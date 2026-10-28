//! Private foreground runner for one manually started ORCH agent.
//!
//! The interactive shell receives only this runner's short command. The server
//! stages the agent command and complete briefing in its owner-only session
//! directory; the runner consumes that record once and passes the briefing to
//! the agent as a structured process argument. Long prompts therefore never
//! cross the fresh PTY's canonical input buffer.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

const OWNERSHIP_WAIT: Duration = Duration::from_secs(5);
const OWNERSHIP_RETRY: Duration = Duration::from_millis(10);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ManualLaunchSpec {
    task_id: String,
    pane: u32,
    agent: String,
    briefing: String,
}

pub(crate) fn stage(
    pane: crate::ids::PaneId,
    task_id: &str,
    agent: &str,
    briefing: &str,
) -> Result<()> {
    crate::persist::ensure_server_session_dir()?;
    let destination = launch_path(pane);
    let temporary = destination.with_file_name(format!(
        ".task-launch-{}-{}",
        pane.0,
        crate::ids::public_id("tmp")
    ));
    let body = serde_json::to_vec(&ManualLaunchSpec {
        task_id: task_id.to_string(),
        pane: pane.0,
        agent: agent.to_string(),
        briefing: briefing.to_string(),
    })?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let result = (|| -> Result<()> {
        let mut file = options.open(&temporary)?;
        file.write_all(&body)?;
        file.flush()?;
        crate::platform::atomic_replace_file(&temporary, &destination)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn discard(pane: crate::ids::PaneId) {
    let _ = std::fs::remove_file(launch_path(pane));
}

#[cfg(test)]
pub(crate) fn staged_for_test(pane: crate::ids::PaneId) -> bool {
    launch_path(pane).exists()
}

pub(crate) fn validate_agent_command(agent: &str) -> Result<(), String> {
    if let Some(descriptor) = crate::agent::registry::find(agent) {
        return if crate::agent::registry::supports_local_task(descriptor) {
            Ok(())
        } else {
            Err(format!(
                "{agent} cannot work in a local ORCH task workspace"
            ))
        };
    }
    custom_agent_argv(agent).map(|_| ())
}

pub(crate) fn run(args: &[String]) -> Result<i32> {
    let [_, command] = args else {
        return Err(anyhow!("invalid internal task worker invocation"));
    };
    if command != "__task-worker" {
        return Err(anyhow!("invalid internal task worker command"));
    }
    let pane = std::env::var("LUVUS_PANE_ID")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .ok_or_else(|| anyhow!("manual task worker has no pane identity"))?;
    let pane = crate::ids::PaneId(pane);
    let staged = peek(pane)?;
    if staged.pane != pane.0 {
        return Err(anyhow!("manual task launch belongs to another pane"));
    }
    // Shell input and app mutations run independently. Keep the one-use record
    // intact until the server has committed the task/pane ownership that
    // authorizes this process to consume it.
    let attempt = wait_for_owner(&staged.task_id, pane.0)?;
    let spec = take(pane)?;
    if spec != staged {
        return Err(anyhow!("manual task launch changed before it was consumed"));
    }
    if crate::orch::contains_multiline_control(&spec.briefing) {
        return Err(anyhow!(
            "task briefing must not contain terminal control characters"
        ));
    }
    let status = match launch(&spec.agent, &spec.briefing, &spec.task_id) {
        Ok(status) => status,
        Err(error) => {
            fail_unstarted(&spec.task_id, pane.0, attempt, &error.to_string());
            return Err(error);
        }
    };
    Ok(status.code().unwrap_or(1))
}

/// How many times the runner re-reads a task whose revision moved before the
/// start failure could be recorded.
const FAIL_UNSTARTED_TRIES: usize = 3;

#[derive(Debug, PartialEq, Eq)]
enum UnstartedOutcome {
    /// The task was marked failed.
    Failed,
    /// The task no longer belongs to this pane's running attempt.
    Changed,
    /// Other mutations kept winning; the task was left untouched.
    Conflicted,
    Error(String),
}

/// A task whose agent never started has no process left to settle it, so the
/// runner records the failure instead of leaving the task `running`. The task
/// may have changed while the agent was starting (completed, blocked, retried
/// or reassigned); only the attempt this pane still owns is marked failed.
fn fail_unstarted(task_id: &str, pane: u32, attempt: u64, detail: &str) {
    match fail_unstarted_with(task_id, pane, attempt, detail, crate::cli::send_request) {
        UnstartedOutcome::Failed => {}
        UnstartedOutcome::Changed => {
            eprintln!("task {task_id} changed while its agent was starting; leaving it as is")
        }
        UnstartedOutcome::Conflicted => eprintln!(
            "could not mark task {task_id} failed: its state kept changing; leaving it as is"
        ),
        UnstartedOutcome::Error(error) => {
            eprintln!("could not mark task {task_id} failed: {error}")
        }
    }
}

/// Read the task, check that this pane still owns the running attempt, and
/// mark it failed only if no other mutation landed since that read. The
/// server applies `if_revision` and the update in one app-loop step, so the
/// ownership check and the write are atomic; a conflict re-reads and decides
/// again.
fn fail_unstarted_with(
    task_id: &str,
    pane: u32,
    attempt: u64,
    detail: &str,
    mut send: impl FnMut(&str, serde_json::Value) -> Result<serde_json::Value>,
) -> UnstartedOutcome {
    for _ in 0..FAIL_UNSTARTED_TRIES {
        let response = match send("task.get", json!({"id":task_id})) {
            Ok(response) => response,
            Err(error) => return UnstartedOutcome::Error(error.to_string()),
        };
        if let Some(error) = response.get("error") {
            return UnstartedOutcome::Error(format!("could not load manual task: {error}"));
        }
        let result = response.get("result");
        let Some(task) = result.and_then(|result| result.get("task")) else {
            return UnstartedOutcome::Error(format!(
                "manual task snapshot is unavailable: {task_id}"
            ));
        };
        if !owns_running_attempt(task, pane, attempt) {
            return UnstartedOutcome::Changed;
        }
        let Some(revision) = result
            .and_then(|result| result.get("revision"))
            .and_then(|revision| revision.as_u64())
        else {
            return UnstartedOutcome::Error("task snapshot has no revision".to_string());
        };
        let response = match send(
            "task.update",
            json!({"id":task_id, "status":"failed", "output":detail, "if_revision":revision}),
        ) {
            Ok(response) => response,
            Err(error) => return UnstartedOutcome::Error(error.to_string()),
        };
        match response.get("error") {
            None => return UnstartedOutcome::Failed,
            Some(error)
                if error.get("code").and_then(|code| code.as_str())
                    == Some("revision_conflict") => {}
            Some(error) => return UnstartedOutcome::Error(error.to_string()),
        }
    }
    UnstartedOutcome::Conflicted
}

fn take(pane: crate::ids::PaneId) -> Result<ManualLaunchSpec> {
    let path = launch_path(pane);
    let claimed = path.with_file_name(format!(
        ".task-launch-consume-{}-{}",
        pane.0,
        crate::ids::public_id("tmp")
    ));
    std::fs::rename(&path, &claimed)
        .map_err(|error| anyhow!("manual task launch is unavailable: {error}"))?;
    let result = std::fs::read(&claimed)
        .map_err(anyhow::Error::from)
        .and_then(|body| {
            serde_json::from_slice(&body)
                .map_err(|error| anyhow!("invalid manual task launch: {error}"))
        });
    let _ = std::fs::remove_file(claimed);
    result
}

fn peek(pane: crate::ids::PaneId) -> Result<ManualLaunchSpec> {
    let body = std::fs::read(launch_path(pane))
        .map_err(|error| anyhow!("manual task launch is unavailable: {error}"))?;
    serde_json::from_slice(&body).map_err(|error| anyhow!("invalid manual task launch: {error}"))
}

fn launch_path(pane: crate::ids::PaneId) -> PathBuf {
    crate::persist::session_dir().join(format!("task-launch-{}.json", pane.0))
}

/// Wait until the server binds the task to this pane, and return the attempt
/// number this runner owns.
fn wait_for_owner(task_id: &str, pane: u32) -> Result<u64> {
    let mut attempt = None;
    wait_for_owner_with(OWNERSHIP_WAIT, || {
        let task = fetch_task(task_id)?;
        let ready = owner_ready(&task, task_id, pane)?;
        if ready {
            attempt = task.get("attempt").and_then(|value| value.as_u64());
        }
        Ok(ready)
    })
    .map_err(|error| anyhow!("manual task ownership was not committed: {error}"))?;
    attempt.ok_or_else(|| anyhow!("manual task snapshot has no attempt number: {task_id}"))
}

fn wait_for_owner_with(timeout: Duration, mut ready: impl FnMut() -> Result<bool>) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if ready()? {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(anyhow!("timed out waiting for the task/pane binding"));
        }
        std::thread::sleep(OWNERSHIP_RETRY);
    }
}

fn fetch_task(task_id: &str) -> Result<serde_json::Value> {
    let response = crate::cli::send_request("task.get", json!({"id":task_id}))?;
    if let Some(error) = response.get("error") {
        return Err(anyhow!("could not load manual task: {error}"));
    }
    response
        .get("result")
        .and_then(|result| result.get("task"))
        .cloned()
        .ok_or_else(|| anyhow!("manual task snapshot is unavailable: {task_id}"))
}

fn owns_running_attempt(task: &serde_json::Value, pane: u32, attempt: u64) -> bool {
    task.get("status").and_then(|value| value.as_str()) == Some("running")
        && task.get("assignee").and_then(|value| value.as_u64()) == Some(u64::from(pane))
        && task.get("attempt").and_then(|value| value.as_u64()) == Some(attempt)
}

fn owner_ready(task: &serde_json::Value, task_id: &str, pane: u32) -> Result<bool> {
    let status = task.get("status").and_then(|value| value.as_str());
    let assignee = task.get("assignee").and_then(|value| value.as_u64());
    if status == Some("running") && assignee == Some(u64::from(pane)) {
        return Ok(true);
    }
    if (status == Some("queued") && assignee.is_none())
        || (status == Some("claimed") && assignee == Some(u64::from(pane)))
    {
        return Ok(false);
    }
    Err(anyhow!("manual task is not owned by this pane: {task_id}"))
}

fn launch(agent: &str, briefing: &str, task_id: &str) -> Result<ExitStatus> {
    let start_error =
        |error: std::io::Error| anyhow!("could not start manual agent {agent}: {error}");
    let mut command = if let Some(descriptor) = crate::agent::registry::find(agent) {
        if !crate::agent::registry::supports_local_task(descriptor) {
            return Err(anyhow!(
                "{agent} cannot work in a local ORCH task workspace"
            ));
        }
        let mut command =
            crate::platform::agent_command(descriptor.launch_command).map_err(start_error)?;
        command.args(descriptor.task_prompt_args);
        command
    } else {
        let argv = custom_agent_argv(agent).map_err(anyhow::Error::msg)?;
        let (executable, arguments) = argv
            .split_first()
            .ok_or_else(|| anyhow!("manual agent command cannot be empty"))?;
        let mut command = crate::platform::agent_command(executable).map_err(start_error)?;
        command.args(arguments);
        command
    };
    command
        .arg(briefing)
        .env("LUVUS_TASK_ID", task_id)
        .status()
        .map_err(start_error)
}

fn custom_agent_argv(agent: &str) -> Result<Vec<String>, String> {
    let argv = shell_words::split(agent)
        .map_err(|error| format!("agent command has invalid quoting: {error}"))?;
    if argv.is_empty() || argv[0].trim().is_empty() {
        return Err("agent command cannot be empty".to_string());
    }
    // Process detection only unwraps plain `env KEY=value command`. Resolve
    // supported options, including nested wrappers, before checking whether
    // the actual executable is allowed in a local task.
    let mut inspected_argv = argv.as_slice();
    while executable_name(&inspected_argv[0]) == "env" {
        inspected_argv = env_command_argv(inspected_argv)?;
    }
    let executable = executable_name(&inspected_argv[0]);
    let descriptor = crate::agent::registry::find(&executable).or_else(|| {
        crate::detect::builtin_agent_in_argv(inspected_argv)
            .and_then(|agent| crate::agent::registry::find(&agent))
    });
    if let Some(descriptor) = descriptor {
        if !crate::agent::registry::supports_local_task(descriptor) {
            return Err(format!(
                "{agent} cannot work in a local ORCH task workspace"
            ));
        }
    }
    Ok(argv)
}

fn executable_name(command: &str) -> String {
    let basename = command.rsplit(['/', '\\']).next().unwrap_or(command);
    let lowercase = basename.to_ascii_lowercase();
    lowercase
        .strip_suffix(".exe")
        .or_else(|| lowercase.strip_suffix(".cmd"))
        .or_else(|| lowercase.strip_suffix(".bat"))
        .unwrap_or(&lowercase)
        .to_string()
}

fn env_command_argv(argv: &[String]) -> Result<&[String], String> {
    let mut index = 1;
    let mut options = true;
    while let Some(arg) = argv.get(index) {
        if options {
            match arg.as_str() {
                "--" => {
                    options = false;
                    index += 1;
                    continue;
                }
                "-" | "-i" | "--ignore-environment" => {
                    index += 1;
                    continue;
                }
                "-u" | "--unset" => {
                    if argv.get(index + 1).is_none_or(|name| name.is_empty()) {
                        return Err("env unset option requires a variable name".to_string());
                    }
                    index += 2;
                    continue;
                }
                _ => {}
            }
            if let Some(name) = arg.strip_prefix("--unset=") {
                if name.is_empty() {
                    return Err("env unset option requires a variable name".to_string());
                }
                index += 1;
                continue;
            }
            if arg.starts_with('-') {
                return Err("unsupported env option for local ORCH task agent".to_string());
            }
        }
        if arg.contains('=') {
            index += 1;
            continue;
        }
        return Ok(&argv[index..]);
    }
    Err("env command cannot be empty".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_launch_is_private_and_consumed_once() {
        let _env = crate::persist::test_env("manual-worker-stage");
        let pane = crate::ids::PaneId(42);
        stage(pane, "t7", "agent-🚀", "line one\nline two").unwrap();
        let path = launch_path(pane);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let spec = take(pane).unwrap();
        assert_eq!(spec.task_id, "t7");
        assert_eq!(spec.agent, "agent-🚀");
        assert_eq!(spec.briefing, "line one\nline two");
        assert!(!path.exists());
        assert!(take(pane).is_err());
    }

    #[test]
    fn ownership_handshake_waits_before_allowing_consumption() {
        let mut attempts = 0;
        wait_for_owner_with(Duration::from_secs(1), || {
            attempts += 1;
            Ok(attempts == 3)
        })
        .unwrap();
        assert_eq!(attempts, 3);

        let error = wait_for_owner_with(Duration::ZERO, || Ok(false)).unwrap_err();
        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn a_start_failure_only_fails_the_attempt_this_pane_still_owns() {
        let task = |status: &str, assignee: Option<u32>, attempt: u64| json!({"id":"t1", "status":status, "assignee":assignee, "attempt":attempt});
        assert!(owns_running_attempt(&task("running", Some(7), 2), 7, 2));
        for status in ["done", "blocked", "review", "failed", "queued", "claimed"] {
            assert!(
                !owns_running_attempt(&task(status, Some(7), 2), 7, 2),
                "{status}"
            );
        }
        // Reassigned to another pane, or retried as a new attempt.
        assert!(!owns_running_attempt(&task("running", Some(8), 2), 7, 2));
        assert!(!owns_running_attempt(&task("running", None, 2), 7, 2));
        assert!(!owns_running_attempt(&task("running", Some(7), 3), 7, 2));
        assert!(!owns_running_attempt(&json!({"status":"running"}), 7, 2));
    }

    type SentLog = std::rc::Rc<std::cell::RefCell<Vec<(String, serde_json::Value)>>>;

    /// Replay scripted server responses and record every request sent.
    fn scripted(
        responses: Vec<serde_json::Value>,
    ) -> (
        impl FnMut(&str, serde_json::Value) -> Result<serde_json::Value>,
        SentLog,
    ) {
        let sent = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let log = sent.clone();
        let mut responses = responses.into_iter();
        let send = move |method: &str, params: serde_json::Value| {
            log.borrow_mut().push((method.to_string(), params));
            responses
                .next()
                .ok_or_else(|| anyhow!("unexpected request: {method}"))
        };
        (send, sent)
    }

    fn task_get(status: &str, revision: u64) -> serde_json::Value {
        json!({"result":{"type":"task", "revision":revision,
            "task":{"id":"t1", "status":status, "assignee":7, "attempt":2}}})
    }

    fn conflict() -> serde_json::Value {
        json!({"error":{"code":"revision_conflict", "message":"socket state changed"}})
    }

    #[test]
    fn an_unstarted_task_still_owned_is_failed_at_the_read_revision() {
        let (send, sent) = scripted(vec![
            task_get("running", 41),
            json!({"result":{"type":"task"}}),
        ]);
        assert_eq!(
            fail_unstarted_with("t1", 7, 2, "program not found", send),
            UnstartedOutcome::Failed
        );
        let sent = sent.borrow();
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[1].0, "task.update");
        assert_eq!(
            sent[1].1,
            json!({"id":"t1", "status":"failed", "output":"program not found", "if_revision":41})
        );
    }

    #[test]
    fn an_unstarted_task_that_changed_is_left_alone() {
        for status in ["done", "blocked", "review", "queued"] {
            let (send, sent) = scripted(vec![task_get(status, 5)]);
            assert_eq!(
                fail_unstarted_with("t1", 7, 2, "x", send),
                UnstartedOutcome::Changed,
                "{status}"
            );
            assert_eq!(sent.borrow().len(), 1, "{status}");
        }
    }

    #[test]
    fn a_revision_conflict_rereads_and_decides_again() {
        // Unrelated mutation first: the task is still ours, so retry.
        let (send, sent) = scripted(vec![
            task_get("running", 5),
            conflict(),
            task_get("running", 6),
            json!({"result":{"type":"task"}}),
        ]);
        assert_eq!(
            fail_unstarted_with("t1", 7, 2, "x", send),
            UnstartedOutcome::Failed
        );
        assert_eq!(sent.borrow()[3].1["if_revision"], 6);

        // The conflicting mutation completed the task: leave it.
        let (send, sent) = scripted(vec![
            task_get("running", 5),
            conflict(),
            task_get("done", 6),
        ]);
        assert_eq!(
            fail_unstarted_with("t1", 7, 2, "x", send),
            UnstartedOutcome::Changed
        );
        assert_eq!(sent.borrow().len(), 3);
    }

    #[test]
    fn exhausted_revision_conflicts_leave_the_task_untouched() {
        let mut responses = Vec::new();
        for revision in 0..FAIL_UNSTARTED_TRIES as u64 {
            responses.push(task_get("running", revision));
            responses.push(conflict());
        }
        let (send, sent) = scripted(responses);
        assert_eq!(
            fail_unstarted_with("t1", 7, 2, "x", send),
            UnstartedOutcome::Conflicted
        );
        let sent = sent.borrow();
        assert_eq!(sent.len(), 2 * FAIL_UNSTARTED_TRIES);
        assert!(sent
            .iter()
            .filter(|(method, _)| method == "task.update")
            .all(|(_, params)| params.get("if_revision").is_some()));
    }

    #[test]
    fn other_update_errors_and_missing_revisions_are_reported() {
        let (send, _) = scripted(vec![
            task_get("running", 5),
            json!({"error":{"code":"task_complete", "message":"t1 is already merged"}}),
        ]);
        assert!(matches!(
            fail_unstarted_with("t1", 7, 2, "x", send),
            UnstartedOutcome::Error(error) if error.contains("task_complete")
        ));
        let (send, sent) = scripted(vec![json!({"result":{"task":
            {"id":"t1", "status":"running", "assignee":7, "attempt":2}}})]);
        assert!(matches!(
            fail_unstarted_with("t1", 7, 2, "x", send),
            UnstartedOutcome::Error(error) if error.contains("revision")
        ));
        assert_eq!(sent.borrow().len(), 1);
    }

    #[test]
    fn remote_sandbox_agent_cannot_claim_a_local_task() {
        assert!(validate_agent_command("arc-studio").is_err());
        assert!(launch("arc-studio", "briefing", "t1").is_err());
        for command in [
            "arc-studio --continue",
            "/usr/local/bin/arc-studio --continue",
            r#""C:\Program Files\nodejs\arc-studio.cmd" --continue"#,
            "node /opt/node_modules/@circle-fin/arc-studio-cli/bin/arc-studio.mjs",
            "node --require helper /opt/node_modules/@circle-fin/arc-studio-cli/bin/arc-studio.mjs",
            r#""C:\Program Files\nodejs\node.exe" "C:\Users\Ada\node_modules\@circle-fin\arc-studio-cli\bin\arc-studio.mjs""#,
            "env FOO=bar arc-studio",
            "env -i arc-studio",
            "env - arc-studio",
            "env -u FOO arc-studio",
            "env -i env -i arc-studio",
            "env - /usr/bin/env -u FOO arc-studio",
            "env FOO=bar -- arc-studio",
            "env --unset=FOO arc-studio",
            "env FOO=bar -i arc-studio",
            "env -i node /opt/node_modules/@circle-fin/arc-studio-cli/bin/arc-studio.mjs",
            "env - node /opt/node_modules/@circle-fin/arc-studio-cli/bin/arc-studio.mjs",
            "env -u FOO node /opt/node_modules/@circle-fin/arc-studio-cli/bin/arc-studio.mjs",
            "env -i env -u FOO node /opt/node_modules/@circle-fin/arc-studio-cli/bin/arc-studio.mjs",
            "env -S arc-studio",
        ] {
            assert!(validate_agent_command(command).is_err(), "{command}");
            assert!(launch(command, "briefing", "t1").is_err(), "{command}");
        }
        assert!(validate_agent_command("codex").is_ok());
        assert!(validate_agent_command("codex --model o3").is_ok());
        assert!(validate_agent_command("custom-agent --flag").is_ok());
        assert!(validate_agent_command("env FOO=bar custom-agent --flag").is_ok());
        for command in [
            "env -i custom-agent",
            "env - custom-agent",
            "env -u FOO custom-agent",
            "env --ignore-environment custom-agent",
            "env --unset=FOO custom-agent",
            "env FOO=bar -- custom-agent",
            "env FOO=bar -i custom-agent",
            "env -i env -u FOO custom-agent",
        ] {
            assert!(validate_agent_command(command).is_ok(), "{command}");
        }
        assert!(validate_agent_command("env -u custom-agent").is_err());
        assert!(
            validate_agent_command("node /opt/node_modules/@other/arc-studio-cli/bin/cli.mjs")
                .is_ok()
        );
        assert!(validate_agent_command("node app.js --example arc-studio").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn direct_launch_delivers_a_multiline_briefing_beyond_canonical_tty_limits() {
        use std::os::unix::fs::PermissionsExt;

        let root = std::env::temp_dir().join(format!(
            "luvus-manual-worker-{}-{}",
            std::process::id(),
            crate::ids::public_id("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let agent = root.join("capture-agent");
        let output = root.join("capture-agent.out");
        let briefing = format!("heading\n{}\nending", "x".repeat(8 * 1024));
        let agent_command = format!("{} --flag 'two words'", agent.display());

        std::fs::write(
            &agent,
            "#!/bin/sh\nprintf '%s\\n%s' \"$1\" \"$2\" > \"$0.args\"\nprintf '%s' \"$3\" > \"$0.out\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&agent, std::fs::Permissions::from_mode(0o700)).unwrap();
        let status = launch(&agent_command, &briefing, "t42").unwrap();

        assert!(status.success());
        assert_eq!(
            std::fs::read_to_string(agent.with_extension("args")).unwrap(),
            "--flag\ntwo words"
        );
        assert_eq!(std::fs::read_to_string(&output).unwrap(), briefing);
        for wrapper in ["env -", "env -i", "env -u FOO"] {
            let wrapped_command = format!("{wrapper} {agent_command}");
            assert!(launch(&wrapped_command, &briefing, "t42")
                .unwrap()
                .success());
            assert_eq!(std::fs::read_to_string(&output).unwrap(), briefing);
        }
        let nested_command = format!("env - /usr/bin/env -i {agent_command}");
        assert!(launch(&nested_command, &briefing, "t42").unwrap().success());
        assert_eq!(std::fs::read_to_string(&output).unwrap(), briefing);
        let _ = std::fs::remove_dir_all(root);
    }
}
