//! AutoShorts 11.0 — Bounded subprocess execution
//!
//! Every external process in the pipeline (Python sidecars, ffmpeg, ffprobe,
//! yt-dlp) must be bounded. `std::process::Command::output()` waits forever:
//! a wedged child, a full pipe, or a slow source hangs the render thread with
//! no diagnostic and no way to recover.
//!
//! `run_bounded` enforces a hard deadline, kills the child on expiry (Windows
//! and Unix), drains stdout/stderr concurrently so a chatty child can never
//! deadlock the parent on a full pipe, and surfaces a typed timeout error
//! callers can log and fall back from.

use anyhow::{anyhow, Result};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Outcome of a bounded child process.
#[derive(Debug, Clone)]
pub struct BoundedOutput {
    pub success: bool,
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub elapsed: Duration,
    pub timed_out: bool,
}

/// Run `cmd` to completion, killing it if it exceeds `timeout`.
///
/// The child is spawned with piped stdout/stderr, and both pipes are drained on
/// dedicated threads. Draining concurrently is what prevents the classic
/// deadlock where the child blocks writing to a full pipe while the parent
/// blocks in `wait()`.
pub fn run_bounded(cmd: &mut Command, timeout: Duration, stage: &str) -> Result<BoundedOutput> {
    let started = Instant::now();
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return Err(anyhow!("{}: spawn failed: {}", stage, e)),
    };

    // Take the pipes before waiting so the readers start immediately.
    let mut out_pipe = child.stdout.take();
    let mut err_pipe = child.stderr.take();

    let (tx_out, rx_out) = mpsc::channel();
    let (tx_err, rx_err) = mpsc::channel();

    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(p) = out_pipe.as_mut() {
            let _ = p.read_to_end(&mut buf);
        }
        let _ = tx_out.send(buf);
    });
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(p) = err_pipe.as_mut() {
            let _ = p.read_to_end(&mut buf);
        }
        let _ = tx_err.send(buf);
    });

    // Poll for exit against the deadline. A short sleep keeps the loop cheap
    // while still reacting well inside the timeout.
    let poll = Duration::from_millis(50);
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if started.elapsed() >= timeout {
                    timed_out = true;
                    // Kill, then reap, so no zombie survives the call. On
                    // Windows `kill()` only signals the direct child, so the
                    // tree is taskkill'd as well; otherwise a shelled-out
                    // grandchild would keep running after we return.
                    kill_process_tree(&mut child);
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(poll);
            }
            Err(e) => {
                let _ = child.kill();
                return Err(anyhow!("{}: wait failed: {}", stage, e));
            }
        }
    };

    // Collect the drained output. The reader threads finish as soon as the
    // pipes close. The wait is short: it exists to collect what was already
    // read, not to wait for a wedged grandchild (Windows `kill()` terminates
    // the direct child only), so a long wait here would reintroduce the very
    // unbounded behavior this module exists to prevent.
    let drain = Duration::from_millis(500);
    let stdout = rx_out.recv_timeout(drain).unwrap_or_default();
    let stderr = rx_err.recv_timeout(drain).unwrap_or_default();

    let elapsed = started.elapsed();

    if timed_out {
        return Ok(BoundedOutput {
            success: false,
            code: None,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: format!(
                "{}\n[timed out and killed after {:.1}s]",
                String::from_utf8_lossy(&stderr),
                elapsed.as_secs_f64()
            ),
            elapsed,
            timed_out: true,
        });
    }

    let status = status.ok_or_else(|| anyhow!("{}: child exited without status", stage))?;
    Ok(BoundedOutput {
        success: status.success(),
        code: status.code(),
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        elapsed,
        timed_out: false,
    })
}

/// Kill a child and, on Windows, its whole process tree.
///
/// `Child::kill()` only terminates the direct child. When the command is a
/// shell wrapper that has itself spawned the real worker (e.g. `cmd /C`),
/// the grandchild would keep running — holding CPU, a file handle, or a port
/// — long after the caller believes the process is gone. `taskkill /T` takes
/// the tree down with it.
fn kill_process_tree(child: &mut std::process::Child) {
    #[cfg(windows)]
    {
        let pid = child.id();
        let _ = std::process::Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
    let _ = child.kill();
}

/// Truncate stderr for logging: enough to diagnose, bounded so a multi-MB
/// ffmpeg spew cannot flood the log.
pub fn sanitize_stderr(stderr: &str, max_chars: usize) -> String {
    let trimmed = stderr.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(max_chars / 2).collect();
    let tail: String = trimmed
        .chars()
        .skip(trimmed.chars().count() - max_chars / 2)
        .collect();
    format!(
        "{} ... [{} chars elided] ... {}",
        head,
        trimmed.chars().count() - max_chars,
        tail
    )
}

/// Stage progress contract.
///
/// Every long or external stage announces START, then exactly one terminal
/// marker. The point is that a user watching the console can always tell which
/// stage owns the current wait, and for how long, without attaching a
/// debugger. Emitted at low frequency — one line per stage transition.
pub struct StageTimer {
    stage: &'static str,
    start: Instant,
}

impl StageTimer {
    pub fn start(stage: &'static str) -> Self {
        println!("[{}] START", stage);
        Self {
            stage,
            start: Instant::now(),
        }
    }

    pub fn complete(&self, detail: impl std::fmt::Display) {
        println!(
            "[{}] COMPLETE in {:.1}s ({})",
            self.stage,
            self.start.elapsed().as_secs_f64(),
            detail
        );
    }

    pub fn failed(&self, reason: impl std::fmt::Display) {
        eprintln!(
            "[{}] FAILED after {:.1}s reason={}",
            self.stage,
            self.start.elapsed().as_secs_f64(),
            reason
        );
    }

    /// Long-running stage that is still legitimately working.
    /// `label` identifies the work item, e.g. a chunk index.
    pub fn heartbeat(&self, label: impl std::fmt::Display) {
        println!(
            "[{}] HEARTBEAT elapsed={:.0}s {}",
            self.stage,
            self.start.elapsed().as_secs_f64(),
            label
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_stdout_and_success() {
        let mut cmd = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args(["/C", "echo hello"]);
        } else {
            cmd.args(["-c", "echo hello"]);
        }
        let out = run_bounded(&mut cmd, Duration::from_secs(30), "T").unwrap();
        assert!(out.success);
        assert!(out.stdout.contains("hello"), "got {:?}", out.stdout);
        assert!(!out.timed_out);
    }

    #[test]
    fn nonzero_exit_is_reported_not_panicked() {
        let mut cmd = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args(["/C", "exit 3"]);
        } else {
            cmd.args(["-c", "exit 3"]);
        }
        let out = run_bounded(&mut cmd, Duration::from_secs(30), "T").unwrap();
        assert!(!out.success);
        assert_eq!(out.code, Some(3));
    }

    #[test]
    fn timeout_kills_child_and_is_flagged() {
        // The whole point: a wedged child must not hang the pipeline.
        let mut cmd = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args(["/C", "ping -n 60 127.0.0.1 >NUL"]);
        } else {
            cmd.args(["-c", "sleep 60"]);
        }
        let start = Instant::now();
        let out = run_bounded(&mut cmd, Duration::from_secs(2), "T").unwrap();
        let wall = start.elapsed();
        assert!(out.timed_out, "child should have been killed");
        assert!(!out.success);
        assert!(
            wall < Duration::from_secs(20),
            "must return promptly, took {:?}",
            wall
        );
        assert!(out.stderr.contains("timed out"), "got {:?}", out.stderr);
    }

    #[test]
    fn large_output_does_not_deadlock() {
        // A child that writes far more than a pipe buffer would deadlock a
        // parent that waits before reading.
        let mut cmd = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            cmd.args([
                "/C",
                "for /L %i in (1,1,20000) do @echo aaaaaaaaaaaaaaaaaaaa",
            ]);
        } else {
            cmd.args([
                "-c",
                "for i in $(seq 1 20000); do echo aaaaaaaaaaaaaaaaaaaa; done",
            ]);
        }
        let out = run_bounded(&mut cmd, Duration::from_secs(60), "T").unwrap();
        assert!(out.success);
        assert!(out.stdout.len() > 100_000, "got {} bytes", out.stdout.len());
    }

    #[test]
    fn missing_binary_is_an_error_not_a_hang() {
        let mut cmd = Command::new("definitely_not_a_real_binary_xyz");
        let r = run_bounded(&mut cmd, Duration::from_secs(5), "T");
        assert!(r.is_err());
    }

    #[test]
    fn sanitize_stderr_is_bounded_and_keeps_both_ends() {
        let long = "A".repeat(5000) + "TAIL";
        let s = sanitize_stderr(&long, 200);
        assert!(s.chars().count() < 400, "len {}", s.chars().count());
        assert!(s.contains("TAIL"), "must keep the tail for diagnosis");
        assert!(s.contains("elided"));
        assert_eq!(sanitize_stderr("  small  ", 200), "small");
    }

    #[test]
    fn render_timeout_scales_with_clip_and_is_bounded() {
        use crate::media::render_timeout_for;
        let short = render_timeout_for(0.0, 10.0);
        assert!(short.as_secs_f64() >= 300.0, "{:?}", short);
        let long = render_timeout_for(0.0, 600.0);
        assert!(long.as_secs_f64() > short.as_secs_f64());
        // Bounded on both ends.
        assert!(long.as_secs_f64() <= 3600.0);
        assert!(render_timeout_for(0.0, 1e9).as_secs_f64() <= 3600.0);
        // Non-finite / reversed inputs must not produce a zero budget.
        assert!(render_timeout_for(f64::NAN, f64::NAN).as_secs_f64() >= 300.0);
        assert!(render_timeout_for(100.0, 10.0).as_secs_f64() >= 300.0);
    }

    #[test]
    fn stage_timer_reports_without_panicking() {
        // The progress contract must never itself be a failure mode.
        let t = StageTimer::start("UnitTestStage");
        t.heartbeat("item 1");
        t.complete("ok");
        let t2 = StageTimer::start("UnitTestStage2");
        t2.failed("synthetic");
    }
}
