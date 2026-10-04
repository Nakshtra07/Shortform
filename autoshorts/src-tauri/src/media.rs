use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::models::{CandidateDraft, MediaProbe, MultimodalStatus, TranscriptWord};

pub fn find_ytdlp_cmd() -> Option<PathBuf> {
    // 1. Check environment variables
    for env_var in &["YTDLP_PATH", "YT_DLP_PATH"] {
        if let Ok(val) = std::env::var(env_var) {
            let path = PathBuf::from(val.trim());
            if path.exists() {
                if Command::new(&path)
                    .arg("--version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Some(path);
                }
            }
        }
    }

    // 2. Try default command on PATH ("yt-dlp" or "yt-dlp.exe")
    let default_cmd = if cfg!(windows) {
        "yt-dlp.exe"
    } else {
        "yt-dlp"
    };
    if let Ok(out) = Command::new(default_cmd).arg("--version").output() {
        if out.status.success() {
            return Some(PathBuf::from(default_cmd));
        }
    }

    // 3. Project-local virtual environment (.venv) â€” mirrors find_python_cmd() discovery
    let mut venv_candidates: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        venv_candidates.push(PathBuf::from("../.venv/Scripts/yt-dlp.exe"));
        venv_candidates.push(PathBuf::from(".venv/Scripts/yt-dlp.exe"));
        venv_candidates.push(PathBuf::from("../../.venv/Scripts/yt-dlp.exe"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                venv_candidates.push(parent.join(".venv/Scripts/yt-dlp.exe"));
                venv_candidates.push(parent.join("../.venv/Scripts/yt-dlp.exe"));
                venv_candidates.push(parent.join("../../.venv/Scripts/yt-dlp.exe"));
            }
        }
    } else {
        venv_candidates.push(PathBuf::from("../.venv/bin/yt-dlp"));
        venv_candidates.push(PathBuf::from(".venv/bin/yt-dlp"));
        venv_candidates.push(PathBuf::from("../../.venv/bin/yt-dlp"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                venv_candidates.push(parent.join(".venv/bin/yt-dlp"));
                venv_candidates.push(parent.join("../.venv/bin/yt-dlp"));
                venv_candidates.push(parent.join("../../.venv/bin/yt-dlp"));
            }
        }
    }

    for path in venv_candidates {
        if path.exists() {
            if let Ok(out) = Command::new(&path).arg("--version").output() {
                if out.status.success() {
                    return Some(path);
                }
            }
        }
    }

    // 4. Known system locations (especially Windows)
    let mut candidates: Vec<PathBuf> = Vec::new();

    if cfg!(windows) {
        candidates.push(PathBuf::from(r"D:\yt-dlp.exe"));
        candidates.push(PathBuf::from(r"C:\yt-dlp.exe"));
        candidates.push(PathBuf::from(r"D:\yt-dlp\yt-dlp.exe"));
        candidates.push(PathBuf::from(r"C:\yt-dlp\yt-dlp.exe"));

        if let Ok(home) = std::env::var("USERPROFILE") {
            let home_path = PathBuf::from(home);
            candidates.push(home_path.join("yt-dlp.exe"));
            candidates.push(home_path.join("bin").join("yt-dlp.exe"));
            candidates.push(
                home_path
                    .join("AppData")
                    .join("Local")
                    .join("bin")
                    .join("yt-dlp.exe"),
            );
            candidates.push(
                home_path
                    .join("AppData")
                    .join("Local")
                    .join("Microsoft")
                    .join("WinGet")
                    .join("Links")
                    .join("yt-dlp.exe"),
            );
            candidates.push(
                home_path
                    .join("AppData")
                    .join("Local")
                    .join("Programs")
                    .join("yt-dlp")
                    .join("yt-dlp.exe"),
            );
            candidates.push(home_path.join("scoop").join("shims").join("yt-dlp.exe"));
        }

        if let Ok(windir) = std::env::var("SystemDrive") {
            candidates.push(PathBuf::from(format!(
                r"{}\Program Files\yt-dlp\yt-dlp.exe",
                windir
            )));
            candidates.push(PathBuf::from(format!(
                r"{}\Program Files (x86)\yt-dlp\yt-dlp.exe",
                windir
            )));
        }

        candidates.push(PathBuf::from(r"C:\Program Files\yt-dlp\yt-dlp.exe"));
        candidates.push(PathBuf::from(r"C:\Program Files (x86)\yt-dlp\yt-dlp.exe"));
        candidates.push(PathBuf::from(r"C:\ProgramData\chocolatey\bin\yt-dlp.exe"));
    } else {
        candidates.push(PathBuf::from("/usr/local/bin/yt-dlp"));
        candidates.push(PathBuf::from("/usr/bin/yt-dlp"));
        candidates.push(PathBuf::from("/opt/homebrew/bin/yt-dlp"));
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".local/bin/yt-dlp"));
        }
    }

    for path in candidates {
        if path.exists() {
            if let Ok(out) = Command::new(&path).arg("--version").output() {
                if out.status.success() {
                    return Some(path);
                }
            }
        }
    }

    None
}

pub fn find_ffmpeg_cmd() -> Option<PathBuf> {
    // 1. Check environment variables
    for env_var in &["FFMPEG_PATH", "AUTOSHORTS_FFMPEG_PATH"] {
        if let Ok(val) = std::env::var(env_var) {
            let path = PathBuf::from(val.trim());
            if path.exists() {
                if Command::new(&path)
                    .arg("-version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Some(path);
                }
            }
        }
    }

    // 2. Try default command on PATH ("ffmpeg" or "ffmpeg.exe")
    let default_cmd = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    if let Ok(out) = Command::new(default_cmd).arg("-version").output() {
        if out.status.success() {
            let which_cmd = if cfg!(windows) { "where.exe" } else { "which" };
            if let Ok(where_out) = Command::new(which_cmd).arg(default_cmd).output() {
                if where_out.status.success() {
                    let first_line = String::from_utf8_lossy(&where_out.stdout)
                        .lines()
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !first_line.is_empty() {
                        let p = PathBuf::from(first_line);
                        if p.exists() {
                            return Some(p);
                        }
                    }
                }
            }
            return Some(PathBuf::from(default_cmd));
        }
    }

    // 3. Project-local virtual environment (.venv)
    let mut venv_candidates: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        venv_candidates.push(PathBuf::from("../.venv/Scripts/ffmpeg.exe"));
        venv_candidates.push(PathBuf::from(".venv/Scripts/ffmpeg.exe"));
        venv_candidates.push(PathBuf::from("../../.venv/Scripts/ffmpeg.exe"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                venv_candidates.push(parent.join(".venv/Scripts/ffmpeg.exe"));
                venv_candidates.push(parent.join("../.venv/Scripts/ffmpeg.exe"));
                venv_candidates.push(parent.join("../../.venv/Scripts/ffmpeg.exe"));
            }
        }
    } else {
        venv_candidates.push(PathBuf::from("../.venv/bin/ffmpeg"));
        venv_candidates.push(PathBuf::from(".venv/bin/ffmpeg"));
        venv_candidates.push(PathBuf::from("../../.venv/bin/ffmpeg"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                venv_candidates.push(parent.join(".venv/bin/ffmpeg"));
                venv_candidates.push(parent.join("../.venv/bin/ffmpeg"));
                venv_candidates.push(parent.join("../../.venv/bin/ffmpeg"));
            }
        }
    }

    for path in venv_candidates {
        if path.exists() {
            if let Ok(out) = Command::new(&path).arg("-version").output() {
                if out.status.success() {
                    return Some(path);
                }
            }
        }
    }

    // 4. Known system locations
    let mut candidates: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        if let Ok(home) = std::env::var("USERPROFILE") {
            let home_path = PathBuf::from(home);
            candidates.push(home_path.join("AppData").join("Local").join("Microsoft").join("WinGet").join("Links").join("ffmpeg.exe"));
            candidates.push(home_path.join("scoop").join("shims").join("ffmpeg.exe"));
            candidates.push(home_path.join("bin").join("ffmpeg.exe"));

            // Check WinGet Packages directory
            let winget_pkg = home_path.join("AppData").join("Local").join("Microsoft").join("WinGet").join("Packages");
            if winget_pkg.exists() {
                if let Ok(entries) = std::fs::read_dir(&winget_pkg) {
                    for entry in entries.flatten() {
                        let sub = entry.path();
                        if let Ok(sub_entries) = std::fs::read_dir(&sub) {
                            for sub_entry in sub_entries.flatten() {
                                let target = sub_entry.path().join("bin").join("ffmpeg.exe");
                                if target.exists() {
                                    candidates.push(target);
                                }
                            }
                        }
                    }
                }
            }
        }
        candidates.push(PathBuf::from(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"));
        candidates.push(PathBuf::from(r"C:\ffmpeg\bin\ffmpeg.exe"));
        candidates.push(PathBuf::from(r"D:\ffmpeg\bin\ffmpeg.exe"));
        candidates.push(PathBuf::from(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"));
    } else {
        candidates.push(PathBuf::from("/usr/local/bin/ffmpeg"));
        candidates.push(PathBuf::from("/usr/bin/ffmpeg"));
        candidates.push(PathBuf::from("/opt/homebrew/bin/ffmpeg"));
        if let Ok(home) = std::env::var("HOME") {
            candidates.push(PathBuf::from(home).join(".local/bin/ffmpeg"));
        }
    }

    for path in candidates {
        if path.exists() {
            if let Ok(out) = Command::new(&path).arg("-version").output() {
                if out.status.success() {
                    return Some(path);
                }
            }
        }
    }

    None
}

pub fn find_ffprobe_cmd() -> Option<PathBuf> {
    // 1. Check environment variables
    for env_var in &["FFPROBE_PATH", "AUTOSHORTS_FFPROBE_PATH"] {
        if let Ok(val) = std::env::var(env_var) {
            let path = PathBuf::from(val.trim());
            if path.exists() {
                if Command::new(&path)
                    .arg("-version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Some(path);
                }
            }
        }
    }

    // 2. If ffmpeg was found with a directory, check for ffprobe in the same directory
    if let Some(ffmpeg) = find_ffmpeg_cmd() {
        if let Some(parent) = ffmpeg.parent() {
            let probe_name = if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" };
            let probe_candidate = parent.join(probe_name);
            if probe_candidate.exists() {
                if Command::new(&probe_candidate)
                    .arg("-version")
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false)
                {
                    return Some(probe_candidate);
                }
            }
        }
    }

    // 3. Try default command on PATH ("ffprobe" or "ffprobe.exe")
    let default_cmd = if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };
    if let Ok(out) = Command::new(default_cmd).arg("-version").output() {
        if out.status.success() {
            let which_cmd = if cfg!(windows) { "where.exe" } else { "which" };
            if let Ok(where_out) = Command::new(which_cmd).arg(default_cmd).output() {
                if where_out.status.success() {
                    let first_line = String::from_utf8_lossy(&where_out.stdout)
                        .lines()
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !first_line.is_empty() {
                        let p = PathBuf::from(first_line);
                        if p.exists() {
                            return Some(p);
                        }
                    }
                }
            }
            return Some(PathBuf::from(default_cmd));
        }
    }

    None
}

pub fn command_exists(name: &str) -> bool {
    if name == "yt-dlp" || name == "yt-dlp.exe" {
        return find_ytdlp_cmd().is_some();
    }
    if name == "ffmpeg" || name == "ffmpeg.exe" {
        return find_ffmpeg_cmd().is_some();
    }
    if name == "ffprobe" || name == "ffprobe.exe" {
        return find_ffprobe_cmd().is_some();
    }
    Command::new(name).arg("-version").output().is_ok()
}

/// Bounded ffprobe budget. A local-file probe is normally sub-second; the
/// bound exists so a wedged ffprobe can never stall the pipeline.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

pub fn probe_media(path: &str) -> Result<MediaProbe> {
    let ffprobe_bin = find_ffprobe_cmd()
        .ok_or_else(|| anyhow!("ffprobe is not installed or not available on PATH"))?;

    let mut cmd = Command::new(&ffprobe_bin);
    cmd.args([
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]);
    // Bounded: this probe is on the critical path (candidate geometry, render
    // geometry) and an unbounded ffprobe on a pathological input would stall
    // the pipeline before any diagnostic is possible.
    let output = crate::proc_guard::run_bounded(&mut cmd, PROBE_TIMEOUT, "media/probe_media")
        .context("running ffprobe")?;

    if !output.success {
        return Err(anyhow!(
            "ffprobe failed (rc={:?}, timed_out={}): {}",
            output.code,
            output.timed_out,
            crate::proc_guard::sanitize_stderr(&output.stderr, 2000)
        ));
    }

    let json: Value = serde_json::from_str(&output.stdout).context("parsing ffprobe JSON")?;
    let streams = json
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let video = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"));
    let audio = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("audio"));

    let duration_sec = json
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(Value::as_str)
        .and_then(|duration| duration.parse::<f64>().ok());

    Ok(MediaProbe {
        duration_sec,
        has_video: video.is_some(),
        width: video
            .and_then(|stream| stream.get("width"))
            .and_then(Value::as_i64),
        height: video
            .and_then(|stream| stream.get("height"))
            .and_then(Value::as_i64),
        video_codec: video
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        audio_codec: audio
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

/// Wall-clock budget for an ffmpeg render, derived from clip duration.
///
/// Render cost is roughly linear in output length, so the budget is a
/// multiple of clip duration with a floor for short clips and a cap so a
/// pathological input cannot occupy the machine indefinitely.
const RENDER_TIMEOUT_FLOOR_SEC: f64 = 300.0;
const RENDER_TIMEOUT_REALTIME_FACTOR: f64 = 6.0;
const RENDER_TIMEOUT_CEILING_SEC: f64 = 3600.0;

pub fn render_timeout_for(start_sec: f64, end_sec: f64) -> std::time::Duration {
    let clip = (end_sec - start_sec).max(0.0);
    let secs = if clip.is_finite() && clip > 0.0 {
        RENDER_TIMEOUT_REALTIME_FACTOR * clip
    } else {
        RENDER_TIMEOUT_FLOOR_SEC
    };
    std::time::Duration::from_secs_f64(
        secs.clamp(RENDER_TIMEOUT_FLOOR_SEC, RENDER_TIMEOUT_CEILING_SEC),
    )
}

/// Budget for the transcribe-audio extraction ffmpeg calls.
///
/// Scaling needs the source duration, which is a property of the file rather
/// than the candidate range, so it is probed (bounded, 30s) and cached in a
/// process-local map: the extraction fallback ladder runs up to three times per
/// source and must not pay for a probe each time. The budget is deliberately
/// generous because this is a MANDATORY stage — the bound exists to stop an
/// unbounded wait, not to cut a legitimate long-source transcode short.
pub fn audio_extraction_timeout_for(source_path: &str) -> std::time::Duration {
    const FLOOR_SEC: f64 = 600.0;
    const REALTIME_FACTOR: f64 = 6.0;
    const CEILING_SEC: f64 = 3600.0;

    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, Option<f64>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));

    // Insert-if-absent: if a probe for this source is already in flight the
    // value is reused, so concurrent callers cannot race a second probe.
    let known = cache
        .lock()
        .ok()
        .and_then(|mut m| match m.get(source_path) {
            Some(v) => Some(*v),
            None => {
                m.insert(source_path.to_string(), None);
                None
            }
        });
    let duration = match known {
        Some(d) => d,
        None => {
            let probed = probe_media_duration_seconds_bounded(source_path);
            if let Ok(mut m) = cache.lock() {
                m.insert(source_path.to_string(), probed);
            }
            probed
        }
    };

    let secs = match duration {
        Some(d) if d.is_finite() && d > 0.0 => REALTIME_FACTOR * d,
        // Unknown duration: assume a long-form source rather than risk cutting
        // a legitimate transcode short on a probe miss.
        _ => CEILING_SEC,
    };
    std::time::Duration::from_secs_f64(secs.clamp(FLOOR_SEC, CEILING_SEC))
}

/// Bounded ffprobe of the source duration, returning None on any failure.
fn probe_media_duration_seconds_bounded(path: &str) -> Option<f64> {
    let ffprobe_bin = find_ffprobe_cmd()?;
    let mut cmd = Command::new(&ffprobe_bin);
    cmd.args([
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]);
    let out = crate::proc_guard::run_bounded(
        &mut cmd,
        std::time::Duration::from_secs(30),
        "extract_audio/duration-probe",
    )
    .ok()?;
    if !out.success {
        return None;
    }
    out.stdout.trim().lines().next()?.trim().parse::<f64>().ok()
}

pub fn extract_audio(source_path: &str, project_dir: &Path) -> Result<PathBuf> {
    let ffmpeg_bin = find_ffmpeg_cmd()
        .ok_or_else(|| anyhow!("ffmpeg is not installed or not available on PATH"))?;

    std::fs::create_dir_all(project_dir)?;

    // Budget for the transcode. Audio extraction is a mandatory stage, so the
    // bound must be generous, but it must exist: an unbounded `Command::output`
    // here can hang the pipeline forever with no way to recover. The MP3/WAV
    // encode of a long source is far faster than realtime, so a multiple of the
    // source duration is comfortable headroom, and the fallback formats retry
    // under the same budget.
    let budget = audio_extraction_timeout_for(source_path);

    // 1. Try voice-optimized MP3 (64k mono 16kHz) for ultra-fast, lightweight upload
    let mp3_path = project_dir.join("transcription_audio.mp3");
    let mut mp3_cmd = Command::new(&ffmpeg_bin);
    mp3_cmd.args([
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
    ])
    .arg(&mp3_path);
    let mp3_output =
        crate::proc_guard::run_bounded(&mut mp3_cmd, budget, "extract_audio/mp3").ok();

    if let Some(out) = mp3_output {
        if out.success && mp3_path.exists() {
            println!(
                "[Audio Extraction] Extracted voice-optimized MP3 audio: {} ({} bytes) in {:.1}s",
                mp3_path.display(),
                std::fs::metadata(&mp3_path).map(|m| m.len()).unwrap_or(0),
                out.elapsed.as_secs_f64()
            );
            return Ok(mp3_path);
        }
    }

    // 2. Fallback: AAC M4A
    let m4a_path = project_dir.join("transcription_audio.m4a");
    let mut m4a_cmd = Command::new(&ffmpeg_bin);
    m4a_cmd.args([
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
    ])
    .arg(&m4a_path);
    let m4a_output =
        crate::proc_guard::run_bounded(&mut m4a_cmd, budget, "extract_audio/m4a").ok();

    if let Some(out) = m4a_output {
        if out.success && m4a_path.exists() {
            println!(
                "[Audio Extraction] Extracted fallback AAC M4A audio: {} ({} bytes) in {:.1}s",
                m4a_path.display(),
                std::fs::metadata(&m4a_path).map(|m| m.len()).unwrap_or(0),
                out.elapsed.as_secs_f64()
            );
            return Ok(m4a_path);
        }
    }

    // 3. Fallback: standard uncompressed WAV
    let wav_path = project_dir.join("transcription_audio.wav");
    let mut wav_cmd = Command::new(&ffmpeg_bin);
    wav_cmd
        .args(["-y", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000"])
        .arg(&wav_path);
    let wav_output = crate::proc_guard::run_bounded(&mut wav_cmd, budget, "extract_audio/wav")
        .context("running ffmpeg audio extraction")?;

    if !wav_output.success {
        return Err(anyhow!(
            "ffmpeg audio extraction failed (rc={:?}, timed_out={}): {}",
            wav_output.code,
            wav_output.timed_out,
            crate::proc_guard::sanitize_stderr(&wav_output.stderr, 2000)
        ));
    }

    println!(
        "[Audio Extraction] Extracted fallback WAV audio: {} ({} bytes)",
        wav_path.display(),
        std::fs::metadata(&wav_path).map(|m| m.len()).unwrap_or(0)
    );
    Ok(wav_path)
}

/// Minimal success/stdout/stderr carrier so the bounded tracker result keeps
/// exactly the shape the existing parser already consumes.
struct TrackerOutput {
    success: bool,
    /// True when the watchdog killed the tracker. Kept distinct from
    /// `success: false` so a timeout is never reported as a script error.
    timed_out: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Wall-clock budget for the speaker-tracker sidecar.
///
/// Per-frame CV analysis is roughly real-time-to-2x on 1080p, so the budget
/// scales with clip length, with a floor for short clips and a ceiling so a
/// pathological source cannot occupy the machine indefinitely.
const TRACKER_TIMEOUT_FLOOR_SEC: f64 = 180.0;
const TRACKER_TIMEOUT_REALTIME_FACTOR: f64 = 8.0;
const TRACKER_TIMEOUT_CEILING_SEC: f64 = 2400.0;

fn tracker_timeout_for(clip_dur: f64) -> std::time::Duration {
    let secs = if clip_dur.is_finite() && clip_dur > 0.0 {
        TRACKER_TIMEOUT_REALTIME_FACTOR * clip_dur
    } else {
        TRACKER_TIMEOUT_FLOOR_SEC
    };
    std::time::Duration::from_secs_f64(
        secs.clamp(TRACKER_TIMEOUT_FLOOR_SEC, TRACKER_TIMEOUT_CEILING_SEC),
    )
}

fn find_speaker_tracker_script() -> Option<PathBuf> {
    if let Ok(override_path) = std::env::var("AUTOSHORTS_SPEAKER_TRACKER_SCRIPT") {
        let trimmed = override_path.trim();
        if trimmed == "disabled" || trimmed == "none" || trimmed.is_empty() {
            return None;
        }
        let p = PathBuf::from(trimmed);
        if p.exists() {
            return Some(p);
        }
        return None;
    }

    // Look for the speaker_tracker.py script relative to the binary or in common locations
    let script_name = "speaker_tracker.py";

    // 1. Next to the executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidates = [
                parent.join(script_name),
                parent.join("scripts").join(script_name),
                parent.join("../../scripts").join(script_name),
                parent.join("../../../scripts").join(script_name),
            ];
            for c in &candidates {
                if c.exists() {
                    return Some(c.clone());
                }
            }
        }
    }

    // 2. In the src-tauri/scripts dir during development
    let dev_paths = [
        PathBuf::from("autoshorts/src-tauri/scripts").join(script_name),
        PathBuf::from("src-tauri/scripts").join(script_name),
        PathBuf::from("scripts").join(script_name),
        PathBuf::from("../src-tauri/scripts").join(script_name),
        PathBuf::from("../../scripts").join(script_name),
    ];
    for p in &dev_paths {
        if p.exists() {
            return Some(p.clone());
        }
    }

    None
}

pub fn find_fonts_dir() -> Option<PathBuf> {
    if let Ok(override_dir) = std::env::var("AUTOSHORTS_FONTS_DIR") {
        let p = PathBuf::from(override_dir.trim());
        if p.exists() && p.is_dir() {
            return Some(p);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidates = [
                parent.join("fonts"),
                parent.join("../fonts"),
                parent.join("../../fonts"),
            ];
            for p in &candidates {
                if p.exists() && p.is_dir() {
                    return Some(p.clone());
                }
            }
        }
    }

    let dev_paths = [
        PathBuf::from("fonts"),
        PathBuf::from("src-tauri/fonts"),
        PathBuf::from("../fonts"),
        PathBuf::from("../src-tauri/fonts"),
        PathBuf::from("../../fonts"),
        PathBuf::from("../../src-tauri/fonts"),
    ];
    for p in &dev_paths {
        if p.exists() && p.is_dir() {
            return Some(p.clone());
        }
    }

    None
}

pub fn find_python_cmd() -> String {
    // 0. Check explicit environment override
    if let Ok(override_cmd) = std::env::var("AUTOSHORTS_PYTHON") {
        let trimmed = override_cmd.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }

    // 1. Check project-local virtual environment (.venv)
    let mut venv_candidates = Vec::new();
    if cfg!(windows) {
        venv_candidates.push(PathBuf::from("../.venv/Scripts/python.exe"));
        venv_candidates.push(PathBuf::from(".venv/Scripts/python.exe"));
        venv_candidates.push(PathBuf::from("../../.venv/Scripts/python.exe"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                venv_candidates.push(parent.join(".venv/Scripts/python.exe"));
                venv_candidates.push(parent.join("../.venv/Scripts/python.exe"));
                venv_candidates.push(parent.join("../../.venv/Scripts/python.exe"));
            }
        }
    } else {
        venv_candidates.push(PathBuf::from("../.venv/bin/python"));
        venv_candidates.push(PathBuf::from(".venv/bin/python"));
        venv_candidates.push(PathBuf::from("../../.venv/bin/python"));
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                venv_candidates.push(parent.join(".venv/bin/python"));
                venv_candidates.push(parent.join("../.venv/bin/python"));
                venv_candidates.push(parent.join("../../.venv/bin/python"));
            }
        }
    }

    for p in venv_candidates {
        if p.exists() {
            if let Ok(out) = Command::new(&p).arg("--version").output() {
                if out.status.success() {
                    return p.to_string_lossy().to_string();
                }
            }
        }
    }

    let candidates: &[&str] = if cfg!(windows) {
        &["python", "py", "python3"]
    } else {
        &["python3", "python"]
    };

    // 2. Prefer an interpreter that has cv2 installed
    for &cmd in candidates {
        if let Ok(out) = Command::new(cmd).args(["-c", "import cv2"]).output() {
            if out.status.success() {
                return cmd.to_string();
            }
        }
    }

    // 3. Fall back to any working python interpreter
    for &cmd in candidates {
        if Command::new(cmd)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return cmd.to_string();
        }
    }

    "python".to_string()
}

pub fn find_multimodal_analyzer_script() -> Option<PathBuf> {
    let script_name = "multimodal_hook_analyzer.py";

    // 1. Next to the executable
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(script_name);
        if candidate.exists() {
            return Some(candidate);
        }
        let candidate = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("scripts")
            .join(script_name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // 2. In the src-tauri/scripts dir during development
    let dev_paths = [
        PathBuf::from("src-tauri/scripts").join(script_name),
        PathBuf::from("scripts").join(script_name),
        PathBuf::from("../src-tauri/scripts").join(script_name),
    ];
    for p in &dev_paths {
        if p.exists() {
            return Some(p.clone());
        }
    }

    None
}

#[derive(Debug, serde::Deserialize)]
struct MultimodalPyResponse {
    status: String,
    visual_analysis: String,
    audio_analysis: String,
    temporal_analysis: String,
    candidate_count_analyzed: usize,
    fallback_to_v92: bool,
    failure_reason: Option<String>,
    failed_modality: Option<String>,
    processing_time_ms: u64,
    candidates: Vec<CandidateDraft>,
}

pub fn analyze_multimodal_hook_signals(
    source_path: &str,
    candidates: &[CandidateDraft],
    source_duration: f64,
    force_fail_modality: Option<&str>,
) -> (Vec<CandidateDraft>, MultimodalStatus) {
    let start_time = std::time::Instant::now();
    let total_count = candidates.len();

    let script_opt = find_multimodal_analyzer_script();
    let python_cmd = find_python_cmd();

    if script_opt.is_none() {
        let elapsed = start_time.elapsed().as_millis() as u64;
        let mut fallback_candidates = candidates.to_vec();
        for c in &mut fallback_candidates {
            c.multimodal_verified = false;
            c.multimodal_status = Some("FAILED".to_string());
            c.fallback_label =
                Some("v9.2 fallback â€” multimodal analysis unavailable".to_string());
            c.multimodal_evidence = Some(vec![
                "Multimodal evidence unavailable; ranking uses v9.2 semantic fallback.".to_string(),
            ]);
        }

        let status = MultimodalStatus {
            status: "FAILED".to_string(),
            visual_analysis: "SKIPPED".to_string(),
            audio_analysis: "SKIPPED".to_string(),
            temporal_analysis: "SKIPPED".to_string(),
            candidate_count_analyzed: total_count,
            fallback_to_v92: true,
            failure_reason: Some(
                "multimodal_hook_analyzer.py script not found on system".to_string(),
            ),
            failed_modality: Some("Script Missing".to_string()),
            processing_time_ms: elapsed,
            source_duration,
            analyzed_start: 0.0,
            analyzed_end: source_duration,
            uncovered_seconds: 0.0,
            windows_processed: 1,
            windows_failed: 0,
        };
        return (fallback_candidates, status);
    }

    let script_path = script_opt.unwrap();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp_input = std::env::temp_dir().join(format!("autoshorts_cand_in_{}.json", ts));
    let tmp_output = std::env::temp_dir().join(format!("autoshorts_cand_out_{}.json", ts));

    if let Err(e) = std::fs::write(
        &tmp_input,
        serde_json::to_string(candidates).unwrap_or_default(),
    ) {
        let elapsed = start_time.elapsed().as_millis() as u64;
        let mut fallback_candidates = candidates.to_vec();
        for c in &mut fallback_candidates {
            c.multimodal_verified = false;
            c.multimodal_status = Some("FAILED".to_string());
            c.fallback_label =
                Some("v9.2 fallback â€” multimodal analysis unavailable".to_string());
        }
        let status = MultimodalStatus {
            status: "FAILED".to_string(),
            visual_analysis: "FAILED".to_string(),
            audio_analysis: "FAILED".to_string(),
            temporal_analysis: "FAILED".to_string(),
            candidate_count_analyzed: total_count,
            fallback_to_v92: true,
            failure_reason: Some(format!("Failed to write candidate temp file: {e}")),
            failed_modality: Some("File IO".to_string()),
            processing_time_ms: elapsed,
            source_duration,
            analyzed_start: 0.0,
            analyzed_end: source_duration,
            uncovered_seconds: 0.0,
            windows_processed: 1,
            windows_failed: 0,
        };
        return (fallback_candidates, status);
    }

    let mut cmd = Command::new(python_cmd);
    cmd.arg(&script_path)
        .arg("--source")
        .arg(source_path)
        .arg("--candidates")
        .arg(&tmp_input)
        .arg("--output")
        .arg(&tmp_output);

    if let Some(fail_mode) = force_fail_modality {
        match fail_mode {
            "visual" => {
                cmd.arg("--force-fail-visual");
            }
            "audio" => {
                cmd.arg("--force-fail-audio");
            }
            "temporal" => {
                cmd.arg("--force-fail-temporal");
            }
            "all" => {
                cmd.arg("--force-fail-all");
            }
            _ => {}
        }
    }

    // Bounded: this analyzer decodes the full source across visual, acoustic
    // and temporal passes, and sits on the candidate-discovery path. Unbounded
    // `.output()` let a wedged child freeze candidate generation silently.
    let budget = multimodal_timeout_for(source_duration);
    println!(
        "[Multimodal] analyzer START budget={:.0}s for {:.0}s source",
        budget.as_secs_f64(),
        source_duration
    );
    let output = match crate::proc_guard::run_bounded(&mut cmd, budget, "Multimodal/analyzer") {
        Ok(o) => {
            if o.timed_out {
                eprintln!(
                    "[Multimodal] analyzer TIMEOUT after {:.0}s (budget {:.0}s) -> v9.2 fallback",
                    o.elapsed.as_secs_f64(),
                    budget.as_secs_f64()
                );
            } else if !o.success {
                eprintln!(
                    "[Multimodal] analyzer failed rc={:?}: {}",
                    o.code,
                    crate::proc_guard::sanitize_stderr(&o.stderr, 600)
                );
            } else {
                println!(
                    "[Multimodal] analyzer COMPLETE in {:.1}s",
                    o.elapsed.as_secs_f64()
                );
            }
            Ok(MultimodalOutput {
                _success: o.success,
                stdout: o.stdout.into_bytes(),
                stderr: o.stderr.into_bytes(),
            })
        }
        Err(e) => {
            eprintln!("[Multimodal] analyzer spawn failed: {}", e);
            Err(e)
        }
    };
    let _ = std::fs::remove_file(&tmp_input);

    let parsed_res: Option<MultimodalPyResponse> = if tmp_output.exists() {
        let content = std::fs::read_to_string(&tmp_output).unwrap_or_default();
        let _ = std::fs::remove_file(&tmp_output);
        serde_json::from_str(&content).ok()
    } else if let Ok(ref out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        serde_json::from_str(&stdout).ok()
    } else {
        None
    };

    let elapsed = start_time.elapsed().as_millis() as u64;

    if let Some(res) = parsed_res {
        let status = MultimodalStatus {
            status: res.status,
            visual_analysis: res.visual_analysis,
            audio_analysis: res.audio_analysis,
            temporal_analysis: res.temporal_analysis,
            candidate_count_analyzed: res.candidate_count_analyzed,
            fallback_to_v92: res.fallback_to_v92,
            failure_reason: res.failure_reason,
            failed_modality: res.failed_modality,
            processing_time_ms: res.processing_time_ms.max(elapsed),
            source_duration,
            analyzed_start: 0.0,
            analyzed_end: source_duration,
            uncovered_seconds: 0.0,
            windows_processed: 1,
            windows_failed: 0,
        };
        (res.candidates, status)
    } else {
        let stderr = output
            .as_ref()
            .map(|o| String::from_utf8_lossy(&o.stderr).to_string())
            .unwrap_or_default();
        let mut fallback_candidates = candidates.to_vec();
        for c in &mut fallback_candidates {
            c.multimodal_verified = false;
            c.multimodal_status = Some("FAILED".to_string());
            c.fallback_label =
                Some("v9.2 fallback â€” multimodal analysis unavailable".to_string());
            c.multimodal_evidence = Some(vec![
                "Multimodal evidence unavailable; ranking uses v9.2 semantic fallback.".to_string(),
            ]);
        }
        let status = MultimodalStatus {
            status: "FAILED".to_string(),
            visual_analysis: "FAILED".to_string(),
            audio_analysis: "FAILED".to_string(),
            temporal_analysis: "FAILED".to_string(),
            candidate_count_analyzed: total_count,
            fallback_to_v92: true,
            failure_reason: Some(format!("Multimodal analyzer process failed: {stderr}")),
            failed_modality: Some("Analyzer Runtime".to_string()),
            processing_time_ms: elapsed,
            source_duration,
            analyzed_start: 0.0,
            analyzed_end: source_duration,
            uncovered_seconds: 0.0,
            windows_processed: 1,
            windows_failed: 0,
        };
        (fallback_candidates, status)
    }
}

/// Wall-clock budget for the multimodal hook analyzer.
///
/// Cost is linear in source runtime (full decode for visual, full audio scan
/// for acoustic); the budget scales with duration under a floor and a ceiling.
const MULTIMODAL_TIMEOUT_FLOOR_SEC: f64 = 300.0;
const MULTIMODAL_TIMEOUT_REALTIME_FACTOR: f64 = 4.0;
const MULTIMODAL_TIMEOUT_CEILING_SEC: f64 = 2400.0;

fn multimodal_timeout_for(duration_sec: f64) -> std::time::Duration {
    let secs = if duration_sec.is_finite() && duration_sec > 0.0 {
        MULTIMODAL_TIMEOUT_REALTIME_FACTOR * duration_sec
    } else {
        MULTIMODAL_TIMEOUT_FLOOR_SEC
    };
    std::time::Duration::from_secs_f64(
        secs.clamp(MULTIMODAL_TIMEOUT_FLOOR_SEC, MULTIMODAL_TIMEOUT_CEILING_SEC),
    )
}

/// success/stdout/stderr carrier for the bounded multimodal analyzer result.
struct MultimodalOutput {
    _success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCropResult {
    pub x: String,
    pub y: String,
    pub w: String,
    pub h: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CropRectExpr {
    #[serde(deserialize_with = "deserialize_string_or_number")]
    pub x: String,
    #[serde(deserialize_with = "deserialize_string_or_number")]
    pub y: String,
    #[serde(deserialize_with = "deserialize_string_or_number")]
    pub w: String,
    #[serde(deserialize_with = "deserialize_string_or_number")]
    pub h: String,
}

fn deserialize_string_or_number<'de, D>(deserializer: D) -> std::result::Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrNumberVisitor;

    impl<'de> serde::de::Visitor<'de> for StringOrNumberVisitor {
        type Value = String;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or a number")
        }

        fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.to_string())
        }

        fn visit_string<E>(self, v: String) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v)
        }

        fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.to_string())
        }

        fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.to_string())
        }

        fn visit_f64<E>(self, v: f64) -> std::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(v.to_string())
        }
    }

    deserializer.deserialize_any(StringOrNumberVisitor)
}

fn deserialize_optional_string_or_number<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StrOrNum {
        Str(String),
        I64(i64),
        U64(u64),
        F64(f64),
    }

    let opt = Option::<StrOrNum>::deserialize(deserializer)?;
    Ok(opt.map(|val| match val {
        StrOrNum::Str(s) => s,
        StrOrNum::I64(i) => i.to_string(),
        StrOrNum::U64(u) => u.to_string(),
        StrOrNum::F64(f) => f.to_string(),
    }))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode")]
pub enum LayoutSegment {
    #[serde(rename = "single")]
    Single {
        start: f64,
        end: f64,
        crop: CropRectExpr,
        #[serde(default)]
        face_bounds: Option<crate::captions::SubjectFaceBounds>,
    },
    #[serde(rename = "dual_stack")]
    DualStack {
        start: f64,
        end: f64,
        #[serde(default)]
        top_track_id: Option<i64>,
        #[serde(default)]
        bottom_track_id: Option<i64>,
        top_crop: CropRectExpr,
        bottom_crop: CropRectExpr,
        #[serde(default)]
        top_face_bounds: Option<crate::captions::SubjectFaceBounds>,
        #[serde(default)]
        bottom_face_bounds: Option<crate::captions::SubjectFaceBounds>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SmartFramingPlan {
    pub mode: String,
    pub x: String,
    pub y: String,
    pub w: String,
    pub h: String,
    pub segments: Vec<LayoutSegment>,
    pub face_bounds: Option<crate::captions::SubjectFaceBounds>,
    /// "original" = full-bleed 9:16 crop; "adaptive" = square inner composition
    /// (full-width square video vertically centered in the 9:16 canvas).
    #[serde(default = "default_original_framing")]
    pub framing: String,
    #[serde(default)]
    pub is_emergency_fallback: bool,
    #[serde(default)]
    pub fallback_reason: Option<String>,
    /// Phase 2 Speaker Intelligence: Re-ID gallery + fused active-speaker
    /// intervals emitted by speaker_tracker.py. Deserialized when present,
    /// never re-serialized into downstream plan JSON, and never consulted by
    /// the deterministic framing consumers.
    #[serde(
        rename = "speakerIntel",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub speaker_intel: Option<serde_json::Value>,
}

fn default_original_framing() -> String {
    "original".to_string()
}

#[derive(Deserialize)]
struct RawSmartFramingPlan {
    #[serde(default = "default_single_mode")]
    mode: String,
    #[serde(default, deserialize_with = "deserialize_optional_string_or_number")]
    x: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string_or_number")]
    y: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string_or_number")]
    w: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_string_or_number")]
    h: Option<String>,
    #[serde(default)]
    segments: Option<Vec<LayoutSegment>>,
    #[serde(default)]
    face_bounds: Option<crate::captions::SubjectFaceBounds>,
    #[serde(default = "default_original_framing")]
    framing: String,
    #[serde(default)]
    is_emergency_fallback: Option<bool>,
    #[serde(default)]
    fallback_reason: Option<String>,
    #[serde(rename = "speakerIntel", default)]
    speaker_intel: Option<serde_json::Value>,
}

fn default_single_mode() -> String {
    "single".to_string()
}

impl<'de> Deserialize<'de> for SmartFramingPlan {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawSmartFramingPlan::deserialize(deserializer)?;
        let x = raw.x.unwrap_or_else(|| "0".to_string());
        let y = raw.y.unwrap_or_else(|| "0".to_string());
        let w = raw.w.unwrap_or_else(|| "0".to_string());
        let h = raw.h.unwrap_or_else(|| "0".to_string());

        let segments = match raw.segments {
            Some(segs) if !segs.is_empty() => segs,
            _ => vec![LayoutSegment::Single {
                start: 0.0,
                end: 0.0,
                crop: CropRectExpr {
                    x: x.clone(),
                    y: y.clone(),
                    w: w.clone(),
                    h: h.clone(),
                },
                face_bounds: raw.face_bounds.clone(),
            }],
        };

        Ok(SmartFramingPlan {
            mode: raw.mode,
            x,
            y,
            w,
            h,
            segments,
            face_bounds: raw.face_bounds,
            framing: raw.framing,
            is_emergency_fallback: raw.is_emergency_fallback.unwrap_or(false),
            fallback_reason: raw.fallback_reason,
            speaker_intel: raw.speaker_intel,
        })
    }
}

impl Default for SmartFramingPlan {
    fn default() -> Self {
        Self {
            mode: "single".to_string(),
            x: "0".to_string(),
            y: "0".to_string(),
            w: "0".to_string(),
            h: "0".to_string(),
            segments: vec![],
            face_bounds: None,
            framing: default_original_framing(),
            is_emergency_fallback: false,
            fallback_reason: None,
            speaker_intel: None,
        }
    }
}

impl SmartFramingPlan {
    pub fn single_fallback(x: String, y: String, w: String, h: String, clip_dur: f64) -> Self {
        let crop = CropRectExpr {
            x: x.clone(),
            y: y.clone(),
            w: w.clone(),
            h: h.clone(),
        };
        Self {
            mode: "single".to_string(),
            x,
            y,
            w,
            h,
            segments: vec![LayoutSegment::Single {
                start: 0.0,
                end: clip_dur.max(0.0),
                crop,
                face_bounds: None,
            }],
            face_bounds: None,
            framing: default_original_framing(),
            is_emergency_fallback: true,
            fallback_reason: Some("single_fallback".to_string()),
            speaker_intel: None,
        }
    }

    pub fn segment_at(&self, t: f64) -> Option<&LayoutSegment> {
        if self.segments.is_empty() {
            return None;
        }
        let is_single = self.segments.len() == 1;
        if is_single {
            let s = &self.segments[0];
            let (start, end) = match s {
                LayoutSegment::Single { start, end, .. } => (*start, *end),
                LayoutSegment::DualStack { start, end, .. } => (*start, *end),
            };
            let s_end = if end <= 0.0 { f64::MAX } else { end };
            if t >= start && t <= s_end {
                return Some(s);
            }
            return Some(s);
        }

        // Multi-segment plan: reject/skip malformed segments where end <= start or end <= 0
        let valid_segments: Vec<&LayoutSegment> = self
            .segments
            .iter()
            .filter(|s| {
                let (start, end) = match s {
                    LayoutSegment::Single { start, end, .. } => (*start, *end),
                    LayoutSegment::DualStack { start, end, .. } => (*start, *end),
                };
                end > start && end > 0.0
            })
            .collect();

        if valid_segments.is_empty() {
            return self.segments.first();
        }

        for (i, s) in valid_segments.iter().enumerate() {
            let (start, end) = match s {
                LayoutSegment::Single { start, end, .. } => (*start, *end),
                LayoutSegment::DualStack { start, end, .. } => (*start, *end),
            };
            if t >= start && t <= end {
                return Some(s);
            }
            // Gap bridging: if t falls in an unmapped timeline gap before the next segment,
            // map it to the current segment so captions don't lose layout context
            if i + 1 < valid_segments.len() {
                let next_start = match valid_segments[i + 1] {
                    LayoutSegment::Single { start, .. } => *start,
                    LayoutSegment::DualStack { start, .. } => *start,
                };
                if t > end && t < next_start {
                    return Some(s);
                }
            }
        }

        let first_start = match valid_segments[0] {
            LayoutSegment::Single { start, .. } => *start,
            LayoutSegment::DualStack { start, .. } => *start,
        };
        if t < first_start {
            return Some(valid_segments[0]);
        }
        Some(valid_segments[valid_segments.len() - 1])
    }

    #[allow(dead_code)]
    pub fn to_speaker_crop_result(&self) -> SpeakerCropResult {
        SpeakerCropResult {
            x: self.x.clone(),
            y: self.y.clone(),
            w: self.w.clone(),
            h: self.h.clone(),
        }
    }
}

impl From<SmartFramingPlan> for SpeakerCropResult {
    fn from(plan: SmartFramingPlan) -> Self {
        SpeakerCropResult {
            x: plan.x,
            y: plan.y,
            w: plan.w,
            h: plan.h,
        }
    }
}

impl From<&SmartFramingPlan> for SpeakerCropResult {
    fn from(plan: &SmartFramingPlan) -> Self {
        plan.to_speaker_crop_result()
    }
}

/// Optional Speaker Intelligence sidecar inputs handed to speaker_tracker.py:
/// cached source-level diarization (absolute-time segments) and the persisted
/// Re-ID gallery for cross-render track association. `None` keeps the legacy
/// CLI contract (transcript-derived evidence only).
pub struct SpeakerIntelSidecarInputs<'a> {
    pub diarization_json: &'a str,
    pub gallery_json: &'a str,
    /// Phase 3 scene intelligence: cached PySceneDetect boundary JSON
    /// ({"scenes":[{"sceneId","start","end"},...]}, absolute source seconds).
    /// None keeps the tracker's own cut detection as the sole source.
    pub scene_cuts_json: Option<&'a str>,
}

pub fn detect_speaker_crop_params(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    iw: i64,
    ih: i64,
    crop_w: i64,
    transcript_words: Option<&[TranscriptWord]>,
    framing_mode: &str,
) -> SmartFramingPlan {
    detect_speaker_crop_params_with_intel(
        source_path,
        start_sec,
        end_sec,
        iw,
        ih,
        crop_w,
        transcript_words,
        framing_mode,
        None,
    )
}

pub fn detect_speaker_crop_params_with_intel(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    iw: i64,
    ih: i64,
    crop_w: i64,
    transcript_words: Option<&[TranscriptWord]>,
    framing_mode: &str,
    speaker_intel_inputs: Option<&SpeakerIntelSidecarInputs<'_>>,
) -> SmartFramingPlan {
    let clip_dur = (end_sec - start_sec).max(0.0);
    // Adaptive Framing crops a square (side = min(iw, ih)); Original 9:16 keeps
    // the full-height 9:16 crop. The horizontal centering bounds and the
    // fallback default_x passed to the tracker must match the effective crop,
    // otherwise the tracker's tail-fill keyframes land off-center.
    let effective_crop_w = if framing_mode.eq_ignore_ascii_case("adaptive") {
        iw.min(ih)
    } else {
        crop_w
    };
    let max_x = (iw - effective_crop_w).max(0);
    let default_x = max_x / 2;

    // A square crop on a portrait source can be exactly the frame width
    // (max_x == 0). That is a valid full-width composition, not a degenerate
    // case, so only the legacy early-return path is taken for original mode.
    if max_x == 0 && !framing_mode.eq_ignore_ascii_case("adaptive") {
        let mut plan = SmartFramingPlan::single_fallback(
            "0".to_string(),
            "0".to_string(),
            iw.to_string(),
            ih.to_string(),
            clip_dur,
        );
        plan.is_emergency_fallback = false;
        plan.fallback_reason = None;
        return plan;
    }

    let start_ms = (start_sec * 1000.0) as i64;
    let end_ms = (end_sec * 1000.0) as i64;

    // 1. Try the standalone multi-speaker script if available
    if let Some(script_path) = find_speaker_tracker_script() {
        let python_cmd = find_python_cmd();


        // D4b: the tracker can die from a transient native crash (non-zero exit,
        // no Python traceback) — e.g. memory pressure from concurrent CV work. One
        // bounded retry distinguishes such transient failures from systematic ones;
        // timeouts are NEVER retried (the budget is the guard), and a missing script
        // never reaches this loop.
        const TRACKER_MAX_ATTEMPTS: usize = 2;
        for tracker_attempt in 1..=TRACKER_MAX_ATTEMPTS {
            // Write transcript words to a temp JSON file so the script can use
            // speaker labels for diarization-driven camera keyframes.
            let temp_transcript_path: Option<std::path::PathBuf> = transcript_words.and_then(|words| {
                if words.is_empty() {
                    return None;
                }
                let tmp = std::env::temp_dir()
                    .join(format!("autoshorts_tracker_{}.json", uuid::Uuid::new_v4()));
                serde_json::to_string(words)
                    .ok()
                    .and_then(|json| std::fs::write(&tmp, json).ok().map(|_| tmp))
            });

            let mut cmd = Command::new(&python_cmd);
            cmd.arg(&script_path)
                .arg(source_path)
                .arg(start_ms.to_string())
                .arg(end_ms.to_string())
                .arg(crop_w.to_string())
                .arg(max_x.to_string())
                .arg(default_x.to_string());

            if let Some(ref tmp_path) = temp_transcript_path {
                cmd.arg(tmp_path);
            }

            // Adaptive Framing (10.0): per-project framing choice. "original"
            // leaves the legacy behavior untouched; "adaptive" disables DualFrame
            // and relaxes the two-shot fit threshold on the Python side.
            cmd.env("AUTOSHORTS_FRAMING_MODE", framing_mode);

            // Phase 2 Speaker Intelligence: cached diarization + persisted Re-ID
            // gallery (source-level, absolute-time) when the engine produced them.
            // Feature flags flow through inherited env vars to the sidecar.
            if let Some(inputs) = speaker_intel_inputs {
                cmd.arg("--diarization-json").arg(inputs.diarization_json);
                cmd.arg("--gallery-json").arg(inputs.gallery_json);
                if let Some(scenes) = inputs.scene_cuts_json {
                    cmd.arg("--scene-cuts-json").arg(scenes);
                }
            }

            // Bounded execution. The tracker decodes every frame of the candidate
            // range, so cost is linear in clip duration; budget accordingly rather
            // than calling `.output()` and waiting forever. A hung or wedged child
            // here presented as a pipeline freeze right after speaker mapping.
            let tracker_budget = tracker_timeout_for(clip_dur);
            eprintln!(
                "[Framing] TRACKER START clip={:.1}s budget={:.0}s mode={}",
                clip_dur,
                tracker_budget.as_secs_f64(),
                framing_mode
            );
            let tracker_started = std::time::Instant::now();
            let result = match crate::proc_guard::run_bounded(
                &mut cmd,
                tracker_budget,
                "Framing/tracker",
            ) {
                Ok(o) => {
                    // The terminal outcome is reported AFTER the child's telemetry
                    // is forwarded below. Printing it here made a timeout appear
                    // BEFORE the "[Framing] START"/"[CV Runtime]" lines that the
                    // killed child had already written, which looked like a
                    // duplicate/orphan framing process that did not exist.
                    // The downstream parser reads exactly success/stderr/stdout.
                    Ok(TrackerOutput {
                        success: o.success,
                        timed_out: o.timed_out,
                        stderr: o.stderr.into_bytes(),
                        stdout: o.stdout.into_bytes(),
                    })
                }
                Err(e) => Err(e),
            };

            // Clean up the temp transcript file regardless of outcome
            if let Some(ref tmp_path) = temp_transcript_path {
                let _ = std::fs::remove_file(tmp_path);
            }

            if let Ok(out) = result {
                // Forward visible runtime CV & DualFrame telemetry to stderr for
                // terminal visibility. These are the child's OWN progress lines and
                // must appear BEFORE the terminal outcome, otherwise a timeout
                // reads as though a second/orphan framing process continued after
                // the kill.
                let stderr_str = String::from_utf8_lossy(&out.stderr);
                for line in stderr_str.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("[CV Backend]")
                        || trimmed.starts_with("[CV Runtime]")
                        || trimmed.starts_with("[DualFrame]")
                        || trimmed.starts_with("[Framing]")
                        || trimmed.starts_with("[ReID]")
                        || trimmed.starts_with("[SceneIntel]")
                    {
                        eprintln!("{}", trimmed);
                    }
                }
                let stdout_str = String::from_utf8_lossy(&out.stdout);
                for line in stdout_str.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("[CV Backend]")
                        || trimmed.starts_with("[CV Runtime]")
                        || trimmed.starts_with("[DualFrame]")
                    {
                        eprintln!("{}", trimmed);
                    }
                }

                // Terminal outcome, reported AFTER telemetry. A timeout is reported
                // as a TIMEOUT (never "script failed"), and a non-zero exit is
                // reported with the real exit code. Both are fallbacks, and both
                // are labelled so Render QA success is never mistaken for Adaptive
                // Framing success.
                if out.timed_out {
                    eprintln!(
                        "[Framing] TIMEOUT after {:.0}s (clip={:.1}s, budget={:.0}s) -> adaptive framing did NOT complete; deterministic fallback",
                        tracker_started.elapsed().as_secs_f64(),
                        clip_dur,
                        tracker_budget.as_secs_f64()
                    );
                } else if !out.success {
                    // Distinguish a genuine crash from benign stderr noise. A
                    // library UserWarning on stderr (e.g. torchreid's "Cython
                    // evaluation ... is unavailable") is informational and is NOT a
                    // script failure; only the exit code decides.
                    let stderr_text = String::from_utf8_lossy(&out.stderr);
                    let has_traceback = stderr_text.contains("Traceback (most recent call last)")
                        || stderr_text.contains("ModuleNotFoundError")
                        || stderr_text.contains("ImportError");
                    eprintln!(
                        "[Framing] FAILED exit={:?} after {:.1}s{}{}",
                        out.success,
                        tracker_started.elapsed().as_secs_f64(),
                        if has_traceback { " (python exception)" } else { " (no python traceback in stderr)" },
                        crate::proc_guard::sanitize_stderr(&stderr_text, 400)
                    );
                } else {
                    eprintln!(
                        "[Framing] TRACKER COMPLETE in {:.1}s",
                        tracker_started.elapsed().as_secs_f64()
                    );
                }

                if out.success {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let valid_lines: Vec<&str> = stdout
                        .lines()
                        .map(|l| l.trim())
                        .filter(|l| !l.is_empty() && !l.starts_with('[') && !l.contains("WARN"))
                        .collect();

                    if let Some(&last_line) = valid_lines.last() {
                        let cleaned = last_line.trim().to_string();
                        if !cleaned.is_empty() {
                            let mode = if transcript_words.map(|w| w.len()).unwrap_or(0) > 0 {
                                "multi-speaker"
                            } else {
                                "single-speaker"
                            };

                            // Attempt to parse SmartFramingPlan JSON (supports both multi-segment and legacy single crop JSON)
                            if let Ok(mut plan) = serde_json::from_str::<SmartFramingPlan>(&cleaned) {
                                plan.is_emergency_fallback = false;
                                plan.fallback_reason = None;
                                eprintln!(
                                    "[Smart Framing/{mode}] dynamic crop trajectory (mode: {}): x={}, y={}, w={}, h={}, segments={}",
                                    plan.mode, plan.x, plan.y, plan.w, plan.h, plan.segments.len()
                                );
                                if plan.segments.len() == 1 {
                                    if let LayoutSegment::Single { ref mut end, .. } = plan.segments[0]
                                    {
                                        if *end == 0.0 && clip_dur > 0.0 {
                                            *end = clip_dur;
                                        }
                                    }
                                }
                                return plan;
                            }

                            // Fallback for plain expression string
                            eprintln!(
                                "[Smart Framing/{mode}] crop trajectory: {} (max_x: {}, default_x: {})",
                                cleaned, max_x, default_x
                            );
                            let mut plan = SmartFramingPlan::single_fallback(
                                cleaned,
                                "0".to_string(),
                                crop_w.to_string(),
                                ih.to_string(),
                                clip_dur,
                            );
                            plan.is_emergency_fallback = false;
                            plan.fallback_reason = None;
                            return plan;
                        }
                    }
                } else {
                    // Do NOT dump the entire stderr blob here: a killed child's
                    // buffer contains progress lines plus library warnings (e.g.
                    // torchreid's Cython UserWarning), which are informational, not
                    // failures. The terminal outcome above already reported the real
                    // cause (timeout vs non-zero exit). Here we only surface the
                    // specific, actionable dependency errors.
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    if stderr.contains("No module named 'cv2'")
                        || stderr.contains("OpenCV (cv2) is not installed")
                    {
                        eprintln!(
                            "[Smart Framing] DEPENDENCY ERROR: OpenCV (cv2) is not installed in the Python environment used by AutoShorts.\n{}",
                            crate::proc_guard::sanitize_stderr(&stderr, 300)
                        );
                    } else if stderr.contains("Traceback (most recent call last)") {
                        eprintln!(
                            "[Smart Framing] PYTHON EXCEPTION:\n{}",
                            crate::proc_guard::sanitize_stderr(&stderr, 600)
                        );
                    }
                    // Otherwise: benign library warnings only. Already reported
                    // above as a timeout or non-zero exit; nothing more to add.

                // D4b: a transient crash (non-timeout, non-zero exit) gets ONE bounded
                // retry; a timeout never retries (the budget is the guard).
                if !out.success && !out.timed_out && tracker_attempt < TRACKER_MAX_ATTEMPTS {
                    eprintln!(
                        "[Framing] RETRY {}/{} after tracker crash (bounded re-run)",
                        tracker_attempt + 1,
                        TRACKER_MAX_ATTEMPTS
                    );
                    continue;
                }
                }
            // Any outcome other than a retryable crash exits here: spawn error,
            // timeout, success (parsed or plain-string plan), or a deterministic
            // parse failure that a retry cannot change.
            break;
        }
        }
    }

    eprintln!(
        "[Framing] FALLBACK reason=adaptive_tracker_unavailable — emergency center crop X: {} (max_x: {}). THIS IS NOT ADAPTIVE FRAMING SUCCESS.",
        default_x, max_x
    );
    let mut plan = SmartFramingPlan::single_fallback(
        default_x.to_string(),
        "0".to_string(),
        crop_w.to_string(),
        ih.to_string(),
        clip_dur,
    );
    plan.is_emergency_fallback = true;
    plan.fallback_reason = Some("tracker_missing_or_failed".to_string());
    plan
}

/// Escapes filesystem paths for use inside single-quoted FFmpeg filtergraph arguments (e.g. subtitles='...':fontsdir='...').
/// Performs:
/// 1. Backslash normalization to forward slashes (\ -> /)
/// 2. Windows drive-letter colon escaping (: -> \:)
/// 3. Filtergraph single-quote escaping (' -> '\'')
pub fn escape_ffmpeg_filter_path(p: &Path) -> String {
    p.to_string_lossy()
        .replace('\\', "/")
        .replace(':', r"\:")
        .replace('\'', r"'\\\''")
}

/// Resolves the `:fontsdir` parameter for ASS subtitles and strictly verifies that:
/// 1. The bundled font directory exists.
/// 2. If the ASS file requires a template font (Bebas Neue, Montserrat, Inter, Poppins),
///    the exact required TTF file exists in that directory.
/// Fails explicitly rather than silently falling back to OS fonts.
pub fn resolve_and_verify_fonts_param(ass_subtitle_path: Option<&Path>) -> Result<Option<String>> {
    let ass_path = match ass_subtitle_path {
        Some(p) => p,
        None => return Ok(None),
    };

    let mut required_fonts: Vec<(String, &'static str)> = Vec::new();
    if ass_path.exists() {
        if let Ok(content) = std::fs::read_to_string(ass_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("Style:") {
                    let parts: Vec<&str> = trimmed["Style:".len()..].split(',').collect();
                    if parts.len() >= 2 {
                        let font_name = parts[1].trim();
                        if let Some(expected_file) =
                            crate::captions::required_font_filename(font_name)
                        {
                            if !required_fonts.iter().any(|(f, _)| f == font_name) {
                                required_fonts.push((font_name.to_string(), expected_file));
                            }
                        }
                    }
                } else if trimmed.starts_with("Dialogue:") {
                    // Check for inline \fn<FontName> tags
                    let mut search_idx = 0;
                    while let Some(pos) = trimmed[search_idx..].find(r"\fn") {
                        let start = search_idx + pos + 3;
                        let end = trimmed[start..]
                            .find(|c| c == '\\' || c == '}')
                            .map(|e| start + e)
                            .unwrap_or(trimmed.len());
                        let font_name = trimmed[start..end].trim();
                        if let Some(expected_file) =
                            crate::captions::required_font_filename(font_name)
                        {
                            if !required_fonts.iter().any(|(f, _)| f == font_name) {
                                required_fonts.push((font_name.to_string(), expected_file));
                            }
                        }
                        search_idx = end;
                    }
                }
            }
        }
    }

    if required_fonts.is_empty() {
        return Ok(None);
    }

    let fonts_dir_opt = find_fonts_dir();
    if let Some(ref fonts_dir) = fonts_dir_opt {
        if !fonts_dir.exists() {
            return Err(anyhow!(
                "Bundled fonts directory does not exist at {:?}. Cannot render required template fonts: {:?}",
                fonts_dir,
                required_fonts.iter().map(|(f, _)| f.as_str()).collect::<Vec<_>>()
            ));
        }

        for (font, expected_file) in &required_fonts {
            let font_file_path = fonts_dir.join(expected_file);
            if !font_file_path.exists() {
                return Err(anyhow!(
                    "Mandatory font file '{}' for template font family '{}' was not found in directory '{:?}'. Silent fallback to OS fonts is prohibited.",
                    expected_file,
                    font,
                    fonts_dir
                ));
            }
        }

        let escaped_fonts = escape_ffmpeg_filter_path(fonts_dir);
        return Ok(Some(format!(":fontsdir='{}'", escaped_fonts)));
    }

    Ok(None)
}

/// Validates and repairs the layout segment timeline for multi-segment plans:
/// 1. Rejects/skips malformed segments (end <= start or end <= 0.0).
/// 2. Ensures contiguous temporal coverage across [0.0, duration_sec] without gaps or overlaps.
/// 3. In single-segment plans, preserves legitimate open-ended segments (end <= 0 expands to duration_sec).
pub fn validate_and_repair_timeline(
    segments: &[LayoutSegment],
    duration_sec: f64,
) -> Vec<LayoutSegment> {
    if segments.is_empty() {
        return Vec::new();
    }

    if segments.len() == 1 {
        let mut seg = segments[0].clone();
        match &mut seg {
            LayoutSegment::Single { start, end, .. } => {
                if *end <= 0.0 || *end <= *start {
                    *start = 0.0;
                    *end = duration_sec;
                }
            }
            LayoutSegment::DualStack { start, end, .. } => {
                if *end <= 0.0 || *end <= *start {
                    *start = 0.0;
                    *end = duration_sec;
                }
            }
        }
        return vec![seg];
    }

    // 1. Filter out malformed segments where end <= start or end <= 0
    let mut valid: Vec<LayoutSegment> = segments
        .iter()
        .filter(|s| {
            let (start, end) = match s {
                LayoutSegment::Single { start, end, .. } => (*start, *end),
                LayoutSegment::DualStack { start, end, .. } => (*start, *end),
            };
            end > start && end > 0.0
        })
        .cloned()
        .collect();

    if valid.is_empty() {
        return Vec::new();
    }

    if valid.len() == 1 {
        match &mut valid[0] {
            LayoutSegment::Single { start, end, .. } => {
                *start = 0.0;
                *end = duration_sec;
            }
            LayoutSegment::DualStack { start, end, .. } => {
                *start = 0.0;
                *end = duration_sec;
            }
        }
        return valid;
    }

    // 2. Validate and bridge the timeline: guarantee contiguous coverage [0.0, duration_sec]
    match &mut valid[0] {
        LayoutSegment::Single { start, .. } => *start = 0.0,
        LayoutSegment::DualStack { start, .. } => *start = 0.0,
    }

    for i in 0..(valid.len() - 1) {
        let end_i = match &valid[i] {
            LayoutSegment::Single { end, .. } => *end,
            LayoutSegment::DualStack { end, .. } => *end,
        };
        let start_next = match &valid[i + 1] {
            LayoutSegment::Single { start, .. } => *start,
            LayoutSegment::DualStack { start, .. } => *start,
        };

        // Bridge gaps or resolve overlaps: ensure next segment starts exactly where current ends
        if (start_next - end_i).abs() > 1e-4 {
            match &mut valid[i + 1] {
                LayoutSegment::Single { start, .. } => *start = end_i,
                LayoutSegment::DualStack { start, .. } => *start = end_i,
            }
        }
    }

    // Ensure the last segment reaches duration_sec
    if let Some(last) = valid.last_mut() {
        match last {
            LayoutSegment::Single { end, .. } => {
                if *end < duration_sec {
                    *end = duration_sec;
                }
            }
            LayoutSegment::DualStack { end, .. } => {
                if *end < duration_sec {
                    *end = duration_sec;
                }
            }
        }
    }

    valid
}

pub fn build_layout_filtergraph(
    plan: &SmartFramingPlan,
    duration_sec: f64,
    ass_subtitle_path: Option<&Path>,
    drawtext_filters: Option<&str>,
) -> Result<(String, String)> {
    let fonts_opt = resolve_and_verify_fonts_param(ass_subtitle_path)?.unwrap_or_default();

    let validated_segments = validate_and_repair_timeline(&plan.segments, duration_sec);
    let effective_segments = if validated_segments.is_empty() {
        &plan.segments[..]
    } else {
        &validated_segments[..]
    };

    let has_dual = effective_segments
        .iter()
        .any(|s| matches!(s, LayoutSegment::DualStack { .. }));
    let is_single = !has_dual && effective_segments.len() <= 1;

    if is_single {
        let (crop_w, crop_h, crop_x, crop_y) =
            if let Some(LayoutSegment::Single { crop, .. }) = effective_segments.first() {
                (
                    crop.w.as_str(),
                    crop.h.as_str(),
                    crop.x.as_str(),
                    crop.y.as_str(),
                )
            } else {
                (
                    plan.w.as_str(),
                    plan.h.as_str(),
                    plan.x.as_str(),
                    plan.y.as_str(),
                )
            };

        let crop_filter = format!(
            "crop=w={}:h={}:x='{}':y='{}'",
            crop_w, crop_h, crop_x, crop_y
        );
        // Adaptive Framing: the crop is a SQUARE inner composition rendered
        // full-width and vertically centered in the 9:16 canvas, reproducing
        // the reference composition (black canvas above and below the video).
        // Original 9:16: full-bleed stretch of the 9:16 crop.
        let scale_filter = if plan.framing == "adaptive" {
            "scale=1080:1080:flags=lanczos+accurate_rnd+full_chroma_int,pad=width=1080:height=1920:x=0:y=420:color=black,setsar=1".to_string()
        } else {
            "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int".to_string()
        };
        let mut filter = format!("{},{}", crop_filter, scale_filter);

        if let Some(ass_path) = ass_subtitle_path {
            let escaped_ass = escape_ffmpeg_filter_path(ass_path);
            filter = format!("{},subtitles='{}'{}", filter, escaped_ass, fonts_opt);
        } else if let Some(drawtext) = drawtext_filters {
            let trimmed = drawtext.trim();
            if !trimmed.is_empty() {
                filter = format!("{},{}", filter, trimmed);
            }
        }

        return Ok((filter, String::new()));
    }

    // Multi-segment or dual-stack layout
    let total_branches: usize = effective_segments
        .iter()
        .map(|s| match s {
            LayoutSegment::Single { .. } => 1,
            LayoutSegment::DualStack { .. } => 2,
        })
        .sum();

    let mut graph = String::new();

    // Explicit source stream split: [0:v]split=K[b0][b1]...[b_{K-1}];
    if total_branches > 1 {
        graph.push_str(&format!("[0:v]split={}", total_branches));
        for b in 0..total_branches {
            graph.push_str(&format!("[b{}]", b));
        }
        graph.push(';');
    }

    let format_time = |val: f64| -> String {
        if (val - val.round()).abs() < 1e-6 {
            format!("{:.0}", val)
        } else {
            let s = format!("{:.3}", val);
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        }
    };

    let mut branch_idx = 0;
    for (i, seg) in effective_segments.iter().enumerate() {
        match seg {
            LayoutSegment::Single {
                start, end, crop, ..
            } => {
                let seg_end = (*end).min(duration_sec);
                let b_in = if total_branches > 1 {
                    format!("[b{}]", branch_idx)
                } else {
                    "[0:v]".to_string()
                };
                branch_idx += 1;

                let scale_seg = if plan.framing == "adaptive" {
                    "scale=1080:1080:flags=lanczos+accurate_rnd+full_chroma_int,pad=width=1080:height=1920:x=0:y=420:color=black,setsar=1"
                } else {
                    "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1"
                };
                graph.push_str(&format!(
                    "{}trim=start={}:end={},setpts=PTS-STARTPTS,crop=w={}:h={}:x='{}':y='{}',{}[v_seg_{}];",
                    b_in,
                    format_time(*start),
                    format_time(seg_end),
                    crop.w,
                    crop.h,
                    crop.x,
                    crop.y,
                    scale_seg,
                    i
                ));
            }
            LayoutSegment::DualStack {
                start,
                end,
                top_crop,
                bottom_crop,
                ..
            } => {
                let seg_end = (*end).min(duration_sec);
                let b_top = format!("[b{}]", branch_idx);
                let b_bot = format!("[b{}]", branch_idx + 1);
                branch_idx += 2;

                // Top panel: scale to 1080:960
                graph.push_str(&format!(
                    "{}trim=start={}:end={},setpts=PTS-STARTPTS,crop=w={}:h={}:x='{}':y='{}',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_top_{}];",
                    b_top,
                    format_time(*start),
                    format_time(seg_end),
                    top_crop.w,
                    top_crop.h,
                    top_crop.x,
                    top_crop.y,
                    i
                ));

                // Bottom panel: scale to 1080:960
                graph.push_str(&format!(
                    "{}trim=start={}:end={},setpts=PTS-STARTPTS,crop=w={}:h={}:x='{}':y='{}',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int,setsar=1[v_bot_{}];",
                    b_bot,
                    format_time(*start),
                    format_time(seg_end),
                    bottom_crop.w,
                    bottom_crop.h,
                    bottom_crop.x,
                    bottom_crop.y,
                    i
                ));

                // Vstack: 960 + 960 = 1920
                graph.push_str(&format!(
                    "[v_top_{}][v_bot_{}]vstack=inputs=2[v_seg_{}];",
                    i, i, i
                ));
            }
        }
    }

    // Concatenate segments
    let mut concat_inputs = String::new();
    for i in 0..effective_segments.len() {
        concat_inputs.push_str(&format!("[v_seg_{}]", i));
    }
    graph.push_str(&format!(
        "{}concat=n={}:v=1:a=0[v_composed]",
        concat_inputs,
        effective_segments.len()
    ));

    // Kinetic subtitles / drawtext overlay
    if let Some(ass_path) = ass_subtitle_path {
        let escaped_ass = escape_ffmpeg_filter_path(ass_path);
        graph.push_str(&format!(
            ";[v_composed]subtitles='{}'{}[v_final]",
            escaped_ass, fonts_opt
        ));
        Ok((graph, "v_final".to_string()))
    } else if let Some(drawtext) = drawtext_filters {
        let trimmed = drawtext.trim();
        if !trimmed.is_empty() {
            graph.push_str(&format!(";[v_composed]{}[v_final]", trimmed));
            Ok((graph, "v_final".to_string()))
        } else {
            Ok((graph, "v_composed".to_string()))
        }
    } else {
        Ok((graph, "v_composed".to_string()))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn render_flat_clip(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    output_path: &Path,
    ass_subtitle_path: Option<&Path>,
    drawtext_filters: Option<&str>,
    transcript_words: Option<&[TranscriptWord]>,
    precomputed_framing: Option<&SmartFramingPlan>,
    audio_filters: Option<&str>,
) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let start = format!("{start_sec:.3}");
    let duration_sec = (end_sec - start_sec).max(0.1);
    let duration = format!("{duration_sec:.3}");

    let probe = probe_media(source_path).ok();
    let has_video = probe.as_ref().map(|p| p.has_video).unwrap_or(false);
    let has_audio = probe
        .as_ref()
        .map(|p| p.audio_codec.is_some())
        .unwrap_or(false);
    let iw = probe.as_ref().and_then(|p| p.width).unwrap_or(1920);
    let ih = probe.as_ref().and_then(|p| p.height).unwrap_or(1080);

    // Audio Intelligence filters are applied only when the source actually
    // carries an audio stream (probed); otherwise the option is ignored so
    // the command stays byte-identical to the legacy render.
    let audio_chain = audio_filters
        .filter(|f| !f.trim().is_empty())
        .filter(|_| has_audio);

    let ffmpeg_bin = find_ffmpeg_cmd().unwrap_or_else(|| PathBuf::from("ffmpeg"));
    let mut cmd = Command::new(&ffmpeg_bin);
    cmd.args(["-y", "-ss", &start, "-i", source_path, "-t", &duration]);

    if has_video {
        let crop_params = match precomputed_framing {
            Some(plan) => plan.clone(),
            None => {
                let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
                if crop_w % 2 != 0 {
                    crop_w -= 1;
                }
                detect_speaker_crop_params(
                    source_path,
                    start_sec,
                    end_sec,
                    iw,
                    ih,
                    crop_w,
                    transcript_words,
                    "original",
                )
            }
        };

        let total_segments = crop_params.segments.len();
        let dual_segments = crop_params
            .segments
            .iter()
            .filter(|s| matches!(s, LayoutSegment::DualStack { .. }))
            .count();
        let single_segments = total_segments.saturating_sub(dual_segments);
        println!(
            "[DualFrame Render] Executing layout plan with {} segments ({} dual-stack, {} single)",
            total_segments, dual_segments, single_segments
        );

        let valid_ass = ass_subtitle_path.filter(|p| p.exists());
        let (filter, output_label) =
            build_layout_filtergraph(&crop_params, duration_sec, valid_ass, drawtext_filters)?;

        if !output_label.is_empty() {
            // Audio Intelligence: when corrections are active, append the
            // staged chain to the SAME filter_complex (audio is routed
            // through its own labeled branch) and map that label instead of
            // the optional `0:a?`. None keeps the legacy command identical.
            let (filter, audio_map) = match audio_chain {
                Some(chain) => (
                    format!("{};[0:a]{}[a_intel]", filter, chain.trim()),
                    vec![
                        "-map".to_string(),
                        format!("[{}]", output_label),
                        "-map".to_string(),
                        "[a_intel]".to_string(),
                    ],
                ),
                None => (
                    filter,
                    vec![
                        "-map".to_string(),
                        format!("[{}]", output_label),
                        "-map".to_string(),
                        "0:a?".to_string(),
                    ],
                ),
            };
            println!(
                "[FFmpeg Render] Complete -filter_complex filter string:\n{}",
                filter
            );
            cmd.args(["-filter_complex", &filter]);
            for a in &audio_map {
                cmd.arg(a);
            }
        } else {
            println!("[FFmpeg Render] Complete -vf filter string:\n{}", filter);
            cmd.args(["-vf", &filter]);
            if let Some(chain) = audio_chain {
                cmd.args(["-af", chain.trim()]);
            }
        }

        cmd.args([
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-movflags",
            "+faststart",
        ]);
    } else {
        cmd.arg("-vn");
        if let Some(chain) = audio_chain {
            cmd.args(["-af", chain.trim()]);
        }
    }

    cmd.args(["-c:a", "aac", "-b:a", "192k"]);
    cmd.arg(output_path);

    // Bounded render. `Command::output()` waits forever: a wedged ffmpeg, or a
    // child that fills the stderr pipe while the parent is blocked in wait(),
    // hangs the render thread with a half-written file on disk. On timeout the
    // process tree is killed and the partial output removed, so a truncated
    // encode can never be mistaken for a finished clip.
    let budget = render_timeout_for(start_sec, end_sec);
    println!(
        "[FFmpeg Render] START budget={:.0}s for {:.1}s of source",
        budget.as_secs_f64(),
        (end_sec - start_sec).max(0.0)
    );
    let started = std::time::Instant::now();
    let result = crate::proc_guard::run_bounded(&mut cmd, budget, "Render/ffmpeg");

    let output = match result {
        Ok(o) => o,
        Err(e) => {
            let _ = std::fs::remove_file(output_path);
            return Err(e).context("running ffmpeg clip render");
        }
    };

    if output.timed_out {
        // Partial output must not survive: downstream treats an existing file
        // as a rendered clip.
        let _ = std::fs::remove_file(output_path);
        return Err(anyhow!(
            "ffmpeg clip render TIMEOUT after {:.0}s (budget {:.0}s); process tree killed and partial output removed",
            output.elapsed.as_secs_f64(),
            budget.as_secs_f64()
        ));
    }

    if !output.success {
        let _ = std::fs::remove_file(output_path);
        let stderr = crate::proc_guard::sanitize_stderr(&output.stderr, 800);
        eprintln!(
            "[FFmpeg Render Error] ffmpeg exited rc={:?} in {:.1}s:\n{}",
            output.code,
            output.elapsed.as_secs_f64(),
            stderr
        );
        return Err(anyhow!(
            "ffmpeg clip render failed after {:.1}s (rc={:?}): {}",
            output.elapsed.as_secs_f64(),
            output.code,
            stderr.trim()
        ));
    }

    println!(
        "[FFmpeg Render] COMPLETE in {:.1}s -> {}",
        started.elapsed().as_secs_f64(),
        output_path.display()
    );

    Ok(output_path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_python_cmd_finds_valid_python_with_cv2() {
        let cmd = find_python_cmd();
        assert!(!cmd.is_empty(), "Python command should not be empty");
        let out = Command::new(&cmd)
            .args(["-c", "import cv2; print(cv2.__version__)"])
            .output();
        assert!(out.is_ok(), "Failed to execute python command: {}", cmd);
        let out = out.unwrap();
        assert!(
            out.status.success(),
            "Python failed to import cv2: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn test_find_ytdlp_cmd_discovers_project_venv_install() {
        // The project venv ships yt-dlp.exe in Scripts/. find_ytdlp_cmd() must
        // discover it (via PATH when the venv is activated, or via its .venv
        // candidate paths otherwise) so the YouTube import feature is available
        // without a system-wide yt-dlp install.
        let found = find_ytdlp_cmd();
        assert!(
            found.is_some(),
            "yt-dlp must be discoverable (project venv install expected)"
        );
        let path = found.unwrap();

        // A bare command name (e.g. "yt-dlp.exe" resolved from PATH when the
        // project venv is activated) has no directory component; Path::exists()
        // would be CWD-relative and meaningless for it. The real invariant is
        // runnability, which Command::new resolves via PATH.
        let has_dir_component = path
            .parent()
            .map(|p| !p.as_os_str().is_empty())
            .unwrap_or(false);
        if has_dir_component {
            assert!(
                path.exists(),
                "Discovered yt-dlp path must exist: {}",
                path.display()
            );
        }
        let out = Command::new(&path)
            .arg("--version")
            .output()
            .expect("yt-dlp --version must execute");
        assert!(
            out.status.success(),
            "Discovered yt-dlp must be runnable: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn test_command_exists_ytdlp_routes_to_find_ytdlp_cmd() {
        // command_exists("yt-dlp") must reflect the same discovery as find_ytdlp_cmd()
        // (environment_status.has_ytdlp depends on this routing).
        assert_eq!(command_exists("yt-dlp"), find_ytdlp_cmd().is_some());
    }

    #[test]
    fn test_find_ffmpeg_cmd_discovers_ffmpeg() {
        let found = find_ffmpeg_cmd();
        assert!(found.is_some(), "ffmpeg must be discoverable on system or PATH");
        let path = found.unwrap();
        let out = Command::new(&path)
            .arg("-version")
            .output()
            .expect("ffmpeg -version must execute");
        assert!(out.status.success(), "Discovered ffmpeg must be runnable");
    }

    #[test]
    fn test_find_ffprobe_cmd_discovers_ffprobe() {
        let found = find_ffprobe_cmd();
        assert!(found.is_some(), "ffprobe must be discoverable on system or PATH");
        let path = found.unwrap();
        let out = Command::new(&path)
            .arg("-version")
            .output()
            .expect("ffprobe -version must execute");
        assert!(out.status.success(), "Discovered ffprobe must be runnable");
    }

    #[test]
    fn test_command_exists_ffmpeg_routes_to_find_ffmpeg_cmd() {
        assert_eq!(command_exists("ffmpeg"), find_ffmpeg_cmd().is_some());
        assert_eq!(command_exists("ffprobe"), find_ffprobe_cmd().is_some());
    }

    static TRACKER_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_find_speaker_tracker_script_exists() {
        let _guard = TRACKER_TEST_MUTEX.lock().unwrap();
        let script = find_speaker_tracker_script();
        assert!(script.is_some(), "speaker_tracker.py should be located");
        assert!(script.unwrap().exists());
    }

    #[test]
    fn test_detect_speaker_crop_params_end_to_end() {
        let video_path = r"C:\Users\naksh\Downloads\AutoShorts_P1OyIznd1aY.mp4";
        if !std::path::Path::new(video_path).exists() {
            return;
        }

        let res =
            detect_speaker_crop_params(video_path, 0.0, 3.0, 1920, 1080, 608, None, "original");

        assert!(!res.x.is_empty());
        assert_ne!(
            res.x, "656",
            "Should not fall back to emergency default 656 when tracker runs successfully"
        );
        assert!(!res.w.is_empty());
        assert!(!res.h.is_empty());
    }

    #[test]
    fn test_render_flat_clip_end_to_end_on_failing_clip() {
        let video_path = r"C:\Users\naksh\Downloads\AutoShorts_sLGXktFmpEM.mp4";
        if !std::path::Path::new(video_path).exists() {
            return;
        }

        let out_file = std::env::temp_dir().join("test_render_6e5a505a.mp4");

        // Render 5 seconds of the failing clip 271.505 to 276.505 through render_flat_clip
        let res = render_flat_clip(
            video_path, 271.505, 276.505, &out_file, None, None, None, None, None,
        );

        assert!(res.is_ok(), "render_flat_clip failed: {:?}", res.err());
        let path = res.unwrap();
        assert!(path.exists());
        let size = std::fs::metadata(&path).unwrap().len();
        println!("[Integration Test] Rendered file size: {} bytes", size);
        assert!(size > 100_000, "Rendered file should not be zero bytes");
    }

    #[test]
    fn test_detect_speaker_crop_params_local_sample() {
        let _guard = TRACKER_TEST_MUTEX.lock().unwrap();
        let candidates = [
            r"../../Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
            r"Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
            r"../Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
            r"d:\College\Autoshorts 5.0\Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
        ];
        let mut video_path = None;
        for c in &candidates {
            if std::path::Path::new(c).exists() {
                video_path = Some(c.to_string());
                break;
            }
        }

        if let Some(path) = video_path {
            let res =
                detect_speaker_crop_params(&path, 15.0, 18.0, 1920, 1080, 608, None, "original");
            assert!(!res.x.is_empty(), "x expression should not be empty");
            assert_eq!(res.w, "608");
            assert_eq!(res.h, "1080");
        }
    }

    #[test]
    fn test_render_flat_clip_local_sample() {
        let candidates = [
            r"../../Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
            r"Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
            r"../Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
            r"d:\College\Autoshorts 5.0\Messi vs Ronaldo Fansï¼š The Psychology Explained [rssDTc086bk].mp4",
        ];
        let mut video_path = None;
        for c in &candidates {
            if std::path::Path::new(c).exists() {
                video_path = Some(c.to_string());
                break;
            }
        }

        if let Some(path) = video_path {
            let out_file = std::env::temp_dir().join("test_autoshorts5_render.mp4");
            let res = render_flat_clip(&path, 15.0, 18.0, &out_file, None, None, None, None, None);
            assert!(res.is_ok(), "render_flat_clip failed: {:?}", res.err());
            let p = res.unwrap();
            assert!(p.exists());
            let size = std::fs::metadata(&p).unwrap().len();
            println!(
                "[Render Verification] Rendered MP4 size: {} bytes at {}",
                size,
                p.display()
            );
            assert!(size > 50_000, "Rendered file should not be zero bytes");
        }
    }

    #[test]
    fn test_parse_legacy_single_crop_json() {
        let json_str = r#"{"x": "100", "y": "0", "w": "608", "h": "1080"}"#;
        let plan: SmartFramingPlan =
            serde_json::from_str(json_str).expect("Failed to deserialize legacy single crop JSON");
        assert_eq!(plan.mode, "single");
        assert_eq!(plan.x, "100");
        assert_eq!(plan.y, "0");
        assert_eq!(plan.w, "608");
        assert_eq!(plan.h, "1080");
        assert_eq!(plan.segments.len(), 1);
        match &plan.segments[0] {
            LayoutSegment::Single { start, crop, .. } => {
                assert_eq!(*start, 0.0);
                assert_eq!(crop.x, "100");
                assert_eq!(crop.y, "0");
                assert_eq!(crop.w, "608");
                assert_eq!(crop.h, "1080");
            }
            _ => panic!("Expected single layout segment"),
        }
    }

    #[test]
    fn test_parse_smart_framing_plan_with_dual_segments() {
        let json_str = r#"{
            "mode": "dual_stack",
            "x": "656",
            "y": "0",
            "w": "608",
            "h": "1080",
            "segments": [
                {
                    "mode": "single",
                    "start": 0.0,
                    "end": 2.5,
                    "crop": {
                        "x": "100",
                        "y": "0",
                        "w": "608",
                        "h": "1080"
                    }
                },
                {
                    "mode": "dual_stack",
                    "start": 2.5,
                    "end": 7.5,
                    "top_track_id": 1,
                    "bottom_track_id": 2,
                    "top_crop": {
                        "x": "200",
                        "y": "50",
                        "w": "1080",
                        "h": "960"
                    },
                    "bottom_crop": {
                        "x": "400",
                        "y": "50",
                        "w": "1080",
                        "h": "960"
                    }
                }
            ]
        }"#;
        let plan: SmartFramingPlan = serde_json::from_str(json_str)
            .expect("Failed to deserialize smart framing plan with dual segments");
        assert_eq!(plan.mode, "dual_stack");
        assert_eq!(plan.x, "656");
        assert_eq!(plan.y, "0");
        assert_eq!(plan.w, "608");
        assert_eq!(plan.h, "1080");
        assert_eq!(plan.segments.len(), 2);

        match &plan.segments[0] {
            LayoutSegment::Single {
                start, end, crop, ..
            } => {
                assert_eq!(*start, 0.0);
                assert_eq!(*end, 2.5);
                assert_eq!(crop.x, "100");
                assert_eq!(crop.y, "0");
                assert_eq!(crop.w, "608");
                assert_eq!(crop.h, "1080");
            }
            _ => panic!("Expected single segment at index 0"),
        }

        match &plan.segments[1] {
            LayoutSegment::DualStack {
                start,
                end,
                top_track_id,
                bottom_track_id,
                top_crop,
                bottom_crop,
                ..
            } => {
                assert_eq!(*start, 2.5);
                assert_eq!(*end, 7.5);
                assert_eq!(*top_track_id, Some(1));
                assert_eq!(*bottom_track_id, Some(2));
                assert_eq!(top_crop.x, "200");
                assert_eq!(top_crop.y, "50");
                assert_eq!(top_crop.w, "1080");
                assert_eq!(top_crop.h, "960");
                assert_eq!(bottom_crop.x, "400");
                assert_eq!(bottom_crop.y, "50");
                assert_eq!(bottom_crop.w, "1080");
                assert_eq!(bottom_crop.h, "960");
            }
            _ => panic!("Expected dual_stack segment at index 1"),
        }
    }

    #[test]
    fn test_smart_framing_plan_conversion_to_speaker_crop_result() {
        let plan = SmartFramingPlan::single_fallback(
            "120".to_string(),
            "0".to_string(),
            "608".to_string(),
            "1080".to_string(),
            5.0,
        );
        let res1 = plan.to_speaker_crop_result();
        assert_eq!(res1.x, "120");
        assert_eq!(res1.y, "0");
        assert_eq!(res1.w, "608");
        assert_eq!(res1.h, "1080");

        let res2: SpeakerCropResult = (&plan).into();
        assert_eq!(res2.x, "120");

        let res3: SpeakerCropResult = plan.into();
        assert_eq!(res3.x, "120");
    }

    #[test]
    fn test_build_single_layout_filtergraph_preserves_legacy_format() {
        let plan = SmartFramingPlan::single_fallback(
            "100".to_string(),
            "0".to_string(),
            "608".to_string(),
            "1080".to_string(),
            10.0,
        );
        let (filter, label) = build_layout_filtergraph(&plan, 10.0, None, None).unwrap();
        assert_eq!(
            filter,
            "crop=w=608:h=1080:x='100':y='0',scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int"
        );
        assert!(
            label.is_empty(),
            "Single layout should return empty output label for -vf"
        );
    }

    #[test]
    fn test_build_multi_segment_filtergraph_with_split_and_vstack() {
        let plan = SmartFramingPlan {
            speaker_intel: None,
            mode: "dual_stack".to_string(),
            x: "656".to_string(),
            y: "0".to_string(),
            w: "608".to_string(),
            h: "1080".to_string(),
            face_bounds: None,
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 2.5,
                    crop: CropRectExpr {
                        x: "100".to_string(),
                        y: "0".to_string(),
                        w: "608".to_string(),
                        h: "1080".to_string(),
                    },
                    face_bounds: None,
                },
                LayoutSegment::DualStack {
                    start: 2.5,
                    end: 7.5,
                    top_track_id: Some(1),
                    bottom_track_id: Some(2),
                    top_crop: CropRectExpr {
                        x: "200".to_string(),
                        y: "50".to_string(),
                        w: "1080".to_string(),
                        h: "960".to_string(),
                    },
                    bottom_crop: CropRectExpr {
                        x: "400".to_string(),
                        y: "50".to_string(),
                        w: "1080".to_string(),
                        h: "960".to_string(),
                    },
                    top_face_bounds: None,
                    bottom_face_bounds: None,
                },
            ],
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let sub_path = Path::new("test.ass");
        let (filter, label) = build_layout_filtergraph(&plan, 7.5, Some(sub_path), None).unwrap();

        // K = 1 (single) + 2 (dual_stack) = 3 branches
        assert!(
            filter.contains("[0:v]split=3[b0][b1][b2];"),
            "Filter should split source into 3 branches: {}",
            filter
        );

        // Single segment 0
        assert!(
            filter.contains("[b0]trim=start=0:end=2.5,setpts=PTS-STARTPTS,crop=w=608:h=1080:x='100':y='0',scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int"),
            "Single segment should trim, setpts, crop and scale: {}", filter
        );
        assert!(
            filter.contains("[v_seg_0];"),
            "Single segment output label should be [v_seg_0];"
        );

        // DualStack segment 1
        assert!(
            filter.contains("[b1]trim=start=2.5:end=7.5,setpts=PTS-STARTPTS,crop=w=1080:h=960:x='200':y='50',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int"),
            "Dual top branch should trim, setpts, crop and scale to 1080:960: {}", filter
        );
        assert!(
            filter.contains("[v_top_1];"),
            "Dual top output label should be [v_top_1];"
        );

        assert!(
            filter.contains("[b2]trim=start=2.5:end=7.5,setpts=PTS-STARTPTS,crop=w=1080:h=960:x='400':y='50',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int"),
            "Dual bottom branch should trim, setpts, crop and scale to 1080:960: {}", filter
        );
        assert!(
            filter.contains("[v_bot_1];"),
            "Dual bottom output label should be [v_bot_1];"
        );

        // Vstack
        assert!(
            filter.contains("[v_top_1][v_bot_1]vstack=inputs=2[v_seg_1];"),
            "DualStack segment should vertically stack top and bottom: {}",
            filter
        );

        // Concat
        assert!(
            filter.contains("[v_seg_0][v_seg_1]concat=n=2:v=1:a=0[v_composed];"),
            "Filter should concatenate all segments: {}",
            filter
        );

        // Subtitles
        assert!(
            filter.contains("[v_composed]subtitles='test.ass'"),
            "Filter should burn subtitles onto composed stream: {}",
            filter
        );
        assert!(
            filter.ends_with("[v_final]"),
            "Filter should terminate with [v_final]: {}",
            filter
        );

        assert_eq!(label, "v_final");
    }

    #[test]
    fn test_resolve_and_verify_fonts_param_behavior() {
        // When no subtitle path is provided, returns Ok(None)
        let res_none = resolve_and_verify_fonts_param(None).unwrap();
        assert!(res_none.is_none());

        // Create a temporary ASS file specifying Bebas Neue
        let temp_dir = std::env::temp_dir();
        let ass_path = temp_dir.join("test_bebas.ass");
        let ass_content = "[Script Info]\nTitle: Test\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\nStyle: Default,Bebas Neue,52\n\n[Events]\n";
        std::fs::write(&ass_path, ass_content).unwrap();

        let res_bebas = resolve_and_verify_fonts_param(Some(&ass_path));
        // Must find bundled font directory and BebasNeue-Regular.ttf
        assert!(
            res_bebas.is_ok(),
            "Bundled font verification should succeed: {:?}",
            res_bebas.err()
        );
        let fonts_param = res_bebas.unwrap().expect("Must have :fontsdir param");
        assert!(fonts_param.contains("fontsdir="));

        let _ = std::fs::remove_file(ass_path);
    }

    #[test]
    fn test_segment_at_valid_single_open_ended() {
        let plan = SmartFramingPlan {
            speaker_intel: None,
            mode: "single".to_string(),
            x: "100".to_string(),
            y: "0".to_string(),
            w: "608".to_string(),
            h: "1080".to_string(),
            face_bounds: None,
            segments: vec![LayoutSegment::Single {
                start: 0.0,
                end: 0.0, // Open-ended single segment
                crop: CropRectExpr {
                    x: "100".to_string(),
                    y: "0".to_string(),
                    w: "608".to_string(),
                    h: "1080".to_string(),
                },
                face_bounds: None,
            }],
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        // Querying arbitrary timestamps inside the clip matches the single open-ended segment
        assert!(plan.segment_at(0.0).is_some());
        assert!(plan.segment_at(15.5).is_some());
        assert!(plan.segment_at(100.0).is_some());
    }

    #[test]
    fn test_segment_at_valid_multi_segment() {
        let crop = CropRectExpr {
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
        };
        let plan = SmartFramingPlan {
            speaker_intel: None,
            mode: "single".to_string(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            face_bounds: None,
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 5.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
                LayoutSegment::Single {
                    start: 5.0,
                    end: 10.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
            ],
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let s0 = plan.segment_at(2.5).unwrap();
        match s0 {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(*start, 0.0);
                assert_eq!(*end, 5.0);
            }
            _ => panic!("Expected single"),
        }

        let s1 = plan.segment_at(7.5).unwrap();
        match s1 {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(*start, 5.0);
                assert_eq!(*end, 10.0);
            }
            _ => panic!("Expected single"),
        }
    }

    #[test]
    fn test_segment_at_zero_length_segment() {
        let crop = CropRectExpr {
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
        };
        let plan = SmartFramingPlan {
            speaker_intel: None,
            mode: "single".to_string(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            face_bounds: None,
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 5.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
                LayoutSegment::Single {
                    start: 5.0,
                    end: 5.0,
                    crop: crop.clone(),
                    face_bounds: None,
                }, // Zero-length
                LayoutSegment::Single {
                    start: 5.0,
                    end: 10.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
            ],
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        // Querying at 7.0 must match segment 2, NOT the zero-length segment
        let s = plan.segment_at(7.0).unwrap();
        match s {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(*start, 5.0);
                assert_eq!(*end, 10.0);
            }
            _ => panic!("Expected segment 2"),
        }
    }

    #[test]
    fn test_segment_at_inverted_segment() {
        let crop = CropRectExpr {
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
        };
        let plan = SmartFramingPlan {
            speaker_intel: None,
            mode: "single".to_string(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            face_bounds: None,
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 5.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
                LayoutSegment::Single {
                    start: 6.0,
                    end: 4.0,
                    crop: crop.clone(),
                    face_bounds: None,
                }, // Inverted
                LayoutSegment::Single {
                    start: 5.0,
                    end: 10.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
            ],
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let s = plan.segment_at(7.0).unwrap();
        match s {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(*start, 5.0);
                assert_eq!(*end, 10.0);
            }
            _ => panic!("Expected segment 2"),
        }
    }

    #[test]
    fn test_segment_at_malformed_intermediate_segment_cannot_swallow_later() {
        let crop = CropRectExpr {
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
        };
        let plan = SmartFramingPlan {
            speaker_intel: None,
            mode: "single".to_string(),
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
            face_bounds: None,
            segments: vec![
                LayoutSegment::Single {
                    start: 0.0,
                    end: 5.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
                LayoutSegment::Single {
                    start: 5.0,
                    end: 0.0,
                    crop: crop.clone(),
                    face_bounds: None,
                }, // Malformed intermediate open-ended
                LayoutSegment::Single {
                    start: 5.0,
                    end: 10.0,
                    crop: crop.clone(),
                    face_bounds: None,
                },
            ],
            framing: "original".to_string(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        // In the legacy code, segment 1 (end <= 0) expanded to f64::MAX and captured t=7.0.
        // In the new code, malformed intermediate segment is skipped, so t=7.0 correctly matches segment 2!
        let s = plan.segment_at(7.0).unwrap();
        match s {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(*start, 5.0);
                assert_eq!(*end, 10.0);
            }
            _ => panic!("Expected segment 2"),
        }
    }

    #[test]
    fn test_validate_and_repair_timeline_bridges_gaps_and_drops_malformed() {
        let crop = CropRectExpr {
            x: "0".into(),
            y: "0".into(),
            w: "608".into(),
            h: "1080".into(),
        };
        let raw_segments = vec![
            LayoutSegment::Single {
                start: 0.0,
                end: 4.0,
                crop: crop.clone(),
                face_bounds: None,
            },
            LayoutSegment::Single {
                start: 4.0,
                end: 4.0,
                crop: crop.clone(),
                face_bounds: None,
            }, // Zero-length
            LayoutSegment::Single {
                start: 5.0,
                end: 0.0,
                crop: crop.clone(),
                face_bounds: None,
            }, // Malformed
            LayoutSegment::Single {
                start: 6.0,
                end: 10.0,
                crop: crop.clone(),
                face_bounds: None,
            }, // Has gap between 4.0 and 6.0
        ];

        let repaired = validate_and_repair_timeline(&raw_segments, 10.0);
        assert_eq!(
            repaired.len(),
            2,
            "Zero-length and malformed intermediate segments must be removed"
        );

        match &repaired[0] {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(*start, 0.0);
                assert_eq!(*end, 4.0);
            }
            _ => panic!("Expected single"),
        }

        // Second segment must bridge the gap from 4.0 to 10.0
        match &repaired[1] {
            LayoutSegment::Single { start, end, .. } => {
                assert_eq!(
                    *start, 4.0,
                    "Timeline validation must bridge the gap so start equals previous end"
                );
                assert_eq!(*end, 10.0);
            }
            _ => panic!("Expected single"),
        }
    }

    #[test]
    fn test_build_layout_filtergraph_escapes_single_quotes_in_ass_path() {
        let plan = SmartFramingPlan::single_fallback(
            "100".to_string(),
            "0".to_string(),
            "608".to_string(),
            "1080".to_string(),
            5.0,
        );

        let test_path = Path::new(r"C:\Users\John's PC\video.ass");
        let (filter, _) = build_layout_filtergraph(&plan, 5.0, Some(test_path), None).unwrap();

        // Must normalize slashes, escape drive colon, and escape single quote as '\''
        assert!(
            filter.contains(r"subtitles='C\:/Users/John'\\\''s PC/video.ass'"),
            "FFmpeg filtergraph must correctly escape apostrophe in subtitle path: {}",
            filter
        );
    }

    #[test]
    fn test_temp_transcript_file_uuid_uniqueness() {
        // Generating multiple unique UUID paths must produce zero collisions
        let mut ids = std::collections::HashSet::new();
        for _ in 0..100 {
            let path = format!("autoshorts_tracker_{}.json", uuid::Uuid::new_v4());
            assert!(
                ids.insert(path),
                "UUID-based temp filename must be globally unique"
            );
        }
    }

    #[test]
    fn test_detect_speaker_crop_params_missing_tracker_marks_emergency_fallback() {
        let _guard = TRACKER_TEST_MUTEX.lock().unwrap();
        std::env::set_var("AUTOSHORTS_SPEAKER_TRACKER_SCRIPT", "disabled");

        let plan = detect_speaker_crop_params(
            "dummy_video.mp4",
            0.0,
            5.0,
            1920,
            1080,
            608,
            None,
            "original",
        );

        std::env::remove_var("AUTOSHORTS_SPEAKER_TRACKER_SCRIPT");

        assert!(
            plan.is_emergency_fallback,
            "Missing tracker must mark is_emergency_fallback as true"
        );
        assert_eq!(
            plan.fallback_reason.as_deref(),
            Some("tracker_missing_or_failed"),
            "Fallback reason must be tracker_missing_or_failed"
        );

        let portrait_plan = detect_speaker_crop_params(
            "dummy_video.mp4",
            0.0,
            5.0,
            608,
            1080,
            608,
            None,
            "original",
        );
        assert!(
            !portrait_plan.is_emergency_fallback,
            "Portrait source must not be emergency fallback"
        );
        assert_eq!(portrait_plan.fallback_reason, None);
    }
}
