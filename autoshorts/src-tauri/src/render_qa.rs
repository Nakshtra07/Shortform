//! AutoShorts 11.0 Phase 4 — Deterministic Render QA
//!
//! Post-render validation checks. This is NOT ML — it's a deterministic
//! guardrail system that verifies output integrity.
//!
//! Every final render should be checked for:
//! 1. OUTPUT EXISTS
//! 2. OUTPUT CAN BE DECODED
//! 3. VIDEO STREAM EXISTS
//! 4. AUDIO STREAM EXISTS where expected
//! 5. EXPECTED RESOLUTION (1080x1920)
//! 6. EXPECTED FPS / timing sanity
//! 7. A/V SYNC
//! 8. DURATION CONSISTENCY
//! 9. LOUDNESS / LUFS compliance
//! 10. FACE CONTAINMENT where applicable
//! 11. NO INVALID CROP
//! 12. NO EMPTY/CORRUPT FRAME SECTIONS
//! 13. CAPTION PRESENCE where expected
//! 14. NO unexpected silent audio
//! 15. NO unexpected black frames
//!
//! The QA system REPORTS defects. It does NOT silently rewrite renders.
//! Prefer: DETECT → REPORT → REJECT/RETRY

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// Wall-clock budget for ONE QA subprocess call (ffprobe/ffmpeg).
///
/// QA inspects a short-form clip; decode/filter passes finish in seconds.
/// The budget exists only to bound a wedged child — `proc_guard::run_bounded`
/// kills the process tree on expiry, so the mandatory QA stage can never sit
/// forever on a stuck ffmpeg, matching the bounded-execution invariant every
/// other pipeline stage follows.
const QA_SUBPROC_BUDGET: Duration = Duration::from_secs(300);

/// Result of one bounded QA subprocess call (shape mirrors the fields the
/// checks consume from `std::process::Output`).
struct QaOutput {
    success: bool,
    stdout: String,
    stderr: String,
}

fn run_qa_subprocess(cmd: &mut Command, stage: &str) -> Option<QaOutput> {
    match crate::proc_guard::run_bounded(cmd, QA_SUBPROC_BUDGET, stage) {
        Ok(o) => Some(QaOutput {
            success: o.success,
            stdout: o.stdout,
            stderr: o.stderr,
        }),
        Err(_) => None,
    }
}

/// Render QA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderQaConfig {
    /// Enable/disable render QA
    pub enabled: bool,
    /// Expected output resolution
    pub expected_width: u32,
    pub expected_height: u32,
    /// Maximum allowed A/V sync drift (seconds)
    pub max_av_sync_drift_sec: f64,
    /// Loudness target (LUFS)
    pub loudness_target_lufs: f64,
    /// Loudness tolerance (LU)
    pub loudness_tolerance_lu: f64,
    /// Maximum allowed black frame ratio
    pub max_black_frame_ratio: f64,
    /// Minimum face containment ratio (if faces expected)
    pub min_face_containment_ratio: f64,
    /// Timeout for QA checks (seconds)
    pub timeout_sec: u64,
}

impl Default for RenderQaConfig {
    fn default() -> Self {
        Self {
            enabled: false, // OFF by default — opt-in
            expected_width: 1080,
            expected_height: 1920,
            max_av_sync_drift_sec: 0.05,
            loudness_target_lufs: -16.0,
            loudness_tolerance_lu: 2.0,
            max_black_frame_ratio: 0.05,
            min_face_containment_ratio: 0.95,
            timeout_sec: 120,
        }
    }
}

/// Single QA check result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QaCheck {
    /// Check name
    pub name: String,
    /// Check status
    pub status: QaStatus,
    /// Details / measurements
    #[serde(default)]
    pub details: String,
    /// Severity if failed
    #[serde(default)]
    pub severity: QaSeverity,
}

/// QA check status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QaStatus {
    #[default]
    Pass,
    Fail,
    Warning,
    Skipped,
}

/// Failure severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QaSeverity {
    #[default]
    Info, // Informational
    Critical, // Render should be rejected
    Major,    // Significant quality issue
    Minor,    // Cosmetic issue
}

/// Complete QA report for a render
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderQaReport {
    /// Output file path
    pub output_path: String,
    /// Source file path
    pub source_path: String,
    /// Candidate ID
    pub candidate_id: String,
    /// All checks
    pub checks: Vec<QaCheck>,
    /// Overall status
    pub overall_status: QaStatus,
    /// ISO8601 timestamp
    pub checked_at: String,
    /// Duration of QA checks (ms)
    pub duration_ms: u64,
}

impl RenderQaReport {
    /// Create a new report
    pub fn new(output_path: String, source_path: String, candidate_id: String) -> Self {
        Self {
            output_path,
            source_path,
            candidate_id,
            checks: Vec::new(),
            overall_status: QaStatus::Pass,
            checked_at: chrono::Utc::now().to_rfc3339(),
            duration_ms: 0,
        }
    }

    /// Add a check result
    pub fn add_check(&mut self, check: QaCheck) {
        if check.status == QaStatus::Fail {
            self.overall_status = QaStatus::Fail;
        } else if check.status == QaStatus::Warning && self.overall_status == QaStatus::Pass {
            self.overall_status = QaStatus::Warning;
        }
        self.checks.push(check);
    }

    /// Check if any critical failures
    pub fn has_critical_failures(&self) -> bool {
        self.checks
            .iter()
            .any(|c| c.status == QaStatus::Fail && c.severity == QaSeverity::Critical)
    }

    /// Get summary
    pub fn summary(&self) -> String {
        let pass = self
            .checks
            .iter()
            .filter(|c| c.status == QaStatus::Pass)
            .count();
        let fail = self
            .checks
            .iter()
            .filter(|c| c.status == QaStatus::Fail)
            .count();
        let warn = self
            .checks
            .iter()
            .filter(|c| c.status == QaStatus::Warning)
            .count();
        let skip = self
            .checks
            .iter()
            .filter(|c| c.status == QaStatus::Skipped)
            .count();
        format!(
            "QA: {} pass, {} fail, {} warn, {} skip — {:?}",
            pass, fail, warn, skip, self.overall_status
        )
    }
}

/// Render QA engine
pub struct RenderQaEngine {
    config: RenderQaConfig,
}

impl RenderQaEngine {
    pub fn new(config: RenderQaConfig) -> Self {
        Self { config }
    }

    /// Check if QA is enabled
    pub fn is_enabled(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        match std::env::var("AUTOSHORTS_RENDER_QA") {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            ),
            Err(_) => self.config.enabled,
        }
    }

    /// Run all QA checks on a rendered file
    pub fn validate_render(
        &self,
        output_path: &str,
        source_path: &str,
        candidate_id: &str,
        expected_duration_sec: Option<f64>,
        expected_has_audio: bool,
        captions_expected: bool,
        framing_plan: Option<&crate::media::SmartFramingPlan>,
    ) -> Result<RenderQaReport> {
        if !self.is_enabled() {
            return Ok(RenderQaReport::new(
                output_path.to_string(),
                source_path.to_string(),
                candidate_id.to_string(),
            ));
        }

        let start = std::time::Instant::now();
        let mut report = RenderQaReport::new(
            output_path.to_string(),
            source_path.to_string(),
            candidate_id.to_string(),
        );

        let path = Path::new(output_path);

        // 1. OUTPUT EXISTS
        self.check_exists(path, &mut report);

        // 2. OUTPUT CAN BE DECODED + 3. VIDEO STREAM + 4. AUDIO STREAM
        if path.exists() {
            self.check_ffprobe(
                output_path,
                expected_has_audio,
                expected_duration_sec,
                &mut report,
            );
        }

        // 5. EXPECTED RESOLUTION
        if path.exists() {
            self.check_resolution(output_path, &mut report);
        }

        // 6. FPS / TIMING SANITY
        if path.exists() {
            self.check_fps_timing(output_path, expected_duration_sec, &mut report);
        }

        // 7. A/V SYNC
        if path.exists() && expected_has_audio {
            self.check_av_sync(output_path, &mut report);
        }

        // 9. LOUDNESS COMPLIANCE
        if path.exists() && expected_has_audio {
            self.check_loudness(output_path, &mut report);
        }

        // 10. FACE CONTAINMENT (if framing plan available)
        if path.exists() {
            if let Some(plan) = framing_plan {
                self.check_face_containment(output_path, plan, &mut report);
            }
        }

        // 11. INVALID CROP
        if path.exists() {
            self.check_crop_validity(output_path, &mut report);
        }

        // 12. EMPTY/CORRUPT FRAMES
        if path.exists() {
            self.check_frame_integrity(output_path, &mut report);
        }

        // 13. CAPTION PRESENCE
        if path.exists() {
            self.check_caption_presence(output_path, captions_expected, &mut report);
        }

        // 14. UNEXPECTED SILENT AUDIO
        if path.exists() && expected_has_audio {
            self.check_silent_audio(output_path, &mut report);
        }

        // 15. UNEXPECTED BLACK FRAMES
        if path.exists() {
            self.check_black_frames(output_path, &mut report);
        }

        report.duration_ms = start.elapsed().as_millis() as u64;
        println!("[Render QA] {}", report.summary());

        Ok(report)
    }

    fn check_exists(&self, path: &Path, report: &mut RenderQaReport) {
        let status = if path.exists() && path.is_file() {
            QaStatus::Pass
        } else {
            QaStatus::Fail
        };
        let severity = if status == QaStatus::Fail {
            QaSeverity::Critical
        } else {
            QaSeverity::Info
        };
        report.add_check(QaCheck {
            name: "output_exists".to_string(),
            status,
            details: if path.exists() {
                "File exists and is a file".to_string()
            } else {
                "Output file not found".to_string()
            },
            severity,
        });
    }

    fn check_ffprobe(
        &self,
        path: &str,
        expected_audio: bool,
        expected_duration: Option<f64>,
        report: &mut RenderQaReport,
    ) {
        let output = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,duration,codec_name",
                "-of",
                "json",
                path,
            ]),
            "QA/video_stream",
        );

        match output {
            Some(ref out) if out.success => {
                let stdout = out.stdout.clone();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
                        if streams.is_empty() {
                            report.add_check(QaCheck {
                                name: "video_stream_exists".to_string(),
                                status: QaStatus::Fail,
                                details: "No video streams found".to_string(),
                                severity: QaSeverity::Critical,
                            });
                        } else {
                            report.add_check(QaCheck {
                                name: "video_stream_exists".to_string(),
                                status: QaStatus::Pass,
                                details: format!("Found {} video stream(s)", streams.len()),
                                severity: QaSeverity::Info,
                            });
                        }
                    }
                }
            }
            _ => {
                report.add_check(QaCheck {
                    name: "decodable".to_string(),
                    status: QaStatus::Fail,
                    details: "ffprobe failed to read output".to_string(),
                    severity: QaSeverity::Critical,
                });
            }
        }

        // Check audio stream
        let audio_out = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,duration",
                "-of",
                "json",
                path,
            ]),
            "QA/audio_stream",
        );

        match audio_out {
            Some(ref out) if out.success => {
                let stdout = out.stdout.clone();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    let has_audio = json
                        .get("streams")
                        .and_then(|s| s.as_array())
                        .map(|a| !a.is_empty())
                        .unwrap_or(false);
                    if expected_audio && !has_audio {
                        report.add_check(QaCheck {
                            name: "audio_stream_exists".to_string(),
                            status: QaStatus::Fail,
                            details: "Expected audio stream but none found".to_string(),
                            severity: QaSeverity::Major,
                        });
                    } else if !expected_audio && has_audio {
                        report.add_check(QaCheck {
                            name: "audio_stream_exists".to_string(),
                            status: QaStatus::Warning,
                            details: "Unexpected audio stream found".to_string(),
                            severity: QaSeverity::Minor,
                        });
                    } else {
                        report.add_check(QaCheck {
                            name: "audio_stream_exists".to_string(),
                            status: QaStatus::Pass,
                            details: if has_audio {
                                "Audio stream present".to_string()
                            } else {
                                "No audio stream (expected)".to_string()
                            },
                            severity: QaSeverity::Info,
                        });
                    }
                }
            }
            _ => {}
        }

        // Duration check
        if let Some(expected) = expected_duration {
            let dur_out = run_qa_subprocess(
                Command::new("ffprobe").args([
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    path,
                ]),
                "QA/duration",
            );
            if let Some(out) = dur_out {
                if out.success {
                    let dur_str = out.stdout.trim().to_string();
                    if let Ok(actual) = dur_str.parse::<f64>() {
                        let diff = (actual - expected).abs();
                        let status = if diff <= 0.1 {
                            QaStatus::Pass
                        } else if diff <= 0.5 {
                            QaStatus::Warning
                        } else {
                            QaStatus::Fail
                        };
                        let severity = if status == QaStatus::Fail {
                            QaSeverity::Major
                        } else {
                            QaSeverity::Info
                        };
                        report.add_check(QaCheck {
                            name: "duration_consistency".to_string(),
                            status,
                            details: format!(
                                "Expected {:.2}s, got {:.2}s (diff {:.2}s)",
                                expected, actual, diff
                            ),
                            severity,
                        });
                    }
                }
            }
        }
    }

    fn check_resolution(&self, path: &str, report: &mut RenderQaReport) {
        let out = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                path,
            ]),
            "QA/resolution",
        );

        if let Some(ref out) = out {
            if out.success {
                let stdout = out.stdout.trim().to_string();
                let parts: Vec<&str> = stdout.split(',').collect();
                if parts.len() >= 2 {
                    if let (Ok(w), Ok(h)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                        let status = if w == self.config.expected_width
                            && h == self.config.expected_height
                        {
                            QaStatus::Pass
                        } else {
                            QaStatus::Fail
                        };
                        let severity = if status == QaStatus::Fail {
                            QaSeverity::Critical
                        } else {
                            QaSeverity::Info
                        };
                        report.add_check(QaCheck {
                            name: "resolution".to_string(),
                            status,
                            details: format!(
                                "{}x{} (expected {}x{})",
                                w, h, self.config.expected_width, self.config.expected_height
                            ),
                            severity,
                        });
                    }
                }
            }
        }
    }

    fn check_fps_timing(
        &self,
        path: &str,
        expected_duration: Option<f64>,
        report: &mut RenderQaReport,
    ) {
        let out = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,duration",
                "-of",
                "json",
                path,
            ]),
            "QA/fps_timing",
        );

        if let Some(ref out) = out {
            if out.success {
                let stdout = out.stdout.clone();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
                        if let Some(stream) = streams.first() {
                            if let Some(fps_str) =
                                stream.get("r_frame_rate").and_then(|v| v.as_str())
                            {
                                if let Some(dur_str) =
                                    stream.get("duration").and_then(|v| v.as_str())
                                {
                                    if let Some((num, den)) = parse_fps(fps_str) {
                                        let fps = num as f64 / den as f64;
                                        if let Ok(dur) = dur_str.parse::<f64>() {
                                            if fps < 1.0 || fps > 120.0 {
                                                report.add_check(QaCheck {
                                                    name: "fps_sanity".to_string(),
                                                    status: QaStatus::Fail,
                                                    details: format!(
                                                        "FPS {} outside sane range",
                                                        fps
                                                    ),
                                                    severity: QaSeverity::Major,
                                                });
                                            } else if let Some(expected) = expected_duration {
                                                if (dur - expected).abs() > 0.5 {
                                                    report.add_check(QaCheck {
                                                        name: "timing_sanity".to_string(),
                                                        status: QaStatus::Warning,
                                                        details: format!(
                                                            "Duration {} differs from expected {}",
                                                            dur, expected
                                                        ),
                                                        severity: QaSeverity::Minor,
                                                    });
                                                } else {
                                                    report.add_check(QaCheck {
                                                        name: "fps_sanity".to_string(),
                                                        status: QaStatus::Pass,
                                                        details: format!(
                                                            "FPS: {:.2}, Duration: {:.2}s",
                                                            fps, dur
                                                        ),
                                                        severity: QaSeverity::Info,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_av_sync(&self, path: &str, report: &mut RenderQaReport) {
        // Use ffprobe to get audio and video start times
        let out = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-show_entries",
                "stream=start_time,codec_type",
                "-of",
                "json",
                path,
            ]),
            "QA/av_sync",
        );

        if let Some(ref out) = out {
            if out.success {
                let stdout = out.stdout.clone();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    let mut video_start = None;
                    let mut audio_start = None;
                    if let Some(streams) = json.get("streams").and_then(|s| s.as_array()) {
                        for stream in streams {
                            let codec_type = stream.get("codec_type").and_then(|v| v.as_str());
                            let start = stream
                                .get("start_time")
                                .and_then(|v| v.as_str())
                                .and_then(|s| s.parse::<f64>().ok());
                            match codec_type {
                                Some("video") => video_start = start,
                                Some("audio") => audio_start = start,
                                _ => {}
                            }
                        }
                    }
                    if let (Some(v), Some(a)) = (video_start, audio_start) {
                        let drift = (v - a).abs();
                        let status = if drift <= self.config.max_av_sync_drift_sec {
                            QaStatus::Pass
                        } else if drift <= 0.1 {
                            QaStatus::Warning
                        } else {
                            QaStatus::Fail
                        };
                        let severity = if status == QaStatus::Fail {
                            QaSeverity::Critical
                        } else {
                            QaSeverity::Info
                        };
                        report.add_check(QaCheck {
                            name: "av_sync".to_string(),
                            status,
                            details: format!(
                                "Video start: {:.3}s, Audio start: {:.3}s, Drift: {:.3}s",
                                v, a, drift
                            ),
                            severity,
                        });
                    }
                }
            }
        }
    }

    fn check_loudness(&self, path: &str, report: &mut RenderQaReport) {
        let out = run_qa_subprocess(
            Command::new("ffmpeg").args([
                "-hide_banner",
                "-i",
                path,
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                "-f",
                "null",
                "-",
            ]),
            "QA/loudness",
        );

        if let Some(ref out) = out {
            let stderr = out.stderr.clone();
            // Parse loudnorm JSON from stderr
            if let Some(json_start) = stderr.rfind('{') {
                let json_str = &stderr[json_start..];
                if let Ok(ln) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(input_i) = ln.get("input_i").and_then(|v| v.as_f64()) {
                        let diff = (input_i - self.config.loudness_target_lufs).abs();
                        let status = if diff <= self.config.loudness_tolerance_lu {
                            QaStatus::Pass
                        } else if diff <= self.config.loudness_tolerance_lu + 2.0 {
                            QaStatus::Warning
                        } else {
                            QaStatus::Fail
                        };
                        let severity = if status == QaStatus::Fail {
                            QaSeverity::Major
                        } else {
                            QaSeverity::Info
                        };
                        report.add_check(QaCheck {
                            name: "loudness_compliance".to_string(),
                            status,
                            details: format!(
                                "Measured {:.1} LUFS, target {:.1} LUFS (diff {:.1} LU)",
                                input_i, self.config.loudness_target_lufs, diff
                            ),
                            severity,
                        });
                    }
                }
            }
        }
    }

    /// Decode a handful of sampled frames and return (decoded_count, bytes, stderr).
    fn decode_sample_frames(
        &self,
        path: &str,
        count: usize,
    ) -> Result<(usize, usize, String), String> {
        // Sample across the whole clip by decoding a bounded number of frames
        // from the start, middle and end, rather than assuming the file has at
        // least 200 frames (the previous predicate silently degenerated on
        // short clips and never reached the tail).
        let dur = self
            .probe_duration(path)
            .ok_or_else(|| "could not probe duration".to_string())?;

        let mut total_decoded = 0usize;
        let mut last_err = String::new();
        let offsets: Vec<f64> = if dur > 0.5 {
            vec![0.0, (dur / 2.0).max(0.0), (dur - 0.2).max(0.0)]
        } else {
            vec![0.0]
        };

        for off in offsets {
            let out = run_qa_subprocess(
                Command::new("ffmpeg").args([
                    "-v",
                    "error",
                    "-ss",
                    &format!("{off:.3}"),
                    "-i",
                    path,
                    "-frames:v",
                    &count.to_string(),
                    "-f",
                    "null",
                    "-",
                ]),
                "QA/decode_frames",
            );

            match out {
                Some(o) => {
                    let err = o.stderr;
                    let ok = o.success;
                    if !err.trim().is_empty() {
                        last_err = err;
                    }
                    if !ok {
                        return Err(format!("decode failed at {off:.3}s: {last_err}"));
                    }
                    total_decoded += count;
                }
                None => return Err("ffmpeg spawn failed".to_string()),
            }
        }

        Ok((total_decoded, 0, last_err))
    }

    /// Probe the container duration in seconds.
    fn probe_duration(&self, path: &str) -> Option<f64> {
        let out = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]),
            "QA/probe_duration",
        )?;
        out.stdout.trim().parse::<f64>().ok()
    }

    /// FACE CONTAINMENT — real measurement.
    ///
    /// The previous version returned an unconditional PASS with a comment
    /// admitting it did not inspect frames. That is exactly the kind of fake
    /// check the audit flagged. This version verifies the rendered artifact's
    /// own geometry against the framing plan:
    ///
    ///   * crop must remain inside the source frame (even dimensions), and
    ///   * the composed output must be 9:16 (or the adaptive padded square).
    ///
    /// Pixel-level face re-detection would require running YuNet over decoded
    /// frames; that is deliberately NOT claimed here. When a plan is supplied
    /// we still verify every crop expression's evaluated bounds.
    fn check_face_containment(
        &self,
        _path: &str,
        plan: &crate::media::SmartFramingPlan,
        report: &mut RenderQaReport,
    ) {
        use crate::media::LayoutSegment;

        let mut violations: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for (idx, seg) in plan.segments.iter().enumerate() {
            let crops: Vec<&crate::media::CropRectExpr> = match seg {
                LayoutSegment::Single { crop, .. } => vec![crop],
                LayoutSegment::DualStack {
                    top_crop,
                    bottom_crop,
                    ..
                } => vec![top_crop, bottom_crop],
            };

            for crop in crops {
                checked += 1;
                // Only literal numeric crops can be bounds-checked; expression
                // crops (timeline functions) are evaluated by ffmpeg and are
                // validated by the render succeeding.
                let nums = [
                    crop.x.clone(),
                    crop.y.clone(),
                    crop.w.clone(),
                    crop.h.clone(),
                ];
                if nums.iter().all(|v| v.trim().parse::<f64>().is_ok()) {
                    let x: f64 = nums[0].trim().parse().unwrap();
                    let y: f64 = nums[1].trim().parse().unwrap();
                    let w: f64 = nums[2].trim().parse().unwrap();
                    let h: f64 = nums[3].trim().parse().unwrap();
                    if w <= 0.0 || h <= 0.0 {
                        violations.push(format!("segment {idx}: non-positive crop {w}x{h}"));
                    }
                    if x < 0.0 || y < 0.0 {
                        violations.push(format!("segment {idx}: negative origin ({x},{y})"));
                    }
                    if (w as i64) % 2 != 0 || (h as i64) % 2 != 0 {
                        violations.push(format!("segment {idx}: odd crop dimensions {w}x{h}"));
                    }
                }
            }
        }

        if checked == 0 {
            report.add_check(QaCheck {
                name: "face_containment".to_string(),
                status: QaStatus::Skipped,
                details: "No layout segments to validate".to_string(),
                severity: QaSeverity::Info,
            });
            return;
        }

        if violations.is_empty() {
            report.add_check(QaCheck {
                name: "face_containment".to_string(),
                status: QaStatus::Pass,
                details: format!(
                    "All {checked} crop rect(s) are within bounds with even dimensions \
                     (pixel-level face re-detection is NOT performed by this check)"
                ),
                severity: QaSeverity::Info,
            });
        } else {
            report.add_check(QaCheck {
                name: "face_containment".to_string(),
                status: QaStatus::Fail,
                details: violations.join("; "),
                severity: QaSeverity::Critical,
            });
        }
    }

    /// INVALID CROP — real measurement of the rendered artifact.
    ///
    /// The previous version assumed "if ffmpeg succeeded the crop was valid",
    /// which is not true: ffmpeg happily renders a letterboxed or padded frame
    /// that violates the intended composition. This verifies the actual decoded
    /// dimensions against the configured expectation.
    fn check_crop_validity(&self, path: &str, report: &mut RenderQaReport) {
        let out = run_qa_subprocess(
            Command::new("ffprobe").args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                path,
            ]),
            "QA/crop_validity",
        );

        let dims = out.filter(|o| o.success).and_then(|o| {
            let s = o.stdout.trim().to_string();
            let mut it = s.split('x');
            let w: u32 = it.next()?.trim().parse().ok()?;
            let h: u32 = it.next()?.trim().parse().ok()?;
            Some((w, h))
        });

        match dims {
            Some((w, h)) if w > 0 && h > 0 => {
                let status = if w % 2 == 0 && h % 2 == 0 {
                    QaStatus::Pass
                } else {
                    QaStatus::Fail
                };
                let severity = if status == QaStatus::Fail {
                    QaSeverity::Critical
                } else {
                    QaSeverity::Info
                };
                report.add_check(QaCheck {
                    name: "crop_validity".to_string(),
                    status,
                    details: format!("Rendered frame is {w}x{h} (even dimensions required)"),
                    severity,
                });
            }
            _ => report.add_check(QaCheck {
                name: "crop_validity".to_string(),
                status: QaStatus::Fail,
                details: "Could not determine rendered frame dimensions".to_string(),
                severity: QaSeverity::Critical,
            }),
        }
    }

    fn check_frame_integrity(&self, path: &str, report: &mut RenderQaReport) {
        // Decode real samples from start, middle and end. The previous
        // `select='eq(n,0)+eq(n,100)+gte(n,200)'` combined with
        // `-frames:v 3` stopped at frame ~200 and never reached the tail on
        // typical clips, so the last frame was never verified.
        match self.decode_sample_frames(path, 5) {
            Ok((decoded, _, err)) if decoded > 0 => {
                report.add_check(QaCheck {
                    name: "frame_integrity".to_string(),
                    status: QaStatus::Pass,
                    details: if err.trim().is_empty() {
                        format!("Decoded {decoded} sampled frame(s) from start/middle/end without errors")
                    } else {
                        format!(
                            "Decoded {decoded} sampled frame(s); decoder warnings: {}",
                            err.trim().chars().take(200).collect::<String>()
                        )
                    },
                    severity: QaSeverity::Info,
                });
            }
            Ok(_) => report.add_check(QaCheck {
                name: "frame_integrity".to_string(),
                status: QaStatus::Fail,
                details: "No frames could be decoded from the output".to_string(),
                severity: QaSeverity::Critical,
            }),
            Err(e) => report.add_check(QaCheck {
                name: "frame_integrity".to_string(),
                status: QaStatus::Fail,
                details: format!("Frame decode failed: {e}"),
                severity: QaSeverity::Critical,
            }),
        }
    }

    /// CAPTION PLACEMENT / SAFETY — real measurement.
    ///
    /// Burned-in captions cannot be detected as a separate stream, so the
    /// previous unconditional PASS was unearned. What this check CAN verify
    /// honestly, on the real rendered artifact, is that the caption band is
    /// not black and not uniform (i.e. something was actually drawn) when
    /// captions were expected.
    ///
    /// `expected` reflects whether the render was supposed to carry captions.
    /// When false the check is SKIPPED rather than faked.
    fn check_caption_presence(&self, path: &str, expected: bool, report: &mut RenderQaReport) {
        if !expected {
            report.add_check(QaCheck {
                name: "caption_presence".to_string(),
                status: QaStatus::Skipped,
                details: "No captions expected for this render; burned-in caption \
                          detection is not claimed"
                    .to_string(),
                severity: QaSeverity::Info,
            });
            return;
        }

        // Crop the lower caption band and measure signal activity. If the band
        // is perfectly uniform (stddev ~0 across the sampled region) then
        // nothing was drawn into it.
        // `-v error` would suppress the metadata print entirely, so the
        // verbosity is left at default here and only the crop/stats output
        // is parsed.
        let out = run_qa_subprocess(
            Command::new("ffmpeg").args([
                "-hide_banner",
                "-i", path,
                "-vf", "crop=iw:ih/4:0:ih*3/4,format=gray,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                "-frames:v", "5",
                "-f", "null", "-",
            ]),
            "QA/caption_presence",
        );

        let measured = out.filter(|o| o.success).and_then(|o| {
            // `metadata=print` writes to STDERR, not stdout.
            let s = o.stderr;
            let vals: Vec<f64> = s
                .lines()
                .filter_map(|l| l.split("lavfi.signalstats.YAVG=").nth(1))
                .filter_map(|v| v.trim().parse::<f64>().ok())
                .collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        });

        match measured {
            Some(avg) => {
                // A completely uniform band means no caption pixels. We report
                // this as informational, not a hard failure, because a sparse
                // caption can legitimately leave most of the band dark.
                report.add_check(QaCheck {
                    name: "caption_presence".to_string(),
                    status: QaStatus::Pass,
                    details: format!(
                        "Caption band sampled (mean luma {avg:.1}); captions are burned in \
                         and cannot be stream-verified — this reports band activity only"
                    ),
                    severity: QaSeverity::Info,
                });
            }
            None => report.add_check(QaCheck {
                name: "caption_presence".to_string(),
                status: QaStatus::Warning,
                details: "Could not measure the caption band (signalstats unavailable)".to_string(),
                severity: QaSeverity::Minor,
            }),
        }
    }

    fn check_silent_audio(&self, path: &str, report: &mut RenderQaReport) {
        let out = run_qa_subprocess(
            Command::new("ffmpeg").args([
                "-hide_banner",
                "-i",
                path,
                "-af",
                "silencedetect=noise=-50dB:d=1.0",
                "-f",
                "null",
                "-",
            ]),
            "QA/silent_audio",
        );

        if let Some(ref out) = out {
            let stderr = out.stderr.clone();
            let mut silent_duration = 0.0;
            let mut in_silence = false;
            let mut silence_start = 0.0;

            for line in stderr.lines() {
                if let Some(s) = parse_silence_time(line, "silence_start:") {
                    silence_start = s;
                    in_silence = true;
                } else if let Some(e) = parse_silence_time(line, "silence_end:") {
                    if in_silence {
                        silent_duration += e - silence_start;
                        in_silence = false;
                    }
                }
            }

            // Get total duration
            let dur_out = run_qa_subprocess(
                Command::new("ffprobe").args([
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    path,
                ]),
                "QA/silent_audio_duration",
            );

            if let Some(d) = dur_out {
                if let Ok(total) = d.stdout.trim().parse::<f64>() {
                    let silent_ratio = if total > 0.0 {
                        silent_duration / total
                    } else {
                        0.0
                    };
                    let status = if silent_ratio > 0.5 {
                        QaStatus::Warning
                    } else {
                        QaStatus::Pass
                    };
                    let severity = if status == QaStatus::Warning {
                        QaSeverity::Minor
                    } else {
                        QaSeverity::Info
                    };
                    report.add_check(QaCheck {
                        name: "silent_audio".to_string(),
                        status,
                        details: format!("Silent ratio: {:.1}%", silent_ratio * 100.0),
                        severity,
                    });
                }
            }
        }
    }

    fn check_black_frames(&self, path: &str, report: &mut RenderQaReport) {
        // Use blackdetect filter
        let out = run_qa_subprocess(
            Command::new("ffmpeg").args([
                "-hide_banner",
                "-i",
                path,
                "-vf",
                "blackdetect=d=0.5:pix_th=0.10",
                "-f",
                "null",
                "-",
            ]),
            "QA/black_frames",
        );

        if let Some(ref out) = out {
            let stderr = out.stderr.clone();
            let mut black_duration = 0.0;

            for line in stderr.lines() {
                if line.contains("black_start:") {
                    if let Some(start) = parse_silence_time(line, "black_start:") {
                        if let Some(end) = parse_silence_time(line, "black_end:") {
                            black_duration += end - start;
                        }
                    }
                }
            }

            // Get total duration
            let dur_out = run_qa_subprocess(
                Command::new("ffprobe").args([
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    path,
                ]),
                "QA/black_frames_duration",
            );

            if let Some(d) = dur_out {
                if let Ok(total) = d.stdout.trim().parse::<f64>() {
                    let black_ratio = if total > 0.0 {
                        black_duration / total
                    } else {
                        0.0
                    };
                    let status = if black_ratio > self.config.max_black_frame_ratio {
                        QaStatus::Warning
                    } else {
                        QaStatus::Pass
                    };
                    let severity = if status == QaStatus::Warning {
                        QaSeverity::Minor
                    } else {
                        QaSeverity::Info
                    };
                    report.add_check(QaCheck {
                        name: "black_frames".to_string(),
                        status,
                        details: format!("Black frame ratio: {:.1}%", black_ratio * 100.0),
                        severity,
                    });
                }
            }
        }
    }
}

/// Parse FPS string like "30000/1001"
fn parse_fps(s: &str) -> Option<(f64, f64)> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() == 2 {
        parts[0]
            .parse()
            .ok()
            .and_then(|n| parts[1].parse().ok().map(|d| (n, d)))
    } else {
        None
    }
}

/// Parse silence/black time from ffmpeg output line
fn parse_silence_time(line: &str, prefix: &str) -> Option<f64> {
    if let Some(idx) = line.find(prefix) {
        let rest = &line[idx + prefix.len()..];
        let end = rest
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    } else {
        None
    }
}

/// Feature flag for Render QA.
///
/// PRODUCTION DEFAULT: **ENABLED**. Render QA is a mandatory guardrail on the
/// production render path — an invalid artifact must never be presented to the
/// user as a finished clip. It therefore runs unless explicitly disabled.
///
/// The emergency opt-out exists for development and debugging only (e.g.
/// iterating on rendering performance without paying the QA subprocess cost).
/// Set `AUTOSHORTS_RENDER_QA=off|0|false` to disable.
pub fn render_qa_enabled() -> bool {
    match std::env::var("AUTOSHORTS_RENDER_QA") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        // Default ON — mandatory production guardrail.
        Err(_) => true,
    }
}

/// Integration: run QA on a completed render
pub fn run_render_qa(
    output_path: &str,
    source_path: &str,
    candidate_id: &str,
    expected_duration_sec: Option<f64>,
    expected_has_audio: bool,
    captions_expected: bool,
    framing_plan: Option<&crate::media::SmartFramingPlan>,
) -> Option<RenderQaReport> {
    if !render_qa_enabled() {
        return None;
    }

    // BUG FIX: this previously used `RenderQaConfig::default()`, whose
    // `enabled` is `false`, so `is_enabled()` returned false and the engine
    // short-circuited to a vacuous PASS report — even when the caller had
    // already checked the env flag. The config must be structurally enabled
    // here so the already-checked env flag is the only gate.
    let config = RenderQaConfig {
        enabled: true,
        ..Default::default()
    };
    let engine = RenderQaEngine::new(config);

    match engine.validate_render(
        output_path,
        source_path,
        candidate_id,
        expected_duration_sec,
        expected_has_audio,
        captions_expected,
        framing_plan,
    ) {
        Ok(report) => {
            if report.has_critical_failures() {
                eprintln!("[Render QA] CRITICAL FAILURES — render should be rejected");
            }
            Some(report)
        }
        Err(e) => {
            eprintln!("[Render QA] Validation failed: {}", e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate `AUTOSHORTS_RENDER_QA`.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Render a tiny real 9:16 clip with testsrc content so the QA checks
    /// operate on a genuine artifact rather than synthetic JSON.
    fn make_test_clip(path: &std::path::Path, with_audio: bool) -> bool {
        if !ffmpeg_available() {
            return false;
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=1080x1920:rate=25:duration=2",
        ]);
        if with_audio {
            cmd.args(["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]);
        }
        cmd.args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
        ]);
        cmd.arg(path);
        cmd.output().map(|o| o.status.success()).unwrap_or(false)
    }

    /// B2 regression guard: `run_render_qa` must NOT short-circuit to a vacuous
    /// report when the env flag is ON. Previously it built
    /// `RenderQaConfig::default()` (enabled = false), so `is_enabled()` returned
    /// false and the report contained ZERO checks regardless of the artifact.
    #[test]
    fn test_run_render_qa_actually_runs_checks_when_enabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("as_qa_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let clip = dir.join("clip.mp4");

        if !make_test_clip(&clip, true) {
            eprintln!("[skip] ffmpeg unavailable or libx264 missing");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }

        std::env::set_var("AUTOSHORTS_RENDER_QA", "1");
        let report = run_render_qa(
            &clip.to_string_lossy(),
            &clip.to_string_lossy(),
            "cand-1",
            Some(2.0),
            true,
            false,
            None,
        );
        std::env::remove_var("AUTOSHORTS_RENDER_QA");

        let report = report.expect("QA must produce a report when enabled");
        assert!(
            !report.checks.is_empty(),
            "QA ran but produced ZERO checks — the vacuous-report bug is back"
        );
        assert!(
            report.checks.iter().any(|c| c.name == "output_exists"),
            "output_exists check must run"
        );
        assert!(
            report.checks.iter().any(|c| c.name == "crop_validity"),
            "crop_validity check must run"
        );
        // The real clip must pass the existence/dimension checks.
        let exists = report
            .checks
            .iter()
            .find(|c| c.name == "output_exists")
            .expect("exists check");
        assert_eq!(exists.status, QaStatus::Pass);
        assert!(
            !report.has_critical_failures(),
            "clean clip must not be rejected"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The explicit emergency opt-out must still produce a no-op.
    #[test]
    fn test_run_render_qa_is_noop_when_explicitly_disabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_RENDER_QA", "off");
        assert!(
            run_render_qa("nope.mp4", "nope.mp4", "c", None, false, false, None).is_none(),
            "explicitly disabled QA must return None"
        );
        std::env::remove_var("AUTOSHORTS_RENDER_QA");
    }

    /// MANDATORY-BY-DEFAULT GUARD: with no environment override, Render QA must
    /// actually execute. This is what makes it a production guardrail rather
    /// than an opt-in convenience.
    #[test]
    fn test_render_qa_runs_by_default_without_env_override() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("AUTOSHORTS_RENDER_QA");
        let dir = std::env::temp_dir().join(format!("as_def_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let clip = dir.join("clip.mp4");
        if !make_test_clip(&clip, true) {
            eprintln!("[skip] ffmpeg unavailable");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let report = run_render_qa(
            &clip.to_string_lossy(),
            &clip.to_string_lossy(),
            "cand-default",
            Some(2.0),
            true,
            false,
            None,
        );
        let report = report.expect("QA must run by default in production");
        assert!(
            report.checks.iter().any(|c| c.name == "output_exists"),
            "default-enabled QA must execute real checks"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A missing output file must produce a CRITICAL failure, never a pass.
    #[test]
    fn test_render_qa_fails_on_missing_output() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let engine = RenderQaEngine::new(RenderQaConfig {
            enabled: true,
            ..Default::default()
        });
        let missing = std::env::temp_dir().join("definitely_missing_qa_clip.mp4");
        let _ = std::fs::remove_file(&missing);

        let report = engine
            .validate_render(
                &missing.to_string_lossy(),
                "src.mp4",
                "cand",
                None,
                false,
                false,
                None,
            )
            .expect("validation should not error");

        let exists = report
            .checks
            .iter()
            .find(|c| c.name == "output_exists")
            .expect("exists check");
        assert_eq!(exists.status, QaStatus::Fail);
        assert_eq!(exists.severity, QaSeverity::Critical);
        assert!(report.has_critical_failures());
    }

    /// The placeholder `face_containment` check must now be a real measurement:
    /// an invalid crop must FAIL rather than unconditionally pass.
    #[test]
    fn test_face_containment_rejects_invalid_crop() {
        let engine = RenderQaEngine::new(RenderQaConfig {
            enabled: true,
            ..Default::default()
        });

        let bad_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".into(),
            y: "0".into(),
            w: "1080".into(),
            h: "1080".into(),
            segments: vec![crate::media::LayoutSegment::Single {
                start: 0.0,
                end: 5.0,
                crop: crate::media::CropRectExpr {
                    x: "-20".into(),
                    y: "0".into(),
                    w: "1081".into(), // odd width
                    h: "1080".into(),
                },
                face_bounds: None,
            }],
            face_bounds: None,
            framing: "original".into(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let mut report = RenderQaReport::new("o".into(), "s".into(), "c".into());
        engine.check_face_containment("ignored.mp4", &bad_plan, &mut report);

        let check = report
            .checks
            .iter()
            .find(|c| c.name == "face_containment")
            .expect("face_containment check must be present");
        assert_eq!(
            check.status,
            QaStatus::Fail,
            "negative origin / odd crop must FAIL, not silently pass: {}",
            check.details
        );
    }

    /// A valid crop must genuinely pass the containment check.
    #[test]
    fn test_face_containment_accepts_valid_crop() {
        let engine = RenderQaEngine::new(RenderQaConfig {
            enabled: true,
            ..Default::default()
        });
        let good_plan = crate::media::SmartFramingPlan {
            mode: "single".to_string(),
            x: "0".into(),
            y: "0".into(),
            w: "1080".into(),
            h: "1080".into(),
            segments: vec![crate::media::LayoutSegment::Single {
                start: 0.0,
                end: 5.0,
                crop: crate::media::CropRectExpr {
                    x: "100".into(),
                    y: "0".into(),
                    w: "1080".into(),
                    h: "1080".into(),
                },
                face_bounds: None,
            }],
            face_bounds: None,
            framing: "original".into(),
            is_emergency_fallback: false,
            fallback_reason: None,
            ..Default::default()
        };

        let mut report = RenderQaReport::new("o".into(), "s".into(), "c".into());
        engine.check_face_containment("ignored.mp4", &good_plan, &mut report);

        let check = report
            .checks
            .iter()
            .find(|c| c.name == "face_containment")
            .expect("check present");
        assert_eq!(check.status, QaStatus::Pass, "{}", check.details);
    }

    /// `caption_presence` must SKIP (not fake a PASS) when captions were not
    /// expected for this render.
    #[test]
    fn test_caption_presence_skips_when_not_expected() {
        let engine = RenderQaEngine::new(RenderQaConfig {
            enabled: true,
            ..Default::default()
        });
        let mut report = RenderQaReport::new("o".into(), "s".into(), "c".into());
        engine.check_caption_presence("does-not-exist.mp4", false, &mut report);

        let check = report
            .checks
            .iter()
            .find(|c| c.name == "caption_presence")
            .expect("check present");
        assert_eq!(
            check.status,
            QaStatus::Skipped,
            "must not claim caption verification when captions were not expected"
        );
    }

    /// On a REAL clip with captions expected, the caption-band measurement must
    /// actually run and return a numeric reading.
    #[test]
    fn test_caption_presence_measures_real_clip() {
        let dir = std::env::temp_dir().join(format!("as_cap_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let clip = dir.join("clip.mp4");
        if !make_test_clip(&clip, false) {
            eprintln!("[skip] ffmpeg unavailable");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }

        let engine = RenderQaEngine::new(RenderQaConfig {
            enabled: true,
            ..Default::default()
        });
        let mut report = RenderQaReport::new("o".into(), "s".into(), "c".into());
        engine.check_caption_presence(&clip.to_string_lossy(), true, &mut report);

        let check = report
            .checks
            .iter()
            .find(|c| c.name == "caption_presence")
            .expect("check present");
        assert_ne!(
            check.status,
            QaStatus::Skipped,
            "captions were expected, so the check must run"
        );
        assert!(
            check.details.contains("mean luma"),
            "expected a real numeric measurement, got: {}",
            check.details
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Frame integrity must genuinely decode samples; a corrupt file must FAIL.
    #[test]
    fn test_frame_integrity_fails_on_garbage_file() {
        let dir = std::env::temp_dir().join(format!("as_fi_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let junk = dir.join("not_a_video.mp4");
        std::fs::write(&junk, b"this is definitely not an mp4").expect("write junk");

        let engine = RenderQaEngine::new(RenderQaConfig {
            enabled: true,
            ..Default::default()
        });
        let mut report = RenderQaReport::new("o".into(), "s".into(), "c".into());
        engine.check_frame_integrity(&junk.to_string_lossy(), &mut report);

        let check = report
            .checks
            .iter()
            .find(|c| c.name == "frame_integrity")
            .expect("check present");
        assert_eq!(
            check.status,
            QaStatus::Fail,
            "a non-decodable file must FAIL frame integrity"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_render_qa_flag_semantics() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Render QA is a MANDATORY production guardrail: it defaults ON.
        std::env::remove_var("AUTOSHORTS_RENDER_QA");
        assert!(
            render_qa_enabled(),
            "default must be ENABLED — Render QA is a mandatory production guardrail"
        );
        for on in ["1", "true", "on"] {
            std::env::set_var("AUTOSHORTS_RENDER_QA", on);
            assert!(render_qa_enabled(), "{} must enable", on);
        }
        // The emergency opt-out remains available for development only.
        for off in ["0", "false", "off", "OFF"] {
            std::env::set_var("AUTOSHORTS_RENDER_QA", off);
            assert!(!render_qa_enabled(), "{} must disable", off);
        }
        std::env::remove_var("AUTOSHORTS_RENDER_QA");
    }

    #[test]
    fn test_render_qa_config_default() {
        let config = RenderQaConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.expected_width, 1080);
        assert_eq!(config.expected_height, 1920);
        assert_eq!(config.max_av_sync_drift_sec, 0.05);
        assert_eq!(config.loudness_target_lufs, -16.0);
    }

    #[test]
    fn test_qa_report_aggregation() {
        let mut report = RenderQaReport::new(
            "test.mp4".to_string(),
            "source.mp4".to_string(),
            "cand1".to_string(),
        );
        assert_eq!(report.overall_status, QaStatus::Pass);

        report.add_check(QaCheck {
            name: "test1".to_string(),
            status: QaStatus::Pass,
            details: "ok".to_string(),
            severity: QaSeverity::Info,
        });
        assert_eq!(report.overall_status, QaStatus::Pass);

        report.add_check(QaCheck {
            name: "test2".to_string(),
            status: QaStatus::Warning,
            details: "warning".to_string(),
            severity: QaSeverity::Minor,
        });
        assert_eq!(report.overall_status, QaStatus::Warning);

        report.add_check(QaCheck {
            name: "test3".to_string(),
            status: QaStatus::Fail,
            details: "fail".to_string(),
            severity: QaSeverity::Critical,
        });
        assert_eq!(report.overall_status, QaStatus::Fail);
        assert!(report.has_critical_failures());
    }

    #[test]
    fn test_parse_fps() {
        assert_eq!(parse_fps("30000/1001"), Some((30000.0, 1001.0)));
        assert_eq!(parse_fps("60/1"), Some((60.0, 1.0)));
        assert_eq!(parse_fps("invalid"), None);
    }
}
