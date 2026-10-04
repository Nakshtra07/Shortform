// ── Audio Intelligence (AutoShorts 8.0) ─────────────────────────────────────
//
// SPEECH QUALITY: "Good audio -> preserve it. Problematic audio -> make the
// smallest safe correction. Uncertain audio -> do not touch it."
//
// This module is the Rust side of a two-part system:
//   1. `scripts/audio_intelligence.py` (sidecar) — measures the clip with
//      FFmpeg (loudnorm / astats / volumedetect / silencedetect) and emits a
//      staged, evidence-gated decision: which corrections are justified, in
//      order, each with its measured evidence. It NEVER touches media.
//   2. This module — invokes the sidecar, validates the plan, and appends the
//      emitted filter chain to the EXISTING render (single encode, no
//      re-encode, no intermediate files). When the plan is None the render is
//      byte-identical to the legacy path (additive-only integration).
//
// Stages (in chain order): highpass (rumble) -> afftdn (noise) ->
// volume@speaker (balance) -> loudnorm (loudness) -> aresample ->
// alimiter (peak safety). Every stage is applied ONLY with measured
// evidence; anything unmeasured or uncertain is skipped.
//
// Kill switch: AUTOSHORTS_AUDIO_INTELLIGENCE=0|false|off disables the
// feature entirely (plan_audio_intelligence returns None).

use crate::media::find_python_cmd;
use crate::models::TranscriptWord;
use crate::pacing::SmartPacingPlan;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;

// ── Plan types (mirror of the sidecar's JSON, camelCase) ────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioStage {
    pub name: String,
    pub applied: bool,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioAnalysis {
    #[serde(default)]
    pub input_i: Option<f64>,
    #[serde(default)]
    pub input_tp: Option<f64>,
    #[serde(default)]
    pub input_lra: Option<f64>,
    #[serde(default)]
    pub input_thresh: Option<f64>,
    #[serde(default)]
    pub target_offset: Option<f64>,
    #[serde(default)]
    pub astats_peak_db: Option<f64>,
    #[serde(default)]
    pub astats_rms_db: Option<f64>,
    #[serde(default)]
    pub clipped: Option<bool>,
    #[serde(default)]
    pub min_level: Option<f64>,
    #[serde(default)]
    pub max_level: Option<f64>,
    #[serde(default)]
    pub highpass_mean_db: Option<f64>,
    #[serde(default)]
    pub noise_floor_db: Option<f64>,
    #[serde(default)]
    pub silence_ratio: Option<f64>,
    #[serde(default)]
    pub speaker_means_db: Option<std::collections::BTreeMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioIntelligencePlan {
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub stages: Vec<AudioStage>,
    #[serde(default)]
    pub filter_chain: String,
    #[serde(default)]
    pub analysis: AudioAnalysis,
}

impl AudioIntelligencePlan {
    /// The plan is actionable only when the engine ran, produced a non-empty
    /// chain, and every filter is known-safe with conservative parameters.
    /// This is the last line of defense between the sidecar and the render.
    pub fn validate(&self) -> Result<()> {
        if self.status != "ok" {
            return Err(anyhow!("status is {:?}", self.status));
        }
        if self.filter_chain.trim().is_empty() {
            return Err(anyhow!("empty filter chain"));
        }
        for part in split_top_level(&self.filter_chain) {
            validate_filter(part.trim())?;
        }
        Ok(())
    }
}

/// Split a filter chain on commas that are NOT inside single quotes (the
/// speaker-balance volume filter embeds `between(t,1,2)+...` in quotes).
fn split_top_level(chain: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quote = false;
    for ch in chain.chars() {
        match ch {
            '\'' => {
                in_quote = !in_quote;
                current.push(ch);
            }
            ',' if !in_quote => {
                parts.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    parts.push(current);
    parts
}

/// Validate one filter element against the conservative allowlist.
fn validate_filter(p: &str) -> Result<()> {
    if p.is_empty() {
        return Ok(()); // trailing commas are tolerated
    }
    // Exact-match filters (no parameters to check).
    if p == "highpass=f=80"
        || p == "aresample=48000"
        || p == "alimiter=limit=0.841:level=false:attack=5:release=50"
    {
        return Ok(());
    }
    // afftdn=nf=<dB>:nr=10 — nf must be within the filter's legal range.
    if let Some(rest) = p.strip_prefix("afftdn=nf=") {
        let mut it = rest.splitn(2, ":nr=");
        let nf: f64 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("bad afftdn nf"))?;
        let nr: f64 = it
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| anyhow!("bad afftdn nr"))?;
        if !(-80.0..=-20.0).contains(&nf) {
            return Err(anyhow!("afftdn nf {} out of range", nf));
        }
        if (nr - 10.0).abs() > 0.01 {
            return Err(anyhow!("afftdn nr {} != 10", nr));
        }
        return Ok(());
    }
    // volume=<gain>dB:enable='<between terms>' — speaker boost, capped.
    if let Some(rest) = p.strip_prefix("volume=") {
        let (gain_s, enable_s) = rest
            .split_once("dB:enable='")
            .ok_or_else(|| anyhow!("malformed volume filter"))?;
        let gain: f64 = gain_s.parse().map_err(|_| anyhow!("bad volume gain"))?;
        if !(0.0..=6.0).contains(&gain) {
            return Err(anyhow!("volume gain {} outside 0..6 dB cap", gain));
        }
        let enable = enable_s
            .strip_suffix('\'')
            .ok_or_else(|| anyhow!("volume enable expression not quoted"))?;
        for term in enable.split('+') {
            let t = term.trim();
            let inner = t
                .strip_prefix("between(t,")
                .and_then(|x| x.strip_suffix(')'))
                .ok_or_else(|| anyhow!("disallowed enable term: {}", t))?;
            let mut nums = inner.split(',');
            let s: f64 = nums
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| anyhow!("bad enable start"))?;
            let e: f64 = nums
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| anyhow!("bad enable end"))?;
            if !(0.0..=700.0).contains(&s) || !(0.0..=700.0).contains(&e) || e < s {
                return Err(anyhow!("enable interval {},{} out of bounds", s, e));
            }
        }
        return Ok(());
    }
    // loudnorm=I=<i>:TP=<tp>:LRA=<lra>:measured_*=...:offset=<o>:linear=true
    // Targets are pinned by the engine; parse them (the sidecar's number
    // formatter strips trailing zeros, so match values, not strings).
    if let Some(rest) = p.strip_prefix("loudnorm=") {
        let mut i: Option<f64> = None;
        let mut tp: Option<f64> = None;
        let mut lra: Option<f64> = None;
        let mut linear = false;
        for kv in rest.split(':') {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| anyhow!("malformed loudnorm parameter: {}", kv))?;
            match k {
                "I" => i = v.parse().ok(),
                "TP" => tp = v.parse().ok(),
                "LRA" => lra = v.parse().ok(),
                "linear" => linear = v == "true",
                _ => {} // measured_* / offset — engine-provided, not pinned
            }
        }
        let i = i.ok_or_else(|| anyhow!("loudnorm missing I"))?;
        let tp = tp.ok_or_else(|| anyhow!("loudnorm missing TP"))?;
        let lra = lra.ok_or_else(|| anyhow!("loudnorm missing LRA"))?;
        if (i - (-16.0)).abs() > 0.01 {
            return Err(anyhow!("loudnorm I {} != -16 LUFS target", i));
        }
        if (tp - (-1.5)).abs() > 0.01 {
            return Err(anyhow!("loudnorm TP {} != -1.5 dBTP ceiling", tp));
        }
        if !(11.0..=50.0).contains(&lra) {
            return Err(anyhow!("loudnorm LRA {} out of range", lra));
        }
        if !linear {
            return Err(anyhow!("loudnorm must be linear (dynamics preserved)"));
        }
        return Ok(());
    }
    Err(anyhow!("disallowed filter element: {}", p))
}

// ── Feature switch ──────────────────────────────────────────────────────────

/// Wall-clock budget for the audio intelligence sidecar (ADVISORY).
const AUDIO_INTEL_TIMEOUT_FLOOR_SEC: f64 = 120.0;
const AUDIO_INTEL_TIMEOUT_REALTIME_FACTOR: f64 = 4.0;
const AUDIO_INTEL_TIMEOUT_CEILING_SEC: f64 = 900.0;

fn audio_intel_timeout_for(clip_dur: f64) -> std::time::Duration {
    let secs = if clip_dur.is_finite() && clip_dur > 0.0 {
        AUDIO_INTEL_TIMEOUT_REALTIME_FACTOR * clip_dur
    } else {
        AUDIO_INTEL_TIMEOUT_FLOOR_SEC
    };
    std::time::Duration::from_secs_f64(secs.clamp(
        AUDIO_INTEL_TIMEOUT_FLOOR_SEC,
        AUDIO_INTEL_TIMEOUT_CEILING_SEC,
    ))
}

/// success/stdout/stderr carrier for the bounded audio intelligence result.
struct AudioIntelOutput {
    success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

pub fn audio_intelligence_enabled() -> bool {
    match std::env::var("AUTOSHORTS_AUDIO_INTELLIGENCE") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

// ── Sidecar discovery (mirrors find_smart_pacing_script) ────────────────────

fn find_audio_intelligence_script() -> Option<PathBuf> {
    let script_name = "audio_intelligence.py";
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
    let dev_paths = [
        PathBuf::from("src-tauri/scripts").join(script_name),
        PathBuf::from("scripts").join(script_name),
        PathBuf::from("../src-tauri/scripts").join(script_name),
        PathBuf::from("autoshorts/src-tauri/scripts").join(script_name),
    ];
    for p in &dev_paths {
        if p.exists() {
            return Some(p.clone());
        }
    }
    None
}

// ── Plan summary (render log / telemetry) ────────────────────────────────────

/// Human-readable one-line summary for the render log.
pub fn audio_summary(plan: &AudioIntelligencePlan) -> String {
    let applied: Vec<&str> = plan
        .stages
        .iter()
        .filter(|s| s.applied)
        .map(|s| s.name.as_str())
        .collect();
    let measured = plan
        .analysis
        .input_i
        .map(|i| format!(" {:.1} LUFS", i))
        .unwrap_or_default();
    format!(
        "[Audio Intelligence] {}{} -> {}",
        if applied.is_empty() {
            "no corrections".to_string()
        } else {
            format!("applied: {}", applied.join(", "))
        },
        measured,
        plan.reason.as_deref().unwrap_or("")
    )
}

// ── Sidecar invocation ──────────────────────────────────────────────────────

/// Run the Python audio-intelligence engine for a candidate range. Returns
/// None when the feature is disabled, the engine is unavailable, or it
/// decided the audio needs no correction — in every such case the caller
/// renders the legacy clip with untouched audio.
///
/// `words` are on the SOURCE timeline (the sidecar remaps them internally
/// when a pacing plan is active, mirroring remap_words). `pacing` is the
/// already-validated pacing plan for the same range, if any.
pub fn plan_audio_intelligence(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    words: Option<&[TranscriptWord]>,
    pacing: Option<&SmartPacingPlan>,
) -> Option<AudioIntelligencePlan> {
    if !audio_intelligence_enabled() {
        return None;
    }
    if end_sec <= start_sec {
        return None;
    }
    let script = find_audio_intelligence_script()?;
    let python = find_python_cmd();

    // Words file (empty file when no transcript — the sidecar treats an
    // empty list the same as no words: speaker analysis simply won't run).
    let words_tmp = std::env::temp_dir().join(format!(
        "autoshorts_audio_words_{}.json",
        uuid::Uuid::new_v4()
    ));
    let words_json = match words {
        Some(w) => serde_json::to_string(w).ok()?,
        None => "[]".to_string(),
    };
    std::fs::write(&words_tmp, words_json).ok()?;

    // Pacing file (only when a validated plan exists).
    let pacing_tmp = pacing.map(|p| {
        let tmp = std::env::temp_dir().join(format!(
            "autoshorts_audio_pacing_{}.json",
            uuid::Uuid::new_v4()
        ));
        (tmp, serde_json::to_string(p).unwrap_or_default())
    });
    if let Some((tmp, json)) = pacing_tmp.as_ref() {
        std::fs::write(tmp, json).ok()?;
    }

    let start_ms = (start_sec * 1000.0) as i64;
    let end_ms = (end_sec * 1000.0) as i64;
    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg(source_path)
        .arg(start_ms.to_string())
        .arg(end_ms.to_string())
        .arg(&words_tmp);
    if let Some((tmp, _)) = pacing_tmp.as_ref() {
        cmd.arg(tmp);
    }
    // Bounded: advisory audio analysis (Silero VAD / pause / denoise /
    // loudness) over the candidate range. Advisory, so a timeout degrades to
    // "no correction chain" and the render proceeds on untouched audio.
    let clip_dur = (end_sec - start_sec).max(0.0);
    let budget = audio_intel_timeout_for(clip_dur);
    eprintln!(
        "[Audio Intelligence] sidecar START budget={:.0}s",
        budget.as_secs_f64()
    );
    let result = match crate::proc_guard::run_bounded(&mut cmd, budget, "AudioIntel/sidecar") {
        Ok(o) => {
            if o.timed_out {
                eprintln!(
                    "[Audio Intelligence] sidecar TIMEOUT after {:.0}s (budget {:.0}s) -> untouched audio",
                    o.elapsed.as_secs_f64(),
                    budget.as_secs_f64()
                );
            } else {
                eprintln!(
                    "[Audio Intelligence] sidecar COMPLETE in {:.1}s rc={:?}",
                    o.elapsed.as_secs_f64(),
                    o.code
                );
            }
            Ok(AudioIntelOutput {
                success: o.success,
                stdout: o.stdout.into_bytes(),
                stderr: o.stderr.into_bytes(),
            })
        }
        Err(e) => Err(e),
    };
    let _ = std::fs::remove_file(&words_tmp);
    if let Some((tmp, _)) = pacing_tmp.as_ref() {
        let _ = std::fs::remove_file(tmp);
    }

    let out = result.ok()?;
    if !out.success {
        eprintln!(
            "[Audio Intelligence] engine exited non-zero: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let last_line = stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('['))
        .last()?
        .to_string();
    let plan: AudioIntelligencePlan = serde_json::from_str(&last_line).ok()?;
    if plan.status != "ok" || plan.filter_chain.trim().is_empty() {
        return None;
    }
    if let Err(e) = plan.validate() {
        eprintln!("[Audio Intelligence] rejecting engine plan: {e}");
        return None;
    }
    eprintln!("{}", audio_summary(&plan));
    Some(plan)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_with_chain(chain: &str) -> AudioIntelligencePlan {
        AudioIntelligencePlan {
            status: "ok".to_string(),
            reason: None,
            stages: vec![],
            filter_chain: chain.to_string(),
            analysis: AudioAnalysis::default(),
        }
    }

    // A. Every chain shape the sidecar can legally emit must validate.
    #[test]
    fn test_validate_accepts_all_engine_chain_shapes() {
        // Real Beat-footage chain (trailing zeros stripped by the engine's
        // number formatter: I=-16, not I=-16.0).
        assert!(plan_with_chain(
            "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-24.48:measured_LRA=1.8:\
             measured_TP=-8.82:measured_thresh=-35.38:offset=-0.24:linear=true,\
             aresample=48000"
        )
        .validate()
        .is_ok());
        // Rumble + loudness with remeasure.
        assert!(plan_with_chain(
            "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-24.98:\
             measured_LRA=2.1:measured_TP=-11.44:measured_thresh=-36.1:\
             offset=-0.1:linear=true,aresample=48000"
        )
        .validate()
        .is_ok());
        // Denoise chain (floor-hunt path).
        assert!(plan_with_chain(
            "afftdn=nf=-31.443:nr=10,loudnorm=I=-16:TP=-1.5:LRA=11:\
             measured_I=-25.1:measured_LRA=3:measured_TP=-9.5:\
             measured_thresh=-36:offset=0:linear=true,aresample=48000"
        )
        .validate()
        .is_ok());
        // Speaker balance: quoted enable with embedded commas + limiter.
        assert!(plan_with_chain(
            "volume=6.00dB:enable='between(t,0,1)+between(t,2,3)',\
             alimiter=limit=0.841:level=false:attack=5:release=50"
        )
        .validate()
        .is_ok());
        // Limiter-only (clipped input).
        assert!(
            plan_with_chain("alimiter=limit=0.841:level=false:attack=5:release=50")
                .validate()
                .is_ok()
        );
        // Highpass only.
        assert!(plan_with_chain("highpass=f=80").validate().is_ok());
    }

    // B. Anything outside the allowlist is rejected.
    #[test]
    fn test_validate_rejects_disallowed_filters() {
        // Arbitrary filters.
        assert!(plan_with_chain("acrusher=bits=10").validate().is_err());
        assert!(plan_with_chain("volume=20dB").validate().is_err()); // no enable
        assert!(plan_with_chain("atempo=2.0").validate().is_err());
        // Wrong targets.
        assert!(plan_with_chain("loudnorm=I=-6:TP=-1.5:LRA=11:linear=true")
            .validate()
            .is_err());
        assert!(
            plan_with_chain("loudnorm=I=-16:TP=0:TP_x=LRA=11:linear=true")
                .validate()
                .is_err()
        );
        // Speaker boost over the 6 dB cap.
        assert!(plan_with_chain("volume=9.00dB:enable='between(t,0,1)'")
            .validate()
            .is_err());
        // afftdn with out-of-range floor.
        assert!(plan_with_chain("afftdn=nf=-5:nr=10").validate().is_err());
        assert!(plan_with_chain("afftdn=nf=-31:nr=25").validate().is_err());
        // Non-linear loudnorm (dynamics would be squashed).
        assert!(
            plan_with_chain("loudnorm=I=-16:TP=-1.5:LRA=11:linear=false")
                .validate()
                .is_err()
        );
        // Empty chain.
        assert!(plan_with_chain("").validate().is_err());
        assert!(plan_with_chain("   ").validate().is_err());
    }

    // C. Quote-aware splitting: commas inside enable='...' stay intact.
    #[test]
    fn test_split_top_level_respects_quotes() {
        let parts =
            split_top_level("volume=6.00dB:enable='between(t,0,1)+between(t,2,3)',aresample=48000");
        assert_eq!(parts.len(), 2);
        assert!(parts[0].contains("between(t,0,1)+between(t,2,3)"));
        assert_eq!(parts[1], "aresample=48000");
    }

    // D. Kill switch parsing (mirrors the pacing switch semantics).
    #[test]
    fn test_kill_switch_env_parsing() {
        std::env::remove_var("AUTOSHORTS_AUDIO_INTELLIGENCE");
        assert!(audio_intelligence_enabled());
        for off in ["0", "false", "off", "OFF", "False"] {
            std::env::set_var("AUTOSHORTS_AUDIO_INTELLIGENCE", off);
            assert!(!audio_intelligence_enabled(), "{off} must disable");
        }
        std::env::set_var("AUTOSHORTS_AUDIO_INTELLIGENCE", "1");
        assert!(audio_intelligence_enabled());
        std::env::remove_var("AUTOSHORTS_AUDIO_INTELLIGENCE");
    }

    // E. Summary formatting.
    #[test]
    fn test_audio_summary_formats() {
        let mut plan = plan_with_chain("highpass=f=80");
        plan.stages = vec![
            AudioStage {
                name: "highpass".into(),
                applied: true,
                reason: String::new(),
            },
            AudioStage {
                name: "loudness".into(),
                applied: false,
                reason: String::new(),
            },
        ];
        plan.analysis.input_i = Some(-24.5);
        let s = audio_summary(&plan);
        assert!(s.contains("[Audio Intelligence]"));
        assert!(s.contains("highpass"));
        assert!(!s.contains("loudness,"));
        assert!(s.contains("-24.5 LUFS"));
    }

    // F. Deserialization of the sidecar's real JSON contract (camelCase,
    // clipped is a BOOLEAN, minLevel/maxLevel present).
    #[test]
    fn test_deserialize_real_sidecar_json() {
        let json = r#"{
            "status": "ok",
            "reason": "processed: loudness",
            "stages": [
                {"name": "loudness", "applied": true,
                 "inputI": -24.48, "targetI": -16.0, "remeasured": false,
                 "reason": "integrated loudness -24.5 LUFS deviates > 2 LU"}
            ],
            "filterChain": "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-24.48:measured_LRA=1.8:measured_TP=-8.82:measured_thresh=-35.38:offset=-0.24:linear=true,aresample=48000",
            "analysis": {
                "inputI": -24.48, "inputTp": -8.82, "inputLra": 1.8,
                "inputThresh": -35.38, "targetOffset": -0.24,
                "astatsPeakDb": -8.82, "astatsRmsDb": -22.1,
                "clipped": false, "minLevel": -0.362, "maxLevel": 0.359,
                "highpassMeanDb": -28.9, "noiseFloorDb": null,
                "silenceRatio": 0.02, "speakerMeansDb": null
            }
        }"#;
        let plan: AudioIntelligencePlan = serde_json::from_str(json).expect("parse");
        assert_eq!(plan.status, "ok");
        assert_eq!(plan.analysis.clipped, Some(false));
        assert_eq!(plan.analysis.min_level, Some(-0.362));
        assert_eq!(plan.analysis.max_level, Some(0.359));
        assert_eq!(plan.analysis.input_i, Some(-24.48));
        assert!(plan.validate().is_ok());
    }

    // G. Skipped/error sidecar outputs never produce a plan.
    #[test]
    fn test_reject_non_ok_and_empty_chain() {
        let mut p = plan_with_chain("highpass=f=80");
        p.status = "skipped".into();
        assert!(p.validate().is_err());
        let mut p = plan_with_chain("");
        p.status = "ok".into();
        assert!(p.validate().is_err());
    }
}
