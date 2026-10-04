//! audio_inspect — AutoShorts 8.0 Audio Intelligence inspection CLI.
//!
//! Exposes the REAL production audio-intelligence pipeline (sidecar plan ->
//! Rust validation -> filter chain -> optional render) as a JSON-emitting
//! command so the Python test suite and forensic validation can exercise the
//! exact code paths the app uses, without launching the Tauri app.
//!
//! Usage:
//!   audio_inspect <source> <start_sec> <end_sec> <words_json_path>
//!                 [pacing_json_path] [audio_out_path]
//!
//! - words_json_path: NormalizedTranscript JSON ("[]" also accepted — no
//!   speaker analysis in that case).
//! - pacing_json_path: a SmartPacingPlan JSON (as emitted by pacing_inspect's
//!   "plan" field) — when present, the sidecar analyzes the OUTPUT timeline
//!   of that edit map, exactly like the production render.
//! - audio_out_path: when given, renders the clip's audio through the
//!   validated chain (via the flat render path) so A/B loudness can be
//!   measured on the result.
//!
//! Emits one JSON object as the LAST stdout line (compact, single-line —
//! the same contract as the Python sidecar). Earlier stdout lines are the
//! render path's pre-existing diagnostics and must be ignored by callers.

use std::path::Path;

use autoshorts_lib::audio::{audio_summary, plan_audio_intelligence, AudioIntelligencePlan};
use autoshorts_lib::models::NormalizedTranscript;
use autoshorts_lib::pacing::SmartPacingPlan;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: audio_inspect <source> <start_sec> <end_sec> <words_json_path> \
             [pacing_json_path] [audio_out_path]"
        );
        std::process::exit(2);
    }
    let source = args[1].clone();
    let start_sec: f64 = args[2].parse().expect("start_sec must be a number");
    let end_sec: f64 = args[3].parse().expect("end_sec must be a number");
    let words_path = args[4].clone();
    // "-" is the explicit "absent" placeholder (PowerShell 5.1 drops empty
    // string arguments, so callers cannot pass "" reliably).
    let pacing_path = args
        .get(5)
        .filter(|p| !p.is_empty() && p.as_str() != "-")
        .cloned();
    let audio_out_path = args
        .get(6)
        .filter(|p| !p.is_empty() && p.as_str() != "-")
        .cloned();

    // Words: NormalizedTranscript {"words": [...]} or a bare list [...].
    let words: Vec<autoshorts_lib::models::TranscriptWord> = {
        let words_json = std::fs::read_to_string(&words_path).expect("failed to read words JSON");
        let trimmed = words_json.trim();
        if trimmed.starts_with('[') {
            serde_json::from_str(trimmed).expect("words JSON must be a list of TranscriptWord")
        } else {
            let transcript: NormalizedTranscript = serde_json::from_str(&words_json)
                .expect("words JSON must be a NormalizedTranscript");
            transcript.words
        }
    };

    // Pacing plan (optional): validated exactly like the production path —
    // an invalid/no-op plan is treated as absent.
    let pacing: Option<SmartPacingPlan> = pacing_path.and_then(|p| {
        let json = std::fs::read_to_string(&p).expect("failed to read pacing JSON");
        let plan: SmartPacingPlan = serde_json::from_str(&json).expect("invalid pacing JSON");
        if plan.status == "ok" && !plan.is_noop() && plan.validate().is_ok() {
            Some(plan)
        } else {
            None
        }
    });

    let mut out = serde_json::json!({
        "source": source,
        "startSec": start_sec,
        "endSec": end_sec,
        "pacingActive": pacing.is_some(),
    });

    let plan: Option<AudioIntelligencePlan> =
        plan_audio_intelligence(&source, start_sec, end_sec, Some(&words), pacing.as_ref());

    match plan {
        Some(plan) => {
            out["summary"] = serde_json::Value::String(audio_summary(&plan));
            out["filterChain"] = serde_json::Value::String(plan.filter_chain.clone());
            out["plan"] = serde_json::to_value(&plan).unwrap();

            // Optional: render the clip's audio through the validated chain
            // using the production flat-render path (video included, as in
            // the app; A/B loudness is then measured on the file).
            if let Some(out_path) = audio_out_path {
                let path = Path::new(&out_path).to_path_buf();
                match autoshorts_lib::media::render_flat_clip(
                    &source,
                    start_sec,
                    end_sec,
                    &path,
                    None,
                    None,
                    None,
                    None,
                    Some(plan.filter_chain.as_str()),
                ) {
                    Ok(rendered) => {
                        out["renderPath"] =
                            serde_json::Value::String(rendered.to_string_lossy().to_string());
                    }
                    Err(e) => {
                        out["renderError"] = serde_json::Value::String(e.to_string());
                    }
                }
            }
        }
        None => {
            out["skippedReason"] = serde_json::Value::String(
                "engine returned no plan (disabled / audio already good / \
                 unavailable / uncertain)"
                    .into(),
            );
        }
    }

    // The JSON goes out LAST and compact (single line): render diagnostics
    // from the production path print to stdout first, so callers take the
    // final line — exactly how the Rust side parses the Python sidecar.
    println!("{}", serde_json::to_string(&out).unwrap());
}
