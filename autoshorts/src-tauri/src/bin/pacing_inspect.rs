//! pacing_inspect — AutoShorts 8.0 Smart Pacing inspection CLI.
//!
//! Exposes the REAL production pacing pipeline (plan -> remap -> pieces ->
//! filtergraph -> ASS) as a JSON-emitting command so the Python test suite
//! and forensic validation can exercise the exact code paths the app uses,
//! without launching the Tauri app.
//!
//! Usage:
//!   pacing_inspect <source> <start_sec> <end_sec> <words_json_path> [style]
//!
//! Emits one JSON object on stdout:
//!   { plan, skippedReason?, remappedWords, pacedFraming, pieces,
//!     filtergraph, ass, summary }

use std::path::Path;

use autoshorts_lib::models::NormalizedTranscript;
use autoshorts_lib::pacing::{
    build_paced_filtergraph, build_render_pieces, plan_smart_pacing_v2, remap_framing_plan,
    remap_words, SmartPacingPlan,
};
use autoshorts_lib::{generate_layout_aware_kinetic_ass_subtitles, generate_srt};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: pacing_inspect <source> <start_sec> <end_sec> <words_json_path> [style]");
        std::process::exit(2);
    }
    let source = args[1].clone();
    let start_sec: f64 = args[2].parse().expect("start_sec must be a number");
    let end_sec: f64 = args[3].parse().expect("end_sec must be a number");
    let words_path = args[4].clone();
    let style = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "modern-box".to_string());

    let words_json = std::fs::read_to_string(&words_path).expect("failed to read words JSON");
    let transcript: NormalizedTranscript =
        serde_json::from_str(&words_json).expect("words JSON must be a NormalizedTranscript");
    let words = transcript.words;

    let probe = autoshorts_lib::media::probe_media(&source).expect("failed to probe source");
    let iw = probe.width.unwrap_or(1920);
    let ih = probe.height.unwrap_or(1080);
    let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
    if crop_w % 2 != 0 {
        crop_w -= 1;
    }

    // Full-range framing (production path: pacing never shortens the range
    // the camera sampler sees).
    let framing = autoshorts_lib::media::detect_speaker_crop_params(
        &source,
        start_sec,
        end_sec,
        iw,
        ih,
        crop_w,
        Some(&words),
        "original",
    );

    let plan: Option<SmartPacingPlan> =
        plan_smart_pacing_v2(&source, start_sec, end_sec, Some(&words));

    let clip_dur = (end_sec - start_sec).max(0.0);

    let mut out = serde_json::json!({
        "source": source,
        "startSec": start_sec,
        "endSec": end_sec,
        "clipDurationSec": clip_dur,
        "framingSegments": framing.segments.len(),
        "framingMode": framing.mode,
    });

    match plan {
        Some(plan) => {
            let remapped_words = remap_words(&plan, &words);
            let paced_framing = remap_framing_plan(&framing, &plan, clip_dur);
            let pieces = build_render_pieces(&framing, &plan, clip_dur);
            let (graph, vlabel, alabel) = build_paced_filtergraph(
                &pieces,
                plan.output_duration_sec,
                probe.audio_codec.is_some(),
                None,
                None,
                None,
            )
            .expect("failed to build paced filtergraph");

            let caption_start = plan.clip_start_sec;
            let caption_end = plan.clip_start_sec + plan.output_duration_sec;
            let ass = if remapped_words.is_empty() {
                String::new()
            } else {
                generate_layout_aware_kinetic_ass_subtitles(
                    &remapped_words,
                    caption_start,
                    caption_end,
                    &style,
                    Some(&paced_framing),
                )
            };
            let srt = if remapped_words.is_empty() {
                String::new()
            } else {
                generate_srt(&remapped_words, caption_start, caption_end)
            };

            out["plan"] = serde_json::to_value(&plan).unwrap();
            out["remappedWords"] = serde_json::to_value(&remapped_words).unwrap();
            out["pacedFraming"] = serde_json::to_value(&paced_framing).unwrap();
            out["pieces"] = serde_json::to_value(
                pieces
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "srcStart": p.src_start,
                            "srcEnd": p.src_end,
                            "outStart": p.out_start,
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            out["filtergraph"] = serde_json::Value::String(graph);
            out["videoLabel"] = serde_json::Value::String(vlabel);
            out["audioLabel"] = alabel
                .map(serde_json::Value::String)
                .unwrap_or(serde_json::Value::Null);
            out["ass"] = serde_json::Value::String(ass);
            out["srt"] = serde_json::Value::String(srt);
            out["summary"] =
                serde_json::Value::String(autoshorts_lib::pacing::pacing_summary(&plan));
        }
        None => {
            out["skippedReason"] = serde_json::Value::String(
                "engine returned no plan (disabled / no safe edits / unavailable)".into(),
            );
        }
    }

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    let _ = Path::new(&words_path);
}
