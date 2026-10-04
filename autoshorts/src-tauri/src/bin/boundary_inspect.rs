//! boundary_inspect — AutoShorts 8.0 Hook & Ending Optimization inspection CLI.
//!
//! Exposes the REAL production boundary-optimization pipeline (optimize →
//! pacing on the optimized range → framing on the optimized range → remap →
//! ASS/SRT) as a JSON-emitting command so the Python test suite and
//! forensic validation can exercise the exact code paths the app uses,
//! without launching the Tauri app.
//!
//! Usage:
//!   boundary_inspect <start_sec> <end_sec> <words_json_path>
//!                    [candidate_json_path] [source_path] [style]
//!
//!   words_json_path     — NormalizedTranscript JSON (or a bare array of
//!                         {text, start, end, speaker?} word objects).
//!   candidate_json_path — optional {hook?, hookStartSec?, hookEndSec?,
//!                         hookConfidence?, openingContextScore?}.
//!   source_path         — optional media file; when present AND probeable,
//!                         pacing + framing + ASS/SRT run on the OPTIMIZED
//!                         range (mirroring the production render path).
//!                         When absent, only the optimization runs
//!                         (pacing/framing are null).
//!
//! Emits one JSON object on stdout:
//!   { startSec, endSec, videoDuration, optimization, pacingRange,
//!     framingRange, pacing, remappedWords, framing, ass, srt, summary }

use autoshorts_lib::boundary::{optimize_boundaries, BoundaryCandidateMeta, BoundaryOptimization};
use autoshorts_lib::models::{NormalizedTranscript, TranscriptWord};
use autoshorts_lib::pacing::{
    build_paced_filtergraph, build_render_pieces, plan_smart_pacing_v2, remap_framing_plan,
    remap_words, SmartPacingPlan,
};
use autoshorts_lib::{generate_layout_aware_kinetic_ass_subtitles, generate_srt};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateJson {
    hook: Option<String>,
    hook_start_sec: Option<f64>,
    hook_end_sec: Option<f64>,
    hook_confidence: Option<f64>,
    opening_context_score: Option<f64>,
    payoff_end_sec: Option<f64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: boundary_inspect <start_sec> <end_sec> <words_json_path> [candidate_json_path] [source_path] [style]"
        );
        std::process::exit(2);
    }
    let start_sec: f64 = args[1].parse().expect("start_sec must be a number");
    let end_sec: f64 = args[2].parse().expect("end_sec must be a number");
    let words_path = args[3].clone();
    let candidate_path = args.get(4).filter(|s| !s.is_empty()).cloned();
    let source = args
        .get(5)
        .filter(|s| !s.is_empty() && std::path::Path::new(s).exists())
        .cloned();
    let style = args
        .get(6)
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| "modern-box".to_string());

    // Words: NormalizedTranscript OR bare array of word objects.
    let words_json = std::fs::read_to_string(&words_path).expect("failed to read words JSON");
    let words: Vec<TranscriptWord> = match serde_json::from_str::<NormalizedTranscript>(&words_json)
    {
        Ok(t) => t.words,
        Err(_) => serde_json::from_str(&words_json)
            .expect("words JSON must be a NormalizedTranscript or a bare array of word objects"),
    };

    // Candidate metadata (optional).
    let candidate: Option<CandidateJson> = candidate_path
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|j| serde_json::from_str(&j).ok());

    // Probe the source when available (duration + dimensions).
    let probe = source.as_ref().and_then(|s| {
        let r = autoshorts_lib::media::probe_media(s).ok();
        if r.is_none() {
            eprintln!("[boundary_inspect] probe_media failed for {}", s);
        }
        r
    });
    let video_duration = probe
        .as_ref()
        .and_then(|p| p.duration_sec)
        .unwrap_or(end_sec);

    // ── Stage 1: boundary optimization (SOURCE → OPTIMIZED) ────────────
    let meta = candidate.as_ref().map(|c| BoundaryCandidateMeta {
        hook: c.hook.as_deref(),
        hook_start_sec: c.hook_start_sec,
        hook_end_sec: c.hook_end_sec,
        hook_confidence: c.hook_confidence,
        opening_context_score: c.opening_context_score,
        payoff_end_sec: c.payoff_end_sec,
    });
    let opt: BoundaryOptimization =
        optimize_boundaries(&words, start_sec, end_sec, video_duration, meta.as_ref());
    let render_start = opt.optimized_start_sec;
    let render_end = opt.optimized_end_sec;

    let mut out = serde_json::json!({
        "startSec": start_sec,
        "endSec": end_sec,
        "videoDuration": video_duration,
        "optimization": serde_json::to_value(&opt).unwrap(),
        "pacingRange": { "startSec": render_start, "endSec": render_end },
        "framingRange": { "startSec": render_start, "endSec": render_end },
        "pacing": serde_json::Value::Null,
        "remappedWords": serde_json::Value::Null,
        "framing": serde_json::Value::Null,
        "ass": serde_json::Value::String(String::new()),
        "srt": serde_json::Value::String(String::new()),
        "summary": serde_json::Value::String(opt.summary()),
    });

    // ── Stage 2: production downstream path on the OPTIMIZED range ──────
    // (only when a probeable source is provided; otherwise optimization-only)
    if let (Some(source), Some(probe)) = (source.as_ref(), probe.as_ref()) {
        let iw = probe.width.unwrap_or(1920);
        let ih = probe.height.unwrap_or(1080);
        let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
        if crop_w % 2 != 0 {
            crop_w -= 1;
        }

        // Framing ALWAYS runs on the full OPTIMIZED range (production path:
        // pacing never shortens the range the camera sampler sees).
        let framing = autoshorts_lib::media::detect_speaker_crop_params(
            source,
            render_start,
            render_end,
            iw,
            ih,
            crop_w,
            Some(&words),
            "original",
        );
        out["framing"] = serde_json::to_value(&framing).unwrap();

        // Pacing on the OPTIMIZED range.
        let plan: Option<SmartPacingPlan> =
            plan_smart_pacing_v2(source, render_start, render_end, Some(&words));
        let clip_dur = (render_end - render_start).max(0.0);

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

                out["pacing"] = serde_json::to_value(&plan).unwrap();
                out["remappedWords"] = serde_json::to_value(&remapped_words).unwrap();
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
                out["pacingSummary"] =
                    serde_json::Value::String(autoshorts_lib::pacing::pacing_summary(&plan));
            }
            None => {
                // Production None-pacing path: legacy captions on the
                // OPTIMIZED range with the original (unremapped) words.
                let caption_start = render_start;
                let caption_end = render_end;
                let ass = if words.is_empty() {
                    String::new()
                } else {
                    generate_layout_aware_kinetic_ass_subtitles(
                        &words,
                        caption_start,
                        caption_end,
                        &style,
                        Some(&framing),
                    )
                };
                let srt = if words.is_empty() {
                    String::new()
                } else {
                    generate_srt(&words, caption_start, caption_end)
                };
                out["remappedWords"] = serde_json::to_value(&words).unwrap();
                out["ass"] = serde_json::Value::String(ass);
                out["srt"] = serde_json::Value::String(srt);
                out["skippedReason"] = serde_json::Value::String(
                    "engine returned no plan (disabled / no safe edits / unavailable)".into(),
                );
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
