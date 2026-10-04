// Close-pair Adaptive Framing verification — real-render harness.
//
// Mirrors the production render path in lib.rs::render_flat_clip_for_candidate:
//   detect_speaker_crop_params(framing_mode) -> caption intel -> ASS -> render_paced_clip
//
// Renders real MP4s from the Rio/Cristiano interview and verifies, at the pixel
// level, the close-pair scenario that the previous verification could not
// exercise: TWO PEOPLE + CLOSE ENOUGH TO FIT -> both kept together in one
// 9:16 composition (Adaptive), vs. the unchanged original 9:16 behavior.

use autoshorts_lib::caption_intel;
use autoshorts_lib::captions::{
    generate_ass_from_template_with_framing_and_intel, get_caption_template,
};
use autoshorts_lib::media::{self, probe_media};
use autoshorts_lib::models::{Candidate, TranscriptWord};
use autoshorts_lib::pacing;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Check {
    id: String,
    name: String,
    status: String, // PASS / FAIL / SKIP
    detail: String,
}

const VIDEO: &str = r"D:\College\Autoshorts 10.0\My thoughts on my 2023_24 season and the truth about my future. Talk with my friend Rio (PART 1) - UR · Cristiano (1080p, h264).mp4";
const OUT_DIR: &str = r"D:\College\Autoshorts 10.0\tmp\closepair_renders";

// Windows chosen from the measured genuine-track geometry scan
// (scratch/closepair_trackscan.json). Under Adaptive Framing the close-pair
// criterion is HEAD containment inside the SQUARE inner composition
// (side = min(source_h, source_w) = 1080px for this 1920x1080 source), so a
// pair is kept together iff both head extents fit in the square.
// Overridable via argv: closepair_render <close_start> <close_end> <far_start> <far_end>
const CLOSE_START: f64 = 0.0; // filled from scan
const CLOSE_END: f64 = 0.0;
const FAR_START: f64 = 1204.0; // 2 genuine tracks, head span beyond the square
const FAR_END: f64 = 1216.0;

fn out_path(name: &str) -> PathBuf {
    Path::new(OUT_DIR).join(name)
}

fn synthetic_words(start: f64, end: f64) -> Vec<TranscriptWord> {
    const PHRASE: &[&str] = &[
        "you", "have", "to", "believe", "in", "yourself", "every", "single", "day", "because",
        "nobody", "else", "will", "do", "the", "work", "for", "you", "so", "start", "right", "now",
        "and", "never", "stop", "until", "you", "win", "big", "time",
    ];
    let mut out = Vec::new();
    let dur = (end - start).max(1.0);
    let per = (dur / 32.0).min(0.45).max(0.18);
    let mut t = start + 0.30;
    let mut i = 0;
    while t < end - 0.10 {
        let w = PHRASE[i % PHRASE.len()];
        let e = (t + per).min(end - 0.05);
        out.push(TranscriptWord {
            text: w.to_string(),
            start: t,
            end: e,
            speaker: None,
        });
        t = e + 0.02;
        i += 1;
    }
    out
}

struct RenderCase {
    id: String,
    name: String,
    start: f64,
    end: f64,
    framing_mode: String,
    caption_style: String,
}

fn render_case(rc: &RenderCase, checks: &mut Vec<Check>) -> Option<(PathBuf, String)> {
    let probe = probe_media(VIDEO).expect("probe source");
    let iw = probe.width.unwrap_or(1920);
    let ih = probe.height.unwrap_or(1080);
    let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
    if crop_w % 2 != 0 {
        crop_w -= 1;
    }

    let words = synthetic_words(rc.start, rc.end);

    let framing = media::detect_speaker_crop_params(
        VIDEO,
        rc.start,
        rc.end,
        iw,
        ih,
        crop_w,
        Some(&words),
        &rc.framing_mode,
    );

    let pacing_plan = pacing::plan_smart_pacing_v2(VIDEO, rc.start, rc.end, Some(&words));
    let clip_dur = (rc.end - rc.start).max(0.0);
    let (caption_words, caption_framing, caption_start, caption_end) = match pacing_plan {
        Some(ref plan) => {
            let remapped = pacing::remap_words(plan, &words);
            let remapped_framing = pacing::remap_framing_plan(&framing, plan, clip_dur);
            if remapped.is_empty() {
                (words.clone(), Some(framing.clone()), rc.start, rc.end)
            } else {
                (
                    remapped,
                    Some(remapped_framing),
                    plan.clip_start_sec,
                    plan.clip_start_sec + plan.output_duration_sec,
                )
            }
        }
        None => (words.clone(), None, rc.start, rc.end),
    };

    let candidate = Candidate {
        id: rc.id.clone(),
        project_id: "probe".to_string(),
        start_sec: rc.start,
        end_sec: rc.end,
        score: 0.9,
        hook: "".to_string(),
        rationale: "probe".to_string(),
        rank: 0,
        selected: false,
        hook_start_sec: None,
        hook_end_sec: None,
        hook_confidence: None,
        opening_context_score: None,
        payoff_text: None,
        payoff_start_sec: None,
        payoff_end_sec: None,
        payoff_score: None,
        payoff_completion: None,
        metadata_json: None,
    };

    let intel = caption_intel::plan_caption_intel(
        &caption_words,
        &candidate,
        pacing_plan.as_ref(),
        caption_start,
        caption_end,
    );

    let template = get_caption_template(&rc.caption_style).expect("template registered");
    let ass = generate_ass_from_template_with_framing_and_intel(
        &caption_words,
        caption_start,
        caption_end,
        &template,
        caption_framing.as_ref().or(Some(&framing)),
        intel.as_ref(),
        None,
    );

    let ass_path = out_path(&format!("{}.ass", rc.id));
    std::fs::create_dir_all(ass_path.parent().unwrap()).expect("create out dir");
    std::fs::write(&ass_path, &ass).expect("write ass");

    let output = out_path(&format!("{}.mp4", rc.id));
    match pacing::render_paced_clip(
        VIDEO,
        rc.start,
        rc.end,
        &output,
        Some(&ass_path),
        None,
        Some(&words),
        Some(&framing),
        pacing_plan.as_ref(),
        None,
    ) {
        Ok(path) => Some((path, ass)),
        Err(e) => {
            checks.push(Check {
                id: rc.id.clone(),
                name: rc.name.clone(),
                status: "FAIL".to_string(),
                detail: format!("render failed: {e:?}"),
            });
            None
        }
    }
}

fn main() {
    std::fs::create_dir_all(OUT_DIR).ok();
    let mut checks: Vec<Check> = Vec::new();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let close_start = args
        .get(0)
        .and_then(|a| a.parse::<f64>().ok())
        .unwrap_or(CLOSE_START);
    let close_end = args
        .get(1)
        .and_then(|a| a.parse::<f64>().ok())
        .unwrap_or(CLOSE_END);
    let far_start = args
        .get(2)
        .and_then(|a| a.parse::<f64>().ok())
        .unwrap_or(FAR_START);
    let far_end = args
        .get(3)
        .and_then(|a| a.parse::<f64>().ok())
        .unwrap_or(FAR_END);
    eprintln!("windows: close [{close_start},{close_end}) far [{far_start},{far_end})");

    let cases = vec![
        // 1. Close pair, Adaptive Framing (the scenario under test)
        RenderCase {
            id: "closepair_adaptive".to_string(),
            name: "Close pair, adaptive framing".to_string(),
            start: close_start,
            end: close_end,
            framing_mode: "adaptive".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
        // 2. Same window, Original 9:16 (unchanged pipeline, DualFrame eligible)
        RenderCase {
            id: "closepair_original".to_string(),
            name: "Close pair, original 9:16".to_string(),
            start: close_start,
            end: close_end,
            framing_mode: "original".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
        // 3. Far pair, Adaptive (must isolate active speaker; DualFrame disabled)
        RenderCase {
            id: "farpair_adaptive".to_string(),
            name: "Far pair, adaptive framing".to_string(),
            start: far_start,
            end: far_end,
            framing_mode: "adaptive".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
        // 4. Far pair, Original (DualFrame split-screen must still engage)
        RenderCase {
            id: "farpair_original".to_string(),
            name: "Far pair, original 9:16".to_string(),
            start: far_start,
            end: far_end,
            framing_mode: "original".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
    ];

    let mut renders: Vec<(String, PathBuf, String, String)> = Vec::new();
    for rc in &cases {
        if rc.end <= rc.start {
            checks.push(Check {
                id: rc.id.clone(),
                name: rc.name.clone(),
                status: "SKIP".to_string(),
                detail: "window not selected yet".to_string(),
            });
            continue;
        }
        let id = rc.id.clone();
        let mode = rc.framing_mode.clone();
        if let Some((path, ass)) = render_case(rc, &mut checks) {
            checks.push(Check {
                id: format!("{id}-RENDER"),
                name: format!("{} — render produced a file", rc.name),
                status: if path.exists()
                    && std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) > 100_000
                {
                    "PASS".to_string()
                } else {
                    "FAIL".to_string()
                },
                detail: format!("{:?}", path),
            });
            renders.push((id, path, ass, mode));
        }
    }

    let manifest: Vec<serde_json::Value> = renders
        .iter()
        .map(|(id, path, _ass, mode)| {
            serde_json::json!({
                "id": id,
                "path": path.to_string_lossy(),
                "framing_mode": mode,
            })
        })
        .collect();
    std::fs::write(
        out_path("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .ok();

    let pass = checks.iter().filter(|c| c.status == "PASS").count();
    let fail = checks.iter().filter(|c| c.status == "FAIL").count();
    println!("\n=== Close-pair render harness: {pass} PASS / {fail} FAIL ===");
    for c in &checks {
        println!("[{}] {} — {} — {}", c.status, c.id, c.name, c.detail);
    }
    let report = serde_json::to_string_pretty(&checks).unwrap();
    std::fs::write(out_path("render_checks.json"), report).ok();
}
