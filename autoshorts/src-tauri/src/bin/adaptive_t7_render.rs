// Adaptive Framing + T7 caption style — real-render verification harness.
//
// Mirrors the production render path in lib.rs::render_flat_clip_for_candidate:
//   detect_speaker_crop_params(framing_mode) -> caption intel -> ASS -> render_paced_clip
// Renders real MP4s from AutoShorts_ngvOyccUzzY.mp4 and verifies each test
// case by inspecting the encoded pixels and the generated ASS.

use autoshorts_lib::caption_intel;
use autoshorts_lib::captions::{
    self, generate_ass_from_template_with_framing_and_intel, get_caption_template,
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

const VIDEO: &str = r"D:\College\Autoshorts 10.0\AutoShorts_ngvOyccUzzY.mp4";
const OUT_DIR: &str = r"D:\College\Autoshorts 10.0\tmp\adaptive_t7_renders";

fn out_path(name: &str) -> PathBuf {
    Path::new(OUT_DIR).join(name)
}

/// Deterministic words spread across the SOURCE window [start, end] so that
/// the pacing remapper retains at least some of them (it filters by source
/// time). Includes long keyword-like words the caption-intel planner can
/// emphasize.
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

/// Full production-path render for one window with a given framing mode +
/// caption style. Returns (output path, ass content, checks).
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
    let iw = probe.width.unwrap_or(3840);
    let ih = probe.height.unwrap_or(2160);
    let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
    if crop_w % 2 != 0 {
        crop_w -= 1;
    }

    // Words on the SOURCE timeline (production passes source-timeline words;
    // the pacing sidecar remaps internally).
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

    // Sanity: T7 template registered and its constants match the spec.
    let t7 = get_caption_template("preset_t7").expect("preset_t7 registered");
    checks.push(Check {
        id: "REG-T7".to_string(),
        name: "T7 template registered with spec constants".to_string(),
        status: if t7.global_styles.font_family == "Matt_Trial-Bold"
            && t7.global_styles.font_size == 110
            && t7.global_styles.font_weight == "700"
            && t7.global_styles.font_color == "#FFFFFF"
            && t7.global_styles.background_box.is_none()
            && t7.global_styles.stroke_color.is_none()
            && t7.global_styles.shadow_color.is_none()
            && t7.global_styles.position_y == 0.60
            && matches!(t7.active_state, captions::CaptionActiveState::SpeechChunkedRolling { ref highlight_color, .. } if highlight_color == "#F9EF07")
        {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        detail: format!("{:?}", t7.global_styles),
    });

    let cases = vec![
        // 1. T7 caption style render — original framing
        RenderCase {
            id: "t7_original".to_string(),
            name: "T7 captions, original 9:16".to_string(),
            start: 490.0,
            end: 510.0,
            framing_mode: "original".to_string(),
            caption_style: "preset_t7".to_string(),
        },
        // 2. Adaptive framing, single-person window
        RenderCase {
            id: "adaptive_single".to_string(),
            name: "Adaptive framing, single person".to_string(),
            start: 300.0,
            end: 320.0,
            framing_mode: "adaptive".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
        // 3. Adaptive framing, two-person window (wide pair)
        RenderCase {
            id: "adaptive_two_wide".to_string(),
            name: "Adaptive framing, two people far apart".to_string(),
            start: 2040.0,
            end: 2060.0,
            framing_mode: "adaptive".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
        // 4. Original framing on the same two-person window (DualFrame eligible)
        RenderCase {
            id: "original_two_wide".to_string(),
            name: "Original 9:16 on two-person window".to_string(),
            start: 2040.0,
            end: 2060.0,
            framing_mode: "original".to_string(),
            caption_style: "preset_mrbeast_pop".to_string(),
        },
        // 5. T7 + adaptive combined
        RenderCase {
            id: "t7_adaptive".to_string(),
            name: "T7 captions + adaptive framing".to_string(),
            start: 490.0,
            end: 510.0,
            framing_mode: "adaptive".to_string(),
            caption_style: "preset_t7".to_string(),
        },
    ];

    let mut renders: Vec<(String, PathBuf, String, String)> = Vec::new();
    for rc in &cases {
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

    // Write the ASS of the T7 render for pixel-independent assertions.
    if let Some((_, _, ass, _)) = renders.iter().find(|(id, _, _, _)| id == "t7_original") {
        std::fs::write(out_path("t7_original.ass"), ass).expect("write t7 ass");
    }

    // Emit a manifest; the Python pixel-inspection step consumes it.
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
    println!("\n=== Adaptive/T7 render harness: {pass} PASS / {fail} FAIL ===");
    for c in &checks {
        println!("[{}] {} — {} — {}", c.status, c.id, c.name, c.detail);
    }
    let report = serde_json::to_string_pretty(&checks).unwrap();
    std::fs::write(out_path("render_checks.json"), report).ok();
}
