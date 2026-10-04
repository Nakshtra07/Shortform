// Real-media render tool for Candidate 4 using the exact production pipeline.

use autoshorts_lib::boundary::{self, BoundaryCandidateMeta};
use autoshorts_lib::caption_intel;
use autoshorts_lib::captions::{generate_ass_from_template_with_framing_and_intel, get_caption_template};
use autoshorts_lib::media::{self, probe_media};
use autoshorts_lib::models::{Candidate, NormalizedTranscript};
use autoshorts_lib::pacing;
use std::path::PathBuf;

const VIDEO: &str = r"D:\College\Autoshorts 10.0\My thoughts on my 2023_24 season and the truth about my future. Talk with my friend Rio (PART 1) - UR · Cristiano (1080p, h264).mp4";
const OUT_LOCAL: &str = r"D:\College\Autoshorts 10.0\clip-04_flat.mp4";
const OUT_DOCS: &str = r"C:\Users\naksh\OneDrive\Documents\AutoShorts\My-thoughts-on-my-2023_24-season-and-the-truth-about-my-future--Talk-with-my-friend-Rio--PART-1----UR---Cristiano--1080p--h264\clips\clip-04_flat.mp4";

fn main() {
    println!("=== AutoShorts Production Render for Candidate 4 ===");
    let words_path = r"D:\College\Autoshorts 10.0\tmp\cand4_words.json";
    let words_json = std::fs::read_to_string(words_path).expect("read words JSON");
    let transcript: NormalizedTranscript =
        serde_json::from_str(&words_json).expect("parse NormalizedTranscript");
    let words = transcript.words;

    let probe = probe_media(VIDEO).expect("probe media");
    let iw = probe.width.unwrap_or(1920);
    let ih = probe.height.unwrap_or(1080);
    let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
    if crop_w % 2 != 0 {
        crop_w -= 1;
    }

    let cand_raw_start = 286.10498;
    let cand_raw_end = 397.395;
    let payoff_end = 397.395;

    let meta = BoundaryCandidateMeta {
        hook: Some("How much pressure did you feel being the first real big star to come out and say and then say to everybody, come on, follow me. How was that big pressure for you?"),
        hook_start_sec: Some(286.10498),
        hook_end_sec: Some(294.85),
        hook_confidence: Some(1.0),
        opening_context_score: Some(1.0),
        payoff_end_sec: Some(payoff_end),
    };

    let boundary_opt = boundary::optimize_boundaries(
        &words,
        cand_raw_start,
        cand_raw_end,
        probe.duration_sec.unwrap_or(cand_raw_end),
        Some(&meta),
    );

    let render_start = boundary_opt.optimized_start_sec;
    let render_end = boundary_opt.optimized_end_sec;
    println!("[Boundary Optimization] {}", boundary_opt.summary());
    assert_eq!(
        render_end, payoff_end,
        "Authoritative payoff end must not change"
    );

    // 1. Framing in Adaptive mode
    println!("[Framing] Running detect_speaker_crop_params in adaptive mode...");
    let framing_mode = "adaptive";
    let framing = media::detect_speaker_crop_params(
        VIDEO,
        render_start,
        render_end,
        iw,
        ih,
        crop_w,
        Some(&words),
        framing_mode,
    );
    println!(
        "[Framing] Done. mode={}, framing={}",
        framing.mode, framing.framing
    );

    // 2. Smart Pacing
    println!(
        "[Smart Pacing] Planning smart pacing on [{}, {}]...",
        render_start, render_end
    );
    let pacing_plan = pacing::plan_smart_pacing_v2(VIDEO, render_start, render_end, Some(&words));
    let clip_dur = (render_end - render_start).max(0.0);
    let (caption_words, caption_framing, caption_start, caption_end) = match pacing_plan {
        Some(ref plan) => {
            let remapped = pacing::remap_words(plan, &words);
            let remapped_framing = pacing::remap_framing_plan(&framing, plan, clip_dur);
            println!(
                "[Smart Pacing] Plan produced: {}",
                pacing::pacing_summary(plan)
            );
            if remapped.is_empty() {
                (
                    words.clone(),
                    Some(framing.clone()),
                    render_start,
                    render_end,
                )
            } else {
                (
                    remapped,
                    Some(remapped_framing),
                    plan.clip_start_sec,
                    plan.clip_start_sec + plan.output_duration_sec,
                )
            }
        }
        None => {
            println!("[Smart Pacing] No pacing edits — using original range");
            (words.clone(), None, render_start, render_end)
        }
    };

    // 3. Audio Intelligence
    println!("[Audio Intelligence] Analyzing audio...");
    let audio_plan = autoshorts_lib::audio::plan_audio_intelligence(
        VIDEO,
        render_start,
        render_end,
        Some(&words),
        pacing_plan.as_ref(),
    );
    if let Some(ref a) = audio_plan {
        println!("[Audio Intelligence] Filters: {}", a.filter_chain);
    } else {
        println!("[Audio Intelligence] Audio within tolerance / untouched");
    }

    // 4. Captions (preset_t7)
    let candidate = Candidate {
        id: "90238d8f-71dd-4df0-9f9e-bfde1a84b3fa".to_string(),
        project_id: "8313ab72-9405-45ef-99c1-6253f0fefb1d".to_string(),
        start_sec: cand_raw_start,
        end_sec: cand_raw_end,
        score: 0.832,
        hook: meta.hook.unwrap().to_string(),
        rationale: "Candidate 4 with full payoff".to_string(),
        rank: 4,
        selected: true,
        hook_start_sec: meta.hook_start_sec,
        hook_end_sec: meta.hook_end_sec,
        hook_confidence: meta.hook_confidence,
        opening_context_score: meta.opening_context_score,
        payoff_text: Some(
            "But I will expect that the football will change because of me.".to_string(),
        ),
        payoff_start_sec: Some(393.475),
        payoff_end_sec: Some(397.395),
        payoff_score: Some(1.0),
        payoff_completion: Some(true),
        metadata_json: None,
    };

    let intel = caption_intel::plan_caption_intel(
        &caption_words,
        &candidate,
        pacing_plan.as_ref(),
        caption_start,
        caption_end,
    );
    let template = get_caption_template("preset_t7").expect("template preset_t7 registered");
    let ass = generate_ass_from_template_with_framing_and_intel(
        &caption_words,
        caption_start,
        caption_end,
        &template,
        caption_framing.as_ref().or(Some(&framing)),
        intel.as_ref(),
        None,
    );

    let ass_path = PathBuf::from(r"D:\College\Autoshorts 10.0\tmp\clip-04.ass");
    std::fs::write(&ass_path, &ass).expect("write ass file");
    println!(
        "[Captions] Generated ASS file with {} lines",
        ass.lines().count()
    );

    // 5. Render to OUT_LOCAL
    println!("[FFmpeg Render] Rendering to {}...", OUT_LOCAL);
    let out_local_path = PathBuf::from(OUT_LOCAL);
    let render_res = pacing::render_paced_clip(
        VIDEO,
        render_start,
        render_end,
        &out_local_path,
        Some(&ass_path),
        None,
        Some(&words),
        Some(&framing),
        pacing_plan.as_ref(),
        audio_plan.as_ref().map(|p| p.filter_chain.as_str()),
    );

    match render_res {
        Ok(p) => {
            println!("[FFmpeg Render] Success! Rendered to {}", p.display());
            // Copy to docs folder as well
            let docs_path = PathBuf::from(OUT_DOCS);
            if let Some(parent) = docs_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            if let Err(e) = std::fs::copy(&p, &docs_path) {
                eprintln!("[Warning] Could not copy to docs path: {}", e);
            } else {
                println!(
                    "[FFmpeg Render] Successfully copied to {}",
                    docs_path.display()
                );
            }
        }
        Err(e) => {
            eprintln!("[FFmpeg Render] ERROR: {:?}", e);
            std::process::exit(1);
        }
    }
}
