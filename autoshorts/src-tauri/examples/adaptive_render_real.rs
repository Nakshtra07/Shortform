//! Real Adaptive Framing render for candidate 74604721 (production call sites).
//!
//! Drives the EXACT functions the Tauri render command calls, in the same order:
//!   boundary optimize -> Adaptive Framing -> Smart Pacing -> captions/T7
//!   -> ASS -> audio plan -> render_paced_clip -> MANDATORY Render QA
//!
//! Nothing is mocked. The purpose is to prove Adaptive Framing completes and
//! the rendered artifact is built from the adaptive crop (NOT the emergency
//! center crop) and still passes mandatory Render QA.

use autoshorts_lib::boundary::{self, BoundaryCandidateMeta};
use autoshorts_lib::models::NormalizedTranscript;
use autoshorts_lib::proc_guard::StageTimer;
use std::path::PathBuf;

const WORDS_JSON: &str = r"D:\College\Autoshorts 11.0\tmp\real_candidate_words.json";

fn main() {
    let _ = dotenvy::dotenv();
    let t_all = StageTimer::start("AdaptiveRender");

    let source = std::env::args()
        .nth(1)
        .expect("pass source video path");
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| r"D:\College\Autoshorts 11.0\tmp\adaptive_real_clip.mp4".to_string());
    let source = source.clone();

    let words_json = std::fs::read_to_string(WORDS_JSON).expect("read words json");
    let transcript: NormalizedTranscript = serde_json::from_str(&words_json).expect("parse words");
    let words = transcript.words.clone();
    println!("[Real] transcript words = {}", words.len());

    let probe = autoshorts_lib::media::probe_media(&source).expect("probe");
    let iw = probe.width.unwrap_or(1920);
    let ih = probe.height.unwrap_or(1080);
    let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
    if crop_w % 2 != 0 {
        crop_w -= 1;
    }
    println!(
        "[Real] source {}x{} crop_w={} duration={:?}",
        iw, ih, crop_w, probe.duration_sec
    );

    // ── Boundary optimization (deterministic, authoritative) ────────────────
    let t_b = StageTimer::start("Boundary");
    let meta = BoundaryCandidateMeta {
        hook: Some("So the performances of Portugal in the tournament in Euros twenty four, like, what's your thoughts on that now you've had a bit of time to reflect?"),
        hook_start_sec: Some(897.485),
        hook_end_sec: Some(904.205),
        hook_confidence: Some(1.0),
        opening_context_score: Some(1.0),
        payoff_end_sec: Some(977.375),
    };
    let bopt = boundary::optimize_boundaries(
        &words,
        897.485,
        977.375,
        probe.duration_sec.unwrap_or(977.375),
        Some(&meta),
    );
    let render_start = bopt.optimized_start_sec;
    let render_end = bopt.optimized_end_sec;
    println!("[Real] boundary: {}", bopt.summary());
    t_b.complete(format!("{} -> {}", render_start, render_end));

    let clip_dur = (render_end - render_start).max(0.0);

    // ── Adaptive Framing (THE stage under repair) ────────────────────────────
    let t_f = StageTimer::start("Framing");
    let framing_mode = "adaptive";
    let framing = autoshorts_lib::media::detect_speaker_crop_params(
        &source,
        render_start,
        render_end,
        iw,
        ih,
        crop_w,
        Some(&words),
        framing_mode,
    );
    println!(
        "[Real] framing mode={} framing={} emergency_fallback={} reason={:?} segments={}",
        framing.mode,
        framing.framing,
        framing.is_emergency_fallback,
        framing.fallback_reason,
        framing.segments.len()
    );
    println!("[Real] crop x={} w={} h={}", framing.x, framing.w, framing.h);

    if framing.is_emergency_fallback {
        eprintln!("[Real] ADAPTIVE FRAMING DID NOT RUN — center crop fallback in use");
        t_f.failed("emergency fallback");
        std::process::exit(2);
    }
    let adaptive_ok = framing.framing.eq_ignore_ascii_case("adaptive");
    println!("[Real] ADAPTIVE FRAMING EXECUTED = {}", adaptive_ok);
    t_f.complete(format!(
        "adaptive={} segments={}",
        adaptive_ok,
        framing.segments.len()
    ));

    // ── Smart Pacing ────────────────────────────────────────────────────────
    let t_p = StageTimer::start("SmartPacing");
    let pacing_plan = autoshorts_lib::pacing::plan_smart_pacing_v2(
        &source,
        render_start,
        render_end,
        Some(&words),
    );
    let (caption_words, caption_framing, caption_start, caption_end) = match pacing_plan.as_ref() {
        Some(p) => {
            let rw = autoshorts_lib::pacing::remap_words(p, &words);
            let rf = autoshorts_lib::pacing::remap_framing_plan(&framing, p, clip_dur);
            (
                rw,
                Some(rf),
                p.clip_start_sec,
                p.clip_start_sec + p.output_duration_sec,
            )
        }
        None => (words.clone(), None, render_start, render_end),
    };
    t_p.complete(format!(
        "pacing={} caption_words={}",
        pacing_plan.is_some(),
        caption_words.len()
    ));

    // ── Audio Intelligence ──────────────────────────────────────────────────
    let t_a = StageTimer::start("AudioIntel");
    let audio_plan = autoshorts_lib::audio::plan_audio_intelligence(
        &source,
        render_start,
        render_end,
        Some(&words),
        pacing_plan.as_ref(),
    );
    t_a.complete(format!(
        "audio_plan={}",
        audio_plan
            .as_ref()
            .map(|a| a.filter_chain.clone())
            .unwrap_or_else(|| "none".into())
    ));

    // ── Captions / ASS (T7-aware) ───────────────────────────────────────────
    let t_c = StageTimer::start("Captions");
    let ass_dir = PathBuf::from(r"D:\College\Autoshorts 11.0\tmp");
    let _ = std::fs::create_dir_all(&ass_dir);
    let ass_path = ass_dir.join("adaptive_real_clip.ass");
    let style_id = "preset_viral_bold";
    let template = autoshorts_lib::captions::get_caption_template(style_id)
        .unwrap_or_else(|| autoshorts_lib::captions::get_caption_template("default").expect("caption template"));
    println!("[Real] caption template = {}", template.template_id);
    let ass_string: String =
        autoshorts_lib::captions::generate_ass_from_template_with_framing_and_intel(
            &caption_words,
            caption_start,
            caption_end,
            &template,
            caption_framing.as_ref().or(Some(&framing)),
            None,
            None,
        );
    std::fs::write(&ass_path, &ass_string).expect("write ASS");
    t_c.complete(format!("{} bytes -> {}", ass_string.len(), ass_path.display()));

    // ── Render ──────────────────────────────────────────────────────────────
    let t_r = StageTimer::start("Render");
    let out = PathBuf::from(&out_path);
    let rendered = autoshorts_lib::pacing::render_paced_clip(
        &source,
        render_start,
        render_end,
        &out,
        Some(ass_path.as_path()),
        None,
        Some(&caption_words),
        Some(&framing),
        pacing_plan.as_ref(),
        audio_plan.as_ref().map(|a| a.filter_chain.as_str()),
    )
    .expect("render_paced_clip");
    t_r.complete(format!("{}", rendered.display()));

    // ── MANDATORY Render QA ─────────────────────────────────────────────────
    let t_q = StageTimer::start("RenderQA");
    let out_str = rendered.to_string_lossy().to_string();
    let qa = autoshorts_lib::render_qa::run_render_qa(
        &out_str,
        &source,
        "74604721-d665-48d5-9f87-40e0422ee42d",
        Some(clip_dur),
        probe.audio_codec.is_some(),
        true,
        Some(&framing),
    )
    .expect("run_render_qa returned None (QA disabled?)");

    println!("[Render QA] {}", qa.summary());
    for c in &qa.checks {
        println!("   [{:?}/{:?}] {} — {}", c.status, c.severity, c.name, c.details);
    }
    let critical = qa.has_critical_failures();
    t_q.complete(format!(
        "critical_failures={} -> {}",
        critical,
        if critical { "REJECTED" } else { "PASS" }
    ));

    println!("\n===== RESULT =====");
    println!("adaptive_framing_executed = {}", adaptive_ok);
    println!("emergency_fallback        = {}", framing.is_emergency_fallback);
    println!("render_qa_critical_fail   = {}", critical);
    println!("output                    = {}", out_str);
    t_all.complete("done");
    std::process::exit(if !critical && adaptive_ok { 0 } else { 1 });
}