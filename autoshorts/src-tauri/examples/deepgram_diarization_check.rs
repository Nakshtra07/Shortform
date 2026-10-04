//! Focused production-path validation harness for the Deepgram diarization fix.
//!
//! This is a VALIDATION TOOL, not a reimplementation. It calls the exact
//! production functions the app uses, in the same order:
//!
//!   media::extract_audio()      <- production audio extraction (media.rs)
//!   SpeakerIntelligenceEngine::process_source_video()
//!                               -> run_diarization() -> speaker_diarization.py
//!   SpeakerIntelligenceEngine::build_speaker_map()
//!
//! It deliberately does NOT run candidate generation, Smart Pacing, rendering,
//! captions, VLM, PANNs, Render QA, or any other pipeline stage.
//!
//! Usage:
//!   cargo run --release --example deepgram_diarization_check -- "<video path>"
//!
//! The Deepgram key is read from .env via dotenvy, exactly as the app does.
//! The key value is NEVER printed -- only its length and a presence flag.

use autoshorts_lib::models::TranscriptWord;
use autoshorts_lib::speaker_intelligence::{SpeakerIntelligenceConfig, SpeakerIntelligenceEngine};
use std::path::PathBuf;

fn main() {
    // 1. Credentials via the application's normal mechanism.
    let _ = dotenvy::dotenv();
    let key_present = std::env::var("DEEPGRAM_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false);
    let key_len = std::env::var("DEEPGRAM_API_KEY")
        .map(|k| k.len())
        .unwrap_or(0);
    println!("=== CREDENTIALS ===");
    println!(
        "DEEPGRAM_API_KEY loaded: {} (len={}, value never printed)",
        key_present, key_len
    );
    if !key_present {
        eprintln!("ABORT: no DEEPGRAM_API_KEY available");
        std::process::exit(2);
    }

    let source = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("ABORT: pass the source video path as argv[1]");
        std::process::exit(2)
    });
    let src = PathBuf::from(&source);
    if !src.exists() {
        eprintln!("ABORT: source not found");
        std::process::exit(2);
    }
    println!("source: {}", src.file_name().unwrap().to_string_lossy());
    println!("source bytes: {}", src.metadata().unwrap().len());

    // 2. PRODUCTION audio extraction (same function the app calls).
    println!("\n=== PRODUCTION AUDIO EXTRACTION (media::extract_audio) ===");
    let work = std::env::temp_dir().join(format!("as_si_val_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    let t0 = std::time::Instant::now();
    let audio =
        autoshorts_lib::media::extract_audio(&source, &work).expect("extract_audio must succeed");
    let extract_secs = t0.elapsed().as_secs_f64();
    let audio_bytes = std::fs::metadata(&audio).unwrap().len();
    let src_bytes = src.metadata().unwrap().len();
    println!("extracted: {}", audio.display());
    println!(
        "extracted format: {}",
        audio.extension().unwrap_or_default().to_string_lossy()
    );
    println!(
        "extracted bytes: {} ({:.2} MB)",
        audio_bytes,
        audio_bytes as f64 / 1e6
    );
    println!("extract elapsed: {:.1}s", extract_secs);
    println!(
        "payload reduction: {:.2} MB -> {:.2} MB ({:.1}x smaller)",
        src_bytes as f64 / 1e6,
        audio_bytes as f64 / 1e6,
        src_bytes as f64 / audio_bytes as f64
    );
    println!(
        "FULL VIDEO SENT TO DEEPGRAM? {}",
        if audio_bytes == src_bytes {
            "YES (BUG)"
        } else {
            "no"
        }
    );

    // 3. PRODUCTION diarization + speaker mapping, on a cache-bypassing engine.
    //    A throwaway cache_dir guarantees a real Deepgram request rather than a
    //    replay of a previously cached fallback.
    println!("\n=== PRODUCTION DIARIZATION (SpeakerIntelligenceEngine) ===");
    let cache_dir = work.join("si_cache_fresh");
    let _ = std::fs::create_dir_all(&cache_dir);

    let db = autoshorts_lib::db::Database::open(&work.join("si_val.sqlite")).expect("open db");
    let project = db
        .create_project(&source, "deepgram", "kinetic", "original", Some(90.0))
        .expect("create project");

    let engine = SpeakerIntelligenceEngine::new(
        SpeakerIntelligenceConfig {
            cache_dir: Some(cache_dir.to_string_lossy().into_owned()),
            // Use the key from .env, never a hardcoded value.
            deepgram_api_key: std::env::var("DEEPGRAM_API_KEY").ok(),
            ..Default::default()
        },
        db,
    )
    .expect("engine");

    // No transcript words: diarization is the stage under test, and an empty
    // word list only affects the question-evidence weighting, not the segments.
    let words: Vec<TranscriptWord> = Vec::new();

    let t1 = std::time::Instant::now();
    let si = engine
        .process_source_video(&project.id, &source, &words)
        .expect("process_source_video is fallback-safe and must not error");
    let diar_secs = t1.elapsed().as_secs_f64();

    let d = &si.diarization;
    println!("diarization elapsed: {:.1}s", diar_secs);
    println!("diarization model: {}", d.model);
    println!("diarization version: {}", d.version);
    println!("diarization confidence: {:.4}", d.confidence);
    println!("SPEAKERS: {}", d.speakers.len());
    println!("SEGMENTS: {}", d.segments.len());
    for s in &d.speakers {
        println!(
            "   speaker {} total_speech={:.1}s segments={} avg_conf={:.3}",
            s.diarization_id, s.total_speech_sec, s.segment_count, s.avg_confidence
        );
    }

    println!("\n=== SPEAKER MAPPING (build_speaker_map) ===");
    println!("MAPPED ENTRIES: {}", si.speaker_map.mappings.len());
    for m in &si.speaker_map.mappings {
        println!(
            "   {} -> {:?} conf={:.3} first={} asker={} override={}",
            m.diarization_id,
            m.application_role,
            m.confidence,
            m.evidence.first_speaker,
            m.evidence.question_asker,
            m.evidence.user_override
        );
    }

    // 4. Long-form timestamp validation, on the real result.
    println!("\n=== LONG-FORM TIMESTAMP VALIDATION ===");
    let mut monotonic = true;
    let mut overlaps = 0usize;
    let mut nonpos = 0usize;
    for w in d.segments.windows(2) {
        if w[1].start < w[0].start - 1e-6 {
            monotonic = false;
        }
        if w[1].start < w[0].end - 1e-6 {
            overlaps += 1;
        }
    }
    for s in &d.segments {
        if s.end <= s.start {
            nonpos += 1;
        }
    }
    let first = d.segments.first();
    let last = d.segments.last();
    println!(
        "first segment: {:?}",
        first.map(|s| (&s.speaker_id, s.start, s.end))
    );
    println!(
        "last  segment: {:?}",
        last.map(|s| (&s.speaker_id, s.start, s.end))
    );
    if let (Some(f), Some(l)) = (first, last) {
        println!("timeline span: {:.1}s .. {:.1}s", f.start, l.end);
    }
    println!("monotonic non-decreasing start: {}", monotonic);
    println!("overlapping segment pairs: {}", overlaps);
    println!("zero/negative-length segments: {}", nonpos);
    let distinct: std::collections::BTreeSet<_> =
        d.segments.iter().map(|s| s.speaker_id.clone()).collect();
    println!("distinct speaker labels: {:?}", distinct);

    // 5. Verdict.
    println!("\n=== VERDICT ===");
    let live = d.model == "deepgram" && !d.segments.is_empty() && !d.speakers.is_empty();
    println!("live diarization data: {}", live);
    println!(
        "speaker mapping non-zero: {}",
        !si.speaker_map.mappings.is_empty()
    );
    if live && !si.speaker_map.mappings.is_empty() && monotonic && nonpos == 0 {
        println!("RESULT: PASS");
    } else {
        println!("RESULT: INCONCLUSIVE/FAIL -- see counters above");
    }
    println!("workdir: {}", work.display());
}
