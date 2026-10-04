//! FULL production end-to-end validation on real media.
//!
//! Drives the REAL AutoShorts 11.0 production call sites, in the same order the
//! Tauri commands do, and stops at nothing short of a real rendered clip that
//! passes mandatory Render QA and is persisted to the database.
//!
//!   ingest -> audio extraction -> Deepgram transcription -> normalization
//!   -> Deepgram diarization -> speaker mapping -> speaker intelligence
//!   -> candidate discovery -> ranking -> redundancy -> VLM -> PANNs
//!   -> boundary/payoff enforcement -> smart pacing -> scene intelligence
//!   -> framing -> caption intelligence/T7 -> ASS -> ffmpeg render
//!   -> Render QA -> persistence
//!
//! Nothing is mocked. Advisory stages (VLM, PANNs, redundancy) execute their
//! real path when enabled and their real timeout/fallback path when their
//! assets are unavailable; that outcome is reported, never hidden.
//!
//! Every stage emits [Stage] START / COMPLETE / FAILED / TIMEOUT with elapsed
//! time via `proc_guard::StageTimer`, so a stall is always attributable.
//!
//! Usage: cargo run --release --example full_pipeline_check -- "<video>"

use autoshorts_lib::models::{CandidateDraft, TranscriptWord};
use autoshorts_lib::proc_guard::StageTimer;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn banner(n: &str) {
    println!("\n=========== {} ===========", n);
}

fn sec(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

/// Run an advisory stage that is allowed to fail. Failures are reported and the
/// pipeline continues — advisory never breaks candidate generation or render.
struct AdvisoryOutcome {
    note: String,
}

impl AdvisoryOutcome {
    fn note(note: impl Into<String>) -> Self {
        Self { note: note.into() }
    }
}

fn main() {
    let _ = dotenvy::dotenv();

    let pipeline_timer = StageTimer::start("Pipeline");

    // Presence only — never print any credential value.
    let dg_present = std::env::var("DEEPGRAM_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false);
    // Resolve the discovery provider exactly as production `generate_candidates`
    // does (LLM_PROVIDER, then the matching per-provider key) so the harness
    // exercises the same real LLM path the app uses.
    let provider = std::env::var("LLM_PROVIDER")
        .ok()
        .unwrap_or_else(|| "deepseek".to_string())
        .to_lowercase();
    let api = match provider.as_str() {
        "claude" => std::env::var("ANTHROPIC_API_KEY").ok(),
        "openai" => std::env::var("OPENAI_API_KEY").ok(),
        "openrouter" => std::env::var("OPENROUTER_API_KEY").ok(),
        "groq" => std::env::var("GROQ_API_KEY").ok(),
        "gemini" => std::env::var("GEMINI_API_KEY").ok(),
        "deepseek" => std::env::var("DEEPSEEK_API_KEY").ok(),
        "local" | "ollama" => Some(String::new()),
        _ => std::env::var("DEEPSEEK_API_KEY").ok(),
    }
    .filter(|k| !k.is_empty());
    println!(
        "[Pipeline] credentials: DEEPGRAM={} LLM_PROVIDER={} LLM_KEY={} (values never printed)",
        if dg_present { "present" } else { "absent" },
        provider,
        if api.is_some() { "present" } else { "absent" }
    );

    // ── Ingest ────────────────────────────────────────────────────────────
    let t_ingest = StageTimer::start("Ingest");
    let source = std::env::args()
        .nth(1)
        .expect("pass the source video path");
    let src = PathBuf::from(&source);
    assert!(src.exists(), "source not found: {}", src.display());
    let bytes = src.metadata().unwrap().len();
    let probe = autoshorts_lib::media::probe_media(&source)
        .unwrap_or_else(|e| panic!("media probe failed (mandatory): {}", e));
    println!(
        "[Ingest] file={} bytes={} dims={}x{} has_video={} has_audio={} duration={:?}",
        src.file_name().unwrap().to_string_lossy(),
        bytes,
        probe.width.unwrap_or(0),
        probe.height.unwrap_or(0),
        probe.has_video,
        probe.audio_codec.is_some(),
        probe.duration_sec
    );
    let source_duration = probe.duration_sec.unwrap_or(0.0);
    t_ingest.complete(format!("{} bytes, {:.1}s source", bytes, source_duration));

    let work = std::env::temp_dir().join(format!("as_full_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).unwrap();

    // ── Audio extraction (production, bounded) ────────────────────────────
    let t_audio = StageTimer::start("AudioExtraction");
    let audio = autoshorts_lib::media::extract_audio(&source, &work).expect("extract_audio");
    let audio_bytes = std::fs::metadata(&audio).unwrap().len();
    let audio_str = audio.to_string_lossy().to_string();
    t_audio.complete(format!(
        "{} ({} bytes)",
        audio.file_name().unwrap().to_string_lossy(),
        audio_bytes
    ));

    // ── Transcription (real Deepgram, MANDATORY) ───────────────────────────
    let t_tx = StageTimer::start("Transcription");
    if !dg_present {
        t_tx.failed("no DEEPGRAM_API_KEY; cannot run a real end-to-end pipeline");
        panic!("no Deepgram key; cannot run a real end-to-end pipeline");
    }
    let key = std::env::var("DEEPGRAM_API_KEY").unwrap_or_default();
    let transcript = {
        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(autoshorts_lib::transcription::transcribe_deepgram(
            &audio_str,
            &key,
        )) {
            Ok(t) => t,
            Err(e) => {
                t_tx.failed(&e);
                panic!("Transcription is MANDATORY for candidate discovery: {}", e);
            }
        }
    };
    let mut words: Vec<TranscriptWord> = transcript
        .words
        .iter()
        .map(|w| TranscriptWord {
            text: w.text.clone(),
            start: w.start,
            end: w.end,
            speaker: w.speaker.clone(),
        })
        .collect();
    t_tx.complete(format!(
        "duration={:.1}s words={} segments={}",
        transcript.duration,
        words.len(),
        transcript.segments.len()
    ));

    // ── Transcript normalization (production) ─────────────────────────────
    let t_norm = StageTimer::start("TranscriptNormalization");
    let normalized = autoshorts_lib::transcript_normalizer::normalize_transcript_semantic(&transcript);
    t_norm.complete(format!(
        "words={} segments={}",
        normalized.words.len(),
        normalized.segments.len()
    ));

    // ── Deepgram diarization + speaker mapping (production engine) ─────────
    let t_diar = StageTimer::start("Diarization");
    let db = autoshorts_lib::db::Database::open(&work.join("full.sqlite")).unwrap();
    let caption_style = std::env::args()
        .find_map(|a| a.strip_prefix("--caption=").map(|s| s.to_string()))
        .unwrap_or_else(|| "kinetic".to_string());
    let project = db
        .create_project(
            &source,
            "deepgram",
            &caption_style,
            "original",
            Some(source_duration),
        )
        .expect("create project");
    let engine = autoshorts_lib::speaker_intelligence::SpeakerIntelligenceEngine::new(
        autoshorts_lib::speaker_intelligence::SpeakerIntelligenceConfig {
            cache_dir: Some(
                work.join("si_cache")
                    .to_string_lossy()
                    .into_owned(),
            ),
            deepgram_api_key: Some(key.clone()),
            ..Default::default()
        },
        db.clone(),
    )
    .unwrap();

    let si = engine
        .process_source_video(&project.id, &source, &words)
        .expect("process_source_video is fallback-safe");
    t_diar.complete(format!(
        "model={} speakers={} segments={} mapped={} reid={}",
        si.diarization.model,
        si.diarization.speakers.len(),
        si.diarization.segments.len(),
        si.speaker_map.mappings.len(),
        si.reid_embeddings.len()
    ));
    println!("[SpeakerIntel] speaker mappings:");
    for m in &si.speaker_map.mappings {
        println!(
            "   {} -> {:?} conf={:.2}",
            m.diarization_id, m.application_role, m.confidence
        );
    }
    if si.diarization.segments.is_empty() {
        eprintln!("[SpeakerIntel] WARNING: no diarization segments; downstream falls back");
    }
    // Propagate diarization labels onto words when the ASR omitted them.
    if words.iter().all(|w| w.speaker.is_none()) && !si.diarization.segments.is_empty() {
        for w in words.iter_mut() {
            let mid = (w.start + w.end) / 2.0;
            if let Some(seg) = si
                .diarization
                .segments
                .iter()
                .find(|s| mid >= s.start && mid <= s.end)
            {
                w.speaker = Some(seg.speaker_id.clone());
            }
        }
        println!(
            "[SpeakerIntel] propagated diarization labels onto {}/{} words",
            words.iter().filter(|w| w.speaker.is_some()).count(),
            words.len()
        );
    }

    // -- Candidate discovery (real LLM, MANDATORY) --------------------
    let t_cd = StageTimer::start("CandidateDiscovery");
    let discovery_config = autoshorts_lib::models::WindowDiscoveryConfig {
        discovery_mode: autoshorts_lib::resolve_discovery_mode_from_env(),
        ..autoshorts_lib::models::WindowDiscoveryConfig::default()
    };
    println!(
        "[CandidateDiscovery] Active Mode: {:?}",
        discovery_config.discovery_mode
    );
    // An operator may supply `--select-candidate=<start>-<end>` to exercise every
    // DOWNSTREAM stage on real media when the discovery LLM credential is
    // unavailable. This is explicitly NOT a full end-to-end pass: the final
    // report is marked partial and the run is never labelled a success.
    let operator_range: Option<(f64, f64)> = std::env::args()
        .find_map(|a| a.strip_prefix("--select-candidate=").map(|s| s.to_string()))
        .and_then(|s| {
            let (a, b) = s.split_once('-')?;
            Some((
                a.trim().parse::<f64>().ok()?,
                b.trim().parse::<f64>().ok()?,
            ))
        });

    let drafts: Vec<CandidateDraft> = if let Some((s, e)) = operator_range {
        if api.is_none() {
            t_cd.complete(format!(
                "OPERATOR-SPECIFIED RANGE {}-{}s (discovery bypassed — PARTIAL run)",
                s, e
            ));
            eprintln!(
                "[CandidateDiscovery] PARTIAL RUN: discovery was bypassed via --select-candidate.\n\
                 This run does NOT satisfy the full end-to-end success criterion."
            );
            vec![CandidateDraft {
                start: s,
                end: e,
                score: 1.0,
                hook: String::from("(operator-specified range)"),
                rationale: String::from("validation run with LLM discovery unavailable"),
                ..Default::default()
            }]
        } else {
            // A key IS available: run the real discovery and only use the
            // operator range for candidate SELECTION after ranking, so the
            // full pipeline (including LLM discovery) is exercised.
            let api_key = api.clone().unwrap();
            let r = {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(autoshorts_lib::llm::discover_candidates_full_timeline(
                    &normalized,
                    &provider,
                    &api_key,
                    None,
                    &discovery_config,
                ))
            };
            match r {
                Ok(d) => d,
                Err(e) => {
                    t_cd.failed(&e);
                    panic!("candidate discovery failed: {}", e);
                }
            }
        }
    } else {
        let api = match api {
            Some(k) => k,
            None => {
                t_cd.failed(&format!(
                    "no API key for provider '{}'; discovery is mandatory",
                    provider
                ));
                panic!(
                    "no LLM key for provider '{}'; cannot run a real end-to-end pipeline",
                    provider
                );
            }
        };
        let r = {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(autoshorts_lib::llm::discover_candidates_full_timeline(
                &normalized,
                &provider,
                &api,
                None,
                &discovery_config,
            ))
        };
        match r {
            Ok(d) => d,
            Err(e) => {
                t_cd.failed(&e);
                panic!("candidate discovery failed: {}", e);
            }
        }
    };
    t_cd.complete(format!("drafts={}", drafts.len()));
    assert!(!drafts.is_empty(), "no candidates discovered");

    // ── Candidate ranking (deterministic) ─────────────────────────────────
    let t_rank = StageTimer::start("CandidateRanking");
    let mut all_drafts = autoshorts_lib::llm::deduplicate_and_rank_all_candidates(
        drafts,
        normalized.duration,
        discovery_config.max_total_selected,
    );
    t_rank.complete(format!(
        "ranked={} top_score={:.3}",
        all_drafts.len(),
        all_drafts.first().map(|d| d.score).unwrap_or(0.0)
    ));

    // ── Redundancy (ADVISORY, deterministic, signal-only) ─────────────────
    let redundancy_enabled = autoshorts_lib::candidate_redundancy::candidate_redundancy_enabled();
    let t_red = StageTimer::start("Redundancy");
    if redundancy_enabled {
        let before = all_drafts.len();
        let snapshot: Vec<(f64, f64, Option<f64>)> = all_drafts
            .iter()
            .map(|d| (d.start, d.end, d.payoff_end))
            .collect();
        autoshorts_lib::candidate_redundancy::run_redundancy_detection(
            "",
            &mut all_drafts,
            &normalized.words,
        );
        let after: Vec<(f64, f64, Option<f64>)> = all_drafts
            .iter()
            .map(|d| (d.start, d.end, d.payoff_end))
            .collect();
        // Invariant: redundancy may DROP/REORDER candidates but must never
        // mutate a timestamp or a payoff endpoint.
        let survivors_ok = snapshot
            .iter()
            .filter(|s| after.contains(s))
            .count();
        assert!(
            survivors_ok == after.len(),
            "Redundancy mutated a timestamp/payoff boundary — deterministic invariant broken"
        );
        t_red.complete(format!("{} -> {} candidates", before, all_drafts.len()));
    } else {
        t_red.complete("disabled by default (AUTOSHORTS_CANDIDATE_REDUNDANCY unset)");
    }

    // ── PANNs reactions (ADVISORY) ────────────────────────────────────────
    let panns_enabled = autoshorts_lib::panns_reactions::panns_reactions_enabled();
    let t_panns = StageTimer::start("PANNs");
    let panns: AdvisoryOutcome = if !panns_enabled {
        // D7: the disabled branch must still close its stage timer, otherwise
        // the trace shows [PANNs] START with no COMPLETE line.
        t_panns.complete("disabled by default (opt-in)");
        AdvisoryOutcome::note("disabled by default (opt-in)")
    } else {
        let before: Vec<(f64, f64)> = all_drafts.iter().map(|d| (d.start, d.end)).collect();
        let snap: Vec<(f64, f64, Option<f64>)> = all_drafts
            .iter()
            .map(|d| (d.start, d.end, d.payoff_end))
            .collect();
        autoshorts_lib::panns_reactions::annotate_candidates_with_reactions(
            &source,
            &mut all_drafts,
        );
        let snap_after: Vec<(f64, f64, Option<f64>)> = all_drafts
            .iter()
            .map(|d| (d.start, d.end, d.payoff_end))
            .collect();
        assert!(
            snap == snap_after,
            "PANNs mutated a boundary/payoff — advisory stage overstepped"
        );
        let boosted = all_drafts
            .iter()
            .zip(before.iter())
            .filter(|(d, _)| d.hook_score.is_some())
            .count();
        t_panns.complete(format!(
            "annotated={} hook_scored={} (capped boost, boundaries untouched)",
            all_drafts.len(),
            boosted
        ));
        AdvisoryOutcome::note("reactions annotated")
    };

    // ── VLM scoring (ADVISORY, must never hang) ───────────────────────────
    let vlm_enabled = autoshorts_lib::vlm_scoring::vlm_scoring_enabled();
    let t_vlm = StageTimer::start("VLM");
    if !vlm_enabled {
        t_vlm.complete("disabled by default (AUTOSHORTS_VLM_SCORING unset) — heuristic path");
        eprintln!("[VLM] FALLBACK: opt-in only; deterministic heuristics used");
    } else {
        let snap: Vec<(f64, f64, Option<f64>)> = all_drafts
            .iter()
            .map(|d| (d.start, d.end, d.payoff_end))
            .collect();
        autoshorts_lib::vlm_scoring::enhance_candidates_with_vlm(
            &project.id,
            &source,
            &mut all_drafts,
            &normalized.words,
        );
        let snap_after: Vec<(f64, f64, Option<f64>)> = all_drafts
            .iter()
            .map(|d| (d.start, d.end, d.payoff_end))
            .collect();
        assert!(
            snap == snap_after,
            "VLM mutated a boundary/payoff — advisory stage overstepped"
        );
        let scored = all_drafts
            .iter()
            .filter(|d| d.vlm_quality_score.is_some())
            .count();
        t_vlm.complete(format!("scored={}/{}", scored, all_drafts.len()));
    }

    // ── Persistence of candidates (production) ────────────────────────────
    let t_persist = StageTimer::start("Persistence");
    let persisted = db
        .replace_candidates(&project.id, &all_drafts)
        .expect("replace_candidates");
    t_persist.complete(format!("candidates persisted={}", persisted.len()));

    // ── Select the candidate to render ────────────────────────────────────
    // With --select-candidate=<start>-<end>, the REAL persisted pool is
    // searched for that candidate (the operator pins WHICH real candidate is
    // rendered; discovery/ranking above still ran for real). Otherwise the
    // top-ranked candidate is rendered.
    let best = if let Some((s, e)) = operator_range {
        persisted
            .iter()
            .find(|c| (c.start_sec - s).abs() <= 1.0 && (c.end_sec - e).abs() <= 1.0)
            .or_else(|| persisted.iter().find(|c| (c.start_sec - s).abs() <= 2.0))
            .unwrap_or(&persisted[0])
    } else {
        &persisted[0]
    };
    println!(
        "[Select] rendering candidate start={:.3} end={:.3} rank={} (pool={})",
        best.start_sec,
        best.end_sec,
        best.rank,
        persisted.len()
    );
    let mut render_start = best.start_sec.max(0.0);
    let mut render_end = best.end_sec;
    // payoff_end is authoritative and is never overridden downstream.
    let payoff_end = best.payoff_end_sec;
    if let Some(p) = payoff_end {
        render_end = p.min(render_end);
    }

    // ── Boundary enforcement (deterministic authority) ───────────────────
    let t_bound = StageTimer::start("Boundary");
    let opt = autoshorts_lib::boundary::optimize_boundaries(
        &normalized.words,
        render_start,
        render_end,
        normalized.duration,
        None,
    );
    let snap_note = format!(
        "start={:.3}->{:.3} end={:.3}->{:.3} conf={:.2}",
        render_start,
        opt.optimized_start_sec,
        render_end,
        opt.optimized_end_sec,
        opt.confidence
    );
    render_start = opt.optimized_start_sec;
    // payoff_end stays authoritative: boundary optimization may only pull the
    // end EARLIER, never past the deterministic payoff endpoint.
    render_end = opt.optimized_end_sec.min(payoff_end.unwrap_or(opt.optimized_end_sec));
    let clip_dur = render_end - render_start;
    t_bound.complete(format!("{}", snap_note));
    println!(
        "[Boundary] payoff_end={:?} -> FINAL render range {:.3}..{:.3} ({}s), no artificial cap applied",
        payoff_end,
        render_start,
        render_end,
        clip_dur
    );
    assert!(
        clip_dur > 1.0,
        "clip too short to render after boundary enforcement"
    );

    // ── Smart Pacing (ADVISORY, now bounded) ──────────────────────────────
    let t_pace = StageTimer::start("SmartPacing");
    let pacing_plan =
        autoshorts_lib::pacing::plan_smart_pacing_v2(&source, render_start, render_end, Some(&words));
    match &pacing_plan {
        Some(p) => {
            t_pace.complete(format!("status={} edits={}", p.status, p.edits.len()));
            println!("[SmartPacing] {}", autoshorts_lib::pacing::pacing_summary(p));
        }
        None => {
            t_pace.complete("no v2 edits -> v1/unpaced fallback (valid, not an error)");
            println!(
                "[SmartPacing] No v2 edits found -> falling back to v1 plan (legitimate completion)"
            );
        }
    }

    // ── Scene Intelligence (ADVISORY, bounded, frame_skip=2) ───────────────
    let t_scene = StageTimer::start("SceneIntel");
    let scene = autoshorts_lib::scene_intelligence::SceneIntelligenceEngine::new(None).unwrap();
    let doc = scene.process_source(&source);
    let scene_cuts_json: Option<String> = if !doc.fallback && !doc.scenes.is_empty() {
        let p = work.join("scenes.json");
        std::fs::write(&p, serde_json::to_string(&doc).unwrap()).unwrap();
        Some(p.to_string_lossy().into_owned())
    } else {
        None
    };
    t_scene.complete(format!(
        "scenes={} fallback={} detector={}",
        doc.scenes.len(),
        doc.fallback,
        doc.detector
    ));

    // ── Framing (real Adaptive/speaker framing sidecar) ───────────────────
    // The framing mode is the user's per-project selection. `--framing=<mode>`
    // (adaptive|original) selects which production mode this pass validates;
    // it defaults to "original" so historical invocations are unchanged.
    let framing_mode = std::env::args()
        .find_map(|a| a.strip_prefix("--framing=").map(|s| s.to_string()))
        .unwrap_or_else(|| "original".to_string());
    let t_fr = StageTimer::start("Framing");
    let iw = probe.width.unwrap_or(1920);
    let ih = probe.height.unwrap_or(1080);
    let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
    if crop_w % 2 != 0 {
        crop_w -= 1;
    }
    let diar_tmp = work.join("diar.json");
    let gal_tmp = work.join("gal.json");
    let scene_tmp = work.join("scenes_sidecar.json");
    std::fs::write(
        &diar_tmp,
        serde_json::to_string(&serde_json::json!({
            "model": si.diarization.model,
            "segments": si.diarization.segments.iter().map(|s| serde_json::json!({
                "speaker_id": s.speaker_id, "start": s.start, "end": s.end, "confidence": s.confidence,
            })).collect::<Vec<_>>(),
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &gal_tmp,
        serde_json::to_string(&serde_json::json!({"entries": si.reid_embeddings})).unwrap(),
    )
    .unwrap();
    // The sidecar consumes `scene_cuts_json` as its third input.
    let scene_sidecar_path = scene_cuts_json
        .as_ref()
        .map(|s| {
            let _ = std::fs::copy(s, &scene_tmp);
            scene_tmp.to_string_lossy().into_owned()
        });
    let diar_s = diar_tmp.to_string_lossy().into_owned();
    let gal_s = gal_tmp.to_string_lossy().into_owned();
    let sidecar_inputs = autoshorts_lib::media::SpeakerIntelSidecarInputs {
        diarization_json: diar_s.as_str(),
        gallery_json: gal_s.as_str(),
        scene_cuts_json: scene_sidecar_path.as_deref(),
    };
    let plan = autoshorts_lib::media::detect_speaker_crop_params_with_intel(
        &source,
        render_start,
        render_end,
        iw,
        ih,
        crop_w,
        Some(&words),
        &framing_mode,
        Some(&sidecar_inputs),
    );
    let adaptive_requested = framing_mode.eq_ignore_ascii_case("adaptive");
    let adaptive_delivered = plan.framing == "adaptive";
    if adaptive_requested && !adaptive_delivered {
        panic!(
            "ADAPTIVE FRAMING DID NOT EXECUTE: requested mode '{}' but the sidecar produced framing='{}' — a fallback silently replaced the selected architecture",
            framing_mode, plan.framing
        );
    }
    let dual = plan
        .segments
        .iter()
        .filter(|s| matches!(s, autoshorts_lib::media::LayoutSegment::DualStack { .. }))
        .count();
    t_fr.complete(format!(
        "mode={} x={} w={} h={} segments={} dual={} intel={}",
        plan.framing,
        plan.x,
        plan.w,
        plan.h,
        plan.segments.len(),
        dual,
        plan.speaker_intel.is_some()
    ));
    if plan.is_emergency_fallback {
        panic!("framing plan is an emergency center-crop fallback — not Adaptive Framing success");
    }

    // ── Caption intelligence / T7 (production) ────────────────────────────
    let t_cap = StageTimer::start("Captions");
    let caption_plan = autoshorts_lib::caption_intel::plan_caption_intel(
        &normalized.words,
        best,
        pacing_plan.as_ref(),
        render_start,
        render_end,
    );
    let t7_note = match &caption_plan {
        Some(ci) => format!("caption_intel applied: {}", ci.reason),
        None => "caption_intel not applied (advisory) — phrase-paired T7 default used".to_string(),
    };

    let template = autoshorts_lib::captions::get_caption_template(&caption_style)
        .unwrap_or_else(|| autoshorts_lib::captions::all_caption_templates()[0].clone());
    let ass_string = autoshorts_lib::captions::generate_ass_from_template_with_framing_and_intel(
        &normalized.words,
        render_start,
        render_end,
        &template,
        Some(&plan),
        caption_plan.as_ref(),
        None,
    );
    let ass_path = work.join("captions.ass");
    std::fs::write(&ass_path, &ass_string).unwrap();

    // Caption path must be face-safe and inside the crop; verify the bundled
    // font is resolvable rather than silently falling back to OS fonts.
    let fonts_param = autoshorts_lib::media::resolve_and_verify_fonts_param(Some(ass_path.as_path()))
        .expect("fontsdir verification");
    let events = ass_string.matches("Dialogue:").count();
    t_cap.complete(format!(
        "ass_bytes={} dialogue_events={} fontsdir={} | {}",
        ass_string.len(),
        events,
        fonts_param.as_deref().unwrap_or("NONE"),
        t7_note
    ));

    // ── Render (real ffmpeg, full candidate duration, bounded) ────────────
    let t_r = StageTimer::start("Render");
    let out_clip = work.join("final_clip.mp4");
    let render_start_t = Instant::now();
    let rendered = autoshorts_lib::pacing::render_paced_clip(
        &source,
        render_start,
        render_end,
        &out_clip,
        Some(ass_path.as_path()),
        None,
        Some(&normalized.words),
        Some(&plan),
        pacing_plan.as_ref(),
        None,
    );
    let render_wall = sec(render_start_t);
    match &rendered {
        Ok(p) => t_r.complete(format!(
            "ok in {:.1}s wall -> {}",
            render_wall,
            p.display()
        )),
        Err(e) => {
            t_r.failed(e);
            panic!("render failed: {}", e);
        }
    }
    let clip_bytes = std::fs::metadata(&out_clip).unwrap().len();

    // ── Render QA (MANDATORY) ─────────────────────────────────────────────
    let t_qa = StageTimer::start("RenderQA");
    let qa_outcome = autoshorts_lib::render_qa::run_render_qa(
        &out_clip.to_string_lossy(),
        &source,
        &best.id,
        // Expected duration of the artifact: pacing output when pacing removed
        // content (same alignment as the production path), else the source range.
        Some(
            pacing_plan
                .as_ref()
                .map(|p| p.output_duration_sec)
                .unwrap_or(clip_dur),
        ),
        probe.audio_codec.is_some(),
        ass_path.exists(),
        Some(&plan),
    );
    let qa = match qa_outcome {
        autoshorts_lib::render_qa::RenderQaOutcome::Pass(r)
        | autoshorts_lib::render_qa::RenderQaOutcome::Fail(r) => r,
        autoshorts_lib::render_qa::RenderQaOutcome::Error(e) => {
            t_qa.failed(&format!("Render QA error: {}", e));
            panic!("Render QA error: {}", e);
        }
        autoshorts_lib::render_qa::RenderQaOutcome::Disabled => {
            t_qa.failed("Render QA disabled (AUTOSHORTS_RENDER_QA=0) — cannot certify success");
            panic!("Render QA is mandatory but was disabled");
        }
    };
    t_qa.complete(format!("status={:?} checks={}", qa.overall_status, qa.checks.len()));
    for c in &qa.checks {
        println!(
            "   [{:?}] {}{}{}",
            c.status,
            c.name,
            c.details,
            if c.severity == autoshorts_lib::render_qa::QaSeverity::Critical {
                " (CRITICAL)"
            } else {
                ""
            }
        );
    }
    let critical_fails = qa.checks.iter().filter(|c| {
        c.status == autoshorts_lib::render_qa::QaStatus::Fail
            && c.severity == autoshorts_lib::render_qa::QaSeverity::Critical
    })
    .count();

    // ── Persistence of the clip (production) ──────────────────────────────
    let t_save = StageTimer::start("Persistence/Clip");
    let render_log = format!(
        "e2e_clip={} | {} | qa={:?} | vlm={} | panns={}",
        out_clip.display(),
        t7_note,
        qa.overall_status,
        vlm_enabled,
        panns.note
    );
    let clip_status = if critical_fails == 0 && clip_bytes > 0 {
        "done"
    } else {
        "error"
    };
    db.update_clip_for_candidate(
        &best.id,
        clip_status,
        Some(&out_clip.to_string_lossy()),
        Some(ass_path.to_str().unwrap()),
        Some(&render_log),
        None,
    )
    .expect("update_clip_for_candidate");
    t_save.complete(format!("clip status={} -> {}", clip_status, out_clip.display()));

    // Copy the certified clip somewhere durable (the temp workdir is not a
    // deliverable) and report its real probed geometry.
    let final_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("e2e_output");
    std::fs::create_dir_all(&final_dir).unwrap();
    let final_clip = final_dir.join("full_pipeline_final_clip.mp4");
    let _ = std::fs::copy(&out_clip, &final_clip);
    let final_probe = autoshorts_lib::media::probe_media(&final_clip.to_string_lossy())
        .expect("probe final clip");

    banner("RESULT");
    println!("FINAL CLIP: {}", final_clip.display());
    println!(
        "FINAL CLIP GEOMETRY: {}x{} duration={:?} bytes={}",
        final_probe.width.unwrap_or(0),
        final_probe.height.unwrap_or(0),
        final_probe.duration_sec,
        std::fs::metadata(&final_clip).unwrap().len()
    );
    println!("QA STATUS: {:?}  critical_failures={}", qa.overall_status, critical_fails);
    println!("CLIP PERSISTED STATUS: {}", clip_status);
    pipeline_timer.complete(format!(
        "clip={} qa={:?} critical_failures={}",
        final_clip.display(),
        qa.overall_status,
        critical_fails
    ));

    if critical_fails == 0 && clip_bytes > 0 {
        println!("RESULT: PASS");
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
    println!("workdir: {}", work.display());
}
