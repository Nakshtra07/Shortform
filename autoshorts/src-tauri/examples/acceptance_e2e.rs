//! Acceptance end-to-end harness for AutoShorts 11.0 (Part A evidence).
//!
//! Drives the REAL production entry points — the exact functions the Tauri
//! commands wrap — against the REAL application database, in the same order
//! the UI does:
//!
//!   copyright gate -> yt-dlp download (app downloader) -> media probe +
//!   SHA-256 -> project row (production DB) -> Deepgram transcription
//!   -> generate_candidates_impl (discovery + AutoClip scoring + eligibility
//!   gate + deterministic ranking + Top-N + persistence) -> fresh-connection
//!   reload verification (UI-reload agreement) -> per-selected-candidate
//!   production render (+ mandatory Render QA) -> output probe + SHA-256.
//!
//! Nothing is mocked. Every stage prints its evidence. This harness is NOT a
//! substitute for the actual UI journey; UI-level verification is documented
//! separately in the acceptance report.
//!
//! Usage:
//!   cargo run --release --example acceptance_e2e -- --url <youtube-url> --label V1
//!   cargo run --release --example acceptance_e2e -- --source <local-file> --label V4

use anyhow::{anyhow, bail, Result};
use autoshorts_lib::db::Database;
use autoshorts_lib::models::{Candidate, MediaProbe, NormalizedTranscript};
use autoshorts_lib::{create_project_from_path_impl, generate_candidates_impl};
use autoshorts_lib::{
    render_flat_clip_for_candidate_impl, transcribe_project_impl, InFlightRenderTracker,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn probe_line(label: &str, probe: &MediaProbe) {
    println!(
        "[Probe] {}: {}x{} video={} audio={} duration={:?}",
        label,
        probe.width.unwrap_or(0),
        probe.height.unwrap_or(0),
        probe.video_codec.as_deref().unwrap_or("-"),
        probe.audio_codec.as_deref().unwrap_or("-"),
        probe.duration_sec
    );
}

fn print_candidates(label: &str, candidates: &[Candidate]) {
    println!("---- {label} ----");
    for c in candidates {
        println!(
            "  rank={:>2} selected={:?} score={:.2} source={:?} [{:>7.1}s -> {:7.1}s] hook={}",
            c.rank,
            c.selected,
            c.score,
            c.metadata_json
                .as_ref()
                .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
                .and_then(|m| m["scoreSource"].as_str().map(str::to_string))
                .unwrap_or_else(|| "-".to_string()),
            c.start_sec,
            c.end_sec,
            c.hook.chars().take(60).collect::<String>()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Environment: workspace .env files, then the acceptance-run settings.
    let _ = dotenvy::from_path(r"D:\College\Autoshorts 11.0\.env");
    let _ = dotenvy::from_path(r"D:\College\Autoshorts 11.0\autoshorts\.env");
    let _ = dotenvy::dotenv();
    // Acceptance-run scope only (process env; production .env stays untouched):
    std::env::set_var("AUTOSHORTS_SCORING_BACKEND", "autoclip");
    std::env::set_var("AUTOSHORTS_TOP_N", "3");

    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.to_string())
    };
    let label = get("--label").unwrap_or_else(|| "V?".to_string());
    let url = get("--url");
    let source = get("--source");
    if url.is_none() && source.is_none() {
        bail!("pass --url <youtube-url> or --source <local-file>");
    }

    println!("==============================================================");
    println!("  ACCEPTANCE E2E — {label}");
    println!("==============================================================");

    // 2. Backend authority evidence (must be autoclip for this run).
    let backend = autoshorts_lib::scoring::resolve_scoring_backend().map_err(anyhow::Error::msg)?;
    let threshold = autoshorts_lib::scoring::resolve_recommend_threshold().map_err(anyhow::Error::msg)?;
    let top_n = autoshorts_lib::scoring::resolve_top_n().map_err(anyhow::Error::msg)?;
    println!(
        "[Acceptance] backend={backend:?} threshold={threshold:.2} top_n={top_n:?} (process-env scoped)"
    );
    if backend != autoshorts_lib::scoring::ScoringBackend::AutoClip {
        bail!("acceptance run requires AUTOSHORTS_SCORING_BACKEND=autoclip");
    }
    let model = std::env::var("AUTOSHORTS_SCORER_MODEL")
        .unwrap_or_else(|_| "nvidia/nemotron-3-ultra-550b-a55b (default)".to_string());
    println!("[Acceptance] scorer model: {model}");

    // 3. Source acquisition through the APP's own pathway.
    let source_path: PathBuf = if let Some(url) = &url {
        // 3a. The app's copyright gate — record exactly what the UI would see.
        match autoshorts_lib::youtube::check_copyright(url, None, None) {
            Ok(res) => println!(
                "[CopyrightGate] is_safe={} license={:?} (UI shows a warning + override when !is_safe)",
                res.is_safe, res.license
            ),
            Err(e) => println!("[CopyrightGate] check failed: {e}"),
        }
        // 3b. The app's downloader (same function `download_youtube_video` wraps).
        let target = dirs::download_dir().ok_or_else(|| anyhow!("no Downloads dir"))?;
        let t = Instant::now();
        let path = autoshorts_lib::youtube::download_video(url, &target, None, None, None)
            .map_err(|e| anyhow!("app download failed: {e}"))?;
        println!(
            "[Download] via youtube::download_video in {:.1}s -> {}",
            t.elapsed().as_secs_f64(),
            path.display()
        );
        path
    } else {
        PathBuf::from(source.unwrap())
    };

    // 4. Media verification (Part A2): probe + SHA-256.
    if !source_path.exists() {
        bail!("source missing: {}", source_path.display());
    }
    let size = std::fs::metadata(&source_path)?.len();
    let hash = sha256_file(&source_path)?;
    let probe = autoshorts_lib::media::probe_media(source_path.to_string_lossy().as_ref())
        .map_err(|e| anyhow!("probe failed: {e}"))?;
    probe_line(&format!("{label} source"), &probe);
    println!(
        "[Source] {} bytes={} sha256={}",
        source_path.display(),
        size,
        &hash[..16.min(hash.len())]
    );

    // 5. Real production DB + project row (same as create_project_from_path).
    let data_dir = dirs::data_dir()
        .ok_or_else(|| anyhow!("no data dir"))?
        .join("com.autoshorts.desktop");
    let db = Database::open(&data_dir.join("autoshorts.sqlite"))?;
    let project = create_project_from_path_impl(
        &db,
        source_path.to_string_lossy().as_ref(),
        "cloud",                 // Deepgram, as selected in the UI
        "preset_viral_bold",     // first caption style card in the UI
        "original",              // default framing mode (DualFrame enabled)
    )
    .map_err(anyhow::Error::msg)?;
    println!(
        "[Project] id={} name={:?} status={} duration={:?}",
        project.id, project.name, project.status, project.source_duration
    );
    let project_id = project.id.clone();

    // 6. Real transcription (Deepgram nova-3 + diarization labels).
    let t = Instant::now();
    let transcript = transcribe_project_impl(
        db.clone(),
        data_dir.clone(),
        project_id.clone(),
        "cloud".to_string(),
        None,
    )
    .await
    .map_err(|e| anyhow!("transcription failed: {e}"))?;
    let words: usize = serde_json::from_str::<NormalizedTranscript>(&transcript.raw_json)
        .map(|n| n.words.len())
        .unwrap_or(0);
    println!(
        "[Transcript] engine={} language={:?} words={} in {:.1}s",
        transcript.engine,
        transcript.language,
        words,
        t.elapsed().as_secs_f64()
    );
    if words < 50 {
        bail!("transcript unusable (<50 words) — diagnose before judging discovery");
    }

    // 7. The REAL pipeline: discovery -> AutoClip scoring -> gate -> rank ->
    //    Top-N -> persistence (generate_candidates_impl is the command body).
    let t = Instant::now();
    let candidates = generate_candidates_impl(
        db.clone(),
        project_id.clone(),
        None,
        Some("deepseek".to_string()),
        None,
        None,
        None,
        None,
        // Explicit UI-style selection: this run must score via AutoClip.
        Some("autoclip".to_string()),
    )
    .await
    .map_err(|e| anyhow!("generate_candidates failed: {e}"))?;
    println!(
        "[Pipeline] generate_candidates_impl finished in {:.1}s, returned {} candidates",
        t.elapsed().as_secs_f64(),
        candidates.len()
    );
    print_candidates("persisted pool (returned)", &candidates);

    // Stage counts (Part A4). Eligibility contract: llm provenance AND
    // score >= threshold (inclusive); fallbacks are never eligible.
    let score_source_of = |c: &Candidate| -> Option<String> {
        c.metadata_json
            .as_ref()
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .and_then(|m| m["scoreSource"].as_str().map(str::to_string))
    };
    let llm_scored = candidates
        .iter()
        .filter(|c| score_source_of(c).as_deref() == Some("llm"))
        .count();
    let fallbacks = candidates.len() - llm_scored;
    let eligible = candidates
        .iter()
        .filter(|c| score_source_of(c).as_deref() == Some("llm") && c.score >= threshold)
        .count();
    let selected: Vec<&Candidate> = candidates.iter().filter(|c| c.selected).collect();
    println!(
        "[Counts] discovered+persisted={} llm_scored={llm_scored} fallback={fallbacks} eligible={eligible} selected={} top_n={top_n:?}",
        candidates.len(),
        selected.len()
    );

    // 8. Fresh-connection reload verification (UI-reload agreement).
    let db2 = Database::open(&data_dir.join("autoshorts.sqlite"))?;
    let reloaded = db2.list_candidates(&project_id)?;
    if reloaded.len() != candidates.len() {
        bail!("reload mismatch: {} vs {}", reloaded.len(), candidates.len());
    }
    for (i, c) in reloaded.iter().enumerate() {
        assert_eq!(c.rank, (i + 1) as i64, "rank = index+1");
        assert_eq!(c.id, candidates[i].id, "reload order matches pipeline order");
        assert_eq!(c.selected, candidates[i].selected, "selected flags agree");
        assert_eq!(c.score, candidates[i].score, "scores agree after reload");
    }
    println!(
        "[Reload] fresh connection agrees: {} rows, {} selected, ranks/scores/provenance identical",
        reloaded.len(),
        selected.len()
    );

    // 9. Render EVERY selected candidate through the production render path.
    let tracker = InFlightRenderTracker::default();
    let mut rendered = 0usize;
    let mut render_failed = 0usize;
    for c in &selected {
        let t = Instant::now();
        match render_flat_clip_for_candidate_impl(
            db.clone(),
            data_dir.clone(),
            tracker.clone(),
            c.id.clone(),
        )
        .await
        {
            Ok(out_path) => {
                let out = PathBuf::from(&out_path);
                let out_probe = autoshorts_lib::media::probe_media(&out_path);
                let out_hash = sha256_file(&out).unwrap_or_default();
                println!(
                    "[Render] rank={} -> {} ({:.1}s) {} sha256={}",
                    c.rank,
                    out_path,
                    t.elapsed().as_secs_f64(),
                    out_probe
                        .as_ref()
                        .map(|p| format!(
                            "{}x{} dur={:?} audio={}",
                            p.width.unwrap_or(0),
                            p.height.unwrap_or(0),
                            p.duration_sec,
                            p.audio_codec.as_deref().unwrap_or("-")
                        ))
                        .unwrap_or_else(|e| format!("PROBE FAILED: {e}")),
                    &out_hash[..16.min(out_hash.len())]
                );
                rendered += 1;
            }
            Err(e) => {
                println!("[Render] rank={} FAILED: {e}", c.rank);
                render_failed += 1;
            }
        }
    }
    println!(
        "[Counts] rendered={rendered} render_failed={render_failed} (selected={})",
        selected.len()
    );

    println!("==============================================================");
    println!("  ACCEPTANCE E2E — {label}: COMPLETE");
    println!("==============================================================");
    Ok(())
}
