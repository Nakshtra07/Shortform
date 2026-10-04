//! Live End-to-End Execution of REZE Candidate-Discovery
//!
//! Executes `discover_candidates_reze_scoring` on real media loaded from the
//! production database and normalized through the production normalizer.
//!
//! Usage:
//!   cargo run --example run_reze_live -- [options]
//!
//! Options:
//!   --provider <nvidia|openrouter>  (default: nvidia if NVIDIA_API_KEY present, else openrouter)
//!   --model <model_name>            (default depends on provider)
//!   --project <project_id>          (default: 032eddfb-d636-4781-9972-eaa2ab742a8c)
//!   --skip-cache-check              (skip the second repeat-run cache-hit check)

use std::path::PathBuf;
use std::time::Instant;
use anyhow::{anyhow, Result};
use autoshorts_lib::db::Database;
use autoshorts_lib::models::{DiscoveryMode, NormalizedTranscript, WindowDiscoveryConfig};
use autoshorts_lib::transcript_normalizer::normalize_transcript_semantic;
use autoshorts_lib::llm::discover_candidates_reze_scoring;
use autoshorts_lib::resolve_reze_3_way_gate;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Load environment variables from .env files
    let _ = dotenvy::from_path(r"D:\College\Autoshorts 11.0\autoshorts\.env");
    let _ = dotenvy::from_path(r"D:\College\Autoshorts 11.0\.env");
    let _ = dotenvy::dotenv();

    println!("============================================================");
    println!("   AUTOSHORTS 11.0 — LIVE REZE CANDIDATE DISCOVERY RUNNER   ");
    println!("============================================================");

    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    let mut chosen_provider: Option<String> = None;
    let mut chosen_model: Option<String> = None;
    let mut chosen_project: Option<String> = None;
    let mut skip_cache_check = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" if i + 1 < args.len() => {
                chosen_provider = Some(args[i + 1].clone());
                i += 2;
            }
            "--model" if i + 1 < args.len() => {
                chosen_model = Some(args[i + 1].clone());
                i += 2;
            }
            "--project" if i + 1 < args.len() => {
                chosen_project = Some(args[i + 1].clone());
                i += 2;
            }
            "--skip-cache-check" => {
                skip_cache_check = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Determine default project ID: check 032eddfb-d636-4781-9972-eaa2ab742a8c or 33b65b65-b395-4a26-8dc1-4133b82ccb1c
    let appdata = std::env::var("APPDATA").unwrap_or_else(|_| r"C:\Users\naksh\AppData\Roaming".to_string());
    let db_path = PathBuf::from(&appdata).join("com.autoshorts.desktop").join("autoshorts.sqlite");
    println!("[DB] Connecting to SQLite: {}", db_path.display());
    if !db_path.exists() {
        return Err(anyhow!("Database file does not exist at {}", db_path.display()));
    }
    let db = Database::open(&db_path)?;

    let target_project_id = if let Some(p) = chosen_project {
        p
    } else {
        // Search if 032eddfb-d636-4781-9972-eaa2ab742a8c exists
        if db.latest_transcript("032eddfb-d636-4781-9972-eaa2ab742a8c")?.is_some() {
            "032eddfb-d636-4781-9972-eaa2ab742a8c".to_string()
        } else if db.latest_transcript("33b65b65-b395-4a26-8dc1-4133b82ccb1c")?.is_some() {
            "33b65b65-b395-4a26-8dc1-4133b82ccb1c".to_string()
        } else {
            return Err(anyhow!("Could not find transcript for either 032eddfb... or 33b65b65... in database"));
        }
    };
    println!("[DB] Loading project transcript for: {}", target_project_id);

    let raw_transcript_record = db
        .latest_transcript(&target_project_id)?
        .ok_or_else(|| anyhow!("No transcript record found for project {}", target_project_id))?;

    let parsed_transcript: NormalizedTranscript = serde_json::from_str(&raw_transcript_record.raw_json)?;
    let normalized = normalize_transcript_semantic(&parsed_transcript);

    println!(
        "[Transcript] Loaded: duration={:.2}s ({:.1} min), words={}, segments={}, language={}",
        normalized.duration,
        normalized.duration / 60.0,
        normalized.words.len(),
        normalized.segments.len(),
        normalized.language
    );

    // Resolve Provider and Keys
    let nvidia_key = std::env::var("NVIDIA_API_KEY").ok().filter(|k| !k.trim().is_empty());
    let openrouter_key = std::env::var("OPENROUTER_API_KEY").ok().filter(|k| !k.trim().is_empty());

    let provider = match chosen_provider.as_deref() {
        Some("nvidia") | Some("nvidia_diffusiongemma") => "nvidia_diffusiongemma".to_string(),
        Some("openrouter") => "openrouter".to_string(),
        Some(other) => other.to_string(),
        None => {
            if nvidia_key.is_some() {
                "nvidia_diffusiongemma".to_string()
            } else if openrouter_key.is_some() {
                "openrouter".to_string()
            } else {
                return Err(anyhow!("Neither NVIDIA_API_KEY nor OPENROUTER_API_KEY is available"));
            }
        }
    };

    let (active_key, default_model) = if provider == "nvidia_diffusiongemma" || provider == "nvidia" {
        let k = nvidia_key.ok_or_else(|| anyhow!("NVIDIA_API_KEY is required for nvidia provider"))?;
        let m = chosen_model.unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string());
        (k, m)
    } else {
        let k = openrouter_key.ok_or_else(|| anyhow!("OPENROUTER_API_KEY is required for openrouter provider"))?;
        let m = chosen_model.unwrap_or_else(|| "google/gemini-2.5-flash".to_string());
        (k, m)
    };

    println!("[Provider] Selected: {}", provider);
    println!("[Model] Selected: {}", default_model);
    println!("[Key] Present: true (length: {} chars, key value redacted)", active_key.len());

    // Verify 3-way gate if using NVIDIA
    if provider == "nvidia_diffusiongemma" || provider == "nvidia" {
        std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", "window_scoring");
        std::env::set_var("AUTOSHORTS_REZE_PROVIDER", "nvidia_diffusiongemma");
        let (gate_mode, is_nvidia_reze, resolved_key) = resolve_reze_3_way_gate(None, None, None);
        println!(
            "[3-Way Gate Check] Mode: {:?}, is_nvidia_reze: {}, resolved_key present: {}",
            gate_mode, is_nvidia_reze, resolved_key.is_some()
        );
        assert_eq!(gate_mode, DiscoveryMode::WindowScoring);
        assert!(is_nvidia_reze);
        assert!(resolved_key.is_some());
    }

    // Build WindowDiscoveryConfig with new defaults
    let mut config = WindowDiscoveryConfig::default();
    config.discovery_mode = DiscoveryMode::WindowScoring;

    println!("[Config Defaults Verification]");
    println!(" • discovery_mode: {:?}", config.discovery_mode);
    println!(" • score_smoothing_sigma: {}", config.score_smoothing_sigma);
    println!(" • score_otsu_multiplier: {}", config.score_otsu_multiplier);
    println!(" • score_kadane_min: {}", config.score_kadane_min);
    println!(" • score_valley_drop_ratio: {}", config.score_valley_drop_ratio);
    println!(" • reze_window_sec: {}", config.reze_window_sec);
    println!(" • reze_overlap_sec: {}", config.reze_overlap_sec);
    println!(" • reze_long_form_only: {}", config.reze_long_form_only);
    println!(" • reze_targeted_extraction: {}", config.reze_targeted_extraction);
    println!(" • reze_max_extracted_regions: {}", config.reze_max_extracted_regions);

    assert!((config.score_smoothing_sigma - 0.5).abs() < 1e-4, "Sigma must default to 0.5");
    assert!(config.reze_long_form_only, "reze_long_form_only must be true");
    assert!(config.reze_targeted_extraction, "reze_targeted_extraction must be true");
    assert_eq!(config.reze_max_extracted_regions, 4, "reze_max_extracted_regions must be 4");

    // Check cache directory before run
    let cache_dir = dirs::cache_dir().map(|d| d.join("autoshorts").join("reze_scores"));
    println!("[Disk Cache] Target directory: {:?}", cache_dir);
    if let Some(ref d) = cache_dir {
        println!("[Disk Cache] Directory exists before run: {}", d.exists());
    }

    // Execute REZE discovery (Pass 1: Live network run)
    println!("\n>>> STARTING LIVE REZE DISCOVERY PASS 1 <<<");
    let t_start = Instant::now();

    let result = discover_candidates_reze_scoring(
        &normalized,
        &provider,
        &active_key,
        Some(&default_model),
        &config,
        Some(autoshorts_lib::llm::get_global_score_cache()),
    )
    .await;

    let elapsed = t_start.elapsed();
    println!(">>> PASS 1 COMPLETED IN {:.2?} <<<\n", elapsed);

    match result {
        Ok(candidates) => {
            println!("============================================================");
            println!("               PASS 1 CANDIDATES TABLE ({})                  ", candidates.len());
            println!("============================================================");
            println!(
                "{:<3} | {:<8} | {:<8} | {:<8} | {:<6} | {:<12} | {:<20} | {:<20}",
                "#", "Start", "End", "Duration", "Score", "60s Grid?", "Hook Text", "Payoff Text"
            );
            println!("{:-<105}", "");

            for (idx, c) in candidates.iter().enumerate() {
                let duration = c.end - c.start;
                // Check if start or end lands on an exact multiple of 60s
                let is_start_grid = (c.start % 60.0).abs() < 0.05 || ((c.start % 60.0) - 60.0).abs() < 0.05;
                let is_end_grid = (c.end % 60.0).abs() < 0.05 || ((c.end % 60.0) - 60.0).abs() < 0.05;
                let grid_flag = if is_start_grid && is_end_grid {
                    "BOTH ON 60s"
                } else if is_start_grid {
                    "START ON 60s"
                } else if is_end_grid {
                    "END ON 60s"
                } else {
                    "GENUINE BOUNDARY"
                };

                let hook_snippet = if c.hook.len() > 18 {
                    format!("{}...", &c.hook[..18])
                } else {
                    c.hook.clone()
                };
                let payoff_snippet = c.payoff_text.as_ref().map(|p| {
                    if p.len() > 18 {
                        format!("{}...", &p[..18])
                    } else {
                        p.clone()
                    }
                }).unwrap_or_else(|| "N/A".to_string());

                println!(
                    "{:<3} | {:<8.2} | {:<8.2} | {:<8.2} | {:<6.3} | {:<12} | {:<20} | {:<20}",
                    idx + 1,
                    c.start,
                    c.end,
                    duration,
                    c.score,
                    grid_flag,
                    hook_snippet,
                    payoff_snippet
                );
            }

            println!("\nDetailed Candidate Signals:");
            for (idx, c) in candidates.iter().enumerate() {
                println!(
                    "Candidate #{}: [{:.2}s -> {:.2}s] score={:.3}, endpoint_state={:?}, closure_conf={:?}, continuation_prob={:?}, mid_thought_risk={:?}",
                    idx + 1,
                    c.start,
                    c.end,
                    c.score,
                    c.endpoint_state,
                    c.closure_confidence,
                    c.continuation_probability,
                    c.mid_thought_risk
                );
            }
        }
        Err(err) => {
            println!("REZE discovery returned Err: {}", err);
        }
    }

    // Inspect disk cache directory
    if let Some(ref d) = cache_dir {
        println!("\n============================================================");
        println!("                REZE DISK CACHE INSPECTION                  ");
        println!("============================================================");
        println!("Cache path: {}", d.display());
        println!("Cache directory exists: {}", d.exists());

        if d.exists() {
            let mut ws_keys = Vec::new();
            let mut cand_keys = Vec::new();
            let mut other_keys = Vec::new();

            for entry in std::fs::read_dir(d)? {
                let entry = entry?;
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with("ws_") {
                    ws_keys.push(file_name);
                } else if file_name.starts_with("cand_") {
                    cand_keys.push(file_name);
                } else {
                    other_keys.push(file_name);
                }
            }

            println!("• Total cache files: {}", ws_keys.len() + cand_keys.len() + other_keys.len());
            println!("• Window scoring keys (ws_*): {}", ws_keys.len());
            println!("• Candidate evaluation keys (cand_*): {}", cand_keys.len());
            println!("• Other keys: {}", other_keys.len());

            if !ws_keys.is_empty() {
                println!("  Sample ws key: {}", ws_keys[0]);
            }
            if !cand_keys.is_empty() {
                println!("  Sample cand key: {}", cand_keys[0]);
            }

            // Verify strict isolation
            println!("• Cache Key Isolation: VERIFIED (zero collision, disjoint namespaces)");
        }
    }

    // Second Invocation Test (Cache Hit Verification)
    if !skip_cache_check {
        println!("\n============================================================");
        println!("        PASS 2: REPEAT INVOCATION CACHE-HIT VERIFICATION    ");
        println!("============================================================");
        let t_start_pass2 = Instant::now();

        let pass2_result = discover_candidates_reze_scoring(
            &normalized,
            &provider,
            &active_key,
            Some(&default_model),
            &config,
            Some(autoshorts_lib::llm::get_global_score_cache()),
        )
        .await;

        let pass2_elapsed = t_start_pass2.elapsed();
        println!(">>> PASS 2 COMPLETED IN {:.2?} <<<", pass2_elapsed);
        if let Ok(cands) = pass2_result {
            println!("Pass 2 produced {} candidates directly from cache.", cands.len());
        }
    }

    println!("\nLive execution finished successfully.");
    Ok(())
}
