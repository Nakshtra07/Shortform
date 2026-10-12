//! Live End-to-End Validation of the AutoClip Holistic Scoring Backend
//!
//! Runs the REAL NVIDIA NIM scorer (`scoring::run_autoclip_scoring`) over a
//! REAL candidate pool + transcript loaded from a DISPOSABLE COPY of the
//! production database, then mirrors the exact AutoClip branch of
//! `generate_candidates` (eligibility gate → dedup/rank → Top-N selection →
//! `replace_candidates_with_selection` → reload verification), and finally
//! re-runs the scorer to prove cache attribution.
//!
//! NON-DESTRUCTIVE: the production database is copied to a temp file first;
//! every write happens against the copy. The only production-side effect is
//! the score cache namespace (`<cache_dir>/autoshorts/final_scores/`), which
//! is the feature working as designed (cached entries are genuine model
//! scores).
//!
//! Usage:
//!   cargo run --example autoclip_live_validation -- [options]
//!
//! Options:
//!   --project <project_id>   (default: project with the most candidates)
//!   --skip-second-run        (skip the repeat-run cache-hit verification)

use anyhow::{anyhow, bail, Result};
use autoshorts_lib::db::Database;
use autoshorts_lib::llm::{
    compute_transcript_source_identity, deduplicate_and_rank_all_candidates_autoclip,
};
use autoshorts_lib::models::{CandidateDraft, NormalizedTranscript};
use autoshorts_lib::scoring;
use std::time::Instant;

fn reset_to_fresh_pool(
    candidates: &[autoshorts_lib::models::Candidate],
) -> Result<Vec<CandidateDraft>> {
    let mut drafts: Vec<CandidateDraft> = candidates
        .iter()
        .map(|c| c.try_metadata_draft())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| anyhow!("reconstructing drafts from metadata: {e}"))?;
    // A fresh discovery pool enters the scorer unscored, exactly like the
    // in-pipeline state between multimodal analysis and AutoClip scoring.
    for d in drafts.iter_mut() {
        d.score_source = None;
        d.scoring_id = None;
    }
    Ok(drafts)
}

fn print_pool(label: &str, drafts: &[CandidateDraft]) {
    println!("---- {label} ----");
    for d in drafts.iter() {
        println!(
            "  {:?} [{:>7.1}s -> {:7.1}s] final_score={:.2} source={:?} reason={}",
            d.scoring_id.as_deref().unwrap_or("-"),
            d.start,
            d.end,
            d.score,
            d.score_source.as_deref().unwrap_or("-"),
            d.rationale.chars().take(90).collect::<String>()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Environment (same convention as run_reze_live.rs)
    let _ = dotenvy::from_path(r"D:\College\Autoshorts 11.0\autoshorts\.env");
    let _ = dotenvy::from_path(r"D:\College\Autoshorts 11.0\.env");
    let _ = dotenvy::dotenv();

    println!("==============================================================");
    println!("  AUTOSHORTS 11.0 — LIVE AUTOCLIP SCORING + TOP-N VALIDATION ");
    println!("==============================================================");

    let args: Vec<String> = std::env::args().collect();
    let chosen_project = args
        .iter()
        .position(|a| a == "--project")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string());
    let skip_second_run = args.iter().any(|a| a == "--skip-second-run");

    let key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
    if key.trim().is_empty() {
        bail!("NVIDIA_API_KEY is not configured; live validation impossible");
    }
    let backend = scoring::resolve_scoring_backend().map_err(anyhow::Error::msg)?;
    let threshold = scoring::resolve_recommend_threshold().map_err(anyhow::Error::msg)?;
    let top_n = scoring::resolve_top_n().map_err(anyhow::Error::msg)?;
    println!(
        "[Validation] AUTOSHORTS_SCORING_BACKEND={:?} threshold={:.2} top_n={:?}",
        backend, threshold, top_n
    );

    // 2. Disposable copy of the production DB (non-destructive validation)
    let prod_db = dirs::data_dir()
        .ok_or_else(|| anyhow!("cannot resolve data dir"))?
        .join("com.autoshorts.desktop")
        .join("autoshorts.sqlite");
    if !prod_db.exists() {
        bail!("production database not found at {}", prod_db.display());
    }
    let tmp_db = std::env::temp_dir().join(format!(
        "autoshorts_autoclip_validation_{}.sqlite",
        uuid::Uuid::new_v4()
    ));
    std::fs::copy(&prod_db, &tmp_db)?;
    println!(
        "[Validation] Working on a DISPOSABLE COPY of the production DB: {}",
        tmp_db.display()
    );
    let db = Database::open(&tmp_db)?;

    // 3. Pick a real project with candidates + transcript
    let project_id = match &chosen_project {
        Some(p) => p.clone(),
        None => {
            let mut best: Option<(usize, String)> = None;
            for project in db.list_projects()? {
                if db.latest_transcript(&project.id)?.is_none() {
                    continue;
                }
                let count = db.list_candidates(&project.id)?.len();
                if count > 0 && best.as_ref().map(|(c, _)| count > *c).unwrap_or(true) {
                    best = Some((count, project.id));
                }
            }
            best.map(|(_, id)| id)
                .ok_or_else(|| anyhow!("no project with candidates + transcript found"))?
        }
    };
    let transcript = db
        .latest_transcript(&project_id)?
        .ok_or_else(|| anyhow!("project {project_id} has no transcript"))?;
    let normalized: NormalizedTranscript = serde_json::from_str(&transcript.raw_json)?;
    if normalized.words.is_empty() {
        bail!("transcript has no words");
    }
    let candidates = db.list_candidates(&project_id)?;
    if candidates.is_empty() {
        bail!("project {project_id} has no candidate pool");
    }
    println!(
        "[Validation] project={project_id} words={} pool={}",
        normalized.words.len(),
        candidates.len()
    );

    // 4. LIVE scoring run #1 (billable NVIDIA NIM call)
    let original_drafts = reset_to_fresh_pool(&candidates)?;
    let mut drafts = original_drafts.clone();
    let source_identity = compute_transcript_source_identity(&normalized);
    let t0 = Instant::now();
    let summary =
        scoring::run_autoclip_scoring(&source_identity, &mut drafts, &normalized.words, &key)
            .await
            .map_err(|f| anyhow!("live scoring failed: {f}"))?;
    println!(
        "[Validation] live run #1 finished in {:.1}s: entered={} cache_hits={} cache_misses={} \
         valid={} unscored={} invalid={} ambiguous={} unknown={} failed_batches={}",
        t0.elapsed().as_secs_f64(),
        summary.candidates_entering,
        summary.cache_hits,
        summary.cache_misses,
        summary.valid_scores,
        summary.unscored,
        summary.invalid_scores,
        summary.ambiguous_ids,
        summary.unknown_ids,
        summary.failed_batches
    );
    print_pool("scored pool (run #1)", &drafts);

    // Contract checks on the scored pool
    for d in drafts.iter() {
        let source = d
            .score_source
            .as_deref()
            .ok_or_else(|| anyhow!("candidate missing score_source"))?;
        if source == scoring::SCORE_SOURCE_LLM {
            assert!(
                (0.0..=1.0).contains(&d.score),
                "llm score out of [0,1]: {}",
                d.score
            );
        } else if source == scoring::SCORE_SOURCE_FALLBACK {
            assert_eq!(d.score, 0.0, "fallback must carry score 0.0");
        } else {
            bail!("unexpected score_source: {source}");
        }
    }
    let run1_scores: Vec<(Option<String>, f64, Option<String>, String)> = drafts
        .iter()
        .map(|d| {
            (
                d.scoring_id.clone(),
                d.score,
                d.score_source.clone(),
                d.rationale.clone(),
            )
        })
        .collect();
    let run1_llm_count = drafts
        .iter()
        .filter(|d| d.score_source.as_deref() == Some(scoring::SCORE_SOURCE_LLM))
        .count();
    let run1_fallback_count = drafts.len() - run1_llm_count;
    println!(
        "[Validation] run #1 attribution: {} genuine llm scores, {} fallbacks (never cached)",
        run1_llm_count, run1_fallback_count
    );

    // 5. Mirror the generate_candidates AutoClip branch: eligibility gate →
    //    dedup/rank → Top-N selection → persistence → reload verification.
    let before_gate = drafts.len();
    let gate_backup = drafts.clone();
    drafts.retain(|d| {
        let llm_scored = d.score_source.as_deref() == Some(scoring::SCORE_SOURCE_LLM);
        !llm_scored || d.score >= threshold
    });
    let gate_removed = before_gate - drafts.len();
    if drafts.is_empty() {
        println!("[Validation] eligibility gate emptied the pool; restoring");
        drafts = gate_backup;
    } else {
        println!(
            "[Validation] eligibility gate (>= {:.2}): {} of {} removed, unscored alternatives retained",
            threshold,
            gate_removed,
            before_gate
        );
    }
    let after_gate = drafts.len();

    let ranked = deduplicate_and_rank_all_candidates_autoclip(drafts, normalized.duration, 12);
    let eligible_count = ranked
        .iter()
        .filter(|d| scoring::is_autoclip_eligible(d, threshold))
        .count();
    let selected_count = scoring::effective_selected_count(top_n, eligible_count);
    println!(
        "[Validation] ranked pool: {} (gate removed {}, dedup removed {}), eligible={}, top_n={:?} -> selected_count={}",
        ranked.len(),
        gate_removed,
        after_gate - ranked.len(),
        eligible_count,
        top_n,
        selected_count
    );
    print_pool("final ranked pool", &ranked);

    db.replace_candidates_with_selection(&project_id, &ranked, selected_count)?;
    let reloaded = db.list_candidates(&project_id)?;

    // Persistence contract: rank = index+1, selected = index < selected_count,
    // Not Selected alternatives retained, provenance survives the roundtrip.
    assert_eq!(reloaded.len(), ranked.len(), "pool size must roundtrip");
    for (idx, c) in reloaded.iter().enumerate() {
        assert_eq!(c.rank, (idx + 1) as i64, "rank = index + 1");
        assert_eq!(
            c.selected,
            idx < selected_count,
            "selected = index < selected_count"
        );
        let meta = c
            .metadata_json
            .as_ref()
            .ok_or_else(|| anyhow!("metadata_json missing after reload"))?;
        let parsed: serde_json::Value = serde_json::from_str(meta)?;
        let source = parsed["scoreSource"].as_str().unwrap_or("-");
        assert_eq!(
            source,
            ranked[idx].score_source.as_deref().unwrap_or("-"),
            "score_source must survive persistence"
        );
    }
    let selected_total: usize = reloaded.iter().filter(|c| c.selected).count();
    assert_eq!(
        selected_total, selected_count,
        "exactly selected_count selected"
    );
    println!(
        "[Validation] persistence verified: {} persisted, {} selected, {} Not Selected alternatives retained",
        reloaded.len(),
        selected_total,
        reloaded.len() - selected_total
    );

    // 6. Second run from the ORIGINAL pool state: cache attribution proof.
    //    - every candidate genuinely scored in run #1 must be a cache hit with
    //      an identical score AND identical recommend_reason (cached entries
    //      ARE genuine model scores, replayed deterministically);
    //    - every run-#1 fallback must be a live miss again (failed/unscored
    //      batches are never cached, so they legitimately retry).
    if !skip_second_run {
        let mut drafts2 = original_drafts;
        let t1 = Instant::now();
        let summary2 =
            scoring::run_autoclip_scoring(&source_identity, &mut drafts2, &normalized.words, &key)
                .await
                .map_err(|f| anyhow!("second run failed: {f}"))?;
        println!(
            "[Validation] run #2 (cache attribution) in {:.1}s: cache_hits={} cache_misses={} valid={}",
            t1.elapsed().as_secs_f64(),
            summary2.cache_hits,
            summary2.cache_misses,
            summary2.valid_scores
        );
        assert!(
            summary2.cache_misses <= run1_fallback_count,
            "only run-#1 fallbacks may retry live (misses={}, fallbacks={}); \
             failures are never cached",
            summary2.cache_misses,
            run1_fallback_count
        );
        assert!(
            summary2.cache_hits >= run1_llm_count,
            "every run-#1 llm score must be replayed from cache (hits={} llm={})",
            summary2.cache_hits,
            run1_llm_count
        );
        let run2_by_id: std::collections::HashMap<
            &Option<String>,
            &autoshorts_lib::models::CandidateDraft,
        > = drafts2.iter().map(|d| (&d.scoring_id, d)).collect();
        for (run1_id, score, source, rationale) in &run1_scores {
            if source.as_deref() == Some(scoring::SCORE_SOURCE_LLM) {
                let run2 = run2_by_id
                    .get(run1_id)
                    .ok_or_else(|| anyhow!("run-#1 llm candidate missing in run #2"))?;
                assert_eq!(
                    run2.score, *score,
                    "cached replay must reproduce the live final_score exactly"
                );
                assert_eq!(
                    &run2.score_source, source,
                    "cached replay must keep score_source=llm"
                );
                assert_eq!(
                    &run2.rationale, rationale,
                    "cached replay must reproduce the live recommend_reason"
                );
            }
        }
        let run2_llm_count = drafts2
            .iter()
            .filter(|d| d.score_source.as_deref() == Some(scoring::SCORE_SOURCE_LLM))
            .count();
        println!(
            "[Validation] cache replay verified: run-#1 llm scores identical via cache; run #2 now has {} llm / {} fallback",
            run2_llm_count,
            drafts2.len() - run2_llm_count
        );
    }

    let _ = std::fs::remove_file(&tmp_db);
    println!("==============================================================");
    println!("  LIVE AUTOCLIP SCORING + TOP-N VALIDATION: ALL CHECKS PASSED");
    println!("==============================================================");
    Ok(())
}
