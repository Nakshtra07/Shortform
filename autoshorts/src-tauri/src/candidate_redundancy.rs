//! AutoShorts 11.0 Phase 4 — Candidate Redundancy Detection
//!
//! Prevents AutoShorts from producing multiple near-identical shorts by
//! detecting and suppressing duplicate/overlapping candidates using
//! transcript embeddings and deterministic similarity rules.
//!
//! Architecture:
//!   FINAL CANDIDATES
//!        ↓
//!   EMBED CANDIDATE CONTENT (transcript-based embeddings)
//!        ↓
//!   SIMILARITY MATRIX
//!        ↓
//!   DETERMINISTIC REDUNDANCY CHECK
//!        ↓
//!   KEEP DISTINCT CANDIDATES (preserve stronger by explicit ranking policy)
//!        ↓
//!   FINAL SELECTION

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Redundancy detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RedundancyConfig {
    /// Enable/disable redundancy detection
    pub enabled: bool,
    /// Embedding model: "tfidf", "sentence_transformers", "openai"
    pub embedding_model: String,
    /// Model version
    pub model_version: String,
    /// Similarity threshold for near-duplicate detection (0.0–1.0)
    pub similarity_threshold: f64,
    /// Minimum time gap between candidates to consider non-overlapping (seconds)
    pub min_time_gap_sec: f64,
    /// Preserve candidate with higher composite score when duplicates found
    pub preserve_higher_scored: bool,
    /// Cache directory
    pub cache_dir: Option<String>,
}

impl Default for RedundancyConfig {
    fn default() -> Self {
        Self {
            enabled: false, // OFF by default — opt-in
            embedding_model: "tfidf".to_string(),
            model_version: "1.0".to_string(),
            similarity_threshold: 0.85,
            min_time_gap_sec: 2.0,
            preserve_higher_scored: true,
            cache_dir: None,
        }
    }
}

/// Candidate embedding for similarity computation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateEmbedding {
    /// Candidate ID
    pub candidate_id: String,
    /// Embedding vector (serialized based on model)
    pub embedding: Vec<f32>,
    /// Source hash
    pub source_hash: String,
    /// Candidate time range
    pub start_sec: f64,
    pub end_sec: f64,
    /// Model that produced this embedding
    pub model: String,
    /// Model version
    pub model_version: String,
    /// ISO8601 timestamp
    pub created_at: String,
}

/// Redundancy analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RedundancyResult {
    /// Total candidates analyzed
    pub total_candidates: usize,
    /// Candidates flagged as redundant
    pub redundant_count: usize,
    /// Pairs of redundant candidates (indices into original array)
    pub redundant_pairs: Vec<RedundantPair>,
    /// Indices of candidates to KEEP (after deduplication)
    pub keep_indices: Vec<usize>,
    /// Indices of candidates to REMOVE
    pub remove_indices: Vec<usize>,
}

/// Pair of redundant candidates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RedundantPair {
    /// Index of first candidate
    pub index_a: usize,
    /// Index of second candidate
    pub index_b: usize,
    /// Similarity score (0.0–1.0)
    pub similarity: f64,
    /// Reason for redundancy classification
    pub reason: String,
}

/// Redundancy detection engine
pub struct RedundancyEngine {
    config: RedundancyConfig,
    _cache_dir: std::path::PathBuf,
}

impl RedundancyEngine {
    pub fn new(config: RedundancyConfig) -> Result<Self> {
        let cache_dir = config
            .cache_dir
            .as_ref()
            .map(|s| std::path::PathBuf::from(s))
            .unwrap_or_else(|| {
                dirs::cache_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join("autoshorts")
                    .join("candidate_redundancy")
            });
        std::fs::create_dir_all(&cache_dir).context("creating redundancy cache dir")?;
        Ok(Self { config, _cache_dir: cache_dir })
    }

    /// Check if redundancy detection is enabled
    pub fn is_enabled(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        match std::env::var("AUTOSHORTS_CANDIDATE_REDUNDANCY") {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            ),
            Err(_) => self.config.enabled,
        }
    }

    /// Compute TF-IDF style embeddings from transcript text
    /// This is a lightweight, dependency-free embedding approach
    fn compute_tfidf_embeddings(
        &self,
        candidates: &[crate::models::CandidateDraft],
        transcript_words: &[crate::models::TranscriptWord],
    ) -> Vec<CandidateEmbedding> {
        // Build vocabulary from all candidate transcripts
        let mut all_terms = HashMap::new();
        let mut candidate_terms = Vec::new();

        for candidate in candidates {
            // Get transcript words in candidate range
            let words: Vec<String> = transcript_words
                .iter()
                .filter(|w| w.end > candidate.start && w.start < candidate.end)
                .map(|w| w.text.to_lowercase())
                .collect();

            // Simple term extraction (words, bigrams)
            let mut terms = HashMap::new();
            for i in 0..words.len() {
                *terms.entry(words[i].clone()).or_insert(0) += 1;
                if i + 1 < words.len() {
                    let bigram = format!("{} {}", words[i], words[i + 1]);
                    *terms.entry(bigram).or_insert(0) += 1;
                }
            }

            for term in terms.keys() {
                *all_terms.entry(term.clone()).or_insert(0) += 1;
            }
            candidate_terms.push(terms);
        }

        // Compute IDF
        let total_docs = candidates.len() as f32;
        let mut idf = HashMap::new();
        for (term, df) in all_terms {
            let idf_val = (total_docs / df as f32).ln() + 1.0;
            idf.insert(term, idf_val);
        }

        // Build TF-IDF vectors
        let vocab: Vec<String> = idf.keys().cloned().collect();
        let mut embeddings = Vec::new();

        for (idx, terms) in candidate_terms.iter().enumerate() {
            let candidate = &candidates[idx];
            let total_terms: f32 = terms.values().sum::<i32>() as f32;
            let mut vector = vec![0.0; vocab.len()];

            for (vocab_idx, term) in vocab.iter().enumerate() {
                if let Some(&tf) = terms.get(term) {
                    let tf_val = tf as f32 / total_terms.max(1.0);
                    let idf_val = *idf.get(term).unwrap_or(&1.0);
                    vector[vocab_idx] = tf_val * idf_val;
                }
            }

            // Normalize
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut vector {
                    *x /= norm;
                }
            }

            embeddings.push(CandidateEmbedding {
                candidate_id: format!(
                    "{:.0}_{:.0}",
                    candidate.start * 1000.0,
                    candidate.end * 1000.0
                ),
                embedding: vector,
                source_hash: String::new(), // Will be set by caller
                start_sec: candidate.start,
                end_sec: candidate.end,
                model: self.config.embedding_model.clone(),
                model_version: self.config.model_version.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
            });
        }

        embeddings
    }

    /// Compute cosine similarity between two embeddings
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)) as f64
    }

    /// Check temporal overlap
    fn temporal_overlap(&self, a: &CandidateEmbedding, b: &CandidateEmbedding) -> bool {
        let overlap_start = a.start_sec.max(b.start_sec);
        let overlap_end = a.end_sec.min(b.end_sec);
        let overlap = overlap_end - overlap_start;
        overlap > 0.0 && overlap > self.config.min_time_gap_sec
    }

    /// Run redundancy detection on candidates
    pub fn detect_redundancy(
        &self,
        source_hash: &str,
        candidates: &[crate::models::CandidateDraft],
        transcript_words: &[crate::models::TranscriptWord],
    ) -> Result<RedundancyResult> {
        if !self.is_enabled() {
            return Ok(RedundancyResult {
                total_candidates: candidates.len(),
                redundant_count: 0,
                redundant_pairs: Vec::new(),
                keep_indices: (0..candidates.len()).collect(),
                remove_indices: Vec::new(),
            });
        }

        if candidates.len() <= 1 {
            return Ok(RedundancyResult {
                total_candidates: candidates.len(),
                redundant_count: 0,
                redundant_pairs: Vec::new(),
                keep_indices: (0..candidates.len()).collect(),
                remove_indices: Vec::new(),
            });
        }

        // Compute embeddings
        let mut embeddings = self.compute_tfidf_embeddings(candidates, transcript_words);
        for emb in &mut embeddings {
            emb.source_hash = source_hash.to_string();
        }

        // Compute similarity matrix and find redundant pairs
        let mut redundant_pairs = Vec::new();
        let n = embeddings.len();

        for i in 0..n {
            for j in (i + 1)..n {
                let sim =
                    Self::cosine_similarity(&embeddings[i].embedding, &embeddings[j].embedding);
                let temporal = self.temporal_overlap(&embeddings[i], &embeddings[j]);

                if sim >= self.config.similarity_threshold && temporal {
                    let reason = if sim >= 0.95 {
                        "near-exact duplicate".to_string()
                    } else if sim >= 0.90 {
                        "high content similarity".to_string()
                    } else {
                        "significant content overlap".to_string()
                    };

                    redundant_pairs.push(RedundantPair {
                        index_a: i,
                        index_b: j,
                        similarity: sim,
                        reason,
                    });
                }
            }
        }

        // Determine which candidates to keep/remove
        let mut remove_set = std::collections::HashSet::new();
        let mut keep_set = std::collections::HashSet::new();

        // Process pairs: remove the lower-scored candidate (or first if equal)
        for pair in &redundant_pairs {
            let cand_a = &candidates[pair.index_a];
            let cand_b = &candidates[pair.index_b];

            let remove_idx = if self.config.preserve_higher_scored {
                if cand_a.score >= cand_b.score {
                    pair.index_b
                } else {
                    pair.index_a
                }
            } else {
                pair.index_a // Remove first by default
            };

            remove_set.insert(remove_idx);
            keep_set.insert(if remove_idx == pair.index_a {
                pair.index_b
            } else {
                pair.index_a
            });
        }

        // Add non-redundant candidates to keep
        for i in 0..n {
            if !remove_set.contains(&i) {
                keep_set.insert(i);
            }
        }

        let keep_indices: Vec<usize> = keep_set.into_iter().collect();
        let remove_indices: Vec<usize> = remove_set.into_iter().collect();

        Ok(RedundancyResult {
            total_candidates: n,
            redundant_count: remove_indices.len(),
            redundant_pairs,
            keep_indices,
            remove_indices,
        })
    }

    /// Apply redundancy filtering to candidate list
    pub fn filter_candidates(
        &self,
        result: &RedundancyResult,
        candidates: Vec<crate::models::CandidateDraft>,
    ) -> Vec<crate::models::CandidateDraft> {
        let keep_set: std::collections::HashSet<usize> =
            result.keep_indices.iter().cloned().collect();
        candidates
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep_set.contains(i))
            .map(|(_, c)| c)
            .collect()
    }
}

/// Feature flag
pub fn candidate_redundancy_enabled() -> bool {
    match std::env::var("AUTOSHORTS_CANDIDATE_REDUNDANCY") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => false, // Default OFF — opt-in
    }
}

/// Integration: run redundancy detection on final candidate list
pub fn run_redundancy_detection(
    source_hash: &str,
    candidates: &mut Vec<crate::models::CandidateDraft>,
    transcript_words: &[crate::models::TranscriptWord],
) {
    if !candidate_redundancy_enabled() {
        return;
    }

    // DOUBLE-DISABLE FIX: `RedundancyConfig::default()` has `enabled: false`,
    // so `engine.is_enabled()` returned false and `detect_redundancy` produced
    // the no-op result even when `AUTOSHORTS_CANDIDATE_REDUNDANCY=1`. The env
    // gate above is already checked, so enable the config structurally here.
    // The kill-switch is NOT bypassed: this function still returns early when
    // `candidate_redundancy_enabled()` is false.
    let config = RedundancyConfig {
        enabled: true,
        ..Default::default()
    };
    let engine = match RedundancyEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[Redundancy] Engine init failed (non-fatal): {}", e);
            return;
        }
    };

    let result = match engine.detect_redundancy(source_hash, candidates, transcript_words) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[Redundancy] Detection failed (non-fatal): {}", e);
            return;
        }
    };

    if result.redundant_count > 0 {
        println!(
            "[Redundancy] Filtered {} redundant candidates ({} kept)",
            result.redundant_count,
            result.keep_indices.len()
        );
        for pair in &result.redundant_pairs {
            println!(
                "  [{}] vs [{}] similarity={:.3} ({})",
                pair.index_a, pair.index_b, pair.similarity, pair.reason
            );
        }
    }

    *candidates = engine.filter_candidates(&result, std::mem::take(candidates));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate `AUTOSHORTS_CANDIDATE_REDUNDANCY`.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn draft(start: f64, end: f64, score: f64) -> crate::models::CandidateDraft {
        crate::models::CandidateDraft {
            start,
            end,
            score,
            hook: format!("hook {start}"),
            rationale: "test".into(),
            payoff_end: Some(end),
            ..Default::default()
        }
    }

    fn words(texts: &[(f64, f64)]) -> Vec<crate::models::TranscriptWord> {
        texts
            .iter()
            .map(|(s, e)| crate::models::TranscriptWord {
                text: "alpha beta gamma delta".into(),
                start: *s,
                end: *e,
                speaker: Some("S1".into()),
            })
            .collect()
    }

    /// DOUBLE-DISABLE REGRESSION GUARD.
    ///
    /// `run_redundancy_detection` used to build `RedundancyConfig::default()`
    /// (enabled = false), so `engine.is_enabled()` returned false and the
    /// engine produced a no-op result even when the environment flag was ON.
    /// This proves the enabled path now performs REAL analysis and REMOVES a
    /// duplicate candidate.
    #[test]
    fn test_redundancy_executes_when_env_enabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", "1");

        // Two near-identical, heavily overlapping candidates sharing the same
        // transcript text -> the weaker must be suppressed.
        let mut candidates = vec![draft(10.0, 40.0, 0.90), draft(12.0, 42.0, 0.50)];
        let w = words(&[(10.0, 20.0), (21.0, 30.0), (31.0, 40.0)]);

        run_redundancy_detection("hash", &mut candidates, &w);

        assert!(
            candidates.len() < 2,
            "redundancy must actually remove a duplicate when enabled; got {} candidates",
            candidates.len()
        );
        // The HIGHER-scored candidate must be the survivor.
        assert!(
            (candidates[0].score - 0.90).abs() < 1e-9,
            "the higher-scored candidate must be preserved, got score {}",
            candidates[0].score
        );
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
    }

    /// OFF must be a genuine no-op: no filtering whatsoever.
    #[test]
    fn test_redundancy_is_noop_when_env_disabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", "0");

        let mut candidates = vec![draft(10.0, 40.0, 0.90), draft(12.0, 42.0, 0.50)];
        let w = words(&[(10.0, 20.0), (21.0, 30.0), (31.0, 40.0)]);

        run_redundancy_detection("hash", &mut candidates, &w);

        assert_eq!(
            candidates.len(),
            2,
            "disabled redundancy must not remove anything"
        );
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
    }

    /// Independent candidates must ALL survive when enabled.
    #[test]
    fn test_redundancy_preserves_independent_candidates() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", "1");

        // Far apart in time (no overlap) AND different text.
        let mut candidates = vec![draft(0.0, 20.0, 0.7), draft(500.0, 520.0, 0.7)];
        let w = vec![
            crate::models::TranscriptWord {
                text: "completely different subject matter here".into(),
                start: 1.0,
                end: 5.0,
                speaker: Some("S1".into()),
            },
            crate::models::TranscriptWord {
                text: "another unrelated conversation segment".into(),
                start: 501.0,
                end: 505.0,
                speaker: Some("S1".into()),
            },
        ];

        run_redundancy_detection("hash", &mut candidates, &w);

        assert_eq!(
            candidates.len(),
            2,
            "independent candidates must both be preserved"
        );
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
    }

    /// Determinism / safety: redundancy may only REMOVE candidates. It must
    /// never mutate a timestamp, payoff endpoint, or score of a survivor.
    #[test]
    fn test_redundancy_never_mutates_boundaries() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", "1");

        let mut candidates = vec![
            draft(10.0, 40.0, 0.90),
            draft(12.0, 42.0, 0.50),
            draft(300.0, 330.0, 0.60),
        ];
        let w = words(&[(10.0, 20.0), (21.0, 30.0), (31.0, 40.0)]);

        // Snapshot every (start, end, payoff_end, score) before.
        let before: Vec<(f64, f64, Option<f64>, f64)> = candidates
            .iter()
            .map(|c| (c.start, c.end, c.payoff_end, c.score))
            .collect();

        run_redundancy_detection("hash", &mut candidates, &w);

        // Every surviving candidate must match its pre-call values exactly.
        for c in &candidates {
            let hit = before.iter().any(|(s, e, p, sc)| {
                (*s - c.start).abs() < 1e-9
                    && (*e - c.end).abs() < 1e-9
                    && *p == c.payoff_end
                    && (*sc - c.score).abs() < 1e-9
            });
            assert!(
                hit,
                "surviving candidate was mutated: start={} end={} payoff_end={:?} score={}",
                c.start, c.end, c.payoff_end, c.score
            );
        }
        assert!(
            candidates.len() <= before.len(),
            "redundancy may only remove candidates, never add them"
        );
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
    }

    /// An empty candidate set must be safe.
    #[test]
    fn test_redundancy_handles_empty_candidate_set() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", "1");
        let mut candidates: Vec<crate::models::CandidateDraft> = Vec::new();
        run_redundancy_detection("hash", &mut candidates, &[]);
        assert!(candidates.is_empty());
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
    }

    #[test]
    fn test_redundancy_flag_semantics() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
        assert!(
            !candidate_redundancy_enabled(),
            "default must be disabled (opt-in)"
        );
        for on in ["1", "true", "on"] {
            std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", on);
            assert!(candidate_redundancy_enabled(), "{} must enable", on);
        }
        for off in ["0", "false", "off", "OFF"] {
            std::env::set_var("AUTOSHORTS_CANDIDATE_REDUNDANCY", off);
            assert!(!candidate_redundancy_enabled(), "{} must disable", off);
        }
        std::env::remove_var("AUTOSHORTS_CANDIDATE_REDUNDANCY");
    }

    #[test]
    fn test_redundancy_config_default() {
        let config = RedundancyConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.similarity_threshold, 0.85);
        assert_eq!(config.min_time_gap_sec, 2.0);
        assert!(config.preserve_higher_scored);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((RedundancyEngine::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((RedundancyEngine::cosine_similarity(&a, &b) - 0.0).abs() < 0.001);

        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.0];
        let _norm = 2.0_f32.sqrt();
        assert!((RedundancyEngine::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temporal_overlap() {
        let config = RedundancyConfig::default();
        let engine = RedundancyEngine::new(config).unwrap();

        let a = CandidateEmbedding {
            candidate_id: "a".to_string(),
            embedding: vec![],
            source_hash: "test".to_string(),
            start_sec: 10.0,
            end_sec: 20.0,
            model: "test".to_string(),
            model_version: "1.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let b = CandidateEmbedding {
            candidate_id: "b".to_string(),
            embedding: vec![],
            source_hash: "test".to_string(),
            start_sec: 15.0,
            end_sec: 25.0,
            model: "test".to_string(),
            model_version: "1.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };
        let c = CandidateEmbedding {
            candidate_id: "c".to_string(),
            embedding: vec![],
            source_hash: "test".to_string(),
            start_sec: 30.0,
            end_sec: 40.0,
            model: "test".to_string(),
            model_version: "1.0".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };

        assert!(engine.temporal_overlap(&a, &b)); // Overlap 15-20 = 5s > 2s
        assert!(!engine.temporal_overlap(&a, &c)); // No overlap
        assert!(!engine.temporal_overlap(&b, &c)); // No overlap
    }
}
