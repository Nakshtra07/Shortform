//! AutoShorts 11.0 Phase 4 — ADVISORY VLM Candidate Scoring via NVIDIA Nemotron
//!
//! Optional Vision-Language Model integration for richer candidate quality signals.
//! The VLM is ADVISORY ONLY — it provides additional metadata/scores that feed into
//! the existing deterministic ranking pipeline. It NEVER produces authoritative
//! candidate boundaries, payoff endpoints, or framing decisions.
//!
//! Architecture (RUNTIME-WIRED as of Phase 4 production wiring):
//!   CandidateDraft (from existing LLM discovery)
//!         ↓
//!   OpenRouter VLM Scoring (optional, flag-gated, cached, per-candidate)
//!         ↓
//!   Advisory scores attached to CandidateDraft.vlm_* (metadata_json persistence)
//!         ↓
//!   Existing deterministic ranking / boundary / payoff system (UNCHANGED)
//!
//! Authority contract (must never be violated):
//!   - The VLM never writes candidate.start / candidate.end / payoff_end.
//!   - The VLM never participates in `composite_score`, boundary snapping, framing,
//!     crop, containment, A/V sync, or loudness.
//!   - Every VLM failure path degrades to "no advisory scores" and NEVER fails
//!     candidate generation or rendering.
//!
//! Production wiring lives in `lib.rs::generate_candidates` (post-discovery,
//! pre-persistence) and `vlm_scoring::enhance_candidates_with_vlm`.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// VLM scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VlmScoringConfig {
    /// Enable/disable VLM scoring. This is the *structural* switch; the
    /// environment flag `AUTOSHORTS_VLM_SCORING` is the *runtime* switch.
    pub enabled: bool,
    /// Model identifier. The active provider is NVIDIA Nemotron 3 Nano Omni;
    /// the previous OpenRouter/Qwen default was removed.
    pub model: String,
    /// Model version string for cache invalidation
    pub model_version: String,
    /// Maximum frames to sample per candidate
    pub max_frames: usize,
    /// Inference timeout (seconds) — ENFORCED by the caller with a watchdog.
    pub timeout_sec: u64,
    /// Cache directory
    pub cache_dir: Option<String>,
    /// Prompt version for cache invalidation
    pub prompt_version: String,
}

impl Default for VlmScoringConfig {
    fn default() -> Self {
        Self {
            // Structurally enabled. The previous `enabled: false` default made
            // `enhance_candidates_with_vlm` a guaranteed no-op even when the
            // environment flag was set: it built `VlmScoringConfig::default()`
            // and `is_enabled()` then returned false. The runtime kill-switch
            // (`AUTOSHORTS_VLM_SCORING`) still defaults to OFF, so shipping
            // behavior is unchanged — but the double-disable is gone.
            enabled: true,
            model: "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning".to_string(),
            model_version: "1.0".to_string(),
            max_frames: 8,
            timeout_sec: 120,
            cache_dir: None,
            prompt_version: "v1.0".to_string(),
        }
    }
}

/// VLM score result for a single candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VlmCandidateScore {
    /// Stable candidate key this score applies to (see `candidate_key`).
    /// Results are associated by this key, never by vector position.
    pub candidate_id: String,
    /// Overall VLM quality score (0.0–1.0)
    pub quality_score: f64,
    /// Visual engagement score (0.0–1.0)
    pub visual_engagement: f64,
    /// Semantic coherence score (0.0–1.0)
    pub semantic_coherence: f64,
    /// Production quality score (0.0–1.0)
    pub production_quality: f64,
    /// Highlight relevance score (0.0–1.0)
    pub highlight_relevance: f64,
    /// Model that produced this score
    pub model: String,
    /// Model version
    pub model_version: String,
    /// Prompt version used
    pub prompt_version: String,
    /// Source video hash
    pub source_hash: String,
    /// Timestamp when score was generated
    pub scored_at: String,
    /// Evidence / reasoning from VLM
    pub evidence: Vec<String>,
    /// True when the score came from the deterministic heuristic fallback
    /// (API unavailable / inference failed) rather than real model inference.
    #[serde(default)]
    pub heuristic_fallback: bool,
    /// Raw VLM response (for debugging)
    #[serde(default)]
    pub raw_response: Option<String>,
}

/// Stable per-candidate identity used to associate VLM results with drafts.
///
/// Derived from the project's own time range so it is deterministic across runs
/// and independent of vector position. A missing/failed/skipped result for one
/// candidate therefore cannot shift any other candidate's score.
pub fn candidate_key(project_id: &str, start_sec: f64, end_sec: f64) -> String {
    format!("{}_{:.3}_{:.3}", project_id, start_sec, end_sec)
}

/// VLM scoring engine
/// Forward sidecar stderr to the backend window without leaking a credential.
///
/// The VLM sidecar never prints its key, but this is a hard boundary: any
/// `Authorization`/`Bearer` token that ever appeared in a future sidecar log
/// line is redacted before it reaches the terminal, an artifact, or a report.
fn sanitize_vlm_log(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("authorization") || lower.contains("bearer ") {
            // Preserve the shape of the log line, never the secret.
            let redacted = if let Some(pos) = lower.find("bearer ") {
                format!("{}[REDACTED]", &line[..pos + 7])
            } else {
                "[REDACTED authorization header]".to_string()
            };
            out.push_str(&redacted);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

pub struct VlmScoringEngine {
    config: VlmScoringConfig,
    cache_dir: PathBuf,
}

impl VlmScoringEngine {
    pub fn new(config: VlmScoringConfig) -> Result<Self> {
        let cache_dir = config
            .cache_dir
            .as_ref()
            .map(|s| PathBuf::from(s))
            .unwrap_or_else(|| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("autoshorts")
                    .join("vlm_scoring")
            });
        std::fs::create_dir_all(&cache_dir).context("creating VLM scoring cache dir")?;
        Ok(Self { config, cache_dir })
    }

    /// Compute source hash for cache key.
    ///
    /// NOTE: this hashes the whole file, which is expensive for large sources.
    /// Callers should compute it once per batch (as `score_candidates` does).
    pub fn compute_source_hash(&self, source_path: &str) -> Result<String> {
        let file = std::fs::File::open(source_path).context("opening source for hashing")?;
        let mut reader = std::io::BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = reader
                .read(&mut buffer)
                .context("reading source for hash")?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Compute cache key for a candidate
    fn cache_key(&self, source_hash: &str, candidate_id: &str) -> String {
        self.cache_key_with(source_hash, candidate_id, self.config.model.as_str())
    }

    /// Cache key with an explicit model override, so a model change provably
    /// produces a cache miss.
    fn cache_key_with(&self, source_hash: &str, candidate_id: &str, model: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(source_hash.as_bytes());
        hasher.update(candidate_id.as_bytes());
        hasher.update(model.as_bytes());
        hasher.update(self.config.model_version.as_bytes());
        hasher.update(self.config.prompt_version.as_bytes());
        format!("{:x}", hasher.finalize())
    }

        /// Resolve the provider credential.
    ///
    /// The active provider is NVIDIA (`NVIDIA_API_KEY`). The OpenRouter
    /// spellings are still accepted as a backward-compatible fallback so an
    /// existing install does not hard-break on upgrade, but they are never the
    /// primary path. Returns `None` when nothing is set so the caller reports a
    /// concrete reason instead of treating "" as a usable key.
    pub(crate) fn resolve_api_key(&self) -> Option<String> {
        [
            "NVIDIA_API_KEY",
            "OPENROUTER_API_KEY",
            "OPEN_ROUTER_API_KEY",
        ]
        .iter()
        .find_map(|n| std::env::var(n).ok().filter(|v| !v.trim().is_empty()))
    }

    /// Cache location for a key. Used by the tests to assert that a poisoned
    /// entry is evicted, and by `load_cache`/`save_cache`.
    pub(crate) fn cache_path_for(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", key))
    }

    /// Load cached VLM score
    fn load_cache(&self, key: &str) -> Option<VlmCandidateScore> {
        let path = self.cache_path_for(key);
        if !path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(&path).ok()?;
        let score: VlmCandidateScore = serde_json::from_str(&content).ok()?;
        // A heuristic-fallback entry is NOT a real VLM result. Serving it on a
        // later run would pin the install to fallback forever: the key does not
        // include whether inference actually happened, so a single degraded run
        // (no key, 429, timeout) would permanently mask a working
        // configuration. Fallback entries are dropped and recomputed.
        if score.heuristic_fallback {
            eprintln!(
                "[VLM Scoring] discarding cached heuristic-fallback entry for {} \
                 (not a real inference result)",
                key
            );
            let _ = std::fs::remove_file(&path);
            return None;
        }
        Some(score)
    }

    /// Save VLM score to cache
    fn save_cache(&self, key: &str, score: &VlmCandidateScore) -> Result<()> {
        let path = self.cache_path_for(key);
        std::fs::write(&path, serde_json::to_string_pretty(score)?)
            .context("writing VLM scoring cache")?;
        Ok(())
    }

    /// Runtime enable check: structural config AND environment kill-switch.
    ///
    /// `AUTOSHORTS_VLM_SCORING` defaults to OFF (opt-in). When the env var is
    /// absent the config flag decides; when present it is authoritative.
    pub fn is_enabled(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        match std::env::var("AUTOSHORTS_VLM_SCORING") {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            ),
            Err(_) => self.config.enabled,
        }
    }

    /// Score a batch of candidates using the OpenRouter VLM.
    ///
    /// Returns a map keyed by `candidate_key(project_id, start, end)` so callers
    /// associate by stable identity. Candidates that fail, time out, or are
    /// skipped are simply absent from the map — they never shift other results.
    pub fn score_candidates_by_key(
        &self,
        project_id: &str,
        source_path: &str,
        candidates: &[crate::models::CandidateDraft],
        transcript_words: &[crate::models::TranscriptWord],
    ) -> HashMap<String, VlmCandidateScore> {
        let mut out: HashMap<String, VlmCandidateScore> = HashMap::new();

        if !self.is_enabled() {
            return out;
        }
        if candidates.is_empty() {
            return out;
        }

        let source_hash = match self.compute_source_hash(source_path) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[VLM Scoring] Failed to compute source hash: {}", e);
                return out;
            }
        };

        for candidate in candidates {
            let key = candidate_key(project_id, candidate.start, candidate.end);
            let cache_key = self.cache_key(&source_hash, &key);

            if let Some(cached) = self.load_cache(&cache_key) {
                if cached.model == self.config.model
                    && cached.model_version == self.config.model_version
                    && cached.prompt_version == self.config.prompt_version
                    && cached.candidate_id == key
                {
                    println!("[VLM Scoring] Cache hit for candidate {}", key);
                    out.insert(key, cached);
                    continue;
                }
            }

            match self.score_single_candidate(
                source_path,
                &source_hash,
                candidate,
                transcript_words,
                &key,
            ) {
                Ok(mut score) => {
                    // The sidecar leaves sourceHash blank; stamp it authoritatively.
                    score.source_hash = source_hash.clone();
                    score.candidate_id = key.clone();
                    if let Err(e) = self.save_cache(&cache_key, &score) {
                        eprintln!("[VLM Scoring] Cache write failed: {}", e);
                    }
                    out.insert(key, score);
                }
                Err(e) => {
                    // VLM failure never blocks the pipeline — this candidate
                    // simply carries no advisory scores.
                    eprintln!("[VLM Scoring] Failed to score candidate {}: {}", key, e);
                }
            }
        }

        out
    }

    /// Backwards-compatible wrapper returning a positional Vec.
    ///
    /// The returned Vec is ALWAYS the same length and order as `candidates`;
    /// entries are `None` where scoring failed. Prefer
    /// `score_candidates_by_key` for new call sites.
    pub fn score_candidates(
        &self,
        project_id: &str,
        source_path: &str,
        candidates: &[crate::models::CandidateDraft],
        transcript_words: &[crate::models::TranscriptWord],
    ) -> Vec<Option<VlmCandidateScore>> {
        let map =
            self.score_candidates_by_key(project_id, source_path, candidates, transcript_words);
        candidates
            .iter()
            .map(|c| map.get(&candidate_key(project_id, c.start, c.end)).cloned())
            .collect()
    }

    /// Score a single candidate by invoking the VLM sidecar (which calls OpenRouter).
    ///
    /// Enforces `config.timeout_sec` with a watchdog: the child is killed when it
    /// exceeds the budget. The previous implementation used `Command::output()`
    /// with no timeout, so a hung API call could stall candidate generation
    /// indefinitely.
    fn score_single_candidate(
        &self,
        source_path: &str,
        _source_hash: &str,
        candidate: &crate::models::CandidateDraft,
        transcript_words: &[crate::models::TranscriptWord],
        key: &str,
    ) -> Result<VlmCandidateScore> {
        let script = self.find_vlm_sidecar()?;
        let python = crate::media::find_python_cmd();

        // Provide the candidate's own transcript excerpt (not the whole
        // transcript). The sidecar filters by range, but sending only the
        // relevant window keeps the payload small and deterministic.
        let excerpt: Vec<crate::models::TranscriptWord> = transcript_words
            .iter()
            .filter(|w| w.end > candidate.start && w.start < candidate.end)
            .cloned()
            .collect();

        let candidate_json = serde_json::to_string(candidate)?;
        let words_json = serde_json::to_string(&excerpt)?;

        let candidate_tmp =
            std::env::temp_dir().join(format!("autoshorts_vlm_cand_{}.json", uuid::Uuid::new_v4()));
        let words_tmp = std::env::temp_dir().join(format!(
            "autoshorts_vlm_words_{}.json",
            uuid::Uuid::new_v4()
        ));

        std::fs::write(&candidate_tmp, candidate_json)?;
        std::fs::write(&words_tmp, words_json)?;

        let start_ms = (candidate.start * 1000.0) as i64;
        let end_ms = (candidate.end * 1000.0) as i64;

        // NVIDIA is the active provider. A missing key degrades to a heuristic
        // score rather than an error, so the reason is logged explicitly and
        // the result is marked heuristic_fallback (and never cached as real).
        let api_key = self.resolve_api_key().unwrap_or_default();
        if api_key.is_empty() {
            eprintln!(
                "[VLM Scoring] FALLBACK reason=no API key found \
                 (tried NVIDIA_API_KEY) -> real inference unavailable"
            );
        }
        let frame_budget = self.config.max_frames.to_string();

        // RAII cleanup so an early return / spawn failure cannot leak temp files.
        let _cleanup = TempFiles(vec![candidate_tmp.clone(), words_tmp.clone()]);

        let mut child = Command::new(&python)
            .arg(&script)
            .arg(source_path)
            .arg(start_ms.to_string())
            .arg(end_ms.to_string())
            .arg(&candidate_tmp)
            .arg(&words_tmp)
            .arg("--candidate-key")
            .arg(key)
            .arg("--max-frames")
            .arg(&frame_budget)
            .env("AUTOSHORTS_VLM_MODEL", &self.config.model)
            .env("AUTOSHORTS_VLM_MODEL_VERSION", &self.config.model_version)
            .env("AUTOSHORTS_VLM_PROMPT_VERSION", &self.config.prompt_version)
            .env(
                "AUTOSHORTS_VLM_TIMEOUT_SEC",
                self.config.timeout_sec.to_string(),
            )
            .env("NVIDIA_API_KEY", api_key)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning VLM sidecar")?;

        // Enforced timeout: poll for completion and kill the child if it overruns.
        let budget = Duration::from_secs(self.config.timeout_sec.max(1));
        let started = Instant::now();
        let status = loop {
            match child.try_wait().context("polling VLM sidecar")? {
                Some(status) => break status,
                None => {
                    if started.elapsed() >= budget {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(anyhow!(
                            "VLM sidecar exceeded timeout of {}s",
                            self.config.timeout_sec
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        };

        let output = child
            .wait_with_output()
            .context("collecting VLM sidecar output")?;

        if !status.success() && !output.stdout.is_empty() {
            // A sidecar that printed a result but exited non-zero is still
            // usable; otherwise surface stderr.
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.lines().any(|l| l.trim_start().starts_with('{')) {
                return parse_sidecar_stdout(&stdout);
            }
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("VLM sidecar failed: {}", stderr.trim()));
        }

        if !status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("[VLM] {}", sanitize_vlm_log(&stderr));
            return Err(anyhow!("VLM sidecar failed: {}", stderr.trim()));
        }

        // Backend visibility REQUIREMENT: the sidecar writes the real model
        // response and its structured parse to stderr (stdout is reserved for
        // the single machine-readable JSON document). Forward it so the
        // developer can inspect what the model actually said, including
        // FALLBACK reasons. Never printed on the failure path above twice.
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            println!("{}", sanitize_vlm_log(&stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        parse_sidecar_stdout(&stdout)
    }

    /// Find the VLM sidecar script
    fn find_vlm_sidecar(&self) -> Result<PathBuf> {
        let script_name = "vlm_scoring.py";
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let candidate = parent.join(script_name);
                if candidate.exists() {
                    return Ok(candidate);
                }
                let candidate = parent.join("scripts").join(script_name);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
        let dev_paths = [
            PathBuf::from("src-tauri/scripts").join(script_name),
            PathBuf::from("scripts").join(script_name),
            PathBuf::from("../src-tauri/scripts").join(script_name),
            PathBuf::from("autoshorts/src-tauri/scripts").join(script_name),
        ];
        for p in &dev_paths {
            if p.exists() {
                return Ok(p.clone());
            }
        }
        Err(anyhow!("vlm_scoring.py not found"))
    }
}

/// Deletes the listed temp files on drop (RAII cleanup).
struct TempFiles(Vec<PathBuf>);

impl Drop for TempFiles {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Parse the sidecar's stdout contract: the LAST non-`[` line is the JSON score.
fn parse_sidecar_stdout(stdout: &str) -> Result<VlmCandidateScore> {
    let last = stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('['))
        .last()
        .ok_or_else(|| anyhow!("no JSON output from VLM sidecar"))?;

    // Reject the sidecar's structured error payloads explicitly so they surface
    // as errors instead of deserializing into an empty/garbage score.
    if last.contains("\"status\"") && last.contains("\"error\"") {
        return Err(anyhow!("VLM sidecar reported an error payload"));
    }

    serde_json::from_str(last).context("parsing VLM sidecar output")
}

/// Feature flag for VLM scoring (runtime kill-switch; default OFF / opt-in).
pub fn vlm_scoring_enabled() -> bool {
    match std::env::var("AUTOSHORTS_VLM_SCORING") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => false, // Default OFF — opt-in
    }
}

/// Production integration point: attach advisory VLM scores to candidates.
///
/// Called from `lib.rs::generate_candidates` AFTER deterministic discovery,
/// ranking, and boundary resolution, and BEFORE persistence. It mutates ONLY
/// `vlm_*` fields. It never touches timestamps, payoff endpoints, scores used
/// for ranking, or framing.
///
/// Every failure path is a silent no-op: VLM problems must never fail
/// candidate generation.
pub fn enhance_candidates_with_vlm(
    project_id: &str,
    source_path: &str,
    candidates: &mut [crate::models::CandidateDraft],
    transcript_words: &[crate::models::TranscriptWord],
) {
    if !vlm_scoring_enabled() {
        return;
    }
    if candidates.is_empty() {
        return;
    }

    let config = VlmScoringConfig::default();
    let engine = match VlmScoringEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[VLM Scoring] Engine init failed (non-fatal): {}", e);
            return;
        }
    };

    let scores =
        engine.score_candidates_by_key(project_id, source_path, candidates, transcript_words);

    // Associate by STABLE KEY, never by vector position, so a failed or
    // cache-skipped candidate cannot shift every subsequent result.
    let mut attached = 0usize;
    for candidate in candidates.iter_mut() {
        let key = candidate_key(project_id, candidate.start, candidate.end);
        let Some(score) = scores.get(&key) else {
            continue;
        };

        candidate.vlm_quality_score = Some(score.quality_score);
        candidate.vlm_visual_engagement = Some(score.visual_engagement);
        candidate.vlm_semantic_coherence = Some(score.semantic_coherence);
        candidate.vlm_production_quality = Some(score.production_quality);
        candidate.vlm_highlight_relevance = Some(score.highlight_relevance);
        candidate.vlm_model = Some(score.model.clone());
        candidate.vlm_model_version = Some(score.model_version.clone());
        candidate.vlm_scored_at = Some(score.scored_at.clone());
        candidate.vlm_evidence = Some(score.evidence.clone());
        attached += 1;
    }

    println!(
        "[VLM Scoring] attached advisory scores to {}/{} candidates (heuristic fallback: {})",
        attached,
        candidates.len(),
        scores.values().filter(|s| s.heuristic_fallback).count()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(start: f64, end: f64) -> crate::models::CandidateDraft {
        crate::models::CandidateDraft {
            start,
            end,
            hook: "a hook".to_string(),
            score: 0.7,
            ..Default::default()
        }
    }

    fn word(text: &str, start: f64, end: f64) -> crate::models::TranscriptWord {
        crate::models::TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: Some("S1".to_string()),
        }
    }

    #[test]
    fn test_vlm_flag_semantics() {
        // These tests share a process-global env var, so they must not run
        // concurrently or they will observe each other's mutations.
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
        assert!(!vlm_scoring_enabled(), "default must be disabled (opt-in)");
        for on in ["1", "true", "on"] {
            std::env::set_var("AUTOSHORTS_VLM_SCORING", on);
            assert!(vlm_scoring_enabled(), "{} must enable", on);
        }
        for off in ["0", "false", "off", "OFF"] {
            std::env::set_var("AUTOSHORTS_VLM_SCORING", off);
            assert!(!vlm_scoring_enabled(), "{} must disable", off);
        }
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    /// Serializes tests that mutate `AUTOSHORTS_VLM_SCORING`.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_vlm_config_default_is_structurally_enabled() {
        // Regression guard for the "double-disable" bug: the config default must
        // be structurally enabled so the env flag alone decides at runtime.
        let config = VlmScoringConfig::default();
        assert!(
            config.enabled,
            "config must default to structurally enabled"
        );
        assert_eq!(config.model, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning");
        assert_eq!(config.max_frames, 8);
        assert_eq!(config.timeout_sec, 120);
    }

    #[test]
    fn test_vlm_engine_env_flag_alone_controls_runtime() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // With the env flag ON, the engine must actually be enabled now
        // (previously impossible because of the double-disable).
        std::env::set_var("AUTOSHORTS_VLM_SCORING", "1");
        let engine = VlmScoringEngine::new(VlmScoringConfig::default()).unwrap();
        assert!(engine.is_enabled(), "env flag ON must enable the engine");

        std::env::set_var("AUTOSHORTS_VLM_SCORING", "off");
        assert!(!engine.is_enabled(), "env flag OFF must disable the engine");
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    #[test]
    fn test_disabled_engine_returns_no_results() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_VLM_SCORING", "0");
        let engine = VlmScoringEngine::new(VlmScoringConfig::default()).unwrap();
        let drafts = vec![draft(0.0, 10.0)];
        let map = engine.score_candidates_by_key("p1", "does-not-exist.mp4", &drafts, &[]);
        assert!(map.is_empty(), "disabled engine must produce no scores");
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    #[test]
    fn test_empty_candidate_set_returns_empty() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_VLM_SCORING", "1");
        let engine = VlmScoringEngine::new(VlmScoringConfig::default()).unwrap();
        let map = engine.score_candidates_by_key("p1", "nope.mp4", &[], &[]);
        assert!(map.is_empty());
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    #[test]
    fn test_candidate_key_is_stable_and_position_independent() {
        let a = candidate_key("proj", 12.5, 42.25);
        let b = candidate_key("proj", 12.5, 42.25);
        let c = candidate_key("proj", 12.5, 42.26);
        assert_eq!(a, b, "same range must yield the same key");
        assert_ne!(a, c, "different range must yield a different key");
        assert_ne!(
            candidate_key("other", 12.5, 42.25),
            a,
            "different project must yield a different key"
        );
    }

    #[test]
    fn test_parse_sidecar_stdout_success() {
        let stdout = "[VLM] progress line\n[frame] 1/8\n{\"candidateId\":\"k\",\"qualityScore\":0.8,\"visualEngagement\":0.7,\"semanticCoherence\":0.6,\"productionQuality\":0.9,\"highlightRelevance\":0.5,\"model\":\"m\",\"modelVersion\":\"1.0\",\"promptVersion\":\"v1.0\",\"sourceHash\":\"h\",\"scoredAt\":\"t\",\"evidence\":[\"e\"],\"heuristicFallback\":false}";
        let score = parse_sidecar_stdout(stdout).expect("valid payload parses");
        assert_eq!(score.candidate_id, "k");
        assert!((score.quality_score - 0.8).abs() < 1e-9);
        assert_eq!(score.evidence.len(), 1);
        assert!(!score.heuristic_fallback);
    }

    #[test]
    fn test_parse_sidecar_stdout_malformed_json_is_error() {
        // Truncated / malformed JSON must be an Err, never a default score.
        let bad = "[VLM] log\n{\"candidateId\":\"k\",\"qualityScore\":0.8,";
        assert!(
            parse_sidecar_stdout(bad).is_err(),
            "malformed JSON must error"
        );
    }

    #[test]
    fn test_parse_sidecar_stdout_error_payload_is_error() {
        let err = "[VLM] log\n{\"status\":\"error\",\"reason\":\"api failure\"}";
        assert!(
            parse_sidecar_stdout(err).is_err(),
            "structured error payload must surface as Err"
        );
    }

    #[test]
    fn test_parse_sidecar_stdout_empty_is_error() {
        assert!(parse_sidecar_stdout("").is_err());
        assert!(parse_sidecar_stdout("[only] [logs]").is_err());
    }

    #[test]
    fn test_cache_roundtrip_and_model_mismatch_is_miss() {
        let engine = VlmScoringEngine::new(VlmScoringConfig {
            cache_dir: Some(
                std::env::temp_dir()
                    .join(format!("autoshorts_vlm_test_{}", uuid::Uuid::new_v4()))
                    .to_string_lossy()
                    .into_owned(),
            ),
            ..Default::default()
        })
        .unwrap();

        let key = engine.cache_key("srchash", "candkey");
        assert!(engine.load_cache(&key).is_none(), "empty cache must miss");

        let score = VlmCandidateScore {
            candidate_id: "candkey".to_string(),
            quality_score: 0.42,
            visual_engagement: 0.5,
            semantic_coherence: 0.5,
            production_quality: 0.5,
            highlight_relevance: 0.5,
            model: engine.config.model.clone(),
            model_version: engine.config.model_version.clone(),
            prompt_version: engine.config.prompt_version.clone(),
            source_hash: "srchash".to_string(),
            scored_at: "2026-01-01T00:00:00Z".to_string(),
            evidence: vec!["ok".to_string()],
            heuristic_fallback: false,
            raw_response: None,
        };
        engine.save_cache(&key, &score).unwrap();

        let hit = engine.load_cache(&key).expect("cache hit after save");
        assert_eq!(hit.candidate_id, "candkey");
        assert!((hit.quality_score - 0.42).abs() < 1e-9);

        // A different model version must produce a different cache key (miss).
        let other = engine.cache_key_with("srchash", "candkey", "other-model");
        assert!(
            engine.load_cache(&other).is_none(),
            "model change must miss"
        );
    }

    #[test]
    fn test_enhance_is_noop_when_disabled_and_preserves_deterministic_fields() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_VLM_SCORING", "0");
        let mut drafts = vec![draft(10.0, 40.0)];
        let before_start = drafts[0].start;
        let before_end = drafts[0].end;
        let before_score = drafts[0].score;

        enhance_candidates_with_vlm("p1", "nope.mp4", &mut drafts, &[]);

        // No advisory fields set...
        assert!(drafts[0].vlm_quality_score.is_none());
        // ...and, critically, no deterministic field was touched.
        assert_eq!(drafts[0].start, before_start);
        assert_eq!(drafts[0].end, before_end);
        assert_eq!(drafts[0].score, before_score);
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    #[test]
    fn test_enhance_on_empty_set_is_safe() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_VLM_SCORING", "1");
        let mut drafts: Vec<crate::models::CandidateDraft> = Vec::new();
        enhance_candidates_with_vlm("p1", "nope.mp4", &mut drafts, &[]);
        assert!(drafts.is_empty());
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    #[test]
    fn test_excerpt_filtering_selects_only_candidate_range() {
        // The Rust side must hand the sidecar only the candidate's own words.
        let words = vec![
            word("before", 0.0, 5.0),
            word("inside", 12.0, 13.0),
            word("also", 30.0, 31.0),
            word("after", 45.0, 46.0),
        ];
        let excerpt: Vec<_> = words
            .iter()
            .filter(|w| w.end > 10.0 && w.start < 40.0)
            .cloned()
            .collect();
        assert_eq!(excerpt.len(), 2, "only the in-range words are sent");
        assert_eq!(excerpt[0].text, "inside");
    }

    #[test]
    fn test_positional_wrapper_preserves_alignment_on_partial_failure() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // score_candidates must return same-length Vec<Option<..>> aligned to
        // the input order, so a partial failure cannot shift results.
        std::env::set_var("AUTOSHORTS_VLM_SCORING", "1");
        let engine = VlmScoringEngine::new(VlmScoringConfig::default()).unwrap();
        let drafts = vec![draft(0.0, 10.0), draft(20.0, 30.0)];
        // Source does not exist -> every candidate fails; alignment must hold.
        let aligned = engine.score_candidates("p1", "definitely-missing.mp4", &drafts, &[]);
        assert_eq!(aligned.len(), drafts.len(), "length must match input");
        assert!(aligned.iter().all(|v| v.is_none()));
        std::env::remove_var("AUTOSHORTS_VLM_SCORING");
    }

    // ── REGRESSION: a degraded run must not pin VLM to fallback forever ────
    //
    // The cache key is (source, candidate, model, model_version,
    // prompt_version) and does NOT include whether inference actually ran. A
    // single heuristic-fallback run therefore wrote an entry that every
    // subsequent run accepted as a cache hit — so a properly credentialed
    // install kept reporting "heuristic fallback" forever.

    /// UNIQUE per test name: cargo runs #[test] fns in parallel threads, so a
    /// shared directory let one test's cache file satisfy another's assertion
    /// (making the fallback-rejection test fail for the wrong reason).
    fn temp_engine(tag: &str) -> VlmScoringEngine {
        let dir = std::env::temp_dir().join(format!(
            "vlm_cache_test_{}_{}",
            std::process::id(),
            tag
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let config = VlmScoringConfig {
            cache_dir: Some(dir.to_string_lossy().to_string()),
            ..VlmScoringConfig::default()
        };
        VlmScoringEngine::new(config).expect("engine")
    }

    fn sample_score(candidate_id: &str, fallback: bool) -> VlmCandidateScore {
        let d = VlmScoringConfig::default();
        VlmCandidateScore {
            candidate_id: candidate_id.to_string(),
            quality_score: 0.8,
            visual_engagement: 0.7,
            semantic_coherence: 0.6,
            production_quality: 0.9,
            highlight_relevance: 0.5,
            model: d.model,
            model_version: d.model_version,
            prompt_version: d.prompt_version,
            source_hash: "h".into(),
            scored_at: "2026-01-01T00:00:00Z".into(),
            evidence: vec!["e".into()],
            heuristic_fallback: fallback,
            raw_response: None,
        }
    }

    #[test]
    fn cached_heuristic_fallback_entry_is_never_served() {
        let engine = temp_engine("fallback");
        let key = engine.cache_key("HASH", "cand_1");
        engine
            .save_cache(&key, &sample_score("cand_1", true))
            .unwrap();

        assert!(
            engine.load_cache(&key).is_none(),
            "a heuristic-fallback entry must not satisfy a later live request"
        );
        // And it is evicted, so it cannot be resurrected by a retry.
        assert!(
            !engine.cache_path_for(&key).exists(),
            "the poisoned entry must be removed, not merely ignored"
        );
        let _ = std::fs::remove_dir_all(&engine.cache_dir);
    }

    #[test]
    fn cached_real_inference_entry_is_served() {
        let engine = temp_engine("real");
        let key = engine.cache_key("HASH", "cand_1");
        engine
            .save_cache(&key, &sample_score("cand_1", false))
            .unwrap();

        let hit = engine
            .load_cache(&key)
            .expect("a real inference result must be cacheable");
        assert!(!hit.heuristic_fallback);
        assert_eq!(hit.candidate_id, "cand_1");
        let _ = std::fs::remove_dir_all(&engine.cache_dir);
    }

    #[test]
    fn cache_key_varies_with_model_and_candidate() {
        let engine = temp_engine("keys");
        let base = engine.cache_key("HASH", "cand_1");
        assert_ne!(
            base,
            engine.cache_key("HASH", "cand_2"),
            "different candidates must not share a key"
        );
        assert_ne!(
            base,
            engine.cache_key_with("HASH", "cand_1", "some/other-model:v1"),
            "a model change must invalidate the entry"
        );
        assert_ne!(
            base,
            engine.cache_key("OTHER_HASH", "cand_1"),
            "a different source must not share a key"
        );
    }

    #[test]
    fn openrouter_key_accepts_both_spellings() {
        // Real .env files use OPEN_ROUTER_API_KEY while the historical code
        // read only OPENROUTER_API_KEY. That mismatch silently disabled real
        // inference for an otherwise correctly-provisioned install, because a
        // missing key degrades to a heuristic score instead of erroring.
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let engine = temp_engine("keynames");
        for name in ["OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"] {
            std::env::remove_var("OPENROUTER_API_KEY");
            std::env::remove_var("OPEN_ROUTER_API_KEY");
            std::env::set_var(name, "test-key-value");
            assert_eq!(
                engine.resolve_api_key(),
                Some("test-key-value".to_string()),
                "{} must be accepted",
                name
            );
        }
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("OPEN_ROUTER_API_KEY");
        assert_eq!(
            engine.resolve_api_key(),
            None,
            "absent keys must report None, not an empty string"
        );
        let _ = std::fs::remove_dir_all(&engine.cache_dir);
    }

    #[test]
    fn sanitizer_redacts_authorization_but_keeps_diagnostics() {
        // A future sidecar change must never be able to print the credential.
        let dirty = "Authorization: Bearer sk-super-secret-value\n[VLM] ok";
        let clean = sanitize_vlm_log(dirty);
        assert!(!clean.contains("sk-super-secret-value"), "{}", clean);
        assert!(clean.contains("[REDACTED]"));
        // Useful diagnostics must survive.
        assert!(clean.contains("[VLM] ok"));
    }

    #[test]
    fn sanitizer_leaves_normal_vlm_output_intact() {
        let clean = sanitize_vlm_log("[VLM] INFERENCE COMPLETE elapsed=3.2s");
        assert_eq!(clean, "[VLM] INFERENCE COMPLETE elapsed=3.2s");
    }

    #[test]
    fn active_provider_is_nvidia_nemotron() {
        let d = VlmScoringConfig::default();
        assert_eq!(d.model, "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning");
    }

    #[test]
    fn nvidia_api_key_is_preferred_over_openrouter() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let engine = temp_engine("nvidiapref");
        std::env::remove_var("NVIDIA_API_KEY");
        std::env::remove_var("OPENROUTER_API_KEY");
        std::env::remove_var("OPEN_ROUTER_API_KEY");
        std::env::set_var("NVIDIA_API_KEY", "nv-key");
        std::env::set_var("OPENROUTER_API_KEY", "or-key");
        assert_eq!(engine.resolve_api_key(), Some("nv-key".to_string()));
        // With only the legacy spelling set, it still resolves (no hard break).
        std::env::remove_var("NVIDIA_API_KEY");
        assert_eq!(engine.resolve_api_key(), Some("or-key".to_string()));
        std::env::remove_var("OPENROUTER_API_KEY");
        assert_eq!(engine.resolve_api_key(), None);
        let _ = std::fs::remove_dir_all(&engine.cache_dir);
    }
}
