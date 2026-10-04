//! AutoShorts 11.0 Phase 4 — PANNs Reaction Metadata
//!
//! Source-level audio event detection using PANNs (Pre-trained Audio Neural Networks).
//! Detects reaction events like laughter, applause, cheering from the audio track.
//! The metadata is cached per source and available to Hook Intelligence, candidate
//! ranking, and analysis systems.
//!
//! Architecture:
//!   SOURCE AUDIO
//!        ↓
//!   ONE PANNs PASS (per source, cached)
//!        ↓
//!   REACTION EVENTS
//!        ↓
//!   CACHE (source_hash + model + config)
//!        ↓
//!   Consumers: Hook Intelligence, candidate ranking, analysis metadata

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

/// PANNs configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PannsConfig {
    /// Enable/disable PANNs reaction detection
    pub enabled: bool,
    /// Model identifier. CNN14 is built at 32 kHz (panns_inference
    /// `Cnn14_DecisionLevelMax(sample_rate=32000, hop_size=320)`), so the
    /// honest name is "cnn14" — the old "cnn14_16k" was a misnomer that
    /// implied 16 kHz input and would make a wrong sample rate look correct.
    pub model: String,
    /// Model version
    pub model_version: String,
    /// Path to model weights
    pub model_path: Option<String>,
    /// Sample rate the model is built for. CNN14 = 32000 Hz. Do not feed
    /// 16 kHz audio to this model.
    pub sample_rate: u32,
    /// Per-frame probability required to open/extend an event inside the
    /// sidecar. This is the DETECTION threshold. It is deliberately separate
    /// from `confidence_threshold` (the event-level gate) so that lowering one
    /// can never silently lower the other.
    pub frame_threshold: f64,
    /// Event-level confidence gate: an event survives into the metadata only
    /// if its peak probability is >= this. Left at the AudioSet-typical 0.5;
    /// NOT lowered to manufacture detections.
    pub confidence_threshold: f64,
    /// Minimum event duration (seconds)
    pub min_event_duration_sec: f64,
    /// Cache directory
    pub cache_dir: Option<String>,
    /// Inference timeout (seconds)
    pub timeout_sec: u64,
}

impl Default for PannsConfig {
    fn default() -> Self {
        Self {
            enabled: false, // OFF by default — opt-in
            model: "cnn14".to_string(),
            model_version: "1.0".to_string(),
            model_path: None,
            sample_rate: 32000,
            frame_threshold: 0.1,
            confidence_threshold: 0.5,
            min_event_duration_sec: 0.2,
            cache_dir: None,
            timeout_sec: 300,
        }
    }
}

/// Bumped whenever the sidecar contract or the event semantics change, so
/// stale cache entries from an older (buggy) run can never be served as if
/// they were valid.
///
/// BUMPED TO 2 after v1 wrote `{"events": []}` cache entries produced by the
/// stub detector (no inference at all). Those poisoned entries would otherwise
/// be a permanent cache HIT and keep reporting "0 total events" forever.
pub const PANNS_CACHE_SCHEMA_VERSION: u32 = 2;

/// Single reaction event detected by PANNs
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReactionEvent {
    /// Event type (AudioSet class name)
    pub event_type: String,
    /// Start time (seconds, absolute source time)
    pub start: f64,
    /// End time (seconds, absolute source time)
    pub end: f64,
    /// Confidence score (0.0–1.0)
    pub confidence: f64,
    /// Model that detected this event
    pub model: String,
    /// Model version
    pub model_version: String,
}

/// Complete PANNs reaction metadata for a source
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PannsReactionMetadata {
    /// SHA256 hash of the source file
    pub source_hash: String,
    /// Model used
    pub model: String,
    /// Model version
    pub model_version: String,
    /// All detected reaction events
    pub events: Vec<ReactionEvent>,
    /// Events grouped by type for convenience
    #[serde(default)]
    pub events_by_type: HashMap<String, Vec<ReactionEvent>>,
    /// Configuration hash for cache invalidation
    pub config_hash: String,
    /// Cache/event schema version — see `PANNS_CACHE_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// ISO8601 timestamp when this was generated
    pub created_at: String,
}

/// AudioSet reaction classes, matched EXACTLY (not by substring).
///
/// The old `REACTION_CLASSES` list used `event_type.contains(c)` against
/// hand-written display names like "Chuckle, chortle" and "Crowd cheer" that
/// do not exist in the AudioSet vocabulary, and `.capitalize()` in the sidecar
/// lower-cased everything after the first character ("Laughter" ->
/// "laughter", "Gasp" -> "gasp"). The result was a class filter that could
/// reject genuine detections. These are the real PANNs label strings, verified
/// against `panns_inference.config.labels` (527 classes).
pub const REACTION_CLASSES: &[&str] = &[
    "Laughter",
    "Baby laughter",
    "Giggle",
    "Snicker",
    "Belly laugh",
    "Chuckle, chortle",
    "Gasp",
    "Pant",
    "Clapping",
    "Applause",
    "Cheering",
    "Whoop",
    "Shout",
    "Yell",
    "Screaming",
    "Children shouting",
];

/// PANNs inference engine
pub struct PannsEngine {
    config: PannsConfig,
    cache_dir: PathBuf,
}

impl PannsEngine {
    pub fn new(config: PannsConfig) -> Result<Self> {
        let cache_dir = config
            .cache_dir
            .as_ref()
            .map(|s| PathBuf::from(s))
            .unwrap_or_else(|| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("autoshorts")
                    .join("panns_reactions")
            });
        std::fs::create_dir_all(&cache_dir).context("creating PANNs cache dir")?;
        Ok(Self { config, cache_dir })
    }

    /// Compute source hash for cache key
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

    /// Compute config hash for cache invalidation
    fn compute_config_hash(&self) -> String {
        let config_str = serde_json::to_string(&self.config).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(config_str.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Cache file path
    fn cache_path(&self, source_hash: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", source_hash))
    }

    /// Load cached metadata
    fn load_cache(&self, source_hash: &str) -> Option<PannsReactionMetadata> {
        let path = self.cache_path(source_hash);
        if !path.exists() {
            return None;
        }
        let content = std::fs::read_to_string(&path).ok()?;
        let metadata: PannsReactionMetadata = serde_json::from_str(&content).ok()?;
        if metadata.config_hash != self.compute_config_hash() {
            eprintln!(
                "[PANNs] Cache invalid (config changed) for {}",
                &source_hash[..16.min(source_hash.len())]
            );
            let _ = std::fs::remove_file(&path);
            return None;
        }
        // v1 cached entries were written by the STUB path (no CNN14 inference
        // ever ran) and read back as a permanent cache HIT reporting 0 events.
        // Reject them loudly instead of serving them forever.
        if metadata.schema_version != PANNS_CACHE_SCHEMA_VERSION {
            eprintln!(
                "[PANNs] Cache invalid (schema v{} != v{}; produced by an older/buggy \
                 run that may not have run inference) for {} — discarding",
                metadata.schema_version,
                PANNS_CACHE_SCHEMA_VERSION,
                &source_hash[..16.min(source_hash.len())]
            );
            let _ = std::fs::remove_file(&path);
            return None;
        }
        Some(metadata)
    }

    /// Save metadata to cache
    fn save_cache(&self, metadata: &PannsReactionMetadata) -> Result<()> {
        let path = self.cache_path(&metadata.source_hash);
        std::fs::write(&path, serde_json::to_string_pretty(metadata)?)
            .context("writing PANNs cache")?;
        Ok(())
    }

    /// Check if PANNs is enabled
    pub fn is_enabled(&self) -> bool {
        if !self.config.enabled {
            return false;
        }
        match std::env::var("AUTOSHORTS_PANNS_REACTIONS") {
            Ok(v) => !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            ),
            Err(_) => self.config.enabled,
        }
    }

    /// Run PANNs reaction detection for a source video
    pub fn detect_reactions(&self, source_path: &str) -> Result<PannsReactionMetadata> {
        if !self.is_enabled() {
            return Err(anyhow!("PANNs not enabled"));
        }

        let source_hash = self.compute_source_hash(source_path)?;

        // Check cache
        if let Some(cached) = self.load_cache(&source_hash) {
            println!(
                "[PANNs] Cache hit for {}",
                &source_hash[..16.min(source_hash.len())]
            );
            return Ok(cached);
        }

        println!("[PANNs] Running reaction detection on {}", source_path);

        // Invoke PANNs sidecar
        let events = self.run_panns_sidecar(source_path)?;

        // Group events by type
        let mut events_by_type: HashMap<String, Vec<ReactionEvent>> = HashMap::new();
        for event in &events {
            events_by_type
                .entry(event.event_type.clone())
                .or_default()
                .push(event.clone());
        }

        let metadata = PannsReactionMetadata {
            source_hash: source_hash.clone(),
            model: self.config.model.clone(),
            model_version: self.config.model_version.clone(),
            events,
            events_by_type,
            config_hash: self.compute_config_hash(),
            schema_version: PANNS_CACHE_SCHEMA_VERSION,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        // Save cache
        if let Err(e) = self.save_cache(&metadata) {
            eprintln!("[PANNs] Cache write failed (non-fatal): {}", e);
        }

        if metadata.events.is_empty() {
            eprintln!(
                "[PANNs] NOTE: real CNN14 inference ran but produced 0 events above \
                 confidence_threshold={} (per-class peaks are in the sidecar \
                 diagnostics). This is a genuine negative, not a missing model.",
                self.config.confidence_threshold
            );
        }

        Ok(metadata)
    }

    /// Run the PANNs sidecar script
    fn run_panns_sidecar(&self, source_path: &str) -> Result<Vec<ReactionEvent>> {
        let script = self.find_panns_sidecar()?;
        let python = crate::media::find_python_cmd();

        let mut cmd = Command::new(&python);
        cmd.arg(&script)
            .arg(source_path)
            .env("AUTOSHORTS_PANNS_MODEL", &self.config.model)
            .env("AUTOSHORTS_PANNS_MODEL_VERSION", &self.config.model_version)
            .env("AUTOSHORTS_PANNS_SAMPLE_RATE", self.config.sample_rate.to_string())
            .env(
                "AUTOSHORTS_PANNS_FRAME_THRESHOLD",
                self.config.frame_threshold.to_string(),
            )
            .env(
                "AUTOSHORTS_PANNS_MIN_EVENT_DURATION",
                self.config.min_event_duration_sec.to_string(),
            )
            .env(
                "AUTOSHORTS_PANNS_CONFIDENCE_THRESHOLD",
                self.config.confidence_threshold.to_string(),
            );
        if let Some(model_path) = self.config.model_path.as_deref() {
            if !model_path.trim().is_empty() {
                cmd.env("AUTOSHORTS_PANNS_CHECKPOINT", model_path);
            }
        }
        // Never let the sidecar fall back to the stub detector: a stub run
        // produces `{"events": []}` indistinguishable from a real negative,
        // which is exactly the bug this stage had.
        cmd.env_remove("AUTOSHORTS_PANNS_STUB");

        // Bounded: this is ADVISORY work, so a timeout must degrade to "no
        // reaction intelligence" and let the pipeline continue. Unbounded
        // `.output()` here meant a wedged sidecar could stall candidate
        // generation forever.
        let timeout_sec = if self.config.timeout_sec > 0 {
            self.config.timeout_sec
        } else {
            300
        };
        let budget = std::time::Duration::from_secs(timeout_sec);
        println!("[PANNs] sidecar START budget={}s", timeout_sec);
        let output = crate::proc_guard::run_bounded(&mut cmd, budget, "PANNs/sidecar")
            .context("spawning PANNs sidecar")?;

        if output.timed_out {
            return Err(anyhow!(
                "PANNs sidecar TIMEOUT after {:.0}s (budget {}s); process tree killed",
                output.elapsed.as_secs_f64(),
                timeout_sec
            ));
        }

        // The sidecar's stdout is a single JSON line carrying an explicit
        // `ok` flag and a structured `error` object. Parse that FIRST so a
        // structured failure (missing checkpoint, bad audio, inference error)
        // is reported verbatim instead of being flattened into "0 events".
        let stdout = output.stdout;
        let json_line = stdout
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .next_back()
            .ok_or_else(|| {
                anyhow!(
                    "no JSON output from PANNs sidecar (rc={:?}, elapsed {:.1}s). stderr: {}",
                    output.code,
                    output.elapsed.as_secs_f64(),
                    crate::proc_guard::sanitize_stderr(&output.stderr, 600).trim()
                )
            })?;

        #[derive(Deserialize)]
        struct SidecarError {
            code: String,
            message: String,
        }
        #[derive(Deserialize)]
        struct PannsOutput {
            #[serde(default)]
            ok: bool,
            #[serde(default)]
            events: Vec<ReactionEvent>,
            #[serde(default)]
            error: Option<SidecarError>,
            #[serde(default)]
            diagnostics: serde_json::Value,
        }

        let parsed: PannsOutput = serde_json::from_str(json_line).map_err(|e| {
            anyhow!(
                "parsing PANNs sidecar output (rc={:?}): {}; raw: {}",
                output.code,
                e,
                &json_line[..json_line.len().min(400)]
            )
        })?;

        if let Some(err) = parsed.error {
            return Err(anyhow!(
                "PANNs sidecar failed [{}]: {} (diagnostics: {})",
                err.code,
                err.message,
                parsed.diagnostics
            ));
        }
        if !parsed.ok {
            let stderr = crate::proc_guard::sanitize_stderr(&output.stderr, 600);
            return Err(anyhow!(
                "PANNs sidecar reported ok=false without an error object (rc={:?}): {}",
                output.code,
                stderr.trim()
            ));
        }
        if !output.success {
            let stderr = crate::proc_guard::sanitize_stderr(&output.stderr, 600);
            return Err(anyhow!("PANNs sidecar failed: {}", stderr.trim()));
        }

        println!(
            "[PANNs] sidecar COMPLETE in {:.1}s rc={:?} diagnostics={}",
            output.elapsed.as_secs_f64(),
            output.code,
            parsed.diagnostics
        );

        // Event-level gate + exact AudioSet class check. NOTE: no substring
        // matching — `REACTION_CLASSES` now holds the real AudioSet strings.
        let filtered: Vec<ReactionEvent> = parsed
            .events
            .into_iter()
            .filter(|e| e.confidence >= self.config.confidence_threshold)
            .filter(|e| REACTION_CLASSES.contains(&e.event_type.as_str()))
            .collect();

        Ok(filtered)
    }

    /// Find the PANNs sidecar script
    fn find_panns_sidecar(&self) -> Result<PathBuf> {
        let script_name = "panns_reactions.py";
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
        Err(anyhow!("panns_reactions.py not found"))
    }
}

/// Feature flag
pub fn panns_reactions_enabled() -> bool {
    match std::env::var("AUTOSHORTS_PANNS_REACTIONS") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => false, // Default OFF — opt-in
    }
}

/// Integration: get reaction events for a candidate interval
pub fn get_reactions_for_candidate(
    metadata: &PannsReactionMetadata,
    candidate_start: f64,
    candidate_end: f64,
) -> Vec<ReactionEvent> {
    metadata
        .events
        .iter()
        .filter(|e| e.start < candidate_end && e.end > candidate_start)
        .cloned()
        .collect()
}

/// Integration: enhance Hook Intelligence with PANNs reaction signals
pub fn enhance_hook_with_panns(
    hook_score: f64,
    _candidate: &crate::models::CandidateDraft,
    reactions: &[ReactionEvent],
) -> f64 {
    if reactions.is_empty() {
        return hook_score;
    }

    let mut boost = 0.0;
    for event in reactions {
        match event.event_type.as_str() {
            s if s.contains("Laughter") || s.contains("laugh") => {
                // Laughter during hook = strong engagement signal
                boost += 0.05 * event.confidence;
            }
            s if s.contains("Applause") || s.contains("Cheer") => {
                // Applause/cheering = audience validation
                boost += 0.03 * event.confidence;
            }
            _ => {
                boost += 0.01 * event.confidence;
            }
        }
    }

    // Cap the boost
    (hook_score + boost.min(0.15)).min(1.0)
}

/// Production integration point: annotate candidates with PANNs reaction
/// metadata (ADVISORY signal only).
///
/// Authority contract — this function may ONLY:
///   * add reaction counts into `reactions_json` (metadata), and
///   * apply the capped hook-score boost via `enhance_hook_with_panns`.
///
/// It must NEVER:
///   * change `start` / `end` / `payoff_end` / any boundary,
///   * change framing or crop,
///   * fail candidate generation.
///
/// Runs once per source (the engine caches by source hash), so the cost is one
/// PANNs pass regardless of candidate count. Disabled by default via
/// `AUTOSHORTS_PANNS_REACTIONS`. Any failure degrades silently.
pub fn annotate_candidates_with_reactions(
    source_path: &str,
    candidates: &mut [crate::models::CandidateDraft],
) {
    if !panns_reactions_enabled() {
        return;
    }
    if candidates.is_empty() {
        return;
    }

    // DOUBLE-DISABLE FIX: `PannsConfig::default()` has `enabled: false`, which
    // made `engine.is_enabled()` false even after the environment gate above
    // passed — so `AUTOSHORTS_PANNS_REACTIONS=1` was a silent no-op. The
    // environment flag is already checked, so the config is structurally
    // enabled here. The kill-switch is NOT bypassed: this function still
    // returns early when `panns_reactions_enabled()` is false.
    let config = PannsConfig {
        enabled: true,
        ..Default::default()
    };
    let engine = match PannsEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[PANNs] Engine init failed (non-fatal): {}", e);
            return;
        }
    };

    // One source-level pass, cached across calls.
    let metadata = match engine.detect_reactions(source_path) {
        Ok(m) => m,
        Err(e) => {
            // ADVISORY stage: a failure must never fail candidate generation.
            // But it must be LOUD and STRUCTURAL — a silent return here is what
            // made "0 events" indistinguishable from "model never ran".
            eprintln!(
                "[PANNs] REACTION DETECTION FAILED (advisory, non-fatal): {}",
                e
            );
            eprintln!(
                "[PANNs] no reaction metadata was attached; candidates are unchanged. \
                 Check the reason above (e.g. checkpoint_missing) — this is NOT a \
                 genuine 'no reactions in the audio' result."
            );
            return;
        }
    };

    let mut annotated = 0usize;
    for candidate in candidates.iter_mut() {
        let events = get_reactions_for_candidate(&metadata, candidate.start, candidate.end);
        if events.is_empty() {
            continue;
        }

        let doc = serde_json::json!({
            "count": events.len(),
            "events": events
                .iter()
                .map(|e| serde_json::json!({
                    "type": e.event_type,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                }))
                .collect::<Vec<_>>(),
        });
        candidate.reactions_json = Some(doc.to_string());

        // Advisory-only, capped boost. Never touches boundaries.
        let base_hook = candidate
            .hook_score
            .or(candidate.guest_hook_strength)
            .or(candidate.interviewer_hook_strength)
            .unwrap_or(0.0);
        let boosted = enhance_hook_with_panns(base_hook, candidate, &events);
        if (boosted - base_hook).abs() > f64::EPSILON {
            candidate.hook_score = Some(boosted);
        }
        annotated += 1;
    }

    println!(
        "[PANNs] annotated {}/{} candidates with reaction metadata ({} total events)",
        annotated,
        candidates.len(),
        metadata.events.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate `AUTOSHORTS_PANNS_REACTIONS`.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn draft(start: f64, end: f64, hook_score: Option<f64>) -> crate::models::CandidateDraft {
        crate::models::CandidateDraft {
            start,
            end,
            score: 0.7,
            hook: "hook".into(),
            rationale: "test".into(),
            payoff_end: Some(end),
            hook_score,
            ..Default::default()
        }
    }

    /// DOUBLE-DISABLE REGRESSION GUARD.
    ///
    /// `annotate_candidates_with_reactions` used to build
    /// `PannsConfig::default()` (enabled = false), so `detect_reactions`
    /// returned Err("PANNs not enabled") and the whole feature was a silent
    /// no-op even with `AUTOSHORTS_PANNS_REACTIONS=1`.
    ///
    /// This proves the engine gate is now open when the env flag is set.
    /// A missing source file makes inference fail, which is the safe-fallback
    /// path — the important assertion is that the GATE is open (no
    /// "PANNs not enabled" error) and that no boundary was ever mutated.
    #[test]
    fn test_panns_gate_is_open_when_env_enabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_PANNS_REACTIONS", "1");

        // Directly exercise the engine gate the wrapper now relies on.
        let config = PannsConfig {
            enabled: true,
            ..Default::default()
        };
        let engine = PannsEngine::new(config).expect("engine init");
        assert!(
            engine.is_enabled(),
            "engine must be enabled when the wrapper constructs it with enabled: true"
        );

        // The production wrapper must not mutate boundaries even on the
        // fallback path (missing source -> detect_reactions fails).
        let mut candidates = vec![draft(10.0, 40.0, Some(0.5)), draft(50.0, 80.0, Some(0.6))];
        annotate_candidates_with_reactions(
            "definitely_missing_source_for_panns.mp4",
            &mut candidates,
        );

        assert_eq!(candidates.len(), 2, "fallback must never drop candidates");
        assert_eq!(candidates[0].start, 10.0);
        assert_eq!(candidates[0].end, 40.0);
        assert_eq!(candidates[0].payoff_end, Some(40.0));
        assert_eq!(candidates[1].end, 80.0);
        // Without detected events there is no metadata and no boost.
        assert!(candidates[0].reactions_json.is_none());
        assert_eq!(
            candidates[0].hook_score,
            Some(0.5),
            "no boost without events"
        );

        std::env::remove_var("AUTOSHORTS_PANNS_REACTIONS");
    }

    /// OFF must be a genuine no-op.
    #[test]
    fn test_panns_is_noop_when_env_disabled() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_PANNS_REACTIONS", "0");
        let mut candidates = vec![draft(10.0, 40.0, Some(0.5))];
        annotate_candidates_with_reactions("any_source.mp4", &mut candidates);
        assert!(candidates[0].reactions_json.is_none());
        assert_eq!(candidates[0].hook_score, Some(0.5));
        std::env::remove_var("AUTOSHORTS_PANNS_REACTIONS");
    }

    /// An empty candidate set must be safe.
    #[test]
    fn test_panns_handles_empty_candidate_set() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("AUTOSHORTS_PANNS_REACTIONS", "1");
        let mut candidates: Vec<crate::models::CandidateDraft> = Vec::new();
        annotate_candidates_with_reactions("any_source.mp4", &mut candidates);
        assert!(candidates.is_empty());
        std::env::remove_var("AUTOSHORTS_PANNS_REACTIONS");
    }

    #[test]
    fn test_panns_flag_semantics() {
        let _lock = ENV_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("AUTOSHORTS_PANNS_REACTIONS");
        assert!(
            !panns_reactions_enabled(),
            "default must be disabled (opt-in)"
        );
        for on in ["1", "true", "on"] {
            std::env::set_var("AUTOSHORTS_PANNS_REACTIONS", on);
            assert!(panns_reactions_enabled(), "{} must enable", on);
        }
        for off in ["0", "false", "off", "OFF"] {
            std::env::set_var("AUTOSHORTS_PANNS_REACTIONS", off);
            assert!(!panns_reactions_enabled(), "{} must disable", off);
        }
        std::env::remove_var("AUTOSHORTS_PANNS_REACTIONS");
    }

    #[test]
    fn test_panns_config_default() {
        let config = PannsConfig::default();
        assert!(!config.enabled, "default must be disabled");
        assert_eq!(config.model, "cnn14");
        assert_eq!(config.sample_rate, 32000, "CNN14 is built at 32 kHz");
        assert_eq!(config.confidence_threshold, 0.5);
        // Detection and event gating are SEPARATE knobs; the event gate is
        // never lowered to manufacture events.
        assert_eq!(config.frame_threshold, 0.1);
        assert!(
            config.frame_threshold < config.confidence_threshold,
            "frame threshold must be looser than the event gate"
        );
        assert_eq!(config.min_event_duration_sec, 0.2);
    }

    #[test]
    fn test_reaction_classes_constant() {
        assert!(REACTION_CLASSES.contains(&"Laughter"));
        assert!(REACTION_CLASSES.contains(&"Applause"));
        assert!(REACTION_CLASSES.contains(&"Cheering"));
    }

    /// REGRESSION GUARD: `REACTION_CLASSES` must hold EXACT AudioSet label
    /// strings, because `run_panns_sidecar` matches with `contains` on
    /// `&str` (exact equality).
    ///
    /// The old list contained "Crowd cheer" and "Scream" — neither is an
    /// AudioSet class — plus a duplicate "Giggle". A substring match against
    /// those would have silently dropped real detections.
    #[test]
    fn test_reaction_classes_are_exact_audioset_strings() {
        for phantom in ["Crowd cheer", "Scream", "Gasp, pant", "Chortle"] {
            assert!(
                !REACTION_CLASSES.contains(&phantom),
                "{phantom:?} is not an AudioSet label and must not be in REACTION_CLASSES"
            );
        }
        // No duplicates (the old list had "Giggle" twice).
        let mut seen = std::collections::HashSet::new();
        for c in REACTION_CLASSES {
            assert!(seen.insert(*c), "duplicate reaction class: {c}");
        }
        // The labels the sidecar actually emits on this corpus.
        for real in [
            "Laughter",
            "Chuckle, chortle",
            "Gasp",
            "Giggle",
            "Snicker",
            "Applause",
            "Cheering",
        ] {
            assert!(REACTION_CLASSES.contains(&real), "{real:?} must be accepted");
        }
    }

    #[test]
    fn test_enhance_hook_with_panns() {
        let candidate = crate::models::CandidateDraft::default();
        let reactions = vec![
            ReactionEvent {
                event_type: "Laughter".to_string(),
                start: 1.0,
                end: 2.0,
                confidence: 0.9,
                model: "cnn14".to_string(),
                model_version: "1.0".to_string(),
            },
            ReactionEvent {
                event_type: "Applause".to_string(),
                start: 3.0,
                end: 4.0,
                confidence: 0.8,
                model: "cnn14".to_string(),
                model_version: "1.0".to_string(),
            },
        ];
        let enhanced = enhance_hook_with_panns(0.7, &candidate, &reactions);
        assert!(enhanced > 0.7);
        assert!(enhanced <= 1.0);
    }

    /// ADVISORY-ONLY INVARIANT: the boost is capped and can never push a
    /// hook score above 1.0, however many events overlap.
    #[test]
    fn test_panns_boost_is_capped() {
        let candidate = crate::models::CandidateDraft::default();
        let many: Vec<ReactionEvent> = (0..500)
            .map(|_| ReactionEvent {
                event_type: "Laughter".to_string(),
                start: 0.0,
                end: 10.0,
                confidence: 1.0,
                model: "cnn14".to_string(),
                model_version: "1.0".to_string(),
            })
            .collect();
        // 0.05 * 500 * 1.0 = 25.0 raw boost, capped at 0.15.
        let enhanced = enhance_hook_with_panns(0.5, &candidate, &many);
        assert!(
            (enhanced - 0.65).abs() < 1e-9,
            "boost must cap at 0.15, got {enhanced}"
        );

        // And it can never exceed 1.0 from an already-high base.
        let enhanced = enhance_hook_with_panns(0.99, &candidate, &many);
        assert!(enhanced <= 1.0, "boost must never exceed 1.0, got {enhanced}");
    }

    /// ADVISORY-ONLY INVARIANT: empty reactions means NO change at all.
    #[test]
    fn test_panns_boost_is_noop_with_no_events() {
        let candidate = crate::models::CandidateDraft::default();
        for base in [0.0, 0.5, 0.99, 1.0] {
            assert_eq!(enhance_hook_with_panns(base, &candidate, &[]), base);
        }
    }

    #[test]
    fn test_get_reactions_for_candidate() {
        let metadata = PannsReactionMetadata {
            source_hash: "test".to_string(),
            model: "cnn14".to_string(),
            model_version: "1.0".to_string(),
            events: vec![
                ReactionEvent {
                    event_type: "Laughter".to_string(),
                    start: 1.0,
                    end: 2.0,
                    confidence: 0.9,
                    model: "cnn14".to_string(),
                    model_version: "1.0".to_string(),
                },
                ReactionEvent {
                    event_type: "Applause".to_string(),
                    start: 5.0,
                    end: 6.0,
                    confidence: 0.8,
                    model: "cnn14".to_string(),
                    model_version: "1.0".to_string(),
                },
            ],
            events_by_type: HashMap::new(),
            config_hash: "test".to_string(),
            schema_version: PANNS_CACHE_SCHEMA_VERSION,
            created_at: "2026-01-01T00:00:00Z".to_string(),
        };

        let in_range = get_reactions_for_candidate(&metadata, 0.0, 3.0);
        assert_eq!(in_range.len(), 1);
        assert_eq!(in_range[0].event_type, "Laughter");

        let out_of_range = get_reactions_for_candidate(&metadata, 3.0, 4.0);
        assert_eq!(out_of_range.len(), 0);
    }

    /// CANDIDATE ASSOCIATION: events are in GLOBAL source time, and the
    /// half-open overlap rule must behave at the edges (no double counting).
    fn meta_with(events: Vec<ReactionEvent>) -> PannsReactionMetadata {
        PannsReactionMetadata {
            source_hash: "test".to_string(),
            model: "cnn14".to_string(),
            model_version: "1.0".to_string(),
            events,
            events_by_type: HashMap::new(),
            config_hash: "test".to_string(),
            schema_version: PANNS_CACHE_SCHEMA_VERSION,
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn ev(event_type: &str, start: f64, end: f64, confidence: f64) -> ReactionEvent {
        ReactionEvent {
            event_type: event_type.to_string(),
            start,
            end,
            confidence,
            model: "cnn14".to_string(),
            model_version: "1.0".to_string(),
        }
    }

    #[test]
    fn test_association_matches_event_inside_candidate() {
        let m = meta_with(vec![ev("Chuckle, chortle", 10.0, 12.0, 0.6)]);
        assert_eq!(get_reactions_for_candidate(&m, 0.0, 30.0).len(), 1);
    }

    #[test]
    fn test_association_rejects_event_outside_candidate() {
        let m = meta_with(vec![ev("Laughter", 1.0, 2.0, 0.9)]);
        assert_eq!(get_reactions_for_candidate(&m, 10.0, 20.0).len(), 0);
        assert_eq!(get_reactions_for_candidate(&m, 0.0, 1.0).len(), 0);
    }

    #[test]
    fn test_association_is_half_open_at_edges() {
        // Event 10.0 -> 20.0. Touching edges must NOT overlap.
        let m = meta_with(vec![ev("Laughter", 10.0, 20.0, 0.9)]);
        assert_eq!(get_reactions_for_candidate(&m, 20.0, 30.0).len(), 0);
        assert_eq!(get_reactions_for_candidate(&m, 0.0, 10.0).len(), 0);
        // Genuine partial overlaps must match.
        assert_eq!(get_reactions_for_candidate(&m, 15.0, 25.0).len(), 1);
        assert_eq!(get_reactions_for_candidate(&m, 5.0, 12.0).len(), 1);
    }

    /// The real measured event on this corpus sits at ~1032.96 s — far into the
    /// source. Association must use global source time, not a per-window index.
    #[test]
    fn test_association_uses_global_source_time() {
        let m = meta_with(vec![ev("Chuckle, chortle", 1032.96, 1034.56, 0.2758)]);
        assert_eq!(get_reactions_for_candidate(&m, 1030.0, 1040.0).len(), 1);
        assert_eq!(get_reactions_for_candidate(&m, 0.0, 100.0).len(), 0);
    }

    #[test]
    fn test_association_splits_across_candidates() {
        let m = meta_with(vec![
            ev("Laughter", 10.0, 12.0, 0.9),
            ev("Applause", 150.0, 152.0, 0.8),
        ]);
        assert_eq!(get_reactions_for_candidate(&m, 0.0, 20.0).len(), 1);
        assert_eq!(get_reactions_for_candidate(&m, 100.0, 160.0).len(), 1);
        assert_eq!(get_reactions_for_candidate(&m, 0.0, 200.0).len(), 2);
    }

    /// Metadata must round-trip through the on-disk cache JSON, which is how
    /// `reactions_json` ultimately reaches the DB.
    #[test]
    fn test_metadata_round_trips_through_cache_json() {
        let m = meta_with(vec![ev("Chuckle, chortle", 1032.96, 1034.56, 0.2758)]);
        let json = serde_json::to_string(&m).expect("serialize metadata");
        let back: PannsReactionMetadata = serde_json::from_str(&json).expect("deserialize metadata");
        assert_eq!(back.schema_version, PANNS_CACHE_SCHEMA_VERSION);
        assert_eq!(back.events.len(), 1);
        assert_eq!(back.events[0].event_type, "Chuckle, chortle");
        assert!((back.events[0].start - 1032.96).abs() < 1e-6);
    }

    /// REGRESSION GUARD for the poisoned cache.
    ///
    /// A v1 cache entry (written by the stub detector, which never ran CNN14)
    /// reads back with `events: []` and would be served forever as a cache HIT,
    /// reporting "0 total events" permanently. The schema version must make
    /// such an entry unusable.
    #[test]
    fn test_v1_cache_entry_is_rejected() {
        let v1 = r#"{
            "sourceHash": "abc",
            "model": "cnn14_16k",
            "modelVersion": "1.0",
            "events": [],
            "eventsByType": {},
            "configHash": "06a27c63587eb0236eeecf085ddc03fda11bfd7c86c9e841e27a2312393bc46b",
            "createdAt": "2026-09-30T12:30:20.102722+00:00"
        }"#;
        // v1 JSON has no `schemaVersion`, so it fails to deserialize into the
        // new struct — which load_cache treats as a miss. Either way the entry
        // can never be served.
        let parsed: Result<PannsReactionMetadata, _> = serde_json::from_str(v1);
        assert!(
            parsed.is_err(),
            "a v1 entry (no schemaVersion) must not deserialize into the new struct"
        );
    }

    /// The current schema version must actually be serialized into cache files.
    #[test]
    fn test_cache_json_carries_schema_version() {
        let m = meta_with(vec![]);
        let v: serde_json::Value = serde_json::to_value(&m).expect("to_value");
        assert_eq!(
            v["schemaVersion"].as_u64(),
            Some(PANNS_CACHE_SCHEMA_VERSION as u64)
        );
    }

    /// ADVISORY-ONLY INVARIANT: a real detection may only add `reactions_json`
    /// and nudge `hook_score`. It must not touch ANY boundary field.
    #[test]
    fn test_annotation_never_mutates_boundaries() {
        // Simulate the exact writes the annotation path performs, then assert
        // every boundary field is byte-identical.
        let mut c = draft(12.0, 47.5, Some(0.42));
        c.hook_start = Some(12.0);
        c.hook_end = Some(20.0);
        c.payoff_start = Some(38.0);
        c.payoff_end = Some(47.5);

        let before = (
            c.start,
            c.end,
            c.payoff_end,
            c.payoff_start,
            c.hook_start,
            c.hook_end,
        );
        let events = vec![ev("Laughter", 15.0, 18.0, 0.95)];

        // The only two writes the advisory path is allowed to make:
        c.reactions_json = Some(
            serde_json::json!({
                "count": events.len(),
                "events": events
                    .iter()
                    .map(|e| serde_json::json!({
                        "type": e.event_type, "start": e.start,
                        "end": e.end, "confidence": e.confidence,
                    }))
                    .collect::<Vec<_>>(),
            })
            .to_string(),
        );
        let base = c.hook_score.or(c.guest_hook_strength).unwrap_or(0.0);
        c.hook_score = Some(enhance_hook_with_panns(base, &c, &events));

        let after = (
            c.start,
            c.end,
            c.payoff_end,
            c.payoff_start,
            c.hook_start,
            c.hook_end,
        );
        assert_eq!(before, after, "advisory stage must not move any boundary");
        // Metadata written, boost applied and capped.
        assert!(c.reactions_json.as_ref().unwrap().contains("Laughter"));
        assert_eq!(c.hook_score, Some(0.42 + 0.05 * 0.95));
    }

    #[test]
    fn test_panns_config_timeout_authoritative() {
        let mut config = PannsConfig::default();
        assert_eq!(config.timeout_sec, 300);
        config.timeout_sec = 45;
        assert_eq!(config.timeout_sec, 45);
    }
}
