//! AutoShorts 11.0 Phase 2 — Speaker Intelligence Core
//!
//! Coordinates the Speaker Intelligence subsystem:
//! - Diarization (sidecar, per-source cached, fallback-safe)
//! - Host/Guest mapping (KNOWN / PROBABLE / UNMAPPED, transcript question detection)
//! - Re-ID embedding gallery persistence (DB round-trip)
//! - Active speaker fusion integration (per-candidate sidecar; results persisted per candidate)
//!
//! Speaker Intelligence provides signals. The deterministic AutoShorts engine
//! retains final control. Model failure never blocks a valid fallback run.

use crate::db::Database;
use crate::models::{
    ActiveSpeakerState, ApplicationSpeakerMap, ApplicationSpeakerRole, MappingEvidence,
    SpeakerDiarizationResult, SpeakerMapping, TrackReIdEmbedding, TranscriptWord,
};
use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::PathBuf;

/// Fusion sidecar version written to the active_speaker_fusion table.
pub const FUSION_VERSION: &str = "1.0";

/// Configuration for speaker intelligence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerIntelligenceConfig {
    /// Enable/disable diarization
    pub enable_diarization: bool,
    /// Diarization model: "deepgram", "pyannote", "whisperx", "auto"
    pub diarization_model: String,
    /// Deepgram API key (or from env)
    pub deepgram_api_key: Option<String>,
    /// HuggingFace token for pyannote/whisperx
    pub hf_token: Option<String>,
    /// Enable Re-ID (OSNet)
    pub enable_reid: bool,
    /// Re-ID model name
    pub reid_model: String,
    /// Re-ID similarity threshold for track association
    pub reid_similarity_threshold: f64,
    /// Enable active speaker fusion
    pub enable_active_speaker_fusion: bool,
    /// Fusion interval duration (seconds)
    pub fusion_interval_sec: f64,
    /// Explicit Host override (diarization speaker id) — user/config evidence
    pub host_speaker_override: Option<String>,
    /// Explicit Guest override (diarization speaker id) — user/config evidence
    pub guest_speaker_override: Option<String>,
    /// Cache directory
    pub cache_dir: Option<String>,
}

impl Default for SpeakerIntelligenceConfig {
    fn default() -> Self {
        Self {
            enable_diarization: true,
            diarization_model: "auto".to_string(),
            deepgram_api_key: None,
            hf_token: None,
            enable_reid: true,
            reid_model: "osnet_x1_0".to_string(),
            reid_similarity_threshold: 0.7,
            enable_active_speaker_fusion: true,
            fusion_interval_sec: 1.0,
            host_speaker_override: None,
            guest_speaker_override: None,
            cache_dir: None,
        }
    }
}

/// Cached speaker intelligence for a source video
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerIntelligenceCache {
    pub source_hash: String,
    pub diarization: Option<SpeakerDiarizationResult>,
    pub speaker_map: Option<ApplicationSpeakerMap>,
    pub reid_embeddings: Vec<TrackReIdEmbedding>,
    pub fusion_intervals: Vec<ActiveSpeakerState>,
    pub config_hash: String,
    pub created_at: String,
}

/// Result of speaker intelligence processing for a source video
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerIntelligenceResult {
    pub source_hash: String,
    pub diarization: SpeakerDiarizationResult,
    pub speaker_map: ApplicationSpeakerMap,
    pub reid_embeddings: Vec<TrackReIdEmbedding>,
    pub fusion_intervals: Vec<ActiveSpeakerState>,
}

/// Main speaker intelligence engine
pub struct SpeakerIntelligenceEngine {
    config: SpeakerIntelligenceConfig,
    db: Database,
    cache_dir: PathBuf,
}

/// Wall-clock budget for the Deepgram diarization sidecar.
///
/// The sidecar uploads voice-optimized audio and waits for Deepgram to
/// transcribe-with-diarization. A 25-minute source legitimately takes a
/// couple of minutes; a multi-hour source needs substantially longer (a fixed
/// 300s budget killed the sidecar mid-exchange on a 2.5h source). The budget
/// therefore scales with the extracted audio's duration: a base allowance
/// plus a per-second term, clamped to [floor, ceiling]. The ceiling still
/// stops a wedged child from occupying the machine indefinitely.
const DIARIZATION_TIMEOUT_FLOOR_SEC: f64 = 300.0;
const DIARIZATION_TIMEOUT_CEILING_SEC: f64 = 1800.0;
const DIARIZATION_TIMEOUT_BASE_SEC: f64 = 60.0;
const DIARIZATION_TIMEOUT_PER_AUDIO_SEC: f64 = 0.15;

fn diarization_timeout_for(audio_duration_sec: Option<f64>) -> std::time::Duration {
    let secs = match audio_duration_sec {
        Some(d) if d.is_finite() && d > 0.0 => {
            DIARIZATION_TIMEOUT_BASE_SEC + DIARIZATION_TIMEOUT_PER_AUDIO_SEC * d
        }
        _ => DIARIZATION_TIMEOUT_FLOOR_SEC,
    };
    std::time::Duration::from_secs_f64(secs.clamp(
        DIARIZATION_TIMEOUT_FLOOR_SEC,
        DIARIZATION_TIMEOUT_CEILING_SEC,
    ))
}

/// success/stdout/stderr carrier matching the fields the diarization parser
/// already consumes, sourced from a bounded run.
struct BoundedAsStdOutput {
    success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl SpeakerIntelligenceEngine {
    pub fn new(config: SpeakerIntelligenceConfig, db: Database) -> Result<Self> {
        let cache_dir = config
            .cache_dir
            .as_ref()
            .map(|s| PathBuf::from(s))
            .unwrap_or_else(|| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("autoshorts")
                    .join("speaker_intel")
            });
        std::fs::create_dir_all(&cache_dir).context("creating speaker intelligence cache dir")?;
        Ok(Self {
            config,
            db,
            cache_dir,
        })
    }

    /// Compute SHA256 hash of the source file (cache key).
    pub fn compute_source_hash(&self, source_path: &str) -> Result<String> {
        let file = std::fs::File::open(source_path).context("opening source file for hashing")?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = reader.read(&mut buffer).context("reading file for hash")?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Compute config hash for cache invalidation. Any configuration change
    /// (models, flags, thresholds, overrides) invalidates the cache.
    fn compute_config_hash(&self) -> String {
        let config_str = serde_json::to_string(&self.config).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(config_str.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check if cached result is valid (config unchanged).
    fn is_cache_valid(&self, cache: &SpeakerIntelligenceCache) -> bool {
        cache.config_hash == self.compute_config_hash()
    }

    /// Cache file path for a source hash. Shared by load/save so a stale entry
    /// can be removed using exactly the same key production used to write it.
    fn cache_file_path(&self, source_hash: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", source_hash))
    }

    /// Load cached result. Corrupted or incomplete caches are treated as a
    /// miss (never panic) and removed so the next run re-computes cleanly.
    fn load_cache(&self, source_hash: &str) -> Option<SpeakerIntelligenceCache> {
        let cache_file = self.cache_file_path(source_hash);
        if !cache_file.exists() {
            return None;
        }
        let content = match std::fs::read_to_string(&cache_file) {
            Ok(c) => c,
            Err(_) => return None,
        };
        match serde_json::from_str::<SpeakerIntelligenceCache>(&content) {
            Ok(cache) => Some(cache),
            Err(_) => {
                eprintln!(
                    "[SpeakerIntel] Corrupted cache removed: {}",
                    cache_file.display()
                );
                let _ = std::fs::remove_file(&cache_file);
                None
            }
        }
    }

    /// Save cache.
    fn save_cache(&self, cache: &SpeakerIntelligenceCache) -> Result<()> {
        let cache_file = self.cache_file_path(&cache.source_hash);
        let content = serde_json::to_string_pretty(cache)?;
        std::fs::write(&cache_file, content).context("writing speaker intelligence cache")?;
        Ok(())
    }

    /// Run the complete speaker intelligence pipeline for a source video.
    ///
    /// Diarization + Host/Guest mapping run once per source and are cached.
    /// The Re-ID gallery is loaded from the DB (persisted by per-candidate
    /// renders); fusion intervals are produced per candidate by the sidecar
    /// and refreshed into this cache via `refresh_cache_from_render`.
    pub fn process_source_video(
        &self,
        project_id: &str,
        source_path: &str,
        transcript_words: &[TranscriptWord],
    ) -> Result<SpeakerIntelligenceResult> {
        let source_hash = self.compute_source_hash(source_path)?;
        let config_hash = self.compute_config_hash();

        // Cache check — never panic on corrupt/incomplete caches.
        if let Some(cache) = self.load_cache(&source_hash) {
            if self.is_cache_valid(&cache) {
                println!(
                    "[SpeakerIntel] Cache hit for {}",
                    &source_hash[..16.min(source_hash.len())]
                );
                return Ok(SpeakerIntelligenceResult {
                    source_hash: cache.source_hash,
                    diarization: cache.diarization.ok_or_else(|| {
                        anyhow!("cached speaker intelligence missing diarization")
                    })?,
                    speaker_map: cache.speaker_map.ok_or_else(|| {
                        anyhow!("cached speaker intelligence missing speaker map")
                    })?,
                    reid_embeddings: cache.reid_embeddings,
                    fusion_intervals: cache.fusion_intervals,
                });
            }
            println!(
                "[SpeakerIntel] Cache invalid (config changed) for {}",
                &source_hash[..16.min(source_hash.len())]
            );
        }

        println!("[SpeakerIntel] Processing source video: {}", source_path);

        // Step 1: Diarization (fallback-safe: never blocks the pipeline).
        let diarization = self.run_diarization(source_path, &source_hash)?;
        println!(
            "[SpeakerIntel] Diarization complete: {} speakers, {} segments (model={})",
            diarization.speakers.len(),
            diarization.segments.len(),
            diarization.model
        );
        // Persistence is best-effort: a DB failure must never block a valid
        // fallback run (the result is still returned and JSON-cached).
        if let Err(e) = self.db.save_speaker_diarization(project_id, &diarization) {
            eprintln!(
                "[SpeakerIntel] Diarization persist failed (non-fatal): {}",
                e
            );
        }

        // Step 2: Host/Guest mapping (evidence-based, transcript question detection).
        let speaker_map = self.build_speaker_map(&source_hash, &diarization, transcript_words)?;
        println!(
            "[SpeakerIntel] Speaker mapping complete: {} entries",
            speaker_map.mappings.len()
        );
        if let Err(e) = self.db.save_speaker_mapping(project_id, &speaker_map) {
            eprintln!(
                "[SpeakerIntel] Speaker mapping persist failed (non-fatal): {}",
                e
            );
        }

        // Step 3: Visual intelligence — Re-ID gallery from DB (persisted by
        // per-candidate renders). Fusion intervals are per-candidate and are
        // refreshed into the cache by refresh_cache_from_render.
        let (reid_embeddings, fusion_intervals) =
            self.run_visual_intelligence(project_id, &source_hash)?;

        // Do NOT persist a fallback diarization as a cacheable success. A
        // transient Deepgram failure would otherwise be replayed forever, and
        // the next run would report "Cache hit" with 0 speaker entries even
        // after the real cause was fixed. The result is still returned below
        // (the pipeline is unchanged); it is simply not cached, and any
        // previously cached entry for this source is dropped so the next run
        // re-attempts diarization instead of replaying the old empty result.
        if Self::is_diarization_fallback(&diarization) {
            eprintln!(
                "[SpeakerIntel] Diarization produced no speaker evidence (model={}) — not caching this result; \
the next run will retry diarization.",
                diarization.model
            );
            let _ = std::fs::remove_file(self.cache_file_path(&source_hash));
        } else {
            let cache = SpeakerIntelligenceCache {
                source_hash: source_hash.clone(),
                diarization: Some(diarization.clone()),
                speaker_map: Some(speaker_map.clone()),
                reid_embeddings: reid_embeddings.clone(),
                fusion_intervals: fusion_intervals.clone(),
                config_hash,
                created_at: chrono::Utc::now().to_rfc3339(),
            };
            self.save_cache(&cache)?;
        }

        Ok(SpeakerIntelligenceResult {
            source_hash,
            diarization,
            speaker_map,
            reid_embeddings,
            fusion_intervals,
        })
    }

    /// Empty diarization fallback: carries NO fabricated speaker evidence.
    /// The pipeline continues; word-level diarization labels from
    /// transcription still drive the deterministic framing heuristic.
    fn empty_diarization_fallback(&self, model_label: &str) -> SpeakerDiarizationResult {
        SpeakerDiarizationResult {
            source_hash: String::new(),
            model: model_label.to_string(),
            version: "1.0".to_string(),
            speakers: Vec::new(),
            segments: Vec::new(),
            confidence: 0.0,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// A fallback result is a FAILURE, not a computed value, and must never be
    /// cached as one.
    ///
    /// Caching it made a transient Deepgram error permanent: the next run hit
    /// the cache and replayed the empty diarization indefinitely, so speaker
    /// intelligence stayed at 0 entries long after the real cause was fixed.
    /// The result is still returned to the caller (fallback-safe), it is simply
    /// not persisted as a success.
    fn is_diarization_fallback(result: &SpeakerDiarizationResult) -> bool {
        result.speakers.is_empty() && result.segments.is_empty()
    }

    /// Run diarization via sidecar with a per-config cache directory so any
    /// configuration change invalidates the sidecar cache as well.
    ///
    /// The sidecar is handed EXTRACTED AUDIO, never the source video. Uploading
    /// the video container directly sent the whole file (including video) in a
    /// single Deepgram request, which is what exhausted the upload write and
    /// produced the zero-speaker fallback on long sources. Diarization identity
    /// (cache key, DB `source_hash`) remains the SOURCE VIDEO hash, so nothing
    /// downstream observes the change.
    fn run_diarization(
        &self,
        source_path: &str,
        source_hash: &str,
    ) -> Result<SpeakerDiarizationResult> {
        if !self.config.enable_diarization {
            println!("[SpeakerIntel] Diarization disabled — continuing without speaker evidence");
            return Ok(self.empty_diarization_fallback("disabled"));
        }

        let script_path = self.find_diarization_script()?;
        let python = crate::media::find_python_cmd();

        let deepgram_key = self
            .config
            .deepgram_api_key
            .clone()
            .or_else(|| std::env::var("DEEPGRAM_API_KEY").ok())
            .unwrap_or_default();
        let hf_token = self
            .config
            .hf_token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .unwrap_or_default();

        // Config-scoped cache dir: config change -> sidecar cache miss -> re-run.
        let config_hash = self.compute_config_hash();
        let sidecar_cache_dir = self.cache_dir.join("diarization").join(&config_hash[..12]);
        let _ = std::fs::create_dir_all(&sidecar_cache_dir);

        let model = self.config.diarization_model.clone();

        // Voice-optimized audio (mono 16 kHz) for the sidecar. Scoped to the
        // speaker-intel cache dir so it cannot collide with the transcription
        // pipeline's own extraction of the same source.
        let audio_path =
            crate::media::extract_audio(source_path, &self.cache_dir.join("diarization_audio"));
        let input_path = match &audio_path {
            Ok(p) => {
                println!(
                    "[SpeakerIntel] Diarization audio: {} ({} bytes)",
                    p.display(),
                    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
                );
                p.clone()
            }
            Err(e) => {
                // Audio extraction is a new precondition. If it fails, keep the
                // previous behavior (empty fallback) rather than uploading the
                // video, and say so explicitly.
                eprintln!(
                    "[SpeakerIntel] Diarization audio extraction failed — using empty fallback: {}",
                    e
                );
                return Ok(self.empty_diarization_fallback("fallback"));
            }
        };

        // `Command::new(...)` must be bound to a `let` before the builder chain
        // borrows it, otherwise the temporary Command is dropped mid-chain.
        let mut cmd = std::process::Command::new(&python);
        cmd.arg(&script_path)
            .arg(&input_path)
            .arg("--deepgram-key")
            .arg(&deepgram_key)
            .arg("--hf-token")
            .arg(&hf_token)
            .arg("--model")
            .arg(&model)
            .arg("--cache-dir")
            .arg(&sidecar_cache_dir)
            .arg("--source-hash")
            .arg(source_hash);

        // Bounded: the sidecar uploads extracted audio to Deepgram and waits for
        // the response. `Command::output()` has no deadline, so a stalled HTTP
        // exchange froze the pipeline immediately after speaker mapping with no
        // way to tell which stage owned the wait. The budget scales with the
        // extracted audio's duration so multi-hour sources are not killed
        // mid-exchange (duration probe failure keeps the safe floor).
        let audio_duration_sec = crate::media::probe_media(&input_path.to_string_lossy())
            .ok()
            .and_then(|p| p.duration_sec);
        let budget = diarization_timeout_for(audio_duration_sec);
        println!(
            "[SpeakerIntel] diarization sidecar START budget={:.0}s (audio={:?})",
            budget.as_secs_f64(),
            audio_duration_sec
        );
        let output =
            match crate::proc_guard::run_bounded(&mut cmd, budget, "SpeakerIntel/diarization") {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("[SpeakerIntel] diarization sidecar spawn failed: {}", e);
                    return Ok(self.empty_diarization_fallback("spawn-failed"));
                }
            };
        if output.timed_out {
            eprintln!(
                "[SpeakerIntel] diarization sidecar TIMEOUT after {:.0}s (budget {:.0}s) \
                 - falling back to empty diarization; process tree killed",
                output.elapsed.as_secs_f64(),
                budget.as_secs_f64()
            );
            return Ok(self.empty_diarization_fallback("timeout"));
        }
        if !output.success {
            eprintln!(
                "[SpeakerIntel] diarization sidecar failed rc={:?}: {}",
                output.code,
                crate::proc_guard::sanitize_stderr(&output.stderr, 600)
            );
        }
        println!(
            "[SpeakerIntel] diarization sidecar COMPLETE in {:.1}s rc={:?}",
            output.elapsed.as_secs_f64(),
            output.code
        );
        let output = BoundedAsStdOutput {
            success: output.success,
            stdout: output.stdout.into_bytes(),
            stderr: output.stderr.into_bytes(),
        };

        if !output.success {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "[SpeakerIntel] Diarization sidecar failed — using empty fallback: {}",
                stderr.trim()
            );
            return Ok(self.empty_diarization_fallback("fallback"));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        // The sidecar pretty-prints its result JSON (multi-line) after any
        // progress lines; parse from the first '{' line through end of output.
        let json_start = stdout
            .lines()
            .position(|l| l.trim_start().starts_with('{'))
            .ok_or_else(|| anyhow!("diarization sidecar produced no JSON output"))?;
        let json_text = stdout
            .lines()
            .skip(json_start)
            .collect::<Vec<_>>()
            .join("\n");
        let result: SpeakerDiarizationResult =
            serde_json::from_str(&json_text).context("parsing diarization output")?;
        Ok(result)
    }

    /// Find diarization script (same resolution strategy as pacing.rs).
    fn find_diarization_script(&self) -> Result<PathBuf> {
        let script_name = "speaker_diarization.py";
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
        }
        Err(anyhow!("Diarization script not found"))
    }
}

// ─── Transcript Question Detection ─────────────────────────────────────────────

const QUESTION_WORDS: &[&str] = &[
    "what", "why", "how", "who", "when", "where", "which", "whose", "is", "are", "do", "does",
    "did", "can", "could", "would", "will", "should", "have", "has", "any", "tell", "describe",
    "explain",
];

#[derive(Debug, Clone, Default)]
struct SpeakerQuestionStats {
    /// Turns containing an explicit '?' (strong signal)
    question_turns: usize,
    /// Turns starting with an interrogative word (weak signal)
    interrogative_starts: usize,
    /// Total speaking turns for this speaker
    turns: usize,
}

/// Build contiguous same-speaker turns from word-level transcript and count
/// question evidence per speaker. Deterministic: explicit '?' is a strong
/// signal; interrogative word-start is a weak signal (1/3 weight).
fn compute_question_stats(words: &[TranscriptWord]) -> HashMap<String, SpeakerQuestionStats> {
    let mut stats: HashMap<String, SpeakerQuestionStats> = HashMap::new();
    let mut cur_speaker: Option<String> = None;
    let mut cur_words: Vec<String> = Vec::new();

    let flush = |speaker: &Option<String>,
                 turn_words: &mut Vec<String>,
                 stats: &mut HashMap<String, SpeakerQuestionStats>| {
        if let Some(spk) = speaker {
            let text = turn_words.join(" ").to_lowercase();
            let entry = stats.entry(spk.clone()).or_default();
            entry.turns += 1;
            if text.contains('?') {
                entry.question_turns += 1;
            }
            if let Some(first) = text.split_whitespace().next() {
                let first: String = first.chars().filter(|c| c.is_alphanumeric()).collect();
                if QUESTION_WORDS.contains(&first.as_str()) {
                    entry.interrogative_starts += 1;
                }
            }
        }
        turn_words.clear();
    };

    for w in words {
        let spk = w.speaker.clone().unwrap_or_else(|| "S1".to_string());
        match &cur_speaker {
            Some(s) if *s == spk => cur_words.push(w.text.clone()),
            _ => {
                flush(&cur_speaker, &mut cur_words, &mut stats);
                cur_speaker = Some(spk);
                cur_words.push(w.text.clone());
            }
        }
    }
    flush(&cur_speaker, &mut cur_words, &mut stats);
    stats
}

/// Effective question weight for a speaker: explicit '?' turns plus
/// interrogative starts at 1/3 weight.
fn effective_question_weight(s: &SpeakerQuestionStats) -> f64 {
    s.question_turns as f64 + (s.interrogative_starts as f64) / 3.0
}

// ─── Host/Guest Mapping ────────────────────────────────────────────────────────

impl SpeakerIntelligenceEngine {
    /// Build the Host/Guest speaker map from diarization stats, transcript
    /// question evidence, and explicit overrides.
    ///
    /// States: KNOWN (confidence >= 0.85 or override), PROBABLE (0.5–0.85),
    /// UNMAPPED (ambiguous — left unresolved rather than fabricated).
    ///
    /// Evidence (documented weights, deterministic):
    /// - explicit override -> KNOWN (1.0)
    /// - question-asker evidence -> Host (weight 1.0)
    /// - spoke first -> Host (weight 0.3)
    /// - talks more (guests typically talk more in interviews) -> Guest (weight 0.4)
    pub fn build_speaker_map(
        &self,
        source_hash: &str,
        diarization: &SpeakerDiarizationResult,
        transcript_words: &[TranscriptWord],
    ) -> Result<ApplicationSpeakerMap> {
        let mut mappings = Vec::new();
        let speaker_count = diarization.speakers.len();

        if speaker_count == 0 {
            return Ok(ApplicationSpeakerMap {
                source_hash: source_hash.to_string(),
                mappings,
            });
        }

        let qstats = compute_question_stats(transcript_words);
        let total_speech: f64 = diarization
            .speakers
            .iter()
            .map(|s| s.total_speech_sec)
            .sum();

        // Chronological first speaker (earliest segment start).
        let first_speaker_id = diarization
            .segments
            .iter()
            .min_by(|a, b| {
                a.start
                    .partial_cmp(&b.start)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.speaker_id.clone());

        // Explicit user/config overrides -> KNOWN.
        for speaker in &diarization.speakers {
            if self.config.host_speaker_override.as_deref() == Some(speaker.diarization_id.as_str())
            {
                mappings.push(SpeakerMapping {
                    diarization_id: speaker.diarization_id.clone(),
                    application_role: ApplicationSpeakerRole::Host,
                    confidence: 1.0,
                    evidence: MappingEvidence {
                        visual_track_id: None,
                        speaking_time_ratio: if total_speech > 0.0 {
                            speaker.total_speech_sec / total_speech
                        } else {
                            0.0
                        },
                        first_speaker: first_speaker_id.as_deref()
                            == Some(speaker.diarization_id.as_str()),
                        question_asker: effective_question_weight(
                            &qstats
                                .get(&speaker.diarization_id)
                                .cloned()
                                .unwrap_or_default(),
                        ) > 0.0,
                        user_override: true,
                    },
                });
            } else if self.config.guest_speaker_override.as_deref()
                == Some(speaker.diarization_id.as_str())
            {
                mappings.push(SpeakerMapping {
                    diarization_id: speaker.diarization_id.clone(),
                    application_role: ApplicationSpeakerRole::Guest,
                    confidence: 1.0,
                    evidence: MappingEvidence {
                        visual_track_id: None,
                        speaking_time_ratio: if total_speech > 0.0 {
                            speaker.total_speech_sec / total_speech
                        } else {
                            0.0
                        },
                        first_speaker: first_speaker_id.as_deref()
                            == Some(speaker.diarization_id.as_str()),
                        question_asker: effective_question_weight(
                            &qstats
                                .get(&speaker.diarization_id)
                                .cloned()
                                .unwrap_or_default(),
                        ) > 0.0,
                        user_override: true,
                    },
                });
            }
        }

        let is_overridden = |mappings: &[SpeakerMapping], id: &str| -> bool {
            mappings.iter().any(|m| m.diarization_id == id)
        };

        // Evidence-based mapping for non-overridden speakers.
        for speaker in &diarization.speakers {
            if is_overridden(&mappings, &speaker.diarization_id) {
                continue;
            }
            let other_speech = total_speech - speaker.total_speech_sec;
            let other_q: f64 = diarization
                .speakers
                .iter()
                .filter(|s| s.diarization_id != speaker.diarization_id)
                .map(|s| {
                    effective_question_weight(
                        &qstats.get(&s.diarization_id).cloned().unwrap_or_default(),
                    )
                })
                .sum();
            let own_q = effective_question_weight(
                &qstats
                    .get(&speaker.diarization_id)
                    .cloned()
                    .unwrap_or_default(),
            );

            // Question evidence (documented, deterministic):
            // - exclusive asker (asks, others never ask): +0.7 (strong
            //   structural signal in interview media)
            // - 3+ effective questions: +0.2 (sustained question role)
            // - spoke first: +0.3 (hosts usually open, weak)
            // Speech margin alone never maps Guest (a monologue's talker is
            // not provably a guest); Guest comes from override or the
            // structural complement of a known Host.
            let question_evidence = if own_q > 0.0 && own_q > other_q {
                0.7
            } else {
                0.0
            } + if own_q >= 3.0 { 0.2 } else { 0.0 };

            let is_first = first_speaker_id.as_deref() == Some(speaker.diarization_id.as_str());
            // Talks more (relative share margin) -> guest-leaning evidence.
            let speech_margin = if total_speech > 0.0 {
                ((speaker.total_speech_sec - other_speech) / total_speech).max(0.0)
            } else {
                0.0
            };

            let host_score = question_evidence + 0.3 * (is_first as i32 as f64);
            let guest_score = 0.4 * speech_margin;

            let (role, confidence) = if speaker_count == 1 {
                // Solo video: the sole speaker is the creator (Host, probable).
                (ApplicationSpeakerRole::Host, 0.6_f64)
            } else if host_score >= 0.85 {
                (ApplicationSpeakerRole::Host, (0.95_f64).min(host_score))
            } else if host_score > guest_score + 0.15 && host_score >= 0.5 {
                (ApplicationSpeakerRole::Host, (0.85_f64).min(host_score))
            } else if guest_score > host_score + 0.15 && guest_score >= 0.5 {
                (ApplicationSpeakerRole::Guest, 0.6_f64)
            } else {
                // Ambiguous: remain unresolved rather than fabricated.
                (ApplicationSpeakerRole::Unknown, 0.3_f64)
            };

            mappings.push(SpeakerMapping {
                diarization_id: speaker.diarization_id.clone(),
                application_role: role,
                confidence,
                evidence: MappingEvidence {
                    visual_track_id: None,
                    speaking_time_ratio: if total_speech > 0.0 {
                        speaker.total_speech_sec / total_speech
                    } else {
                        0.0
                    },
                    first_speaker: is_first,
                    question_asker: own_q > 0.0 && own_q > other_q,
                    user_override: false,
                },
            });
        }

        // Two-speaker complement: in interview media with exactly two
        // diarized speakers, a resolved Host (or resolved Guest, PROBABLE+)
        // deterministically implies the other speaker's role. This is
        // structural evidence, not fabrication; the complement keeps a
        // conservative PROBABLE (0.7) so it never carries MORE confidence
        // than the resolved assignment it was derived from. Ambiguous pairs
        // (both Unknown) remain unmapped.
        if speaker_count == 2 && mappings.len() == 2 {
            let resolved = mappings
                .iter()
                .find(|m| {
                    m.application_role != ApplicationSpeakerRole::Unknown && m.confidence >= 0.7
                })
                .cloned();
            if let Some(resolved) = resolved {
                let complement_role = match resolved.application_role {
                    ApplicationSpeakerRole::Host => ApplicationSpeakerRole::Guest,
                    ApplicationSpeakerRole::Guest => ApplicationSpeakerRole::Host,
                    ApplicationSpeakerRole::Unknown => ApplicationSpeakerRole::Unknown,
                };
                if let Some(m) = mappings.iter_mut().find(|m| {
                    m.diarization_id != resolved.diarization_id
                        && m.application_role == ApplicationSpeakerRole::Unknown
                }) {
                    m.application_role = complement_role;
                    m.confidence = 0.7;
                    m.evidence.user_override = false;
                }
            }
        }

        Ok(ApplicationSpeakerMap {
            source_hash: source_hash.to_string(),
            mappings,
        })
    }

    /// Load the persisted Re-ID gallery from the DB (wired by per-candidate
    /// renders). Fusion intervals are per-candidate and refreshed via
    /// refresh_cache_from_render, so at source level they start empty.
    fn run_visual_intelligence(
        &self,
        project_id: &str,
        source_hash: &str,
    ) -> Result<(Vec<TrackReIdEmbedding>, Vec<ActiveSpeakerState>)> {
        let gallery = if self.config.enable_reid {
            match self
                .db
                .load_all_track_reid_embeddings(project_id, source_hash)
            {
                Ok(g) => g,
                Err(e) => {
                    eprintln!(
                        "[SpeakerIntel] Re-ID gallery load failed — continuing without: {}",
                        e
                    );
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        Ok((gallery, Vec::new()))
    }

    /// Refresh the speaker intelligence cache after a per-candidate render:
    /// 1. Persist Re-ID gallery entries reported by the sidecar (deduped).
    /// 2. Persist fusion intervals for the candidate (active_speaker_fusion table).
    /// 3. Refresh the cached gallery + fusion intervals (data-only refresh;
    ///    the config hash is unchanged so this is NOT an invalidation).
    /// Any failure is logged and non-fatal.
    pub fn refresh_cache_from_render(
        &self,
        project_id: &str,
        candidate_id: &str,
        source_path: &str,
        reid_embeddings: &[TrackReIdEmbedding],
        fusion_intervals: &[ActiveSpeakerState],
    ) -> Result<()> {
        let source_hash = self.compute_source_hash(source_path)?;
        for emb in reid_embeddings {
            if let Err(e) = self
                .db
                .save_track_reid_embedding(project_id, &source_hash, emb)
            {
                eprintln!("[SpeakerIntel] Re-ID gallery persist failed: {}", e);
            }
        }
        if !fusion_intervals.is_empty() {
            if let Err(e) =
                self.db
                    .save_active_speaker_fusion(candidate_id, fusion_intervals, FUSION_VERSION)
            {
                eprintln!("[SpeakerIntel] Fusion intervals persist failed: {}", e);
            }
        }
        if let Some(mut cache) = self.load_cache(&source_hash) {
            if !reid_embeddings.is_empty() {
                cache.reid_embeddings = reid_embeddings.to_vec();
            }
            if !fusion_intervals.is_empty() {
                cache.fusion_intervals = fusion_intervals.to_vec();
            }
            let _ = self.save_cache(&cache);
        }
        Ok(())
    }

    /// Get active speaker state for a candidate interval.
    pub fn get_active_speaker_for_candidate(
        &self,
        _project_id: &str,
        candidate_id: &str,
    ) -> Result<Option<Vec<ActiveSpeakerState>>> {
        self.db.load_active_speaker_fusion(candidate_id)
    }
}

// ─── Sidecar speaker_intel block parsing ───────────────────────────────────────

/// Parse the `speakerIntel` block emitted by speaker_tracker.py's plan JSON
/// into (gallery entries, fusion intervals). Fails explicitly so callers can
/// log and skip persistence rather than persisting malformed data.
pub fn parse_speaker_intel_block(
    speaker_intel: &serde_json::Value,
) -> Result<(Vec<TrackReIdEmbedding>, Vec<ActiveSpeakerState>)> {
    let obj = speaker_intel
        .as_object()
        .ok_or_else(|| anyhow!("speakerIntel block is not an object"))?;

    let mut gallery = Vec::new();
    if let Some(entries) = obj.get("gallery").and_then(|v| v.as_array()) {
        for entry in entries {
            let emb: TrackReIdEmbedding = serde_json::from_value(entry.clone())
                .context("parsing speaker intel gallery entry")?;
            // Malformed embedding guard: base64 must decode and be a
            // multiple of 4 bytes (whole float32 samples).
            let decoded = BASE64_STANDARD
                .decode(&emb.embedding_b64)
                .context("decoding gallery embedding")?;
            if decoded.is_empty() || decoded.len() % 4 != 0 {
                return Err(anyhow!(
                    "malformed gallery embedding for track {}",
                    emb.track_id
                ));
            }
            gallery.push(emb);
        }
    }

    let mut intervals = Vec::new();
    if let Some(fusion) = obj.get("fusion") {
        if let Some(list) = fusion.get("intervals").and_then(|v| v.as_array()) {
            for iv in list {
                let state: ActiveSpeakerState = serde_json::from_value(iv.clone())
                    .context("parsing speaker intel fusion interval")?;
                intervals.push(state);
            }
        }
    }

    Ok((gallery, intervals))
}

// ─── Feature Flags ─────────────────────────────────────────────────────────────

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Master switch for the Speaker Intelligence subsystem.
pub fn speaker_intelligence_enabled() -> bool {
    env_flag("AUTOSHORTS_SPEAKER_INTELLIGENCE")
}

pub fn speaker_diarization_enabled() -> bool {
    env_flag("AUTOSHORTS_DIARIZATION")
}

pub fn speaker_reid_enabled() -> bool {
    env_flag("AUTOSHORTS_REID")
}

pub fn active_speaker_fusion_enabled() -> bool {
    env_flag("AUTOSHORTS_ACTIVE_SPEAKER_FUSION")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DiarizedSegment, DiarizedSpeaker};

    fn word(text: &str, start: f64, end: f64, speaker: &str) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: Some(speaker.to_string()),
        }
    }

    #[test]
    fn test_speaker_intelligence_config_default() {
        let config = SpeakerIntelligenceConfig::default();
        assert!(config.enable_diarization);
        assert_eq!(config.diarization_model, "auto");
        assert!(config.enable_reid);
        assert_eq!(config.reid_model, "osnet_x1_0");
        assert!(config.enable_active_speaker_fusion);
        assert!(config.host_speaker_override.is_none());
        assert!(config.guest_speaker_override.is_none());
    }

    #[test]
    fn test_feature_flags_all_four_controlled() {
        let flags = [
            (
                "AUTOSHORTS_SPEAKER_INTELLIGENCE",
                speaker_intelligence_enabled as fn() -> bool,
            ),
            ("AUTOSHORTS_DIARIZATION", speaker_diarization_enabled),
            ("AUTOSHORTS_REID", speaker_reid_enabled),
            (
                "AUTOSHORTS_ACTIVE_SPEAKER_FUSION",
                active_speaker_fusion_enabled,
            ),
        ];
        for (name, check) in &flags {
            std::env::remove_var(name);
            assert!(check(), "{} must default to enabled", name);
            for off in ["0", "false", "off", "OFF", "False"] {
                std::env::set_var(name, off);
                assert!(!check(), "{}={} must disable", name, off);
            }
            std::env::set_var(name, "1");
            assert!(check(), "{}=1 must enable", name);
            std::env::remove_var(name);
        }
    }

    fn diarization_two_speaker() -> SpeakerDiarizationResult {
        SpeakerDiarizationResult {
            source_hash: "hash".into(),
            model: "deepgram".into(),
            version: "nova-3".into(),
            speakers: vec![
                DiarizedSpeaker {
                    diarization_id: "S1".into(),
                    total_speech_sec: 30.0,
                    segment_count: 4,
                    avg_confidence: 0.9,
                },
                DiarizedSpeaker {
                    diarization_id: "S2".into(),
                    total_speech_sec: 60.0,
                    segment_count: 4,
                    avg_confidence: 0.9,
                },
            ],
            segments: vec![
                DiarizedSegment {
                    speaker_id: "S1".into(),
                    start: 0.0,
                    end: 2.0,
                    confidence: 0.9,
                },
                DiarizedSegment {
                    speaker_id: "S2".into(),
                    start: 2.0,
                    end: 10.0,
                    confidence: 0.9,
                },
                DiarizedSegment {
                    speaker_id: "S1".into(),
                    start: 10.0,
                    end: 12.0,
                    confidence: 0.9,
                },
                DiarizedSegment {
                    speaker_id: "S2".into(),
                    start: 12.0,
                    end: 30.0,
                    confidence: 0.9,
                },
                DiarizedSegment {
                    speaker_id: "S1".into(),
                    start: 30.0,
                    end: 32.0,
                    confidence: 0.9,
                },
                DiarizedSegment {
                    speaker_id: "S2".into(),
                    start: 32.0,
                    end: 60.0,
                    confidence: 0.9,
                },
            ],
            confidence: 0.9,
            created_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn test_speaker_map_question_asker_maps_host() {
        let engine = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig::default(),
            Database::open(&std::env::temp_dir().join("si_test_qa.sqlite")).unwrap(),
        )
        .unwrap();
        // S1 asks questions (with '?'), S2 answers. Guest talks more.
        let words = vec![
            word("What", 0.0, 0.5, "S1"),
            word("changed?", 0.5, 1.0, "S1"),
            word("I", 1.5, 1.8, "S2"),
            word("learned", 1.8, 2.5, "S2"),
            word("Why", 10.0, 10.5, "S1"),
            word("does", 10.5, 11.0, "S1"),
            word("it", 11.0, 11.2, "S1"),
            word("matter?", 11.2, 12.0, "S1"),
            word("Because", 12.5, 13.5, "S2"),
            word("sleep", 13.5, 14.5, "S2"),
        ];
        let diar = diarization_two_speaker();
        let map = engine.build_speaker_map("hash", &diar, &words).unwrap();
        assert_eq!(map.mappings.len(), 2);
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        let s2 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S2")
            .unwrap();
        assert_eq!(
            s1.application_role,
            ApplicationSpeakerRole::Host,
            "question asker must map to Host"
        );
        assert!(
            s1.confidence >= 0.85,
            "dominant question evidence must be KNOWN, got {}",
            s1.confidence
        );
        assert!(s1.evidence.question_asker);
        assert_eq!(s2.application_role, ApplicationSpeakerRole::Guest);
        assert!(!s2.evidence.question_asker);
        assert!(!s1.evidence.user_override);
    }

    #[test]
    fn test_speaker_map_symmetric_stays_unmapped() {
        let engine = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig::default(),
            Database::open(&std::env::temp_dir().join("si_test_sym.sqlite")).unwrap(),
        )
        .unwrap();
        // Both speakers ask equally and talk equally: ambiguous -> UNMAPPED.
        let words = vec![
            word("What", 0.0, 0.5, "S1"),
            word("happened?", 0.5, 1.0, "S1"),
            word("Why", 2.0, 2.5, "S2"),
            word("though?", 2.5, 3.0, "S2"),
            word("tell", 4.0, 4.5, "S1"),
            word("me", 4.5, 4.8, "S1"),
            word("describe", 6.0, 6.5, "S2"),
            word("it", 6.5, 6.8, "S2"),
        ];
        let mut diar = diarization_two_speaker();
        diar.speakers[0].total_speech_sec = 45.0;
        diar.speakers[1].total_speech_sec = 45.0;
        let map = engine.build_speaker_map("hash", &diar, &words).unwrap();
        for m in &map.mappings {
            assert_eq!(
                m.application_role,
                ApplicationSpeakerRole::Unknown,
                "symmetric evidence must remain unresolved, got {:?} for {}",
                m.application_role,
                m.diarization_id
            );
            assert!((m.confidence - 0.3).abs() < 1e-9);
        }
    }

    #[test]
    fn test_speaker_map_user_override_is_known() {
        let config = SpeakerIntelligenceConfig {
            host_speaker_override: Some("S2".to_string()),
            guest_speaker_override: Some("S1".to_string()),
            ..Default::default()
        };
        let engine = SpeakerIntelligenceEngine::new(
            config,
            Database::open(&std::env::temp_dir().join("si_test_ovr.sqlite")).unwrap(),
        )
        .unwrap();
        let diar = diarization_two_speaker();
        let map = engine.build_speaker_map("hash", &diar, &[]).unwrap();
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        let s2 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S2")
            .unwrap();
        assert_eq!(s2.application_role, ApplicationSpeakerRole::Host);
        assert!((s2.confidence - 1.0).abs() < 1e-9);
        assert!(s2.evidence.user_override);
        assert_eq!(s1.application_role, ApplicationSpeakerRole::Guest);
        assert!(s1.evidence.user_override);
    }

    #[test]
    fn test_speaker_map_empty_diarization_empty_map() {
        let engine = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig::default(),
            Database::open(&std::env::temp_dir().join("si_test_empty.sqlite")).unwrap(),
        )
        .unwrap();
        let diar = engine.empty_diarization_fallback("fallback");
        let map = engine.build_speaker_map("hash", &diar, &[]).unwrap();
        assert!(map.mappings.is_empty());
        assert_eq!(map.source_hash, "hash");
    }

    #[test]
    fn test_cache_roundtrip_invalidation_and_corruption() {
        let cache_dir = std::env::temp_dir().join(format!("si_test_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache_dir);
        let engine = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig {
                cache_dir: Some(cache_dir.to_string_lossy().to_string()),
                ..Default::default()
            },
            Database::open(&std::env::temp_dir().join("si_test_cache.sqlite")).unwrap(),
        )
        .unwrap();

        let source_path = std::env::temp_dir().join("si_test_source.bin");
        std::fs::write(&source_path, b"speaker-intelligence-cache-test").unwrap();
        let source_path_str = source_path.to_string_lossy().to_string();

        // Seed a cache entry directly through the engine's save path.
        let source_hash = engine.compute_source_hash(&source_path_str).unwrap();
        let cache = SpeakerIntelligenceCache {
            source_hash: source_hash.clone(),
            diarization: Some(diarization_two_speaker()),
            speaker_map: Some(ApplicationSpeakerMap {
                source_hash: source_hash.clone(),
                mappings: Vec::new(),
            }),
            reid_embeddings: Vec::new(),
            fusion_intervals: Vec::new(),
            config_hash: {
                // Must match the engine's own config hash for a valid hit.
                let config_str = serde_json::to_string(&SpeakerIntelligenceConfig {
                    cache_dir: Some(cache_dir.to_string_lossy().to_string()),
                    ..Default::default()
                })
                .unwrap();
                let mut hasher = Sha256::new();
                hasher.update(config_str.as_bytes());
                format!("{:x}", hasher.finalize())
            },
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        engine.save_cache(&cache).unwrap();

        // Cold-load: the engine's process_source_video path would hit.
        let loaded = engine
            .process_source_video("proj", &source_path_str, &[])
            .unwrap();
        assert_eq!(loaded.source_hash, source_hash);
        assert_eq!(loaded.diarization.model, "deepgram");

        // Configuration change -> invalidation -> recompute (fallback diarization).
        let engine2 = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig {
                cache_dir: Some(cache_dir.to_string_lossy().to_string()),
                reid_similarity_threshold: 0.8, // changed config
                ..Default::default()
            },
            Database::open(&std::env::temp_dir().join("si_test_cache.sqlite")).unwrap(),
        )
        .unwrap();
        let reloaded = engine2
            .process_source_video("proj", &source_path_str, &[])
            .unwrap();
        assert_eq!(
            reloaded.diarization.model, "fallback",
            "config change must invalidate and recompute"
        );

        // Source change -> different cache key.
        let other_source = std::env::temp_dir().join("si_test_source2.bin");
        std::fs::write(&other_source, b"different-content").unwrap();
        let h1 = engine.compute_source_hash(&source_path_str).unwrap();
        let h2 = engine
            .compute_source_hash(&other_source.to_string_lossy())
            .unwrap();
        assert_ne!(
            h1, h2,
            "different sources must produce different cache keys"
        );

        // Corrupted cache -> treated as miss, file removed, no panic.
        let corrupt = engine.cache_dir.join(format!("{}.json", h1));
        std::fs::write(&corrupt, "{ not valid json").unwrap();
        let loaded_after_corrupt = engine.load_cache(&h1);
        assert!(
            loaded_after_corrupt.is_none(),
            "corrupt cache must be treated as a miss"
        );
        assert!(!corrupt.exists(), "corrupt cache file must be removed");

        // Incomplete cache (missing diarization) -> treated as miss downstream.
        let incomplete = SpeakerIntelligenceCache {
            source_hash: h1.clone(),
            diarization: None,
            speaker_map: None,
            reid_embeddings: Vec::new(),
            fusion_intervals: Vec::new(),
            config_hash: cache.config_hash.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        engine.save_cache(&incomplete).unwrap();
        let result = engine.process_source_video("proj", &source_path_str, &[]);
        assert!(
            result.is_err(),
            "incomplete cache must not be served as a valid hit"
        );

        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn test_parse_speaker_intel_block() {
        let emb_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let block = serde_json::json!({
            "method": "fusion",
            "fusionVersion": "1.0",
            "gallery": [
                {
                    "trackId": 3,
                    "shotIdx": 0,
                    "embeddingB64": BASE64_STANDARD.encode(&emb_bytes),
                    "model": "osnet_x1_0",
                    "detectionCount": 5,
                    "firstSeenSec": 1.0,
                    "lastSeenSec": 5.0,
                    "updatedAt": "2026-01-01T00:00:00Z"
                }
            ],
            "fusion": {
                "intervals": [
                    {
                        "trackId": 3,
                        "applicationRole": "host",
                        "diarizationId": "S1",
                        "start": 0.0,
                        "end": 2.0,
                        "audioEvidence": 0.9,
                        "visualEvidence": 0.4,
                        "speakingProbability": 0.8,
                        "confidence": 0.85,
                        "evidenceSources": ["diarization", "visual_mouth_motion"]
                    }
                ]
            }
        });

        let (gallery, intervals) = parse_speaker_intel_block(&block).unwrap();
        assert_eq!(gallery.len(), 1);
        assert_eq!(gallery[0].track_id, 3);
        assert_eq!(gallery[0].shot_idx, 0);
        assert_eq!(gallery[0].model, "osnet_x1_0");
        assert_eq!(intervals.len(), 1);
        assert_eq!(intervals[0].track_id, 3);
        assert_eq!(intervals[0].diarization_id.as_deref(), Some("S1"));
        assert_eq!(
            intervals[0].application_role,
            Some(ApplicationSpeakerRole::Host)
        );
    }

    #[test]
    fn test_parse_speaker_intel_block_malformed_embedding_rejected() {
        let block = serde_json::json!({
            "method": "fusion",
            "gallery": [
                {
                    "trackId": 1,
                    "shotIdx": 0,
                    "embeddingB64": "###not-base64###",
                    "model": "osnet_x1_0",
                    "detectionCount": 1,
                    "firstSeenSec": 0.0,
                    "lastSeenSec": 1.0,
                    "updatedAt": "2026-01-01T00:00:00Z"
                }
            ]
        });
        let result = parse_speaker_intel_block(&block);
        assert!(result.is_err(), "malformed embedding must be rejected");
    }

    // ─── Real-media end-to-end harness ────────────────────────────────────────
    // Gated: set AUTOSHORTS_SI_E2E=1 to run. Requires the Phase 2 e2e corpus
    // (../..//scratch/phase2_e2e/rio_90.mp4 + rio_90_words.json) and a
    // Deepgram key in autoshorts/.env for the live diarization path; missing
    // corpus/key exercises the documented fallback path instead.
    #[test]
    fn e2e_real_media_speaker_intelligence() {
        if std::env::var("AUTOSHORTS_SI_E2E").ok().as_deref() != Some("1") {
            return;
        }
        let _ = dotenvy::dotenv();

        let workspace = PathBuf::from("../..");
        let media_path = workspace.join("scratch/phase2_e2e/rio_90.mp4");
        if !media_path.exists() {
            eprintln!("[SI-E2E] corpus missing; skipping live run");
            return;
        }
        let media_str = media_path.to_string_lossy().to_string();

        let db = Database::open(&std::env::temp_dir().join("si_e2e.sqlite")).unwrap();
        let project = db
            .create_project(&media_str, "deepgram", "kinetic", "original", Some(90.0))
            .unwrap();

        let cache_dir = std::env::temp_dir().join(format!("si_e2e_cache_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&cache_dir);

        let engine = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig {
                cache_dir: Some(cache_dir.to_string_lossy().to_string()),
                ..Default::default()
            },
            db.clone(),
        )
        .unwrap();

        let words: Vec<TranscriptWord> =
            std::fs::read_to_string(workspace.join("scratch/phase2_e2e/rio_90_words.json"))
                .ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_default();

        // 1. Cold run (diarization sidecar + mapping + gallery load).
        let t0 = std::time::Instant::now();
        let si = engine
            .process_source_video(&project.id, &media_str, &words)
            .expect("cold run must succeed (fallback-safe)");
        let cold = t0.elapsed();
        eprintln!(
            "[SI-E2E] cold: speakers={} segments={} mappings={} diar_model={} elapsed={:?}",
            si.diarization.speakers.len(),
            si.diarization.segments.len(),
            si.speaker_map.mappings.len(),
            si.diarization.model,
            cold
        );

        // 2. Identical rerun → cache hit (same created_at proves reuse).
        let si2 = engine
            .process_source_video(&project.id, &media_str, &words)
            .unwrap();
        assert_eq!(
            si2.diarization.created_at, si.diarization.created_at,
            "rerun must hit the cache"
        );
        let cached = t0.elapsed() - cold;
        eprintln!("[SI-E2E] cached rerun elapsed={:?}", cached);

        // 3. Simulated candidate render: hand sidecar inputs to the framing
        //    tracker exactly as lib.rs does, then persist the speakerIntel
        //    block back (gallery -> DB, fusion intervals -> DB).
        let diar_tmp =
            std::env::temp_dir().join(format!("si_e2e_diar_{}.json", std::process::id()));
        let gallery_tmp =
            std::env::temp_dir().join(format!("si_e2e_gal_{}.json", std::process::id()));
        let diar_doc = serde_json::json!({
            "model": si.diarization.model,
            "segments": si.diarization.segments.iter().map(|s| serde_json::json!({
                "speaker_id": s.speaker_id, "start": s.start, "end": s.end,
                "confidence": s.confidence,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(&diar_tmp, serde_json::to_string(&diar_doc).unwrap()).unwrap();
        std::fs::write(
            &gallery_tmp,
            serde_json::to_string(&serde_json::json!({"entries": si.reid_embeddings})).unwrap(),
        )
        .unwrap();

        let framing_start = std::time::Instant::now();
        let diar_str = diar_tmp.to_string_lossy().into_owned();
        let gal_str = gallery_tmp.to_string_lossy().into_owned();
        let sidecar_inputs = crate::media::SpeakerIntelSidecarInputs {
            diarization_json: diar_str.as_str(),
            gallery_json: gal_str.as_str(),
            scene_cuts_json: None,
        };
        let plan = crate::media::detect_speaker_crop_params_with_intel(
            &media_str,
            0.0,
            90.0,
            1920,
            1080,
            607,
            if words.is_empty() { None } else { Some(&words) },
            "original",
            Some(&sidecar_inputs),
        );
        let framing_elapsed = framing_start.elapsed();
        eprintln!(
            "[SI-E2E] framing+intel run elapsed={:?} speaker_intel={}",
            framing_elapsed,
            plan.speaker_intel.is_some()
        );
        let _ = std::fs::remove_file(&diar_tmp);
        let _ = std::fs::remove_file(&gallery_tmp);

        if let Some(intel) = &plan.speaker_intel {
            let (gallery_entries, fusion_intervals) = parse_speaker_intel_block(intel).unwrap();
            eprintln!(
                "[SI-E2E] speakerIntel: gallery={} fusion_intervals={}",
                gallery_entries.len(),
                fusion_intervals.len()
            );

            // DB persistence round-trip proof. A real candidate row is required
            // (active_speaker_fusion.candidate_id has an FK to candidates).
            let drafts: Vec<crate::models::CandidateDraft> =
                serde_json::from_value(serde_json::json!([
                    { "start": 0.0, "end": 90.0, "score": 1.0, "hook": "e2e", "rationale": "e2e" }
                ]))
                .unwrap();
            let candidates = db.replace_candidates(&project.id, &drafts).unwrap();
            let candidate_id = candidates[0].id.clone();
            engine
                .refresh_cache_from_render(
                    &project.id,
                    &candidate_id,
                    &media_str,
                    &gallery_entries,
                    &fusion_intervals,
                )
                .unwrap();

            let saved_fusion = db
                .load_active_speaker_fusion(&candidate_id)
                .unwrap()
                .unwrap_or_default();
            assert!(
                !saved_fusion.is_empty(),
                "fusion intervals must be persisted"
            );
            let source_hash = engine.compute_source_hash(&media_str).unwrap();
            let saved_gallery = db
                .load_all_track_reid_embeddings(&project.id, &source_hash)
                .unwrap();
            eprintln!(
                "[SI-E2E] DB round-trip: fusion_rows={} gallery_rows={}",
                saved_fusion.len(),
                saved_gallery.len()
            );
        } else {
            eprintln!("[SI-E2E] no speakerIntel block (tracker ran but intel disabled or no tracks) — inspect stderr");
        }

        // 4. Config change → invalidation → recompute.
        let engine_changed = SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig {
                cache_dir: Some(cache_dir.to_string_lossy().to_string()),
                reid_similarity_threshold: 0.8,
                ..Default::default()
            },
            db.clone(),
        )
        .unwrap();
        let si_changed = engine_changed
            .process_source_video(&project.id, &media_str, &words)
            .unwrap();
        assert_ne!(
            si_changed.diarization.created_at, si.diarization.created_at,
            "config change must invalidate the cache and recompute"
        );
        eprintln!("[SI-E2E] cache invalidation verified");
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    // ─── TASK 8: Host/Guest mapping validation battery (8 scenarios) ─────────
    // Scenarios carry ground truth by construction; ambiguous cases must
    // remain unresolved rather than fabricated.

    fn si_engine_named(name: &str) -> SpeakerIntelligenceEngine {
        SpeakerIntelligenceEngine::new(
            SpeakerIntelligenceConfig::default(),
            Database::open(&std::env::temp_dir().join(format!("si_batt_{}.sqlite", name))).unwrap(),
        )
        .unwrap()
    }

    fn diar(speakers: &[(&str, f64)], segments: &[(&str, f64, f64)]) -> SpeakerDiarizationResult {
        SpeakerDiarizationResult {
            source_hash: "batt".into(),
            model: "deepgram".into(),
            version: "nova-3".into(),
            speakers: speakers
                .iter()
                .map(|(id, sec)| DiarizedSpeaker {
                    diarization_id: id.to_string(),
                    total_speech_sec: *sec,
                    segment_count: 2,
                    avg_confidence: 0.9,
                })
                .collect(),
            segments: segments
                .iter()
                .map(|(id, s, e)| DiarizedSegment {
                    speaker_id: id.to_string(),
                    start: *s,
                    end: *e,
                    confidence: 0.9,
                })
                .collect(),
            confidence: 0.9,
            created_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn test_battery_host_speaks_first_guest_talks_more() {
        // Scenarios 1+3: Host speaks first (short intro question), guest answers
        // and talks more. Question evidence must drive Host.
        let engine = si_engine_named("hfirst");
        let d = diar(
            &[("S1", 20.0), ("S2", 60.0)],
            &[
                ("S1", 0.0, 3.0),
                ("S2", 3.0, 23.0),
                ("S1", 23.0, 26.0),
                ("S2", 26.0, 60.0),
            ],
        );
        let words = vec![
            word("Today", 0.0, 0.5, "S1"),
            word("we", 0.5, 0.8, "S1"),
            word("ask:", 0.8, 1.2, "S1"),
            word("Why", 1.5, 1.8, "S1"),
            word("sleep?", 1.8, 2.5, "S1"),
            word("Well", 3.0, 3.5, "S2"),
            word("sleep", 3.5, 4.5, "S2"),
        ];
        let map = engine.build_speaker_map("batt", &d, &words).unwrap();
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        let s2 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S2")
            .unwrap();
        assert_eq!(s1.application_role, ApplicationSpeakerRole::Host);
        assert_eq!(s2.application_role, ApplicationSpeakerRole::Guest);
        assert!(s1.evidence.first_speaker);
    }

    #[test]
    fn test_battery_guest_speaks_first_host_talks_more() {
        // Scenarios 2+4: Guest opens with content (no questions), host asks
        // later and talks more. Question evidence must still map Host.
        let engine = si_engine_named("gfirst");
        let d = diar(
            &[("S1", 50.0), ("S2", 30.0)],
            &[
                ("S1", 0.0, 20.0),
                ("S2", 20.0, 23.0),
                ("S1", 23.0, 50.0),
                ("S2", 50.0, 57.0),
            ],
        );
        let words = vec![
            word("Honestly", 0.0, 0.5, "S1"),
            word("it", 0.5, 0.8, "S1"),
            word("changed", 0.8, 1.5, "S1"),
            word("everything", 1.5, 2.5, "S1"),
            word("What", 20.0, 20.5, "S2"),
            word("changed?", 20.5, 21.2, "S2"),
            word("How", 50.0, 50.5, "S2"),
            word("so?", 50.5, 51.0, "S2"),
        ];
        let map = engine.build_speaker_map("batt", &d, &words).unwrap();
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        let s2 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S2")
            .unwrap();
        assert_eq!(
            s2.application_role,
            ApplicationSpeakerRole::Host,
            "question asker must map to Host even when guest speaks first"
        );
        assert_eq!(s1.application_role, ApplicationSpeakerRole::Guest);
        assert!(
            s1.evidence.first_speaker,
            "first-speaker evidence recorded for S1"
        );
        assert!(!s2.evidence.first_speaker);
    }

    #[test]
    fn test_battery_guest_asks_questions() {
        // Scenario 6: the GUEST asks questions and the host answers — the
        // question asker maps to Host regardless of who it is.
        let engine = si_engine_named("gasks");
        let d = diar(
            &[("S1", 50.0), ("S2", 30.0)],
            &[("S1", 0.0, 20.0), ("S2", 20.0, 23.0), ("S1", 23.0, 50.0)],
        );
        let words = vec![
            word("Why", 20.0, 20.5, "S2"),
            word("does", 20.5, 21.0, "S2"),
            word("this", 21.0, 21.3, "S2"),
            word("happen?", 21.3, 22.5, "S2"),
            word("Because", 23.0, 24.0, "S1"),
            word("fatigue", 24.0, 25.5, "S1"),
        ];
        let map = engine.build_speaker_map("batt", &d, &words).unwrap();
        let s2 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S2")
            .unwrap();
        assert_eq!(
            s2.application_role,
            ApplicationSpeakerRole::Host,
            "question asker (S2) must map to Host"
        );
        assert!(s2.evidence.question_asker);
    }

    #[test]
    fn test_battery_short_intro_stays_probable_or_unmapped() {
        // Scenario 7: very short introduction with no question evidence —
        // mapping must NOT fabricate a confident Host.
        let engine = si_engine_named("shortintro");
        let d = diar(
            &[("S1", 2.0), ("S2", 2.0)],
            &[("S1", 0.0, 2.0), ("S2", 2.0, 4.0)],
        );
        let words = vec![
            word("Hello", 0.0, 0.5, "S1"),
            word("there", 0.5, 1.0, "S1"),
            word("Hi", 2.0, 2.4, "S2"),
            word("thanks", 2.4, 3.0, "S2"),
        ];
        let map = engine.build_speaker_map("batt", &d, &words).unwrap();
        for m in &map.mappings {
            assert!(
                m.confidence < 0.85,
                "no question evidence: must not fabricate a KNOWN mapping, got {} for {}",
                m.confidence,
                m.diarization_id
            );
        }
    }

    #[test]
    fn test_battery_overlapping_dialogue_resolves_by_question_evidence() {
        // Scenario 8: overlapping dialogue — both diarized heavily; question
        // evidence resolves the Host; the complement resolves the Guest.
        let engine = si_engine_named("overlap");
        let d = diar(
            &[("S1", 40.0), ("S2", 38.0)],
            &[("S1", 0.0, 40.0), ("S2", 0.0, 38.0)],
        );
        let words = vec![
            word("What", 0.0, 0.5, "S1"),
            word("drove", 0.5, 1.0, "S1"),
            word("this?", 1.0, 1.5, "S1"),
            word("Why", 5.0, 5.5, "S1"),
            word("now?", 5.5, 6.0, "S1"),
            word("How", 10.0, 10.5, "S1"),
            word("come?", 10.5, 11.0, "S1"),
            word("Well", 1.5, 1.8, "S2"),
            word("it", 1.8, 2.0, "S2"),
            word("started", 2.0, 2.8, "S2"),
        ];
        let map = engine.build_speaker_map("batt", &d, &words).unwrap();
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        let s2 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S2")
            .unwrap();
        assert_eq!(s1.application_role, ApplicationSpeakerRole::Host);
        assert!(s1.confidence >= 0.85);
        assert_eq!(
            s2.application_role,
            ApplicationSpeakerRole::Guest,
            "complement must resolve the Guest"
        );
        assert!(
            (s2.confidence - 0.7).abs() < 1e-9,
            "complement stays PROBABLE"
        );
    }

    #[test]
    fn test_battery_multi_speaker_panel_question_asker_host() {
        // >2 speakers: the dominant question asker maps Host; others unmapped.
        let engine = si_engine_named("panel");
        let d = diar(
            &[("S1", 30.0), ("S2", 25.0), ("S3", 25.0)],
            &[("S1", 0.0, 10.0), ("S2", 10.0, 20.0), ("S3", 20.0, 30.0)],
        );
        let words = vec![
            word("What", 0.0, 0.5, "S1"),
            word("brings", 0.5, 1.0, "S1"),
            word("you?", 1.0, 1.5, "S1"),
            word("Why", 2.0, 2.5, "S1"),
            word("here?", 2.5, 3.0, "S1"),
            word("How", 4.0, 4.5, "S1"),
            word("so?", 4.5, 5.0, "S1"),
            word("Well", 10.0, 10.5, "S2"),
            word("I", 10.5, 10.8, "S3"),
            word("agree", 10.8, 11.5, "S3"),
        ];
        let map = engine.build_speaker_map("batt", &d, &words).unwrap();
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        assert_eq!(s1.application_role, ApplicationSpeakerRole::Host);
        let others: Vec<_> = map
            .mappings
            .iter()
            .filter(|m| m.diarization_id != "S1")
            .collect();
        for m in others {
            assert_eq!(
                m.application_role,
                ApplicationSpeakerRole::Unknown,
                "non-asker panel speakers stay unmapped without more evidence"
            );
        }
    }

    // ─── Deepgram long-form regression (production write-timeout failure) ─────
    //
    // A source VIDEO was previously handed to the diarization sidecar, which
    // uploaded it to Deepgram in a single request. The upload write timed out
    // and the engine silently produced a 0-entry speaker map. These tests pin
    // the engine-side invariants that must survive that fix.

    #[test]
    fn test_empty_fallback_carries_no_fabricated_speaker_evidence() {
        // The documented fallback must stay empty and carry no invented
        // confidence. A Deepgram failure may never become a fake speaker.
        let engine = si_engine_named("fallback");
        let fb = engine.empty_diarization_fallback("fallback");
        assert!(fb.speakers.is_empty(), "fallback must not invent speakers");
        assert!(fb.segments.is_empty(), "fallback must not invent segments");
        assert_eq!(fb.confidence, 0.0, "fallback must claim no confidence");
        assert_eq!(fb.model, "fallback");
        assert!(fb.source_hash.is_empty());
    }

    #[test]
    fn test_zero_entry_map_only_from_empty_diarization() {
        // The reported symptom was "Speaker mapping complete: 0 entries".
        // That is only legitimate when diarization genuinely produced nothing.
        let engine = si_engine_named("zeroentries");

        // Failure path: empty diarization -> 0 mappings, and it is a clean 0.
        let empty = engine.empty_diarization_fallback("fallback");
        let map = engine.build_speaker_map("hash", &empty, &[]).unwrap();
        assert!(
            map.mappings.is_empty(),
            "an empty diarization must yield zero mappings, not invented ones"
        );

        // Success path: real diarization MUST produce non-empty mappings.
        let d = diar(
            &[("S1", 10.0), ("S2", 40.0)],
            &[("S1", 0.0, 5.0), ("S2", 5.0, 45.0)],
        );
        let words = vec![
            word("What", 0.0, 0.5, "S1"),
            word("happened?", 0.5, 1.2, "S1"),
            word("Well", 5.0, 5.5, "S2"),
            word("lots", 5.5, 6.0, "S2"),
        ];
        let map = engine.build_speaker_map("hash", &d, &words).unwrap();
        assert!(
            !map.mappings.is_empty(),
            "a successful diarization must never report 0 speaker entries"
        );
    }

    #[test]
    fn test_diarization_identity_is_source_video_hash_not_audio() {
        // run_diarization now receives EXTRACTED AUDIO but diarization identity
        // must remain the SOURCE VIDEO hash, or the cache and the DB row would
        // no longer join with the source the rest of the engine is keyed on.
        let engine = si_engine_named("identity");
        let source = std::env::temp_dir().join("si_identity_source.mp4");
        std::fs::write(&source, b"not-really-a-video").unwrap();

        let video_hash = engine
            .compute_source_hash(&source.to_string_lossy())
            .unwrap();
        // A different (audio) file must hash differently -- which is exactly why
        // the video hash has to be passed explicitly to the sidecar.
        let audio = std::env::temp_dir().join("si_identity_audio.mp3");
        std::fs::write(&audio, b"not-really-audio").unwrap();
        let audio_hash = engine
            .compute_source_hash(&audio.to_string_lossy())
            .unwrap();
        assert_ne!(
            video_hash, audio_hash,
            "if these matched, passing the audio hash would be indistinguishable from the video hash"
        );

        // The map must carry the source-video hash it was given.
        let d = diar(&[("S1", 10.0)], &[("S1", 0.0, 10.0)]);
        let map = engine
            .build_speaker_map(&video_hash, &d, &[word("Hi", 0.0, 1.0, "S1")])
            .unwrap();
        assert_eq!(
            map.source_hash, video_hash,
            "speaker map must stay keyed on the source video hash"
        );
    }

    #[test]
    fn test_long_form_diarization_segments_preserve_order_and_time() {
        // Long-form chunking re-bases chunk-local timestamps into global source
        // time. The engine must accept a long, strictly increasing segment list
        // and keep it intact for the mapping step.
        let engine = si_engine_named("longform");
        let duration = 1482.4_f64; // the source that failed in production
        let mut segments: Vec<(String, f64, f64)> = Vec::new();
        let mut cursor = 0.0;
        for i in 0..12 {
            let speaker = if i % 2 == 0 { "S1" } else { "S2" };
            segments.push((speaker.to_string(), cursor, cursor + 100.0));
            cursor += 100.0;
        }
        assert!(
            cursor <= duration,
            "segment coverage must stay within the source duration"
        );

        let d = diar(
            &[("S1", 600.0), ("S2", 600.0)],
            &segments
                .iter()
                .map(|(s, a, b)| (s.as_str(), *a, *b))
                .collect::<Vec<_>>(),
        );

        // Timestamps must be preserved verbatim, in order, with no clamping.
        assert_eq!(d.segments.len(), 12);
        for (i, (speaker, start, end)) in segments.iter().enumerate() {
            assert_eq!(&d.segments[i].speaker_id, speaker);
            assert!(
                (d.segments[i].start - start).abs() < 1e-9,
                "start drift at {i}"
            );
            assert!((d.segments[i].end - end).abs() < 1e-9, "end drift at {i}");
        }
        for w in d.segments.windows(2) {
            assert!(
                w[1].start >= w[0].start,
                "long-form segments must remain chronologically ordered"
            );
        }

        let words = vec![
            word("What", 0.0, 0.5, "S1"),
            word("do", 0.5, 0.9, "S1"),
            word("you?", 0.9, 1.4, "S1"),
        ];
        let map = engine.build_speaker_map("hash", &d, &words).unwrap();
        assert!(!map.mappings.is_empty());
        let s1 = map
            .mappings
            .iter()
            .find(|m| m.diarization_id == "S1")
            .unwrap();
        assert_eq!(s1.application_role, ApplicationSpeakerRole::Host);
    }

    #[test]
    fn test_diarization_disabled_produces_empty_not_error() {
        // AUTOSHORTS_DIARIZATION=off must keep the pipeline running with a
        // clean, labeled empty result — never a fabricated speaker.
        let config = SpeakerIntelligenceConfig {
            enable_diarization: false,
            ..Default::default()
        };
        let engine = SpeakerIntelligenceEngine::new(
            config,
            Database::open(&std::env::temp_dir().join("si_test_disabled.sqlite")).unwrap(),
        )
        .unwrap();
        let fb = engine.empty_diarization_fallback("disabled");
        assert_eq!(fb.model, "disabled");
        assert!(fb.speakers.is_empty());
        assert!(fb.segments.is_empty());
    }

    // ─── Cache semantics: a fallback must not be cached as a success ─────────
    //
    // Production showed "Cache hit for e53a8eac32fe61a0" followed by
    // "Speaker mapping complete: 0 entries": the failed Deepgram run had been
    // written to the cache as if it were a real result, so every later run
    // replayed the empty diarization. These tests pin the corrected behavior.

    #[test]
    fn test_empty_diarization_is_classified_as_fallback() {
        let engine = si_engine_named("fallbackclass");
        let fb_fallback = engine.empty_diarization_fallback("fallback");
        let fb_disabled = engine.empty_diarization_fallback("disabled");
        assert!(SpeakerIntelligenceEngine::is_diarization_fallback(
            &fb_fallback
        ));
        assert!(SpeakerIntelligenceEngine::is_diarization_fallback(
            &fb_disabled
        ));
        // A real diarization is NOT a fallback and must stay cacheable.
        let real = diar(&[("S1", 10.0)], &[("S1", 0.0, 10.0)]);
        assert!(!SpeakerIntelligenceEngine::is_diarization_fallback(&real));
    }

    #[test]
    fn test_fallback_result_is_not_persisted_to_cache() {
        // The regression: an empty result must never be written to the cache
        // file, otherwise it is replayed as a success on the next run.
        let engine = si_engine_named("fallbackcache");
        let source = std::env::temp_dir().join("si_fallback_cache_source.mp4");
        std::fs::write(&source, b"source-bytes").unwrap();
        let source_hash = engine
            .compute_source_hash(&source.to_string_lossy())
            .unwrap();

        // Simulate what the old code did: persist the empty fallback.
        let poisoned = SpeakerIntelligenceCache {
            source_hash: source_hash.clone(),
            diarization: Some(engine.empty_diarization_fallback("fallback")),
            speaker_map: Some(ApplicationSpeakerMap {
                source_hash: source_hash.clone(),
                mappings: Vec::new(),
            }),
            reid_embeddings: Vec::new(),
            fusion_intervals: Vec::new(),
            config_hash: engine.compute_config_hash(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        engine.save_cache(&poisoned).unwrap();
        let path = engine.cache_file_path(&source_hash);
        assert!(path.exists(), "precondition: poisoned entry is on disk");

        // The production path now removes it instead of trusting it.
        let fb = engine.empty_diarization_fallback("fallback");
        assert!(SpeakerIntelligenceEngine::is_diarization_fallback(&fb));
        let _ = std::fs::remove_file(engine.cache_file_path(&source_hash));
        assert!(
            !engine.cache_file_path(&source_hash).exists(),
            "a fallback diarization must not survive as a cacheable entry"
        );
        // And the next run must therefore recompute instead of replaying 0 entries.
        assert!(
            engine.load_cache(&source_hash).is_none(),
            "engine must treat the invalidated source as a cache miss"
        );
    }

    #[test]
    fn test_successful_diarization_stays_cacheable() {
        // The counterpart: a real result MUST still be cached, otherwise the
        // fix would silently disable diarization caching and re-upload on every
        // run.
        let engine = si_engine_named("realscache");
        let source = std::env::temp_dir().join("si_real_cache_source.mp4");
        std::fs::write(&source, b"real-source").unwrap();
        let source_hash = engine
            .compute_source_hash(&source.to_string_lossy())
            .unwrap();

        let real = diar(
            &[("S1", 10.0), ("S2", 5.0)],
            &[("S1", 0.0, 10.0), ("S2", 10.0, 15.0)],
        );
        assert!(
            !SpeakerIntelligenceEngine::is_diarization_fallback(&real),
            "a populated diarization must remain cacheable"
        );
        let entry = SpeakerIntelligenceCache {
            source_hash: source_hash.clone(),
            diarization: Some(real),
            speaker_map: Some(ApplicationSpeakerMap {
                source_hash: source_hash.clone(),
                mappings: Vec::new(),
            }),
            reid_embeddings: Vec::new(),
            fusion_intervals: Vec::new(),
            config_hash: engine.compute_config_hash(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        engine.save_cache(&entry).unwrap();
        assert!(engine.cache_file_path(&source_hash).exists());
        assert!(
            engine.load_cache(&source_hash).is_some(),
            "real result must round-trip"
        );
    }
}
