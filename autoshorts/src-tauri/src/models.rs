use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentStatus {
    pub data_dir: String,
    pub has_ffmpeg: bool,
    pub has_ffprobe: bool,
    pub has_deepgram_key: bool,
    pub has_anthropic_key: bool,
    pub has_deepseek_key: bool,
    pub has_gemini_key: bool,
    pub has_openai_key: bool,
    pub has_openrouter_key: bool,
    pub has_groq_key: bool,
    pub llm_provider: String,
    pub has_local_whisper_model: bool,
    pub has_ollama: bool,
    pub has_ytdlp: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaProbe {
    pub duration_sec: Option<f64>,
    pub has_video: bool,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Project {
    pub id: String,
    pub name: Option<String>,
    pub source_path: String,
    pub source_duration: Option<f64>,
    pub status: String,
    pub transcription_mode: String,
    pub caption_style: Option<String>,
    pub framing_mode: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transcript {
    pub id: String,
    pub project_id: String,
    pub engine: String,
    pub raw_json: String,
    #[serde(default)]
    pub raw_transcript_json: Option<String>,
    pub language: Option<String>,
    pub created_at: String,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MetadataError {
    #[error("Corrupted candidate metadata JSON: {0}")]
    CorruptedJson(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub id: String,
    pub project_id: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub score: f64,
    pub hook: String,
    pub rationale: String,
    pub rank: i64,
    pub selected: bool,
    #[serde(default)]
    pub hook_start_sec: Option<f64>,
    #[serde(default)]
    pub hook_end_sec: Option<f64>,
    #[serde(default)]
    pub hook_confidence: Option<f64>,
    #[serde(default)]
    pub opening_context_score: Option<f64>,
    #[serde(default)]
    pub payoff_text: Option<String>,
    #[serde(default)]
    pub payoff_start_sec: Option<f64>,
    #[serde(default)]
    pub payoff_end_sec: Option<f64>,
    #[serde(default)]
    pub payoff_score: Option<f64>,
    #[serde(default)]
    pub payoff_completion: Option<bool>,
    #[serde(default)]
    pub metadata_json: Option<String>,
}

impl Candidate {
    pub fn try_metadata_draft(&self) -> Result<CandidateDraft, MetadataError> {
        match &self.metadata_json {
            None => Ok(self.reconstruct_legacy_draft()),
            Some(raw) if raw.trim().is_empty() => Ok(self.reconstruct_legacy_draft()),
            Some(raw) => serde_json::from_str::<CandidateDraft>(raw)
                .map_err(|e| MetadataError::CorruptedJson(format!("{}: {}", e, raw))),
        }
    }

    pub fn reconstruct_legacy_draft(&self) -> CandidateDraft {
        CandidateDraft {
            start: self.start_sec,
            end: self.end_sec,
            score: self.score,
            hook: self.hook.clone(),
            rationale: self.rationale.clone(),
            hook_start: self.hook_start_sec,
            hook_end: self.hook_end_sec,
            hook_confidence: self.hook_confidence,
            opening_context_score: self.opening_context_score,
            payoff_text: self.payoff_text.clone(),
            payoff_start: self.payoff_start_sec,
            payoff_end: self.payoff_end_sec,
            payoff_score: self.payoff_score,
            payoff_completion: self.payoff_completion,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Clip {
    pub id: String,
    pub candidate_id: String,
    pub status: String,
    pub output_path: Option<String>,
    pub face_track_json: Option<String>,
    pub caption_ass_path: Option<String>,
    pub render_log: Option<String>,
    #[serde(default)]
    pub applied_features: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipCopy {
    pub id: String,
    pub clip_id: String,
    pub platform: String,
    pub hook_text: Option<String>,
    pub caption_text: Option<String>,
    pub hashtags: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MultimodalStatus {
    pub status: String,            // "SUCCESS" | "PARTIAL" | "FAILED"
    pub visual_analysis: String,   // "SUCCESS" | "FAILED" | "SKIPPED"
    pub audio_analysis: String,    // "SUCCESS" | "FAILED" | "SKIPPED"
    pub temporal_analysis: String, // "SUCCESS" | "FAILED" | "SKIPPED"
    pub candidate_count_analyzed: usize,
    pub fallback_to_v92: bool,
    pub failure_reason: Option<String>,
    pub failed_modality: Option<String>,
    pub processing_time_ms: u64,
    pub source_duration: f64,
    pub analyzed_start: f64,
    pub analyzed_end: f64,
    pub uncovered_seconds: f64,
    pub windows_processed: usize,
    pub windows_failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectDetail {
    pub project: Project,
    pub transcript: Option<Transcript>,
    pub candidates: Vec<Candidate>,
    pub clips: Vec<Clip>,
    pub copy: Vec<ClipCopy>,
    pub multimodal_status: Option<MultimodalStatus>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WordCorrection {
    pub index: usize,
    pub raw_word: String,
    pub corrected_word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub rule: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptCorrectionMetadata {
    pub total_words: usize,
    pub corrections_count: usize,
    pub high_confidence_count: usize,
    pub rules_applied: Vec<String>,
    #[serde(default)]
    pub corrections: Vec<WordCorrection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedTranscript {
    pub language: String,
    pub duration: f64,
    pub speakers: Vec<String>,
    pub words: Vec<TranscriptWord>,
    pub segments: Vec<TranscriptSegment>,
    #[serde(default)]
    pub raw_words: Option<Vec<TranscriptWord>>,
    #[serde(default)]
    pub correction_metadata: Option<TranscriptCorrectionMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: Option<String>,
    pub text: String,
}

/// Transient LLM output — not persisted directly to DB.
/// `start`, `end`, `score`, `hook`, `rationale` map directly to the `candidates` table.
/// All other fields carry the structured Hook→Story→Payoff metadata for scoring and logging.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CandidateDraft {
    // Core (→ DB)
    pub start: f64,
    pub end: f64,
    pub score: f64,
    pub hook: String,
    pub rationale: String,

    // Structured timing (from LLM, transient)
    #[serde(default)]
    pub hook_start: Option<f64>,
    #[serde(default)]
    pub hook_end: Option<f64>,
    #[serde(default)]
    pub payoff_text: Option<String>,
    #[serde(default)]
    pub payoff_start: Option<f64>,
    #[serde(default)]
    pub payoff_end: Option<f64>,

    // Narrative classification
    /// e.g. "question_answer_lesson", "lesson_quote", "story_arc",
    ///       "insight_explanation", "emotional_moment"
    #[serde(default)]
    pub structure: Option<String>,

    // Quality scores (0.0–1.0)
    #[serde(default)]
    pub hook_score: Option<f64>,
    #[serde(default)]
    pub coherence_score: Option<f64>,
    #[serde(default)]
    pub payoff_score: Option<f64>,

    // Adaptive Question-Hook & Answer metadata (v3.2)
    #[serde(default)]
    pub question_hook_used: Option<bool>,
    #[serde(default)]
    pub question_start: Option<f64>,
    #[serde(default)]
    pub question_end: Option<f64>,
    #[serde(default)]
    pub question_text: Option<String>,
    #[serde(default)]
    pub question_hook_score: Option<f64>,
    #[serde(default)]
    pub question_relevance_score: Option<f64>,
    #[serde(default)]
    pub answer_start: Option<f64>,
    #[serde(default)]
    pub answer_end: Option<f64>,
    #[serde(default)]
    pub answer_text: Option<String>,
    #[serde(default)]
    pub answer_strength_score: Option<f64>,

    // Multilingual & Hook-First v9 metadata
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub script: Option<String>,
    #[serde(default)]
    pub hook_type: Option<String>,
    #[serde(default)]
    pub speakers: Option<Vec<String>>,
    #[serde(default)]
    pub summary: Option<String>,

    // Fine-grained scoring dimensions (0.0-1.0)
    #[serde(default)]
    pub curiosity_score: Option<f64>,
    #[serde(default)]
    pub emotional_impact_score: Option<f64>,
    #[serde(default)]
    pub surprise_score: Option<f64>,
    #[serde(default)]
    pub value_score: Option<f64>,
    #[serde(default)]
    pub story_quality_score: Option<f64>,
    #[serde(default)]
    pub context_completeness_score: Option<f64>,
    #[serde(default)]
    pub shareability_score: Option<f64>,

    // Semantic Conversation & Narrative Clip Engine v9.1 metadata
    #[serde(default)]
    pub hook_speaker: Option<String>,
    #[serde(default)]
    pub context_text: Option<String>,
    #[serde(default)]
    pub conversation_type: Option<String>,
    #[serde(default)]
    pub start_reason: Option<String>,
    #[serde(default)]
    pub end_reason: Option<String>,
    #[serde(default)]
    pub semantic_complete: Option<bool>,
    #[serde(default)]
    pub interviewer_hook_strength: Option<f64>,
    #[serde(default)]
    pub guest_hook_strength: Option<f64>,
    #[serde(default)]
    pub answer_completeness: Option<f64>,
    #[serde(default)]
    pub semantic_closure: Option<f64>,

    // Multimodal Hook Intelligence v9.3 metadata
    #[serde(default)]
    pub multimodal_verified: bool,
    #[serde(default)]
    pub multimodal_status: Option<String>,
    #[serde(default)]
    pub fallback_label: Option<String>,
    #[serde(default)]
    pub multimodal_score: Option<f64>,
    #[serde(default)]
    pub visual_score: Option<f64>,
    #[serde(default)]
    pub audio_score: Option<f64>,
    #[serde(default)]
    pub temporal_score: Option<f64>,
    #[serde(default)]
    pub semantic_score: Option<f64>,

    // Detailed Multimodal Signals
    #[serde(default)]
    pub visual_scene_change: Option<f64>,
    #[serde(default)]
    pub visual_reaction_strength: Option<f64>,
    #[serde(default)]
    pub visual_expression_change: Option<f64>,
    #[serde(default)]
    pub visual_gesture_strength: Option<f64>,
    #[serde(default)]
    pub visual_framing_change: Option<f64>,
    #[serde(default)]
    pub visual_saliency: Option<f64>,

    #[serde(default)]
    pub audio_energy_change: Option<f64>,
    #[serde(default)]
    pub audio_peak_strength: Option<f64>,
    #[serde(default)]
    pub audio_pitch_change: Option<f64>,
    #[serde(default)]
    pub audio_pause_emphasis: Option<f64>,
    #[serde(default)]
    pub audio_laughter: Option<f64>,

    #[serde(default)]
    pub temporal_escalation: Option<f64>,
    #[serde(default)]
    pub temporal_turning_point: Option<f64>,
    #[serde(default)]
    pub temporal_narrative_progression: Option<f64>,
    #[serde(default)]
    pub temporal_payoff_alignment: Option<f64>,

    #[serde(default)]
    pub multimodal_evidence: Option<Vec<String>>,

    // ==========================================
    // Phase 4 — PANNs reaction metadata (ADVISORY signal only)
    // JSON document of reaction events overlapping this candidate's range.
    // Purely informational: never read by ranking, boundary, or render logic.
    #[serde(default)]
    pub reactions_json: Option<String>,

    // ==========================================
    // v10.x Hook Intelligence Dimensions (R3)
    // ==========================================
    #[serde(default)]
    pub hook_topic_clarity: Option<f64>,
    #[serde(default)]
    pub hook_curiosity: Option<f64>,
    #[serde(default)]
    pub hook_relevance: Option<f64>,
    #[serde(default)]
    pub hook_contrast: Option<f64>,
    #[serde(default)]
    pub hook_confusion_penalty: Option<f64>,
    #[serde(default)]
    pub hook_delay_penalty: Option<f64>,
    #[serde(default)]
    pub hook_irrelevance_penalty: Option<f64>,
    #[serde(default)]
    pub hook_disinterest_penalty: Option<f64>,

    // ==========================================
    // v10.x Semantic Closure Dimensions (R3 & R4)
    // ==========================================
    #[serde(default)]
    pub continuation_probability: Option<f64>,
    #[serde(default)]
    pub breath_pause_risk: Option<f64>,
    #[serde(default)]
    pub mid_thought_risk: Option<f64>,
    #[serde(default)]
    pub closure_confidence: Option<f64>,
    #[serde(default)]
    pub answer_complete: Option<bool>,
    #[serde(default)]
    pub story_completeness: Option<bool>,
    #[serde(default)]
    pub payoff_completion: Option<bool>,
    #[serde(default)]
    pub ending_naturalness: Option<f64>,
    #[serde(default)]
    pub ending_type: Option<String>,

    // ==========================================
    // v10.1 Six-State Endpoint Classification
    // ==========================================
    /// Endpoint state classification from the LLM (or Rust enforcement).
    /// Deserialises from "breath_pause", "sentence_complete", etc.
    #[serde(default)]
    pub endpoint_state: Option<EndpointState>,

    /// Original LLM-suggested end timestamp BEFORE snap_to_semantic_boundaries.
    /// Transient — never written to the candidates DB table.
    #[serde(skip)]
    pub raw_llm_end: Option<f64>,

    // ==========================================
    // v10.x Verified Hook Pipeline Metadata
    // ==========================================
    #[serde(default)]
    pub hook_confidence: Option<f64>,
    #[serde(default)]
    pub opening_context_score: Option<f64>,
    #[serde(default)]
    pub opening_unresolved_reference: Option<String>,
    #[serde(default)]
    pub opening_continuation_marker: Option<String>,

    // ==========================================
    // Phase 4: Local VLM Candidate Scoring
    // ==========================================
    #[serde(default)]
    pub vlm_quality_score: Option<f64>,
    #[serde(default)]
    pub vlm_visual_engagement: Option<f64>,
    #[serde(default)]
    pub vlm_semantic_coherence: Option<f64>,
    #[serde(default)]
    pub vlm_production_quality: Option<f64>,
    #[serde(default)]
    pub vlm_highlight_relevance: Option<f64>,
    #[serde(default)]
    pub vlm_model: Option<String>,
    #[serde(default)]
    pub vlm_model_version: Option<String>,
    #[serde(default)]
    pub vlm_scored_at: Option<String>,
    #[serde(default)]
    pub vlm_evidence: Option<Vec<String>>,
}

/// Central configuration for the verified hook pipeline thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HookPipelineConfig {
    /// Soft penalty subtracted from score for discourse continuation openers (default: 0.04)
    pub continuation_penalty: f64,
    /// Penalty for ungrounded demonstratives without immediate noun (default: 0.03)
    pub demonstrative_penalty: f64,
    /// Minimum confidence for token alignment to be considered valid (default: 0.60)
    pub min_token_alignment_confidence: f64,
    /// Maximum allowable micro-breath lead-in before hook word (default: 0.15s, NEVER crosses sentence)
    pub max_start_breath_leadin_sec: f64,
    /// Hard reject when opening pronoun has zero antecedent and no host setup (default: true)
    pub hard_reject_unresolved_pronoun: bool,
}

impl Default for HookPipelineConfig {
    fn default() -> Self {
        Self {
            continuation_penalty: 0.04,
            demonstrative_penalty: 0.03,
            min_token_alignment_confidence: 0.60,
            max_start_breath_leadin_sec: 0.15,
            hard_reject_unresolved_pronoun: true,
        }
    }
}

/// Configuration for Universal Full-Timeline Windowed Discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WindowDiscoveryConfig {
    /// Videos under this duration (sec) use a single monolithic discovery prompt (default: 720.0s = 12 min)
    pub short_video_threshold_sec: f64,
    /// Window size for medium-length videos (12-45 min) (default: 480.0s = 8 min)
    pub medium_window_sec: f64,
    /// Overlap for medium-length videos (default: 90.0s = 1.5 min)
    pub medium_overlap_sec: f64,
    /// Window size for long videos (45 min - 2 hours) (default: 600.0s = 10 min)
    pub long_window_sec: f64,
    /// Overlap for long videos (default: 120.0s = 2 min)
    pub long_overlap_sec: f64,
    /// Window size for extra-long videos (>2 hours) (default: 180.0s = 3 min)
    pub extra_long_window_sec: f64,
    /// Overlap for extra-long videos (default: 30.0s = 0.5 min)
    pub extra_long_overlap_sec: f64,
    /// Maximum concurrent LLM requests (default: 3)
    pub max_concurrency: usize,
    /// Minimum target candidates requested per window (default: 2)
    pub candidates_per_window_min: usize,
    /// Maximum target candidates requested per window (default: 4)
    pub candidates_per_window_max: usize,
    /// Minimum composite score threshold to accept a candidate (default: 0.65)
    pub min_candidate_score: f64,
    /// Total maximum candidates selected for output (default: 12)
    pub max_total_selected: usize,
    /// Discovery mode: TimestampGeneration (default) or WindowScoring (REZE)
    pub discovery_mode: DiscoveryMode,
    /// Prompt version string for cache invalidation (bump on any scoring prompt change)
    pub scoring_prompt_version: String,
    /// Gaussian smoothing sigma for REZE aggregation (default: 1.5)
    pub score_smoothing_sigma: f64,
    /// Otsu threshold multiplier (default: 1.0)
    pub score_otsu_multiplier: f64,
    /// Minimum window score for Kadane selection (default: 0.35)
    pub score_kadane_min: f64,
    /// Valley drop ratio for Kadane span separation (default: 0.70 = 30% drop from peak forces new span)
    pub score_valley_drop_ratio: f64,
    /// Window size for REZE narrative scoring (default: 75.0s)
    pub reze_window_sec: f64,
    /// Overlap for REZE narrative scoring (default: 15.0s)
    pub reze_overlap_sec: f64,
    /// Restrict REZE window scoring to long-form content (> short_video_threshold_sec) (default: true)
    pub reze_long_form_only: bool,
    /// Enable targeted extraction inside detected highlight spans (default: true)
    pub reze_targeted_extraction: bool,
    /// Maximum candidate regions to extract from detected spans (default: 4)
    pub reze_max_extracted_regions: usize,
}

impl WindowDiscoveryConfig {
    pub fn window_and_overlap_for_duration(&self, duration: f64) -> (f64, f64) {
        if duration > 7200.0 {
            // Extra-long: >2 hours -> use finer 3-min windows
            (self.extra_long_window_sec, self.extra_long_overlap_sec)
        } else if duration > 2700.0 {
            // Long: 45 min - 2 hours -> use 10-min windows
            (self.long_window_sec, self.long_overlap_sec)
        } else {
            // Medium: 12-45 min -> use 8-min windows
            (self.medium_window_sec, self.medium_overlap_sec)
        }
    }

    /// Narrative window sizing for REZE window scoring:
    /// Discretizes the video into short-form narrative intervals (75s - 90s)
    /// so Kadane and Otsu can extract distinct short-form highlights.
    pub fn reze_window_and_overlap_for_duration(&self, duration: f64) -> (f64, f64) {
        if duration > 7200.0 {
            (self.extra_long_window_sec, self.extra_long_overlap_sec)
        } else {
            (self.reze_window_sec, self.reze_overlap_sec)
        }
    }
}

impl Default for WindowDiscoveryConfig {
    fn default() -> Self {
        Self {
            short_video_threshold_sec: 720.0,
            medium_window_sec: 480.0,
            medium_overlap_sec: 90.0,
            long_window_sec: 600.0,
            long_overlap_sec: 120.0,
            extra_long_window_sec: 90.0,
            extra_long_overlap_sec: 15.0,
            max_concurrency: 3,
            candidates_per_window_min: 2,
            candidates_per_window_max: 4,
            min_candidate_score: 0.65,
            max_total_selected: 12,
            discovery_mode: DiscoveryMode::TimestampGeneration,
            scoring_prompt_version: "v1.0".to_string(),
            score_smoothing_sigma: 0.5,
            score_otsu_multiplier: 1.0,
            score_kadane_min: 0.35,
            score_valley_drop_ratio: 0.70,
            reze_window_sec: 75.0,
            reze_overlap_sec: 15.0,
            reze_long_form_only: true,
            reze_targeted_extraction: true,
            reze_max_extracted_regions: 4,
        }
    }
}

/// Detailed diagnostics result for opening 1-3s hook context evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct OpeningContextResult {
    pub score: f64,
    pub penalty: f64,
    pub is_standalone: bool,
    pub has_unresolved_reference: bool,
    pub unresolved_reference: Option<String>,
    pub has_continuation_opener: bool,
    pub matched_marker: Option<String>,
    pub hard_reject: bool,
    pub explanation: String,
}

/// Six endpoint states (v10.1).
/// Ordered from weakest to strongest closure — only ANSWER_COMPLETE and above
/// are normally valid termination points for a conversational clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EndpointState {
    /// Short pause only — same speaker will continue. Never a valid endpoint.
    BreathPause,
    /// Grammatically complete sentence, but thought continues. Valid only when
    /// genuinely self-contained AND closure_confidence >= 0.80.
    SentenceComplete,
    /// Complete thought, but part of a larger arc (story, Q&A, lesson). Valid
    /// only when self-contained AND closure_confidence >= 0.80.
    ThoughtComplete,
    /// The triggering question is fully answered. Valid endpoint.
    AnswerComplete,
    /// Story arc (setup → conflict → resolution) is complete. Valid endpoint.
    StoryComplete,
    /// Punchline / lesson / emotional payoff has landed. Valid endpoint.
    PayoffComplete,
    /// Classification not provided by LLM; fall through to existing logic.
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClosureSignals {
    #[serde(default)]
    pub continuation_probability: Option<f64>,
    #[serde(default)]
    pub breath_pause_risk: Option<f64>,
    #[serde(default)]
    pub mid_thought_risk: Option<f64>,
    #[serde(default)]
    pub answer_complete: Option<bool>,
    #[serde(default)]
    pub story_completeness: Option<bool>,
    #[serde(default)]
    pub payoff_completion: Option<bool>,
    #[serde(default)]
    pub closure_confidence: Option<f64>,
    #[serde(default)]
    pub ending_type: Option<String>,
    #[serde(default)]
    pub answer_end: Option<f64>,
    #[serde(default)]
    pub payoff_end: Option<f64>,
    /// v10.1: six-state endpoint classification
    #[serde(default)]
    pub endpoint_state: Option<EndpointState>,
}

impl ClosureSignals {
    pub fn from_draft(draft: &CandidateDraft) -> Self {
        Self {
            continuation_probability: draft.continuation_probability,
            breath_pause_risk: draft.breath_pause_risk,
            mid_thought_risk: draft.mid_thought_risk,
            answer_complete: draft.answer_complete,
            story_completeness: draft.story_completeness,
            payoff_completion: draft.payoff_completion,
            closure_confidence: draft.closure_confidence,
            ending_type: draft.ending_type.clone(),
            answer_end: draft.answer_end,
            payoff_end: draft.payoff_end,
            endpoint_state: draft.endpoint_state.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionChunk {
    pub start_rel: f64,
    pub end_rel: f64,
    pub text: String,
}

/// REZE-style per-window scoring result.
/// The LLM answers only a binary/ordinal relevance question per window.
/// NO timestamps are emitted in scoring mode.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WindowScoreResult {
    /// Which window (0-indexed) this score applies to
    pub window_idx: usize,
    /// Window start time in seconds (for span reconstruction)
    pub window_start: f64,
    /// Window end time in seconds
    pub window_end: f64,
    /// Raw relevance score from LLM: 0.0 = not highlight, 1.0 = strong highlight
    pub raw_score: f64,
    /// Optional sub-scores
    pub hook_relevance: Option<f64>,
    pub narrative_completeness: Option<f64>,
    pub payoff_presence: Option<f64>,
    pub engagement_signal: Option<f64>,
    /// Provider that generated this score
    pub provider: String,
    /// Model name used
    pub model: String,
    /// Cache key for this score
    pub cache_key: String,
    /// Prompt version (for cache invalidation)
    pub prompt_version: String,
}

/// Feature flag / A-B control for candidate discovery mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMode {
    /// Current timestamp-generation path (backward-compatible, safe default)
    #[default]
    TimestampGeneration,
    /// REZE-style per-window scoring + deterministic aggregation
    WindowScoring,
}

// ─── Phase 2: Speaker Intelligence Types ──────────────────────────────────────

/// Result of speaker diarization for a source video.
/// Cached per source video hash to avoid re-running diarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerDiarizationResult {
    /// SHA256 hash of the source audio/video file
    pub source_hash: String,
    /// Model used: "deepgram-nova3", "pyannote-3.1", "whisperx"
    pub model: String,
    /// Model version string
    pub version: String,
    /// Diarized speakers with aggregate stats
    pub speakers: Vec<DiarizedSpeaker>,
    /// Contiguous speech segments with speaker labels
    pub segments: Vec<DiarizedSegment>,
    /// Overall confidence in the diarization (0.0–1.0)
    pub confidence: f64,
    /// ISO8601 timestamp when this result was generated
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiarizedSpeaker {
    /// Diarization-internal speaker ID (e.g., "S1", "S2")
    pub diarization_id: String,
    /// Total speech duration for this speaker (seconds)
    pub total_speech_sec: f64,
    /// Number of contiguous segments
    pub segment_count: usize,
    /// Average confidence across segments
    pub avg_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiarizedSegment {
    /// References DiarizedSpeaker.diarization_id
    pub speaker_id: String,
    /// Segment start time (absolute seconds from source start)
    pub start: f64,
    /// Segment end time (absolute seconds from source start)
    pub end: f64,
    /// Confidence for this segment (0.0–1.0)
    pub confidence: f64,
}

/// Mapping from diarization speaker IDs to application-level roles.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplicationSpeakerMap {
    /// Source video hash this mapping applies to
    pub source_hash: String,
    /// Per-speaker role assignments
    pub mappings: Vec<SpeakerMapping>,
}

/// Single speaker role mapping with evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerMapping {
    /// Diarization speaker ID (e.g., "S1")
    pub diarization_id: String,
    /// Application-level role
    pub application_role: ApplicationSpeakerRole,
    /// Confidence in this mapping (0.0–1.0)
    pub confidence: f64,
    /// Evidence supporting this mapping
    pub evidence: MappingEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApplicationSpeakerRole {
    Host,
    Guest,
    Unknown,
}

impl std::fmt::Display for ApplicationSpeakerRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplicationSpeakerRole::Host => write!(f, "host"),
            ApplicationSpeakerRole::Guest => write!(f, "guest"),
            ApplicationSpeakerRole::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MappingEvidence {
    /// Visual track ID this speaker maps to (from speaker_tracker.py)
    pub visual_track_id: Option<i32>,
    /// Ratio of total speaking time for this speaker
    pub speaking_time_ratio: f64,
    /// Whether this speaker spoke first in the conversation
    pub first_speaker: bool,
    /// Whether this speaker asks questions (question-asker heuristic)
    pub question_asker: bool,
    /// Whether this mapping came from explicit user override
    pub user_override: bool,
}

/// Active speaker state for a track over a time interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveSpeakerState {
    /// Visual track ID
    pub track_id: i32,
    /// Application speaker role (if mapped)
    pub application_role: Option<ApplicationSpeakerRole>,
    /// Diarization speaker ID (if mapped)
    pub diarization_id: Option<String>,
    /// Time interval (absolute source seconds)
    pub start: f64,
    pub end: f64,
    /// Audio evidence score (0.0–1.0)
    pub audio_evidence: f64,
    /// Visual evidence score (mouth motion, 0.0–1.0)
    pub visual_evidence: f64,
    /// Fused active-speaking probability (0.0–1.0)
    pub speaking_probability: f64,
    /// Confidence in the fusion result (0.0–1.0)
    pub confidence: f64,
    /// Sources that contributed to this decision
    pub evidence_sources: Vec<SpeakerEvidenceSource>,
}

/// Evidence source for active speaker decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerEvidenceSource {
    Diarization,
    VisualMouthMotion,
    ReIdMatch,
    TemporalContinuity,
    UserOverride,
}

/// Re-ID embedding for a visual track.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackReIdEmbedding {
    pub track_id: i32,
    /// Shot index the track belongs to (track IDs are per-shot; required for
    /// unambiguous gallery keying).
    #[serde(default)]
    pub shot_idx: i32,
    /// 512-dimensional float32 embedding (base64 encoded for JSON)
    pub embedding_b64: String,
    /// Model name: "osnet_x1_0", "osnet_x0_75", etc.
    pub model: String,
    /// Number of detections used to compute this embedding (EMA count)
    pub detection_count: usize,
    /// Timestamp of first detection (seconds from source start)
    pub first_seen_sec: f64,
    /// Timestamp of last detection (seconds from source start)
    pub last_seen_sec: f64,
    /// Timestamp of last update
    pub updated_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_draft_full_17_fields_deserialization() {
        let json_data = r#"{
            "start": 12.5,
            "end": 45.0,
            "score": 0.94,
            "hook": "Why you never wake up refreshed",
            "rationale": "Strong opening question and complete resolution",
            "hookTopicClarity": 0.95,
            "hookCuriosity": 0.90,
            "hookRelevance": 0.88,
            "hookContrast": 0.82,
            "hookConfusionPenalty": 0.05,
            "hookDelayPenalty": 0.0,
            "hookIrrelevancePenalty": 0.0,
            "hookDisinterestPenalty": 0.10,
            "continuationProbability": 0.12,
            "breathPauseRisk": 0.08,
            "midThoughtRisk": 0.04,
            "closureConfidence": 0.96,
            "answerComplete": true,
            "storyCompleteness": true,
            "payoffCompletion": true,
            "endingNaturalness": 0.91,
            "endingType": "answer_completion",
            "endpointState": "answer_complete",
            "answerEnd": 44.5,
            "payoffEnd": 44.5
        }"#;

        let draft: CandidateDraft =
            serde_json::from_str(json_data).expect("Failed to deserialize full CandidateDraft");

        assert_eq!(draft.start, 12.5);
        assert_eq!(draft.end, 45.0);
        assert_eq!(draft.score, 0.94);
        assert_eq!(draft.hook, "Why you never wake up refreshed");
        assert_eq!(
            draft.rationale,
            "Strong opening question and complete resolution"
        );

        // 8 Hook dimensions
        assert_eq!(draft.hook_topic_clarity, Some(0.95));
        assert_eq!(draft.hook_curiosity, Some(0.90));
        assert_eq!(draft.hook_relevance, Some(0.88));
        assert_eq!(draft.hook_contrast, Some(0.82));
        assert_eq!(draft.hook_confusion_penalty, Some(0.05));
        assert_eq!(draft.hook_delay_penalty, Some(0.0));
        assert_eq!(draft.hook_irrelevance_penalty, Some(0.0));
        assert_eq!(draft.hook_disinterest_penalty, Some(0.10));

        // 9 Closure dimensions
        assert_eq!(draft.continuation_probability, Some(0.12));
        assert_eq!(draft.breath_pause_risk, Some(0.08));
        assert_eq!(draft.mid_thought_risk, Some(0.04));
        assert_eq!(draft.closure_confidence, Some(0.96));
        assert_eq!(draft.answer_complete, Some(true));
        assert_eq!(draft.story_completeness, Some(true));
        assert_eq!(draft.payoff_completion, Some(true));
        assert_eq!(draft.ending_naturalness, Some(0.91));
        assert_eq!(draft.ending_type.as_deref(), Some("answer_completion"));
        assert_eq!(draft.answer_end, Some(44.5));
        assert_eq!(draft.payoff_end, Some(44.5));

        // v10.1: endpoint_state
        assert_eq!(draft.endpoint_state, Some(EndpointState::AnswerComplete));
        // raw_llm_end is #[serde(skip)] — always None after deserialization
        assert!(draft.raw_llm_end.is_none());
    }

    #[test]
    fn test_candidate_draft_legacy_deserialization_defaults() {
        let legacy_json = r#"{
            "start": 5.0,
            "end": 25.0,
            "score": 0.80,
            "hook": "Legacy candidate",
            "rationale": "Legacy test rationale"
        }"#;

        let draft: CandidateDraft =
            serde_json::from_str(legacy_json).expect("Failed to deserialize legacy CandidateDraft");

        assert_eq!(draft.start, 5.0);
        assert_eq!(draft.end, 25.0);
        assert_eq!(draft.score, 0.80);
        assert_eq!(draft.hook, "Legacy candidate");

        // All new fields must default to None
        assert!(draft.hook_topic_clarity.is_none());
        assert!(draft.hook_curiosity.is_none());
        assert!(draft.hook_relevance.is_none());
        assert!(draft.hook_contrast.is_none());
        assert!(draft.hook_confusion_penalty.is_none());
        assert!(draft.hook_delay_penalty.is_none());
        assert!(draft.hook_irrelevance_penalty.is_none());
        assert!(draft.hook_disinterest_penalty.is_none());

        assert!(draft.continuation_probability.is_none());
        assert!(draft.breath_pause_risk.is_none());
        assert!(draft.mid_thought_risk.is_none());
        assert!(draft.closure_confidence.is_none());
        assert!(draft.answer_complete.is_none());
        assert!(draft.story_completeness.is_none());
        assert!(draft.payoff_completion.is_none());
        assert!(draft.ending_naturalness.is_none());
        assert!(draft.ending_type.is_none());

        // v10.1 fields
        assert!(draft.endpoint_state.is_none());
        assert!(draft.raw_llm_end.is_none());
    }

    #[test]
    fn test_endpoint_state_serde_variants() {
        // All 7 variants must round-trip correctly
        let cases = vec![
            (EndpointState::BreathPause, "\"breath_pause\""),
            (EndpointState::SentenceComplete, "\"sentence_complete\""),
            (EndpointState::ThoughtComplete, "\"thought_complete\""),
            (EndpointState::AnswerComplete, "\"answer_complete\""),
            (EndpointState::StoryComplete, "\"story_complete\""),
            (EndpointState::PayoffComplete, "\"payoff_complete\""),
            (EndpointState::Unknown, "\"unknown\""),
        ];
        for (variant, expected_json) in cases {
            let serialized = serde_json::to_string(&variant).unwrap();
            assert_eq!(
                serialized, expected_json,
                "Serialization mismatch for {:?}",
                variant
            );
            let deserialized: EndpointState = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, variant, "Round-trip failed for {:?}", variant);
        }
    }

    #[test]
    fn test_endpoint_state_default_is_unknown() {
        let state = EndpointState::default();
        assert_eq!(state, EndpointState::Unknown);
    }

    #[test]
    fn test_closure_signals_from_draft_mapping() {
        let draft = CandidateDraft {
            start: 10.0,
            end: 40.0,
            score: 0.88,
            hook: "Test hook".to_string(),
            rationale: "Test rationale".to_string(),
            continuation_probability: Some(0.85),
            breath_pause_risk: Some(0.70),
            mid_thought_risk: Some(0.65),
            closure_confidence: Some(0.40),
            answer_complete: Some(false),
            story_completeness: Some(true),
            payoff_completion: Some(false),
            ending_naturalness: Some(0.60),
            ending_type: Some("sentence_completion".to_string()),
            endpoint_state: Some(EndpointState::SentenceComplete),
            answer_end: Some(38.0),
            payoff_end: Some(42.0),
            raw_llm_end: Some(40.0),
            ..Default::default()
        };

        let signals = ClosureSignals::from_draft(&draft);

        assert_eq!(signals.continuation_probability, Some(0.85));
        assert_eq!(signals.breath_pause_risk, Some(0.70));
        assert_eq!(signals.mid_thought_risk, Some(0.65));
        assert_eq!(signals.closure_confidence, Some(0.40));
        assert_eq!(signals.answer_complete, Some(false));
        assert_eq!(signals.story_completeness, Some(true));
        assert_eq!(signals.payoff_completion, Some(false));
        assert_eq!(signals.ending_type.as_deref(), Some("sentence_completion"));
        assert_eq!(
            signals.endpoint_state,
            Some(EndpointState::SentenceComplete)
        );
        assert_eq!(signals.answer_end, Some(38.0));
        assert_eq!(signals.payoff_end, Some(42.0));
    }

    #[test]
    fn test_closure_signals_from_draft_defaults() {
        let draft = CandidateDraft::default();
        let signals = ClosureSignals::from_draft(&draft);

        assert!(signals.continuation_probability.is_none());
        assert!(signals.breath_pause_risk.is_none());
        assert!(signals.mid_thought_risk.is_none());
        assert!(signals.answer_complete.is_none());
        assert!(signals.story_completeness.is_none());
        assert!(signals.payoff_completion.is_none());
        assert!(signals.closure_confidence.is_none());
        assert!(signals.ending_type.is_none());
        assert!(signals.endpoint_state.is_none());
        assert!(signals.answer_end.is_none());
        assert!(signals.payoff_end.is_none());
    }

    #[test]
    fn test_closure_signals_serde_camel_case() {
        let signals = ClosureSignals {
            continuation_probability: Some(0.75),
            breath_pause_risk: Some(0.30),
            mid_thought_risk: Some(0.20),
            answer_complete: Some(true),
            story_completeness: Some(true),
            payoff_completion: Some(true),
            closure_confidence: Some(0.95),
            ending_type: Some("punchline".to_string()),
            answer_end: Some(50.0),
            payoff_end: Some(52.0),
            endpoint_state: Some(EndpointState::PayoffComplete),
        };

        let json_str = serde_json::to_string(&signals).expect("Failed to serialize ClosureSignals");
        assert!(json_str.contains("continuationProbability"));
        assert!(json_str.contains("breathPauseRisk"));
        assert!(json_str.contains("midThoughtRisk"));
        assert!(json_str.contains("answerComplete"));
        assert!(json_str.contains("storyCompleteness"));
        assert!(json_str.contains("payoffCompletion"));
        assert!(json_str.contains("closureConfidence"));
        assert!(json_str.contains("endpointState"));

        let deserialized: ClosureSignals =
            serde_json::from_str(&json_str).expect("Failed to deserialize ClosureSignals");
        assert_eq!(deserialized, signals);
    }

    #[test]
    fn test_candidate_try_metadata_draft_success() {
        let draft = CandidateDraft {
            start: 10.0,
            end: 40.0,
            score: 0.9,
            hook: "Test hook".into(),
            rationale: "Test rationale".into(),
            structure: Some("story_arc".into()),
            hook_score: Some(0.95),
            closure_confidence: Some(0.88),
            endpoint_state: Some(EndpointState::AnswerComplete),
            ..Default::default()
        };

        let json = serde_json::to_string(&draft).expect("serialize draft");
        let candidate = Candidate {
            id: "c1".into(),
            project_id: "p1".into(),
            start_sec: 10.0,
            end_sec: 40.0,
            score: 0.9,
            hook: "Test hook".into(),
            rationale: "Test rationale".into(),
            rank: 1,
            selected: true,
            hook_start_sec: None,
            hook_end_sec: None,
            hook_confidence: None,
            opening_context_score: None,
            payoff_text: None,
            payoff_start_sec: None,
            payoff_end_sec: None,
            payoff_score: None,
            payoff_completion: None,
            metadata_json: Some(json),
        };

        let restored = candidate.try_metadata_draft().expect("parse draft");
        assert_eq!(restored, draft);
    }

    #[test]
    fn test_candidate_try_metadata_draft_none_and_whitespace_reconstructs_legacy() {
        let candidate = Candidate {
            id: "c1".into(),
            project_id: "p1".into(),
            start_sec: 15.0,
            end_sec: 45.0,
            score: 0.88,
            hook: "Legacy hook".into(),
            rationale: "Legacy rationale".into(),
            rank: 1,
            selected: true,
            hook_start_sec: Some(15.0),
            hook_end_sec: Some(18.0),
            hook_confidence: Some(0.92),
            opening_context_score: Some(0.85),
            payoff_text: Some("Legacy payoff".into()),
            payoff_start_sec: Some(40.0),
            payoff_end_sec: Some(45.0),
            payoff_score: Some(0.90),
            payoff_completion: Some(true),
            metadata_json: None,
        };

        let legacy_draft = candidate.try_metadata_draft().expect("reconstruct legacy");
        assert_eq!(legacy_draft.start, 15.0);
        assert_eq!(legacy_draft.end, 45.0);
        assert_eq!(legacy_draft.score, 0.88);
        assert_eq!(legacy_draft.hook, "Legacy hook");
        assert_eq!(legacy_draft.rationale, "Legacy rationale");
        assert_eq!(legacy_draft.hook_start, Some(15.0));
        assert_eq!(legacy_draft.hook_end, Some(18.0));
        assert_eq!(legacy_draft.hook_confidence, Some(0.92));
        assert_eq!(legacy_draft.opening_context_score, Some(0.85));
        assert_eq!(legacy_draft.payoff_text, Some("Legacy payoff".into()));
        assert_eq!(legacy_draft.payoff_start, Some(40.0));
        assert_eq!(legacy_draft.payoff_end, Some(45.0));
        assert_eq!(legacy_draft.payoff_score, Some(0.90));
        assert_eq!(legacy_draft.payoff_completion, Some(true));

        let mut candidate_ws = candidate.clone();
        candidate_ws.metadata_json = Some("   \t\n  ".into());
        let legacy_draft_ws = candidate_ws
            .try_metadata_draft()
            .expect("reconstruct legacy on whitespace");
        assert_eq!(legacy_draft_ws, legacy_draft);
    }

    #[test]
    fn test_candidate_try_metadata_draft_corrupted_json_returns_error() {
        let candidate = Candidate {
            id: "c1".into(),
            project_id: "p1".into(),
            start_sec: 10.0,
            end_sec: 40.0,
            score: 0.9,
            hook: "Test hook".into(),
            rationale: "Test rationale".into(),
            rank: 1,
            selected: true,
            hook_start_sec: None,
            hook_end_sec: None,
            hook_confidence: None,
            opening_context_score: None,
            payoff_text: None,
            payoff_start_sec: None,
            payoff_end_sec: None,
            payoff_score: None,
            payoff_completion: None,
            metadata_json: Some("{ corrupt json: [not valid".into()),
        };

        let result = candidate.try_metadata_draft();
        match result {
            Err(MetadataError::CorruptedJson(err_msg)) => {
                assert!(err_msg.contains("{ corrupt json: [not valid"));
            }
            Ok(_) => panic!("Expected Err(MetadataError::CorruptedJson), got Ok"),
        }
    }

    #[test]
    fn test_window_score_result_default_fields() {
        let result = WindowScoreResult::default();
        assert_eq!(result.window_idx, 0);
        assert_eq!(result.raw_score, 0.0);
        assert!(result.hook_relevance.is_none());
        assert!(result.cache_key.is_empty());
    }

    #[test]
    fn test_discovery_mode_serde_roundtrip() {
        let mode = DiscoveryMode::WindowScoring;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"window_scoring\"");
        let back: DiscoveryMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, DiscoveryMode::WindowScoring);

        let default_mode = DiscoveryMode::default();
        assert_eq!(default_mode, DiscoveryMode::TimestampGeneration);
        let json2 = serde_json::to_string(&default_mode).unwrap();
        assert_eq!(json2, "\"timestamp_generation\"");
    }

    #[test]
    fn test_window_discovery_config_includes_mode_and_scoring_fields() {
        let config = WindowDiscoveryConfig::default();
        assert_eq!(config.discovery_mode, DiscoveryMode::TimestampGeneration);
        assert_eq!(config.scoring_prompt_version, "v1.0");
        assert!((config.score_smoothing_sigma - 0.5).abs() < 0.001);
        assert!((config.score_otsu_multiplier - 1.0).abs() < 0.001);
        assert!((config.score_kadane_min - 0.35).abs() < 0.001);
        assert!(config.reze_long_form_only);
        assert!(config.reze_targeted_extraction);
        assert_eq!(config.reze_max_extracted_regions, 4);
        // Existing fields must still be correct
        assert_eq!(config.max_concurrency, 3);
        assert!((config.short_video_threshold_sec - 720.0).abs() < 0.001);
    }
}
