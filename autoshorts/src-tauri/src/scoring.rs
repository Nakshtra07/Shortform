//! AutoClip-style holistic candidate scoring backend (NVIDIA NIM).
//!
//! Evaluates the full candidate pool produced by discovery + boundary
//! processing with a single LLM scorer that judges each candidate
//! holistically on four dimensions and returns ONE authoritative
//! `final_score` in `[0.0, 1.0]` plus a concise `recommend_reason`.
//!
//! Authority contract (explicit `autoclip` mode only):
//! - The accepted `final_score` REPLACES the discovery-level
//!   `CandidateDraft.score` and becomes the authoritative ranking score.
//! - `score_source = "llm"` marks genuine model-produced scores.
//! - Omitted/invalid/unrecoverable results become `score = 0.0` with
//!   `score_source = "fallback"` and are NEVER eligible for selection.
//! - The legacy `composite_score()` path is untouched and remains the
//!   authority in `legacy` mode (the default).
//!
//! Alignment contract:
//! - Every candidate in a scoring run receives a unique transient identity
//!   `cand-%04d` derived from its position in the scoring pool.
//! - Results are aligned by EXACT identity, never by response order.
//! - Duplicate/ambiguous ids are not assigned; unknown ids are ignored;
//!   missing ids stay unscored.

use crate::models::{CandidateDraft, TranscriptWord};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Version of the final-score prompt + output contract. Bump on any change
/// to the prompt text, input construction, or expected output schema so the
/// score cache and any live evidence are invalidated coherently.
pub const FINAL_SCORE_PROMPT_VERSION: &str = "v2.0";

/// Default eligibility threshold for AutoClip mode (historic 11.1 value).
/// NOT calibrated on a fixed validation set — see
/// docs/development/AUTOCLIP_SCORING_BACKEND.md.
pub const DEFAULT_RECOMMEND_THRESHOLD: f64 = 0.65;

/// Default Top-N selection limit (`AUTOSHORTS_TOP_N <= 0` disables the cap).
pub const DEFAULT_TOP_N: usize = 12;

/// Default holistic scoring model on NVIDIA NIM (historic 11.1 value).
pub const DEFAULT_SCORER_MODEL: &str = "nvidia/nemotron-3-ultra-550b-a55b";

/// Default NVIDIA NIM OpenAI-compatible base URL.
pub const DEFAULT_SCORER_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

const MAX_TRANSCRIPT_EXCERPT_CHARS: usize = 1200;
const MAX_EVIDENCE_ITEMS: usize = 3;
const MAX_EVIDENCE_CHARS: usize = 160;
const MAX_SCORING_RETRIES: u32 = 3;
const SCORING_BATCH_SIZE: usize = 6;
const SCORING_CONCURRENCY: usize = 2;
const SCORING_CONNECT_TIMEOUT_SEC: u64 = 30;
const SCORING_REQUEST_TIMEOUT_SEC: u64 = 180;
const SCORING_TEMPERATURE: f64 = 0.15;
const SCORING_MAX_TOKENS: u64 = 4096;

/// Multimodal fallback marker injected by `media::analyze_multimodal_hook_signals`
/// when evidence is genuinely unavailable. It is not evidence and must never be
/// sent to the scorer as if it were.
const MULTIMODAL_FALLBACK_MARKER: &str =
    "Multimodal evidence unavailable; ranking uses v9.2 semantic fallback.";

// ---------------------------------------------------------------------------
// Backend switch
// ---------------------------------------------------------------------------

/// Authoritative scoring backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringBackend {
    /// Original composite-score authority (default).
    Legacy,
    /// NVIDIA NIM holistic `final_score` authority.
    AutoClip,
}

/// Resolve `AUTOSHORTS_SCORING_BACKEND`. Historical default is `legacy`.
/// Unknown values are a hard configuration error — never a silent substitution.
pub fn resolve_scoring_backend() -> Result<ScoringBackend, String> {
    match std::env::var("AUTOSHORTS_SCORING_BACKEND") {
        Ok(raw) => {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "" | "legacy" => Ok(ScoringBackend::Legacy),
                "autoclip" => Ok(ScoringBackend::AutoClip),
                other => Err(format!(
                    "[AutoClip Scoring] Invalid AUTOSHORTS_SCORING_BACKEND value '{other}'. \
                     Supported values: 'legacy' (default) or 'autoclip'."
                )),
            }
        }
        Err(_) => Ok(ScoringBackend::Legacy),
    }
}

/// Resolve the scoring backend for ONE generate-candidates call.
///
/// `scoring_mode` is the explicit per-call UI selection (the scoring-mode
/// selector in the import/caption-style dialog):
/// - `None` or empty → the env default (`resolve_scoring_backend`) applies.
/// - `"legacy"` → force the legacy backend for this call, even when the env
///   default is `autoclip`.
/// - `"autoclip"` → force the AutoClip backend for this call, even when the
///   env default is `legacy` (the production default).
/// - any other value is a hard configuration error — never a silent
///   substitution.
pub fn resolve_scoring_backend_with_override(
    scoring_mode: Option<&str>,
) -> Result<ScoringBackend, String> {
    match scoring_mode {
        None => resolve_scoring_backend(),
        Some(raw) => {
            let trimmed = raw.trim().to_ascii_lowercase();
            match trimmed.as_str() {
                "" => resolve_scoring_backend(),
                "legacy" => Ok(ScoringBackend::Legacy),
                "autoclip" => Ok(ScoringBackend::AutoClip),
                other => Err(format!(
                    "[AutoClip Scoring] Invalid scoring-mode selection '{other}'. \
                     Supported values: 'legacy' or 'autoclip'."
                )),
            }
        }
    }
}

/// Verified NVIDIA scorer credential for a call resolved to the AutoClip
/// backend. A missing credential is a HARD error before any discovery work —
/// AutoClip mode never silently falls back to the legacy backend.
pub fn resolve_scorer_credential(backend: ScoringBackend) -> Result<Option<String>, String> {
    match backend {
        ScoringBackend::Legacy => Ok(None),
        ScoringBackend::AutoClip => {
            let key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
            if key.trim().is_empty() {
                return Err(
                    "[AutoClip Scoring] AutoClip Scoring (NVIDIA) requires NVIDIA_API_KEY \
                     to be configured (environment or .env). Fix the key, or select \
                     Legacy Scoring instead."
                        .to_string(),
                );
            }
            Ok(Some(key))
        }
    }
}

/// One-call scoring authority used by `generate_candidates`: the explicit UI
/// selection (`scoring_mode`) overrides the env default for this call only,
/// and an AutoClip call must carry a verified NVIDIA credential before any
/// discovery work is spent.
pub fn resolve_scoring_authority_for_call(
    scoring_mode: Option<&str>,
) -> Result<(ScoringBackend, Option<String>), String> {
    let backend = resolve_scoring_backend_with_override(scoring_mode)?;
    let credential = resolve_scorer_credential(backend)?;
    Ok((backend, credential))
}

/// Resolve the AutoClip eligibility threshold.
/// `AUTOCLIP_RECOMMEND_THRESHOLD` overrides the documented default of 0.65.
pub fn resolve_recommend_threshold() -> Result<f64, String> {
    match std::env::var("AUTOCLIP_RECOMMEND_THRESHOLD") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(DEFAULT_RECOMMEND_THRESHOLD);
            }
            match trimmed.parse::<f64>() {
                Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => Ok(v),
                _ => Err(format!(
                    "[AutoClip Scoring] Invalid AUTOCLIP_RECOMMEND_THRESHOLD '{trimmed}'. \
                     Expected a finite number in [0.0, 1.0]."
                )),
            }
        }
        Err(_) => Ok(DEFAULT_RECOMMEND_THRESHOLD),
    }
}

/// Resolve the effective Top-N selection limit.
/// `None` means the cutoff is disabled: select ALL eligible candidates.
pub fn resolve_top_n() -> Result<Option<usize>, String> {
    match std::env::var("AUTOSHORTS_TOP_N") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(Some(DEFAULT_TOP_N));
            }
            match trimmed.parse::<i64>() {
                Ok(n) if n > 0 => Ok(Some(n as usize)),
                Ok(_) => Ok(None),
                Err(_) => Err(format!(
                    "[AutoClip Scoring] Invalid AUTOSHORTS_TOP_N '{trimmed}'. \
                     Expected an integer (<= 0 disables the cutoff)."
                )),
            }
        }
        Err(_) => Ok(Some(DEFAULT_TOP_N)),
    }
}

// ---------------------------------------------------------------------------
// Scorer configuration
// ---------------------------------------------------------------------------

/// Bounded NVIDIA NIM scorer configuration (historic 11.1 values).
#[derive(Debug, Clone)]
pub struct AutoClipScorerConfig {
    /// Full OpenAI-compatible chat-completions endpoint URL.
    pub endpoint: String,
    pub model: String,
    /// Reasoning/thinking toggle (historic default: off).
    pub thinking: bool,
    pub temperature: f64,
    pub max_tokens: u64,
    pub batch_size: usize,
    pub concurrency: usize,
    pub connect_timeout_sec: u64,
    pub request_timeout_sec: u64,
}

impl Default for AutoClipScorerConfig {
    fn default() -> Self {
        Self {
            endpoint: format!("{DEFAULT_SCORER_BASE_URL}/chat/completions"),
            model: DEFAULT_SCORER_MODEL.to_string(),
            thinking: false,
            temperature: SCORING_TEMPERATURE,
            max_tokens: SCORING_MAX_TOKENS,
            batch_size: SCORING_BATCH_SIZE,
            concurrency: SCORING_CONCURRENCY,
            connect_timeout_sec: SCORING_CONNECT_TIMEOUT_SEC,
            request_timeout_sec: SCORING_REQUEST_TIMEOUT_SEC,
        }
    }
}

impl AutoClipScorerConfig {
    /// Resolve from the environment. Overrides:
    /// - `NVIDIA_SCORER_BASE_URL` (base URL; `/chat/completions` appended when missing)
    /// - `AUTOSHORTS_SCORER_MODEL`
    /// - `AUTOSHORTS_SCORER_THINKING` (default off)
    /// - `AUTOSHORTS_SCORER_MAX_TOKENS` (default 4096)
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(base) = std::env::var("NVIDIA_SCORER_BASE_URL") {
            let base = base.trim().trim_end_matches('/').to_string();
            if !base.is_empty() {
                config.endpoint = if base.ends_with("/chat/completions") {
                    base
                } else {
                    format!("{base}/chat/completions")
                };
            }
        }
        if let Ok(model) = std::env::var("AUTOSHORTS_SCORER_MODEL") {
            let model = model.trim();
            if !model.is_empty() {
                config.model = model.to_string();
            }
        }
        if let Ok(thinking) = std::env::var("AUTOSHORTS_SCORER_THINKING") {
            let flag = thinking.trim().to_ascii_lowercase();
            if matches!(flag.as_str(), "1" | "true" | "on" | "yes") {
                config.thinking = true;
            }
        }
        if let Ok(raw) = std::env::var("AUTOSHORTS_SCORER_MAX_TOKENS") {
            if let Ok(max_tokens) = raw.trim().parse::<u64>() {
                if max_tokens > 0 {
                    config.max_tokens = max_tokens;
                }
            }
        }
        config
    }
}

// ---------------------------------------------------------------------------
// Transient scoring identity
// ---------------------------------------------------------------------------

/// Unique transient identity for one candidate within a scoring run.
/// Derived from the candidate's position in the scoring pool — never from
/// its score — and stable across retries and batch boundaries.
pub fn transient_scoring_id(pool_position: usize) -> String {
    format!("cand-{:04}", pool_position)
}

// ---------------------------------------------------------------------------
// Scoring input construction
// ---------------------------------------------------------------------------

/// One candidate's holistic scoring input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringCandidateInput {
    /// Transient scoring identity (`cand-%04d`).
    pub id: String,
    /// AutoClip `outline` analog: the AutoShorts hook.
    pub hook: String,
    /// AutoClip `content` analog: the AutoShorts rationale/context.
    pub context: String,
    /// Timestamps as floating-point seconds.
    pub start_seconds: f64,
    pub end_seconds: f64,
    /// Word-aligned transcript excerpt (capped at 1200 characters).
    pub transcript: String,
    /// Up to 3 naturally available multimodal evidence strings, each capped
    /// at 160 characters. Empty when no genuine evidence exists.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
}

/// Word-aligned transcript excerpt for `[start, end]`, capped at
/// `MAX_TRANSCRIPT_EXCERPT_CHARS` characters on a character boundary.
pub fn transcript_excerpt(
    words: &[TranscriptWord],
    start: f64,
    end: f64,
    max_chars: usize,
) -> String {
    let mut excerpt = String::new();
    for word in words {
        // Include a word when its [start, end) overlaps the candidate range.
        if word.end <= start || word.start >= end {
            continue;
        }
        if !excerpt.is_empty() {
            excerpt.push(' ');
        }
        excerpt.push_str(&word.text);
        if excerpt.chars().count() >= max_chars {
            break;
        }
    }
    excerpt.chars().take(max_chars).collect()
}

/// Collect naturally available multimodal evidence: up to
/// [`MAX_EVIDENCE_ITEMS`] strings, each capped at [`MAX_EVIDENCE_CHARS`].
/// The known fallback marker is filtered out — unavailable evidence is not
/// fabricated evidence.
pub fn collect_evidence(draft: &CandidateDraft) -> Vec<String> {
    draft
        .multimodal_evidence
        .as_ref()
        .map(|items| {
            items
                .iter()
                .filter(|e| {
                    let trimmed = e.trim();
                    !trimmed.is_empty() && trimmed != MULTIMODAL_FALLBACK_MARKER
                })
                .take(MAX_EVIDENCE_ITEMS)
                .map(|e| e.chars().take(MAX_EVIDENCE_CHARS).collect::<String>())
                .collect()
        })
        .unwrap_or_default()
}

/// Build the scoring input list for the whole pool, assigning each candidate
/// its transient identity by pool position.
pub fn build_scoring_inputs(drafts: &[CandidateDraft], words: &[TranscriptWord]) -> Vec<ScoringCandidateInput> {
    drafts
        .iter()
        .enumerate()
        .map(|(position, draft)| ScoringCandidateInput {
            id: transient_scoring_id(position),
            hook: draft.hook.clone(),
            context: draft.rationale.clone(),
            start_seconds: draft.start,
            end_seconds: draft.end,
            transcript: transcript_excerpt(words, draft.start, draft.end, MAX_TRANSCRIPT_EXCERPT_CHARS),
            evidence: collect_evidence(draft),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Prompt
// ---------------------------------------------------------------------------

/// Structured, stable holistic-scoring prompt with explicit scoring
/// definitions, the exact output schema and a no-fabrication instruction.
pub fn build_final_score_prompt(inputs: &[ScoringCandidateInput]) -> String {
    let payload = serde_json::to_string_pretty(inputs).unwrap_or_else(|_| "[]".to_string());
    format!(
        r#"You are an expert short-form video strategist. You will evaluate candidate clips extracted from a longer video transcript.

Evaluate each candidate HOLISTICALLY across four dimensions:
1. Information value: Does it deliver a unique insight, knowledge, or high information density?
2. Emotional resonance: Does it provoke a strong emotion (surprise, joy, anger, curiosity, empathy)? Is the stance clear?
3. Shareability: Does it contain quotable lines or discussion-worthy ideas people would forward?
4. Structural completeness: Does the clip stand alone with a clear opening and a satisfying payoff?

Weigh all four dimensions together and produce ONE authoritative final_score in the range 0.0 to 1.0, plus one concise recommend_reason (a single sentence of roughly 15-30 words) explaining the score.

Candidates (JSON array; timestamps are floating-point seconds):
{payload}

Required output — a JSON array with exactly one object per evaluated candidate and NO other text:
[
  {{"id": "<the exact input id>", "final_score": <0.0-1.0>, "recommend_reason": "<one concise sentence>"}}
]

Rules:
- Echo the exact id of each candidate you evaluated. Never rename, drop, or invent ids.
- final_score must be a finite number between 0.0 and 1.0.
- Base the score and the reason strictly on the provided transcript, hook, context and evidence. Never fabricate content.
- If a candidate lacks substance, completeness, or payoff, give it a LOW score; never invent a high score to fill a quota.
- Respond with the complete JSON array only — no markdown fences, no commentary."#
    )
}

// ---------------------------------------------------------------------------
// Score normalization
// ---------------------------------------------------------------------------

/// Normalize a model-produced score value into `[0.0, 1.0]`.
///
/// Recognition ladder (strict — invalid values are rejected, never clamped):
/// - JSON booleans are rejected (they are not scores).
/// - Numbers (and numeric strings) are parsed.
/// - Non-finite values (NaN / ±infinity), negatives, and values above 100
///   are rejected.
/// - Values in (1.0, 10.0] are interpreted as 0-10 scales and divided by 10.
/// - Values in (10.0, 100.0] are interpreted as 0-100 scales and divided by 100.
/// - Accepted scores are rounded to 2 decimals (matching the reference
///   normalization behavior) for cache/report stability.
pub fn to_final_score(value: &serde_json::Value) -> Option<f64> {
    if value.is_boolean() {
        return None;
    }
    let as_f64 = match value {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    if !as_f64.is_finite() || as_f64 < 0.0 || as_f64 > 100.0 {
        return None;
    }
    let normalized = if as_f64 > 1.0 {
        if as_f64 <= 10.0 {
            as_f64 / 10.0
        } else {
            as_f64 / 100.0
        }
    } else {
        as_f64
    };
    Some((normalized * 100.0).round() / 100.0)
}

// ---------------------------------------------------------------------------
// Response parsing (bounded salvage)
// ---------------------------------------------------------------------------

/// A single raw scoring result object as returned by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringResultRaw {
    pub id: String,
    pub final_score: serde_json::Value,
    #[serde(default)]
    pub recommend_reason: Option<String>,
}

/// Bounded salvage parse of the model response content.
///
/// 1. Strip ```json / ``` markdown fences when present.
/// 2. Locate the outermost JSON array via bracket scanning (string-aware).
/// 3. `serde_json` parse; each element must be an object carrying an `id`.
///    Invalid elements are dropped (counted by the caller), never guessed.
///
/// This is deliberately bounded: no regex engine, no general-purpose repair,
/// and no positional assumptions — identity comes from the `id` field only.
pub fn parse_scoring_response_text(text: &str) -> Result<Vec<ScoringResultRaw>, String> {
    let stripped = strip_markdown_fences(text);
    let array_slice = find_outer_json_array(&stripped).ok_or_else(|| {
        format!(
            "[AutoClip Scoring] Response contained no JSON array (content preview: `{}`)",
            response_preview(&stripped)
        )
    })?;
    let parsed: serde_json::Value = serde_json::from_str(array_slice)
        .map_err(|e| format!("[AutoClip Scoring] JSON parse failed: {e}"))?;
    let elements = parsed
        .as_array()
        .ok_or_else(|| "[AutoClip Scoring] Parsed JSON was not an array".to_string())?;
    let mut results = Vec::new();
    for element in elements {
        let Some(obj) = element.as_object() else { continue };
        let Some(id) = obj.get("id").and_then(|v| v.as_str()).map(str::to_string) else {
            continue;
        };
        results.push(ScoringResultRaw {
            id,
            final_score: obj.get("final_score").cloned().unwrap_or(serde_json::Value::Null),
            recommend_reason: obj
                .get("recommend_reason")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        });
    }
    Ok(results)
}

/// One-line, length-capped preview of a model response, for parse-failure
/// diagnostics (refusal prose vs truncated JSON vs dict-only output).
fn response_preview(text: &str) -> String {
    let mut preview: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(200)
        .collect();
    if text.chars().count() > 200 {
        preview.push_str("...");
    }
    preview.trim().to_string()
}

fn strip_markdown_fences(text: &str) -> String {
    let trimmed = text.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed.to_string();
    };
    // Skip an optional language tag on the opening fence.
    let body_start = after_open
        .find('\n')
        .map(|i| i + 1)
        .unwrap_or(after_open.len());
    let body = &after_open[body_start..];
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim().to_string()
}

/// Find the outermost `[ ... ]` JSON array in `text` using a string-aware
/// bracket scan. Returns the inclusive slice including both brackets.
fn find_outer_json_array(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut depth: i64 = 0;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'[' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b']' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(s) = start {
                        return Some(&text[s..=i]);
                    }
                }
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Exact-identity alignment
// ---------------------------------------------------------------------------

/// The outcome of aligning one candidate's result.
#[derive(Debug, Clone, PartialEq)]
pub enum AlignedOutcome {
    /// A valid, unambiguous model score (rounded, in [0.0, 1.0]).
    Scored { score: f64, reason: Option<String> },
    /// The result existed but the score value was invalid.
    InvalidScore,
    /// No usable result referenced this candidate's id.
    Unscored,
}

/// Summary of an alignment pass for diagnostics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AlignmentSummary {
    pub aligned: usize,
    pub invalid_score: usize,
    pub unscored: usize,
    pub ambiguous_ids: usize,
    pub unknown_ids: usize,
}

/// Align raw results to scoring inputs by EXACT candidate identity.
///
/// Rules (identity alignment — never positional):
/// - Exact unique id → aligned to its candidate.
/// - Duplicate id in the response → ambiguous; NEITHER result is assigned.
/// - Unknown id → ignored (counted).
/// - Missing id → candidate stays unscored.
/// - Invalid score value → rejected safely (candidate stays unscored).
/// - Reordered or partial responses still align correctly.
pub fn align_final_scores(
    inputs: &[ScoringCandidateInput],
    results: &[ScoringResultRaw],
) -> (Vec<AlignedOutcome>, AlignmentSummary) {
    // Map result ids: duplicate id → ambiguous (None).
    let mut by_id: HashMap<&str, &ScoringResultRaw> = HashMap::new();
    let mut ambiguous: HashMap<&str, ()> = HashMap::new();
    let known: std::collections::HashSet<&str> = inputs.iter().map(|i| i.id.as_str()).collect();

    for result in results {
        if ambiguous.contains_key(result.id.as_str()) {
            continue;
        }
        if by_id.insert(result.id.as_str(), result).is_some() {
            by_id.remove(result.id.as_str());
            ambiguous.insert(result.id.as_str(), ());
        }
    }

    let mut summary = AlignmentSummary::default();
    summary.ambiguous_ids = ambiguous.len();
    summary.unknown_ids = by_id
        .keys()
        .filter(|id| !known.contains(**id))
        .count();

    let outcomes = inputs
        .iter()
        .map(|input| {
            if ambiguous.contains_key(input.id.as_str()) {
                summary.unscored += 1;
                return AlignedOutcome::Unscored;
            }
            match by_id.get(input.id.as_str()) {
                Some(result) => match to_final_score(&result.final_score) {
                    Some(score) => {
                        summary.aligned += 1;
                        AlignedOutcome::Scored {
                            score,
                            reason: result.recommend_reason.clone(),
                        }
                    }
                    None => {
                        summary.invalid_score += 1;
                        AlignedOutcome::InvalidScore
                    }
                },
                None => {
                    summary.unscored += 1;
                    AlignedOutcome::Unscored
                }
            }
        })
        .collect();

    (outcomes, summary)
}

// ---------------------------------------------------------------------------
// Final-score disk cache (versioned, atomic, success-only)
// ---------------------------------------------------------------------------

/// A cached successful model score. Failed/unscored results are NEVER cached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFinalScore {
    pub score: f64,
    #[serde(default)]
    pub recommend_reason: Option<String>,
    pub model: String,
    pub prompt_version: String,
}

/// Versioned score cache under `<cache_dir>/autoshorts/final_scores/`,
/// separate from all REZE/window-discovery caches.
pub struct FinalScoreCache {
    dir: PathBuf,
}

impl FinalScoreCache {
    pub fn new() -> Self {
        let dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("autoshorts")
            .join("final_scores");
        Self { dir }
    }

    pub fn with_dir(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    /// Cache identity over every materially different input: source/media
    /// identity, candidate timestamps, hook, context, exact transcript
    /// excerpt, included evidence, provider, model and the authoritative
    /// prompt version. The TRANSIENT scoring id is deliberately NOT part of
    /// the key (it is pool-position-based): caching must be stable across
    /// re-runs and pool reordering, not position-sensitive.
    pub fn compute_key(input: &ScoringCandidateInput, source_identity: &str, model: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(source_identity.as_bytes());
        hasher.update([0]);
        hasher.update(format!("{:.3}", input.start_seconds));
        hasher.update([0]);
        hasher.update(format!("{:.3}", input.end_seconds));
        hasher.update([0]);
        hasher.update(input.hook.as_bytes());
        hasher.update([0]);
        hasher.update(input.context.as_bytes());
        hasher.update([0]);
        hasher.update(input.transcript.as_bytes());
        hasher.update([0]);
        hasher.update(input.evidence.join("\u{1f}").as_bytes());
        hasher.update([0]);
        hasher.update(b"nvidia_nim");
        hasher.update([0]);
        hasher.update(model.as_bytes());
        hasher.update([0]);
        hasher.update(FINAL_SCORE_PROMPT_VERSION.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(format!("final_score_{key}.json"))
    }

    /// Cache hit → the stored successful model score, preserving provenance.
    pub fn get(&self, key: &str) -> Option<CachedFinalScore> {
        let raw = std::fs::read_to_string(self.path_for(key)).ok()?;
        let cached: CachedFinalScore = serde_json::from_str(&raw).ok()?;
        // Versioned entries only: a mismatched prompt/model is a miss.
        if cached.prompt_version != FINAL_SCORE_PROMPT_VERSION {
            return None;
        }
        Some(cached)
    }

    /// Atomic write (`.tmp` + rename). Only successful model scores are stored.
    pub fn put(&self, key: &str, value: &CachedFinalScore) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let final_path = self.path_for(key);
        let tmp_path = final_path.with_extension("json.tmp");
        std::fs::write(&tmp_path, serde_json::to_string_pretty(value).unwrap_or_default())?;
        std::fs::rename(&tmp_path, &final_path)
    }
}

impl Default for FinalScoreCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Provider request (bounded, retry, error classification)
// ---------------------------------------------------------------------------

/// Classified provider failure for accurate diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringFailure {
    MissingCredential,
    Auth(String),
    RateLimit,
    Transient(String),
    Timeout,
    InvalidResponse(String),
}

impl std::fmt::Display for ScoringFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScoringFailure::MissingCredential => {
                write!(f, "NVIDIA_API_KEY is not configured")
            }
            ScoringFailure::Auth(msg) => write!(f, "authentication/authorization failure: {msg}"),
            ScoringFailure::RateLimit => write!(f, "rate limit / resource exhaustion (HTTP 429)"),
            ScoringFailure::Transient(msg) => write!(f, "transient server error: {msg}"),
            ScoringFailure::Timeout => write!(f, "request timed out"),
            ScoringFailure::InvalidResponse(msg) => write!(f, "invalid response: {msg}"),
        }
    }
}

fn scorer_http_client(config: &AutoClipScorerConfig) -> Result<reqwest::Client, ScoringFailure> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.request_timeout_sec))
        .connect_timeout(std::time::Duration::from_secs(config.connect_timeout_sec))
        .pool_idle_timeout(Some(std::time::Duration::from_secs(30)))
        .build()
        .map_err(|e| ScoringFailure::InvalidResponse(format!("client build failed: {e}")))
}

/// Classify an HTTP status for retry semantics.
fn classify_status(status: reqwest::StatusCode) -> ScoringFailure {
    if status.as_u16() == 429 {
        ScoringFailure::RateLimit
    } else if status.as_u16() == 401 || status.as_u16() == 403 {
        ScoringFailure::Auth(format!("HTTP {}", status.as_u16()))
    } else if status.is_server_error() {
        ScoringFailure::Transient(format!("HTTP {}", status.as_u16()))
    } else {
        // Other 4xx: invalid configuration — do not retry indefinitely.
        ScoringFailure::InvalidResponse(format!("HTTP {}", status.as_u16()))
    }
}

/// One bounded NVIDIA NIM chat-completions request for a batch prompt.
/// Retries transient failures (429 / eligible 5xx) with exponential backoff;
/// invalid-configuration 4xx errors fail fast.
async fn request_final_scores(
    client: &reqwest::Client,
    config: &AutoClipScorerConfig,
    api_key: &str,
    prompt: &str,
) -> Result<Vec<ScoringResultRaw>, ScoringFailure> {
    let mut payload = serde_json::json!({
        "model": config.model,
        "messages": [
            { "role": "user", "content": prompt }
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    });
    // Reasoning control: historic default is OFF. Only send the toggle the
    // endpoint understands, and only when thinking is explicitly enabled.
    if config.thinking {
        payload["chat_template_kwargs"] = serde_json::json!({ "thinking": true });
    }

    let mut last_failure: Option<ScoringFailure> = None;
    for attempt in 0..MAX_SCORING_RETRIES {
        if attempt > 0 {
            let backoff = std::time::Duration::from_secs(1u64 << (attempt - 1));
            tokio::time::sleep(backoff).await;
        }
        let response = client
            .post(&config.endpoint)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Accept", "application/json")
            .json(&payload)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if !status.is_success() {
                    let failure = classify_status(status);
                    match failure {
                        ScoringFailure::RateLimit | ScoringFailure::Transient(_) => {
                            eprintln!(
                                "[AutoClip Scoring] Batch request failed ({}), retry {}/{}",
                                failure, attempt + 1, MAX_SCORING_RETRIES
                            );
                            last_failure = Some(failure);
                            continue;
                        }
                        _ => return Err(failure),
                    }
                }
                let body: serde_json::Value = resp
                    .json()
                    .await
                    .map_err(|e| ScoringFailure::InvalidResponse(format!("body decode: {e}")))?;
                let content = body
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| {
                        ScoringFailure::InvalidResponse(
                            "missing choices[0].message.content".to_string(),
                        )
                    })?
                    .to_string();
                return parse_scoring_response_text(&content)
                    .map_err(ScoringFailure::InvalidResponse);
            }
            Err(e) => {
                if e.is_timeout() {
                    eprintln!(
                        "[AutoClip Scoring] Batch request timed out, retry {}/{}",
                        attempt + 1, MAX_SCORING_RETRIES
                    );
                    last_failure = Some(ScoringFailure::Timeout);
                    continue;
                }
                if e.is_connect() {
                    last_failure = Some(ScoringFailure::InvalidResponse(format!("connect: {e}")));
                } else {
                    last_failure = Some(ScoringFailure::InvalidResponse(e.to_string()));
                }
                return Err(last_failure.unwrap());
            }
        }
    }
    Err(last_failure.unwrap_or(ScoringFailure::Transient(
        "retries exhausted".to_string(),
    )))
}

// ---------------------------------------------------------------------------
// Scoring run orchestration
// ---------------------------------------------------------------------------

/// Result summary of one AutoClip scoring run for diagnostics.
#[derive(Debug, Clone, Default)]
pub struct ScoringRunSummary {
    pub candidates_entering: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub live_batches: usize,
    pub valid_scores: usize,
    pub unscored: usize,
    pub invalid_scores: usize,
    pub ambiguous_ids: usize,
    pub unknown_ids: usize,
    pub failed_batches: usize,
}

/// Persistent provenance markers.
pub const SCORE_SOURCE_LLM: &str = "llm";
pub const SCORE_SOURCE_FALLBACK: &str = "fallback";

/// Run the AutoClip holistic scorer over the candidate pool.
///
/// - Assigns each candidate its transient identity by pool position.
/// - Serves valid cached scores first (never as "llm" provenance lies: cache
///   entries ARE genuine model scores, so they retain `score_source = "llm"`).
/// - Sends uncached candidates to NVIDIA NIM in bounded batches with bounded
///   concurrency and retry/backoff.
/// - Aligns results by exact identity; assigns `final_score` as the
///   authoritative `CandidateDraft.score`.
/// - Marks every omitted/invalid/unrecoverable candidate as
///   `score = 0.0`, `score_source = "fallback"` — visibly unscored, never
///   eligible for AutoClip Top-N selection.
pub async fn run_autoclip_scoring(
    source_identity: &str,
    drafts: &mut [CandidateDraft],
    words: &[TranscriptWord],
    api_key: &str,
) -> Result<ScoringRunSummary, ScoringFailure> {
    if api_key.trim().is_empty() {
        return Err(ScoringFailure::MissingCredential);
    }
    let config = AutoClipScorerConfig::from_env();
    let cache = FinalScoreCache::new();
    let inputs = build_scoring_inputs(drafts, words);
    let mut summary = ScoringRunSummary {
        candidates_entering: inputs.len(),
        ..Default::default()
    };
    if inputs.is_empty() {
        return Ok(summary);
    }

    println!(
        "[AutoClip Scoring] Backend=autoclip provider=nvidia_nim model={} prompt_version={} \
         candidates={} batch_size={} concurrency={}",
        config.model,
        FINAL_SCORE_PROMPT_VERSION,
        inputs.len(),
        config.batch_size,
        config.concurrency
    );

    // Persist the transient identity on every candidate for reproducible
    // tie-breaks and diagnostics.
    for (idx, input) in inputs.iter().enumerate() {
        drafts[idx].scoring_id = Some(input.id.clone());
    }

    // 1. Cache pass (keyed per candidate; hits preserve llm provenance).
    struct PendingScoring {
        input_idx: usize,
    }
    let mut pending: Vec<PendingScoring> = Vec::new();
    for (idx, input) in inputs.iter().enumerate() {
        let key = FinalScoreCache::compute_key(input, source_identity, &config.model);
        match cache.get(&key) {
            Some(cached) => {
                drafts[idx].score = cached.score;
                drafts[idx].score_source = Some(SCORE_SOURCE_LLM.to_string());
                if let Some(reason) = cached.recommend_reason.filter(|r| !r.trim().is_empty()) {
                    drafts[idx].rationale = reason;
                }
                summary.cache_hits += 1;
                // A cache hit is a genuine, previously validated model score.
                summary.valid_scores += 1;
            }
            None => {
                summary.cache_misses += 1;
                pending.push(PendingScoring { input_idx: idx });
            }
        }
    }

    // 2. Batch the uncached candidates.
    let semaphore = Arc::new(tokio::sync::Semaphore::new(config.concurrency));
    let client = scorer_http_client(&config)?;
    let mut handles = Vec::new();

    for (batch_no, chunk) in pending.chunks(config.batch_size).enumerate() {
        let batch_inputs: Vec<ScoringCandidateInput> =
            chunk.iter().map(|p| inputs[p.input_idx].clone()).collect();
        let prompt = build_final_score_prompt(&batch_inputs);
        let client = client.clone();
        let api_key = api_key.to_string();
        let config = config.clone();
        let semaphore = semaphore.clone();
        let batch_no = batch_no + 1;
        let total_batches = (pending.len() + config.batch_size - 1) / config.batch_size.max(1);

        handles.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("scoring semaphore");
            println!(
                "[AutoClip Scoring] Batch {}/{}: {} candidates (prompt {} chars)",
                batch_no,
                total_batches,
                batch_inputs.len(),
                prompt.len()
            );
            let result =
                request_final_scores(&client, &config, &api_key, &prompt).await;
            (batch_inputs, result)
        }));
    }

    // 3. Align each batch response to its exact candidate identities.
    for handle in handles {
        let (batch_inputs, result) = handle
            .await
            .map_err(|e| ScoringFailure::InvalidResponse(format!("task join: {e}")))?;
        summary.live_batches += 1;
        match result {
            Ok(raw_results) => {
                let (outcomes, alignment) = align_final_scores(&batch_inputs, &raw_results);
                println!(
                    "[AutoClip Scoring] Batch aligned: {} scored, {} unscored, {} invalid, \
                     {} ambiguous ids, {} unknown ids",
                    alignment.aligned,
                    alignment.unscored,
                    alignment.invalid_score,
                    alignment.ambiguous_ids,
                    alignment.unknown_ids
                );
                summary.valid_scores += alignment.aligned;
                summary.unscored += alignment.unscored;
                summary.invalid_scores += alignment.invalid_score;
                summary.ambiguous_ids += alignment.ambiguous_ids;
                summary.unknown_ids += alignment.unknown_ids;

                for (batch_idx, outcome) in outcomes.iter().enumerate() {
                    let input = &batch_inputs[batch_idx];
                    let pool_idx = input
                        .id
                        .strip_prefix("cand-")
                        .and_then(|n| n.parse::<usize>().ok());
                    if let Some(pool_idx) = pool_idx {
                        if let AlignedOutcome::Scored { score, reason } = outcome {
                            drafts[pool_idx].score = *score;
                            drafts[pool_idx].score_source = Some(SCORE_SOURCE_LLM.to_string());
                            if let Some(reason) = reason
                                .as_ref()
                                .filter(|r| !r.trim().is_empty())
                            {
                                drafts[pool_idx].rationale = reason.clone();
                            }
                            let key = FinalScoreCache::compute_key(
                                input,
                                source_identity,
                                &config.model,
                            );
                            let cached = CachedFinalScore {
                                score: *score,
                                recommend_reason: reason.clone(),
                                model: config.model.clone(),
                                prompt_version: FINAL_SCORE_PROMPT_VERSION.to_string(),
                            };
                            if let Err(e) = cache.put(&key, &cached) {
                                eprintln!("[AutoClip Scoring] Cache write failed: {e}");
                            }
                        }
                    }
                }
            }
            Err(failure) => {
                summary.failed_batches += 1;
                eprintln!(
                    "[AutoClip Scoring] Batch failed and will not be retried further: {failure}"
                );
            }
        }
    }

    // 4. Every candidate that did not receive a valid model score becomes an
    //    explicit unscored fallback: score 0.0 + fallback provenance. A failed
    //    scoring operation never destroys the candidate record.
    for draft in drafts.iter_mut() {
        if draft.score_source.as_deref() != Some(SCORE_SOURCE_LLM) {
            draft.score = 0.0;
            draft.score_source = Some(SCORE_SOURCE_FALLBACK.to_string());
        }
    }

    println!(
        "[AutoClip Scoring] Run complete: entered={} cache_hits={} live_batches={} valid={} \
         unscored={} invalid={} failed_batches={} — final_score is AUTHORITATIVE in autoclip mode",
        summary.candidates_entering,
        summary.cache_hits,
        summary.live_batches,
        summary.valid_scores,
        summary.unscored,
        summary.invalid_scores,
        summary.failed_batches
    );
    if summary.valid_scores == 0 {
        eprintln!(
            "[AutoClip Scoring] WARNING: no usable scoring result was obtained; all {} \
             candidates are explicit unscored fallbacks",
            summary.candidates_entering
        );
    }
    Ok(summary)
}

// ---------------------------------------------------------------------------
// AutoClip ranking + eligibility
// ---------------------------------------------------------------------------

/// A candidate is eligible for AutoClip Top-N selection only when it carries
/// a genuine model score (`score_source == "llm"`) AND meets the threshold.
/// Unscored/fallback candidates are NEVER eligible, even though their
/// numeric fallback score is 0.0.
pub fn is_autoclip_eligible(draft: &CandidateDraft, threshold: f64) -> bool {
    draft.score_source.as_deref() == Some(SCORE_SOURCE_LLM) && draft.score >= threshold
}

/// Deterministic AutoClip ranking comparator (total order):
/// 1. Valid model-scored candidates before unscored/fallback candidates.
/// 2. `final_score` descending.
/// 3. Start timestamp ascending.
/// 4. End timestamp ascending.
/// 5. Transient scoring identity ascending as the final tie-break.
///
/// Uses `total_cmp` for all floating-point comparisons — never
/// subtraction-based ordering.
pub fn autoclip_rank_cmp(a: &CandidateDraft, b: &CandidateDraft) -> std::cmp::Ordering {
    let a_scored = a.score_source.as_deref() == Some(SCORE_SOURCE_LLM);
    let b_scored = b.score_source.as_deref() == Some(SCORE_SOURCE_LLM);
    // Scored candidates sort BEFORE unscored/fallback candidates.
    b_scored
        .cmp(&a_scored)
        .then(b.score.total_cmp(&a.score))
        .then(a.start.total_cmp(&b.start))
        .then(a.end.total_cmp(&b.end))
        .then_with(|| match (&a.scoring_id, &b.scoring_id) {
            (Some(x), Some(y)) => x.cmp(y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        })
}

/// Effective Top-N: `selected_count = min(N, eligible_count)`; disabled
/// cutoff selects all eligible candidates. Never fabricates candidates and
/// never selects ineligible ones.
pub fn effective_selected_count(top_n: Option<usize>, eligible_count: usize) -> usize {
    match top_n {
        Some(n) => n.min(eligible_count),
        None => eligible_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(start: f64, end: f64, score: f64) -> CandidateDraft {
        CandidateDraft {
            start,
            end,
            score,
            hook: format!("hook {start}"),
            rationale: format!("rationale {start}"),
            ..Default::default()
        }
    }

    // ---------------- Scoring configuration ----------------

    /// Scorer configuration is resolved from process-env variables, which are
    /// global state shared by every test in this binary (tests run in
    /// parallel threads). Verified defect: in a full-suite run,
    /// `scorer_config_defaults_match_documented_contract` raced with
    /// `scorer_config_endpoint_and_model_overrides` and failed despite passing
    /// in isolation. Every test that reads or writes a scorer env var must
    /// hold this lock for its whole body.
    static SCORER_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock_scorer_env() -> std::sync::MutexGuard<'static, ()> {
        SCORER_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn scoring_backend_defaults_to_legacy() {
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
        assert_eq!(resolve_scoring_backend().unwrap(), ScoringBackend::Legacy);
    }

    #[test]
    fn scoring_backend_explicit_autoclip() {
        let _env = lock_scorer_env();
        std::env::set_var("AUTOSHORTS_SCORING_BACKEND", "autoclip");
        assert_eq!(resolve_scoring_backend().unwrap(), ScoringBackend::AutoClip);
        std::env::set_var("AUTOSHORTS_SCORING_BACKEND", "AUTOCLIP");
        assert_eq!(resolve_scoring_backend().unwrap(), ScoringBackend::AutoClip);
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
    }

    #[test]
    fn scoring_backend_invalid_is_hard_error_not_silent_substitution() {
        let _env = lock_scorer_env();
        std::env::set_var("AUTOSHORTS_SCORING_BACKEND", "openai");
        assert!(resolve_scoring_backend().is_err());
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
    }

    // ---------------- UI scoring-mode selector (per-call override) ----------------

    #[test]
    fn scoring_mode_override_none_and_empty_fall_back_to_env_default() {
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
        assert_eq!(
            resolve_scoring_backend_with_override(None).unwrap(),
            ScoringBackend::Legacy
        );
        assert_eq!(
            resolve_scoring_backend_with_override(Some("")).unwrap(),
            ScoringBackend::Legacy
        );
        std::env::set_var("AUTOSHORTS_SCORING_BACKEND", "autoclip");
        assert_eq!(
            resolve_scoring_backend_with_override(None).unwrap(),
            ScoringBackend::AutoClip
        );
        assert_eq!(
            resolve_scoring_backend_with_override(Some("   ")).unwrap(),
            ScoringBackend::AutoClip
        );
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
    }

    #[test]
    fn scoring_mode_override_legacy_forces_legacy_even_when_env_is_autoclip() {
        let _env = lock_scorer_env();
        std::env::set_var("AUTOSHORTS_SCORING_BACKEND", "autoclip");
        // The explicit Legacy selection reaches the backend and overrides the
        // configured default for this call.
        assert_eq!(
            resolve_scoring_backend_with_override(Some("legacy")).unwrap(),
            ScoringBackend::Legacy
        );
        // Case-insensitive + whitespace-tolerant, mirroring the env resolver.
        assert_eq!(
            resolve_scoring_backend_with_override(Some(" Legacy ")).unwrap(),
            ScoringBackend::Legacy
        );
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
    }

    #[test]
    fn scoring_mode_override_autoclip_forces_autoclip_even_when_env_default_is_legacy() {
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
        // Production default is legacy; the user's AutoClip selection must be
        // authoritative for this call.
        assert_eq!(
            resolve_scoring_backend_with_override(Some("autoclip")).unwrap(),
            ScoringBackend::AutoClip
        );
        assert_eq!(
            resolve_scoring_backend_with_override(Some("AutoClip")).unwrap(),
            ScoringBackend::AutoClip
        );
    }

    #[test]
    fn scoring_mode_override_unknown_selection_is_a_hard_error() {
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
        let err = resolve_scoring_backend_with_override(Some("nvidia")).unwrap_err();
        assert!(err.contains("Invalid scoring-mode selection"), "{err}");
        assert!(err.contains("'nvidia'"), "{err}");
    }

    #[test]
    fn scorer_credential_gate_missing_key_is_hard_error_not_silent_legacy_fallback() {
        let _env = lock_scorer_env();
        std::env::remove_var("NVIDIA_API_KEY");
        // Selecting AutoClip without a credential must produce a clear error,
        // never a silent switch to the legacy backend.
        let err = resolve_scorer_credential(ScoringBackend::AutoClip).unwrap_err();
        assert!(err.contains("NVIDIA_API_KEY"), "{err}");
        let err = resolve_scoring_authority_for_call(Some("autoclip")).unwrap_err();
        assert!(err.contains("NVIDIA_API_KEY"), "{err}");
        assert!(err.contains("Legacy Scoring"), "{err}");
    }

    #[test]
    fn scorer_credential_gate_legacy_needs_no_credential() {
        let _env = lock_scorer_env();
        std::env::remove_var("NVIDIA_API_KEY");
        assert_eq!(resolve_scorer_credential(ScoringBackend::Legacy).unwrap(), None);
        let (backend, credential) =
            resolve_scoring_authority_for_call(Some("legacy")).unwrap();
        assert_eq!(backend, ScoringBackend::Legacy);
        assert_eq!(credential, None);
    }

    #[test]
    fn scorer_credential_gate_autoclip_with_key_returns_verified_credential() {
        let _env = lock_scorer_env();
        std::env::set_var("NVIDIA_API_KEY", "nvapi-test-selector-key");
        let (backend, credential) =
            resolve_scoring_authority_for_call(Some("autoclip")).unwrap();
        assert_eq!(backend, ScoringBackend::AutoClip);
        assert_eq!(credential.as_deref(), Some("nvapi-test-selector-key"));
        std::env::remove_var("NVIDIA_API_KEY");
    }

    #[test]
    fn scorer_config_defaults_match_documented_contract() {
        let _env = lock_scorer_env();
        std::env::remove_var("NVIDIA_SCORER_BASE_URL");
        std::env::remove_var("AUTOSHORTS_SCORER_MODEL");
        std::env::remove_var("AUTOSHORTS_SCORER_THINKING");
        std::env::remove_var("AUTOSHORTS_SCORER_MAX_TOKENS");
        let config = AutoClipScorerConfig::from_env();
        assert_eq!(
            config.endpoint,
            "https://integrate.api.nvidia.com/v1/chat/completions"
        );
        assert_eq!(config.model, "nvidia/nemotron-3-ultra-550b-a55b");
        assert!(!config.thinking);
        assert!((config.temperature - 0.15).abs() < 1e-9);
        assert_eq!(config.max_tokens, 4096);
        assert_eq!(config.batch_size, 6);
        assert_eq!(config.concurrency, 2);
        assert_eq!(config.connect_timeout_sec, 30);
        assert_eq!(config.request_timeout_sec, 180);
    }

    #[test]
    fn scorer_config_endpoint_and_model_overrides() {
        let _env = lock_scorer_env();
        std::env::set_var("NVIDIA_SCORER_BASE_URL", "https://example.test/v1/");
        std::env::set_var("AUTOSHORTS_SCORER_MODEL", "custom/model-x");
        let config = AutoClipScorerConfig::from_env();
        assert_eq!(config.endpoint, "https://example.test/v1/chat/completions");
        assert_eq!(config.model, "custom/model-x");
        std::env::remove_var("NVIDIA_SCORER_BASE_URL");
        std::env::remove_var("AUTOSHORTS_SCORER_MODEL");

        // A full endpoint override is respected as-is.
        std::env::set_var(
            "NVIDIA_SCORER_BASE_URL",
            "https://example.test/v1/chat/completions",
        );
        let config = AutoClipScorerConfig::from_env();
        assert_eq!(
            config.endpoint,
            "https://example.test/v1/chat/completions"
        );
        std::env::remove_var("NVIDIA_SCORER_BASE_URL");
    }

    #[test]
    fn thinking_flag_and_max_tokens_overrides() {
        let _env = lock_scorer_env();
        std::env::set_var("AUTOSHORTS_SCORER_THINKING", "1");
        std::env::set_var("AUTOSHORTS_SCORER_MAX_TOKENS", "2048");
        let config = AutoClipScorerConfig::from_env();
        assert!(config.thinking);
        assert_eq!(config.max_tokens, 2048);
        std::env::remove_var("AUTOSHORTS_SCORER_THINKING");
        std::env::remove_var("AUTOSHORTS_SCORER_MAX_TOKENS");
    }

    #[test]
    fn legacy_mode_never_requires_nvidia_credential() {
        // The backend switch itself is independent of credentials; the
        // credential requirement applies only to the explicit autoclip path.
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOSHORTS_SCORING_BACKEND");
        assert_eq!(resolve_scoring_backend().unwrap(), ScoringBackend::Legacy);
    }

    // ---------------- Threshold / Top-N resolution ----------------

    #[test]
    fn threshold_defaults_to_documented_0_65_and_supports_override() {
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOCLIP_RECOMMEND_THRESHOLD");
        assert!((resolve_recommend_threshold().unwrap() - 0.65).abs() < 1e-9);
        std::env::set_var("AUTOCLIP_RECOMMEND_THRESHOLD", "0.5");
        assert!((resolve_recommend_threshold().unwrap() - 0.5).abs() < 1e-9);
        std::env::set_var("AUTOCLIP_RECOMMEND_THRESHOLD", "2");
        assert!(resolve_recommend_threshold().is_err());
        std::env::remove_var("AUTOCLIP_RECOMMEND_THRESHOLD");
    }

    #[test]
    fn top_n_defaults_to_12_and_non_positive_disables() {
        let _env = lock_scorer_env();
        std::env::remove_var("AUTOSHORTS_TOP_N");
        assert_eq!(resolve_top_n().unwrap(), Some(12));
        std::env::set_var("AUTOSHORTS_TOP_N", "3");
        assert_eq!(resolve_top_n().unwrap(), Some(3));
        std::env::set_var("AUTOSHORTS_TOP_N", "0");
        assert_eq!(resolve_top_n().unwrap(), None);
        std::env::set_var("AUTOSHORTS_TOP_N", "-5");
        assert_eq!(resolve_top_n().unwrap(), None);
        std::env::set_var("AUTOSHORTS_TOP_N", "abc");
        assert!(resolve_top_n().is_err());
        std::env::remove_var("AUTOSHORTS_TOP_N");
    }

    // ---------------- Identity ----------------

    #[test]
    fn transient_identity_is_position_based_and_unique() {
        assert_eq!(transient_scoring_id(0), "cand-0000");
        assert_eq!(transient_scoring_id(7), "cand-0007");
        assert_eq!(transient_scoring_id(1234), "cand-1234");
        assert_ne!(transient_scoring_id(1), transient_scoring_id(2));
    }

    // ---------------- Input construction ----------------

    fn word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: None,
        }
    }

    #[test]
    fn transcript_excerpt_selects_only_candidate_range_and_caps_1200() {
        let words: Vec<TranscriptWord> = (0..300)
            .map(|i| word(&format!("w{i}"), i as f64, i as f64 + 0.5))
            .collect();
        let excerpt = transcript_excerpt(&words, 10.0, 20.0, MAX_TRANSCRIPT_EXCERPT_CHARS);
        assert!(excerpt.contains("w10"));
        assert!(excerpt.contains("w19"));
        assert!(!excerpt.contains("w5 "));
        assert!(excerpt.chars().count() <= MAX_TRANSCRIPT_EXCERPT_CHARS);
    }

    #[test]
    fn evidence_excludes_fallback_marker_and_caps_items_and_chars() {
        let mut d = draft(0.0, 10.0, 0.5);
        d.multimodal_evidence = Some(vec![
            MULTIMODAL_FALLBACK_MARKER.to_string(),
            "e1".to_string(),
            format!("{}{}", "x".repeat(300), "tail"),
            "e3".to_string(),
            "e4".to_string(),
        ]);
        let evidence = collect_evidence(&d);
        assert_eq!(evidence.len(), 3);
        assert!(!evidence.iter().any(|e| e.contains("unavailable")));
        assert!(evidence.iter().all(|e| e.chars().count() <= 160));
        assert_eq!(evidence[0], "e1");
    }

    #[test]
    fn build_scoring_inputs_assigns_identity_by_pool_position() {
        let drafts = vec![draft(0.0, 10.0, 0.5), draft(10.0, 20.0, 0.6)];
        let inputs = build_scoring_inputs(&drafts, &[]);
        assert_eq!(inputs[0].id, "cand-0000");
        assert_eq!(inputs[1].id, "cand-0001");
        assert_eq!(inputs[0].hook, "hook 0");
        assert_eq!(inputs[1].start_seconds, 10.0);
        assert_eq!(inputs[1].end_seconds, 20.0);
    }

    // ---------------- Prompt ----------------

    #[test]
    fn prompt_contains_ids_timestamps_contract_and_no_fabrication() {
        let inputs = vec![ScoringCandidateInput {
            id: "cand-0000".to_string(),
            hook: "the hook".to_string(),
            context: "why it matters".to_string(),
            start_seconds: 12.5,
            end_seconds: 45.25,
            transcript: "spoken words".to_string(),
            evidence: vec!["face visible".to_string()],
        }];
        let prompt = build_final_score_prompt(&inputs);
        assert!(prompt.contains("cand-0000"));
        assert!(prompt.contains("12.5"));
        assert!(prompt.contains("45.25"));
        assert!(prompt.contains("the hook"));
        assert!(prompt.contains("spoken words"));
        assert!(prompt.contains("face visible"));
        assert!(prompt.contains("Information value"));
        assert!(prompt.contains("Emotional resonance"));
        assert!(prompt.contains("Shareability"));
        assert!(prompt.contains("Structural completeness"));
        assert!(prompt.contains("final_score"));
        assert!(prompt.contains("recommend_reason"));
        assert!(prompt.contains("Never fabricate"));
    }

    #[test]
    fn prompt_omits_evidence_field_entirely_when_unavailable() {
        let inputs = vec![ScoringCandidateInput {
            id: "cand-0001".to_string(),
            hook: "h".to_string(),
            context: "c".to_string(),
            start_seconds: 0.0,
            end_seconds: 1.0,
            transcript: "t".to_string(),
            evidence: vec![],
        }];
        let prompt = build_final_score_prompt(&inputs);
        assert!(!prompt.contains("evidence\":"));
    }

    // ---------------- Normalization ladder ----------------

    #[test]
    fn valid_scores_in_unit_range_are_kept() {
        for v in [0.0, 0.25, 0.65, 1.0] {
            let got = to_final_score(&serde_json::json!(v)).unwrap();
            assert!((got - v).abs() < 1e-9, "{v} -> {got}");
        }
    }

    #[test]
    fn scores_from_0_10_and_0_100_scales_convert() {
        assert!((to_final_score(&serde_json::json!(8.0)).unwrap() - 0.8).abs() < 1e-9);
        assert!((to_final_score(&serde_json::json!(65.0)).unwrap() - 0.65).abs() < 1e-9);
    }

    #[test]
    fn numeric_strings_parse() {
        assert!((to_final_score(&serde_json::json!("0.72")).unwrap() - 0.72).abs() < 1e-9);
        assert!((to_final_score(&serde_json::json!("70")).unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn invalid_scores_are_rejected_not_clamped() {
        assert!(to_final_score(&serde_json::json!(true)).is_none());
        assert!(to_final_score(&serde_json::json!(false)).is_none());
        assert!(to_final_score(&serde_json::json!(-0.1)).is_none());
        assert!(to_final_score(&serde_json::json!(101.0)).is_none());
        assert!(to_final_score(&serde_json::json!(f64::NAN)).is_none());
        assert!(to_final_score(&serde_json::json!(f64::INFINITY)).is_none());
        assert!(to_final_score(&serde_json::json!(f64::NEG_INFINITY)).is_none());
        assert!(to_final_score(&serde_json::json!("not a number")).is_none());
        assert!(to_final_score(&serde_json::Value::Null).is_none());
        assert!(to_final_score(&serde_json::json!({"score": 0.5})).is_none());
        assert!(to_final_score(&serde_json::json!([])).is_none());
    }

    // ---------------- Response parsing ----------------

    #[test]
    fn parse_accepts_plain_and_fenced_json() {
        let plain = r#"[{"id":"cand-0000","final_score":0.8,"recommend_reason":"great"}]"#;
        let fenced = format!("```json\n{plain}\n```");
        for text in [plain, &fenced] {
            let parsed = parse_scoring_response_text(text).unwrap();
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].id, "cand-0000");
        }
    }

    #[test]
    fn parse_rejects_malformed_and_non_array() {
        assert!(parse_scoring_response_text("no json here").is_err());
        assert!(parse_scoring_response_text("{\"id\":\"cand-0000\"}").is_err());
        assert!(parse_scoring_response_text("[{broken").is_err());
    }

    #[test]
    fn parse_survives_leading_prose_and_nested_brackets() {
        let text = r#"Here are the scores:
        [{"id":"cand-0000","final_score":0.9,"recommend_reason":"ok [bracketed]"}]
        Done."#;
        let parsed = parse_scoring_response_text(text).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].id, "cand-0000");
    }

    // ---------------- Exact-identity alignment ----------------

    fn result(id: &str, score: serde_json::Value) -> ScoringResultRaw {
        ScoringResultRaw {
            id: id.to_string(),
            final_score: score,
            recommend_reason: Some(format!("reason for {id}")),
        }
    }

    fn two_inputs() -> Vec<ScoringCandidateInput> {
        vec![
            ScoringCandidateInput {
                id: "cand-0000".to_string(),
                hook: "a".into(),
                context: "a".into(),
                start_seconds: 0.0,
                end_seconds: 1.0,
                transcript: String::new(),
                evidence: vec![],
            },
            ScoringCandidateInput {
                id: "cand-0001".to_string(),
                hook: "b".into(),
                context: "b".into(),
                start_seconds: 1.0,
                end_seconds: 2.0,
                transcript: String::new(),
                evidence: vec![],
            },
        ]
    }

    #[test]
    fn alignment_is_by_id_not_position() {
        let inputs = two_inputs();
        // Response order deliberately reversed relative to input order.
        let results = vec![result("cand-0001", serde_json::json!(0.4)), result("cand-0000", serde_json::json!(0.9))];
        let (outcomes, summary) = align_final_scores(&inputs, &results);
        assert_eq!(summary.aligned, 2);
        assert!(matches!(
            outcomes[0],
            AlignedOutcome::Scored { score, .. } if (score - 0.9).abs() < 1e-9
        ));
        assert!(matches!(
            outcomes[1],
            AlignedOutcome::Scored { score, .. } if (score - 0.4).abs() < 1e-9
        ));
    }

    #[test]
    fn duplicate_id_is_ambiguous_and_never_assigned() {
        let inputs = two_inputs();
        let results = vec![
            result("cand-0000", serde_json::json!(0.9)),
            result("cand-0000", serde_json::json!(0.1)),
        ];
        let (outcomes, summary) = align_final_scores(&inputs, &results);
        assert_eq!(summary.ambiguous_ids, 1);
        assert_eq!(summary.unscored, 2);
        assert!(outcomes.iter().all(|o| *o == AlignedOutcome::Unscored));
    }

    #[test]
    fn unknown_id_is_ignored_and_logged_via_summary() {
        let inputs = two_inputs();
        let results = vec![result("cand-9999", serde_json::json!(0.9))];
        let (outcomes, summary) = align_final_scores(&inputs, &results);
        assert_eq!(summary.unknown_ids, 1);
        assert_eq!(summary.unscored, 2);
        assert!(outcomes.iter().all(|o| *o == AlignedOutcome::Unscored));
    }

    #[test]
    fn missing_ids_stay_unscored_on_partial_response() {
        let inputs = two_inputs();
        let results = vec![result("cand-0000", serde_json::json!(0.9))];
        let (outcomes, summary) = align_final_scores(&inputs, &results);
        assert_eq!(summary.aligned, 1);
        assert_eq!(summary.unscored, 1);
        assert!(matches!(outcomes[0], AlignedOutcome::Scored { .. }));
        assert_eq!(outcomes[1], AlignedOutcome::Unscored);
    }

    #[test]
    fn invalid_score_value_is_rejected_safely() {
        let inputs = two_inputs();
        let results = vec![
            result("cand-0000", serde_json::json!(true)),
            result("cand-0001", serde_json::json!(f64::NAN)),
        ];
        let (outcomes, summary) = align_final_scores(&inputs, &results);
        assert_eq!(summary.invalid_score, 2);
        assert!(outcomes
            .iter()
            .all(|o| *o == AlignedOutcome::InvalidScore));
    }

    #[test]
    fn empty_results_mark_everything_unscored() {
        let inputs = two_inputs();
        let (outcomes, summary) = align_final_scores(&inputs, &[]);
        assert_eq!(summary.unscored, 2);
        assert!(outcomes.iter().all(|o| *o == AlignedOutcome::Unscored));
    }

    // ---------------- Cache ----------------

    fn cache_input(id: &str, hook: &str, transcript: &str, evidence: Vec<String>) -> ScoringCandidateInput {
        ScoringCandidateInput {
            id: id.to_string(),
            hook: hook.to_string(),
            context: "ctx".to_string(),
            start_seconds: 1.0,
            end_seconds: 2.0,
            transcript: transcript.to_string(),
            evidence,
        }
    }

    fn temp_cache() -> (FinalScoreCache, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "autoshorts_scoring_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        (FinalScoreCache::with_dir(dir.clone()), dir)
    }

    fn base_input() -> ScoringCandidateInput {
        cache_input("cand-0000", "hook", "transcript words", vec!["ev1".to_string()])
    }

    #[test]
    fn cache_key_changes_with_every_material_input() {
        let base = base_input();
        let key = FinalScoreCache::compute_key(&base, "src", "model-a");

        let mut other = base.clone();
        other.start_seconds = 1.5;
        assert_ne!(key, FinalScoreCache::compute_key(&other, "src", "model-a"));

        let mut other = base.clone();
        other.end_seconds = 2.5;
        assert_ne!(key, FinalScoreCache::compute_key(&other, "src", "model-a"));

        let mut other = base.clone();
        other.hook = "different".to_string();
        assert_ne!(key, FinalScoreCache::compute_key(&other, "src", "model-a"));

        let mut other = base.clone();
        other.context = "different".to_string();
        assert_ne!(key, FinalScoreCache::compute_key(&other, "src", "model-a"));

        let mut other = base.clone();
        other.transcript = "changed excerpt".to_string();
        assert_ne!(key, FinalScoreCache::compute_key(&other, "src", "model-a"));

        let mut other = base.clone();
        other.evidence = vec!["changed".to_string()];
        assert_ne!(key, FinalScoreCache::compute_key(&other, "src", "model-a"));

        assert_ne!(key, FinalScoreCache::compute_key(&base, "other-src", "model-a"));
        assert_ne!(key, FinalScoreCache::compute_key(&base, "src", "model-b"));
    }

    #[test]
    fn cache_key_is_position_independent() {
        // The transient scoring id is pool-position-based and deliberately
        // excluded from the cache key: the same candidate content must hit
        // the same cache entry regardless of where it sits in the pool.
        let a = cache_input("cand-0000", "hook", "t", vec![]);
        let b = cache_input("cand-0001", "hook", "t", vec![]);
        assert_eq!(
            FinalScoreCache::compute_key(&a, "src", "m"),
            FinalScoreCache::compute_key(&b, "src", "m")
        );
    }

    #[test]
    fn cache_roundtrip_hit_and_version_mismatch_miss() {
        let (cache, _dir) = temp_cache();
        let key = FinalScoreCache::compute_key(&base_input(), "src", "m");
        let value = CachedFinalScore {
            score: 0.83,
            recommend_reason: Some("solid insight".to_string()),
            model: "m".to_string(),
            prompt_version: FINAL_SCORE_PROMPT_VERSION.to_string(),
        };
        cache.put(&key, &value).unwrap();
        let hit = cache.get(&key).unwrap();
        assert!((hit.score - 0.83).abs() < 1e-9);
        assert_eq!(hit.recommend_reason.as_deref(), Some("solid insight"));

        // A mismatched prompt version is a miss, not stale reuse.
        let stale = CachedFinalScore {
            score: 0.99,
            recommend_reason: None,
            model: "m".to_string(),
            prompt_version: "v1.0".to_string(),
        };
        cache.put(&key, &stale).unwrap();
        assert!(cache.get(&key).is_none());

        // Unknown key misses.
        let other_key = FinalScoreCache::compute_key(&base_input(), "different-src", "m");
        assert!(cache.get(&other_key).is_none());
        let _ = std::fs::remove_dir_all(cache.dir());
    }

    #[test]
    fn cache_isolated_per_candidate_no_cross_contamination() {
        let (cache, _dir) = temp_cache();
        let input_a = cache_input("cand-0000", "hookA", "ta", vec![]);
        let input_b = cache_input("cand-0001", "hookB", "tb", vec![]);
        let key_a = FinalScoreCache::compute_key(&input_a, "src", "m");
        let key_b = FinalScoreCache::compute_key(&input_b, "src", "m");
        assert_ne!(key_a, key_b);
        cache
            .put(
                &key_a,
                &CachedFinalScore {
                    score: 0.1,
                    recommend_reason: None,
                    model: "m".into(),
                    prompt_version: FINAL_SCORE_PROMPT_VERSION.into(),
                },
            )
            .unwrap();
        // B must not see A's score.
        assert!(cache.get(&key_b).is_none());
        let _ = std::fs::remove_dir_all(cache.dir());
    }

    // ---------------- Eligibility / ranking ----------------

    #[test]
    fn eligibility_requires_llm_provenance_and_threshold_inclusive() {
        let threshold = 0.65;
        let mut d = draft(0.0, 10.0, 0.65);
        d.score_source = Some(SCORE_SOURCE_LLM.to_string());
        // Score exactly equal to the threshold passes (>=).
        assert!(is_autoclip_eligible(&d, threshold));

        d.score = 0.6499;
        assert!(!is_autoclip_eligible(&d, threshold));

        // Fallback candidates are never eligible, whatever the numeric score.
        let mut fb = draft(0.0, 10.0, 0.0);
        fb.score_source = Some(SCORE_SOURCE_FALLBACK.to_string());
        assert!(!is_autoclip_eligible(&fb, threshold));
        // Even a "high" fallback score stays ineligible (provenance rules).
        fb.score = 0.99;
        assert!(!is_autoclip_eligible(&fb, threshold));

        // No provenance (legacy record) is not AutoClip-eligible.
        let legacy = draft(0.0, 10.0, 0.99);
        assert!(!is_autoclip_eligible(&legacy, threshold));
    }

    #[test]
    fn genuine_model_score_of_zero_keeps_llm_provenance() {
        // A real model score of 0.0 must retain llm provenance — it is not
        // "unscored" merely because its value is zero.
        let mut d = draft(0.0, 10.0, 0.0);
        d.score_source = Some(SCORE_SOURCE_LLM.to_string());
        assert_eq!(d.score_source.as_deref(), Some("llm"));
        assert!(!is_autoclip_eligible(&d, 0.65));
    }

    fn scored_draft(start: f64, end: f64, score: f64, id: Option<&str>) -> CandidateDraft {
        let mut d = draft(start, end, score);
        d.score_source = Some(SCORE_SOURCE_LLM.to_string());
        d.scoring_id = id.map(str::to_string);
        d
    }

    fn fallback_draft(start: f64, end: f64) -> CandidateDraft {
        let mut d = draft(start, end, 0.0);
        d.score_source = Some(SCORE_SOURCE_FALLBACK.to_string());
        d.scoring_id = Some("cand-9999".to_string());
        d
    }

    #[test]
    fn comparator_orders_scored_before_fallback_then_score_desc() {
        let mut pool = vec![
            fallback_draft(0.0, 1.0),
            scored_draft(5.0, 6.0, 0.4, Some("cand-0003")),
            scored_draft(1.0, 2.0, 0.9, Some("cand-0001")),
            scored_draft(3.0, 4.0, 0.95, Some("cand-0002")),
        ];
        pool.sort_by(|a, b| autoclip_rank_cmp(a, b));
        assert_eq!(pool[0].scoring_id.as_deref(), Some("cand-0002"));
        assert_eq!(pool[1].scoring_id.as_deref(), Some("cand-0001"));
        assert_eq!(pool[2].scoring_id.as_deref(), Some("cand-0003"));
        assert_eq!(pool[3].score_source.as_deref(), Some("fallback"));
    }

    #[test]
    fn comparator_tie_breaks_start_end_then_id_deterministically() {
        let mut pool = vec![
            scored_draft(1.0, 2.0, 0.9, Some("cand-0001")),
            scored_draft(0.0, 2.0, 0.9, Some("cand-0002")),
            scored_draft(0.0, 1.0, 0.9, Some("cand-0003")),
            scored_draft(7.0, 8.0, 0.9, Some("cand-0000")),
        ];
        pool.sort_by(|a, b| autoclip_rank_cmp(a, b));
        let ids: Vec<&str> = pool.iter().map(|d| d.scoring_id.as_deref().unwrap()).collect();
        // Equal scores: start asc (7.0 last), then end asc, then id asc.
        assert_eq!(ids, ["cand-0003", "cand-0002", "cand-0001", "cand-0000"]);

        // Determinism: repeated sorts of shuffled copies give identical order.
        for seed in 0..8 {
            let mut shuffled = pool.clone();
            // Rotate by seed to vary input order.
            let n = shuffled.len();
            shuffled.rotate_left(seed % n);
            shuffled.sort_by(|a, b| autoclip_rank_cmp(a, b));
            let got: Vec<&str> = shuffled.iter().map(|d| d.scoring_id.as_deref().unwrap()).collect();
            assert_eq!(got, ids, "seed {seed}");
        }
    }

    #[test]
    fn comparator_is_stable_across_arbitrary_input_orders() {
        let base = vec![
            scored_draft(0.0, 1.0, 0.9, Some("cand-0000")),
            scored_draft(0.0, 1.0, 0.8, Some("cand-0001")),
            scored_draft(0.5, 1.5, 0.95, Some("cand-0002")),
            scored_draft(0.5, 1.5, 0.7, Some("cand-0003")),
            fallback_draft(0.0, 1.0),
            scored_draft(2.0, 3.0, 0.9, Some("cand-0005")),
        ];
        let mut reference = base.clone();
        reference.sort_by(|a, b| autoclip_rank_cmp(a, b));
        for seed in 0..12 {
            let mut shuffled = base.clone();
            let n = shuffled.len();
            shuffled.rotate_left(seed % n);
            shuffled.sort_by(|a, b| autoclip_rank_cmp(a, b));
            let ref_ids: Vec<Option<String>> =
                reference.iter().map(|d| d.scoring_id.clone()).collect();
            let got_ids: Vec<Option<String>> =
                shuffled.iter().map(|d| d.scoring_id.clone()).collect();
            assert_eq!(ref_ids, got_ids, "seed {seed}");
        }
    }

    // ---------------- Top-N ----------------

    #[test]
    fn top_n_never_exceeds_eligible_count() {
        assert_eq!(effective_selected_count(Some(12), 5), 5);
        assert_eq!(effective_selected_count(Some(3), 5), 3);
        assert_eq!(effective_selected_count(Some(5), 5), 5);
        assert_eq!(effective_selected_count(Some(12), 0), 0);
        assert_eq!(effective_selected_count(None, 7), 7);
        assert_eq!(effective_selected_count(None, 0), 0);
    }

    // ---------------- Run-level fallback semantics ----------------

    #[tokio::test]
    async fn missing_credential_is_a_hard_error_before_any_scoring() {
        let _env = lock_scorer_env();
        let mut drafts = vec![draft(0.0, 1.0, 0.5)];
        let err = run_autoclip_scoring("src", &mut drafts, &[], "  ").await;
        assert_eq!(err.unwrap_err(), ScoringFailure::MissingCredential);
    }

    #[tokio::test]
    async fn empty_pool_is_a_clean_noop() {
        let _env = lock_scorer_env();
        let summary = run_autoclip_scoring("src", &mut [], &[], "key").await.unwrap();
        assert_eq!(summary.candidates_entering, 0);
    }
}
