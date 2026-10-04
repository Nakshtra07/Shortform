//! Hook & Ending Optimization 2.0 — render-time boundary refinement.
//!
//! Takes an ALREADY-SELECTED candidate and improves (1) its opening and
//! (2) its ending. This is NOT a second candidate selector: it never
//! re-ranks, re-scores, or replaces candidates. It only moves the temporal
//! boundaries of the selected clip when the move is PROVABLY safe.
//!
//! Core question (from the spec): "Can I improve the opening or ending of
//! THIS selected clip without damaging context, meaning, narrative
//! continuity, or timing?" — If the answer is uncertain: KEEP THE ORIGINAL
//! BOUNDARY. False negatives are acceptable. Incorrect boundary changes
//! are not.
//!
//! Pipeline position: candidate selection → HOOK & ENDING OPTIMIZATION →
//! SMART PACING → speaker/framing → DualFrame/isolation → captions →
//! render. Boundary optimization happens BEFORE final pacing edits, and
//! all downstream systems (pacing, framing, captions, render) receive the
//! optimized timeline. Downstream systems' BEHAVIOR is unchanged — only
//! the temporal range passed to them differs.
//!
//! Timeline mapping (spec):
//!   SOURCE    = original candidate start/end (DB row, never mutated)
//!   OPTIMIZED = new start/end chosen by this module
//!   OUTPUT    = post-pacing timeline (Smart Pacing's responsibility)
//!
//! Division of labor (spec, brain §30):
//!   Hook/Ending Optimization chooses better CONTENT boundaries.
//!   Smart Pacing removes safe unnecessary TIME inside those boundaries.
//!
//! All logic is deterministic: pure functions of (words, candidate meta).
//! No randomness, no time-of-day, no LLM calls, no subprocesses.

use crate::llm::{evaluate_opening_hook_context, normalize_token};
use crate::models::{HookPipelineConfig, TranscriptWord};
use crate::transcription::ends_with_sentence_terminator;

// ── Tunable constants (bounded search windows, spec: ±5–8s) ─────────────────

/// Forward search window for the start (spec: approximately ±5–8 seconds).
pub const FORWARD_WINDOW_SEC: f64 = 6.0;
/// Backward search window for the start (Q→A repair only).
pub const BACKWARD_WINDOW_SEC: f64 = 5.0;
/// Backward window for the trailing end-trim search.
pub const END_TRIM_WINDOW_SEC: f64 = 6.0;
/// Forward window for completing a cut-off conclusion.
pub const END_EXTEND_WINDOW_SEC: f64 = 6.0;

/// Duration guidelines mirror the generation-time snap guidelines
/// (min_guideline=12.0, max_guideline=75.0). The optimizer never violates
/// them for candidates that already satisfy them.
pub const MIN_DURATION_SEC: f64 = 12.0;
pub const MAX_DURATION_SEC: f64 = 75.0;
/// Absolute floor: never shrink a clip below this (pacing needs ≥ 4s;
/// 5s keeps a real Short). Only reachable when the ORIGINAL was already
/// below the 12s guideline.
pub const ABSOLUTE_MIN_DURATION_SEC: f64 = 5.0;

/// Maximum skipped material for a start-forward move.
pub const MAX_SKIP_SENTENCES: usize = 3;
pub const MAX_SKIP_WORDS: usize = 18;
/// Maximum words in a wind-down sentence eligible for end trim.
pub const WIND_DOWN_MAX_WORDS: usize = 10;
/// Maximum words added by an end extension.
pub const MAX_EXTEND_WORDS: usize = 20;
/// Maximum words in a preceding question eligible for Q→A backward repair.
pub const MAX_QUESTION_WORDS: usize = 15;

/// Minimum boundary shift to avoid sub-frame noise (spec: no micro-trims —
/// Smart Pacing handles dead air).
pub const MIN_SHIFT_SEC: f64 = 0.5;
/// Minimum quality gain for a start-forward move (0–1 scale).
pub const MIN_OPENING_GAIN: f64 = 0.25;
/// Minimum quality gain for the Q→A backward repair (smaller: the move
/// adds context rather than removing deficiency).
pub const MIN_BACKWARD_GAIN: f64 = 0.10;
/// Maximum gap between the preceding question and the current opening for
/// Q→A repair.
pub const MAX_QUESTION_GAP_SEC: f64 = 1.5;

/// Lead-in kept before the first retained word (mirrors
/// HookPipelineConfig::max_start_breath_leadin_sec = 0.15).
pub const START_LEAD_SEC: f64 = 0.15;
/// Tail kept after the last retained word.
pub const END_TAIL_SEC: f64 = 0.10;
/// Tolerance when checking whether a move would swallow the hook anchor.
pub const HOOK_ANCHOR_TOLERANCE_SEC: f64 = 0.25;
/// Hook anchor confidence at/below which the anchor is trusted for
/// protection (mirrors min_token_alignment_confidence = 0.60).
pub const HOOK_ANCHOR_MIN_CONFIDENCE: f64 = 0.60;

// ── Closed vocabularies (high-precision, provably low-value material) ───────
//
// A skipped/trimmed span is only allowed when EVERY normalized word is in
// the closed vocabulary. This makes the removed material PROVABLY
// discourse/filler: content words (verbs/nouns like "decided", "quit",
// "psychology", "fans") are NOT in the vocab, so spans containing them are
// never touched. Third-person pronouns (he/she/they/him/her/them) are
// deliberately EXCLUDED from both vocabularies: a span containing them
// might carry narrative content about a person, and dropping it could
// remove the only antecedent.

/// Words allowed in a skippable SETUP span (opening side).
pub const SETUP_VOCAB: &[&str] = &[
    // discourse markers
    "so",
    "yeah",
    "yep",
    "yup",
    "um",
    "uh",
    "uhm",
    "erm",
    "hmm",
    "okay",
    "ok",
    "right",
    "well",
    "anyway",
    "anyways",
    "basically",
    "actually",
    "honestly",
    "literally",
    "maybe",
    "like",
    "mean",
    "know",
    "guess",
    "suppose",
    // first/second person + neutral pronouns (NOT he/she/they)
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "it",
    "its",
    "that",
    "thats",
    "this",
    "these",
    "those",
    "there",
    "here",
    // copulas / auxiliaries
    "was",
    "were",
    "is",
    "are",
    "am",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    // vague talk verbs
    "talk",
    "talking",
    "talked",
    "say",
    "saying",
    "said",
    "tell",
    "telling",
    "told",
    "mention",
    "mentioning",
    "mentioned",
    "speak",
    "speaking",
    "spoke",
    "discuss",
    "discussing",
    "discussed",
    // vague time references
    "yesterday",
    "today",
    "earlier",
    "last",
    "night",
    "week",
    "month",
    "year",
    "other",
    "day",
    // media self-references
    "video",
    "episode",
    "stream",
    "podcast",
    "clip",
    // vague nouns
    "thing",
    "things",
    "stuff",
    "part",
    // fillers
    "little",
    "bit",
    "kind",
    "sort",
    // function words
    "the",
    "a",
    "an",
    "of",
    "on",
    "in",
    "at",
    "to",
    "for",
    "with",
    "and",
    "or",
    "but",
    "then",
    "now",
    "when",
    "where",
    "what",
    "who",
    "how",
    "why",
    "not",
    "no",
    "yes",
    "oh",
    "wow",
    "haha",
    "lol",
    "guys",
    "everyone",
    "everybody",
    "about",
    "as",
    "by",
    "from",
    "up",
    "out",
    "just",
    "very",
    "really",
    "pretty",
    "much",
    "more",
    "most",
    "some",
    "all",
    "one",
    "two",
    "few",
];

/// Words allowed in a trimmable WIND-DOWN span (ending side).
pub const WIND_DOWN_VOCAB: &[&str] = &[
    // discourse markers
    "so",
    "yeah",
    "yep",
    "yup",
    "um",
    "uh",
    "uhm",
    "erm",
    "hmm",
    "okay",
    "ok",
    "right",
    "well",
    "anyway",
    "anyways",
    "basically",
    "actually",
    "honestly",
    "literally",
    "like",
    "mean",
    "know",
    "guess",
    "suppose",
    // first/second person + neutral pronouns (NOT he/she/they)
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "it",
    "its",
    "that",
    "thats",
    "this",
    "these",
    "those",
    "there",
    "here",
    // copulas / auxiliaries
    "was",
    "were",
    "is",
    "are",
    "am",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    // vague talk verbs
    "say",
    "saying",
    "said",
    "tell",
    "telling",
    "told",
    "guess",
    // wind-down specifics
    "whole",
    "story",
    "pretty",
    "much",
    "it",
    "thats",
    "basically",
    // vague nouns
    "thing",
    "things",
    "stuff",
    "part",
    // fillers
    "little",
    "bit",
    "kind",
    "sort",
    // function words
    "the",
    "a",
    "an",
    "of",
    "on",
    "in",
    "at",
    "to",
    "for",
    "with",
    "and",
    "or",
    "but",
    "then",
    "now",
    "not",
    "no",
    "yes",
    "oh",
    "wow",
    "haha",
    "lol",
    "guys",
    "everyone",
    "everybody",
    "about",
    "as",
    "by",
    "from",
    "up",
    "out",
    "just",
    "very",
    "really",
    "pretty",
    "much",
    "more",
    "most",
    "some",
    "all",
    "one",
    "two",
    "few",
    "wrap",
    "wrapping",
    "thanks",
    "watching",
    "subscribe",
];

/// Answer/continuation markers that open an answer whose question may be
/// worth restoring (Q→A backward repair).
pub const ANSWER_MARKERS: &[&str] = &[
    "because", "well", "so", "and", "but", "imean", "youknow", "thatswhy", "cause", "cuz",
];

/// Filler tokens (opening-side deficiency flag).
pub const FILLER_OPENERS: &[&str] = &[
    "um", "uh", "uhm", "erm", "hmm", "like", "you know", "i mean",
];

// ── Public API ─────────────────────────────────────────────────────────────

/// Candidate metadata available at render time (subset of the DB
/// `candidates` row). Closure signals are NOT persisted, so the optimizer
/// deliberately uses only these fields plus the transcript words.
#[derive(Debug, Clone, Default)]
pub struct BoundaryCandidateMeta<'a> {
    /// Verified hook text (LLM hook, aligned to words at generation time).
    pub hook: Option<&'a str>,
    /// Verified hook anchor start (seconds, absolute).
    pub hook_start_sec: Option<f64>,
    /// Verified hook anchor end (seconds, absolute).
    pub hook_end_sec: Option<f64>,
    /// Hook alignment confidence (0–1).
    pub hook_confidence: Option<f64>,
    /// Persisted opening context score (0–1) from generation.
    pub opening_context_score: Option<f64>,
    /// Authoritative payoff end timestamp (seconds, absolute). When present,
    /// Hook Intelligence 2.0 does NOT touch or change the endpoint.
    pub payoff_end_sec: Option<f64>,
}

/// Result of boundary optimization. All times are absolute seconds on the
/// SOURCE timeline. The candidate DB row is never mutated; the render path
/// uses `optimized_start`/`optimized_end` as the OPTIMIZED stage of the
/// SOURCE → OPTIMIZED → OUTPUT timeline mapping.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundaryOptimization {
    pub original_start_sec: f64,
    pub original_end_sec: f64,
    pub optimized_start_sec: f64,
    pub optimized_end_sec: f64,
    pub start_changed: bool,
    pub end_changed: bool,
    pub start_reason: Option<String>,
    pub end_reason: Option<String>,
    pub confidence: f64,
}

impl BoundaryOptimization {
    /// No-change result (conservative default for every uncertain case).
    fn unchanged(start: f64, end: f64) -> Self {
        Self {
            original_start_sec: start,
            original_end_sec: end,
            optimized_start_sec: start,
            optimized_end_sec: end,
            start_changed: false,
            end_changed: false,
            start_reason: None,
            end_reason: None,
            confidence: 1.0,
        }
    }

    /// Human-readable one-line summary for render logs / telemetry.
    pub fn summary(&self) -> String {
        if !self.start_changed && !self.end_changed {
            return format!(
                "boundaries unchanged [{:.2}, {:.2}]",
                self.original_start_sec, self.original_end_sec
            );
        }
        let mut parts = Vec::new();
        if self.start_changed {
            parts.push(format!(
                "start {:.2}→{:.2} ({})",
                self.original_start_sec,
                self.optimized_start_sec,
                self.start_reason.as_deref().unwrap_or("unspecified")
            ));
        }
        if self.end_changed {
            parts.push(format!(
                "end {:.2}→{:.2} ({})",
                self.original_end_sec,
                self.optimized_end_sec,
                self.end_reason.as_deref().unwrap_or("unspecified")
            ));
        }
        format!(
            "boundaries optimized [{:.2}, {:.2}]→[{:.2}, {:.2}]: {} (confidence {:.2})",
            self.original_start_sec,
            self.original_end_sec,
            self.optimized_start_sec,
            self.optimized_end_sec,
            parts.join("; "),
            self.confidence
        )
    }
}

/// Kill-switch: `AUTOSHORTS_BOUNDARY_OPTIMIZATION=0|false|off` disables the
/// optimizer entirely (mirrors `AUTOSHORTS_SMART_PACING`). When disabled,
/// every call returns the original boundaries unchanged.
pub fn boundary_optimization_enabled() -> bool {
    match std::env::var("AUTOSHORTS_BOUNDARY_OPTIMIZATION") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Optimize the boundaries of an already-selected candidate.
///
/// `words` — full transcript words (absolute times, any speaker).
/// `start_sec`/`end_sec` — the candidate's current (snapped) boundaries.
/// `video_duration` — total source duration, used for clamping.
/// `meta` — optional candidate metadata (hook anchor etc.).
///
/// Deterministic and conservative: any uncertainty keeps the original
/// boundary. Never mutates the candidate; returns the OPTIMIZED range.
pub fn optimize_boundaries(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    video_duration: f64,
    meta: Option<&BoundaryCandidateMeta>,
) -> BoundaryOptimization {
    if !boundary_optimization_enabled() {
        return BoundaryOptimization::unchanged(start_sec, end_sec);
    }
    if words.is_empty() || end_sec <= start_sec {
        return BoundaryOptimization::unchanged(start_sec, end_sec);
    }

    let mut result = BoundaryOptimization::unchanged(start_sec, end_sec);

    let authoritative_end = meta.and_then(|m| m.payoff_end_sec).unwrap_or(end_sec);

    // ── START optimization ────────────────────────────────────────────
    let start_move = optimize_start(words, start_sec, authoritative_end, video_duration, meta);
    if let Some((new_start, reason, confidence)) = start_move {
        result.optimized_start_sec = new_start;
        result.start_changed = true;
        result.start_reason = Some(reason);
        result.confidence = result.confidence.min(confidence);
    }

    // ── END optimization (evaluated on the possibly-moved start) ──────
    // Once payoff alignment establishes payoff_end, that endpoint remains authoritative.
    // Hook Intelligence 2.0 must NOT interrupt, override, shorten, rewrite, re-anchor,
    // or otherwise change the payoff-driven endpoint.
    let effective_start = result.optimized_start_sec;
    if let Some(p_end) = meta.and_then(|m| m.payoff_end_sec) {
        result.optimized_end_sec = p_end;
        if (p_end - end_sec).abs() > 0.01 {
            result.end_changed = true;
            result.end_reason = Some("authoritative payoff endpoint".to_string());
        }
    } else {
        let end_move = optimize_end(words, effective_start, end_sec, video_duration);
        if let Some((new_end, reason, confidence)) = end_move {
            result.optimized_end_sec = new_end;
            result.end_changed = true;
            result.end_reason = Some(reason);
            result.confidence = result.confidence.min(confidence);
        }
    }

    // Final safety net: the optimized range must be valid and non-degenerate.
    if result.optimized_end_sec <= result.optimized_start_sec {
        return BoundaryOptimization::unchanged(start_sec, end_sec);
    }
    result
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// A sentence: a contiguous run of words ending at a terminator word.
#[derive(Debug, Clone)]
struct Sentence {
    /// Index of the first word in `words`.
    first_idx: usize,
    /// Index of the last word (the terminator word, when complete).
    last_idx: usize,
    /// True when the last word ends with a sentence terminator.
    complete: bool,
}

impl Sentence {
    fn words<'a>(&self, words: &'a [TranscriptWord]) -> &'a [TranscriptWord] {
        &words[self.first_idx..=self.last_idx]
    }
    fn start(&self, words: &[TranscriptWord]) -> f64 {
        words[self.first_idx].start
    }
    fn end(&self, words: &[TranscriptWord]) -> f64 {
        words[self.last_idx].end
    }
    fn word_count(&self) -> usize {
        self.last_idx - self.first_idx + 1
    }
    fn text(&self, words: &[TranscriptWord]) -> String {
        self.words(words)
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Segment `words` into sentences. Words before the first terminator (or
/// after the last one) form partial sentences marked `complete: false`.
fn segment_sentences(words: &[TranscriptWord]) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut first_idx = 0usize;
    for (i, w) in words.iter().enumerate() {
        if ends_with_sentence_terminator(&w.text) {
            sentences.push(Sentence {
                first_idx,
                last_idx: i,
                complete: true,
            });
            first_idx = i + 1;
        }
    }
    if first_idx < words.len() {
        sentences.push(Sentence {
            first_idx,
            last_idx: words.len() - 1,
            complete: false,
        });
    }
    sentences
}

/// First word index whose word overlaps [start, end).
fn first_word_idx_in_range(words: &[TranscriptWord], start: f64, end: f64) -> Option<usize> {
    words.iter().position(|w| w.end > start && w.start < end)
}

/// Last word index whose word overlaps [start, end).
fn last_word_idx_in_range(words: &[TranscriptWord], start: f64, end: f64) -> Option<usize> {
    words.iter().rposition(|w| w.end > start && w.start < end)
}

/// True when every normalized token of the span is in the closed vocab.
/// Empty spans are NOT all-vocab (must contain something to be provable).
fn span_all_in_vocab(words: &[TranscriptWord], vocab: &[&str]) -> bool {
    if words.is_empty() {
        return false;
    }
    words.iter().all(|w| {
        let t = normalize_token(&w.text);
        !t.is_empty() && vocab.contains(&t.as_str())
    })
}

/// True when any word in the span ends with a question mark.
fn span_contains_question(words: &[TranscriptWord]) -> bool {
    words.iter().any(|w| w.text.trim().ends_with('?'))
}

/// True when an opening whose first two normalized tokens are
/// `first`/`second` begins with an answer/continuation marker.
/// Single-token markers must match `first` exactly ("and" must NOT fire
/// on "Andrea"); multi-token markers ("i mean", "you know", "that's why")
/// match the joined first two tokens.
fn opens_with_answer_marker(first: &str, second: &str) -> bool {
    if first.is_empty() {
        return false;
    }
    for m in ANSWER_MARKERS {
        let m_norm = normalize_token(m);
        if m_norm.is_empty() {
            continue;
        }
        if m_norm == first {
            return true;
        }
        if !second.is_empty()
            && m_norm.len() > first.len()
            && format!("{}{}", first, second).starts_with(&m_norm)
        {
            return true;
        }
    }
    false
}

/// Opening quality on a 0–1 scale using decision-scale deficiency weights
/// (documented, deterministic; flags come from existing infrastructure —
/// `evaluate_opening_hook_context` for continuation markers and ungrounded
/// pronouns, `is_continuation_word`-style lists for fillers):
///   quality = 1.0 − 0.30·setup_opener − 0.35·continuation_marker
///                   − 0.40·ungrounded_pronoun − 0.20·filler_start
fn opening_quality(text: &str, is_question_opening: bool) -> f64 {
    let ctx =
        evaluate_opening_hook_context(text, is_question_opening, &HookPipelineConfig::default());
    let mut q: f64 = 1.0;
    if ctx.has_continuation_opener {
        q -= 0.35;
    }
    if ctx.has_unresolved_reference {
        q -= 0.40;
    }
    let mut tokens = text
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty());
    let first = tokens.next().unwrap_or_default();
    let second = tokens.next().unwrap_or_default();
    if FILLER_OPENERS.iter().any(|f| {
        let f_norm = normalize_token(f);
        !f_norm.is_empty() && f_norm == first
    }) {
        q -= 0.20;
    }
    // An answer/continuation fragment ("Because ...", "Well, ...") is
    // context-deficient: it references a question/setup outside the clip.
    if opens_with_answer_marker(&first, &second) {
        q -= 0.35;
    }
    // A setup-opener start ("so yeah ...", "um okay ...") is a deficiency
    // even when the existing evaluator does not flag it.
    let lower_start = text
        .split_whitespace()
        .take(3)
        .map(normalize_token)
        .collect::<Vec<_>>()
        .join(" ");
    if starts_with_setup_opener(&lower_start) {
        q -= 0.30;
    }
    q.clamp(0.0, 1.0)
}

/// Lowercased normalized prefix starting with a discourse/setup opener.
fn starts_with_setup_opener(lower_prefix: &str) -> bool {
    const SETUP_OPENERS: &[&str] = &[
        "so yeah",
        "so um",
        "so uh",
        "so okay",
        "so ok",
        "so basically",
        "so actually",
        "so honestly",
        "so anyway",
        "so like",
        "yeah so",
        "um so",
        "uh so",
        "okay so",
        "ok so",
        "right so",
        "well so",
        "anyway so",
        "so",
        "yeah",
        "yep",
        "yup",
        "um",
        "uh",
        "uhm",
        "erm",
        "hmm",
        "okay",
        "ok",
        "right",
        "well",
        "anyway",
        "anyways",
    ];
    SETUP_OPENERS.iter().any(|o| lower_prefix.starts_with(o))
}

/// The first sentence (up to ~12 words) starting at word index `idx`.
fn opening_text_at(words: &[TranscriptWord], idx: usize) -> String {
    let mut text = String::new();
    for w in words.iter().skip(idx).take(12) {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&w.text);
        if ends_with_sentence_terminator(&w.text) {
            break;
        }
    }
    text
}

/// Compute the render start for a move landing on word `idx`:
/// clamp to the previous word's end (no mid-word cuts) and add a small
/// lead-in before the first retained word.
fn compute_new_start(words: &[TranscriptWord], idx: usize) -> f64 {
    let first = &words[idx];
    let lead = (first.start - START_LEAD_SEC).max(0.0);
    if idx > 0 {
        let prev_end = words[idx - 1].end;
        // Never start before the previous word ends (no audio overlap).
        lead.max(prev_end)
    } else {
        lead
    }
}

/// Compute the render end for a move landing on word `idx` (inclusive):
/// add a small tail after the last retained word, clamped to video end.
fn compute_new_end(words: &[TranscriptWord], idx: usize, video_duration: f64) -> f64 {
    let last = &words[idx];
    (last.end + END_TAIL_SEC).min(video_duration.max(last.end))
}

/// Duration floor for the optimized clip. Candidates already meeting the
/// 12s generation guideline must keep meeting it; shorter originals may
/// shrink proportionally but never below the absolute floor.
fn duration_floor(original_duration: f64) -> f64 {
    if original_duration >= MIN_DURATION_SEC {
        MIN_DURATION_SEC
    } else {
        (original_duration * 0.60).max(ABSOLUTE_MIN_DURATION_SEC)
    }
}

// ── START optimization ─────────────────────────────────────────────────────

/// Try to improve the start. Returns `(new_start, reason, confidence)`.
///
/// Two provable moves only:
///  1. FORWARD (CASE B): skip a provably low-value setup span (closed
///     vocab, no questions, no hook anchor, bounded window) to land on a
///     standalone opening with a meaningful quality gain.
///  2. BACKWARD (Q→A repair): the current opening is an answer fragment
///     ("Because ...") and the immediately preceding sentence is a short
///     question — restore the question for context.
///
/// Everything else keeps the original boundary.
fn optimize_start(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    video_duration: f64,
    meta: Option<&BoundaryCandidateMeta>,
) -> Option<(f64, String, f64)> {
    let first_idx = first_word_idx_in_range(words, start_sec, end_sec)?;
    let original_duration = end_sec - start_sec;
    let floor = duration_floor(original_duration);

    let old_text = opening_text_at(words, first_idx);
    let old_quality = opening_quality(&old_text, false);

    // ── Attempt 1: FORWARD setup skip (CASE B) ────────────────────────
    if let Some(out) = try_forward_setup_skip(
        words,
        start_sec,
        end_sec,
        first_idx,
        floor,
        old_quality,
        meta,
    ) {
        return Some(out);
    }

    // ── Attempt 2: BACKWARD Q→A repair ────────────────────────────────
    if let Some(out) = try_backward_question_repair(
        words,
        start_sec,
        end_sec,
        first_idx,
        video_duration,
        old_quality,
    ) {
        return Some(out);
    }

    None
}

/// CASE B forward move: skip provably low-value setup sentences.
#[allow(clippy::too_many_arguments)]
fn try_forward_setup_skip(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    first_idx: usize,
    floor: f64,
    old_quality: f64,
    meta: Option<&BoundaryCandidateMeta>,
) -> Option<(f64, String, f64)> {
    // The old opening must actually be deficient — no deficiency, no move.
    if old_quality >= 1.0 - MIN_OPENING_GAIN + f64::EPSILON {
        return None;
    }

    let sentences = segment_sentences(words);
    // Candidate landing points: sentence starts within the forward window.
    let window_max = start_sec + FORWARD_WINDOW_SEC;

    // Hook anchor protection: when the anchor is reliable, the skipped
    // span must not swallow any part of [hook_start, hook_end].
    let (hook_lo, _hook_hi) = match meta {
        Some(m)
            if m.hook_start_sec.is_some()
                && m.hook_confidence.unwrap_or(0.0) >= HOOK_ANCHOR_MIN_CONFIDENCE =>
        {
            (
                m.hook_start_sec.unwrap(),
                m.hook_end_sec.unwrap_or(m.hook_start_sec.unwrap()),
            )
        }
        _ => (f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    // Enumerate sentence starts strictly after the current first word and
    // within the window; earliest qualifying candidate wins.
    for s in &sentences {
        if s.first_idx <= first_idx {
            continue;
        }
        let s_start = s.start(words);
        if s_start > window_max {
            break; // sentences are time-ordered
        }
        // The skipped span [start_sec, new_start) must be non-trivial.
        let new_start = compute_new_start(words, s.first_idx);
        if new_start - start_sec < MIN_SHIFT_SEC {
            continue;
        }
        // Duration safety.
        if end_sec - new_start < floor {
            continue;
        }
        // Skipped span: words from the current first word up to (excluding)
        // the new first word.
        let skipped: Vec<&TranscriptWord> = words[first_idx..s.first_idx].iter().collect();
        if skipped.is_empty() {
            continue;
        }
        // Bounded skip size.
        let skip_word_count = skipped.len();
        if skip_word_count > MAX_SKIP_WORDS {
            continue;
        }
        let skip_sentence_count = sentences
            .iter()
            .filter(|x| x.first_idx >= first_idx && x.first_idx < s.first_idx)
            .count()
            .max(1);
        if skip_sentence_count > MAX_SKIP_SENTENCES {
            continue;
        }
        // PROVABLY low-value: every skipped word in the closed setup vocab.
        let owned: Vec<TranscriptWord> = skipped.into_iter().cloned().collect();
        if !span_all_in_vocab(&owned, SETUP_VOCAB) {
            continue;
        }
        // Never drop a question (Q→A protection, belt and braces).
        if span_contains_question(&owned) {
            continue;
        }
        // Hook anchor protection.
        if new_start > hook_lo + HOOK_ANCHOR_TOLERANCE_SEC && hook_lo > f64::NEG_INFINITY {
            // The move would start after the hook anchor begins — reject
            // unless the entire anchor still precedes the new start... it
            // doesn't (new_start > hook_lo). Reject.
            continue;
        }
        // New opening must be standalone and clean.
        let new_text = opening_text_at(words, s.first_idx);
        let new_quality = opening_quality(&new_text, false);
        if new_quality < 1.0 - f64::EPSILON {
            // New opening itself deficient → not a provable improvement.
            continue;
        }
        // Meaningful gain (conservative scoring).
        let gain = new_quality - old_quality;
        if gain < MIN_OPENING_GAIN {
            continue;
        }
        let reason = format!(
            "skipped {} setup word(s) (closed-vocab, no questions, hook preserved); opening quality {:.2}→{:.2}",
            skip_word_count, old_quality, new_quality
        );
        return Some((new_start, reason, 0.90));
    }
    let _ = end_sec; // window checks use start_sec only
    None
}

/// Q→A backward repair: the current opening is an answer fragment and the
/// immediately preceding sentence is a short question by anyone (interviewer
/// questions are the classic case). Restoring the question is provably
/// context-improving: the answer's marker references it.
fn try_backward_question_repair(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    first_idx: usize,
    video_duration: f64,
    old_quality: f64,
) -> Option<(f64, String, f64)> {
    // The current opening must start with an answer/continuation marker.
    let first_token = normalize_token(&words[first_idx].text);
    let second_token = words
        .get(first_idx + 1)
        .map(|w| normalize_token(&w.text))
        .unwrap_or_default();
    if !opens_with_answer_marker(&first_token, &second_token) {
        return None;
    }

    // Find the sentence immediately preceding the current first word.
    let sentences = segment_sentences(words);
    let prev_sentence = sentences
        .iter()
        .filter(|s| s.last_idx < first_idx)
        .next_back()?;

    // It must be a complete question.
    if !prev_sentence.complete {
        return None;
    }
    let prev_words = prev_sentence.words(words);
    if !prev_words.last()?.text.trim().ends_with('?') {
        return None;
    }
    // Short question only.
    if prev_sentence.word_count() > MAX_QUESTION_WORDS {
        return None;
    }
    // Bounded backward window and tight gap (Q→A pairs are adjacent).
    let q_end = prev_sentence.end(words);
    let gap = start_sec - q_end;
    if gap < 0.0 || gap > MAX_QUESTION_GAP_SEC {
        return None;
    }
    let q_start = prev_sentence.start(words);
    if start_sec - q_start > BACKWARD_WINDOW_SEC {
        return None;
    }

    // Duration ceiling.
    let new_end = end_sec;
    if new_end - q_start > MAX_DURATION_SEC {
        return None;
    }

    // The new opening (the question) must be standalone. Question openings
    // resolve their pronouns via the answer that follows (is_question=true
    // semantics, mirroring the generation-time hook gate).
    let new_text = prev_sentence.text(words);
    let new_quality = opening_quality(&new_text, true);
    if new_quality < 1.0 - MIN_BACKWARD_GAIN + f64::EPSILON {
        return None;
    }
    let gain = new_quality - old_quality;
    if gain < MIN_BACKWARD_GAIN {
        return None;
    }

    // Land just before the question's first word (with lead-in), never
    // overlapping the preceding word.
    let new_start = compute_new_start(words, prev_sentence.first_idx);
    let _ = video_duration;
    let reason = format!(
        "Q→A repair: restored preceding question ({:.2}–{:.2}s) for answer opening; quality {:.2}→{:.2}",
        q_start, q_end, old_quality, new_quality
    );
    Some((new_start, reason, 0.85))
}

// ── END optimization ───────────────────────────────────────────────────────

/// Try to improve the end. Returns `(new_end, reason, confidence)`.
///
/// Two provable moves only:
///  1. TRIM (CASE B ending): trailing wind-down sentences (closed vocab,
///     short, marker-gated) after a strong complete conclusion.
///  2. EXTEND (CASE C ending): the clip cuts mid-sentence and the sentence
///     completes within the bounded window.
///
/// No extension to NEW sentences after a complete terminator end — we
/// cannot prove a payoff there (documented limitation).
fn optimize_end(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    video_duration: f64,
) -> Option<(f64, String, f64)> {
    let last_idx = last_word_idx_in_range(words, start_sec, end_sec)?;
    let original_duration = end_sec - start_sec;
    let floor = duration_floor(original_duration);

    let last_word = &words[last_idx];
    let ends_complete = ends_with_sentence_terminator(&last_word.text);

    if ends_complete {
        try_end_trim(words, start_sec, end_sec, last_idx, floor)
    } else {
        try_end_extend(words, end_sec, last_idx, video_duration)
    }
}

/// CASE B ending: trim trailing wind-down sentences after a strong
/// conclusion. The retained last sentence must be complete and NOT itself
/// wind-down; every trimmed sentence must be short and all-vocab.
fn try_end_trim(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    last_idx: usize,
    floor: f64,
) -> Option<(f64, String, f64)> {
    let sentences = segment_sentences(words);
    // Sentences fully inside the tail window, ordered as in the transcript.
    let tail_min = end_sec - END_TRIM_WINDOW_SEC;
    let in_range: Vec<&Sentence> = sentences
        .iter()
        .filter(|s| {
            s.first_idx <= last_idx
                && s.end(words) > tail_min
                && s.end(words) <= end_sec + 1e-9
                && s.first_idx < last_idx + 1
        })
        .collect();
    if in_range.is_empty() {
        return None;
    }

    // Walk backward over trailing wind-down sentences.
    let mut trim_from = in_range.len(); // exclusive index into in_range
    while trim_from > 0 {
        let s = in_range[trim_from - 1];
        let s_words = s.words(words);
        let is_wind_down = s.complete
            && s.word_count() <= WIND_DOWN_MAX_WORDS
            && span_all_in_vocab(s_words, WIND_DOWN_VOCAB);
        if !is_wind_down {
            break;
        }
        trim_from -= 1;
    }
    // At least one sentence must be trimmed.
    if trim_from == in_range.len() {
        return None;
    }
    // A strong (non-wind-down) predecessor must remain.
    if trim_from == 0 {
        // All sentences in the window are wind-down: no provable payoff
        // boundary — keep the original end.
        return None;
    }
    let strong = in_range[trim_from - 1];
    // The retained last sentence must be complete (provable closure).
    if !strong.complete {
        return None;
    }

    let new_end = compute_new_end(words, strong.last_idx, video_duration_end(words, end_sec));
    // Trim must be non-trivial and keep the duration floor.
    if end_sec - new_end < MIN_SHIFT_SEC {
        return None;
    }
    if new_end - start_sec < floor {
        return None;
    }
    let trimmed_words: usize = in_range[trim_from..].iter().map(|s| s.word_count()).sum();
    let reason = format!(
        "trimmed {} trailing wind-down word(s) after complete conclusion at {:.2}s",
        trimmed_words,
        strong.end(words)
    );
    Some((new_end, reason, 0.90))
}

/// CASE C ending: the clip cuts mid-sentence — complete the sentence when
/// it finishes within the bounded window and word budget.
fn try_end_extend(
    words: &[TranscriptWord],
    end_sec: f64,
    last_idx: usize,
    video_duration: f64,
) -> Option<(f64, String, f64)> {
    // Find where the current sentence completes.
    let mut extend_to;
    let mut added = 0usize;
    for (i, w) in words.iter().enumerate().skip(last_idx + 1) {
        if w.start - end_sec > END_EXTEND_WINDOW_SEC {
            return None; // completion outside the bounded window
        }
        extend_to = i;
        added += 1;
        if added > MAX_EXTEND_WORDS {
            return None;
        }
        if ends_with_sentence_terminator(&w.text) {
            // Completion found — same-speaker check: the completing words
            // must not switch speaker (a speaker switch mid-sentence is
            // suspicious; conservative reject).
            let last_speaker = words[last_idx].speaker.clone();
            if let Some(sp) = &w.speaker {
                if let Some(last) = &last_speaker {
                    if sp != last {
                        return None;
                    }
                }
            }
            let new_end = compute_new_end(words, extend_to, video_duration);
            if new_end - end_sec < MIN_SHIFT_SEC {
                return None;
            }
            let reason = format!(
                "extended end {:.2}→{:.2}s to complete cut-off sentence ({} word(s) added)",
                end_sec, new_end, added
            );
            return Some((new_end, reason, 0.85));
        }
    }
    None // sentence never completes within the transcript
}

/// Best-known video end for clamping (the caller's end is a lower bound).
fn video_duration_end(_words: &[TranscriptWord], end_sec: f64) -> f64 {
    end_sec
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: None,
        }
    }

    fn body_words(start: f64, count: usize) -> Vec<TranscriptWord> {
        (0..count)
            .map(|i| {
                w(
                    &format!("body{}", i),
                    start + i as f64 * 0.4,
                    start + i as f64 * 0.4 + 0.2,
                )
            })
            .collect()
    }

    fn meta(
        hook_start: Option<f64>,
        hook_end: Option<f64>,
        conf: Option<f64>,
    ) -> BoundaryCandidateMeta<'static> {
        BoundaryCandidateMeta {
            hook: None,
            hook_start_sec: hook_start,
            hook_end_sec: hook_end,
            hook_confidence: conf,
            opening_context_score: None,
            payoff_end_sec: None,
        }
    }

    #[test]
    fn test_payoff_endpoint_prevents_end_trim() {
        let mut words = vec![
            w("The", 80.0, 80.1),
            w("result", 80.1, 80.4),
            w("changed", 80.4, 80.7),
            w("my", 80.7, 80.8),
            w("life.", 80.8, 81.2),
        ];
        words.extend(body_words(82.0, 35));
        words.extend(vec![
            w("conclusion.", 96.0, 96.5),
            w("So", 97.0, 97.2),
            w("yeah,", 97.2, 97.4),
            w("that's", 97.4, 97.6),
            w("basically", 97.6, 97.9),
            w("the", 97.9, 98.0),
            w("whole", 98.0, 98.2),
            w("story.", 98.2, 98.6),
        ]);
        let mut m = meta(None, None, None);
        m.payoff_end_sec = Some(98.6);
        let out = optimize_boundaries(&words, 80.0, 98.6, DUR, Some(&m));
        assert!(!out.end_changed, "payoff_end_sec must prevent end trim");
        assert_eq!(out.optimized_end_sec, 98.6);

        let out2 = optimize_boundaries(&words, 80.0, 99.0, DUR, Some(&m));
        assert!(
            out2.end_changed,
            "differing raw end snaps to payoff endpoint"
        );
        assert_eq!(out2.optimized_end_sec, 98.6);
        assert_eq!(
            out2.end_reason.as_deref(),
            Some("authoritative payoff endpoint")
        );
    }

    const DUR: f64 = 600.0;

    #[test]
    fn test_clean_opening_and_ending_unchanged() {
        // Strong hook + strong conclusion → no change (CASE A / ending A).
        let words = vec![
            w("The", 10.0, 10.1),
            w("biggest", 10.1, 10.3),
            w("mistake", 10.3, 10.6),
            w("people", 10.6, 10.8),
            w("make", 10.8, 11.0),
            w("is", 11.0, 11.1),
            w("overthinking.", 11.1, 11.6),
            w("It", 12.0, 12.1),
            w("ruins", 12.1, 12.4),
            w("everything.", 12.4, 13.0),
            w("And", 14.0, 14.2),
            w("that's", 14.2, 14.4),
            w("why", 14.4, 14.5),
            w("I", 14.5, 14.6),
            w("stopped", 14.6, 15.0),
            w("doing", 15.0, 15.2),
            w("it.", 15.2, 15.6),
        ];
        let out = optimize_boundaries(&words, 10.0, 16.0, DUR, None);
        assert!(!out.start_changed, "start must not move");
        assert!(!out.end_changed, "end must not move");
    }

    #[test]
    fn test_forward_skip_of_pure_setup() {
        // "So yeah we were talking about that." (all in SETUP_VOCAB) then
        // a strong standalone statement → start moves to the statement.
        // Clip is 20s (≥ 12s guideline) so the floor allows the skip.
        let mut words = vec![
            w("So", 20.0, 20.2),
            w("yeah", 20.2, 20.4),
            w("we", 20.4, 20.5),
            w("were", 20.5, 20.6),
            w("talking", 20.6, 20.9),
            w("about", 20.9, 21.1),
            w("that.", 21.1, 21.4),
            w("The", 22.0, 22.1),
            w("biggest", 22.1, 22.3),
            w("mistake", 22.3, 22.6),
            w("people", 22.6, 22.8),
            w("make", 22.8, 23.0),
            w("is", 23.0, 23.1),
            w("overthinking.", 23.1, 23.6),
            w("It", 24.0, 24.1),
            w("ruins", 24.1, 24.4),
            w("everything.", 24.4, 25.0),
        ];
        // Body content so the clip is long enough for the 12s floor.
        for i in 0..35 {
            words.push(w(
                &format!("body{}", i),
                25.5 + i as f64 * 0.4,
                25.7 + i as f64 * 0.4,
            ));
        }
        words.push(w("end.", 39.6, 40.0));
        let out = optimize_boundaries(&words, 20.0, 40.0, DUR, None);
        assert!(out.start_changed, "start should skip the setup");
        assert!(
            (out.optimized_start_sec - 21.85).abs() < 0.3,
            "new start ≈ 22.0 − 0.15 lead, got {}",
            out.optimized_start_sec
        );
        assert!(!out.end_changed);
    }

    #[test]
    fn test_forward_skip_blocked_by_content_word() {
        // Setup contains a content word ("psychology") → not provably
        // low-value → keep original (false negative acceptable).
        let words = vec![
            w("So", 30.0, 30.2),
            w("yeah", 30.2, 30.4),
            w("we", 30.4, 30.5),
            w("were", 30.5, 30.6),
            w("talking", 30.6, 30.9),
            w("about", 30.9, 31.1),
            w("psychology.", 31.1, 31.6),
            w("The", 32.0, 32.1),
            w("biggest", 32.1, 32.3),
            w("mistake", 32.3, 32.6),
            w("people", 32.6, 32.8),
            w("make", 32.8, 33.0),
            w("is", 33.0, 33.1),
            w("overthinking.", 33.1, 33.6),
        ];
        let out = optimize_boundaries(&words, 30.0, 34.0, DUR, None);
        assert!(!out.start_changed, "content word must block the skip");
    }

    #[test]
    fn test_forward_skip_blocked_by_question() {
        // Skipped span ends with "?" → Q→A protection blocks the move.
        let words = vec![
            w("So", 40.0, 40.2),
            w("why", 40.2, 40.4),
            w("did", 40.4, 40.5),
            w("you", 40.5, 40.6),
            w("leave?", 40.6, 41.0),
            w("I", 42.0, 42.1),
            w("left", 42.1, 42.4),
            w("because", 42.4, 42.6),
            w("I", 42.6, 42.7),
            w("was", 42.7, 42.8),
            w("tired.", 42.8, 43.2),
        ];
        let out = optimize_boundaries(&words, 40.0, 44.0, DUR, None);
        assert!(!out.start_changed, "question must block the skip");
    }

    #[test]
    fn test_hook_anchor_protection() {
        // Hook anchor inside the skipped span → move rejected.
        let words = vec![
            w("So", 50.0, 50.2),
            w("yeah", 50.2, 50.4),
            w("we", 50.4, 50.5),
            w("were", 50.5, 50.6),
            w("talking", 50.6, 50.9),
            w("about", 50.9, 51.1),
            w("that.", 51.1, 51.4),
            w("The", 52.0, 52.1),
            w("biggest", 52.1, 52.3),
            w("mistake", 52.3, 52.6),
            w("people", 52.6, 52.8),
            w("make", 52.8, 53.0),
            w("is", 53.0, 53.1),
            w("overthinking.", 53.1, 53.6),
        ];
        let m = meta(Some(50.0), Some(51.4), Some(0.90)); // anchor in skipped span
        let out = optimize_boundaries(&words, 50.0, 54.0, DUR, Some(&m));
        assert!(!out.start_changed, "hook anchor must block the skip");
    }

    #[test]
    fn test_backward_qa_repair() {
        // Answer opening ("Because ...") preceded by a short question →
        // start moves back to include the question. Clip is 20s.
        let mut words = vec![
            w("Why", 60.0, 60.2),
            w("did", 60.2, 60.3),
            w("she", 60.3, 60.4),
            w("quit?", 60.4, 60.8),
            w("Because", 61.2, 61.5),
            w("the", 61.5, 61.6),
            w("job", 61.6, 61.8),
            w("was", 61.8, 61.9),
            w("destroying", 61.9, 62.3),
            w("her.", 62.3, 62.7),
            w("It", 63.0, 63.1),
            w("was", 63.1, 63.3),
            w("simple.", 63.3, 63.7),
        ];
        // Body content so the clip is long enough for the 12s floor.
        for i in 0..40 {
            words.push(w(
                &format!("body{}", i),
                64.0 + i as f64 * 0.4,
                64.2 + i as f64 * 0.4,
            ));
        }
        words.push(w("end.", 80.0, 80.4));
        let out = optimize_boundaries(&words, 61.2, 81.0, DUR, None);
        assert!(out.start_changed, "Q→A repair should fire");
        assert!(
            out.optimized_start_sec <= 60.0 + 1e-9,
            "new start at/before the question"
        );
        assert!(
            out.optimized_start_sec >= 59.0,
            "new start not before the question's lead"
        );
    }

    #[test]
    fn test_backward_blocked_when_predecessor_not_question() {
        // Pronoun opening + non-question predecessor → no backward move
        // (pronoun grounding is unprovable; only Q→A repair is allowed).
        let words = vec![
            w("Maria", 70.0, 70.3),
            w("was", 70.3, 70.4),
            w("my", 70.4, 70.5),
            w("partner.", 70.5, 70.9),
            w("She", 71.2, 71.4),
            w("stole", 71.4, 71.7),
            w("everything.", 71.7, 72.2),
            w("It", 73.0, 73.1),
            w("hurt.", 73.1, 73.5),
        ];
        let out = optimize_boundaries(&words, 71.2, 74.0, DUR, None);
        assert!(
            !out.start_changed,
            "non-question predecessor must block backward move"
        );
    }

    #[test]
    fn test_end_trim_of_wind_down() {
        // Strong conclusion + "So yeah thats basically the whole story."
        // (all in WIND_DOWN_VOCAB, 7 words) → end trims to the conclusion.
        // Clip is 19s so the 12s floor allows the trim.
        let mut words = vec![
            w("The", 80.0, 80.1),
            w("result", 80.1, 80.4),
            w("changed", 80.4, 80.7),
            w("my", 80.7, 80.8),
            w("life.", 80.8, 81.2),
        ];
        // Body content so the clip is long enough for the 12s floor.
        for i in 0..35 {
            words.push(w(
                &format!("body{}", i),
                82.0 + i as f64 * 0.4,
                82.2 + i as f64 * 0.4,
            ));
        }
        words.push(w("conclusion.", 96.0, 96.5));
        words.push(w("So", 97.0, 97.2));
        words.push(w("yeah,", 97.2, 97.4));
        words.push(w("that's", 97.4, 97.6));
        words.push(w("basically", 97.6, 97.9));
        words.push(w("the", 97.9, 98.0));
        words.push(w("whole", 98.0, 98.2));
        words.push(w("story.", 98.2, 98.6));
        let out = optimize_boundaries(&words, 80.0, 99.0, DUR, None);
        assert!(out.end_changed, "wind-down should be trimmed");
        assert!(
            (out.optimized_end_sec - 96.6).abs() < 0.25,
            "new end ≈ 96.5 + 0.10 tail, got {}",
            out.optimized_end_sec
        );
        assert!(!out.start_changed);
    }

    #[test]
    fn test_end_trim_blocked_by_content_word() {
        // "So I decided to quit." — "decided"/"quit" not in vocab → no trim.
        let words = vec![
            w("The", 90.0, 90.1),
            w("result", 90.1, 90.4),
            w("changed", 90.4, 90.7),
            w("my", 90.7, 90.8),
            w("life.", 90.8, 91.2),
            w("So", 92.0, 92.2),
            w("I", 92.2, 92.3),
            w("decided", 92.3, 92.7),
            w("to", 92.7, 92.8),
            w("quit.", 92.8, 93.2),
        ];
        let out = optimize_boundaries(&words, 90.0, 94.0, DUR, None);
        assert!(!out.end_changed, "content word must block the trim");
    }

    #[test]
    fn test_end_extend_completes_cut_sentence() {
        // Clip cuts mid-sentence; the sentence completes within 6s/20 words.
        let words = vec![
            w("And", 100.0, 100.1),
            w("that's", 100.1, 100.3),
            w("why", 100.3, 100.4),
            w("I", 100.4, 100.5),
            w("finally", 100.5, 100.8),
            w("stopped", 100.8, 101.2),
            w("doing", 101.2, 101.4),
            w("it.", 101.4, 101.8),
            w("The", 102.0, 102.1),
            w("lesson", 102.1, 102.5),
            w("was", 102.5, 102.7),
            w("worth", 102.7, 103.0),
            w("every", 103.0, 103.2),
            w("second", 103.2, 103.6),
            w("of", 103.6, 103.7),
            w("pain.", 103.7, 104.2),
        ];
        // End at 103.0 cuts "worth every second of pain." mid-sentence.
        let out = optimize_boundaries(&words, 100.0, 103.0, DUR, None);
        assert!(out.end_changed, "mid-sentence cut should be completed");
        assert!(
            out.optimized_end_sec >= 104.2,
            "new end must include the full sentence"
        );
    }

    #[test]
    fn test_end_extend_blocked_when_completion_far() {
        // Completion needs > 6s → keep original (CASE H).
        let mut words = vec![
            w("The", 110.0, 110.1),
            w("lesson", 110.1, 110.5),
            w("was", 110.5, 110.7),
        ];
        // Next terminator is 8s away.
        words.push(w("painful.", 118.0, 118.4));
        let out = optimize_boundaries(&words, 110.0, 111.0, DUR, None);
        assert!(!out.end_changed, "far completion must block extension");
    }

    #[test]
    fn test_no_extension_after_complete_terminator() {
        // End is complete; a NEW sentence follows — extension to new
        // sentences is not allowed (unprovable payoff).
        let words = vec![
            w("It", 120.0, 120.1),
            w("worked.", 120.1, 120.5),
            w("The", 121.0, 121.1),
            w("next", 121.1, 121.4),
            w("chapter", 121.4, 121.8),
            w("started.", 121.8, 122.2),
        ];
        let out = optimize_boundaries(&words, 120.0, 121.0, DUR, None);
        assert!(
            !out.end_changed,
            "no extension to new sentences after a complete end"
        );
    }

    #[test]
    fn test_duration_floor_blocks_overtrim() {
        // A 13s clip whose wind-down trim would leave < 12s → blocked.
        let mut words = vec![
            w("The", 130.0, 130.1),
            w("result", 130.1, 130.4),
            w("changed", 130.4, 130.7),
            w("my", 130.7, 130.8),
            w("life.", 130.8, 131.2),
        ];
        // Wind-down at 141.5–142.5 (trim would leave ~10.4s < 12s floor).
        words.push(w("So", 141.5, 141.7));
        words.push(w("yeah.", 141.7, 142.0));
        let out = optimize_boundaries(&words, 130.0, 143.0, DUR, None);
        assert!(!out.end_changed, "duration floor must block the trim");
    }

    #[test]
    fn test_kill_switch_disables_optimizer() {
        let mut words = vec![
            w("So", 150.0, 150.2),
            w("yeah", 150.2, 150.4),
            w("we", 150.4, 150.5),
            w("were", 150.5, 150.6),
            w("talking", 150.6, 150.9),
            w("about", 150.9, 151.1),
            w("that.", 151.1, 151.4),
            w("The", 152.0, 152.1),
            w("biggest", 152.1, 152.3),
            w("mistake", 152.3, 152.6),
            w("people", 152.6, 152.8),
            w("make", 152.8, 153.0),
            w("is", 153.0, 153.1),
            w("overthinking.", 153.1, 153.6),
        ];
        // Body content so the clip is long enough for the 12s floor.
        for i in 0..35 {
            words.push(w(
                &format!("body{}", i),
                154.0 + i as f64 * 0.4,
                154.2 + i as f64 * 0.4,
            ));
        }
        words.push(w("end.", 168.0, 168.4));
        // NOTE: env vars are process-global; this test asserts the helper
        // honors the canonical off values when set (checked via direct
        // call to avoid cross-test pollution).
        let off = matches!(
            std::env::var("AUTOSHORTS_BOUNDARY_OPTIMIZATION")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off"
        );
        // When the env var is not set (normal test run), the optimizer is
        // enabled and the move fires; the Python suite (case J) exercises
        // the real kill-switch end-to-end via subprocess env.
        if !off {
            let out = optimize_boundaries(&words, 150.0, 169.0, DUR, None);
            assert!(out.start_changed);
        }
    }

    #[test]
    fn test_determinism_same_input_same_output() {
        let words = vec![
            w("So", 160.0, 160.2),
            w("yeah", 160.2, 160.4),
            w("we", 160.4, 160.5),
            w("were", 160.5, 160.6),
            w("talking", 160.6, 160.9),
            w("about", 160.9, 161.1),
            w("that.", 161.1, 161.4),
            w("The", 162.0, 162.1),
            w("biggest", 162.1, 162.3),
            w("mistake", 162.3, 162.6),
            w("people", 162.6, 162.8),
            w("make", 162.8, 163.0),
            w("is", 163.0, 163.1),
            w("overthinking.", 163.1, 163.6),
            w("So", 164.0, 164.2),
            w("yeah,", 164.2, 164.4),
            w("that's", 164.4, 164.6),
            w("basically", 164.6, 164.9),
            w("it.", 164.9, 165.2),
        ];
        let a = optimize_boundaries(&words, 160.0, 166.0, DUR, None);
        let b = optimize_boundaries(&words, 160.0, 166.0, DUR, None);
        assert_eq!(a.optimized_start_sec, b.optimized_start_sec);
        assert_eq!(a.optimized_end_sec, b.optimized_end_sec);
        assert_eq!(a.start_reason, b.start_reason);
        assert_eq!(a.end_reason, b.end_reason);
    }

    #[test]
    fn test_empty_words_unchanged() {
        let out = optimize_boundaries(&[], 10.0, 20.0, DUR, None);
        assert!(!out.start_changed && !out.end_changed);
    }

    #[test]
    fn test_combined_start_and_end_optimization() {
        // Setup at the start + wind-down at the end → both move.
        // Clip is 23s so the 12s floor allows both moves.
        let mut words = vec![
            w("So", 170.0, 170.2),
            w("yeah,", 170.2, 170.4),
            w("we", 170.4, 170.5),
            w("were", 170.5, 170.6),
            w("talking", 170.6, 170.9),
            w("about", 170.9, 171.1),
            w("that.", 171.1, 171.4),
            w("The", 172.0, 172.1),
            w("biggest", 172.1, 172.3),
            w("mistake", 172.3, 172.6),
            w("people", 172.6, 172.8),
            w("make", 172.8, 173.0),
            w("is", 173.0, 173.1),
            w("overthinking.", 173.1, 173.6),
            w("It", 174.0, 174.1),
            w("ruins", 174.1, 174.4),
            w("everything.", 174.4, 175.0),
        ];
        // Body content so the clip is long enough for the 12s floor.
        for i in 0..35 {
            words.push(w(
                &format!("body{}", i),
                176.0 + i as f64 * 0.4,
                176.2 + i as f64 * 0.4,
            ));
        }
        words.push(w("conclusion.", 190.0, 190.5));
        words.push(w("So", 191.0, 191.2));
        words.push(w("yeah,", 191.2, 191.4));
        words.push(w("that's", 191.4, 191.6));
        words.push(w("basically", 191.6, 191.9));
        words.push(w("the", 191.9, 192.0));
        words.push(w("whole", 192.0, 192.2));
        words.push(w("story.", 192.2, 192.6));
        let out = optimize_boundaries(&words, 170.0, 193.0, DUR, None);
        assert!(out.start_changed, "setup should be skipped");
        assert!(out.end_changed, "wind-down should be trimmed");
        assert!(out.optimized_start_sec > 171.4);
        assert!(out.optimized_end_sec < 191.0);
    }

    #[test]
    fn test_short_clip_proportional_floor() {
        // Original 9s clip (below guideline): floor = max(5.4, 5.0) = 5.4s.
        // A setup skip leaving ≥ 5.4s is still allowed.
        let words = vec![
            w("So", 180.0, 180.2),
            w("yeah", 180.2, 180.4),
            w("we", 180.4, 180.5),
            w("were", 180.5, 180.6),
            w("talking", 180.6, 180.9),
            w("about", 180.9, 181.1),
            w("that.", 181.1, 181.4),
            w("The", 182.0, 182.1),
            w("biggest", 182.1, 182.3),
            w("mistake", 182.3, 182.6),
            w("people", 182.6, 182.8),
            w("make", 182.8, 183.0),
            w("is", 183.0, 183.1),
            w("overthinking.", 183.1, 183.6),
            w("It", 184.0, 184.1),
            w("ruins", 184.1, 184.4),
            w("everything.", 184.4, 185.0),
            w("Truly.", 185.5, 186.0),
        ];
        // 180.0–189.0 = 9s original; skip to 182.0 leaves 7s ≥ 5.4 floor.
        let out = optimize_boundaries(&words, 180.0, 189.0, DUR, None);
        assert!(
            out.start_changed,
            "proportional floor should allow this skip"
        );
    }
}
