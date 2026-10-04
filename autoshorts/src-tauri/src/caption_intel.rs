//! Caption Intelligence 2.0
//!
//! An additive decision layer that produces a lightweight `CaptionIntelPlan` at render time.
//! Intelligently determines:
//! 1. Which words/phrases deserve emphasis (represented as semantic spans).
//! 2. Which words belong to the hook and payoff.
//! 3. Where semantic line breaks should preferably occur (represented as global word indices).
//! 4. Whether confidence is high enough (>= 0.75) to apply changes or fall back safely to baseline behavior.

use serde::{Deserialize, Serialize};

use crate::models::{Candidate, TranscriptWord};
use crate::pacing::SmartPacingPlan;

/// Represents a contiguous semantic span of words marked for emphasis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmphasisSpan {
    /// Inclusive start index into the caption words slice.
    pub start_idx: usize,
    /// Inclusive end index into the caption words slice.
    pub end_idx: usize,
    /// Human-readable explanation of why this span is emphasized.
    pub reason: String,
}

/// The high-level intent plan produced by Caption Intelligence 2.0.
/// Represents intent only — does not directly generate ASS subtitles.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionIntelPlan {
    /// Confidence score in [0.0, 1.0]. Only plans with confidence >= 0.75 are applied.
    pub confidence: f64,
    /// Semantic spans of emphasized words.
    pub emphasis_spans: Vec<EmphasisSpan>,
    /// Flattened unique indices of all words marked for emphasis.
    pub emphasis_word_indices: Vec<usize>,
    /// Global word indices after which a line break is preferred.
    pub line_break_hints: Vec<usize>,
    /// Global word indices that belong to the hook.
    pub hook_word_indices: Vec<usize>,
    /// Global word indices that belong to the payoff.
    pub payoff_word_indices: Vec<usize>,
    /// Summary explanation of the plan decisions.
    pub reason: String,
}

/// Feature / kill switch for Caption Intelligence 2.0.
/// Environment variable: AUTOSHORTS_CAPTION_INTEL
/// Disabled when set to: "0", "false", "off" (case-insensitive).
pub fn caption_intelligence_enabled() -> bool {
    match std::env::var("AUTOSHORTS_CAPTION_INTEL") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Normalizes a token for matching (lowercase, alphanumeric characters only).
pub fn normalize_token(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Maps a source-timeline interval `[src_start, src_end]` into zero, one, or multiple
/// retained output intervals on the output timeline `[clip_start_sec, clip_start_sec + output_duration_sec]`.
///
/// **MANDATORY CORRECTION 1**: Preserves discontinuous retained intervals when a source
/// interval crosses one or more Smart Pacing cuts. NEVER collapses discontinuous intervals
/// into a single false continuous span.
pub fn map_source_interval_to_output_intervals(
    src_start: f64,
    src_end: f64,
    pacing_plan: Option<&SmartPacingPlan>,
) -> Vec<(f64, f64)> {
    if src_end <= src_start {
        return Vec::new();
    }

    let plan = match pacing_plan {
        Some(p) if !p.is_noop() => p,
        _ => return vec![(src_start, src_end)],
    };

    let mut intervals = Vec::new();
    const EPS: f64 = 0.001;

    for r in &plan.retained {
        // Check if retained piece overlaps the source interval
        if r.src_end_sec > src_start + EPS && r.src_start_sec < src_end - EPS {
            let overlap_src_start = src_start.max(r.src_start_sec);
            let overlap_src_end = src_end.min(r.src_end_sec);

            if overlap_src_end > overlap_src_start + EPS {
                let out_rel_start = plan
                    .src_to_out(overlap_src_start)
                    .unwrap_or(r.out_start_sec);
                let out_rel_end = plan.src_to_out(overlap_src_end).unwrap_or(r.out_end_sec);

                let abs_out_start = plan.clip_start_sec + out_rel_start;
                let abs_out_end = plan.clip_start_sec + out_rel_end;

                if abs_out_end > abs_out_start {
                    intervals.push((abs_out_start, abs_out_end));
                }
            }
        }
    }

    intervals
}

/// Checks if a time interval `[w_start, w_end]` overlaps any interval in `intervals`.
pub fn overlaps_any_interval(w_start: f64, w_end: f64, intervals: &[(f64, f64)]) -> bool {
    const EPS: f64 = 0.02;
    for &(start, end) in intervals {
        if w_end > start + EPS && w_start < end - EPS {
            return true;
        }
    }
    false
}

/// Checks whether a token is a generic lexical / viral word that CANNOT independently trigger CI.
/// (MANDATORY CORRECTION 3)
pub fn is_supporting_lexical_token(token: &str) -> bool {
    const SUPPORTING_TERMS: &[&str] = &[
        "never",
        "always",
        "worst",
        "best",
        "secret",
        "mistake",
        "truth",
        "problem",
        "key",
        "crucial",
        "essential",
        "proven",
        "simple",
        "insane",
        "crazy",
        "shocking",
        "biggest",
        "million",
        "billion",
        "dollars",
    ];
    SUPPORTING_TERMS.contains(&token) || is_numeric_token(token)
}

/// Checks whether a token represents numbers or metrics (e.g. "10", "100%", "$50k").
pub fn is_numeric_token(s: &str) -> bool {
    s.chars().any(|c| c.is_ascii_digit())
}

/// Main planner for Caption Intelligence 2.0.
///
/// Returns `Some(CaptionIntelPlan)` only when confidence >= 0.75.
/// Otherwise returns `None`, guaranteeing safe fallback to existing caption behavior.
pub fn plan_caption_intel(
    words: &[TranscriptWord],
    candidate: &Candidate,
    pacing_plan: Option<&SmartPacingPlan>,
    start_sec: f64,
    end_sec: f64,
) -> Option<CaptionIntelPlan> {
    // 1. Kill switch check
    if !caption_intelligence_enabled() {
        return None;
    }

    // 2. Empty words check
    if words.is_empty() || end_sec <= start_sec {
        return None;
    }

    // Filter candidate words within the rendered window
    let candidate_word_indices: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.end > start_sec && w.start < end_sec)
        .map(|(i, _)| i)
        .collect();

    if candidate_word_indices.is_empty() {
        return None;
    }

    // 3. Hook-Aware Intelligence (Source -> Output Remapping)
    let mut hook_word_indices = Vec::new();
    let mut hook_reason = String::new();

    if let (Some(h_start), Some(h_end)) = (candidate.hook_start_sec, candidate.hook_end_sec) {
        let h_conf = candidate.hook_confidence.unwrap_or(0.0);
        if h_conf >= 0.70 {
            // Map hook interval across Smart Pacing cuts (preserving discontinuous intervals)
            let out_hook_intervals =
                map_source_interval_to_output_intervals(h_start, h_end, pacing_plan);

            if !out_hook_intervals.is_empty() {
                let hook_tokens: Vec<String> = candidate
                    .hook
                    .split_whitespace()
                    .map(normalize_token)
                    .filter(|t| !t.is_empty())
                    .collect();

                for &idx in &candidate_word_indices {
                    let w = &words[idx];
                    if overlaps_any_interval(w.start, w.end, &out_hook_intervals) {
                        let token = normalize_token(&w.text);
                        // Require token overlap with candidate hook text or strong timing overlap
                        if hook_tokens.contains(&token) || hook_tokens.is_empty() {
                            hook_word_indices.push(idx);
                        }
                    }
                }

                if !hook_word_indices.is_empty() {
                    hook_reason = format!(
                        "Hook verified (conf={:.2}, {} words)",
                        h_conf,
                        hook_word_indices.len()
                    );
                }
            }
        }
    }

    // 4. Payoff-Aware Intelligence (Source -> Output Remapping)
    let mut payoff_word_indices = Vec::new();
    let mut payoff_reason = String::new();

    if let (Some(p_start), Some(p_end), Some(p_score)) = (
        candidate.payoff_start_sec,
        candidate.payoff_end_sec,
        candidate.payoff_score,
    ) {
        // Payoff requires score >= 0.75 and completion not explicitly false
        if p_score >= 0.75 && candidate.payoff_completion != Some(false) {
            let out_payoff_intervals =
                map_source_interval_to_output_intervals(p_start, p_end, pacing_plan);

            if !out_payoff_intervals.is_empty() {
                let payoff_tokens: Vec<String> = candidate
                    .payoff_text
                    .as_deref()
                    .unwrap_or("")
                    .split_whitespace()
                    .map(normalize_token)
                    .filter(|t| !t.is_empty())
                    .collect();

                for &idx in &candidate_word_indices {
                    let w = &words[idx];
                    if overlaps_any_interval(w.start, w.end, &out_payoff_intervals) {
                        let token = normalize_token(&w.text);
                        if payoff_tokens.contains(&token) || payoff_tokens.is_empty() {
                            payoff_word_indices.push(idx);
                        }
                    }
                }

                if !payoff_word_indices.is_empty() {
                    payoff_reason = format!(
                        "Payoff verified (score={:.2}, {} words)",
                        p_score,
                        payoff_word_indices.len()
                    );
                }
            }
        }
    }

    // 5. Supporting Semantic Signals (Numbers, superlatives, contrast terms)
    // MANDATORY CORRECTION 3: Generic lexical words CANNOT independently trigger CI.
    let mut supporting_word_indices = Vec::new();
    for &idx in &candidate_word_indices {
        let token = normalize_token(&words[idx].text);
        if is_supporting_lexical_token(&token) {
            supporting_word_indices.push(idx);
        }
    }

    // 6. Overall Confidence Calculation
    // Hook confidence contribution: up to h_conf when verified
    let hook_contrib = if !hook_word_indices.is_empty() {
        candidate.hook_confidence.unwrap_or(0.0)
    } else {
        0.0
    };

    // Payoff confidence contribution: up to p_score when verified
    let payoff_contrib = if !payoff_word_indices.is_empty() {
        candidate.payoff_score.unwrap_or(0.0)
    } else {
        0.0
    };

    let base_confidence = if hook_contrib > 0.0 && payoff_contrib > 0.0 {
        // Both hook and payoff verified: combined confidence
        (hook_contrib.max(payoff_contrib) + 0.10).min(1.0)
    } else if hook_contrib > 0.0 {
        hook_contrib
    } else if payoff_contrib > 0.0 {
        payoff_contrib
    } else {
        0.0
    };

    // Supporting lexical signals can only add up to +0.05 IF base_confidence > 0.0,
    // and CANNOT independently trigger CI (base_confidence must be > 0.0).
    let supporting_confidence_contrib =
        if base_confidence > 0.0 && !supporting_word_indices.is_empty() {
            0.05f64
        } else {
            0.0f64
        };

    let total_confidence = (base_confidence + supporting_confidence_contrib).clamp(0.0, 1.0);

    // Confidence gating: threshold is 0.75
    const CONFIDENCE_THRESHOLD: f64 = 0.75;
    if total_confidence < CONFIDENCE_THRESHOLD {
        return None;
    }

    // 7. Form Semantic Emphasis Spans (MANDATORY CORRECTION 4)
    // Cluster marked words into contiguous spans.
    // Default rule: prioritize hook and payoff spans, limit density to meaningful semantic units.
    let mut candidate_spans: Vec<EmphasisSpan> = Vec::new();

    // Hook span
    if !hook_word_indices.is_empty() {
        let mut start = hook_word_indices[0];
        let mut prev = start;
        for &idx in &hook_word_indices[1..] {
            if idx == prev + 1 {
                prev = idx;
            } else {
                candidate_spans.push(EmphasisSpan {
                    start_idx: start,
                    end_idx: prev,
                    reason: "Hook phrase".to_string(),
                });
                start = idx;
                prev = idx;
            }
        }
        candidate_spans.push(EmphasisSpan {
            start_idx: start,
            end_idx: prev,
            reason: "Hook phrase".to_string(),
        });
    }

    // Payoff span
    if !payoff_word_indices.is_empty() {
        let mut start = payoff_word_indices[0];
        let mut prev = start;
        for &idx in &payoff_word_indices[1..] {
            if idx == prev + 1 {
                prev = idx;
            } else {
                candidate_spans.push(EmphasisSpan {
                    start_idx: start,
                    end_idx: prev,
                    reason: "Payoff punchline".to_string(),
                });
                start = idx;
                prev = idx;
            }
        }
        candidate_spans.push(EmphasisSpan {
            start_idx: start,
            end_idx: prev,
            reason: "Payoff punchline".to_string(),
        });
    }

    // Supporting spans (only include contiguous supporting words that form a phrase or attach to hook/payoff)
    if !supporting_word_indices.is_empty() {
        let mut i = 0;
        while i < supporting_word_indices.len() {
            let start = supporting_word_indices[i];
            let mut prev = start;
            let mut j = i + 1;
            while j < supporting_word_indices.len() && supporting_word_indices[j] == prev + 1 {
                prev = supporting_word_indices[j];
                j += 1;
            }
            // If it's a numeric or multi-word phrase, include as supporting span
            let span_len = prev - start + 1;
            let is_num = is_numeric_token(&normalize_token(&words[start].text));
            if span_len >= 2 || is_num {
                // Avoid duplicating existing spans
                let already_covered = candidate_spans
                    .iter()
                    .any(|s| s.start_idx <= start && s.end_idx >= prev);
                if !already_covered {
                    candidate_spans.push(EmphasisSpan {
                        start_idx: start,
                        end_idx: prev,
                        reason: if is_num {
                            "Salient metric".to_string()
                        } else {
                            "Salient semantic phrase".to_string()
                        },
                    });
                }
            }
            i = j;
        }
    }

    // Collect all unique word indices from emphasis spans
    let mut emphasis_word_indices = Vec::new();
    for span in &candidate_spans {
        for idx in span.start_idx..=span.end_idx {
            if !emphasis_word_indices.contains(&idx) {
                emphasis_word_indices.push(idx);
            }
        }
    }
    emphasis_word_indices.sort_unstable();

    // 8. Generate Global Line-Break Hints (MANDATORY CORRECTION 2)
    // Global word indices after which a line break is preferred (e.g. at clause punctuation, conjunctions, or span boundaries).
    let conjunctions = [
        "and", "but", "or", "so", "because", "when", "if", "that", "with", "from", "into", "every",
        "on", "your", "my", "our", "their", "this", "which", "aur", "lekin", "par", "ki", "toh",
        "kyunki", "agar", "jab", "tab",
    ];

    let mut line_break_hints = Vec::new();
    for &idx in &candidate_word_indices {
        let w = &words[idx];
        let clean = normalize_token(&w.text);
        let ends_punct = w.text.ends_with(',')
            || w.text.ends_with(';')
            || w.text.ends_with(':')
            || crate::transcription::ends_with_sentence_terminator(&w.text);
        let is_conj = conjunctions.contains(&clean.as_str());

        // Also prefer breaks right after an emphasis span ends
        let ends_span = candidate_spans.iter().any(|s| s.end_idx == idx);

        if ends_punct || ends_span {
            line_break_hints.push(idx);
        } else if is_conj && idx > 0 {
            // Break before conjunction (i.e. after idx - 1)
            line_break_hints.push(idx - 1);
        }
    }
    line_break_hints.sort_unstable();
    line_break_hints.dedup();

    let mut reasons = Vec::new();
    if !hook_reason.is_empty() {
        reasons.push(hook_reason);
    }
    if !payoff_reason.is_empty() {
        reasons.push(payoff_reason);
    }
    if !supporting_word_indices.is_empty() {
        reasons.push(format!(
            "Supporting metrics/phrases ({})",
            supporting_word_indices.len()
        ));
    }

    Some(CaptionIntelPlan {
        confidence: total_confidence,
        emphasis_spans: candidate_spans,
        emphasis_word_indices,
        line_break_hints,
        hook_word_indices,
        payoff_word_indices,
        reason: reasons.join("; "),
    })
}

/// Evaluates global line break hints against a local frame / group.
///
/// **MANDATORY CORRECTION 2**:
/// 1. Intersects global hints with words present in the current group.
/// 2. Converts global indices into 1-based local split points (`split_at`, number of words on line 1).
/// 3. Strictly enforces legal bounds:
///    `min_line1 <= local_split <= max_words_per_line` and `total_words - local_split <= max_words_per_line`.
/// 4. Returns `Some(local_split)` if a legal hint exists, or `None` to preserve standard balanced wrapping.
pub fn select_legal_line_break(
    global_indices_in_group: &[usize],
    line_break_hints: &[usize],
    max_words_per_line: usize,
    max_lines_per_frame: usize,
) -> Option<usize> {
    let total = global_indices_in_group.len();
    if max_lines_per_frame <= 1 || total <= max_words_per_line || total == 0 {
        return None;
    }

    let min_line1 = total.saturating_sub(max_words_per_line).max(1);
    let max_line1 = max_words_per_line.min(total.saturating_sub(1));

    if min_line1 > max_line1 {
        return None;
    }

    let default_target = (total + 1) / 2;

    // Find all legal local split positions derived from hints
    let mut valid_splits = Vec::new();
    for (local_idx, &global_idx) in global_indices_in_group.iter().enumerate() {
        // A hint at global_idx means breaking AFTER word local_idx (so line 1 has local_idx + 1 words)
        if line_break_hints.contains(&global_idx) {
            let split_candidate = local_idx + 1;
            if split_candidate >= min_line1
                && split_candidate <= max_line1
                && (total - split_candidate) <= max_words_per_line
            {
                valid_splits.push(split_candidate);
            }
        }
    }

    // Pick the valid split closest to balanced default
    valid_splits
        .into_iter()
        .min_by_key(|&s| (s as isize - default_target as isize).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pacing::{RetainedInterval, SmartPacingEdit};
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn dummy_word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: None,
        }
    }

    fn dummy_candidate() -> Candidate {
        Candidate {
            id: "cand-1".to_string(),
            project_id: "proj-1".to_string(),
            start_sec: 0.0,
            end_sec: 30.0,
            score: 0.95,
            hook: "Never make this mistake".to_string(),
            rationale: "Rationale".to_string(),
            rank: 1,
            selected: true,
            hook_start_sec: Some(0.0),
            hook_end_sec: Some(4.0),
            hook_confidence: Some(0.92),
            opening_context_score: Some(0.90),
            payoff_text: Some("And that saves your business".to_string()),
            payoff_start_sec: Some(25.0),
            payoff_end_sec: Some(29.5),
            payoff_score: Some(0.94),
            payoff_completion: Some(true),
            metadata_json: None,
        }
    }

    // ── Discontinuous Smart Pacing Remapping Tests (Correction 1) ─────────────

    #[test]
    fn test_map_source_interval_no_pacing() {
        let intervals = map_source_interval_to_output_intervals(10.0, 20.0, None);
        assert_eq!(intervals, vec![(10.0, 20.0)]);
    }

    #[test]
    fn test_map_source_interval_pacing_before_hook() {
        // Cut from 2.0 to 5.0; hook is at 10.0 to 15.0
        let plan = SmartPacingPlan {
            status: "ok".to_string(),
            reason: None,
            clip_start_sec: 0.0,
            clip_end_sec: 30.0,
            output_duration_sec: 27.0,
            removed_total_sec: 3.0,
            edits: vec![SmartPacingEdit {
                edit_type: "pause".to_string(),
                src_start_sec: 2.0,
                src_end_sec: 5.0,
                out_start_sec: 2.0,
                confidence: 1.0,
                reason: String::new(),
            }],
            retained: vec![
                RetainedInterval {
                    src_start_sec: 0.0,
                    src_end_sec: 2.0,
                    out_start_sec: 0.0,
                    out_end_sec: 2.0,
                },
                RetainedInterval {
                    src_start_sec: 5.0,
                    src_end_sec: 30.0,
                    out_start_sec: 2.0,
                    out_end_sec: 27.0,
                },
            ],
        };

        let intervals = map_source_interval_to_output_intervals(10.0, 15.0, Some(&plan));
        assert_eq!(intervals.len(), 1);
        // Source 10.0..15.0 shifted by 3.0s -> 7.0..12.0
        assert!((intervals[0].0 - 7.0).abs() < 0.01);
        assert!((intervals[0].1 - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_map_source_interval_pacing_inside_hook_preserves_discontinuity() {
        // Cut from 14.0 to 16.0 INSIDE hook [10.0, 20.0]
        let plan = SmartPacingPlan {
            status: "ok".to_string(),
            reason: None,
            clip_start_sec: 0.0,
            clip_end_sec: 30.0,
            output_duration_sec: 28.0,
            removed_total_sec: 2.0,
            edits: vec![SmartPacingEdit {
                edit_type: "pause".to_string(),
                src_start_sec: 14.0,
                src_end_sec: 16.0,
                out_start_sec: 14.0,
                confidence: 1.0,
                reason: String::new(),
            }],
            retained: vec![
                RetainedInterval {
                    src_start_sec: 0.0,
                    src_end_sec: 14.0,
                    out_start_sec: 0.0,
                    out_end_sec: 14.0,
                },
                RetainedInterval {
                    src_start_sec: 16.0,
                    src_end_sec: 30.0,
                    out_start_sec: 14.0,
                    out_end_sec: 28.0,
                },
            ],
        };

        let intervals = map_source_interval_to_output_intervals(10.0, 20.0, Some(&plan));
        // MUST produce TWO separate output intervals, NOT a collapsed single one!
        assert_eq!(
            intervals.len(),
            2,
            "Discontinuous cut inside hook must produce 2 intervals"
        );
        assert!((intervals[0].0 - 10.0).abs() < 0.01);
        assert!((intervals[0].1 - 14.0).abs() < 0.01);
        assert!((intervals[1].0 - 14.0).abs() < 0.01);
        assert!((intervals[1].1 - 18.0).abs() < 0.01);
    }

    #[test]
    fn test_map_source_interval_complete_removal() {
        // Cut from 10.0 to 20.0 completely removing the hook [12.0, 18.0]
        let plan = SmartPacingPlan {
            status: "ok".to_string(),
            reason: None,
            clip_start_sec: 0.0,
            clip_end_sec: 30.0,
            output_duration_sec: 20.0,
            removed_total_sec: 10.0,
            edits: vec![SmartPacingEdit {
                edit_type: "pause".to_string(),
                src_start_sec: 10.0,
                src_end_sec: 20.0,
                out_start_sec: 10.0,
                confidence: 1.0,
                reason: String::new(),
            }],
            retained: vec![
                RetainedInterval {
                    src_start_sec: 0.0,
                    src_end_sec: 10.0,
                    out_start_sec: 0.0,
                    out_end_sec: 10.0,
                },
                RetainedInterval {
                    src_start_sec: 20.0,
                    src_end_sec: 30.0,
                    out_start_sec: 10.0,
                    out_end_sec: 20.0,
                },
            ],
        };

        let intervals = map_source_interval_to_output_intervals(12.0, 18.0, Some(&plan));
        assert!(
            intervals.is_empty(),
            "Completely cut interval must return empty intervals"
        );
    }

    // ── Kill Switch Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_kill_switch_env_parsing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
        assert!(caption_intelligence_enabled());

        for off in ["0", "false", "off", "OFF", "False"] {
            std::env::set_var("AUTOSHORTS_CAPTION_INTEL", off);
            assert!(!caption_intelligence_enabled(), "{off} must disable");
        }

        std::env::set_var("AUTOSHORTS_CAPTION_INTEL", "1");
        assert!(caption_intelligence_enabled());
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
    }

    // ── Supporting Lexical Non-Trigger Test (Correction 3) ─────────────────────

    #[test]
    fn test_generic_lexical_word_alone_does_not_trigger_ci() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
        // Candidate with NO hook confidence and NO payoff
        let mut candidate = dummy_candidate();
        candidate.hook_confidence = Some(0.20);
        candidate.payoff_score = None;
        candidate.payoff_text = None;

        // Words contain generic viral words: "problem", "secret", "never"
        let words = vec![
            dummy_word("This", 0.0, 0.5),
            dummy_word("is", 0.5, 0.8),
            dummy_word("the", 0.8, 1.0),
            dummy_word("biggest", 1.0, 1.5),
            dummy_word("secret", 1.5, 2.0),
            dummy_word("problem", 2.0, 2.5),
        ];

        let plan = plan_caption_intel(&words, &candidate, None, 0.0, 5.0);
        assert!(
            plan.is_none(),
            "Generic lexical words alone must NEVER exceed confidence threshold 0.75"
        );
    }

    // ── High Confidence Hook & Payoff Tests ────────────────────────────────────

    #[test]
    fn test_high_confidence_hook_alone_triggers_ci() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
        let mut candidate = dummy_candidate();
        candidate.hook = "Never make this mistake".to_string();
        candidate.hook_confidence = Some(0.92);
        candidate.hook_start_sec = Some(0.0);
        candidate.hook_end_sec = Some(2.0);
        candidate.payoff_score = None;
        candidate.payoff_text = None;

        let words = vec![
            dummy_word("Never", 0.0, 0.5),
            dummy_word("make", 0.5, 1.0),
            dummy_word("this", 1.0, 1.5),
            dummy_word("mistake", 1.5, 2.0),
            dummy_word("today", 2.0, 2.5),
        ];

        let plan = plan_caption_intel(&words, &candidate, None, 0.0, 3.0);
        assert!(plan.is_some(), "Strong hook alone must trigger CI");
        let p = plan.unwrap();
        assert!(p.confidence >= 0.75);
        assert_eq!(p.hook_word_indices, vec![0, 1, 2, 3]);
        assert!(p.payoff_word_indices.is_empty());
    }

    #[test]
    fn test_high_confidence_payoff_alone_triggers_ci() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
        let mut candidate = dummy_candidate();
        candidate.hook_confidence = Some(0.40); // Low confidence hook
        candidate.payoff_score = Some(0.88);
        candidate.payoff_start_sec = Some(1.5);
        candidate.payoff_end_sec = Some(3.0);
        candidate.payoff_text = Some("saves your business".to_string());
        candidate.payoff_completion = Some(true);

        let words = vec![
            dummy_word("Now", 0.0, 0.5),
            dummy_word("this", 0.5, 1.0),
            dummy_word("really", 1.0, 1.5),
            dummy_word("saves", 1.5, 2.0),
            dummy_word("your", 2.0, 2.5),
            dummy_word("business", 2.5, 3.0),
        ];

        let plan = plan_caption_intel(&words, &candidate, None, 0.0, 4.0);
        assert!(plan.is_some(), "Strong payoff alone must trigger CI");
        let p = plan.unwrap();
        assert!(p.confidence >= 0.75);
        assert_eq!(p.payoff_word_indices, vec![3, 4, 5]);
        assert!(p.hook_word_indices.is_empty());
    }

    #[test]
    fn test_low_confidence_hook_no_emphasis() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
        let mut candidate = dummy_candidate();
        candidate.hook_confidence = Some(0.50); // Under 0.70 threshold
        candidate.payoff_score = None;

        let words = vec![
            dummy_word("Never", 0.0, 0.5),
            dummy_word("make", 0.5, 1.0),
            dummy_word("this", 1.0, 1.5),
            dummy_word("mistake", 1.5, 2.0),
        ];

        let plan = plan_caption_intel(&words, &candidate, None, 0.0, 3.0);
        assert!(plan.is_none(), "Low-confidence hook must not trigger CI");
    }

    #[test]
    fn test_empty_words_returns_none() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("AUTOSHORTS_CAPTION_INTEL");
        let candidate = dummy_candidate();
        assert!(plan_caption_intel(&[], &candidate, None, 0.0, 10.0).is_none());
    }

    // ── Legal Group Line-Break Tests (Correction 2) ───────────────────────────

    #[test]
    fn test_select_legal_line_break_respects_hard_limits() {
        let global_indices = vec![10, 11, 12, 13, 14, 15]; // 6 words
        let hints = vec![12]; // break after global word 12 (local word 3 -> 3 words on line 1)

        let split = select_legal_line_break(&global_indices, &hints, 4, 2);
        assert_eq!(
            split,
            Some(3),
            "Hint after 3rd word splits 3 and 3 (both <= 4)"
        );

        // Out of bounds hint: break after global word 10 (1 word on line 1, leaves 5 words on line 2, exceeding max 4)
        let bad_hints = vec![10];
        let bad_split = select_legal_line_break(&global_indices, &bad_hints, 4, 2);
        assert_eq!(
            bad_split, None,
            "Illegal break that leaves 5 words on line 2 must be rejected"
        );
    }
}
