#!/usr/bin/env python3
"""
test_hook_closure_suite.py — Comprehensive Unit & Regression Test Suite for AutoShorts v10.1
Hook Quality Engine, True Conversation Unit Engine, Semantic Closure Engine, and Opening Hook Alignment (R1 – R8.8)

Covers:
- R8.1: Hook Quality Engine (8 unit tests)
- R8.2: Semantic Closure LLM Layer (6 unit tests)
- R8.3: Deterministic Enforcement in snap_to_semantic_boundaries (8 unit tests)
- R8.4: CandidateDraft & ClosureSignals Schema & Deserialization (5 unit tests)
- R8.5: v10.1 True Conversation Unit & Boundary Scenarios (§30 Tests 26–30) (5 unit tests)
- R8.6: v10.1 Six-State Endpoint Classification (Tests 31–37) (7 unit tests)
- R8.7: v10.1 Release-Safe Final Endpoint Validation (Tests 38–40) (3 unit tests)
- R8.8: Hook Alignment, Opening Context, and Boundary Anchoring (Tests 43–47) (5 unit tests)

Total: 47 Comprehensive Tests
"""

import json
import re
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ==============================================================================
# SECTION 1: Core Models & Schemas (Replicating models.rs)
# ==============================================================================

@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


@dataclass
class NormalizedTranscript:
    language: str
    duration: float
    speakers: List[str]
    words: List[TranscriptWord]
    segments: List[TranscriptSegment]


@dataclass
class CandidateDraft:
    # Core fields
    start: float = 0.0
    end: float = 0.0
    score: float = 0.80
    hook: str = ""
    rationale: str = ""

    # Structured timing
    hook_start: Optional[float] = None
    hook_end: Optional[float] = None
    payoff_text: Optional[str] = None
    payoff_start: Optional[float] = None
    payoff_end: Optional[float] = None

    # Narrative classification
    structure: Optional[str] = None
    hook_score: Optional[float] = None
    coherence_score: Optional[float] = None
    payoff_score: Optional[float] = None

    # Adaptive Question-Hook & Answer metadata
    question_hook_used: Optional[bool] = None
    question_start: Optional[float] = None
    question_end: Optional[float] = None
    question_text: Optional[str] = None
    question_hook_score: Optional[float] = None
    question_relevance_score: Optional[float] = None
    answer_start: Optional[float] = None
    answer_end: Optional[float] = None
    answer_text: Optional[str] = None
    answer_strength_score: Optional[float] = None

    # Multilingual & Hook-First metadata
    language: Optional[str] = None
    script: Optional[str] = None
    hook_type: Optional[str] = None
    speakers: Optional[List[str]] = None
    summary: Optional[str] = None

    # Fine-grained scoring dimensions
    curiosity_score: Optional[float] = None
    emotional_impact_score: Optional[float] = None
    surprise_score: Optional[float] = None
    value_score: Optional[float] = None
    story_quality_score: Optional[float] = None
    context_completeness_score: Optional[float] = None
    shareability_score: Optional[float] = None

    # Conversation unit metadata
    hook_speaker: Optional[str] = None
    context_text: Optional[str] = None
    conversation_type: Optional[str] = None
    start_reason: Optional[str] = None
    end_reason: Optional[str] = None
    semantic_complete: Optional[bool] = None
    interviewer_hook_strength: Optional[float] = None
    guest_hook_strength: Optional[float] = None
    answer_completeness: Optional[float] = None
    semantic_closure: Optional[float] = None

    # Multimodal metadata
    multimodal_verified: bool = False
    multimodal_status: Optional[str] = None
    fallback_label: Optional[str] = None
    multimodal_score: Optional[float] = None
    visual_score: Optional[float] = None
    audio_score: Optional[float] = None
    temporal_score: Optional[float] = None
    semantic_score: Optional[float] = None

    # ==========================================
    # v10.x Hook Intelligence Dimensions (R3) - 8 fields
    # ==========================================
    hook_topic_clarity: Optional[float] = None
    hook_curiosity: Optional[float] = None
    hook_relevance: Optional[float] = None
    hook_contrast: Optional[float] = None
    hook_confusion_penalty: Optional[float] = None
    hook_delay_penalty: Optional[float] = None
    hook_irrelevance_penalty: Optional[float] = None
    hook_disinterest_penalty: Optional[float] = None

    # ==========================================
    # v10.x Semantic Closure Dimensions (R3 & R4) - 9 fields
    # ==========================================
    continuation_probability: Optional[float] = None
    breath_pause_risk: Optional[float] = None
    mid_thought_risk: Optional[float] = None
    closure_confidence: Optional[float] = None
    answer_complete: Optional[bool] = None
    story_completeness: Optional[bool] = None
    payoff_completion: Optional[bool] = None
    ending_naturalness: Optional[float] = None
    ending_type: Optional[str] = None

    # ==========================================
    # v10.1 Six-State Endpoint & Validation
    # ==========================================
    endpoint_state: Optional[str] = None
    raw_llm_end: Optional[float] = None

    # ==========================================
    # v10.1 Opening Context & Hook Alignment (R8.8)
    # ==========================================
    hook_confidence: Optional[float] = None
    opening_context_score: Optional[float] = None
    opening_unresolved_reference: Optional[bool] = None
    opening_continuation_marker: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateDraft":
        """Deserialize from snake_case or camelCase dictionary with default fallbacks."""
        def get_field(keys: Sequence[str], default: Any = None) -> Any:
            for k in keys:
                if k in data and data[k] is not None:
                    return data[k]
            return default

        return cls(
            start=float(get_field(["start", "clip_start", "startTime"], 0.0)),
            end=float(get_field(["end", "clip_end", "endTime"], 0.0)),
            score=float(get_field(["score", "overall_score", "overallScore"], 0.8)),
            hook=str(get_field(["hook", "hook_text", "hookText"], "")),
            rationale=str(get_field(["rationale", "reason", "explanation"], "")),
            hook_start=get_field(["hook_start", "hookStart"]),
            hook_end=get_field(["hook_end", "hookEnd"]),
            payoff_text=get_field(["payoff_text", "payoffText"]),
            payoff_start=get_field(["payoff_start", "payoffStart"]),
            payoff_end=get_field(["payoff_end", "payoffEnd"]),
            structure=get_field(["structure", "hook_type", "hookType"]),
            hook_score=get_field(["hook_score", "hookScore", "hook_strength"]),
            coherence_score=get_field(["coherence_score", "coherenceScore"]),
            payoff_score=get_field(["payoff_score", "payoffScore"]),
            question_hook_used=get_field(["question_hook_used", "questionHookUsed"]),
            question_start=get_field(["question_start", "questionStart"]),
            question_end=get_field(["question_end", "questionEnd"]),
            question_text=get_field(["question_text", "questionText"]),
            question_hook_score=get_field(["question_hook_score", "questionHookScore"]),
            question_relevance_score=get_field(["question_relevance_score", "questionRelevanceScore"]),
            answer_start=get_field(["answer_start", "answerStart"]),
            answer_end=get_field(["answer_end", "answerEnd"]),
            answer_text=get_field(["answer_text", "answerText"]),
            answer_strength_score=get_field(["answer_strength_score", "answerStrengthScore"]),
            language=get_field(["language"]),
            script=get_field(["script"]),
            hook_type=get_field(["hook_type", "hookType"]),
            speakers=get_field(["speakers"]),
            summary=get_field(["summary"]),
            curiosity_score=get_field(["curiosity_score", "curiosityScore", "curiosity"]),
            emotional_impact_score=get_field(["emotional_impact_score", "emotionalImpactScore", "emotional_impact"]),
            surprise_score=get_field(["surprise_score", "surpriseScore", "surprise"]),
            value_score=get_field(["value_score", "valueScore", "value"]),
            story_quality_score=get_field(["story_quality_score", "storyQualityScore"]),
            context_completeness_score=get_field(["context_completeness_score", "contextCompletenessScore"]),
            shareability_score=get_field(["shareability_score", "shareabilityScore"]),
            hook_speaker=get_field(["hook_speaker", "hookSpeaker"]),
            context_text=get_field(["context_text", "contextText"]),
            conversation_type=get_field(["conversation_type", "conversationType"]),
            start_reason=get_field(["start_reason", "startReason"]),
            end_reason=get_field(["end_reason", "endReason"]),
            semantic_complete=get_field(["semantic_complete", "semanticComplete"]),
            interviewer_hook_strength=get_field(["interviewer_hook_strength", "interviewerHookStrength"]),
            guest_hook_strength=get_field(["guest_hook_strength", "guestHookStrength"]),
            answer_completeness=get_field(["answer_completeness", "answerCompleteness"]),
            semantic_closure=get_field(["semantic_closure", "semanticClosure"]),
            multimodal_verified=bool(get_field(["multimodal_verified", "multimodalVerified"], False)),
            multimodal_status=get_field(["multimodal_status", "multimodalStatus"]),
            fallback_label=get_field(["fallback_label", "fallbackLabel"]),
            multimodal_score=get_field(["multimodal_score", "multimodalScore"]),
            visual_score=get_field(["visual_score", "visualScore"]),
            audio_score=get_field(["audio_score", "audioScore"]),
            temporal_score=get_field(["temporal_score", "temporalScore"]),
            semantic_score=get_field(["semantic_score", "semanticScore"]),
            # 17 new fields (R3)
            hook_topic_clarity=get_field(["hook_topic_clarity", "hookTopicClarity"]),
            hook_curiosity=get_field(["hook_curiosity", "hookCuriosity"]),
            hook_relevance=get_field(["hook_relevance", "hookRelevance"]),
            hook_contrast=get_field(["hook_contrast", "hookContrast"]),
            hook_confusion_penalty=get_field(["hook_confusion_penalty", "hookConfusionPenalty"]),
            hook_delay_penalty=get_field(["hook_delay_penalty", "hookDelayPenalty"]),
            hook_irrelevance_penalty=get_field(["hook_irrelevance_penalty", "hookIrrelevancePenalty"]),
            hook_disinterest_penalty=get_field(["hook_disinterest_penalty", "hookDisinterestPenalty"]),
            continuation_probability=get_field(["continuation_probability", "continuationProbability"]),
            breath_pause_risk=get_field(["breath_pause_risk", "breathPauseRisk"]),
            mid_thought_risk=get_field(["mid_thought_risk", "midThoughtRisk"]),
            closure_confidence=get_field(["closure_confidence", "closureConfidence"]),
            answer_complete=get_field(["answer_complete", "answerComplete"]),
            story_completeness=get_field(["story_completeness", "storyCompleteness"]),
            payoff_completion=get_field(["payoff_completion", "payoffCompletion"]),
            ending_naturalness=get_field(["ending_naturalness", "endingNaturalness"]),
            ending_type=get_field(["ending_type", "endingType"]),
            endpoint_state=get_field(["endpoint_state", "endpointState", "endpoint_classification", "endpointClassification"]),
            raw_llm_end=get_field(["raw_llm_end", "rawLlmEnd"]),
            hook_confidence=get_field(["hook_confidence", "hookConfidence"]),
            opening_context_score=get_field(["opening_context_score", "openingContextScore"]),
            opening_unresolved_reference=get_field(["opening_unresolved_reference", "openingUnresolvedReference"]),
            opening_continuation_marker=get_field(["opening_continuation_marker", "openingContinuationMarker"]),
        )


@dataclass
class HookPipelineConfig:
    """Configurable thresholds and penalties for hook context evaluation and anchoring."""
    hard_reject_unresolved_pronoun: bool = True
    continuation_penalty: float = 0.04
    min_hook_confidence: float = 0.50
    lead_in_buffer_sec: float = 0.15


@dataclass
class OpeningContextResult:
    """Result of opening 1-3 seconds hook context evaluation."""
    score: float
    penalty: float
    is_standalone: bool
    has_unresolved_reference: bool
    unresolved_reference: Optional[str]
    has_continuation_opener: bool
    matched_marker: Optional[str]
    hard_reject: bool
    explanation: str


@dataclass
class ClosureSignals:
    continuation_probability: Optional[float] = None
    breath_pause_risk: Optional[float] = None
    mid_thought_risk: Optional[float] = None
    answer_complete: Optional[bool] = None
    story_completeness: Optional[bool] = None
    payoff_completion: Optional[bool] = None
    closure_confidence: Optional[float] = None
    ending_type: Optional[str] = None
    answer_end: Optional[float] = None
    payoff_end: Optional[float] = None
    endpoint_state: Optional[str] = None

    @classmethod
    def from_draft(cls, draft: CandidateDraft) -> "ClosureSignals":
        return cls(
            continuation_probability=draft.continuation_probability,
            breath_pause_risk=draft.breath_pause_risk,
            mid_thought_risk=draft.mid_thought_risk,
            answer_complete=draft.answer_complete,
            story_completeness=draft.story_completeness,
            payoff_completion=draft.payoff_completion,
            closure_confidence=draft.closure_confidence,
            ending_type=draft.ending_type,
            answer_end=draft.answer_end,
            payoff_end=draft.payoff_end,
            endpoint_state=draft.endpoint_state,
        )


# ==============================================================================
# SECTION 2: Token & Language Processing (Matching lib.rs & transcription.rs)
# ==============================================================================

CONTINUATION_WORDS = {
    "and", "but", "so", "because", "although", "while", "or", "then",
    "if", "that", "which", "who", "when", "where",
    "with", "from", "at", "as", "by",
    # Hindi / Hinglish continuation tokens
    "कम", "से", "क्योंकि", "लेकिन", "और", "फिर", "इसलिए", "उसके", "बाद", "तब",
    "जब", "अगर", "मतलब", "यानी", "तो", "जिससे", "जिसके", "कारण", "वहां", "उस",
    "टाइम", "मैंने", "हम", "कि", "जैसे", "वैसे", "पर", "भी", "ही", "करके", "लेकर", "देकर",
}


def ends_with_sentence_terminator(text: str) -> bool:
    """Check if text ends with English or Hindi/Devanagari sentence terminator."""
    trimmed = text.strip()
    if not trimmed:
        return False
    return trimmed.endswith(('.', '?', '!', '।', '॥'))


def is_continuation_word(text: str) -> bool:
    """Check if word is an English or Hindi continuation/conjunction token."""
    clean = re.sub(r'^[^\w\u0900-\u097F]+|[^\w\u0900-\u097F]+$', '', text.lower().strip())
    return clean in CONTINUATION_WORDS


def normalize_token(s: str) -> str:
    """Normalize a word token by removing punctuation while preserving Unicode characters."""
    return re.sub(r'[^\w\u0900-\u097F]+', '', s.lower().strip())


def align_hook_to_transcript_words(
    hook_text: str,
    candidate_start: float,
    candidate_end: float,
    words: Sequence[TranscriptWord],
) -> Optional[Tuple[float, float, float, int, int]]:
    """
    Aligns LLM hook text to exact words in transcript (matching llm.rs:align_hook_to_transcript_words).
    Returns (hook_start_sec, hook_end_sec, confidence, start_idx, end_idx).
    """
    if not words:
        return None

    hook_tokens = [normalize_token(w) for w in hook_text.split() if normalize_token(w)]
    if not hook_tokens:
        closest_idx = min(range(len(words)), key=lambda i: abs(words[i].start - candidate_start))
        cw = words[closest_idx]
        return (cw.start, cw.end, 0.40, closest_idx, closest_idx)

    search_window_min = max(0.0, candidate_start - 12.0)
    search_window_max = candidate_start + 25.0
    candidate_word_indices = [
        idx for idx, w in enumerate(words)
        if w.start >= search_window_min and w.start <= search_window_max
    ]
    search_indices = candidate_word_indices if candidate_word_indices else list(range(len(words)))

    search_len = min(len(hook_tokens), 6)
    search_prefix = hook_tokens[:search_len]

    best_match_idx = None
    best_score = 0.0

    for idx in search_indices:
        match_score = 0.0
        for k, target_token in enumerate(search_prefix):
            if idx + k < len(words):
                w_norm = normalize_token(words[idx + k].text)
                if w_norm == target_token:
                    match_score += 1.0
                elif w_norm and (target_token in w_norm or w_norm in target_token):
                    match_score += 0.8
        score = match_score / search_len
        if score > best_score:
            best_score = score
            best_match_idx = idx

    if best_match_idx is not None and best_score >= 0.50:
        start_idx = best_match_idx
        end_idx = min(len(words) - 1, start_idx + len(hook_tokens) - 1)
        return (words[start_idx].start, words[end_idx].end, best_score, start_idx, end_idx)
    else:
        closest_idx = min(range(len(words)), key=lambda i: abs(words[i].start - candidate_start))
        cw = words[closest_idx]
        return (cw.start, cw.end, 0.40, closest_idx, closest_idx)


CONTINUATION_OPENERS = [
    "now here is", "now heres", "at this point", "as i said",
    "the same thing", "whether they", "and this is where", "so then",
    "which is why", "and so", "at that point",
]

THIRD_PERSON_PRONOUNS = {"he", "she", "they", "him", "her", "them"}


def evaluate_opening_hook_context(
    hook_text: str,
    is_question_hook: bool = False,
    config: Optional[HookPipelineConfig] = None,
) -> OpeningContextResult:
    """
    Evaluates opening 1-3 seconds of speech for instant comprehension (matching llm.rs:evaluate_opening_hook_context).
    Detects discourse continuation openers and ungrounded third-person references.
    """
    if config is None:
        config = HookPipelineConfig()
    clean_text = hook_text.strip()
    if not clean_text:
        return OpeningContextResult(
            score=0.0,
            penalty=0.50,
            is_standalone=False,
            has_unresolved_reference=True,
            unresolved_reference="empty_hook",
            has_continuation_opener=False,
            matched_marker=None,
            hard_reject=True,
            explanation="Hook text is empty",
        )

    words = clean_text.split()
    lower_first_few = " ".join(normalize_token(w) for w in words[:6])

    matched_continuation = None
    for opener in CONTINUATION_OPENERS:
        if lower_first_few.startswith(opener):
            matched_continuation = opener
            break

    first_sentence = re.split(r'[.!?।]', clean_text)[0]
    first_sentence_words = first_sentence.split()
    normalized_words = [normalize_token(w) for w in first_sentence_words]

    has_unresolved_3rd_person = False
    unresolved_pronoun = None

    if normalized_words:
        first_word_norm = normalized_words[0]
        if first_word_norm in THIRD_PERSON_PRONOUNS:
            if is_question_hook:
                has_unresolved_3rd_person = False
            else:
                has_proper_noun = any(
                    w and w[0].isupper() and w.isalpha()
                    for w in first_sentence_words[1:]
                )
                has_relative_clause = any(
                    w in ("who", "whom", "whose", "which", "that")
                    for w in normalized_words[1:]
                )
                if not has_proper_noun and not has_relative_clause:
                    has_unresolved_3rd_person = True
                    unresolved_pronoun = first_word_norm

    total_penalty = 0.0
    hard_reject = False
    explanation_parts = []

    if has_unresolved_3rd_person:
        total_penalty += 0.12
        explanation_parts.append(f"ungrounded pronoun '{unresolved_pronoun}'")
        if config.hard_reject_unresolved_pronoun:
            hard_reject = True

    if matched_continuation:
        total_penalty += config.continuation_penalty
        explanation_parts.append(f"continuation marker '{matched_continuation}'")

    final_score = max(0.0, min(1.0, 1.0 - total_penalty))
    is_standalone = not has_unresolved_3rd_person and matched_continuation is None
    explanation = "Clean standalone opening hook" if is_standalone else "; ".join(explanation_parts)

    return OpeningContextResult(
        score=final_score,
        penalty=total_penalty,
        is_standalone=is_standalone,
        has_unresolved_reference=has_unresolved_3rd_person,
        unresolved_reference=unresolved_pronoun,
        has_continuation_opener=matched_continuation is not None,
        matched_marker=matched_continuation,
        hard_reject=hard_reject,
        explanation=explanation,
    )


def closure_requires_extension(cs: Optional[ClosureSignals]) -> bool:
    """Check if closure signals demand extension (matching lib.rs:closure_requires_extension)."""
    if cs is None:
        return False
    cont_prob_trigger = (cs.continuation_probability or 0.0) > 0.70
    breath_risk_trigger = (cs.breath_pause_risk or 0.0) > 0.65
    mid_thought_trigger = (cs.mid_thought_risk or 0.0) > 0.65
    answer_incomplete_trigger = cs.answer_complete is False
    story_incomplete_trigger = cs.story_completeness is False
    payoff_incomplete_trigger = cs.payoff_completion is False
    breath_pause_state_trigger = cs.endpoint_state in ("breath_pause", "breathpause")
    incomplete_state_trigger = False
    if cs.endpoint_state in ("sentence_complete", "sentencecomplete", "thought_complete", "thoughtcomplete"):
        incomplete_state_trigger = (cs.closure_confidence or 0.0) < 0.80

    return (
        cont_prob_trigger
        or breath_risk_trigger
        or mid_thought_trigger
        or answer_incomplete_trigger
        or story_incomplete_trigger
        or payoff_incomplete_trigger
        or breath_pause_state_trigger
        or incomplete_state_trigger
    )


def force_extend_to_closure(
    words: Sequence[TranscriptWord],
    from_sec: float,
    ceiling_sec: float,
) -> Optional[float]:
    """Word-level forward scan up to ceiling_sec for next valid terminator (matching lib.rs)."""
    for idx, w in enumerate(words):
        if w.end > from_sec + 0.1 and w.end <= ceiling_sec:
            if not ends_with_sentence_terminator(w.text):
                continue
            if idx + 1 < len(words):
                next_w = words[idx + 1]
                pause = next_w.start - w.end
                if pause < 0.30 and is_continuation_word(next_w.text):
                    continue
            if not is_continuation_word(w.text):
                return w.end
    return None


def validate_and_enforce_final_endpoint(
    words: Sequence[TranscriptWord],
    raw_llm_end: float,
    snapped_end: float,
    closure: ClosureSignals,
    video_duration: float,
) -> float:
    """Production-safe release behavior enforcement (matching lib.rs:validate_and_enforce_final_endpoint)."""
    extension_required = closure_requires_extension(closure)
    was_extended = abs(snapped_end - raw_llm_end) > 0.05

    if extension_required and not was_extended:
        ceiling = min(video_duration, snapped_end + 35.0)
        salvaged = force_extend_to_closure(words, snapped_end, ceiling)
        if salvaged is not None:
            return salvaged
        raise ValueError(
            f"Candidate with incomplete closure (raw_end={raw_llm_end:.2f}) could not be salvaged "
            f"within safety ceiling ({ceiling:.2f}s). Rejecting to prevent premature render."
        )
    return snapped_end


# ==============================================================================
# SECTION 3: Deterministic Semantic Boundary Snapping (Matching lib.rs)
# ==============================================================================

def snap_to_semantic_boundaries_with_hook_anchor(
    words: Sequence[TranscriptWord],
    raw_start: float,
    raw_end: float,
    video_duration: float,
    closure: Optional[ClosureSignals] = None,
    hook_anchor: Optional[float] = None,
    is_question_hook: bool = False,
) -> Tuple[float, float]:
    """
    Deterministic semantic boundary snapping with hook start anchor protection (matching lib.rs).
    Prevents backward drift across sentence boundaries when a verified hook anchor is present.
    """
    if not words:
        s = max(0.0, min(raw_start, video_duration))
        e = min(video_duration, max(raw_end, s + 1.0))
        return (s, e)

    min_guideline = 12.0
    max_guideline = 75.0

    # 1. Snap start timestamp with strict sentence boundary protection
    chosen_start = max(0.0, raw_start)

    if hook_anchor is not None:
        # Find word closest to anchor
        h_idx, h_word = min(enumerate(words), key=lambda item: abs(item[1].start - hook_anchor))
        if is_question_hook and raw_start < hook_anchor - 1.0:
            q_idx, q_word = min(enumerate(words), key=lambda item: abs(item[1].start - raw_start))
            if q_idx > 0 and ends_with_sentence_terminator(words[q_idx - 1].text):
                chosen_start = max(words[q_idx - 1].end, q_word.start - 0.15)
            else:
                chosen_start = q_word.start
        else:
            # Direct Guest Statement Hook:
            # NEVER cross backward across words[h_idx - 1] if words[h_idx - 1] has a sentence terminator!
            if h_idx > 0 and ends_with_sentence_terminator(words[h_idx - 1].text):
                chosen_start = max(words[h_idx - 1].end, h_word.start - 0.15)
            elif h_idx > 0:
                # Hook begins mid-sentence: find the start of THIS sentence (up to max 1.5s prior)
                sent_start_idx = h_idx
                for k in range(h_idx - 1, -1, -1):
                    if ends_with_sentence_terminator(words[k].text):
                        sent_start_idx = k + 1
                        break
                    if (h_word.start - words[k].start) > 2.0:
                        break
                chosen_start = words[sent_start_idx].start
            else:
                chosen_start = h_word.start
    else:
        # Fallback without hook anchor
        start_candidate_words = [
            (idx, w) for idx, w in enumerate(words)
            if abs(w.start - raw_start) <= 4.5
        ]

        if start_candidate_words:
            sentence_start = None
            for idx, w in start_candidate_words:
                if idx == 0:
                    sentence_start = (idx, w)
                    break
                prev = words[idx - 1]
                if ends_with_sentence_terminator(prev.text):
                    sentence_start = (idx, w)
                    break

            if sentence_start is not None:
                chosen_start = sentence_start[1].start
            else:
                closest = min(start_candidate_words, key=lambda item: abs(item[1].start - raw_start))
                chosen_start = closest[1].start
        else:
            closest_all = min(words, key=lambda w: abs(w.start - raw_start))
            chosen_start = closest_all.start

    # 2. Snap end timestamp with Look-Ahead Semantic Closure Engine
    chosen_end = min(raw_end, video_duration)

    must_extend = closure_requires_extension(closure)
    target_extension_end: Optional[float] = None

    if closure is not None:
        if closure.answer_complete is False and closure.answer_end is not None:
            target_extension_end = closure.answer_end
        elif (closure.payoff_completion is False or closure.story_completeness is False) and closure.payoff_end is not None:
            target_extension_end = closure.payoff_end

    if must_extend:
        search_max = min(chosen_start + max_guideline, video_duration)
        if closure is not None and closure.payoff_end is not None:
            search_max = min(max(chosen_start + max_guideline, closure.payoff_end + 0.5), video_duration)
        target_min_end = target_extension_end if target_extension_end is not None else raw_end

        forward_candidates = [
            (idx, w) for idx, w in enumerate(words)
            if w.end > raw_end + 0.1 and w.end >= target_min_end - 0.1 and w.end <= search_max
        ]

        valid_forward_terminators = []
        for idx, w in forward_candidates:
            if not ends_with_sentence_terminator(w.text):
                continue
            if idx + 1 < len(words):
                next_w = words[idx + 1]
                pause = next_w.start - w.end
                if pause < 0.30 and is_continuation_word(next_w.text):
                    continue
            if not is_continuation_word(w.text):
                valid_forward_terminators.append((idx, w))

        low_closure_confidence = (closure.closure_confidence or 1.0) < 0.50 if closure else False
        if valid_forward_terminators:
            if low_closure_confidence:
                chosen_end = valid_forward_terminators[-1][1].end
            else:
                chosen_end = valid_forward_terminators[0][1].end
        else:
            fallback = [w for w in words if w.end > raw_end and w.end <= search_max and ends_with_sentence_terminator(w.text) and not is_continuation_word(w.text)]
            if fallback:
                chosen_end = fallback[0].end
            else:
                best_in_bounds = [w for w in words if w.end >= chosen_start + min_guideline and w.end <= search_max and ends_with_sentence_terminator(w.text) and not is_continuation_word(w.text)]
                if best_in_bounds:
                    chosen_end = best_in_bounds[-1].end
                elif forward_candidates:
                    chosen_end = max(forward_candidates, key=lambda item: item[1].end)[1].end
    else:
        end_candidate_words = [
            (idx, w) for idx, w in enumerate(words)
            if w.end > chosen_start and (w.end >= raw_end - 2.5 and w.end <= raw_end + 8.5)
        ]

        terminating_candidates = []
        for idx, w in end_candidate_words:
            if not ends_with_sentence_terminator(w.text):
                continue
            if idx + 1 < len(words):
                next_w = words[idx + 1]
                pause = next_w.start - w.end
                if pause < 0.30 and is_continuation_word(next_w.text):
                    continue
            if not is_continuation_word(w.text):
                terminating_candidates.append((idx, w))

        bias_later = False
        if closure is not None and closure.closure_confidence is not None:
            if closure.closure_confidence < 0.50:
                bias_later = True

        if terminating_candidates:
            def sort_key(item: Tuple[int, TranscriptWord]) -> float:
                w = item[1]
                penalty = (4.0 if bias_later else 2.0) if w.end < raw_end else 1.0
                return abs(w.end - raw_end) * penalty

            chosen_end = min(terminating_candidates, key=sort_key)[1].end
        else:
            forward_terminators = [
                w for w in words
                if w.end >= raw_end
                and w.end <= min(raw_end + 8.0, chosen_start + max_guideline)
                and ends_with_sentence_terminator(w.text)
                and not is_continuation_word(w.text)
            ]
            if forward_terminators:
                chosen_end = forward_terminators[0].end
            elif end_candidate_words:
                closest = min(end_candidate_words, key=lambda item: abs(item[1].end - raw_end))
                chosen_end = closest[1].end

    # 3. Look-Ahead Check: If word at chosen_end is a continuation word, extend forward
    end_idx_match = None
    for idx, w in enumerate(words):
        if abs(w.end - chosen_end) < 0.2:
            end_idx_match = idx

    if end_idx_match is not None and end_idx_match + 1 < len(words):
        next_w = words[end_idx_match + 1]
        pause = next_w.start - chosen_end
        if pause < 0.30 and is_continuation_word(next_w.text):
            for fw in words[end_idx_match + 1:]:
                if fw.end - chosen_start > max_guideline:
                    break
                if ends_with_sentence_terminator(fw.text) and not is_continuation_word(fw.text):
                    chosen_end = fw.end
                    break

    # 4. Minimum duration guideline
    dur = chosen_end - chosen_start
    if dur < min_guideline and (video_duration - chosen_start) >= min_guideline:
        for w in words:
            if w.end >= chosen_start + min_guideline and ends_with_sentence_terminator(w.text) and not is_continuation_word(w.text):
                chosen_end = w.end
                break

    # 5. Maximum duration guideline
    dur = chosen_end - chosen_start
    if dur > max_guideline:
        valid_in_range = [
            w for w in words
            if w.end >= chosen_start + 35.0
            and w.end <= chosen_start + max_guideline
            and ends_with_sentence_terminator(w.text)
            and not is_continuation_word(w.text)
        ]
        if valid_in_range:
            chosen_end = valid_in_range[-1].end

    final_start = max(0.0, chosen_start)
    final_end = max(final_start + 1.0, min(chosen_end, video_duration))

    return (final_start, final_end)


def snap_to_semantic_boundaries(
    words: Sequence[TranscriptWord],
    raw_start: float,
    raw_end: float,
    video_duration: float,
    closure: Optional[ClosureSignals] = None,
    hook_anchor: Optional[float] = None,
    is_question_hook: bool = False,
) -> Tuple[float, float]:
    """
    Backward-compatible wrapper for snap_to_semantic_boundaries_with_hook_anchor.
    """
    return snap_to_semantic_boundaries_with_hook_anchor(
        words, raw_start, raw_end, video_duration, closure, hook_anchor, is_question_hook
    )


# ==============================================================================
# SECTION 4: Composite Scoring & Rejection (Matching llm.rs)
# ==============================================================================

def compute_composite_score(c: CandidateDraft) -> float:
    """Matching Rust llm.rs:composite_score formula."""
    base = c.score

    raw_hook = c.hook_score if c.hook_score is not None else (
        c.guest_hook_strength if c.guest_hook_strength is not None else (
            c.interviewer_hook_strength if c.interviewer_hook_strength is not None else base
        )
    )

    clarity = c.hook_topic_clarity if c.hook_topic_clarity is not None else raw_hook
    curiosity = c.hook_curiosity if c.hook_curiosity is not None else (
        c.curiosity_score if c.curiosity_score is not None else raw_hook
    )
    relevance = c.hook_relevance if c.hook_relevance is not None else 0.8
    contrast = c.hook_contrast if c.hook_contrast is not None else 0.7

    positive_hook = min(1.0, max(0.0, clarity * 0.35 + curiosity * 0.35 + relevance * 0.15 + contrast * 0.15))

    delay_pen = c.hook_delay_penalty or 0.0
    conf_pen = c.hook_confusion_penalty or 0.0
    irrel_pen = c.hook_irrelevance_penalty or 0.0
    disint_pen = c.hook_disinterest_penalty or 0.0
    total_hook_penalty = delay_pen * 0.25 + conf_pen * 0.40 + irrel_pen * 0.20 + disint_pen * 0.15
    effective_hook = min(1.0, max(0.0, positive_hook - total_hook_penalty))

    coh = c.coherence_score if c.coherence_score is not None else (
        c.context_completeness_score if c.context_completeness_score is not None else base
    )
    pay = c.payoff_score if c.payoff_score is not None else (
        c.semantic_closure if c.semantic_closure is not None else base
    )
    closure_conf = c.closure_confidence if c.closure_confidence is not None else 0.85
    cont_prob = c.continuation_probability if c.continuation_probability is not None else 0.15
    closure_factor = min(1.0, max(0.0, closure_conf * 0.70 + (1.0 - cont_prob) * 0.30))

    if c.answer_complete is not None:
        ans_comp = 1.0 if c.answer_complete else 0.3
    else:
        ans_comp = c.answer_completeness if c.answer_completeness is not None else coh

    if c.story_completeness is not None:
        story_comp = 1.0 if c.story_completeness else 0.4
    else:
        story_comp = c.story_quality_score if c.story_quality_score is not None else coh

    emo = c.emotional_impact_score if c.emotional_impact_score is not None else 0.7

    is_qa = (c.question_hook_used is True) or (c.hook_speaker == "Host") or (c.conversation_type == "question_answer")

    if is_qa:
        q_strength = c.interviewer_hook_strength if c.interviewer_hook_strength is not None else (
            c.question_relevance_score if c.question_relevance_score is not None else effective_hook
        )
        a_strength = c.answer_strength_score if c.answer_strength_score is not None else pay
        final_score = min(1.0, max(0.0, effective_hook * 0.20 + q_strength * 0.15 + a_strength * 0.20 + coh * 0.15 + pay * 0.10 + ans_comp * 0.10 + closure_factor * 0.10))
    else:
        final_score = min(1.0, max(0.0, effective_hook * 0.30 + coh * 0.15 + pay * 0.15 + ans_comp * 0.10 + story_comp * 0.10 + closure_factor * 0.10 + emo * 0.10))

    # Opening context penalty from v10.1
    if c.opening_context_score is not None and c.opening_context_score < 0.99:
        penalty = (1.0 - c.opening_context_score) * 0.50
        final_score = max(0.0, final_score - penalty)

    if conf_pen > 0.50 or clarity < 0.40:
        return min(final_score, 0.50)

    return final_score


# ==============================================================================
# SECTION 5: Unit & Regression Test Suites
# ==============================================================================

class TestHookQualityEngine(unittest.TestCase):
    """R8.1: Hook Quality Engine Tests (8 Tests)"""

    def test_interviewer_question_hook_high_score(self):
        cand = CandidateDraft(
            hook="What was the single hardest decision of your entire life?",
            hook_speaker="Host",
            hook_type="interviewer_question",
            hook_topic_clarity=0.95,
            hook_curiosity=0.92,
            hook_relevance=0.90,
            hook_confusion_penalty=0.0,
            hook_delay_penalty=0.0,
            answer_complete=True,
            story_completeness=True,
            payoff_completion=True,
            closure_confidence=0.95
        )
        score = compute_composite_score(cand)
        self.assertGreater(score, 0.80)

    def test_guest_insight_hook_self_contained(self):
        cand = CandidateDraft(
            hook="I spent five years in prison for a crime I did not commit.",
            hook_speaker="Guest",
            hook_type="guest_statement",
            hook_topic_clarity=0.92,
            hook_curiosity=0.95,
            hook_relevance=0.88,
            hook_confusion_penalty=0.0,
            hook_delay_penalty=0.0,
            answer_complete=True,
            story_completeness=True,
            payoff_completion=True,
            closure_confidence=0.90
        )
        score = compute_composite_score(cand)
        self.assertGreater(score, 0.80)

    def test_delay_failure_penalized(self):
        clean = CandidateDraft(
            hook="Quantum computing will break all modern encryption.",
            hook_topic_clarity=0.90,
            hook_curiosity=0.85,
            hook_delay_penalty=0.0,
            closure_confidence=0.90
        )
        delayed = CandidateDraft(
            hook="So yeah, like I was saying, you know, quantum computing will break all encryption.",
            hook_topic_clarity=0.60,
            hook_curiosity=0.65,
            hook_delay_penalty=0.80,
            closure_confidence=0.90
        )
        self.assertGreater(compute_composite_score(clean), compute_composite_score(delayed) + 0.10)

    def test_confusion_failure_dangling_pronoun_penalized(self):
        confused = CandidateDraft(
            hook="That was the hardest thing about it when he told me.",
            hook_topic_clarity=0.25,
            hook_curiosity=0.40,
            hook_confusion_penalty=0.85,
            closure_confidence=0.80
        )
        score = compute_composite_score(confused)
        self.assertLessEqual(score, 0.50)

    def test_irrelevance_failure_penalized(self):
        irrelevant = CandidateDraft(
            hook="The weather in Seattle is quite rainy on Tuesday afternoons.",
            hook_topic_clarity=0.80,
            hook_curiosity=0.20,
            hook_irrelevance_penalty=0.75,
            closure_confidence=0.85
        )
        score = compute_composite_score(irrelevant)
        self.assertLess(score, 0.70)

    def test_disinterest_failure_penalized(self):
        flat = CandidateDraft(
            hook="There are several different categories of administrative paperwork.",
            hook_topic_clarity=0.70,
            hook_curiosity=0.15,
            hook_disinterest_penalty=0.80,
            closure_confidence=0.85
        )
        score = compute_composite_score(flat)
        self.assertLess(score, 0.70)

    def test_hindi_interviewer_question_hook_scored(self):
        hindi_cand = CandidateDraft(
            hook="आपने अपनी जिंदगी का सबसे बड़ा जोखिम कब उठाया था?",
            language="hi",
            script="devanagari",
            hook_speaker="Host",
            hook_type="interviewer_question",
            hook_topic_clarity=0.95,
            hook_curiosity=0.92,
            hook_relevance=0.90,
            hook_confusion_penalty=0.0,
            answer_complete=True,
            story_completeness=True,
            payoff_completion=True,
            closure_confidence=0.95
        )
        score = compute_composite_score(hindi_cand)
        self.assertGreater(score, 0.80)

    def test_repaired_hook_beats_broken_hook(self):
        broken = CandidateDraft(
            hook="Because when I woke up, the entire database was gone.",
            hook_confusion_penalty=0.70,
            hook_topic_clarity=0.40,
            closure_confidence=0.70
        )
        repaired = CandidateDraft(
            hook="What was the single worst production outage you ever caused?",
            context_text="What was the single worst production outage you ever caused?",
            hook_confusion_penalty=0.0,
            hook_topic_clarity=0.95,
            hook_curiosity=0.90,
            closure_confidence=0.90
        )
        self.assertGreater(compute_composite_score(repaired), compute_composite_score(broken))


class TestSemanticClosureLLMLayer(unittest.TestCase):
    """R8.2: Semantic Closure LLM Layer Tests (6 Tests)"""

    def test_continuation_probability_high_reduces_score(self):
        complete = CandidateDraft(continuation_probability=0.10, closure_confidence=0.95)
        incomplete = CandidateDraft(continuation_probability=0.85, closure_confidence=0.35)
        self.assertGreater(compute_composite_score(complete), compute_composite_score(incomplete))

    def test_breath_pause_risk_high_reduces_score(self):
        clean = CandidateDraft(breath_pause_risk=0.05, closure_confidence=0.95)
        risky = CandidateDraft(breath_pause_risk=0.80, closure_confidence=0.40)
        self.assertGreater(compute_composite_score(clean), compute_composite_score(risky))

    def test_mid_thought_risk_high_reduces_score(self):
        clean = CandidateDraft(mid_thought_risk=0.05, closure_confidence=0.90)
        cut = CandidateDraft(mid_thought_risk=0.75, closure_confidence=0.40)
        self.assertGreater(compute_composite_score(clean), compute_composite_score(cut))

    def test_answer_completeness_false_penalized(self):
        complete = CandidateDraft(answer_complete=True, closure_confidence=0.90)
        incomplete = CandidateDraft(answer_complete=False, closure_confidence=0.40)
        self.assertGreater(compute_composite_score(complete), compute_composite_score(incomplete))

    def test_story_completeness_false_penalized(self):
        complete = CandidateDraft(story_completeness=True, closure_confidence=0.90)
        incomplete = CandidateDraft(story_completeness=False, closure_confidence=0.40)
        self.assertGreater(compute_composite_score(complete), compute_composite_score(incomplete))

    def test_payoff_completion_false_penalized(self):
        complete = CandidateDraft(payoff_completion=True, closure_confidence=0.90)
        incomplete = CandidateDraft(payoff_completion=False, closure_confidence=0.40)
        self.assertGreater(compute_composite_score(complete), compute_composite_score(incomplete))


class TestDeterministicSemanticBoundaries(unittest.TestCase):
    """R8.3: Deterministic Boundary Snapping Tests (8 Tests)"""

    def test_snap_high_continuation_probability_extends(self):
        words = [
            TranscriptWord("First", 0.0, 0.5),
            TranscriptWord("part.", 0.5, 15.0),
            TranscriptWord("And", 15.5, 16.0),
            TranscriptWord("second", 16.0, 16.5),
            TranscriptWord("part.", 16.5, 25.0),
        ]
        closure = ClosureSignals(continuation_probability=0.85)
        _, snapped_end = snap_to_semantic_boundaries(words, 0.0, 15.0, 40.0, closure)
        self.assertGreater(snapped_end, 15.0)
        self.assertEqual(snapped_end, 25.0)

    def test_snap_breath_pause_risk_extends(self):
        words = [
            TranscriptWord("The", 0.0, 0.5),
            TranscriptWord("speaker", 0.5, 10.0),
            TranscriptWord("pauses", 10.0, 12.0),
            TranscriptWord("and", 12.5, 13.0),
            TranscriptWord("finishes", 13.0, 13.5),
            TranscriptWord("now.", 13.5, 18.0),
        ]
        closure = ClosureSignals(breath_pause_risk=0.80)
        _, snapped_end = snap_to_semantic_boundaries(words, 0.0, 12.0, 30.0, closure)
        self.assertGreater(snapped_end, 12.0)
        self.assertEqual(snapped_end, 18.0)

    def test_snap_mid_thought_risk_extends(self):
        words = [
            TranscriptWord("He", 10.0, 10.5),
            TranscriptWord("said", 10.5, 18.0),
            TranscriptWord("because", 18.0, 18.5),
            TranscriptWord("everyone", 18.5, 19.3),
            TranscriptWord("listens.", 19.3, 21.0),
        ]
        closure = ClosureSignals(mid_thought_risk=0.70)
        _, snapped_end = snap_to_semantic_boundaries(words, 10.0, 18.5, 30.0, closure)
        self.assertGreater(snapped_end, 18.5)
        self.assertEqual(snapped_end, 21.0)

    def test_snap_answer_incomplete_extends(self):
        words = [
            TranscriptWord("Host", 0.0, 0.5),
            TranscriptWord("asked.", 0.5, 1.0),
            TranscriptWord("Guest", 1.5, 2.0),
            TranscriptWord("talking", 2.0, 20.0),
            TranscriptWord("and", 20.5, 21.0),
            TranscriptWord("finally", 25.0, 26.0),
            TranscriptWord("concluded.", 26.0, 32.0),
        ]
        closure = ClosureSignals(answer_complete=False, answer_end=32.0)
        _, snapped_end = snap_to_semantic_boundaries(words, 0.0, 20.0, 50.0, closure)
        self.assertGreaterEqual(snapped_end, 32.0)

    def test_snap_payoff_incomplete_extends(self):
        words = [
            TranscriptWord("Setup", 10.0, 11.0),
            TranscriptWord("development", 11.0, 28.0),
            TranscriptWord("and", 28.5, 29.0),
            TranscriptWord("the", 35.0, 36.0),
            TranscriptWord("payoff.", 36.0, 45.0),
        ]
        closure = ClosureSignals(payoff_completion=False, payoff_end=45.0)
        _, snapped_end = snap_to_semantic_boundaries(words, 10.0, 28.0, 60.0, closure)
        self.assertGreaterEqual(snapped_end, 45.0)

    def test_snap_none_closure_backwards_compatible(self):
        words = [
            TranscriptWord("Hello", 0.0, 0.5),
            TranscriptWord("world.", 0.5, 1.0),
            TranscriptWord("This", 1.2, 1.6),
            TranscriptWord("is", 1.6, 1.9),
            TranscriptWord("great.", 1.9, 15.0),
        ]
        snapped_start, snapped_end = snap_to_semantic_boundaries(words, 0.0, 15.0, 30.0, None)
        self.assertEqual(snapped_start, 0.0)
        self.assertEqual(snapped_end, 15.0)

    def test_snap_high_continuation_timestamp_differs_from_raw(self):
        words = [
            TranscriptWord("First", 10.0, 10.5),
            TranscriptWord("part", 10.5, 25.0),
            TranscriptWord("second", 25.2, 25.8),
            TranscriptWord("conclusion.", 25.8, 28.0),
        ]
        raw_end = 25.0
        closure = ClosureSignals(continuation_probability=0.90)
        _, snapped_end = snap_to_semantic_boundaries(words, 10.0, raw_end, 40.0, closure)
        self.assertNotEqual(snapped_end, raw_end)
        self.assertEqual(snapped_end, 28.0)

    def test_snap_existing_rust_baselines_preserved(self):
        words_45s = [
            TranscriptWord("What", 0.0, 0.5),
            TranscriptWord("is", 0.5, 0.8),
            TranscriptWord("the", 0.8, 1.0),
            TranscriptWord("biggest", 1.0, 1.5),
            TranscriptWord("lesson?", 1.5, 2.0),
            TranscriptWord("The", 2.5, 3.0),
            TranscriptWord("biggest", 3.0, 3.5),
            TranscriptWord("lesson", 3.5, 4.0),
            TranscriptWord("is", 4.0, 4.3),
            TranscriptWord("consistency.", 4.3, 5.0),
            TranscriptWord("That's", 14.0, 14.5),
            TranscriptWord("the", 14.5, 14.8),
            TranscriptWord("truth.", 14.8, 44.5),
        ]
        s1, e1 = snap_to_semantic_boundaries(words_45s, 0.0, 44.5, 60.0)
        self.assertEqual(s1, 0.0)
        self.assertEqual(e1, 44.5)
        self.assertGreater(e1 - s1, 40.0)


class TestCandidateDraftSchema(unittest.TestCase):
    """R8.4: Schema & Deserialization Tests (5 Tests)"""

    def test_candidate_draft_all_fields_present(self):
        json_data = {
            "start": 12.5,
            "end": 45.0,
            "score": 0.95,
            "hook": "Why do 90% of tech startups fail within 12 months?",
            "rationale": "High-curiosity hook with structured answer and narrative payoff.",
            "hookTopicClarity": 0.98,
            "hookCuriosity": 0.94,
            "hookRelevance": 0.90,
            "hookContrast": 0.85,
            "hookConfusionPenalty": 0.0,
            "hookDelayPenalty": 0.0,
            "hookIrrelevancePenalty": 0.0,
            "hookDisinterestPenalty": 0.0,
            "continuationProbability": 0.12,
            "breathPauseRisk": 0.08,
            "midThoughtRisk": 0.05,
            "closureConfidence": 0.93,
            "answerComplete": True,
            "storyCompleteness": True,
            "payoffCompletion": True,
            "endingNaturalness": 0.90,
            "endingType": "answer_completion",
            "endpointState": "answer_complete"
        }

        cand = CandidateDraft.from_dict(json_data)
        self.assertEqual(cand.hook_topic_clarity, 0.98)
        self.assertEqual(cand.endpoint_state, "answer_complete")
        self.assertEqual(cand.continuation_probability, 0.12)

    def test_candidate_draft_legacy_json_deserializes(self):
        legacy_data = {
            "start": 10.0,
            "end": 35.0,
            "score": 0.82,
            "hook": "Legacy hook without new fields.",
            "rationale": "Legacy rationale."
        }
        cand = CandidateDraft.from_dict(legacy_data)
        self.assertEqual(cand.start, 10.0)
        self.assertIsNone(cand.endpoint_state)
        self.assertIsNone(cand.raw_llm_end)

    def test_closure_signals_from_draft_mapping(self):
        draft = CandidateDraft(
            start=20.0,
            end=45.0,
            continuation_probability=0.78,
            breath_pause_risk=0.60,
            answer_complete=False,
            answer_end=48.0,
            closure_confidence=0.45,
            ending_type="mid_answer",
            endpoint_state="sentence_complete"
        )
        signals = ClosureSignals.from_draft(draft)
        self.assertEqual(signals.continuation_probability, 0.78)
        self.assertEqual(signals.endpoint_state, "sentence_complete")

    def test_endpoint_state_all_seven_variants(self):
        variants = [
            "breath_pause", "sentence_complete", "thought_complete",
            "answer_complete", "story_complete", "payoff_complete", "unknown"
        ]
        for v in variants:
            cand = CandidateDraft.from_dict({"endpointState": v})
            self.assertEqual(cand.endpoint_state, v)

    def test_raw_llm_end_preservation(self):
        cand = CandidateDraft(start=10.0, end=25.0, raw_llm_end=25.0)
        self.assertEqual(cand.raw_llm_end, 25.0)


class TestConversationUnitAndBoundaryScenarios(unittest.TestCase):
    """R8.5: v10.1 True Conversation Unit & Boundary Scenarios (§30 Tests 26–30)"""

    def test_26_genuine_life_lesson_closure(self):
        """Test 26: Genuine life lesson endpoint terminates naturally without forced extension."""
        words = [
            TranscriptWord("And", 0.0, 0.5),
            TranscriptWord("that", 0.5, 1.0),
            TranscriptWord("taught", 1.0, 1.5),
            TranscriptWord("me", 1.5, 1.8),
            TranscriptWord("that", 1.8, 2.0),
            TranscriptWord("patience", 2.0, 2.8),
            TranscriptWord("is", 2.8, 3.0),
            TranscriptWord("everything.", 3.0, 18.0),
            TranscriptWord("Moving", 18.5, 19.0),
            TranscriptWord("to", 19.0, 19.5),
            TranscriptWord("the", 19.5, 20.0),
            TranscriptWord("next", 20.0, 20.5),
            TranscriptWord("question.", 20.5, 25.0),
        ]
        closure = ClosureSignals(
            ending_type="life_lesson",
            payoff_completion=True,
            closure_confidence=0.95,
            continuation_probability=0.08,
            endpoint_state="payoff_complete"
        )
        s, e = snap_to_semantic_boundaries(words, 0.0, 18.0, 30.0, closure)
        self.assertEqual(s, 0.0)
        self.assertEqual(e, 18.0, "Life lesson with high closure confidence must cleanly terminate at 18.0s")

    def test_27_unrelated_new_topic_terminates_naturally(self):
        """Test 27: Unrelated new topic following resolved thought does not cause over-extension."""
        words = [
            TranscriptWord("I", 0.0, 0.4),
            TranscriptWord("never", 0.4, 0.8),
            TranscriptWord("looked", 0.8, 1.2),
            TranscriptWord("back.", 1.2, 16.0),
            TranscriptWord("By", 16.5, 17.0),
            TranscriptWord("the", 17.0, 17.3),
            TranscriptWord("way,", 17.3, 18.0),
            TranscriptWord("did", 18.0, 18.3),
            TranscriptWord("you", 18.3, 18.6),
            TranscriptWord("see", 18.6, 19.0),
            TranscriptWord("the", 19.0, 19.3),
            TranscriptWord("game?", 19.3, 22.0),
        ]
        closure = ClosureSignals(
            ending_type="unrelated_topic",
            answer_complete=True,
            closure_confidence=0.90,
            endpoint_state="answer_complete"
        )
        s, e = snap_to_semantic_boundaries(words, 0.0, 16.0, 30.0, closure)
        self.assertEqual(e, 16.0, "Boundary engine must not drag in unrelated next topic")

    def test_28_promotional_insertion_terminates_before_cta(self):
        """Test 28: Promotional insertion cleanly terminates before CTA begins."""
        words = [
            TranscriptWord("That", 0.0, 0.5),
            TranscriptWord("is", 0.5, 0.8),
            TranscriptWord("the", 0.8, 1.0),
            TranscriptWord("secret.", 1.0, 20.0),
            TranscriptWord("Before", 20.5, 21.0),
            TranscriptWord("we", 21.0, 21.3),
            TranscriptWord("continue,", 21.3, 22.0),
            TranscriptWord("sponsor", 22.0, 22.8),
            TranscriptWord("shoutout.", 22.8, 30.0),
        ]
        closure = ClosureSignals(
            answer_complete=True,
            closure_confidence=0.92,
            endpoint_state="answer_complete"
        )
        s, e = snap_to_semantic_boundaries(words, 0.0, 20.0, 35.0, closure)
        self.assertEqual(e, 20.0, "Must terminate before sponsor/promo insertion")

    def test_29_speaker_turn_after_completed_thought(self):
        """Test 29: Speaker turn after complete thought constitutes valid endpoint."""
        words = [
            TranscriptWord("Host", 0.0, 0.5),
            TranscriptWord("Q?", 0.5, 1.0),
            TranscriptWord("Guest", 1.2, 1.6),
            TranscriptWord("Answer", 1.6, 2.0),
            TranscriptWord("complete.", 2.0, 15.0),
            TranscriptWord("Host", 15.5, 16.0),
            TranscriptWord("Totally", 16.0, 16.5),
            TranscriptWord("agree.", 16.5, 17.5),
        ]
        closure = ClosureSignals(
            ending_type="speaker_turn",
            answer_complete=True,
            closure_confidence=0.88,
            endpoint_state="answer_complete"
        )
        s, e = snap_to_semantic_boundaries(words, 0.0, 15.0, 25.0, closure)
        self.assertEqual(e, 15.0)

    def test_30_long_complete_answer_over_50s_accepted(self):
        """Test 30: Long complete answer > 50s accepted without arbitrary truncation."""
        words = [
            TranscriptWord("Why", 0.0, 0.5),
            TranscriptWord("start?", 0.5, 1.0),
            TranscriptWord("Deep", 1.5, 2.0),
            TranscriptWord("thought", 2.0, 25.0),
            TranscriptWord("development", 25.0, 45.0),
            TranscriptWord("final", 45.0, 50.0),
            TranscriptWord("resolution.", 50.0, 54.0),
        ]
        closure = ClosureSignals(
            answer_complete=True,
            story_completeness=True,
            payoff_completion=True,
            closure_confidence=0.95,
            endpoint_state="story_complete"
        )
        s, e = snap_to_semantic_boundaries(words, 0.0, 54.0, 75.0, closure)
        self.assertEqual(s, 0.0)
        self.assertEqual(e, 54.0)
        self.assertGreater(e - s, 50.0, "Long self-contained answer must not be truncated")


class TestSixStateEndpointClassification(unittest.TestCase):
    """R8.6: v10.1 Six-State Endpoint Classification Tests (Tests 31–37)"""

    def test_31_breath_pause_forces_extension(self):
        words = [
            TranscriptWord("Start", 0.0, 1.0),
            TranscriptWord("midway", 1.0, 15.0),
            TranscriptWord("and", 15.5, 16.0),
            TranscriptWord("concluded.", 16.0, 24.0),
        ]
        closure = ClosureSignals(endpoint_state="breath_pause")
        s, e = snap_to_semantic_boundaries(words, 0.0, 15.0, 30.0, closure)
        self.assertEqual(e, 24.0, "BreathPause endpoint state must force extension")

    def test_32_sentence_complete_low_confidence_forces_extension(self):
        words = [
            TranscriptWord("First", 0.0, 1.0),
            TranscriptWord("clause.", 1.0, 14.0),
            TranscriptWord("Second", 14.5, 15.0),
            TranscriptWord("resolution.", 15.0, 22.0),
        ]
        closure = ClosureSignals(endpoint_state="sentence_complete", closure_confidence=0.55)
        s, e = snap_to_semantic_boundaries(words, 0.0, 14.0, 30.0, closure)
        self.assertEqual(e, 22.0, "SentenceComplete with low confidence (<0.80) must extend")

    def test_33_sentence_complete_high_confidence_accepted(self):
        words = [
            TranscriptWord("Self", 0.0, 1.0),
            TranscriptWord("contained.", 1.0, 14.0),
            TranscriptWord("Next", 14.5, 15.0),
            TranscriptWord("topic.", 15.0, 22.0),
        ]
        closure = ClosureSignals(endpoint_state="sentence_complete", closure_confidence=0.88)
        s, e = snap_to_semantic_boundaries(words, 0.0, 14.0, 30.0, closure)
        self.assertEqual(e, 14.0, "SentenceComplete with high confidence (>=0.80) accepted")

    def test_34_thought_complete_low_confidence_forces_extension(self):
        words = [
            TranscriptWord("Setup", 0.0, 1.0),
            TranscriptWord("thought.", 1.0, 15.0),
            TranscriptWord("Payoff", 15.5, 16.0),
            TranscriptWord("delivered.", 16.0, 25.0),
        ]
        closure = ClosureSignals(endpoint_state="thought_complete", closure_confidence=0.60)
        s, e = snap_to_semantic_boundaries(words, 0.0, 15.0, 35.0, closure)
        self.assertEqual(e, 25.0, "ThoughtComplete with low confidence must extend to payoff")

    def test_35_thought_complete_high_confidence_accepted(self):
        words = [
            TranscriptWord("Complete", 0.0, 1.0),
            TranscriptWord("point.", 1.0, 15.0),
            TranscriptWord("New", 15.5, 16.0),
            TranscriptWord("discussion.", 16.0, 25.0),
        ]
        closure = ClosureSignals(endpoint_state="thought_complete", closure_confidence=0.90)
        s, e = snap_to_semantic_boundaries(words, 0.0, 15.0, 35.0, closure)
        self.assertEqual(e, 15.0)

    def test_36_answer_complete_accepted(self):
        words = [
            TranscriptWord("Full", 0.0, 1.0),
            TranscriptWord("answer.", 1.0, 20.0),
            TranscriptWord("Next", 20.5, 21.0),
            TranscriptWord("question.", 21.0, 30.0),
        ]
        closure = ClosureSignals(endpoint_state="answer_complete", closure_confidence=0.95)
        s, e = snap_to_semantic_boundaries(words, 0.0, 20.0, 40.0, closure)
        self.assertEqual(e, 20.0)

    def test_37_payoff_complete_accepted(self):
        words = [
            TranscriptWord("Setup", 0.0, 1.0),
            TranscriptWord("punchline!", 1.0, 18.0),
            TranscriptWord("Laughter", 18.5, 19.0),
            TranscriptWord("subsides.", 19.0, 25.0),
        ]
        closure = ClosureSignals(endpoint_state="payoff_complete", closure_confidence=0.96)
        s, e = snap_to_semantic_boundaries(words, 0.0, 18.0, 30.0, closure)
        self.assertEqual(e, 18.0)


class TestReleaseSafeEndpointValidation(unittest.TestCase):
    """R8.7: v10.1 Release-Safe Final Endpoint Validation Tests (Tests 38–40)"""

    def test_38_validation_success_when_already_extended(self):
        words = [
            TranscriptWord("First", 0.0, 1.0),
            TranscriptWord("part.", 1.0, 10.0),
            TranscriptWord("Second", 10.5, 11.0),
            TranscriptWord("part.", 11.0, 20.0),
        ]
        closure = ClosureSignals(continuation_probability=0.85)
        final_end = validate_and_enforce_final_endpoint(words, 10.0, 20.0, closure, 30.0)
        self.assertEqual(final_end, 20.0)

    def test_39_validation_salvages_incomplete_when_not_extended(self):
        words = [
            TranscriptWord("First", 0.0, 1.0),
            TranscriptWord("part.", 1.0, 10.0),
            TranscriptWord("Second", 10.5, 11.0),
            TranscriptWord("part.", 11.0, 20.0),
        ]
        closure = ClosureSignals(continuation_probability=0.85)
        final_end = validate_and_enforce_final_endpoint(words, 10.0, 10.0, closure, 30.0)
        self.assertEqual(final_end, 20.0)

    def test_40_validation_rejects_unsalvageable_candidate(self):
        words = [
            TranscriptWord("First", 0.0, 1.0),
            TranscriptWord("part.", 1.0, 10.0),
            TranscriptWord("and", 10.5, 11.0),
            TranscriptWord("dangling", 11.0, 12.0),
        ]
        closure = ClosureSignals(continuation_probability=0.85)
        with self.assertRaises(ValueError) as ctx:
            validate_and_enforce_final_endpoint(words, 10.0, 10.0, closure, 15.0)
        self.assertIn("Rejecting to prevent premature render", str(ctx.exception))


class TestHookPipelineFixes(unittest.TestCase):
    """R8.8: Hook Alignment, Opening Context, and Boundary Anchoring (Tests 43–47)"""

    def test_43_clip1_hook_anchor_prevents_backward_drift(self):
        """Test 43: Clip 1 regression - Hook at 7522.78s must not drift backward to 7519.98s."""
        words = [
            TranscriptWord("She", 7519.98, 7520.22),
            TranscriptWord("looks", 7520.22, 7520.38),
            TranscriptWord("at", 7520.38, 7520.54),
            TranscriptWord("him", 7520.54, 7520.70),
            TranscriptWord("and", 7520.70, 7520.78),
            TranscriptWord("says", 7520.78, 7520.94),
            TranscriptWord("exactly", 7520.94, 7521.50),
            TranscriptWord("the", 7521.50, 7521.66),
            TranscriptWord("same", 7521.66, 7521.90),
            TranscriptWord("thing.", 7521.90, 7522.38),
            TranscriptWord("And", 7522.78, 7523.34),
            TranscriptWord("this", 7523.50, 7523.74),
            TranscriptWord("is", 7523.74, 7523.98),
            TranscriptWord("where", 7523.98, 7524.46),
            TranscriptWord("Krishna", 7524.46, 7524.94),
            TranscriptWord("teaches", 7524.94, 7525.34),
            TranscriptWord("us.", 7525.34, 7525.50),
        ]
        hook_text = "And this is where Krishna teaches us"
        align_res = align_hook_to_transcript_words(hook_text, 7522.0, 7580.0, words)
        self.assertIsNotNone(align_res)
        h_start, h_end, conf, s_idx, _ = align_res
        self.assertEqual(s_idx, 10)
        self.assertAlmostEqual(h_start, 7522.78, places=2)
        self.assertGreaterEqual(conf, 0.90)

        # Without hook anchor, legacy snap demonstrated backward drift to 7519.98s
        old_s, _ = snap_to_semantic_boundaries(words, 7522.0, 7525.5, 7600.0, None, hook_anchor=None)
        self.assertAlmostEqual(old_s, 7519.98, places=2, msg="Old snap demonstrates the backward drift bug")

        # With hook anchor, snap MUST anchor at or after words[9].end (7522.38s)
        s, e = snap_to_semantic_boundaries(words, 7522.0, 7525.5, 7600.0, None, hook_anchor=h_start)
        self.assertGreaterEqual(s, 7522.0, "Start must not drift backward into preceding sentence")
        self.assertAlmostEqual(s, 7522.63, delta=0.3)

    def test_44_clip2_continuation_marker_soft_penalty_not_hard_reject(self):
        """Test 44: Clip 2 regression - Continuation opener receives soft penalty (-0.04), not hard reject."""
        config = HookPipelineConfig()
        hook = "Now here is where the mystery is."
        result = evaluate_opening_hook_context(hook, is_question_hook=False, config=config)
        self.assertTrue(result.has_continuation_opener)
        self.assertAlmostEqual(result.penalty, 0.04, places=3)
        self.assertFalse(result.hard_reject, "Continuation marker must NOT be hard rejected")
        self.assertAlmostEqual(result.score, 0.96, places=2)

    def test_45_clip3_ungrounded_pronoun_hard_rejected_unless_host_question(self):
        """Test 45: Clip 3 regression - Ungrounded pronoun without antecedent is hard-rejected; host setup resolves it."""
        config = HookPipelineConfig()
        hook = "She looks at him and says exactly the same thing."

        # Case A: Isolated guest statement -> hard reject
        result_isolated = evaluate_opening_hook_context(hook, is_question_hook=False, config=config)
        self.assertTrue(result_isolated.has_unresolved_reference)
        self.assertTrue(result_isolated.hard_reject, "Ungrounded 3rd-person pronoun must be hard-rejected")
        self.assertEqual(result_isolated.unresolved_reference, "she")

        # Case B: Host question setup -> antecedent provided, valid!
        result_qa = evaluate_opening_hook_context(hook, is_question_hook=True, config=config)
        self.assertFalse(result_qa.has_unresolved_reference)
        self.assertFalse(result_qa.hard_reject, "Host question setup resolves antecedent")

    def test_46_clip4_hindi_hook_anchor_boundary_protection(self):
        """Test 46: Clip 4 regression - Hindi hook at 906.78s does not drift to 903.99s."""
        words = [
            TranscriptWord("तो", 903.00, 903.30),
            TranscriptWord("ये", 903.30, 903.60),
            TranscriptWord("हुआ।", 903.60, 903.99),
            TranscriptWord("कमसा", 906.78, 907.30),
            TranscriptWord("gets", 907.30, 907.80),
            TranscriptWord("मुक्ति।", 907.80, 908.50),
        ]
        hook_text = "कमसा gets मुक्ति"
        align_res = align_hook_to_transcript_words(hook_text, 906.0, 909.0, words)
        self.assertIsNotNone(align_res)
        h_start, _, conf, s_idx, _ = align_res
        self.assertEqual(s_idx, 3)
        self.assertAlmostEqual(h_start, 906.78, places=2)

        # Boundary snapping with hook anchor
        s, e = snap_to_semantic_boundaries(words, 906.0, 908.5, 950.0, None, hook_anchor=h_start)
        self.assertGreaterEqual(s, 906.0, "Hindi hook must not drift backward into preceding Hindi sentence")
        self.assertAlmostEqual(s, 906.63, delta=0.3)

    def test_47_valid_hooks_corpus_invariants(self):
        """Test 47: Valid conversational hooks across corpus receive 0 penalty and no rejection."""
        config = HookPipelineConfig()
        valid_hooks = [
            "How many people in the world do you completely trust?",
            "I have 4,000 plus active clients.",
            "We built this company from nothing.",
            "You cannot imagine what happened next.",
            "These are the three c's you must know.",
            "The same Kamsa who had his sister imprisoned realized his fate.",
            "It turns out that consistency is everything.",
        ]
        for hook in valid_hooks:
            res = evaluate_opening_hook_context(hook, is_question_hook=False, config=config)
            self.assertFalse(res.hard_reject, f"Valid hook '{hook}' should not be rejected")
            self.assertEqual(res.penalty, 0.0, f"Valid hook '{hook}' should receive zero penalty")
            self.assertTrue(res.is_standalone, f"Valid hook '{hook}' should be standalone")
            self.assertEqual(res.score, 1.0)


# ==============================================================================
# SECTION 6: Runner
# ==============================================================================

if __name__ == "__main__":
    unittest.main()
