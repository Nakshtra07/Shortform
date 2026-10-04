use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::models::{
    CandidateDraft, DiscoveryMode, HookPipelineConfig, NormalizedTranscript, OpeningContextResult,
    TranscriptSegment, TranscriptWord, WindowDiscoveryConfig, WindowScoreResult,
};

// ─── Grounding guard (appended to every prompt) ──────────────────────────────

const GROUNDING_INSTRUCTION: &str = "\
IMPORTANT GROUNDING RULES:
- Use ONLY timestamps and content present in the supplied transcript.
- Every clip_start and clip_end MUST correspond to real timestamps in the transcript.
- Do NOT invent dialogue, claims, facts, or people not present in the transcript.
- Do NOT fabricate hook text or payoff text — quote or closely paraphrase actual transcript text.
- If the transcript contains no strong hook-payoff moments, return {\"candidates\":[]} rather than fabricating candidates.";

// ─── Shared system / user prompt ─────────────────────────────────────────────

/// Build the single shared semantic-selection prompt used by every provider.
/// Upgraded in v10.x with Hook Quality Engine (R1), Semantic Closure Engine (R2),
/// Multilingual (Hindi/Hinglish) Support (R7), and 17 Structured Scoring Dimensions.
fn build_semantic_prompt_core(
    transcript_text: &str,
    lang_category: &str,
    script_type: &str,
    timeline_header: Option<&str>,
    candidate_instruction: &str,
) -> String {
    let header_block = timeline_header.unwrap_or("");
    format!(
        r#"{header_block}You are a master podcast short-form editor and narrative strategist. Your objective is to discover and extract the highest-performing viral standalone short-form clip candidates from the podcast transcript below.

CRITICAL PRINCIPLE — AutoShorts is a SEMANTIC CONVERSATION EDITOR, NOT a random sentence extractor.
Do NOT score isolated sentences out of context.
Always identify the complete narrative unit:
  HOOK → CONTEXT → DEVELOPMENT → PAYOFF / RESOLUTION

==================================================
1. SEGMENT-FIRST HOOK DISCOVERY & REPAIR WORKFLOW (R1)
==================================================
Follow this 4-step workflow strictly:
Step 1: SEGMENT THE TRANSCRIPT into coherent topical conversation units.
Step 2: IDENTIFY QUESTION/ANSWER RELATIONSHIPS & NARRATIVE STRUCTURE (Hook → Context → Development → Payoff).
Step 3: EVALUATE THE OPENING 1-3 SECONDS of each unit against the Topic Clarity + Curiosity Gate and 4 Failure Mechanisms:
        1. DELAY FAILURE: Generic intro, conversational filler ("so yeah", "like I was saying", "you know"), or stalling before getting to the topic. (Penalize with hookDelayPenalty)
        2. CONFUSION FAILURE: Unexplained pronouns ("he told me", "that was it"), dangling references, starting mid-sentence or mid-answer, or missing prior context. (Penalize with hookConfusionPenalty or reject)
        3. IRRELEVANCE FAILURE: No discernible reason why the viewer should care, lacks stakes or relevance. (Penalize with hookIrrelevancePenalty or reject)
        4. DISINTEREST FAILURE: Flat delivery, no curiosity loop, no question, no contrast, no surprise. (Penalize with hookDisinterestPenalty)
        * RULE: NEVER treat an isolated interesting sentence as an effective short-form hook if it lacks immediate topic clarity or requires missing prior context.
Step 4: APPLY HOOK REPAIR STRATEGY if a promising segment has a weak opening:
        - Move start BACKWARD to the containing interviewer question that establishes the topic and context.
        - Move start BACKWARD to the sentence that introduces the topic context.
        - Move start FORWARD past weak filler words to a strong standalone opening statement.
        - REJECT the candidate if no valid self-contained opening exists.
        * STRICT INVARIANT: NEVER fabricate narration, reorder words, or paraphrase the speaker.

==================================================
2. INTERVIEWER QUESTIONS AS FIRST-CLASS HOOKS & SPEAKER-TURN GRAPH
==================================================
In podcast conversations, structure follows a local speaker-turn graph:
HOST → QUESTION → GUEST → ANSWER → [GUEST → ELABORATION]* → GUEST → RESOLUTION
or:
HOST → QUESTION → GUEST → STORY → HOST REACTION → GUEST PUNCHLINE / LESSON

- The local conversation arc determines BOTH the required start and end of the clip!
- Start: Choose the earliest turn that establishes topic clarity + curiosity (prefer interviewer question when guest answer lacks immediate self-contained context).
- End: The clip MUST NOT terminate mid-arc. Continue through development until the resolution or payoff turn is delivered.
- An interviewer (Host) question establishes immediate topic clarity, context, and curiosity.
- A strong host question followed by a resolved guest answer MUST be prioritized over an isolated guest statement that requires missing prior context.
- If a guest answer is profound but begins with "Because...", "That's why...", "So when...", "उस वजह से...", "तो मैंने...", start the clip at the preceding Interviewer Question.
- Examples of strong interviewer hooks:
  - "What was the hardest decision of your life?"
  - "Why did you walk away when you were at the top?"
  - "What is the single biggest mistake first-time founders make?"
  - "आपने अपनी जिंदगी का सबसे कठिन फैसला कब लिया?"
  - "क्या आपको कभी लगा कि सब कुछ खत्म हो गया?"
  STRUCTURE: INTERVIEWER QUESTION → GUEST ANSWER → EXPLANATION / STORY → PAYOFF / LESSON

==================================================
3. GUEST STATEMENT HOOKS & CONTEXT SUFFICIENCY
==================================================
A direct revelation, contrarian insight, life lesson, shocking fact, emotional turning point, joke setup, or philosophical quote from the guest.
- Example: "I fight against my own mind every single morning."
- MANDATORY CONTEXT TEST: "Would a viewer who has never heard this podcast understand what this sentence means without prior context?"
- If NO, you MUST move the clip start earlier to include the relevant host question or setup context!
- FORBID DANGLING PRONOUNS / UNEXPLAINED REFERENCES:
  Never start a clip on an ambiguous reference where the viewer cannot understand what is being referenced:
  - "That was the hardest thing..." (What was?)
  - "He told me that I had failed..." (Who is he?)
  - "That's why I left..." (Why?)
  - "It changed everything..." (What did?)
  - "उस वजह से मैंने छोड़ दिया..."
  - "उसने मुझसे ऐसा कहा..."
  In all such cases, start at the preceding Interviewer Question or setup sentence that provides the necessary context!

==================================================
4. SEMANTIC THOUGHT COMPLETION, SIX-STATE ENDPOINTS & 8-POINT CLOSURE CRITERIA (R2)
==================================================
- NEVER terminate a clip merely because it reached 20s, 30s, or 45s.
- The clip MUST continue until the complete thought, answer, story resolution, lesson, or punchline is fully delivered (allowed dynamic range: 15s to 65s+).
- NEVER cut the speaker off mid-sentence, mid-thought, mid-answer, or mid-story! Breath/micro-pause alone is NEVER closure.
- Look-ahead MUST extend until answer/story/punchline/lesson resolves, or an unrelated new topical unit begins.

SIX ENDPOINT STATES (classify every candidate into exactly one):
1. breath_pause      — Short pause/breath only; same speaker will continue. NEVER a valid endpoint.
2. sentence_complete — Grammatically complete sentence, but thought/answer continues. Valid ONLY if genuinely self-contained.
3. thought_complete  — Complete thought, but part of a larger developing arc. Valid ONLY if self-contained.
4. answer_complete   — The triggering interviewer question is fully answered and resolved. VALID termination.
5. story_complete    — Story arc (setup → conflict → resolution) is complete. VALID termination.
6. payoff_complete   — Punchline, life lesson, or emotional payoff has landed. VALID termination.

* RULE: Only states 4, 5, or 6 (answer_complete, story_complete, payoff_complete) should normally terminate a conversational clip!

VALID CLOSURE REQUIRES ALL 8 CONDITIONS:
1. Grammatical completion of the final sentence.
2. Semantic completion of the current thought.
3. Narrative completion (Q&A arc, story arc, or lesson arc resolved).
4. No unresolved continuation dependency ("because...", "and so...", "aur fir...").
5. No obvious imminent payoff within look-ahead window.
6. Natural pause OR speaker turn.
7. No dangling conjunction in the final spoken word.
8. No unfinished list or causal chain.

==================================================
5. DETECTED TRANSCRIPT LANGUAGE & SCRIPT INTEGRITY (R7)
==================================================
- Primary Classification: {lang_category}
- Dominant Script: {script_type}

LANGUAGE & SCRIPT INTEGRITY (MANDATORY INVARIANTS):
- Analyze Hindi and Hinglish conversations with equal semantic depth as English (गहरा सवाल, जीवन का सबक, कहानी और संघर्ष, अप्रत्याशित मोड़, मज़ाकिया प्रसंग और पंचलाइन).
- Preserve the exact spoken script returned in the transcript.
- If the transcript contains Hindi in Devanagari (e.g. "आपको पता है"), KEEP IT IN DEVANAGARI.
- If the transcript contains Roman Hindi (e.g. "Aapko pata hai"), KEEP IT IN ROMAN SCRIPT.
- If the speaker says English (e.g. "this is actually a huge mistake"), KEEP IT IN ENGLISH LATIN SCRIPT.
- NEVER translate Hindi into English or English into Hindi in the output text.
- NEVER invent, transliterate, or rewrite spoken dialogue. All quotes MUST be verbatim from transcript.
- Internal English translation is permitted solely for reasoning; final JSON fields ("hookText", "contextText", "payoffText") MUST be verbatim transcript text.

==================================================
6. HARD REJECTION RULES (R6)
==================================================
REJECT any candidate if:
1. Opening lacks topic clarity within 1-3s (hookTopicClarity < 0.35).
2. Opening contains unexplained pronoun or dangling reference without compensating hook strength (hookConfusionPenalty > 0.60).
3. Hook is vague suspense with no identifiable subject.
4. Hook starts mid-answer without sufficient prior context.
5. The guest is still answering the question or the story is unresolved when the clip ends (continuationProbability > 0.85 without extension).
6. The final sentence is cut off mid-explanation or ends on a dangling conjunction ("because...", "aur fir...").
7. Candidate is marked semantically incomplete (semanticComplete == false).

==================================================
7. JSON RETURN FORMAT
==================================================
Respond with ONLY valid JSON (no markdown formatting, no commentary outside JSON):
{{
  "candidates": [
    {{
      "start": <number>,
      "end": <number>,
      "hookStart": <number>,
      "hookEnd": <number>,
      "speakers": ["Host", "Guest"],
      "hookSpeaker": "Host | Guest",
      "hookType": "interviewer_question | guest_statement | story_opening | revelation | lesson | punchline",
      "conversationType": "question_answer | question_answer_followup | story | claim_explanation | problem_experience_lesson | setup_joke_punchline | revelation_emotional_payoff | life_lesson_insight",
      "language": "hi | en | hi-en | multi",
      "script": "devanagari | latin | mixed | roman_hindi",
      "hookText": "<opening hook verbatim from transcript>",
      "contextText": "<introductory question or context verbatim from transcript, or empty string>",
      "summary": "<1-sentence summary of the complete narrative>",
      "startReason": "<why the clip begins here with sufficient context>",
      "endReason": "<why the thought/answer/story is semantically complete here>",
      "payoffText": "<final concluding sentence delivering payoff verbatim from transcript>",
      "semanticComplete": true,

      "hookTopicClarity": <0.0-1.0>,
      "hookCuriosity": <0.0-1.0>,
      "hookRelevance": <0.0-1.0>,
      "hookContrast": <0.0-1.0>,
      "hookConfusionPenalty": <0.0-1.0>,
      "hookDelayPenalty": <0.0-1.0>,
      "hookIrrelevancePenalty": <0.0-1.0>,
      "hookDisinterestPenalty": <0.0-1.0>,

      "continuationProbability": <0.0-1.0>,
      "breathPauseRisk": <0.0-1.0>,
      "midThoughtRisk": <0.0-1.0>,
      "closureConfidence": <0.0-1.0>,
      "answerComplete": true | false,
      "storyCompleteness": true | false,
      "payoffCompletion": true | false,
      "endingNaturalness": <0.0-1.0>,
      "endingType": "sentence_completion | answer_completion | story_resolution | punchline | life_lesson | speaker_turn | unrelated_topic",
      "endpointState": "breath_pause | sentence_complete | thought_complete | answer_complete | story_complete | payoff_complete",

      "scores": {{
        "interviewerHookStrength": <0.0-1.0>,
        "guestHookStrength": <0.0-1.0>,
        "hookStrength": <0.0-1.0>,
        "contextCompleteness": <0.0-1.0>,
        "narrativeCompleteness": <0.0-1.0>,
        "answerCompleteness": <0.0-1.0>,
        "payoffStrength": <0.0-1.0>,
        "semanticClosure": <0.0-1.0>,
        "curiosity": <0.0-1.0>,
        "emotionalImpact": <0.0-1.0>,
        "shareability": <0.0-1.0>,
        "score": <0.0-1.0>
      }}
    }}
  ]
}}

{candidate_instruction}

{GROUNDING_INSTRUCTION}

TRANSCRIPT:
{transcript_text}"#,
        header_block = header_block,
        lang_category = lang_category,
        script_type = script_type,
        candidate_instruction = candidate_instruction,
        GROUNDING_INSTRUCTION = GROUNDING_INSTRUCTION,
        transcript_text = transcript_text,
    )
}

/// Build the single shared semantic-selection prompt used by every provider.
/// Upgraded in v10.x with Hook Quality Engine (R1), Semantic Closure Engine (R2),
/// Multilingual (Hindi/Hinglish) Support (R7), and 17 Structured Scoring Dimensions.
pub fn build_semantic_prompt(
    transcript_text: &str,
    lang_category: &str,
    script_type: &str,
) -> String {
    let instruction = "Generate 6 to 10 top viral candidates with a healthy balance of Interviewer Question Hooks and Guest Insight Hooks.\nRank them by overall score descending.";
    build_semantic_prompt_core(
        transcript_text,
        lang_category,
        script_type,
        None,
        instruction,
    )
}

/// Build a window-specific semantic prompt with absolute timestamp grounding and bounded candidate count.
pub fn build_window_semantic_prompt(
    transcript_text: &str,
    lang_category: &str,
    script_type: &str,
    window_start: f64,
    window_end: f64,
    window_idx: usize,
    total_windows: usize,
    target_count_min: usize,
    target_count_max: usize,
) -> String {
    let header = format!(
        "==================================================\n\
         TIMELINE COVERAGE CONTEXT (WINDOW {cur} OF {total})\n\
         ==================================================\n\
         - This transcript section covers Window {cur} of {total}: [{window_start:.1}s to {window_end:.1}s] ({dur:.1}s duration).\n\
         - CRITICAL GROUNDING INVARIANT: All timestamps in this transcript are ABSOLUTE seconds from the beginning (00:00) of the full video.\n\
         - Every candidate start, end, hookStart, and hookEnd MUST use these exact absolute timestamps verbatim. Do NOT reset timestamps to 00:00.\n\n",
        cur = window_idx + 1,
        total = total_windows,
        window_start = window_start,
        window_end = window_end,
        dur = window_end - window_start,
    );

    let instruction = format!(
        "Generate {target_count_min} to {target_count_max} top viral candidates from this timeline section.\n\
         Select ONLY genuinely strong, engaging moments with clear hooks and payoffs.\n\
         If this section contains fewer than {target_count_min} strong moments, return only the strong moments or return {{\"candidates\":[]}} rather than fabricating weak filler.\n\
         Rank them by overall score descending.",
        target_count_min = target_count_min,
        target_count_max = target_count_max,
    );

    build_semantic_prompt_core(
        transcript_text,
        lang_category,
        script_type,
        Some(&header),
        &instruction,
    )
}

/// Build a REZE-style window scoring prompt that rates the viral highlight potential of a transcript window.
///
/// CRITICAL INVARIANT: This prompt strictly forbids timestamp generation (`start`, `end`, etc.).
/// All temporal boundaries and clip extents are computed deterministically by the downstream Rust engine.
pub fn build_window_scoring_prompt(
    transcript_text: &str,
    lang_category: &str,
    script_type: &str,
    window_start: f64,
    window_end: f64,
    window_idx: usize,
    total_windows: usize,
) -> String {
    let cur = window_idx + 1;
    let dur = (window_end - window_start).max(0.0);

    let lang_upper = lang_category.to_uppercase();
    let script_lower = script_type.to_lowercase();
    let is_hindi = lang_upper.contains("HINDI")
        || script_lower.contains("devanagari")
        || script_lower.contains("roman_hindi");

    let lang_integrity_block = if is_hindi {
        "LANGUAGE & SCRIPT INTEGRITY (MANDATORY INVARIANTS):\n\
         - Analyze Hindi and Hinglish conversations with equal semantic depth as English (गहरा सवाल, जीवन का सबक, कहानी और संघर्ष, अप्रत्याशित मोड़, मज़ाकिया प्रसंग और पंचलाइन).\n\
         - Preserve the exact spoken script returned in the transcript without translation.\n\
         - Evaluate cultural resonance, colloquial punchlines, and emotional intensity accurately.\n\n"
    } else {
        ""
    };

    format!(
        r#"==================================================
REZE WINDOW SCORING TASK (WINDOW {cur} OF {total})
==================================================
- Timeline Section: [{window_start:.1}s to {window_end:.1}s] ({dur:.1}s duration).
- Primary Classification: {lang_category}
- Dominant Script: {script_type}

You are an expert viral highlight evaluator.
Assess whether this transcript segment contains a viral, high-impact highlight moment suitable for a standalone short-form clip.

Rate highlight potential from 0.0 to 1.0 (highlight_score):
- 0.0 - 0.3: Mundane conversational filler, administrative talk, flat delivery, or lack of standalone interest.
- 0.4 - 0.6: Moderately interesting conversation or answer, but lacks an explosive hook or decisive payoff.
- 0.7 - 0.8: Strong insight, engaging story, clear emotional or intellectual hook, and solid payoff.
- 0.9 - 1.0: Exceptional, explosive viral moment with an immediate grab, compelling narrative, and unforgettable payoff.

Score the following sub-dimensions (each 0.0 - 1.0):
- hook_relevance: Does this segment contain or open with a clear, intriguing question, shocking statement, or compelling hook?
- narrative_completeness: Does this segment convey a coherent thought, story, or idea rather than fragmented words?
- payoff_presence: Does this segment contain a satisfying resolution, punchline, revelation, or actionable lesson?
- engagement_signal: How strong is the emotional resonance, curiosity loop, or conversational intensity?

Provide a brief reasoning (one concise sentence).

{lang_integrity_block}==================================================
STRICT GROUNDING & SCORING CONTRACT:
- DO NOT emit timestamps, clip boundaries, or time codes.
- DO NOT emit fields named start, end, hookStart, hookEnd, payoffStart, payoffEnd, clip_start, or clip_end.
- The Rust engine computes all boundaries deterministically. Your sole job is quality and relevance scoring.
- Output ONLY a single JSON object. No markdown preamble, no commentary.

SCHEMA:
```json
{{
  "highlight_score": <0.0-1.0>,
  "hook_relevance": <0.0-1.0>,
  "narrative_completeness": <0.0-1.0>,
  "payoff_presence": <0.0-1.0>,
  "engagement_signal": <0.0-1.0>,
  "reasoning": "<short sentence>"
}}
```

TRANSCRIPT SEGMENT:
{transcript_text}
"#,
        cur = cur,
        total = total_windows,
        window_start = window_start,
        window_end = window_end,
        dur = dur,
        lang_category = lang_category,
        script_type = script_type,
        lang_integrity_block = lang_integrity_block,
        transcript_text = transcript_text,
    )
}

/// Builds the prompt for evaluating an already-extracted candidate clip as a finished standalone short.
/// Strictly respects the no-timestamps contract.
pub fn build_candidate_evaluation_prompt(
    transcript_text: &str,
    lang_category: &str,
    script_type: &str,
    clip_start: f64,
    clip_end: f64,
) -> String {
    let lang_upper = lang_category.to_uppercase();
    let script_lower = script_type.to_lowercase();
    let is_hindi = lang_upper.contains("HINDI")
        || script_lower.contains("devanagari")
        || script_lower.contains("roman_hindi");

    let lang_integrity_block = if is_hindi {
        "LANGUAGE & SCRIPT INTEGRITY (MANDATORY INVARIANTS):\n\
         - Analyze Hindi and Hinglish conversations with equal semantic depth as English (गहरा सवाल, जीवन का सबक, कहानी और संघर्ष, अप्रत्याशित मोड़, मज़ाकिया प्रसंग और पंचलाइन).\n\
         - Preserve the exact spoken script returned in the transcript without translation.\n\
         - Evaluate cultural resonance, colloquial punchlines, and emotional intensity accurately.\n\n"
    } else {
        ""
    };

    let dur = (clip_end - clip_start).max(0.0);

    format!(
        r#"==================================================
CANDIDATE CLIP EVALUATION TASK
==================================================
- Clip Span: [{clip_start:.1}s to {clip_end:.1}s] ({dur:.1}s duration).
- Primary Classification: {lang_category}
- Dominant Script: {script_type}

You are evaluating a finished short-form clip. Does it work as a STANDALONE short?
Do the first 3 seconds grab a viewer? Does it land a satisfying payoff, or does it cut off mid-thought? Rate clip quality 0.0-1.0. Classify the ending.

Rate the candidate clip according to these dimensions:
- clip_score: 0.0-1.0 standalone quality — the ranking signal
- hook_first_3s: 0.0-1.0 whether the first 3 seconds immediately grab a viewer
- payoff_lands: 0.0-1.0 whether the clip delivers a satisfying conclusion or punchline
- standalone_works: boolean (true if this clip is completely self-contained and makes sense on its own)
- endpoint_state: one of: SentenceComplete | ThoughtComplete | AnswerComplete | AnswerPartial | BreathPause | MidThought
- mid_thought_risk: 0.0-1.0 risk that this clip cuts off in the middle of an incomplete thought
- answer_complete: boolean (false if the clip ends mid-answer; for non-QA clips, mirror standalone_works)
- story_completeness: boolean (true if the narrative or idea in the clip reaches full resolution)
- reasoning: string (ONE concise sentence explaining the score and ending classification)

{lang_integrity_block}==================================================
STRICT GROUNDING & SCORING CONTRACT:
- DO NOT emit timestamps, clip boundaries, or time codes.
- DO NOT emit fields named start, end, hookStart, hookEnd, payoffStart, payoffEnd, clip_start, or clip_end.
- Candidate boundaries are already fixed. Your sole job is evaluating standalone quality and closure.
- Output ONLY a single JSON object. No markdown preamble, no commentary.

SCHEMA:
```json
{{
  "clip_score": <0.0-1.0>,
  "hook_first_3s": <0.0-1.0>,
  "payoff_lands": <0.0-1.0>,
  "standalone_works": <true|false>,
  "endpoint_state": "<SentenceComplete|ThoughtComplete|AnswerComplete|AnswerPartial|BreathPause|MidThought>",
  "mid_thought_risk": <0.0-1.0>,
  "answer_complete": <true|false>,
  "story_completeness": <true|false>,
  "reasoning": "<one sentence>"
}}
```

TRANSCRIPT CLIP:
{transcript_text}
"#,
        clip_start = clip_start,
        clip_end = clip_end,
        dur = dur,
        lang_category = lang_category,
        script_type = script_type,
        lang_integrity_block = lang_integrity_block,
        transcript_text = transcript_text,
    )
}

// ─── Transcript formatting ────────────────────────────────────────────────────

/// Format transcript within an optional time range [t_start, t_end]
pub fn format_transcript_window(
    segments: &[TranscriptSegment],
    words: &[TranscriptWord],
    t_start: f64,
    t_end: f64,
) -> String {
    let filtered_words: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| w.end >= t_start && w.start <= t_end)
        .take(8000)
        .collect();

    let word_section: String = if !filtered_words.is_empty() {
        let word_lines: Vec<String> = filtered_words
            .iter()
            .map(|w| {
                let speaker = w.speaker.as_deref().unwrap_or("Speaker");
                format!("  [{:.2}–{:.2}] {}: {}", w.start, w.end, speaker, w.text)
            })
            .collect();

        format!(
            "=== WORD-LEVEL TIMESTAMPS (use for precise clip_start / clip_end) ===\n{}\n\n",
            word_lines.join("\n")
        )
    } else {
        String::new()
    };

    let segment_lines: Vec<String> = segments
        .iter()
        .filter(|s| s.end >= t_start && s.start <= t_end)
        .map(|seg| {
            let speaker = seg.speaker.as_deref().unwrap_or("Speaker");
            format!(
                "[{:.2}–{:.2}] {}: {}",
                seg.start, seg.end, speaker, seg.text
            )
        })
        .collect();

    let segment_section = format!("=== SEGMENTS ===\n{}", segment_lines.join("\n"));

    format!("{}{}", word_section, segment_section)
}

/// Format the entire transcript for LLM consumption.
pub fn format_transcript_for_llm(
    segments: &[TranscriptSegment],
    words: &[TranscriptWord],
) -> String {
    format_transcript_window(segments, words, 0.0, f64::MAX)
}

// ─── Response structs for each provider ──────────────────────────────────────

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeepseekMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeepseekChoice {
    message: DeepseekMessage,
}

#[derive(Debug, Deserialize)]
struct DeepseekResponse {
    choices: Vec<DeepseekChoice>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
}

#[derive(Debug, Serialize)]
struct ClaudeMessage<'a> {
    role: &'a str,
    content: String,
}

// ─── Provider implementations & Full-Timeline Discovery Engine ──────────────

/// Total budget for a single provider HTTP request.
///
/// `reqwest::Client::new()` has NO request timeout by default: a socket that
/// accepts the connection and then stalls (server-side hang, proxy that holds
/// the response open, dropped route with no RST) blocks `send().await` for as
/// long as the OS keeps the connection alive. Candidate discovery calls these
/// providers on the render thread, so that stall presented as a frozen app
/// with no stage identifiable. Every provider call goes through this client.
const PROVIDER_REQUEST_TIMEOUT_SEC: u64 = 180;
const PROVIDER_CONNECT_TIMEOUT_SEC: u64 = 30;

/// Build a reqwest client with a hard total timeout and a bounded connect
/// timeout. Connection reuse is disabled so a pooled keep-alive connection
/// cannot hand a request to a peer that has already gone away.
fn provider_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(PROVIDER_REQUEST_TIMEOUT_SEC))
        .connect_timeout(std::time::Duration::from_secs(PROVIDER_CONNECT_TIMEOUT_SEC))
        .pool_idle_timeout(Some(std::time::Duration::from_secs(30)))
        .build()
        .context("building provider HTTP client")
}

/// Generic raw provider query dispatcher.
/// Executes a prompt against the requested LLM provider without retry.
pub async fn query_provider_prompt_raw(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
    min_duration_secs: f64,
) -> Result<Vec<CandidateDraft>> {
    match provider {
        "claude" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("ANTHROPIC_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "claude-3-5-sonnet-latest".to_string());

            println!(
                "[Claude] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 0.15,
                    "messages": [
                        ClaudeMessage { role: "user", content: prompt.to_string() }
                    ]
                }))
                .send()
                .await
                .context("calling Claude")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Claude request failed ({status}): {body}"));
            }

            let message: AnthropicMessage =
                response.json().await.context("parsing Claude response")?;
            let text = message
                .content
                .into_iter()
                .find_map(|content| content.text)
                .ok_or_else(|| anyhow!("Claude response did not include text content"))?;

            println!("[Claude] Received response (length: {} chars)", text.len());
            parse_candidate_json(&text, min_duration_secs)
        }
        "gemini" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("GEMINI_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "gemini-2.5-flash".to_string());
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );

            println!(
                "[Gemini] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post(&url)
                .json(&json!({
                    "contents": [{ "parts": [{ "text": prompt }] }],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.15,
                        "maxOutputTokens": 8192
                    }
                }))
                .send()
                .await
                .context("calling Gemini")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Gemini request failed ({status}): {body}"));
            }

            let res_body: GeminiResponse =
                response.json().await.context("parsing Gemini response")?;
            let text = res_body
                .candidates
                .first()
                .and_then(|c| c.content.parts.first())
                .and_then(|p| p.text.clone())
                .ok_or_else(|| anyhow!("Gemini response did not include content text"))?;

            println!("[Gemini] Received response (length: {} chars)", text.len());
            parse_candidate_json(&text, min_duration_secs)
        }
        "openai" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OPENAI_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "gpt-4o-mini".to_string());

            println!(
                "[OpenAI] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling OpenAI")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("OpenAI request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse =
                response.json().await.context("parsing OpenAI response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("OpenAI response did not include choices content"))?;

            println!("[OpenAI] Received response (length: {} chars)", text.len());
            parse_candidate_json(&text, min_duration_secs)
        }
        "openrouter" => {
            let default_model = "google/gemini-2.5-flash".to_string();
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OPENROUTER_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or(default_model);

            println!(
                "[OpenRouter] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://openrouter.ai/api/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling OpenRouter")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("OpenRouter request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse = response
                .json()
                .await
                .context("parsing OpenRouter response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("OpenRouter response did not include choices content"))?;

            println!(
                "[OpenRouter] Received response (length: {} chars)",
                text.len()
            );
            parse_candidate_json(&text, min_duration_secs)
        }
        "groq" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("GROQ_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "llama-3.3-70b-versatile".to_string());

            println!(
                "[Groq] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling Groq")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Groq request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse =
                response.json().await.context("parsing Groq response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("Groq response did not include choices content"))?;

            println!("[Groq] Received response (length: {} chars)", text.len());
            parse_candidate_json(&text, min_duration_secs)
        }
        "local" | "ollama" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OLLAMA_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "llama3.2".to_string());

            let system = "You are a professional podcast-to-shorts editor. \
                You identify strong narrative arcs (hook → story → payoff) in podcast transcripts. \
                You always return valid JSON with the exact schema requested. \
                You never truncate clips mid-sentence. \
                You only include candidates with a clear payoff and standalone coherence.";

            println!(
                "[Ollama] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("http://localhost:11434/api/chat")
                .json(&json!({
                    "model": model,
                    "messages": [
                        { "role": "system", "content": system },
                        { "role": "user",   "content": prompt }
                    ],
                    "stream": false,
                    "options": { "temperature": 0.15, "num_predict": 4096 },
                    "format": "json"
                }))
                .send()
                .await
                .context("calling local Ollama")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Local Ollama request failed ({status}): {body}"));
            }

            let res_body: OllamaResponse = response
                .json()
                .await
                .context("parsing local Ollama response")?;
            println!(
                "[Ollama] Received response (length: {} chars)",
                res_body.message.content.len()
            );
            parse_candidate_json(&res_body.message.content, min_duration_secs)
        }
        _ => {
            // Default: DeepSeek
            let default_model = "deepseek-chat".to_string();
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("DEEPSEEK_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or(default_model);

            println!(
                "[DeepSeek] Calling API with model: {} (prompt length: {} chars)",
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.deepseek.com/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling DeepSeek")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("DeepSeek request failed ({status}): {body}"));
            }

            let res_body: DeepseekResponse =
                response.json().await.context("parsing DeepSeek response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("DeepSeek response did not include choices content"))?;

            println!(
                "[DeepSeek] Received response (length: {} chars)",
                text.len()
            );
            parse_candidate_json(&text, min_duration_secs)
        }
    }
}

/// Provider query wrapper with automatic exponential backoff retry on 429 and transient 5xx errors.
pub async fn query_provider_with_retry(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
    min_duration_secs: f64,
    max_retries: usize,
) -> Result<Vec<CandidateDraft>> {
    let mut attempts = 0;
    loop {
        attempts += 1;
        match query_provider_prompt_raw(provider, api_key, model_name, prompt, min_duration_secs)
            .await
        {
            Ok(candidates) => return Ok(candidates),
            Err(err) => {
                if attempts > max_retries {
                    return Err(err);
                }
                let err_str = err.to_string();
                let is_rate_limit =
                    err_str.contains("429") || err_str.to_lowercase().contains("rate limit");
                let is_server_error =
                    err_str.contains("500") || err_str.contains("502") || err_str.contains("503");

                if is_rate_limit || is_server_error {
                    let backoff_secs = if is_rate_limit {
                        2.0 * (attempts as f64)
                    } else {
                        1.0 * (attempts as f64)
                    };
                    eprintln!(
                        "[Provider Retry] Request failed with transient error: {}. Retrying in {:.1}s (attempt {}/{})",
                        err_str, backoff_secs, attempts, max_retries
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs_f64(backoff_secs)).await;
                } else {
                    return Err(err);
                }
            }
        }
    }
}

pub async fn detect_candidates_with_deepseek(
    transcript: &NormalizedTranscript,
    api_key: &str,
    model_name: Option<&str>,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "deepseek",
        api_key,
        model_name,
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

pub async fn detect_candidates_with_gemini(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "gemini",
        api_key,
        None,
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

pub async fn detect_candidates_with_openai(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "openai",
        api_key,
        None,
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

pub async fn detect_candidates_with_openrouter(
    transcript: &NormalizedTranscript,
    api_key: &str,
    model_name: Option<&str>,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "openrouter",
        api_key,
        model_name,
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

pub async fn detect_candidates_with_groq(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "groq",
        api_key,
        None,
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

pub async fn detect_candidates_with_claude(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "claude",
        api_key,
        None,
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

pub async fn detect_candidates_with_local_llm(
    transcript: &NormalizedTranscript,
    model_name: &str,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
    let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
    query_provider_prompt_raw(
        "local",
        "",
        Some(model_name),
        &prompt,
        min_duration(transcript.duration),
    )
    .await
}

/// Universal Full-Timeline Candidate Discovery Engine (Timestamp Generation Mode).
/// - Analyzes 100% of the timeline regardless of video length.
/// - Automatically determines windowing strategy based on duration (single window <= 12m, sliding windows > 12m).
/// - Ensures 0 uncovered seconds across the video timeline.
/// - Executes parallel queries across windows with bounded concurrency (e.g. 3).
/// - Guarantees absolute timestamps are preserved directly from transcript word timings.
/// - Tolerates individual window failures gracefully with retries and fallback.
/// - Pre-snap deduplicates overlapping boundary moments.
pub async fn discover_candidates_timestamp_generation(
    transcript: &NormalizedTranscript,
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    config: &WindowDiscoveryConfig,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let duration = transcript.duration;

    // 1. Check if video qualifies for single-window discovery
    if duration <= config.short_video_threshold_sec {
        println!(
            "[Universal Discovery] Short video detected ({:.1} min <= {:.1} min). Running single-window discovery.",
            duration / 60.0,
            config.short_video_threshold_sec / 60.0
        );
        let transcript_text = format_transcript_for_llm(&transcript.segments, &transcript.words);
        let prompt = build_semantic_prompt(&transcript_text, &lang_cat, &script_type);
        let raw = query_provider_with_retry(
            provider,
            api_key,
            model_name,
            &prompt,
            min_duration(duration),
            2,
        )
        .await?;
        return Ok(raw);
    }

    // 2. Multi-window duration classification
    let (win_size, win_overlap) = if duration <= 2700.0 {
        // Medium: 12-45 min
        (config.medium_window_sec, config.medium_overlap_sec)
    } else {
        // Long / Extra-Long: >45 min
        (config.long_window_sec, config.long_overlap_sec)
    };

    let windows = calculate_sliding_coverage_windows(duration, win_size, win_overlap);
    let total_windows = windows.len();
    println!("\n================================================================================");
    println!(
        "[Universal Discovery] Long-form video ({:.1} min). Spawning {} sliding windows (size: {:.1}m, overlap: {:.1}m)",
        duration / 60.0,
        total_windows,
        win_size / 60.0,
        win_overlap / 60.0
    );
    println!("================================================================================\n");

    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(config.max_concurrency));
    let mut handles = Vec::new();

    for (idx, (w_start, w_end)) in windows.into_iter().enumerate() {
        let sem = semaphore.clone();
        let provider = provider.to_string();
        let api_key = api_key.to_string();
        let model_name = model_name.map(|s| s.to_string());
        let lang_cat = lang_cat.clone();
        let script_type = script_type.clone();
        let transcript_text =
            format_transcript_window(&transcript.segments, &transcript.words, w_start, w_end);
        let target_min = config.candidates_per_window_min;
        let target_max = config.candidates_per_window_max;
        let prompt = build_window_semantic_prompt(
            &transcript_text,
            &lang_cat,
            &script_type,
            w_start,
            w_end,
            idx,
            total_windows,
            target_min,
            target_max,
        );

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore acquired");
            println!(
                "[Window {}/{}] Analyzing [{:.1}s -> {:.1}s] ({:.1} min)...",
                idx + 1,
                total_windows,
                w_start,
                w_end,
                (w_end - w_start) / 60.0
            );
            let res = query_provider_with_retry(
                &provider,
                &api_key,
                model_name.as_deref(),
                &prompt,
                min_duration(duration),
                2,
            )
            .await;
            (idx, w_start, w_end, res)
        }));
    }

    let mut all_drafts = Vec::new();
    let mut failed_windows = 0;

    for handle in handles {
        match handle.await {
            Ok((idx, w_start, w_end, res)) => {
                match res {
                    Ok(mut drafts) => {
                        println!(
                            "[Window {}/{}] Completed [{:.1}s -> {:.1}s]: returned {} raw candidates",
                            idx + 1, total_windows, w_start, w_end, drafts.len()
                        );
                        // Validate timestamps lie within/near window bounds
                        drafts.retain(|cand| {
                            validate_window_candidate_bounds(cand, w_start, w_end, duration)
                        });
                        all_drafts.extend(drafts);
                    }
                    Err(err) => {
                        eprintln!(
                            "[Window {}/{}] Warning: Window [{:.1}s -> {:.1}s] failed after retries: {}. Continuing with remaining windows.",
                            idx + 1, total_windows, w_start, w_end, err
                        );
                        failed_windows += 1;
                    }
                }
            }
            Err(join_err) => {
                eprintln!("[Universal Discovery] Task join error: {}", join_err);
                failed_windows += 1;
            }
        }
    }

    if all_drafts.is_empty() {
        return Err(anyhow!(
            "Universal Discovery failed: 0 candidates returned across all {} windows ({} windows failed)",
            total_windows, failed_windows
        ));
    }

    println!(
        "[Universal Discovery] Aggregated {} raw candidates across {} windows ({} windows failed). Running pre-snap deduplication...",
        all_drafts.len(), total_windows, failed_windows
    );

    let deduped = deduplicate_window_candidates(all_drafts);
    println!(
        "[Universal Discovery] Pre-snap deduplication complete: {} distinct candidates retained across full timeline.",
        deduped.len()
    );

    Ok(deduped)
}

/// REZE-Style candidate discovery using sliding window scoring + continuous highlight span aggregation.
///
/// 1. Classifies language and script.
/// 2. Generates coverage windows (single window for duration <= short_video_threshold_sec, sliding coverage otherwise).
/// 3. Scores windows concurrently under semaphore bounded concurrency, checking `cache` before querying.
/// 4. Normalizes provider scores, aggregates into highlight spans using Kadane + Otsu, and converts to candidate drafts.
pub async fn discover_candidates_reze_scoring(
    transcript: &NormalizedTranscript,
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    config: &WindowDiscoveryConfig,
    cache: Option<&WindowScoreCache>,
) -> Result<Vec<CandidateDraft>> {
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);
    let duration = transcript.duration;

    // 2. Generate coverage windows (granular narrative intervals for REZE scoring)
    let windows = if duration <= 75.0 {
        vec![(0.0, duration)]
    } else {
        let (win_size, win_overlap) = config.reze_window_and_overlap_for_duration(duration);
        calculate_sliding_coverage_windows(duration, win_size, win_overlap)
    };

    let total_windows = windows.len();
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(config.max_concurrency));
    let mut handles = Vec::new();
    let effective_model = resolve_effective_model_name(provider, model_name);
    let source_identity = compute_transcript_source_identity(transcript);
    let active_cache = cache.cloned().or_else(|| Some(get_global_score_cache().clone()));

    // 3. Parallel window scoring with tokio::sync::Semaphore(config.max_concurrency)
    for (idx, (w_start, w_end)) in windows.into_iter().enumerate() {
        let sem = semaphore.clone();
        let provider = provider.to_string();
        let api_key = api_key.to_string();
        let model_name = model_name.map(|s| s.to_string());
        let effective_model = effective_model.clone();
        let lang_cat = lang_cat.clone();
        let script_type = script_type.clone();
        let transcript_text =
            format_transcript_window(&transcript.segments, &transcript.words, w_start, w_end);
        let cache_clone = active_cache.clone();
        let source_identity = source_identity.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore acquired");

            let cache_key = WindowScoreCache::compute_cache_key(
                &source_identity,
                idx,
                w_start,
                w_end,
                &transcript_text,
                &provider,
                &effective_model,
                "v1.0",
            );

            if let Some(ref c) = cache_clone {
                if let Some(cached) = c.get(&cache_key) {
                    println!(
                        "[REZE Scoring] Cache hit for window {}/{} [{:.1}s -> {:.1}s]",
                        idx + 1,
                        total_windows,
                        w_start,
                        w_end
                    );
                    return (idx, w_start, w_end, Ok(cached));
                }
            }

            println!(
                "[REZE Scoring] Analyzing window {}/{} [{:.1}s -> {:.1}s] ({:.1} min)...",
                idx + 1,
                total_windows,
                w_start,
                w_end,
                (w_end - w_start) / 60.0
            );

            let prompt = build_window_scoring_prompt(
                &transcript_text,
                &lang_cat,
                &script_type,
                w_start,
                w_end,
                idx,
                total_windows,
            );

            let res = score_window_with_provider_with_retry(
                &provider,
                &api_key,
                model_name.as_deref(),
                &prompt,
                idx,
                w_start,
                w_end,
                2,
            )
            .await;

            match res {
                Ok(mut score_result) => {
                    score_result.cache_key = cache_key.clone();
                    if let Some(ref c) = cache_clone {
                        c.insert(cache_key, score_result.clone());
                    }
                    (idx, w_start, w_end, Ok(score_result))
                }
                Err(err) => (idx, w_start, w_end, Err(err)),
            }
        }));
    }

    let mut scores = Vec::new();
    let mut failed_windows = 0;

    for handle in handles {
        match handle.await {
            Ok((idx, w_start, w_end, res)) => match res {
                Ok(score) => {
                    scores.push(score);
                }
                Err(err) => {
                    eprintln!(
                        "[REZE Scoring] Warning: Window {}/{} [{:.1}s -> {:.1}s] failed after retries: {}. Continuing with remaining windows.",
                        idx + 1,
                        total_windows,
                        w_start,
                        w_end,
                        err
                    );
                    failed_windows += 1;
                }
            },
            Err(join_err) => {
                eprintln!("[REZE Scoring] Task join error: {}", join_err);
                failed_windows += 1;
            }
        }
    }

    // 4. Aggregation
    if scores.is_empty() {
        return Err(anyhow!(
            "All window scoring attempts failed ({} of {} windows failed)",
            failed_windows,
            total_windows
        ));
    }

    scores.sort_by(|a, b| {
        a.window_start
            .partial_cmp(&b.window_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Phase 0.4: Scorer-discrimination guard on RAW scores (before normalization)
    let raw_scores: Vec<f64> = scores.iter().map(|s| s.raw_score).collect();
    let mut unique_bits: Vec<u64> = raw_scores.iter().map(|x| x.to_bits()).collect();
    unique_bits.sort_unstable();
    unique_bits.dedup();
    let distinct = unique_bits.len();

    let mean = raw_scores.iter().sum::<f64>() / raw_scores.len().max(1) as f64;
    let variance = raw_scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / raw_scores.len().max(1) as f64;
    let std = variance.sqrt();
    let min = raw_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if distinct <= 3 || std < 0.05 || range < 0.10 {
        eprintln!(
            "[REZE Scorer] Degenerate score signal: distinct={}, std={:.4}, range={:.4}, provider={}, model={}. Scorer did not discriminate – falling back to timestamp generation.",
            distinct, std, range, provider, effective_model
        );
        return Err(anyhow!(
            "REZE scorer produced a degenerate signal (distinct={}, std={:.4}, range={:.4})",
            distinct, std, range
        ));
    }

    normalize_provider_scores(&mut scores, 3);

    let spans = aggregate_window_scores_to_spans(
        &scores,
        config.score_smoothing_sigma,
        config.score_kadane_min,
        config.score_otsu_multiplier,
        config.score_valley_drop_ratio,
    );

    // Phase 0.5: Observability run diagnostics
    let smoothed_for_obs = gaussian_smooth_scores(&raw_scores, config.score_smoothing_sigma);
    let t_star_obs = otsu_threshold(&smoothed_for_obs);
    let otsu_term_obs = t_star_obs * config.score_otsu_multiplier;
    let kadane_term_obs = config.score_kadane_min;
    let winning_tau_term = if otsu_term_obs >= kadane_term_obs {
        "Otsu (t_star * otsu_multiplier)"
    } else {
        "Kadane (score_kadane_min)"
    };
    let tau_obs = otsu_term_obs.max(kadane_term_obs);
    let above_tau_count = smoothed_for_obs.iter().filter(|&&s| s >= tau_obs).count();
    let total_span_duration: f64 = spans.iter().map(|(s, e)| (e - s).max(0.0)).sum();
    let coverage_pct = if transcript.duration > 0.0 {
        (total_span_duration / transcript.duration) * 100.0
    } else {
        0.0
    };

    println!(
        "[REZE Run Diagnostics]\n\
         • Raw Scores: distinct={}, std={:.4}, min={:.3}, max={:.3}\n\
         • Tau Cutoff: {:.4} (winning term: {})\n\
         • Windows Above Tau: {}/{} ({:.1}%)\n\
         • Detected Spans: {} spans\n\
         • Timeline Coverage: {:.1}s / {:.1}s ({:.1}% of video)",
        distinct, std, min, max,
        tau_obs, winning_tau_term,
        above_tau_count, smoothed_for_obs.len(),
        (above_tau_count as f64 / smoothed_for_obs.len().max(1) as f64) * 100.0,
        spans.len(),
        total_span_duration, transcript.duration, coverage_pct
    );

    if spans.is_empty() {
        return Err(anyhow!("0 spans aggregated from scoring windows"));
    }

    // Phase 2: Convert spans to candidate drafts via targeted extraction (or 75s partition fallback)
    let drafts = spans_to_candidate_drafts(
        &spans,
        &scores,
        transcript,
        &lang_cat,
        &script_type,
        config,
        Some(provider),
        Some(api_key),
        model_name,
    )
    .await;

    // Phase 1.3: Run second-pass quality ranker on each candidate draft with bounded concurrency
    let eval_semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(config.max_concurrency));
    let mut eval_handles = Vec::new();

    for (cand_idx, draft) in drafts.into_iter().enumerate() {
        let sem = eval_semaphore.clone();
        let transcript_clone = transcript.clone();
        let provider_str = provider.to_string();
        let api_key_str = api_key.to_string();
        let model_name_str = model_name.map(|s| s.to_string());
        let config_clone = config.clone();
        let cache_clone = active_cache.clone();

        eval_handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.expect("evaluator semaphore acquired");
            let res = evaluate_candidate_clip(
                &transcript_clone,
                &draft,
                &provider_str,
                &api_key_str,
                model_name_str.as_deref(),
                &config_clone,
                cache_clone.as_ref(),
                cand_idx,
            )
            .await;

            match res {
                Ok(updated) => updated,
                Err(err) => {
                    eprintln!(
                        "[REZE Evaluator] Warning: Candidate {} [{:.1}s -> {:.1}s] evaluation failed: {}. Keeping pre-evaluation draft.",
                        cand_idx + 1,
                        draft.start,
                        draft.end,
                        err
                    );
                    draft
                }
            }
        }));
    }

    let mut evaluated_drafts = Vec::new();
    for handle in eval_handles {
        match handle.await {
            Ok(draft) => evaluated_drafts.push(draft),
            Err(join_err) => {
                eprintln!("[REZE Evaluator] Task join error: {}", join_err);
            }
        }
    }

    let deduped = deduplicate_window_candidates(evaluated_drafts);
    Ok(deduped)
}

/// Universal Full-Timeline Candidate Discovery Engine.
/// - Respects `config.discovery_mode`:
///   - `DiscoveryMode::TimestampGeneration`: Legacy timestamp-generation path (safe production default).
///   - `DiscoveryMode::WindowScoring`: REZE-style sliding window scoring with deterministic aggregation,
///     falling back automatically to timestamp generation on any error or 0 candidates.
pub async fn discover_candidates_full_timeline(
    transcript: &NormalizedTranscript,
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    config: &WindowDiscoveryConfig,
) -> Result<Vec<CandidateDraft>> {
    match config.discovery_mode {
        DiscoveryMode::TimestampGeneration => {
            discover_candidates_timestamp_generation(
                transcript, provider, api_key, model_name, config,
            )
            .await
        }
        DiscoveryMode::WindowScoring => {
            // Phase 0.6: Restrict REZE to long-form content
            if config.reze_long_form_only && transcript.duration <= config.short_video_threshold_sec {
                println!(
                    "[REZE Scoring] Warning: Video duration ({:.1}s) <= short_video_threshold_sec ({:.1}s). REZE is not appropriate below the long-form threshold. Dispatching to timestamp generation instead.",
                    transcript.duration, config.short_video_threshold_sec
                );
                return discover_candidates_timestamp_generation(
                    transcript, provider, api_key, model_name, config,
                )
                .await;
            }
            match discover_candidates_reze_scoring(
                transcript,
                provider,
                api_key,
                model_name,
                config,
                Some(get_global_score_cache()),
            )
            .await
            {
                Ok(drafts) if !drafts.is_empty() => Ok(drafts),
                Ok(_) | Err(_) => {
                    eprintln!(
                        "[REZE Scoring] Warning: Scoring discovery failed or produced 0 candidates. Falling back safely to timestamp generation with DeepSeek."
                    );
                    let (fallback_provider, fallback_key, fallback_model) =
                        if provider == "openrouter"
                            || provider == "nvidia_diffusiongemma"
                            || provider == "nvidia"
                        {
                            if let Ok(ds_key) = std::env::var("DEEPSEEK_API_KEY") {
                                ("deepseek", ds_key, Some("deepseek-chat"))
                            } else {
                                (provider, api_key.to_string(), model_name)
                            }
                        } else {
                            (provider, api_key.to_string(), model_name)
                        };
                    discover_candidates_timestamp_generation(
                        transcript,
                        fallback_provider,
                        &fallback_key,
                        fallback_model,
                        config,
                    )
                    .await
                }
            }
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn min_duration(transcript_duration: f64) -> f64 {
    if transcript_duration < 60.0 {
        (transcript_duration * 0.4).max(5.0)
    } else {
        20.0
    }
}

// ─── JSON parser — handles clean, markdown-fenced, reasoning-tagged, and truncated outputs ──

/// Preprocess raw LLM response: strip `<think>...</think>` tags and markdown code fences.
pub fn preprocess_llm_response(raw: &str) -> String {
    let mut text = raw.trim().to_string();

    // 1. Strip reasoning / thinking tags (<think>...</think>)
    while let Some(start) = text.find("<think>") {
        if let Some(end) = text[start..].find("</think>") {
            let full_end = start + end + "</think>".len();
            text.replace_range(start..full_end, "");
        } else {
            text.replace_range(start..start + "<think>".len(), "");
        }
    }

    let trimmed = text.trim();

    // 2. Extract content from markdown code fences if present
    if let Some(start_fence) = trimmed.find("```") {
        let after_fence = &trimmed[start_fence + 3..];
        let content_start = if let Some(newline_idx) = after_fence.find('\n') {
            start_fence + 3 + newline_idx + 1
        } else {
            start_fence + 3
        };

        if let Some(end_fence) = trimmed[content_start..].rfind("```") {
            let extracted = &trimmed[content_start..content_start + end_fence];
            return extracted.trim().to_string();
        }
    }

    trimmed.to_string()
}

/// Find a balanced JSON object `{...}` or array `[...]` in text.
/// Uses string-aware brace matching that correctly ignores braces and quotes inside strings.
pub fn find_bounded_json(text: &str) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut start_pos = None;
    let mut open_char = None;
    let mut close_char = None;

    // Find the first opening brace '{' or bracket '['
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'{' {
            start_pos = Some(i);
            open_char = Some(b'{');
            close_char = Some(b'}');
            break;
        } else if byte == b'[' {
            start_pos = Some(i);
            open_char = Some(b'[');
            close_char = Some(b']');
            break;
        }
    }

    let start_idx = start_pos?;
    let open_b = open_char?;
    let close_b = close_char?;

    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for i in start_idx..bytes.len() {
        let byte = bytes[i];
        if escape_next {
            escape_next = false;
            continue;
        }
        if byte == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if byte == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if byte == open_b {
                depth += 1;
            } else if byte == close_b {
                depth -= 1;
                if depth == 0 {
                    return Some((start_idx, i));
                }
            }
        }
    }

    None
}

/// Sanitize common LLM syntax defects like trailing commas before closing braces/brackets.
pub fn sanitize_json_text(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut in_string = false;
    let mut escape_next = false;

    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if escape_next {
            escape_next = false;
            result.push(b);
            i += 1;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            result.push(b);
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            result.push(b);
            i += 1;
            continue;
        }
        if !in_string && b == b',' {
            // Lookahead: check if next non-whitespace char is '}' or ']'
            let mut j = i + 1;
            while j < bytes.len()
                && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'\r' || bytes[j] == b'\n')
            {
                j += 1;
            }
            if j < bytes.len() && (bytes[j] == b'}' || bytes[j] == b']') {
                // Skip trailing comma
                i += 1;
                continue;
            }
        }
        result.push(b);
        i += 1;
    }

    String::from_utf8_lossy(&result).to_string()
}

/// Salvage complete candidate objects from a truncated or unclosed LLM response.
pub fn salvage_candidate_objects(text: &str) -> Vec<serde_json::Value> {
    let mut items = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'{' {
            let mut depth = 0;
            let mut in_string = false;
            let mut escape_next = false;
            let mut obj_end = None;

            for j in i..bytes.len() {
                let b = bytes[j];
                if escape_next {
                    escape_next = false;
                    continue;
                }
                if b == b'\\' && in_string {
                    escape_next = true;
                    continue;
                }
                if b == b'"' {
                    in_string = !in_string;
                    continue;
                }
                if !in_string {
                    if b == b'{' {
                        depth += 1;
                    } else if b == b'}' {
                        depth -= 1;
                        if depth == 0 {
                            obj_end = Some(j);
                            break;
                        }
                    }
                }
            }

            if let Some(end_idx) = obj_end {
                let candidate_slice = &text[i..=end_idx];
                let sanitized = sanitize_json_text(candidate_slice);
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&sanitized) {
                    if val.is_object()
                        && (val.get("clip_start").is_some() || val.get("start").is_some())
                    {
                        items.push(val);
                    }
                }
                i = end_idx + 1;
                continue;
            }
        }
        i += 1;
    }

    items
}

/// Parse the LLM JSON response into a list of `CandidateDraft`.
///
/// Supports:
/// - Pure JSON objects `{ "candidates": [...] }`
/// - Array responses `[ { ... }, ... ]`
/// - Markdown-fenced responses with preambles and commentary
/// - Reasoning-tagged outputs (`<think>...</think>`)
/// - Trailing commas and syntax repairs
/// - Salvaging truncated candidate objects if response hits max tokens
pub fn parse_candidate_json(text: &str, min_duration_secs: f64) -> Result<Vec<CandidateDraft>> {
    let preprocessed = preprocess_llm_response(text);

    // Stage 1: Try direct parse of preprocessed text
    let mut parsed_val = serde_json::from_str::<serde_json::Value>(&preprocessed).ok();

    // Stage 2: If direct parse failed, extract bounded JSON object/array
    if parsed_val.is_none() {
        if let Some((start, end)) = find_bounded_json(&preprocessed) {
            let bounded = &preprocessed[start..=end];
            parsed_val = serde_json::from_str::<serde_json::Value>(bounded).ok();

            // Stage 3: Sanitize trailing commas on bounded snippet
            if parsed_val.is_none() {
                let sanitized = sanitize_json_text(bounded);
                parsed_val = serde_json::from_str::<serde_json::Value>(&sanitized).ok();
            }
        }
    }

    // Stage 4: Try sanitizing the whole preprocessed text
    if parsed_val.is_none() {
        let sanitized = sanitize_json_text(&preprocessed);
        parsed_val = serde_json::from_str::<serde_json::Value>(&sanitized).ok();
    }

    // Stage 5: Salvage candidate objects if root object is truncated or unclosed
    let val = match parsed_val {
        Some(v) => v,
        None => {
            let salvaged = salvage_candidate_objects(&preprocessed);
            if !salvaged.is_empty() {
                println!("[LLM Candidate Parser] Successfully salvaged {} candidates from unclosed response", salvaged.len());
                serde_json::json!({ "candidates": salvaged })
            } else {
                eprintln!("[LLM Candidate Parser] Failed to parse candidate JSON. Raw output snippet:\n{}", &text[..text.len().min(1000)]);
                return Err(anyhow!(
                    "Failed to parse candidate JSON from LLM output (length: {} chars)",
                    text.len()
                ));
            }
        }
    };

    // Locate the candidates array — support "candidates", "moments", "clips", etc.
    let candidates_arr = extract_candidates_array(&val).ok_or_else(|| {
        anyhow!(
            "No candidates array found in LLM output:\n{}",
            &preprocessed[..preprocessed.len().min(500)]
        )
    })?;

    let mut drafts: Vec<CandidateDraft> = Vec::new();

    for item in &candidates_arr {
        // ── timestamps ───────────────────────────────────────────────────────
        let start = coerce_f64(
            item,
            &[
                "start",
                "clip_start",
                "clipStart",
                "start_sec",
                "startTime",
                "start_time",
            ],
        )
        .unwrap_or(0.0);
        let end = coerce_f64(
            item,
            &[
                "end", "clip_end", "clipEnd", "end_sec", "endTime", "end_time",
            ],
        )
        .unwrap_or(0.0);

        if end <= start {
            continue; // skip degenerate entries
        }

        // ── score ─────────────────────────────────────────────────────────
        let mut score = extract_score_field(
            item,
            &[
                "score",
                "virality_score",
                "viralityScore",
                "overall_score",
                "overallScore",
            ],
        )
        .unwrap_or(0.8);
        score = normalize_score(score);

        // ── hook (new schema: hook_text or object; legacy: string/array) ───
        let (hook_text, hook_start, hook_end) = extract_hook(item);

        if hook_text.trim().is_empty() {
            continue;
        }

        // ── payoff ────────────────────────────────────────────────────────
        let (payoff_text, mut payoff_start, mut payoff_end) = extract_payoff(item);
        if payoff_start.is_none() {
            payoff_start = coerce_f64(item, &["payoff_start", "payoffStart"]).or_else(|| {
                item.get("scores")
                    .and_then(|s| coerce_f64(s, &["payoff_start", "payoffStart"]))
            });
        }
        if payoff_end.is_none() {
            payoff_end = coerce_f64(item, &["payoff_end", "payoffEnd"]).or_else(|| {
                item.get("scores")
                    .and_then(|s| coerce_f64(s, &["payoff_end", "payoffEnd"]))
            });
        }

        // ── structure / hook_type ─────────────────────────────────────────
        let structure = extract_string_field(item, &["structure", "hook_type", "hookType"]);
        let hook_type = extract_string_field(item, &["hook_type", "hookType", "structure"]);
        let language = extract_string_field(item, &["language", "lang"]);
        let script = extract_string_field(item, &["script", "script_type", "scriptType"]);
        let summary = extract_string_field(item, &["summary", "description"]);

        let speakers = item.get("speakers").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str().map(|str| str.to_string()))
                .collect::<Vec<_>>()
        });

        let hook_speaker = extract_string_field(item, &["hook_speaker", "hookSpeaker", "speaker"]);
        let context_text = extract_string_field(item, &["context_text", "contextText"]);
        let conversation_type = extract_string_field(
            item,
            &["conversation_type", "conversationType", "structure"],
        );
        let start_reason = extract_string_field(item, &["start_reason", "startReason"]);
        let end_reason = extract_string_field(item, &["end_reason", "endReason"]);

        let semantic_complete =
            extract_bool_field(item, &["semantic_complete", "semanticComplete"]);

        // ── question_hook ──────────────────────────────────────────────────
        let (q_used, q_text, q_start, q_end, q_score) = extract_question_hook(item);

        // ── answer ─────────────────────────────────────────────────────────
        let (a_text, mut a_start, mut a_end) = extract_answer(item);
        if a_start.is_none() {
            a_start = coerce_f64(item, &["answer_start", "answerStart"]).or_else(|| {
                item.get("scores")
                    .and_then(|s| coerce_f64(s, &["answer_start", "answerStart"]))
            });
        }
        if a_end.is_none() {
            a_end = coerce_f64(item, &["answer_end", "answerEnd"]).or_else(|| {
                item.get("scores")
                    .and_then(|s| coerce_f64(s, &["answer_end", "answerEnd"]))
            });
        }

        // ── quality scores ────────────────────────────────────────────────
        let hook_score = extract_score_field(
            item,
            &["hook_strength", "hookStrength", "hook_score", "hookScore"],
        );
        let interviewer_hook_strength = extract_score_field(
            item,
            &[
                "interviewer_hook_strength",
                "interviewerHookStrength",
                "interviewer_hook",
                "interviewerHook",
            ],
        );
        let guest_hook_strength = extract_score_field(
            item,
            &[
                "guest_hook_strength",
                "guestHookStrength",
                "guest_hook",
                "guestHook",
            ],
        );
        let question_relevance_score = extract_score_field(
            item,
            &[
                "question_relevance_score",
                "questionRelevanceScore",
                "question_relevance",
                "questionRelevance",
            ],
        );
        let answer_strength_score = extract_score_field(
            item,
            &[
                "answer_strength_score",
                "answerStrengthScore",
                "answer_strength",
                "answerStrength",
            ],
        );
        let answer_completeness = extract_score_field(
            item,
            &[
                "answer_completeness",
                "answerCompleteness",
                "answer_strength_score",
                "answerStrengthScore",
            ],
        );
        let coherence_score = extract_score_field(
            item,
            &[
                "narrative_completeness",
                "narrativeCompleteness",
                "coherence_score",
                "coherenceScore",
                "coherence",
                "context_completeness",
                "contextCompleteness",
            ],
        );
        let payoff_score = extract_score_field(
            item,
            &[
                "payoff_strength",
                "payoffStrength",
                "payoff_score",
                "payoffScore",
            ],
        );
        let semantic_closure = extract_score_field(
            item,
            &[
                "semantic_closure",
                "semanticClosure",
                "payoff_strength",
                "payoffStrength",
                "payoff_score",
                "payoffScore",
            ],
        );

        let curiosity_score =
            extract_score_field(item, &["curiosity", "curiosity_score", "curiosityScore"]);
        let emotional_impact_score = extract_score_field(
            item,
            &[
                "emotional_impact",
                "emotionalImpact",
                "emotional_impact_score",
                "emotionalImpactScore",
            ],
        );
        let surprise_score =
            extract_score_field(item, &["surprise", "surprise_score", "surpriseScore"]);
        let value_score = extract_score_field(item, &["value", "value_score", "valueScore"]);
        let story_quality_score = extract_score_field(
            item,
            &[
                "story_quality",
                "storyQuality",
                "story_quality_score",
                "storyQualityScore",
            ],
        );
        let context_completeness_score = extract_score_field(
            item,
            &[
                "context_completeness",
                "contextCompleteness",
                "context_completeness_score",
                "contextCompletenessScore",
            ],
        );
        let shareability_score = extract_score_field(
            item,
            &["shareability", "shareability_score", "shareabilityScore"],
        );

        // ── v10.x Hook Intelligence Dimensions (8 fields) ───────────────────
        let hook_topic_clarity = extract_score_field(
            item,
            &[
                "hook_topic_clarity",
                "hookTopicClarity",
                "topic_clarity",
                "topicClarity",
            ],
        );
        let hook_curiosity =
            extract_score_field(item, &["hook_curiosity", "hookCuriosity"]).or(curiosity_score);
        let hook_relevance = extract_score_field(
            item,
            &[
                "hook_relevance",
                "hookRelevance",
                "relevance_score",
                "relevanceScore",
                "relevance",
            ],
        );
        let hook_contrast = extract_score_field(
            item,
            &[
                "hook_contrast",
                "hookContrast",
                "contrast_score",
                "contrastScore",
                "contrast",
            ],
        );
        let hook_confusion_penalty = extract_score_field(
            item,
            &[
                "hook_confusion_penalty",
                "hookConfusionPenalty",
                "confusion_penalty",
                "confusionPenalty",
                "confusion",
            ],
        );
        let hook_delay_penalty = extract_score_field(
            item,
            &[
                "hook_delay_penalty",
                "hookDelayPenalty",
                "delay_penalty",
                "delayPenalty",
                "delay",
            ],
        );
        let hook_irrelevance_penalty = extract_score_field(
            item,
            &[
                "hook_irrelevance_penalty",
                "hookIrrelevancePenalty",
                "irrelevance_penalty",
                "irrelevancePenalty",
                "irrelevance",
            ],
        );
        let hook_disinterest_penalty = extract_score_field(
            item,
            &[
                "hook_disinterest_penalty",
                "hookDisinterestPenalty",
                "disinterest_penalty",
                "disinterestPenalty",
                "disinterest",
            ],
        );

        // ── v10.x Semantic Closure Dimensions (9 fields) ────────────────────
        let continuation_probability = extract_score_field(
            item,
            &[
                "continuation_probability",
                "continuationProbability",
                "continuation_prob",
                "continuationProb",
            ],
        );
        let breath_pause_risk = extract_score_field(
            item,
            &[
                "breath_pause_risk",
                "breathPauseRisk",
                "breath_risk",
                "breathRisk",
            ],
        );
        let mid_thought_risk = extract_score_field(
            item,
            &[
                "mid_thought_risk",
                "midThoughtRisk",
                "mid_thought",
                "midThought",
            ],
        );
        let closure_confidence = extract_score_field(
            item,
            &[
                "closure_confidence",
                "closureConfidence",
                "closure_conf",
                "closureConf",
            ],
        );
        let answer_complete = extract_bool_field(
            item,
            &[
                "answer_complete",
                "answerComplete",
                "is_answer_complete",
                "isAnswerComplete",
            ],
        );
        let story_completeness = extract_bool_field(
            item,
            &[
                "story_completeness",
                "storyCompleteness",
                "story_complete",
                "storyComplete",
                "is_story_complete",
                "isStoryComplete",
            ],
        );
        let payoff_completion = extract_bool_field(
            item,
            &[
                "payoff_completion",
                "payoffCompletion",
                "payoff_complete",
                "payoffComplete",
                "is_payoff_complete",
                "isPayoffComplete",
            ],
        );
        let ending_naturalness = extract_score_field(
            item,
            &["ending_naturalness", "endingNaturalness", "naturalness"],
        );
        let ending_type = extract_string_field(
            item,
            &["ending_type", "endingType", "closure_type", "closureType"],
        );

        // ── v10.1 Six-State Endpoint Classification ─────────────────────────
        let endpoint_state_str = extract_string_field(
            item,
            &[
                "endpoint_state",
                "endpointState",
                "endpoint_classification",
                "endpointClassification",
            ],
        );
        let endpoint_state =
            endpoint_state_str
                .as_deref()
                .and_then(|s| match s.to_lowercase().trim() {
                    "breath_pause" | "breathpause" | "breath_pause_risk" => {
                        Some(crate::models::EndpointState::BreathPause)
                    }
                    "sentence_complete" | "sentencecomplete" | "sentence_completion" => {
                        Some(crate::models::EndpointState::SentenceComplete)
                    }
                    "thought_complete" | "thoughtcomplete" => {
                        Some(crate::models::EndpointState::ThoughtComplete)
                    }
                    "answer_complete" | "answercomplete" | "answer_completion" => {
                        Some(crate::models::EndpointState::AnswerComplete)
                    }
                    "story_complete" | "storycomplete" | "story_resolution" => {
                        Some(crate::models::EndpointState::StoryComplete)
                    }
                    "payoff_complete" | "payoffcomplete" | "punchline" | "life_lesson" => {
                        Some(crate::models::EndpointState::PayoffComplete)
                    }
                    _ => None,
                });

        // ── R6 Hard Rejection Gate for Broken / Weak Opening Candidates ─────
        if semantic_complete == Some(false) {
            println!("[LLM Candidate Parser] Hard rejecting candidate marked semantically incomplete: {:?}", hook_text);
            continue;
        }

        // 1. Topic clarity failure within first 1-3 seconds (< 0.35)
        if let Some(clarity) = hook_topic_clarity {
            if clarity < 0.35 {
                println!("[LLM Candidate Parser] Hard rejecting candidate failing topic clarity ({:.2} < 0.35): {:?}", clarity, hook_text);
                continue;
            }
        }

        // 2. Unexplained pronouns or dangling references without high compensating curiosity
        if let Some(conf_pen) = hook_confusion_penalty {
            let cur = hook_curiosity.unwrap_or(0.0);
            if conf_pen > 0.60 && cur < 0.90 {
                println!("[LLM Candidate Parser] Hard rejecting candidate with high confusion penalty ({:.2}) without high curiosity: {:?}", conf_pen, hook_text);
                continue;
            }
        }

        // 3. Excessive delay penalty (> 0.75)
        if let Some(delay_pen) = hook_delay_penalty {
            if delay_pen > 0.75 {
                println!("[LLM Candidate Parser] Hard rejecting candidate with extreme delay penalty ({:.2}): {:?}", delay_pen, hook_text);
                continue;
            }
        }

        // 4. Starts mid-answer without context or interviewer question
        let ctx_empty = context_text.as_deref().unwrap_or("").trim().is_empty();
        let is_q_hook = q_used.unwrap_or(false) || hook_speaker.as_deref() == Some("Host");
        if ctx_empty && !is_q_hook && hook_confusion_penalty.unwrap_or(0.0) > 0.40 {
            println!("[LLM Candidate Parser] Hard rejecting candidate starting mid-answer with missing context and confusion penalty > 0.40: {:?}", hook_text);
            continue;
        }

        // 5. Extreme continuation probability with low closure confidence
        if continuation_probability.unwrap_or(0.0) > 0.85
            && closure_confidence.unwrap_or(1.0) < 0.30
        {
            println!("[LLM Candidate Parser] Hard rejecting candidate with extreme continuation probability (>0.85) and low closure confidence (<0.30): {:?}", hook_text);
            continue;
        }

        // ── rationale / reason ────────────────────────────────────────────
        let rationale = extract_string_field(
            item,
            &[
                "reason",
                "rationale",
                "explanation",
                "description",
                "summary",
            ],
        )
        .unwrap_or_default();

        drafts.push(CandidateDraft {
            start,
            end,
            score,
            hook: hook_text,
            rationale,
            hook_start,
            hook_end,
            payoff_text,
            payoff_start,
            payoff_end,
            structure,
            hook_score,
            coherence_score,
            payoff_score,
            question_hook_used: q_used,
            question_start: q_start,
            question_end: q_end,
            question_text: q_text,
            question_hook_score: q_score,
            question_relevance_score,
            answer_start: a_start,
            answer_end: a_end,
            answer_text: a_text,
            answer_strength_score,
            language,
            script,
            hook_type,
            speakers,
            summary,
            curiosity_score,
            emotional_impact_score,
            surprise_score,
            value_score,
            story_quality_score,
            context_completeness_score,
            shareability_score,
            hook_speaker,
            context_text,
            conversation_type,
            start_reason,
            end_reason,
            semantic_complete,
            interviewer_hook_strength,
            guest_hook_strength,
            answer_completeness,
            semantic_closure,
            hook_topic_clarity,
            hook_curiosity,
            hook_relevance,
            hook_contrast,
            hook_confusion_penalty,
            hook_delay_penalty,
            hook_irrelevance_penalty,
            hook_disinterest_penalty,
            continuation_probability,
            breath_pause_risk,
            mid_thought_risk,
            closure_confidence,
            answer_complete,
            story_completeness,
            payoff_completion,
            ending_naturalness,
            ending_type,
            endpoint_state,
            ..Default::default()
        });
    }

    // ── Filter by minimum duration (allowing dynamic narrative units down to 12s) ──
    let mut candidates: Vec<CandidateDraft> = drafts
        .iter()
        .filter(|c| (c.end - c.start) >= min_duration_secs.min(15.0))
        .cloned()
        .collect();

    // Fall back with looser filter if nothing passed
    if candidates.is_empty() {
        candidates = drafts
            .into_iter()
            .filter(|c| (c.end - c.start) >= 5.0)
            .collect();
    }

    // ── Pre-Render Quality Gate: filter out candidates with low composite score ──
    let high_quality: Vec<_> = candidates
        .iter()
        .filter(|c| composite_score(c) >= 0.65)
        .cloned()
        .collect();
    if !high_quality.is_empty() {
        candidates = high_quality;
    }

    // ── Stage B: High-Precision Viral Ranking & Deduplication with Adaptive Temporal Bins ──
    let target_count = 12;
    let final_candidates = deduplicate_and_balance_candidates(candidates, 3600.0, target_count);

    println!("[LLM Candidate Parser] Successfully parsed {} high-quality candidates after deduplication and temporal balancing", final_candidates.len());
    Ok(final_candidates)
}

/// Parse the LLM JSON response for a REZE window scoring query into a `WindowScoreResult`.
///
/// Preprocesses the raw response, sanitizes syntax defects, extracts `highlight_score`
/// (and supported aliases) plus sub-dimensions, and enforces that any erroneously emitted
/// timestamp fields are ignored with a warning log.
pub fn parse_window_score_json(
    text: &str,
    window_idx: usize,
    window_start: f64,
    window_end: f64,
    provider: &str,
    model: &str,
) -> Result<WindowScoreResult> {
    let preprocessed = preprocess_llm_response(text);

    // Stage 1: Try direct parse of preprocessed text
    let mut parsed_val = serde_json::from_str::<serde_json::Value>(&preprocessed).ok();

    // Stage 2: If direct parse failed, extract bounded JSON object/array
    if parsed_val.is_none() {
        if let Some((start, end)) = find_bounded_json(&preprocessed) {
            let bounded = &preprocessed[start..=end];
            parsed_val = serde_json::from_str::<serde_json::Value>(bounded).ok();

            // Stage 3: Sanitize trailing commas on bounded snippet
            if parsed_val.is_none() {
                let sanitized = sanitize_json_text(bounded);
                parsed_val = serde_json::from_str::<serde_json::Value>(&sanitized).ok();
            }
        }
    }

    // Stage 4: Try sanitizing the whole preprocessed text
    if parsed_val.is_none() {
        let sanitized = sanitize_json_text(&preprocessed);
        parsed_val = serde_json::from_str::<serde_json::Value>(&sanitized).ok();
    }

    let val = match parsed_val {
        Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
        Some(serde_json::Value::Array(arr)) if !arr.is_empty() && arr[0].is_object() => {
            arr[0].clone()
        }
        _ => {
            return Err(anyhow!(
                "Failed to parse window score JSON from LLM output (length: {} chars)",
                text.len()
            ));
        }
    };

    // Extract highlight_score with supported aliases
    let raw_score = extract_score_field(
        &val,
        &[
            "highlight_score",
            "highlightScore",
            "raw_score",
            "rawScore",
            "score",
            "virality_score",
            "viralityScore",
            "relevance_score",
            "relevanceScore",
        ],
    )
    .ok_or_else(|| anyhow!("No valid score found in window score JSON: {:?}", val))?;

    // Extract optional sub-scores
    let hook_relevance = extract_score_field(
        &val,
        &["hook_relevance", "hookRelevance", "hook_score", "hookScore"],
    );
    let narrative_completeness = extract_score_field(
        &val,
        &[
            "narrative_completeness",
            "narrativeCompleteness",
            "coherence_score",
            "coherenceScore",
            "coherence",
        ],
    );
    let payoff_presence = extract_score_field(
        &val,
        &[
            "payoff_presence",
            "payoffPresence",
            "payoff_score",
            "payoffScore",
        ],
    );
    let engagement_signal = extract_score_field(
        &val,
        &[
            "engagement_signal",
            "engagementSignal",
            "engagement_score",
            "engagementScore",
        ],
    );

    // Check if forbidden timestamp fields were emitted.
    // If present, log a warning and ignore them.
    let forbidden_timestamp_keys = [
        "start",
        "end",
        "clip_start",
        "clip_end",
        "clipStart",
        "clipEnd",
        "hookStart",
        "hookEnd",
        "hook_start",
        "hook_end",
        "payoffStart",
        "payoffEnd",
        "payoff_start",
        "payoff_end",
    ];

    if let Some(obj) = val.as_object() {
        if forbidden_timestamp_keys
            .iter()
            .any(|k| obj.contains_key(*k))
        {
            eprintln!("[REZE Scoring] Warning: LLM emitted timestamp field in scoring mode, ignoring timestamps.");
        }
    }

    Ok(WindowScoreResult {
        window_idx,
        window_start,
        window_end,
        raw_score,
        hook_relevance,
        narrative_completeness,
        payoff_presence,
        engagement_signal,
        provider: provider.to_string(),
        model: model.to_string(),
        cache_key: String::new(),
        prompt_version: "v1.0".to_string(),
    })
}

/// Structured parsed result of candidate clip evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateEvaluationResult {
    pub clip_score: f64,
    pub hook_first_3s: f64,
    pub payoff_lands: f64,
    pub standalone_works: bool,
    pub endpoint_state: Option<crate::models::EndpointState>,
    pub mid_thought_risk: f64,
    pub answer_complete: bool,
    pub story_completeness: bool,
    pub reasoning: String,
}

/// Parses the LLM's candidate evaluation JSON response with robust preprocessing, bounded JSON extraction,
/// and trailing-comma sanitization.
pub fn parse_candidate_evaluation_json(text: &str) -> Result<CandidateEvaluationResult> {
    let preprocessed = preprocess_llm_response(text);

    // Stage 1: Try direct parse of preprocessed text
    let mut parsed_val = serde_json::from_str::<serde_json::Value>(&preprocessed).ok();

    // Stage 2: If direct parse failed, extract bounded JSON object/array
    if parsed_val.is_none() {
        if let Some((start, end)) = find_bounded_json(&preprocessed) {
            let bounded = &preprocessed[start..=end];
            parsed_val = serde_json::from_str::<serde_json::Value>(bounded).ok();

            // Stage 3: Sanitize trailing commas on bounded snippet
            if parsed_val.is_none() {
                let sanitized = sanitize_json_text(bounded);
                parsed_val = serde_json::from_str::<serde_json::Value>(&sanitized).ok();
            }
        }
    }

    // Stage 4: Try sanitizing the whole preprocessed text
    if parsed_val.is_none() {
        let sanitized = sanitize_json_text(&preprocessed);
        parsed_val = serde_json::from_str::<serde_json::Value>(&sanitized).ok();
    }

    let val = match parsed_val {
        Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
        Some(serde_json::Value::Array(arr)) if !arr.is_empty() && arr[0].is_object() => {
            arr[0].clone()
        }
        _ => {
            return Err(anyhow!(
                "Failed to parse candidate evaluation JSON from LLM output (length: {} chars)",
                text.len()
            ));
        }
    };

    let clip_score = extract_score_field(
        &val,
        &[
            "clip_score",
            "clipScore",
            "score",
            "highlight_score",
            "highlightScore",
            "raw_score",
        ],
    )
    .unwrap_or(0.5)
    .clamp(0.0, 1.0);

    let hook_first_3s = extract_score_field(
        &val,
        &["hook_first_3s", "hookFirst3s", "hook_score", "hookScore"],
    )
    .unwrap_or(0.5)
    .clamp(0.0, 1.0);

    let payoff_lands = extract_score_field(
        &val,
        &["payoff_lands", "payoffLands", "payoff_score", "payoffScore"],
    )
    .unwrap_or(0.5)
    .clamp(0.0, 1.0);

    let standalone_works = val
        .get("standalone_works")
        .or_else(|| val.get("standaloneWorks"))
        .or_else(|| val.get("is_standalone"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let endpoint_state_str = val
        .get("endpoint_state")
        .or_else(|| val.get("endpointState"))
        .and_then(|v| v.as_str());

    let endpoint_state = endpoint_state_str.and_then(|s| match s.to_lowercase().trim() {
        "sentence_complete" | "sentencecomplete" | "sentence_completion" => {
            Some(crate::models::EndpointState::SentenceComplete)
        }
        "thought_complete" | "thoughtcomplete" => {
            Some(crate::models::EndpointState::ThoughtComplete)
        }
        "answer_complete" | "answercomplete" | "answer_completion" => {
            Some(crate::models::EndpointState::AnswerComplete)
        }
        "answer_partial" | "answerpartial" => {
            Some(crate::models::EndpointState::ThoughtComplete)
        }
        "breath_pause" | "breathpause" | "breath_pause_risk" | "midthought" | "mid_thought" => {
            Some(crate::models::EndpointState::BreathPause)
        }
        "story_complete" | "storycomplete" | "story_resolution" => {
            Some(crate::models::EndpointState::StoryComplete)
        }
        "payoff_complete" | "payoffcomplete" | "punchline" | "life_lesson" => {
            Some(crate::models::EndpointState::PayoffComplete)
        }
        _ => None,
    });

    let mid_thought_risk = extract_score_field(
        &val,
        &[
            "mid_thought_risk",
            "midThoughtRisk",
            "continuation_risk",
            "incomplete_risk",
        ],
    )
    .unwrap_or(0.1)
    .clamp(0.0, 1.0);

    let answer_complete = val
        .get("answer_complete")
        .or_else(|| val.get("answerComplete"))
        .and_then(|v| v.as_bool())
        .unwrap_or(standalone_works);

    let story_completeness = val
        .get("story_completeness")
        .or_else(|| val.get("storyCompleteness"))
        .and_then(|v| v.as_bool())
        .unwrap_or(standalone_works);

    let reasoning = val
        .get("reasoning")
        .or_else(|| val.get("explanation"))
        .or_else(|| val.get("rationale"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // Check if forbidden timestamp fields were emitted
    let forbidden_keys = [
        "start", "end", "clip_start", "clip_end", "clipStart", "clipEnd",
        "hookStart", "hookEnd", "hook_start", "hook_end", "payoffStart", "payoffEnd",
    ];
    if let Some(obj) = val.as_object() {
        if forbidden_keys.iter().any(|k| obj.contains_key(*k)) {
            eprintln!("[REZE Evaluator] Warning: LLM emitted timestamp fields in evaluation mode, ignoring timestamps.");
        }
    }

    Ok(CandidateEvaluationResult {
        clip_score,
        hook_first_3s,
        payoff_lands,
        standalone_works,
        endpoint_state,
        mid_thought_risk,
        answer_complete,
        story_completeness,
        reasoning,
    })
}

/// Merges highlight spans that overlap by more than `threshold` fraction (e.g. 0.20 = 20%).
pub fn merge_overlapping_spans(spans: &[(f64, f64)], threshold: f64) -> Vec<(f64, f64)> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<(f64, f64)> = spans
        .iter()
        .filter(|(s, e)| !s.is_nan() && !e.is_nan() && e > s)
        .cloned()
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for span in sorted {
        if let Some(last) = merged.last_mut() {
            let overlap = (last.1.min(span.1) - last.0.max(span.0)).max(0.0);
            let dur_last = (last.1 - last.0).max(1e-6);
            let dur_span = (span.1 - span.0).max(1e-6);
            let min_dur = dur_last.min(dur_span);
            if overlap / min_dur >= threshold {
                last.0 = last.0.min(span.0);
                last.1 = last.1.max(span.1);
                continue;
            }
        }
        merged.push(span);
    }
    merged
}

/// Raw LLM query helper for REZE evaluation and targeted extraction.
pub async fn query_reze_provider_text_raw(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
) -> Result<String> {
    let prov_norm = provider.trim().to_ascii_lowercase();
    match prov_norm.as_str() {
        "nvidia_diffusiongemma" | "nvidia" => {
            let base_url = std::env::var("NVIDIA_REZE_BASE_URL")
                .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".to_string());
            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("NVIDIA_REZE_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string());

            let payload = build_nvidia_diffusiongemma_payload(&model, prompt);
            let response = provider_client()?
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(60))
                .json(&payload)
                .send()
                .await
                .context("calling NVIDIA NIM endpoint")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("NVIDIA NIM request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse = response
                .json()
                .await
                .context("parsing NVIDIA NIM ChatCompletion response")?;

            res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("NVIDIA NIM response did not include choices content"))
        }
        "claude" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("ANTHROPIC_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "claude-3-5-sonnet-latest".to_string());

            let response = provider_client()?
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": model,
                    "max_tokens": 4096,
                    "temperature": 0.15,
                    "messages": [
                        ClaudeMessage { role: "user", content: prompt.to_string() }
                    ]
                }))
                .send()
                .await
                .context("calling Claude")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Claude request failed ({status}): {body}"));
            }

            let message: AnthropicMessage =
                response.json().await.context("parsing Claude response")?;
            message
                .content
                .into_iter()
                .find_map(|content| content.text)
                .ok_or_else(|| anyhow!("Claude response did not include text content"))
        }
        "gemini" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("GEMINI_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "gemini-2.5-flash".to_string());
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );

            let response = provider_client()?
                .post(&url)
                .json(&json!({
                    "contents": [{ "parts": [{ "text": prompt }] }],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.15,
                        "maxOutputTokens": 4096
                    }
                }))
                .send()
                .await
                .context("calling Gemini")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Gemini request failed ({status}): {body}"));
            }

            let res_body: GeminiResponse =
                response.json().await.context("parsing Gemini response")?;
            res_body
                .candidates
                .first()
                .and_then(|c| c.content.parts.first())
                .and_then(|p| p.text.clone())
                .ok_or_else(|| anyhow!("Gemini response did not include content text"))
        }
        "openai" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OPENAI_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "gpt-4o-mini".to_string());

            let response = provider_client()?
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling OpenAI")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("OpenAI request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse =
                response.json().await.context("parsing OpenAI response")?;
            res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("OpenAI response did not include choices content"))
        }
        "openrouter" => {
            let default_model = "google/gemini-2.5-flash".to_string();
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OPENROUTER_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or(default_model);

            let response = provider_client()?
                .post("https://openrouter.ai/api/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling OpenRouter")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("OpenRouter request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse = response
                .json()
                .await
                .context("parsing OpenRouter response")?;
            res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("OpenRouter response did not include choices content"))
        }
        _ => {
            // Default: DeepSeek
            let default_model = "deepseek-chat".to_string();
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("DEEPSEEK_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or(default_model);

            let response = provider_client()?
                .post("https://api.deepseek.com/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 4096,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling DeepSeek")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("DeepSeek request failed ({status}): {body}"));
            }

            let res_body: DeepseekResponse =
                response.json().await.context("parsing DeepSeek response")?;
            res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("DeepSeek response did not include choices content"))
        }
    }
}

/// Automatic exponential backoff retry wrapper for query_reze_provider_text_raw.
pub async fn query_reze_provider_text_with_retry(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
    max_retries: usize,
) -> Result<String> {
    let mut attempts = 0;
    loop {
        attempts += 1;
        match query_reze_provider_text_raw(provider, api_key, model_name, prompt).await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if attempts > max_retries {
                    return Err(err);
                }
                let err_str = err.to_string();
                let is_rate_limit =
                    err_str.contains("429") || err_str.to_lowercase().contains("rate limit");
                let is_server_error =
                    err_str.contains("500") || err_str.contains("502") || err_str.contains("503");

                if is_rate_limit || is_server_error {
                    let backoff_secs = if is_rate_limit {
                        2.0 * (attempts as f64)
                    } else {
                        1.0 * (attempts as f64)
                    };
                    tokio::time::sleep(tokio::time::Duration::from_secs_f64(backoff_secs)).await;
                } else {
                    return Err(err);
                }
            }
        }
    }
}

/// Phase 1: Evaluates an extracted CandidateDraft with a second-pass quality ranker.
/// Populates score, hook/payoff scores, closure metadata, and reasoning.
pub async fn evaluate_candidate_clip(
    transcript: &NormalizedTranscript,
    draft: &CandidateDraft,
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    _config: &WindowDiscoveryConfig,
    cache: Option<&WindowScoreCache>,
    cand_idx: usize,
) -> Result<CandidateDraft> {
    let clip_text = format_transcript_window(
        &transcript.segments,
        &transcript.words,
        draft.start,
        draft.end,
    );
    let (lang_cat, script_type) =
        crate::transcription::classify_language_and_script(&transcript.language, &transcript.words);

    let prompt = build_candidate_evaluation_prompt(
        &clip_text,
        &lang_cat,
        &script_type,
        draft.start,
        draft.end,
    );

    let effective_model = resolve_effective_model_name(provider, model_name);
    let cache_key = format!(
        "cand_{}_{:.1}_{:.1}_{}_{}_{}",
        cand_idx, draft.start, draft.end, provider, effective_model, prompt.len()
    );

    if let Some(c) = cache {
        if let Some(cached) = c.get(&cache_key) {
            if let Ok(eval) =
                serde_json::from_str::<CandidateEvaluationResult>(&cached.prompt_version)
            {
                let mut updated = draft.clone();
                updated.score = eval.clip_score;
                updated.hook_score = Some(eval.hook_first_3s);
                updated.payoff_score = Some(eval.payoff_lands);
                updated.endpoint_state = eval.endpoint_state;
                updated.mid_thought_risk = Some(eval.mid_thought_risk);
                updated.answer_complete = Some(eval.answer_complete);
                updated.story_completeness = Some(eval.story_completeness);
                updated.semantic_complete = Some(eval.standalone_works);
                let closure_conf =
                    (eval.clip_score * 0.5 + eval.payoff_lands * 0.5).clamp(0.0, 1.0);
                updated.closure_confidence = Some(closure_conf);
                let cont_prob = (1.0 - eval.payoff_lands).clamp(0.0, 1.0);
                updated.continuation_probability = Some(cont_prob);
                if !eval.reasoning.trim().is_empty() {
                    updated.rationale = eval.reasoning;
                }
                return Ok(updated);
            }
        }
    }

    let raw_text = query_reze_provider_text_with_retry(
        provider,
        api_key,
        model_name,
        &prompt,
        2,
    )
    .await?;

    let eval = parse_candidate_evaluation_json(&raw_text)?;

    if let Some(c) = cache {
        let eval_json = serde_json::to_string(&eval).unwrap_or_default();
        let cached_score = WindowScoreResult {
            window_idx: cand_idx,
            window_start: draft.start,
            window_end: draft.end,
            raw_score: eval.clip_score,
            hook_relevance: Some(eval.hook_first_3s),
            payoff_presence: Some(eval.payoff_lands),
            provider: provider.to_string(),
            model: effective_model,
            cache_key: cache_key.clone(),
            prompt_version: eval_json,
            ..Default::default()
        };
        c.insert(cache_key, cached_score);
    }

    let mut updated = draft.clone();
    updated.score = eval.clip_score;
    updated.hook_score = Some(eval.hook_first_3s);
    updated.payoff_score = Some(eval.payoff_lands);
    updated.endpoint_state = eval.endpoint_state;
    updated.mid_thought_risk = Some(eval.mid_thought_risk);
    updated.answer_complete = Some(eval.answer_complete);
    updated.story_completeness = Some(eval.story_completeness);
    updated.semantic_complete = Some(eval.standalone_works);
    let closure_conf = (eval.clip_score * 0.5 + eval.payoff_lands * 0.5).clamp(0.0, 1.0);
    updated.closure_confidence = Some(closure_conf);
    let cont_prob = (1.0 - eval.payoff_lands).clamp(0.0, 1.0);
    updated.continuation_probability = Some(cont_prob);
    if !eval.reasoning.trim().is_empty() {
        updated.rationale = eval.reasoning;
    }

    Ok(updated)
}

/// Computes a 64-bit FNV-1a hash over byte slice.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Thread-safe in-memory cache for REZE window scoring results.
/// Prevents redundant LLM calls across repeated runs and overlapping window queries.
#[derive(Debug, Clone, Default)]
pub struct WindowScoreCache {
    entries: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, WindowScoreResult>>>,
}

impl WindowScoreCache {
    /// Create a new, empty thread-safe window score cache.
    pub fn new() -> Self {
        Self {
            entries: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Compute a deterministic 64-bit cache key based on the 8 core query parameters.
    pub fn compute_cache_key(
        source_identity: &str,
        window_idx: usize,
        window_start: f64,
        window_end: f64,
        transcript_slice_text: &str,
        provider: &str,
        model: &str,
        prompt_version: &str,
    ) -> String {
        let start_bits = if window_start == 0.0 {
            0.0_f64.to_bits()
        } else {
            window_start.to_bits()
        };
        let end_bits = if window_end == 0.0 {
            0.0_f64.to_bits()
        } else {
            window_end.to_bits()
        };

        let key_material = format!(
            "src:{}\0idx:{}\0start:{}\0end:{}\0text:{}\0prov:{}\0model:{}\0ver:{}",
            source_identity,
            window_idx,
            start_bits,
            end_bits,
            transcript_slice_text,
            provider,
            model,
            prompt_version,
        );
        let hash = fnv1a_64(key_material.as_bytes());
        format!("reze_score_{:016x}", hash)
    }

    /// Retrieve a cached scoring result by key if present.
    ///
    /// Memory first; on a miss, the process-independent disk cache is consulted
    /// and a hit is promoted into memory. Malformed or unreadable disk entries
    /// are treated as absent (the caller re-scores that window).
    pub fn get(&self, key: &str) -> Option<WindowScoreResult> {
        {
            let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = guard.get(key).cloned() {
                return Some(hit);
            }
        }
        let path = reze_disk_cache_path(key)?;
        let json = std::fs::read_to_string(path).ok()?;
        let parsed: WindowScoreResult = serde_json::from_str(&json).ok()?;
        {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            guard.insert(key.to_string(), parsed.clone());
        }
        Some(parsed)
    }

    /// Insert a score result into the cache under the specified key.
    ///
    /// Also persists the entry to the process-independent disk cache
    /// (`<cache_dir>/autoshorts/reze_scores/<key>.json`) so a NEW pipeline
    /// process reuses window scores instead of re-paying every window LLM
    /// call. The key already pins source identity, window bounds, transcript
    /// slice, provider, model and prompt version, so a stale or version-
    /// changed entry can never collide. Disk IO is best-effort: any failure
    /// keeps the in-memory cache fully functional.
    pub fn insert(&self, key: String, result: WindowScoreResult) {
        {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            guard.insert(key.clone(), result.clone());
        }
        if let Some(path) = reze_disk_cache_path(&key) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string(&result) {
                let tmp = path.with_extension("json.tmp");
                if std::fs::write(&tmp, json).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
        }
    }

    /// Return the number of cached window scoring results.
    pub fn len(&self) -> usize {
        let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.len()
    }

    /// Check whether the cache is currently empty.
    pub fn is_empty(&self) -> bool {
        let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.is_empty()
    }

    /// Clear all entries from the cache (both memory and disk).
    pub fn clear(&self) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        for key in guard.keys() {
            if let Some(path) = reze_disk_cache_path(key) {
                let _ = std::fs::remove_file(path);
            }
        }
        guard.clear();
    }
}

/// Disk cache path for one REZE window score or candidate evaluation entry,
/// or None when the platform cache directory is unavailable.
fn reze_disk_cache_path(key: &str) -> Option<std::path::PathBuf> {
    let safe_key: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' })
        .collect();
    if safe_key.is_empty() {
        return None;
    }
    dirs::cache_dir().map(|d| d.join("autoshorts").join("reze_scores").join(format!("{safe_key}.json")))
}

static GLOBAL_SCORE_CACHE: std::sync::OnceLock<WindowScoreCache> = std::sync::OnceLock::new();

/// Access the global singleton `WindowScoreCache` instance.
pub fn get_global_score_cache() -> &'static WindowScoreCache {
    GLOBAL_SCORE_CACHE.get_or_init(WindowScoreCache::new)
}

/// Helper to derive a deterministic source identity string from transcript metadata.
pub fn compute_transcript_source_identity(transcript: &NormalizedTranscript) -> String {
    let word_summary =
        if let (Some(first), Some(last)) = (transcript.words.first(), transcript.words.last()) {
            format!("{}_{}_{}", transcript.words.len(), first.text, last.text)
        } else {
            format!("{}", transcript.words.len())
        };
    format!("dur_{:.1}_{}", transcript.duration, word_summary)
}

/// Resolves effective model name across providers when model_name is None.
fn resolve_effective_model_name(provider: &str, model_name: Option<&str>) -> String {
    let clean = model_name
        .filter(|m| !m.trim().is_empty())
        .map(|m| m.trim().to_string());
    let provider_norm = provider.trim().to_ascii_lowercase();
    match provider_norm.as_str() {
        "claude" => clean
            .or_else(|| {
                std::env::var("ANTHROPIC_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "claude-3-5-sonnet-latest".to_string()),
        "gemini" => clean
            .or_else(|| {
                std::env::var("GEMINI_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "gemini-2.5-flash".to_string()),
        "openai" => clean
            .or_else(|| {
                std::env::var("OPENAI_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "gpt-4o-mini".to_string()),
        "openrouter" => clean
            .or_else(|| {
                std::env::var("OPENROUTER_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "google/gemini-2.5-flash".to_string()),
        "groq" => clean
            .or_else(|| {
                std::env::var("GROQ_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "llama-3.3-70b-versatile".to_string()),
        "local" | "ollama" => clean
            .or_else(|| {
                std::env::var("OLLAMA_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "llama3.2".to_string()),
        "nvidia_diffusiongemma" | "nvidia" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("NVIDIA_REZE_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string()),
        _ => clean
            .or_else(|| {
                std::env::var("DEEPSEEK_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "deepseek-chat".to_string()),
    }
}

/// Normalize provider scores across windows using z-score normalization mapped via logistic sigmoid.
///
/// Groups `scores` by provider name. For groups with at least `min_samples_for_zscore` and
/// standard deviation $\sigma > 1e-6$, scores are transformed to $z = (x - \mu) / \sigma$ and
/// mapped to $[0.0, 1.0]$ via $\frac{1}{1 + e^{-z}}$.
/// Small sample groups (< `min_samples_for_zscore`) or near-zero variance groups ($\sigma \le 1e-6$)
/// retain their original scores clamped to $[0.0, 1.0]$.
pub fn normalize_provider_scores(scores: &mut [WindowScoreResult], min_samples_for_zscore: usize) {
    if scores.is_empty() {
        return;
    }

    let mut provider_indices: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, score) in scores.iter().enumerate() {
        provider_indices
            .entry(score.provider.clone())
            .or_default()
            .push(idx);
    }

    for indices in provider_indices.values() {
        let n = indices.len();
        if n < min_samples_for_zscore {
            for &idx in indices {
                scores[idx].raw_score = scores[idx].raw_score.clamp(0.0, 1.0);
            }
            continue;
        }

        let mean: f64 = indices
            .iter()
            .map(|&idx| scores[idx].raw_score)
            .sum::<f64>()
            / (n as f64);
        let variance: f64 = indices
            .iter()
            .map(|&idx| {
                let diff = scores[idx].raw_score - mean;
                diff * diff
            })
            .sum::<f64>()
            / (n as f64);

        let std_dev = variance.sqrt();
        if std_dev <= 1e-6 {
            for &idx in indices {
                scores[idx].raw_score = scores[idx].raw_score.clamp(0.0, 1.0);
            }
        } else {
            for &idx in indices {
                let z = (scores[idx].raw_score - mean) / std_dev;
                let normalized = 1.0 / (1.0 + (-z).exp());
                scores[idx].raw_score = normalized.clamp(0.0, 1.0);
            }
        }
    }
}

/// Helper to construct the NVIDIA NIM diffusiongemma request payload.
pub fn build_nvidia_diffusiongemma_payload(model: &str, prompt: &str) -> serde_json::Value {
    json!({
        "model": model,
        "messages": [
            { "role": "user", "content": prompt }
        ],
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 0.95,
        "chat_template_kwargs": {
            "enable_thinking": true
        }
    })
}

/// Dispatches a window scoring prompt to the requested LLM provider and parses the response.
pub async fn score_window_with_provider_raw(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
    window_idx: usize,
    window_start: f64,
    window_end: f64,
) -> Result<WindowScoreResult> {
    let prov_norm = provider.trim().to_ascii_lowercase();
    match prov_norm.as_str() {
        "claude" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("ANTHROPIC_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "claude-3-5-sonnet-latest".to_string());

            println!(
                "[Claude] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&json!({
                    "model": model,
                    "max_tokens": 2048,
                    "temperature": 0.15,
                    "messages": [
                        ClaudeMessage { role: "user", content: prompt.to_string() }
                    ]
                }))
                .send()
                .await
                .context("calling Claude")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Claude request failed ({status}): {body}"));
            }

            let message: AnthropicMessage =
                response.json().await.context("parsing Claude response")?;
            let text = message
                .content
                .into_iter()
                .find_map(|content| content.text)
                .ok_or_else(|| anyhow!("Claude response did not include text content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        "gemini" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("GEMINI_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "gemini-2.5-flash".to_string());
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
                model, api_key
            );

            println!(
                "[Gemini] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post(&url)
                .json(&json!({
                    "contents": [{ "parts": [{ "text": prompt }] }],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.15,
                        "maxOutputTokens": 2048
                    }
                }))
                .send()
                .await
                .context("calling Gemini")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Gemini request failed ({status}): {body}"));
            }

            let res_body: GeminiResponse =
                response.json().await.context("parsing Gemini response")?;
            let text = res_body
                .candidates
                .first()
                .and_then(|c| c.content.parts.first())
                .and_then(|p| p.text.clone())
                .ok_or_else(|| anyhow!("Gemini response did not include content text"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        "openai" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OPENAI_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "gpt-4o-mini".to_string());

            println!(
                "[OpenAI] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 2048,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling OpenAI")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("OpenAI request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse =
                response.json().await.context("parsing OpenAI response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("OpenAI response did not include choices content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        "openrouter" => {
            let default_model = "google/gemini-2.5-flash".to_string();
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OPENROUTER_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or(default_model);

            println!(
                "[OpenRouter] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://openrouter.ai/api/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 2048,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling OpenRouter")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("OpenRouter request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse = response
                .json()
                .await
                .context("parsing OpenRouter response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("OpenRouter response did not include choices content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        "groq" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("GROQ_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "llama-3.3-70b-versatile".to_string());

            println!(
                "[Groq] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.groq.com/openai/v1/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 2048,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling Groq")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Groq request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse =
                response.json().await.context("parsing Groq response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("Groq response did not include choices content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        "local" | "ollama" => {
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("OLLAMA_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "llama3.2".to_string());

            let system = "You are a professional podcast editor and viral short-form analyst. \
                You score transcript windows for virality, hook potential, and payoff presence. \
                You always return valid JSON conforming to the requested schema. \
                You never output timestamps.";

            println!(
                "[Ollama] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("http://localhost:11434/api/chat")
                .json(&json!({
                    "model": model,
                    "messages": [
                        { "role": "system", "content": system },
                        { "role": "user",   "content": prompt }
                    ],
                    "stream": false,
                    "options": { "temperature": 0.15, "num_predict": 2048 },
                    "format": "json"
                }))
                .send()
                .await
                .context("calling local Ollama")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("Local Ollama request failed ({status}): {body}"));
            }

            let res_body: OllamaResponse = response
                .json()
                .await
                .context("parsing local Ollama response")?;
            parse_window_score_json(
                &res_body.message.content,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        "nvidia_diffusiongemma" | "nvidia" => {
            let base_url = std::env::var("NVIDIA_REZE_BASE_URL")
                .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".to_string());
            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("NVIDIA_REZE_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string());

            println!(
                "[NVIDIA NIM] Scoring window {} with model: {} at {} (prompt length: {} chars)",
                window_idx,
                model,
                url,
                prompt.len()
            );

            let payload = build_nvidia_diffusiongemma_payload(&model, prompt);

            let response = provider_client()?
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(60))
                .json(&payload)
                .send()
                .await
                .context("calling NVIDIA NIM endpoint")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("NVIDIA NIM request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse = response
                .json()
                .await
                .context("parsing NVIDIA NIM ChatCompletion response")?;

            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("NVIDIA NIM response did not include choices content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        _ => {
            // Default: DeepSeek
            let default_model = "deepseek-chat".to_string();
            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("DEEPSEEK_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or(default_model);

            println!(
                "[DeepSeek] Scoring window {} with model: {} (prompt length: {} chars)",
                window_idx,
                model,
                prompt.len()
            );

            let response = provider_client()?
                .post("https://api.deepseek.com/chat/completions")
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&json!({
                    "model": model,
                    "messages": [{ "role": "user", "content": prompt }],
                    "temperature": 0.15,
                    "max_tokens": 2048,
                    "response_format": { "type": "json_object" }
                }))
                .send()
                .await
                .context("calling DeepSeek")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("DeepSeek request failed ({status}): {body}"));
            }

            let res_body: DeepseekResponse =
                response.json().await.context("parsing DeepSeek response")?;
            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("DeepSeek response did not include choices content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
    }
}

/// Provider window scoring query wrapper with automatic exponential backoff retry on 429 and transient 5xx errors.
pub async fn score_window_with_provider_with_retry(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
    window_idx: usize,
    window_start: f64,
    window_end: f64,
    max_retries: usize,
) -> Result<WindowScoreResult> {
    let mut attempts = 0;
    loop {
        attempts += 1;
        match score_window_with_provider_raw(
            provider,
            api_key,
            model_name,
            prompt,
            window_idx,
            window_start,
            window_end,
        )
        .await
        {
            Ok(result) => return Ok(result),
            Err(err) => {
                if attempts > max_retries {
                    return Err(err);
                }
                let err_str = err.to_string();
                let is_rate_limit =
                    err_str.contains("429") || err_str.to_lowercase().contains("rate limit");
                let is_server_error =
                    err_str.contains("500") || err_str.contains("502") || err_str.contains("503");

                if is_rate_limit || is_server_error {
                    let backoff_secs = if is_rate_limit {
                        2.0 * (attempts as f64)
                    } else {
                        1.0 * (attempts as f64)
                    };
                    eprintln!(
                        "[Window Scoring Retry] Request failed with transient error: {}. Retrying in {:.1}s (attempt {}/{})",
                        err_str, backoff_secs, attempts, max_retries
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs_f64(backoff_secs)).await;
                } else {
                    return Err(err);
                }
            }
        }
    }
}

// ─── REZE Score Aggregation: Gaussian Smoothing, Otsu & Kadane ───────────────

/// Gaussian smoothing over discrete window scores per REZE protocol (arXiv:2608.04480):
///
/// $$w(k) = \exp\left(-\frac{k^2}{2\sigma^2}\right), \quad k \in [-R, R], \quad R = \lceil 3\sigma \rceil$$
/// $$\hat{s}_i = \frac{\sum_{k=-R}^R w(k) \cdot s_{\text{clamp}(i+k, 0, N-1)}}{\sum_{k=-R}^R w(k)}$$
///
/// Smooths window score noise across adjacent windows while preserving sequence length.
pub fn gaussian_smooth_scores(values: &[f64], sigma: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let n = values.len();
    if n == 1 || sigma <= 1e-6 {
        return values.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as isize;
    if radius <= 0 {
        return values.to_vec();
    }

    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut weights = Vec::with_capacity((2 * radius + 1) as usize);
    let mut weight_sum = 0.0;
    for k in -radius..=radius {
        let w = (-((k * k) as f64) / two_sigma_sq).exp();
        weights.push(w);
        weight_sum += w;
    }

    if weight_sum <= 1e-12 {
        return values.to_vec();
    }

    let mut smoothed = Vec::with_capacity(n);
    for i in 0..n {
        let mut weighted_val_sum = 0.0;
        for (idx, k) in (-radius..=radius).enumerate() {
            let neighbor_idx = (i as isize + k).clamp(0, (n - 1) as isize) as usize;
            weighted_val_sum += weights[idx] * values[neighbor_idx];
        }
        smoothed.push(weighted_val_sum / weight_sum);
    }

    smoothed
}

/// Determines the optimal binarization threshold $t^*$ that maximizes between-class variance:
///
/// $$\sigma_B^2(t) = \omega_0(t)\,\omega_1(t)\,(\mu_0(t) - \mu_1(t))^2$$
///
/// where $\omega_0, \omega_1$ are probabilities of scores $\le t$ and $> t$,
/// and $\mu_0, \mu_1$ are the respective class means.
/// If values are empty, returns 0.0. If all values are identical, returns that value.
pub fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    if (sorted[n - 1] - sorted[0]).abs() <= 1e-9 {
        return sorted[0];
    }

    let total_sum: f64 = sorted.iter().sum();
    let mut sum_0 = 0.0;
    let mut max_variance = -1.0;
    let mut best_thresholds: Vec<f64> = Vec::new();

    for i in 0..(n - 1) {
        sum_0 += sorted[i];
        if sorted[i] >= sorted[i + 1] - 1e-9 {
            continue;
        }

        let n0 = (i + 1) as f64;
        let n1 = (n - (i + 1)) as f64;
        let w0 = n0 / (n as f64);
        let w1 = n1 / (n as f64);

        let mu0 = sum_0 / n0;
        let mu1 = (total_sum - sum_0) / n1;

        let diff = mu0 - mu1;
        let variance = w0 * w1 * diff * diff;

        if variance > max_variance + 1e-9 {
            max_variance = variance;
            best_thresholds.clear();
            best_thresholds.push((sorted[i] + sorted[i + 1]) / 2.0);
        } else if (variance - max_variance).abs() <= 1e-9 {
            best_thresholds.push((sorted[i] + sorted[i + 1]) / 2.0);
        }
    }

    if best_thresholds.is_empty() {
        sorted[0]
    } else {
        best_thresholds.iter().sum::<f64>() / (best_thresholds.len() as f64)
    }
}

/// Extracts contiguous intervals with maximal positive score accumulation using Kadane's algorithm
/// enhanced with local valley-reset for long-form content.
///
/// $$y_i = \hat{s}_i - \tau$$
///
/// Standard Kadane only resets when cumulative sum goes negative. This merges all sustained
/// above-threshold regions into one giant span. For long-form conversational content, we add
/// a **local valley reset**: when the score's EXCESS OVER THRESHOLD drops significantly from
/// the LOCAL EXCURSION'S peak excess (valley_drop_ratio), we force the current excursion to
/// end at the PREVIOUS window and start a new one at the CURRENT window, separating distinct
/// local highlights even when both are above threshold.
///
/// Returns time spans `(start_time, end_time)` mapped from window boundaries.
pub fn kadane_score_spans(
    scores: &[f64],
    window_starts: &[f64],
    window_ends: &[f64],
    threshold: f64,
    valley_drop_ratio: f64,
) -> Vec<(f64, f64)> {
    let n = scores.len().min(window_starts.len()).min(window_ends.len());
    if n == 0 {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let mut current_sum = 0.0;
    let mut current_max = 0.0;
    let mut current_start: Option<usize> = None;
    let mut best_end: usize = 0;
    let mut excursion_peak_excess: f64 = 0.0;

    for i in 0..n {
        let y = scores[i] - threshold;

        if current_sum + y > 0.0 {
            // We're in (or starting) an active excursion
            if current_start.is_none() {
                // Starting a new excursion - reset local peak excess
                current_start = Some(i);
                excursion_peak_excess = y.max(0.0);
            }
            // Track peak excess WITHIN this excursion
            excursion_peak_excess = excursion_peak_excess.max(y.max(0.0));

            // Local valley reset: excess drops from THIS EXCURSION'S peak excess
            let current_excess = y.max(0.0);
            let valley_drop = excursion_peak_excess > 0.0
                && current_excess < excursion_peak_excess * valley_drop_ratio;

            if valley_drop && current_start.is_some() && i > current_start.unwrap() {
                // End current span at PREVIOUS window (i-1)
                if current_max > 0.0 {
                    spans.push((window_starts[current_start.unwrap()], window_ends[i - 1]));
                }
                // Start new excursion at CURRENT window
                current_sum = y;
                current_max = if y > 0.0 { y } else { 0.0 };
                current_start = if y > 0.0 { Some(i) } else { None };
                best_end = i;
                excursion_peak_excess = y.max(0.0); // Reset peak excess for new excursion
            } else {
                current_sum += y;
                if current_sum > current_max {
                    current_max = current_sum;
                    best_end = i;
                }
            }
        } else {
            // Standard Kadane reset: cumulative sum went negative
            if let Some(start_idx) = current_start {
                if current_max > 0.0 {
                    spans.push((window_starts[start_idx], window_ends[best_end]));
                }
            }
            current_sum = 0.0;
            current_max = 0.0;
            current_start = None;
            excursion_peak_excess = 0.0; // Reset for next excursion
        }
    }

    // Flush any remaining active excursion at the end of the array
    if let Some(start_idx) = current_start {
        if current_max > 0.0 {
            spans.push((window_starts[start_idx], window_ends[best_end]));
        }
    }

    spans
}

/// Backward-compatible wrapper using default valley_drop_ratio of 0.70 (30% drop from peak).
pub fn kadane_score_spans_default(
    scores: &[f64],
    window_starts: &[f64],
    window_ends: &[f64],
    threshold: f64,
) -> Vec<(f64, f64)> {
    kadane_score_spans(scores, window_starts, window_ends, threshold, 0.70)
}

/// Deterministic REZE Score Aggregation pipeline:
///
/// 1. Extracts raw scores and window boundaries.
/// 2. Applies Gaussian smoothing across adjacent window scores ($\sigma$).
/// 3. Computes Otsu threshold $t^*$ maximizing between-class variance.
/// 4. Determines adaptive cutoff $\tau = (t^* \times \text{otsu\_multiplier}).\max(\text{kadane\_min\_score})$.
/// 5. Applies Kadane span extraction with valley-reset to extract contiguous maximal positive score intervals.
pub fn aggregate_window_scores_to_spans(
    scores: &[WindowScoreResult],
    sigma: f64,
    kadane_min_score: f64,
    otsu_multiplier: f64,
    valley_drop_ratio: f64,
) -> Vec<(f64, f64)> {
    if scores.is_empty() {
        return Vec::new();
    }

    let raw_scores: Vec<f64> = scores.iter().map(|s| s.raw_score).collect();
    let window_starts: Vec<f64> = scores.iter().map(|s| s.window_start).collect();
    let window_ends: Vec<f64> = scores.iter().map(|s| s.window_end).collect();

    let smoothed = gaussian_smooth_scores(&raw_scores, sigma);
    let min_score = smoothed.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_score = smoothed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let is_uniform = smoothed.len() == 1 || (max_score - min_score).abs() <= 1e-9;

    let tau = if is_uniform {
        kadane_min_score
    } else {
        let t_star = otsu_threshold(&smoothed);
        (t_star * otsu_multiplier).max(kadane_min_score)
    };

    kadane_score_spans(
        &smoothed,
        &window_starts,
        &window_ends,
        tau,
        valley_drop_ratio,
    )
}

/// Backward-compatible wrapper using default valley_drop_ratio of 0.70.
pub fn aggregate_window_scores_to_spans_default(
    scores: &[WindowScoreResult],
    sigma: f64,
    kadane_min_score: f64,
    otsu_multiplier: f64,
) -> Vec<(f64, f64)> {
    aggregate_window_scores_to_spans(scores, sigma, kadane_min_score, otsu_multiplier, 0.70)
}

/// Fallback / 75s Partitioning: Converts REZE continuous score highlight spans into `CandidateDraft`s.
///
/// 1. Filters out degenerate spans (`span_end <= span_start` or duration < 8.0s).
/// 2. Partitions spans > 75.0s into short-form tiles.
/// 3. Extracts overlapping words from the transcript to create verbatim hook and payoff text.
/// 4. Leaves closure metadata unset (None) to be populated by the Phase 1 second-pass evaluator.
pub fn spans_to_candidate_drafts_partition(
    spans: &[(f64, f64)],
    scores: &[WindowScoreResult],
    transcript: &NormalizedTranscript,
    lang_category: &str,
    script_type: &str,
) -> Vec<CandidateDraft> {
    let mut drafts = Vec::new();

    // Partition any oversized spans to adhere to short-form video constraints (<= 75.0s)
    let max_cand_duration: f64 = 75.0;
    let mut resolved_spans: Vec<(f64, f64)> = Vec::new();
    for &(span_start, span_end) in spans {
        if span_start.is_nan()
            || span_end.is_nan()
            || span_end <= span_start
            || (span_end - span_start) < 8.0
        {
            continue;
        }

        let dur = span_end - span_start;
        if dur <= max_cand_duration {
            resolved_spans.push((span_start, span_end));
        } else {
            let mut cur = span_start;
            while cur < span_end {
                let next_end = (cur + max_cand_duration).min(span_end);
                if next_end - cur >= 15.0 {
                    resolved_spans.push((cur, next_end));
                }
                if next_end >= span_end {
                    break;
                }
                cur += (max_cand_duration - 15.0).max(45.0);
            }
        }
    }

    for (span_start, span_end) in resolved_spans {
        // 2. Transcript Alignment: extract words overlapping [span_start, span_end]
        let span_words: Vec<&TranscriptWord> = transcript
            .words
            .iter()
            .filter(|w| w.end >= span_start && w.start <= span_end)
            .collect();

        if span_words.is_empty() {
            continue;
        }

        // Hook: First 6-12 words in the span
        let hook_count = span_words.len().min(10);
        let hook_slice = &span_words[..hook_count];
        let hook = hook_slice
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let hook_start = Some(hook_slice.first().unwrap().start);
        let hook_end = Some(hook_slice.last().unwrap().end);

        // Payoff text: Concluding 6-12 words in the span
        let payoff_count = span_words.len().min(10);
        let payoff_slice = &span_words[span_words.len() - payoff_count..];
        let payoff_text = payoff_slice
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let payoff_start = Some(payoff_slice.first().unwrap().start);
        let payoff_end = Some(payoff_slice.last().unwrap().end);

        // 3. Score Mapping: locate overlapping windows
        let overlapping_scores: Vec<&WindowScoreResult> = scores
            .iter()
            .filter(|w| w.window_end >= span_start && w.window_start <= span_end)
            .collect();

        let score = overlapping_scores
            .iter()
            .map(|w| w.raw_score)
            .fold(0.0_f64, |acc, s| acc.max(s))
            .clamp(0.0, 1.0);

        let hook_score = overlapping_scores
            .iter()
            .filter_map(|w| w.hook_relevance)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
            .map(|v| v.clamp(0.0, 1.0));

        let coherence_score = overlapping_scores
            .iter()
            .filter_map(|w| w.narrative_completeness)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
            .map(|v| v.clamp(0.0, 1.0));

        let payoff_score = overlapping_scores
            .iter()
            .filter_map(|w| w.payoff_presence)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
            .map(|v| v.clamp(0.0, 1.0));

        let curiosity_score = overlapping_scores
            .iter()
            .filter_map(|w| w.engagement_signal)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
            .map(|v| v.clamp(0.0, 1.0));

        // 4. Metadata Preservation
        let rationale = format!(
            "REZE highlight span [{:.2}s - {:.2}s] with score {:.2}",
            span_start, span_end, score
        );

        drafts.push(CandidateDraft {
            start: span_start,
            end: span_end,
            score,
            hook,
            rationale,
            hook_start,
            hook_end,
            payoff_text: Some(payoff_text),
            payoff_start,
            payoff_end,
            hook_score,
            coherence_score,
            payoff_score,
            curiosity_score,
            language: Some(lang_category.to_lowercase()),
            script: Some(script_type.to_lowercase()),
            semantic_complete: None,
            closure_confidence: None,
            continuation_probability: None,
            ..Default::default()
        });
    }

    drafts
}

/// Converts REZE continuous score highlight spans into `CandidateDraft`s.
///
/// When `config.reze_targeted_extraction` is true:
/// 1. Merges spans overlapping > 20%.
/// 2. Ranks merged spans by mean smoothed score, taking the top `reze_max_extracted_regions`.
/// 3. Extracts candidate clips from each region using `build_semantic_prompt` and `parse_candidate_json`.
/// 4. Validates clip bounds against region with `validate_window_candidate_bounds`.
/// 5. Falls back to 75s partitioning for any region yielding zero valid candidates.
///
/// When `config.reze_targeted_extraction` is false or no API key is provided, falls back to 75s partitioning.
pub async fn spans_to_candidate_drafts(
    spans: &[(f64, f64)],
    scores: &[WindowScoreResult],
    transcript: &NormalizedTranscript,
    lang_category: &str,
    script_type: &str,
    config: &WindowDiscoveryConfig,
    provider: Option<&str>,
    api_key: Option<&str>,
    model_name: Option<&str>,
) -> Vec<CandidateDraft> {
    if !config.reze_targeted_extraction || api_key.as_deref().unwrap_or("").trim().is_empty() {
        return spans_to_candidate_drafts_partition(
            spans,
            scores,
            transcript,
            lang_category,
            script_type,
        );
    }

    // 1. Merge spans that overlap by more than 20%
    let merged_spans = merge_overlapping_spans(spans, 0.20);
    if merged_spans.is_empty() {
        return Vec::new();
    }

    // 2. Rank merged spans by mean smoothed score, taking top reze_max_extracted_regions
    let raw_scores: Vec<f64> = scores.iter().map(|s| s.raw_score).collect();
    let smoothed = gaussian_smooth_scores(&raw_scores, config.score_smoothing_sigma);

    let mut scored_regions: Vec<((f64, f64), f64)> = merged_spans
        .into_iter()
        .map(|(r_start, r_end)| {
            let mut win_scores = Vec::new();
            for (idx, w) in scores.iter().enumerate() {
                let overlap = (w.window_end.min(r_end) - w.window_start.max(r_start)).max(0.0);
                if overlap > 0.0 {
                    let s = if idx < smoothed.len() {
                        smoothed[idx]
                    } else {
                        w.raw_score
                    };
                    win_scores.push(s);
                }
            }
            let mean_score = if win_scores.is_empty() {
                0.0
            } else {
                win_scores.iter().sum::<f64>() / win_scores.len() as f64
            };
            ((r_start, r_end), mean_score)
        })
        .collect();

    scored_regions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_regions: Vec<(f64, f64)> = scored_regions
        .into_iter()
        .take(config.reze_max_extracted_regions)
        .map(|(span, _)| span)
        .collect();

    let mut all_drafts = Vec::new();
    let prov = provider.unwrap_or("deepseek");
    let key = api_key.unwrap_or("");

    for (r_start, r_end) in top_regions {
        let region_text =
            format_transcript_window(&transcript.segments, &transcript.words, r_start, r_end);
        let prompt = build_semantic_prompt(&region_text, lang_category, script_type);

        let mut region_drafts = Vec::new();
        match query_reze_provider_text_with_retry(prov, key, model_name, &prompt, 2).await {
            Ok(raw_text) => {
                if let Ok(candidates) = parse_candidate_json(&raw_text, 15.0) {
                    for cand in candidates {
                        // 4. Validate every extracted clip against its parent region
                        if validate_window_candidate_bounds(&cand, r_start, r_end, transcript.duration) {
                            region_drafts.push(cand);
                        }
                    }
                }
            }
            Err(err) => {
                eprintln!(
                    "[REZE Extraction] Warning: Region [{:.1}s -> {:.1}s] extraction failed: {}. Falling back to 75s partition for region.",
                    r_start, r_end, err
                );
            }
        }

        // 6. Fallback: if extraction yields zero valid clips for a region,
        // fall back to the old 75s partition for that region only.
        if region_drafts.is_empty() {
            let fallback_drafts = spans_to_candidate_drafts_partition(
                &[(r_start, r_end)],
                scores,
                transcript,
                lang_category,
                script_type,
            );
            all_drafts.extend(fallback_drafts);
        } else {
            all_drafts.extend(region_drafts);
        }
    }

    all_drafts
}

/// Stage B: High-Precision Viral Ranking & Deduplication Engine
/// - Deduplicates overlapping candidates (merging candidates within 6.0s of each other while preserving distinct hooks)
/// - Partitions candidates into:
///   1. Top `target_count` temporally balanced selections across the entire duration (ranks 1..target_count)
///   2. All remaining valid, distinct discovered candidates sorted by composite quality score (ranks target_count+1..N)
/// - Preserves the entire discovered candidate pool without silently dropping candidates.
pub fn deduplicate_and_rank_all_candidates(
    mut raw_candidates: Vec<CandidateDraft>,
    podcast_duration: f64,
    target_count: usize,
) -> Vec<CandidateDraft> {
    if raw_candidates.is_empty() {
        return Vec::new();
    }

    // 1. Sort by composite quality score descending
    raw_candidates.sort_by(|a, b| composite_score(b).total_cmp(&composite_score(a)));

    // 2. Deduplicate overlapping candidates
    let mut deduplicated: Vec<CandidateDraft> = Vec::new();
    for cand in raw_candidates {
        let is_dup = deduplicated.iter().any(|existing| {
            let start_diff = (existing.start - cand.start).abs();
            let end_diff = (existing.end - cand.end).abs();
            let overlap_start = existing.start.max(cand.start);
            let overlap_end = existing.end.min(cand.end);
            let overlap_dur = (overlap_end - overlap_start).max(0.0);
            let min_cand_dur = (existing.end - existing.start)
                .min(cand.end - cand.start)
                .max(1.0);

            (start_diff < 5.0 && end_diff < 6.0)
                || (overlap_dur / min_cand_dur > 0.65 && existing.hook == cand.hook)
        });

        if !is_dup {
            deduplicated.push(cand);
        }
    }

    // 3. Adaptive Temporal Bins: Dynamically divide podcast into bins based on total duration
    if podcast_duration > 600.0 && deduplicated.len() > target_count {
        let max_bins = target_count.min(12).max(3);
        let num_bins = ((podcast_duration / 540.0).round() as usize).clamp(3, max_bins);
        let bin_size = podcast_duration / num_bins as f64;
        let mut balanced: Vec<CandidateDraft> = Vec::new();
        let mut by_bin: Vec<Vec<CandidateDraft>> = vec![Vec::new(); num_bins];

        for cand in deduplicated {
            let bin_idx = ((cand.start / bin_size).floor() as usize).min(num_bins - 1);
            by_bin[bin_idx].push(cand);
        }

        // Take top candidate from each non-empty bin first to ensure full timeline coverage.
        // Note: by_bin was populated from deduplicated (sorted descending by composite_score).
        // Therefore index 0 is the highest score; remove(0) takes the best candidate.
        for bin in &mut by_bin {
            if !bin.is_empty() {
                let top = bin.remove(0);
                balanced.push(top);
            }
        }

        // Collect all remaining unchosen candidates from all bins
        let mut remaining: Vec<CandidateDraft> = by_bin.into_iter().flatten().collect();
        remaining.sort_by(|a, b| composite_score(b).total_cmp(&composite_score(a)));

        // Fill up to target_count slots for the balanced selection
        if balanced.len() < target_count {
            let slots_needed = (target_count - balanced.len()).min(remaining.len());
            let fill: Vec<CandidateDraft> = remaining.drain(..slots_needed).collect();
            balanced.extend(fill);
        }

        // Sort the primary balanced selection by composite score descending
        balanced.sort_by(|a, b| composite_score(b).total_cmp(&composite_score(a)));

        // Append the remaining discovered pool (sorted descending by composite score)
        balanced.extend(remaining);
        return balanced;
    }

    deduplicated
}

/// Backward-compatible wrapper that returns only the top `target_count` balanced candidates.
pub fn deduplicate_and_balance_candidates(
    raw_candidates: Vec<CandidateDraft>,
    podcast_duration: f64,
    target_count: usize,
) -> Vec<CandidateDraft> {
    let mut all =
        deduplicate_and_rank_all_candidates(raw_candidates, podcast_duration, target_count);
    all.truncate(target_count);
    all
}

/// Validate candidate proposed timestamps against the timeline window bounds.
/// Filters out hallucinated timestamps that lie significantly outside the window.
pub fn validate_window_candidate_bounds(
    cand: &CandidateDraft,
    window_start: f64,
    window_end: f64,
    total_duration: f64,
) -> bool {
    let min_allowed = (window_start - 10.0).max(0.0);
    let max_allowed = (window_end + 30.0).min(total_duration);

    if cand.start < min_allowed || cand.start > max_allowed {
        eprintln!(
            "[Window Validator] Dropping candidate starting at {:.2}s outside window [{:.1}s-{:.1}s]",
            cand.start, window_start, window_end
        );
        return false;
    }
    if cand.end <= cand.start || (cand.end - cand.start) < 10.0 {
        return false;
    }
    true
}

/// Pre-snap cross-window candidate deduplication.
/// Merges duplicate moments detected across overlapping window boundaries.
pub fn deduplicate_window_candidates(mut candidates: Vec<CandidateDraft>) -> Vec<CandidateDraft> {
    if candidates.is_empty() {
        return Vec::new();
    }

    candidates.sort_by(|a, b| composite_score(b).total_cmp(&composite_score(a)));

    let mut deduplicated: Vec<CandidateDraft> = Vec::new();
    for cand in candidates {
        let is_dup = deduplicated.iter().any(|existing| {
            let start_diff = (existing.start - cand.start).abs();
            let end_diff = (existing.end - cand.end).abs();
            let overlap_start = existing.start.max(cand.start);
            let overlap_end = existing.end.min(cand.end);
            let overlap_dur = (overlap_end - overlap_start).max(0.0);
            let min_dur = (existing.end - existing.start)
                .min(cand.end - cand.start)
                .max(1.0);

            (start_diff < 6.0 && end_diff < 8.0)
                || (overlap_dur / min_dur > 0.50 && existing.hook == cand.hook)
        });

        if !is_dup {
            deduplicated.push(cand);
        }
    }

    deduplicated
}

/// Calculate overlapping sliding coverage windows ensuring 0 uncovered seconds (00:00 -> duration)
pub fn calculate_sliding_coverage_windows(
    duration: f64,
    window_size: f64,
    overlap: f64,
) -> Vec<(f64, f64)> {
    if duration <= window_size {
        return vec![(0.0, duration)];
    }

    let mut windows = Vec::new();
    let step = (window_size - overlap).max(60.0);
    let mut current_start = 0.0;

    while current_start < duration {
        let current_end = (current_start + window_size).min(duration);
        windows.push((current_start, current_end));

        if current_end >= duration {
            break;
        }

        current_start += step;
        if current_start + (window_size * 0.4) >= duration {
            // Last trailing slice: anchor directly to duration end to ensure 0 uncovered seconds
            let last_start = (duration - window_size).max(0.0);
            if last_start > current_start - step {
                windows.push((last_start, duration));
            }
            break;
        }
    }

    windows
}

/// Compute v9.3 Composite Multimodal Score:
///   Semantic / Narrative: 50%
///   Visual: 15%
///   Audio: 15%
///   Temporal: 20%
/// Hard Quality Gate: Meaning dominates loudness/flashiness.
pub fn compute_multimodal_composite_score(
    semantic_score: f64,
    visual_score: f64,
    audio_score: f64,
    temporal_score: f64,
) -> f64 {
    let w_sem = 0.50;
    let w_vis = 0.15;
    let w_aud = 0.15;
    let w_temp = 0.20;

    let raw = (semantic_score * w_sem)
        + (visual_score * w_vis)
        + (audio_score * w_aud)
        + (temporal_score * w_temp);

    // Hard Semantic Quality Gate: If semantic meaning is weak (<0.45), hard cap score at 0.55
    if semantic_score < 0.45 {
        return raw.min(0.55);
    }

    raw.clamp(0.0, 1.0)
}

/// Composite score = weighted average of multi-dimensional quality signals.
/// Upgraded in v10.x to:
/// - Incorporate positive hook dimensions (topic clarity, curiosity, relevance, contrast)
/// - Subtract hook penalties (delay, confusion, irrelevance, disinterest)
/// - Factor in semantic closure confidence and continuation risk
/// - Factor in answer completeness, story completeness, and emotional impact
/// - Hard gate: cap composite score at 0.50 if confusion penalty is high or topic clarity is dismal
pub fn composite_score(c: &CandidateDraft) -> f64 {
    let base = c.score;

    // 1. Hook Score calculation with topic clarity, curiosity, relevance, contrast, and penalty subtractions
    let raw_hook = c
        .hook_score
        .or(c.guest_hook_strength)
        .or(c.interviewer_hook_strength)
        .unwrap_or(base);

    let clarity = c.hook_topic_clarity.unwrap_or(raw_hook);
    let curiosity = c.hook_curiosity.or(c.curiosity_score).unwrap_or(raw_hook);
    let relevance = c.hook_relevance.unwrap_or(0.8);
    let contrast = c.hook_contrast.unwrap_or(0.7);

    let positive_hook =
        (clarity * 0.35 + curiosity * 0.35 + relevance * 0.15 + contrast * 0.15).clamp(0.0, 1.0);

    // Penalties (DELAY, CONFUSION, IRRELEVANCE, DISINTEREST)
    let delay_pen = c.hook_delay_penalty.unwrap_or(0.0);
    let conf_pen = c.hook_confusion_penalty.unwrap_or(0.0);
    let irrel_pen = c.hook_irrelevance_penalty.unwrap_or(0.0);
    let disint_pen = c.hook_disinterest_penalty.unwrap_or(0.0);
    let total_hook_penalty =
        delay_pen * 0.25 + conf_pen * 0.40 + irrel_pen * 0.20 + disint_pen * 0.15;

    // Opening context penalty (calibrated: soft penalty if opening_context_score is less than 1.0)
    let opening_context_pen = c
        .opening_context_score
        .map(|s| (1.0 - s).max(0.0) * 0.10)
        .unwrap_or(0.0);

    let effective_hook = (positive_hook - total_hook_penalty - opening_context_pen).clamp(0.0, 1.0);

    // 2. Closure & Narrative Score
    let coh = c
        .coherence_score
        .or(c.context_completeness_score)
        .unwrap_or(base);
    let pay = c.payoff_score.or(c.semantic_closure).unwrap_or(base);
    let closure_conf = c.closure_confidence.unwrap_or(0.85);
    let cont_prob = c.continuation_probability.unwrap_or(0.15);
    let closure_factor = (closure_conf * 0.70 + (1.0 - cont_prob) * 0.30).clamp(0.0, 1.0);

    let ans_comp = if let Some(ac) = c.answer_complete {
        if ac {
            1.0
        } else {
            0.3
        }
    } else {
        c.answer_completeness.unwrap_or(coh)
    };

    let story_comp = if let Some(sc) = c.story_completeness {
        if sc {
            1.0
        } else {
            0.4
        }
    } else {
        c.story_quality_score.unwrap_or(coh)
    };

    let emo = c.emotional_impact_score.unwrap_or(0.7);

    // 3. Composite formulation
    let is_qa = c.question_hook_used.unwrap_or(false)
        || c.hook_speaker.as_deref() == Some("Host")
        || c.conversation_type.as_deref() == Some("question_answer");

    let text_score = if is_qa {
        let q_strength = c
            .interviewer_hook_strength
            .or(c.question_relevance_score)
            .unwrap_or(effective_hook);
        let a_strength = c.answer_strength_score.unwrap_or(pay);
        (effective_hook * 0.20
            + q_strength * 0.15
            + a_strength * 0.20
            + coh * 0.15
            + pay * 0.10
            + ans_comp * 0.10
            + closure_factor * 0.10)
            .clamp(0.0, 1.0)
    } else {
        (effective_hook * 0.30
            + coh * 0.15
            + pay * 0.15
            + ans_comp * 0.10
            + story_comp * 0.10
            + closure_factor * 0.10
            + emo * 0.10)
            .clamp(0.0, 1.0)
    };

    // Blend multimodal score with text narrative score without bypassing hook quality
    let blended = if let Some(mm_score) = c.multimodal_score {
        (text_score * 0.60 + mm_score * 0.40).clamp(0.0, 1.0)
    } else if let (Some(vis), Some(aud), Some(temp)) =
        (c.visual_score, c.audio_score, c.temporal_score)
    {
        let mm_comp = compute_multimodal_composite_score(text_score, vis, aud, temp);
        (text_score * 0.60 + mm_comp * 0.40).clamp(0.0, 1.0)
    } else {
        text_score
    };

    // Hard gate: If hook confusion is high or topic clarity is dismal, cap composite score
    if conf_pen > 0.50 || clarity < 0.40 {
        return blended.min(0.50);
    }

    blended
}

/// Helper to extract a score from either top-level or nested "scores" object
fn extract_score_field(item: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    if let Some(s) = coerce_f64(item, keys) {
        return Some(normalize_score(s));
    }
    if let Some(scores_obj) = item.get("scores") {
        if scores_obj.is_object() {
            if let Some(s) = coerce_f64(scores_obj, keys) {
                return Some(normalize_score(s));
            }
        }
    }
    None
}

/// Helper to extract a boolean from either top-level or nested "scores" object
fn extract_bool_field(item: &serde_json::Value, keys: &[&str]) -> Option<bool> {
    for key in keys {
        if let Some(v) = item.get(*key) {
            if let Some(b) = v.as_bool() {
                return Some(b);
            }
            if let Some(s) = v.as_str() {
                match s.trim().to_lowercase().as_str() {
                    "true" | "1" | "yes" => return Some(true),
                    "false" | "0" | "no" => return Some(false),
                    _ => {}
                }
            }
            if let Some(i) = v.as_i64() {
                return Some(i > 0);
            }
        }
    }
    if let Some(scores_obj) = item.get("scores") {
        if scores_obj.is_object() {
            for key in keys {
                if let Some(v) = scores_obj.get(*key) {
                    if let Some(b) = v.as_bool() {
                        return Some(b);
                    }
                    if let Some(s) = v.as_str() {
                        match s.trim().to_lowercase().as_str() {
                            "true" | "1" | "yes" => return Some(true),
                            "false" | "0" | "no" => return Some(false),
                            _ => {}
                        }
                    }
                    if let Some(i) = v.as_i64() {
                        return Some(i > 0);
                    }
                }
            }
        }
    }
    None
}

/// Helper to extract a string from either top-level or nested "scores" object
fn extract_string_field(item: &serde_json::Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(v) = item.get(*key) {
            if let Some(s) = v.as_str() {
                if !s.trim().is_empty() {
                    return Some(s.to_string());
                }
            }
        }
    }
    if let Some(scores_obj) = item.get("scores") {
        if scores_obj.is_object() {
            for key in keys {
                if let Some(v) = scores_obj.get(*key) {
                    if let Some(s) = v.as_str() {
                        if !s.trim().is_empty() {
                            return Some(s.to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract `(text, start, end)` from hook field
fn extract_hook(item: &serde_json::Value) -> (String, Option<f64>, Option<f64>) {
    let h_start = coerce_f64(item, &["hook_start", "hookStart"]);
    let h_end = coerce_f64(item, &["hook_end", "hookEnd"]);

    if let Some(ht) = item
        .get("hook_text")
        .or_else(|| item.get("hookText"))
        .and_then(|v| v.as_str())
    {
        if !ht.trim().is_empty() {
            return (ht.to_string(), h_start, h_end);
        }
    }
    if let Some(hook_obj) = item.get("hook") {
        if hook_obj.is_object() {
            let text = hook_obj
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let start = hook_obj.get("start").and_then(|v| v.as_f64()).or(h_start);
            let end = hook_obj.get("end").and_then(|v| v.as_f64()).or(h_end);
            return (text, start, end);
        }
        if let Some(s) = hook_obj.as_str() {
            return (s.to_string(), h_start, h_end);
        }
        if let Some(arr) = hook_obj.as_array() {
            let joined = arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            if !joined.is_empty() {
                return (joined, h_start, h_end);
            }
        }
    }
    (String::new(), h_start, h_end)
}

/// Extract `(text, start, end)` from the `payoff` field.
fn extract_payoff(item: &serde_json::Value) -> (Option<String>, Option<f64>, Option<f64>) {
    if let Some(pt) = item
        .get("payoff_text")
        .or_else(|| item.get("payoffText"))
        .and_then(|v| v.as_str())
    {
        if !pt.trim().is_empty() {
            let start = coerce_f64(item, &["payoff_start", "payoffStart"]);
            let end = coerce_f64(item, &["payoff_end", "payoffEnd"]);
            return (Some(pt.to_string()), start, end);
        }
    }
    if let Some(pay_obj) = item.get("payoff") {
        if pay_obj.is_object() {
            let text = pay_obj
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let start = pay_obj.get("start").and_then(|v| v.as_f64());
            let end = pay_obj.get("end").and_then(|v| v.as_f64());
            return (text, start, end);
        }
        if let Some(s) = pay_obj.as_str() {
            return (Some(s.to_string()), None, None);
        }
    }
    (None, None, None)
}

/// Locate the candidates array from any common LLM response shape.
fn extract_candidates_array(val: &serde_json::Value) -> Option<Vec<serde_json::Value>> {
    if let Some(arr) = val.as_array() {
        return Some(arr.clone());
    }
    for key in &[
        "candidates",
        "Candidates",
        "moments",
        "clips",
        "segments",
        "results",
        "data",
    ] {
        if let Some(arr) = val.get(*key).and_then(|v| v.as_array()) {
            return Some(arr.clone());
        }
    }
    // Any top-level array field
    if let Some(obj) = val.as_object() {
        for (_k, v) in obj {
            if let Some(arr) = v.as_array() {
                return Some(arr.clone());
            }
        }
    }
    // Single candidate object
    if val.get("start").is_some() || val.get("clip_start").is_some() {
        return Some(vec![val.clone()]);
    }
    None
}

/// Extract question hook metadata: `(used, text, start, end, score)`
fn extract_question_hook(
    item: &serde_json::Value,
) -> (
    Option<bool>,
    Option<String>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
) {
    if let Some(q_obj) = item
        .get("question_hook")
        .or_else(|| item.get("questionHook"))
    {
        if q_obj.is_object() {
            let used = q_obj.get("used").and_then(|v| v.as_bool()).unwrap_or(true);
            let text = q_obj
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let start = q_obj.get("start").and_then(|v| v.as_f64());
            let end = q_obj.get("end").and_then(|v| v.as_f64());
            let score = q_obj
                .get("score")
                .and_then(|v| v.as_f64())
                .map(normalize_score);
            return (Some(used), text, start, end, score);
        }
    }
    // Fallback: check hook.type == "question"
    if let Some(hook_obj) = item.get("hook") {
        if let Some(t) = hook_obj.get("type").and_then(|v| v.as_str()) {
            if t == "question" {
                let text = hook_obj
                    .get("text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let start = hook_obj.get("start").and_then(|v| v.as_f64());
                let end = hook_obj.get("end").and_then(|v| v.as_f64());
                let score = hook_obj
                    .get("score")
                    .and_then(|v| v.as_f64())
                    .map(normalize_score);
                return (Some(true), text, start, end, score);
            }
        }
    }
    (None, None, None, None, None)
}

/// Extract answer metadata: `(text, start, end)`
fn extract_answer(item: &serde_json::Value) -> (Option<String>, Option<f64>, Option<f64>) {
    if let Some(at) = item
        .get("answer_text")
        .or_else(|| item.get("answerText"))
        .and_then(|v| v.as_str())
    {
        if !at.trim().is_empty() {
            let start = coerce_f64(item, &["answer_start", "answerStart"]);
            let end = coerce_f64(item, &["answer_end", "answerEnd"]);
            return (Some(at.to_string()), start, end);
        }
    }
    if let Some(a_obj) = item.get("answer") {
        if a_obj.is_object() {
            let text = a_obj
                .get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let start = a_obj.get("start").and_then(|v| v.as_f64());
            let end = a_obj.get("end").and_then(|v| v.as_f64());
            return (text, start, end);
        }
        if let Some(s) = a_obj.as_str() {
            return (Some(s.to_string()), None, None);
        }
    }
    (None, None, None)
}

/// Try multiple field names and coerce to f64.
fn coerce_f64(item: &serde_json::Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = item.get(*key) {
            if let Some(f) = v.as_f64() {
                return Some(f);
            }
            if let Some(s) = v.as_str() {
                return s.parse::<f64>().ok();
            }
            if let Some(i) = v.as_i64() {
                return Some(i as f64);
            }
        }
    }
    None
}

/// Normalize a score from 0–100 or 0–10 scale into 0.0–1.0.
fn normalize_score(s: f64) -> f64 {
    if s > 100.0 {
        1.0
    } else if s > 10.0 {
        (s / 100.0).clamp(0.0, 1.0)
    } else if s > 1.0 {
        (s / 10.0).clamp(0.0, 1.0)
    } else {
        s.clamp(0.0, 1.0)
    }
}

/// Normalize a word token for linguistic matching: lowercase, strip punctuation, retain unicode letters/digits.
pub fn normalize_token(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Align hook text to the exact word boundaries in `transcript.words`.
/// Searches within a window around `candidate_start` (-12.0s to +25.0s).
/// Returns `Some((verified_start, verified_end, confidence, start_word_idx, end_word_idx))`.
pub fn align_hook_to_transcript_words(
    hook_text: &str,
    candidate_start: f64,
    _candidate_end: f64,
    words: &[TranscriptWord],
) -> Option<(f64, f64, f64, usize, usize)> {
    if words.is_empty() {
        return None;
    }

    let hook_tokens: Vec<String> = hook_text
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();

    if hook_tokens.is_empty() {
        // Fallback to closest word to candidate_start
        let (closest_idx, closest_w) = words.iter().enumerate().min_by(|(_, a), (_, b)| {
            (a.start - candidate_start)
                .abs()
                .total_cmp(&(b.start - candidate_start).abs())
        })?;
        return Some((
            closest_w.start,
            closest_w.end,
            0.40,
            closest_idx,
            closest_idx,
        ));
    }

    // Filter candidate words within a search window around candidate_start
    let search_window_min = (candidate_start - 12.0).max(0.0);
    let search_window_max = candidate_start + 25.0;

    let candidate_word_indices: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.start >= search_window_min && w.start <= search_window_max)
        .map(|(idx, _)| idx)
        .collect();

    let search_indices = if candidate_word_indices.is_empty() {
        (0..words.len()).collect::<Vec<_>>()
    } else {
        candidate_word_indices
    };

    let search_len = hook_tokens.len().min(6);
    let search_prefix = &hook_tokens[..search_len];

    let mut best_match_idx = None;
    let mut best_score = 0.0;

    for &idx in &search_indices {
        let mut match_score = 0.0;
        for (k, target_token) in search_prefix.iter().enumerate() {
            if idx + k < words.len() {
                let w_norm = normalize_token(&words[idx + k].text);
                if &w_norm == target_token {
                    match_score += 1.0;
                } else if !w_norm.is_empty()
                    && (w_norm.contains(target_token) || target_token.contains(&w_norm))
                {
                    match_score += 0.8;
                }
            }
        }
        let score = match_score / (search_len as f64);
        if score > best_score {
            best_score = score;
            best_match_idx = Some(idx);
        }
    }

    if let (Some(start_idx), true) = (best_match_idx, best_score >= 0.50) {
        let start_word = &words[start_idx];
        let end_idx = (start_idx + hook_tokens.len() - 1).min(words.len() - 1);
        let end_word = &words[end_idx];
        Some((
            start_word.start,
            end_word.end,
            best_score,
            start_idx,
            end_idx,
        ))
    } else {
        // Fallback: find word closest to candidate_start
        let (closest_idx, closest_w) = words.iter().enumerate().min_by(|(_, a), (_, b)| {
            (a.start - candidate_start)
                .abs()
                .total_cmp(&(b.start - candidate_start).abs())
        })?;
        Some((
            closest_w.start,
            closest_w.end,
            0.40,
            closest_idx,
            closest_idx,
        ))
    }
}

/// Align payoff text to the exact word boundaries in `transcript.words`.
/// Locates the LLM-predicted payoff in the transcript and returns its real timestamps.
/// Returns `Some((verified_start, verified_end, confidence, start_word_idx, end_word_idx))` on success.
/// Returns `None` on failure (no fallback, no substitution).
pub fn align_payoff_to_transcript_words(
    payoff_text: &str,
    candidate_start: f64,
    candidate_end: f64,
    words: &[TranscriptWord],
) -> Option<(f64, f64, f64, usize, usize)> {
    if words.is_empty() {
        return None;
    }

    let payoff_tokens: Vec<String> = payoff_text
        .split_whitespace()
        .map(normalize_token)
        .filter(|t| !t.is_empty())
        .collect();

    if payoff_tokens.is_empty() {
        return None;
    }

    let search_window_min = (candidate_start - 5.0).max(0.0);

    let candidate_word_indices: Vec<usize> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.start >= search_window_min)
        .map(|(idx, _)| idx)
        .collect();

    let search_indices = if candidate_word_indices.is_empty() {
        (0..words.len()).collect::<Vec<_>>()
    } else {
        candidate_word_indices
    };

    let search_len = payoff_tokens.len();
    let mut best_match_idx = None;
    let mut best_score = 0.0;
    let mut best_dist = f64::MAX;

    for &idx in &search_indices {
        let mut match_score = 0.0;
        for (k, target_token) in payoff_tokens.iter().enumerate() {
            if idx + k < words.len() {
                let w_norm = normalize_token(&words[idx + k].text);
                if &w_norm == target_token {
                    match_score += 1.0;
                } else if !w_norm.is_empty()
                    && (w_norm.contains(target_token) || target_token.contains(&w_norm))
                {
                    match_score += 0.8;
                }
            }
        }
        let score = match_score / (search_len as f64);
        let dist = (words[idx].start - candidate_end).abs();
        if score > best_score || (score == best_score && dist < best_dist) {
            best_score = score;
            best_match_idx = Some(idx);
            best_dist = dist;
        }
    }

    if let (Some(start_idx), true) = (best_match_idx, best_score >= 0.50) {
        let start_word = &words[start_idx];
        let end_idx = (start_idx + payoff_tokens.len() - 1).min(words.len() - 1);
        let end_word = &words[end_idx];
        Some((
            start_word.start,
            end_word.end,
            best_score,
            start_idx,
            end_idx,
        ))
    } else {
        None
    }
}

/// Evaluates the opening 1-3 seconds of speech for instant comprehension,
/// detecting discourse continuation openers and ungrounded third-person references.
pub fn evaluate_opening_hook_context(
    hook_text: &str,
    is_question_hook: bool,
    config: &HookPipelineConfig,
) -> OpeningContextResult {
    let clean_text = hook_text.trim();
    if clean_text.is_empty() {
        return OpeningContextResult {
            score: 0.0,
            penalty: 0.50,
            is_standalone: false,
            has_unresolved_reference: true,
            unresolved_reference: Some("empty_hook".to_string()),
            has_continuation_opener: false,
            matched_marker: None,
            hard_reject: true,
            explanation: "Hook text is empty".to_string(),
        };
    }

    let words: Vec<&str> = clean_text.split_whitespace().collect();
    let lower_first_few = words
        .iter()
        .take(6)
        .map(|w| normalize_token(w))
        .collect::<Vec<_>>()
        .join(" ");

    // 1. Discourse continuation openers
    const CONTINUATION_OPENERS: &[&str] = &[
        "now here is",
        "now heres",
        "at this point",
        "as i said",
        "the same thing",
        "whether they",
        "and this is where",
        "so then",
        "which is why",
        "and so",
        "at that point",
    ];

    let mut matched_continuation = None;
    for &opener in CONTINUATION_OPENERS {
        if lower_first_few.starts_with(opener) {
            matched_continuation = Some(opener.to_string());
            break;
        }
    }

    // 2. Third-person personal pronouns in opening phrase
    let first_sentence = clean_text
        .split(|c| c == '.' || c == '!' || c == '?' || c == '।')
        .next()
        .unwrap_or(clean_text);
    let first_sentence_words: Vec<&str> = first_sentence.split_whitespace().collect();
    let normalized_words: Vec<String> = first_sentence_words
        .iter()
        .map(|w| normalize_token(w))
        .collect();

    const THIRD_PERSON_PRONOUNS: &[&str] = &["he", "she", "they", "him", "her", "them"];
    let mut has_unresolved_3rd_person = false;
    let mut unresolved_pronoun = None;

    if !normalized_words.is_empty() {
        let first_word_norm = &normalized_words[0];
        if THIRD_PERSON_PRONOUNS.contains(&first_word_norm.as_str()) {
            if is_question_hook {
                // Host question setup provides the antecedent -> valid!
                has_unresolved_3rd_person = false;
            } else {
                // Check if sentence provides an intra-sentence antecedent
                // e.g. proper noun / capitalized word after the pronoun ("She, Maria, was...")
                let has_proper_noun = first_sentence_words.iter().skip(1).any(|w| {
                    let chars: Vec<char> = w.chars().collect();
                    !chars.is_empty()
                        && chars[0].is_uppercase()
                        && chars.iter().all(|c| c.is_alphabetic())
                });
                let has_relative_clause = normalized_words.iter().skip(1).any(|w| {
                    w == "who" || w == "whom" || w == "whose" || w == "which" || w == "that"
                });

                if !has_proper_noun && !has_relative_clause {
                    has_unresolved_3rd_person = true;
                    unresolved_pronoun = Some(first_word_norm.clone());
                }
            }
        }
    }

    // 3. Compute penalties
    let mut total_penalty = 0.0;
    let mut hard_reject = false;
    let mut explanation_parts = Vec::new();

    if has_unresolved_3rd_person {
        total_penalty += 0.12;
        explanation_parts.push(format!(
            "ungrounded pronoun '{}'",
            unresolved_pronoun.as_deref().unwrap_or("unknown")
        ));
        if config.hard_reject_unresolved_pronoun {
            hard_reject = true;
        }
    }

    if let Some(ref marker) = matched_continuation {
        total_penalty += config.continuation_penalty;
        explanation_parts.push(format!("continuation marker '{}'", marker));
    }

    let final_score = (1.0 - total_penalty).clamp(0.0, 1.0);
    let is_standalone = !has_unresolved_3rd_person && matched_continuation.is_none();

    let explanation = if is_standalone {
        "Clean standalone opening hook".to_string()
    } else {
        explanation_parts.join("; ")
    };

    OpeningContextResult {
        score: final_score,
        penalty: total_penalty,
        is_standalone,
        has_unresolved_reference: has_unresolved_3rd_person,
        unresolved_reference: unresolved_pronoun,
        has_continuation_opener: matched_continuation.is_some(),
        matched_marker: matched_continuation,
        hard_reject,
        explanation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_candidate_json_structured_schema() {
        let sample_json = r#"{
            "candidates": [
                {
                    "clip_start": 142.35,
                    "clip_end": 188.72,
                    "score": 0.95,
                    "structure": "question_answer_lesson",
                    "hook": {
                        "text": "What was the hardest decision you ever made?",
                        "start": 142.35,
                        "end": 147.20
                    },
                    "body": {
                        "start": 147.20,
                        "end": 177.40
                    },
                    "payoff": {
                        "text": "Leaving my company taught me that failure is the greatest teacher.",
                        "start": 177.40,
                        "end": 188.72
                    },
                    "hook_score": 0.92,
                    "coherence_score": 0.96,
                    "payoff_score": 0.94,
                    "reason": "Complete question-answer-lesson arc with powerful takeaway."
                }
            ]
        }"#;

        let result =
            parse_candidate_json(sample_json, 20.0).expect("Failed to parse candidate JSON");
        assert_eq!(result.len(), 1);

        let candidate = &result[0];
        assert_eq!(candidate.start, 142.35);
        assert_eq!(candidate.end, 188.72);
        assert_eq!(
            candidate.hook,
            "What was the hardest decision you ever made?"
        );
        assert_eq!(candidate.hook_start, Some(142.35));
        assert_eq!(candidate.hook_end, Some(147.20));
        assert_eq!(
            candidate.payoff_text.as_deref(),
            Some("Leaving my company taught me that failure is the greatest teacher.")
        );
        assert_eq!(
            candidate.structure.as_deref(),
            Some("question_answer_lesson")
        );
        assert_eq!(candidate.hook_score, Some(0.92));
        assert_eq!(candidate.coherence_score, Some(0.96));
        assert_eq!(candidate.payoff_score, Some(0.94));
        assert!(candidate.score >= 0.90);
    }

    #[test]
    fn test_parse_candidate_json_legacy_schema_backwards_compatible() {
        let legacy_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 45.0,
                    "score": 0.88,
                    "hook": "This changed everything.",
                    "rationale": "High emotional engagement."
                }
            ]
        }"#;

        let result = parse_candidate_json(legacy_json, 20.0).expect("Failed to parse legacy JSON");
        assert_eq!(result.len(), 1);

        let candidate = &result[0];
        assert_eq!(candidate.start, 10.0);
        assert_eq!(candidate.end, 45.0);
        assert_eq!(candidate.hook, "This changed everything.");
        assert_eq!(candidate.rationale, "High emotional engagement.");
    }

    #[test]
    fn test_parse_candidate_json_markdown_fences_and_preamble() {
        let text = "Here are the top candidates for your short-form video:\n\n```json\n{\n  \"candidates\": [\n    {\n      \"clip_start\": 10.0,\n      \"clip_end\": 40.0,\n      \"score\": 0.9,\n      \"hook\": \"Wait until you hear this...\",\n      \"reason\": \"Viral moment\"\n    }\n  ]\n}\n```\nHope this helps!";
        let result =
            parse_candidate_json(text, 20.0).expect("Failed to parse fenced JSON with preamble");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 10.0);
        assert_eq!(result[0].end, 40.0);
        assert_eq!(result[0].hook, "Wait until you hear this...");
    }

    #[test]
    fn test_parse_candidate_json_deepseek_r1_reasoning_think_tags() {
        let text = "<think>\nLet me analyze the transcript.\n\"Quotes inside thinking\"\nSelecting the best candidate...\n</think>\n```json\n{\n  \"candidates\": [\n    {\n      \"clip_start\": 15.0,\n      \"clip_end\": 48.0,\n      \"score\": 0.95,\n      \"hook\": \"I fight against my mind every day.\",\n      \"reason\": \"Emotional insight\"\n    }\n  ]\n}\n```";
        let result = parse_candidate_json(text, 20.0).expect("Failed to parse DeepSeek R1 output");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 15.0);
        assert_eq!(result[0].end, 48.0);
        assert_eq!(result[0].hook, "I fight against my mind every day.");
    }

    #[test]
    fn test_parse_candidate_json_direct_array_root() {
        let text = "[\n  {\n    \"clip_start\": 12.0,\n    \"clip_end\": 45.0,\n    \"score\": 0.88,\n    \"hook\": \"Direct array test\",\n    \"reason\": \"Valid candidate\"\n  }\n]";
        let result = parse_candidate_json(text, 20.0).expect("Failed to parse array root JSON");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 12.0);
        assert_eq!(result[0].end, 45.0);
    }

    #[test]
    fn test_parse_candidate_json_trailing_commas() {
        let text = "{\n  \"candidates\": [\n    {\n      \"clip_start\": 20.0,\n      \"clip_end\": 60.0,\n      \"score\": 0.85,\n      \"hook\": \"Trailing comma test\",\n      \"reason\": \"Clean parse\",\n    },\n  ],\n}";
        let result =
            parse_candidate_json(text, 20.0).expect("Failed to parse JSON with trailing commas");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 20.0);
        assert_eq!(result[0].end, 60.0);
    }

    #[test]
    fn test_parse_candidate_json_salvage_truncated_output() {
        let text = "{\n  \"candidates\": [\n    {\n      \"clip_start\": 10.0,\n      \"clip_end\": 42.0,\n      \"score\": 0.92,\n      \"hook\": \"Candidate 1 is complete\",\n      \"reason\": \"Good hook\"\n    },\n    {\n      \"clip_start\": 50.0,\n      \"clip_end\": 85.0,\n      \"score\": 0.88,\n      \"hook\": \"Candidate 2 is complete\",\n      \"reason\": \"Good payoff\"\n    },\n    {\n      \"clip_start\": 90.0,\n      \"clip_end\":";
        let result = parse_candidate_json(text, 20.0).expect("Failed to salvage truncated JSON");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].start, 10.0);
        assert_eq!(result[1].start, 50.0);
    }

    #[test]
    fn test_parse_candidate_json_adaptive_question_hook() {
        let sample_json = r#"{
            "candidates": [
                {
                    "clip_start": 10.0,
                    "clip_end": 52.0,
                    "score": 0.94,
                    "structure": "question_answer_lesson",
                    "hook": {
                        "type": "question",
                        "text": "How do you fight against your own mind?",
                        "start": 10.0,
                        "end": 14.5,
                        "score": 0.93
                    },
                    "question_hook": {
                        "used": true,
                        "text": "How do you fight against your own mind?",
                        "start": 10.0,
                        "end": 14.5,
                        "score": 0.93
                    },
                    "answer": {
                        "text": "I fight against my mind every single day through strict physical discipline.",
                        "start": 14.5,
                        "end": 45.0
                    },
                    "payoff": {
                        "text": "Discipline is the bridge between goals and accomplishment.",
                        "start": 45.0,
                        "end": 52.0
                    },
                    "question_relevance_score": 0.95,
                    "answer_strength_score": 0.94,
                    "hook_score": 0.93,
                    "coherence_score": 0.96,
                    "payoff_score": 0.92,
                    "reason": "High-curiosity host question with profound guest answer and memorable payoff."
                }
            ]
        }"#;

        let result =
            parse_candidate_json(sample_json, 20.0).expect("Failed to parse question hook JSON");
        assert_eq!(result.len(), 1);

        let candidate = &result[0];
        assert_eq!(candidate.start, 10.0);
        assert_eq!(candidate.end, 52.0);
        assert_eq!(candidate.question_hook_used, Some(true));
        assert_eq!(
            candidate.question_text.as_deref(),
            Some("How do you fight against your own mind?")
        );
        assert_eq!(candidate.question_start, Some(10.0));
        assert_eq!(candidate.question_end, Some(14.5));
        assert_eq!(candidate.answer_start, Some(14.5));
        assert_eq!(candidate.answer_end, Some(45.0));
        assert_eq!(candidate.question_relevance_score, Some(0.95));
        assert_eq!(candidate.answer_strength_score, Some(0.94));
        assert!(candidate.score >= 0.90);
    }

    #[test]
    fn test_parse_candidate_json_direct_statement_hook() {
        let direct_json = r#"{
            "candidates": [
                {
                    "clip_start": 20.0,
                    "clip_end": 55.0,
                    "score": 0.92,
                    "structure": "statement_development_payoff",
                    "hook": {
                        "type": "direct_statement",
                        "text": "I fight against my mind every single day.",
                        "start": 20.0,
                        "end": 24.0,
                        "score": 0.95
                    },
                    "question_hook": {
                        "used": false,
                        "text": "",
                        "start": null,
                        "end": null
                    },
                    "payoff": {
                        "text": "When you conquer your morning, you conquer your life.",
                        "start": 48.0,
                        "end": 55.0
                    },
                    "hook_score": 0.95,
                    "coherence_score": 0.92,
                    "payoff_score": 0.90,
                    "reason": "Iconic direct statement hook with strong narrative development."
                }
            ]
        }"#;

        let result =
            parse_candidate_json(direct_json, 20.0).expect("Failed to parse direct hook JSON");
        assert_eq!(result.len(), 1);

        let candidate = &result[0];
        assert_eq!(candidate.start, 20.0);
        assert_eq!(candidate.end, 55.0);
        assert_eq!(candidate.question_hook_used, Some(false));
        assert_eq!(candidate.hook, "I fight against my mind every single day.");
        assert_eq!(
            candidate.structure.as_deref(),
            Some("statement_development_payoff")
        );
    }

    #[test]
    fn test_format_transcript_for_llm_contains_words_and_segments() {
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: 5.0,
            speaker: Some("Host".to_string()),
            text: "Welcome to the podcast.".to_string(),
        }];
        let words = vec![
            TranscriptWord {
                start: 0.0,
                end: 0.8,
                speaker: Some("Host".to_string()),
                text: "Welcome".to_string(),
            },
            TranscriptWord {
                start: 0.8,
                end: 1.0,
                speaker: Some("Host".to_string()),
                text: "to".to_string(),
            },
        ];

        let formatted = format_transcript_for_llm(&segments, &words);
        assert!(formatted.contains("=== WORD-LEVEL TIMESTAMPS"));
        assert!(formatted.contains("=== SEGMENTS ==="));
        assert!(formatted.contains("Welcome to the podcast."));
        assert!(formatted.contains("[0.00–0.80] Host: Welcome"));
    }

    #[test]
    fn test_parse_candidate_json_v9_multilingual_hindi() {
        let v9_hindi_json = r#"{
            "candidates": [
                {
                    "start": 15.2,
                    "end": 48.6,
                    "hook_start": 15.2,
                    "hook_end": 19.8,
                    "speakers": ["Host", "Guest"],
                    "hook_type": "question_answer_lesson",
                    "language": "hi",
                    "script": "devanagari",
                    "hook_text": "आपने अपनी ज़िंदगी का सबसे बड़ा फ़ैसला कब लिया?",
                    "summary": "Guest reveals how turning down an offer changed their whole trajectory.",
                    "reason": "Intense question hook with authentic Hindi narrative resolution.",
                    "scores": {
                        "hook_strength": 0.96,
                        "narrative_completeness": 0.94,
                        "payoff_strength": 0.92,
                        "emotional_impact": 0.88,
                        "curiosity": 0.95,
                        "context_completeness": 0.92,
                        "shareability": 0.91,
                        "score": 0.94
                    }
                }
            ]
        }"#;

        let result =
            parse_candidate_json(v9_hindi_json, 15.0).expect("Failed to parse v9 Hindi JSON");
        assert_eq!(result.len(), 1);

        let candidate = &result[0];
        assert_eq!(candidate.start, 15.2);
        assert_eq!(candidate.end, 48.6);
        assert_eq!(candidate.hook, "आपने अपनी ज़िंदगी का सबसे बड़ा फ़ैसला कब लिया?");
        assert_eq!(candidate.language.as_deref(), Some("hi"));
        assert_eq!(candidate.script.as_deref(), Some("devanagari"));
        assert_eq!(candidate.hook_score, Some(0.96));
        assert_eq!(candidate.coherence_score, Some(0.94));
        assert_eq!(candidate.payoff_score, Some(0.92));
        assert_eq!(candidate.curiosity_score, Some(0.95));
        assert!(candidate.score >= 0.90);
    }

    #[test]
    fn test_parse_candidate_json_v9_1_conversation_unit() {
        let v9_1_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 52.4,
                    "hook_start": 10.0,
                    "hook_end": 14.5,
                    "speakers": ["Host", "Guest"],
                    "hook_speaker": "Host",
                    "hook_type": "interviewer_question",
                    "conversation_type": "question_answer",
                    "language": "en",
                    "script": "latin",
                    "hook_text": "What was the hardest decision you made when starting this company?",
                    "context_text": "What was the hardest decision you made when starting this company?",
                    "summary": "Guest explains why turning down a $5M buyout saved the company.",
                    "start_reason": "Opens directly on the high-curiosity host question.",
                    "end_reason": "Guest delivers the final life lesson on long-term conviction.",
                    "payoff_text": "Looking back, that decision changed everything for us.",
                    "semantic_complete": true,
                    "scores": {
                        "interviewer_hook_strength": 0.98,
                        "guest_hook_strength": 0.92,
                        "hook_strength": 0.98,
                        "context_completeness": 0.96,
                        "narrative_completeness": 0.95,
                        "answer_completeness": 0.94,
                        "payoff_strength": 0.95,
                        "semantic_closure": 0.96,
                        "curiosity": 0.97,
                        "emotional_impact": 0.90,
                        "shareability": 0.94,
                        "score": 0.96
                    }
                }
            ]
        }"#;

        let result =
            parse_candidate_json(v9_1_json, 15.0).expect("Failed to parse v9.1 conversation JSON");
        assert_eq!(result.len(), 1);

        let c = &result[0];
        assert_eq!(c.hook_speaker.as_deref(), Some("Host"));
        assert_eq!(c.conversation_type.as_deref(), Some("question_answer"));
        assert_eq!(c.semantic_complete, Some(true));
        assert_eq!(c.interviewer_hook_strength, Some(0.98));
        assert_eq!(c.answer_completeness, Some(0.94));
        assert_eq!(c.semantic_closure, Some(0.96));
        assert_eq!(
            c.context_text.as_deref(),
            Some("What was the hardest decision you made when starting this company?")
        );
        assert_eq!(
            c.start_reason.as_deref(),
            Some("Opens directly on the high-curiosity host question.")
        );
        assert_eq!(
            c.end_reason.as_deref(),
            Some("Guest delivers the final life lesson on long-term conviction.")
        );
        assert_eq!(
            c.payoff_text.as_deref(),
            Some("Looking back, that decision changed everything for us.")
        );
    }

    #[test]
    fn test_candidate_hard_rejection_rules() {
        // Candidate with semantic_complete = false should be rejected
        let incomplete_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 35.0,
                    "hook_text": "He told me that we were done.",
                    "semantic_complete": false,
                    "scores": {
                        "hook_strength": 0.90,
                        "narrative_completeness": 0.30,
                        "payoff_strength": 0.20,
                        "score": 0.50
                    }
                },
                {
                    "start": 40.0,
                    "end": 75.0,
                    "hook_text": "The greatest lesson I learned was patience.",
                    "semantic_complete": true,
                    "scores": {
                        "hook_strength": 0.95,
                        "narrative_completeness": 0.90,
                        "payoff_strength": 0.90,
                        "score": 0.92
                    }
                }
            ]
        }"#;

        let result =
            parse_candidate_json(incomplete_json, 15.0).expect("Failed to parse candidates");
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].hook,
            "The greatest lesson I learned was patience."
        );
    }

    #[test]
    fn test_deduplicate_and_balance_candidates_adaptive_bins() {
        let mut raw = Vec::new();
        // Candidate 1 & 2 overlap heavily (start: 100, end: 140 vs start: 102, end: 142)
        raw.push(CandidateDraft {
            start: 100.0,
            end: 140.0,
            score: 0.95,
            hook: "First moment".to_string(),
            hook_score: Some(0.95),
            coherence_score: Some(0.90),
            payoff_score: Some(0.90),
            semantic_complete: Some(true),
            ..Default::default()
        });
        raw.push(CandidateDraft {
            start: 102.0,
            end: 142.0,
            score: 0.80,
            hook: "First moment".to_string(),
            hook_score: Some(0.80),
            coherence_score: Some(0.80),
            payoff_score: Some(0.80),
            semantic_complete: Some(true),
            ..Default::default()
        });

        // Candidate 3 is in late timeline (2400s)
        raw.push(CandidateDraft {
            start: 2400.0,
            end: 2445.0,
            score: 0.90,
            hook: "Late timeline moment".to_string(),
            hook_score: Some(0.90),
            coherence_score: Some(0.90),
            payoff_score: Some(0.90),
            semantic_complete: Some(true),
            ..Default::default()
        });

        let balanced = deduplicate_and_balance_candidates(raw.clone(), 3600.0, 5);
        assert_eq!(balanced.len(), 2, "Duplicate candidate 2 should be merged");
        assert_eq!(balanced[0].start, 100.0);
        assert_eq!(balanced[1].start, 2400.0);

        let all = deduplicate_and_rank_all_candidates(raw, 3600.0, 5);
        assert_eq!(
            all.len(),
            2,
            "Both distinct candidates preserved in full pool"
        );
    }

    #[test]
    fn test_deduplicate_and_rank_all_candidates_preserves_full_pool_and_orders_selected() {
        let mut raw = Vec::new();
        // Generate 25 distinct candidates spaced across a 3600s podcast
        for i in 0..25 {
            let start = (i as f64) * 140.0;
            raw.push(CandidateDraft {
                start,
                end: start + 30.0,
                score: 0.70 + (i as f64) * 0.01,
                hook: format!("Moment {}", i),
                hook_score: Some(0.70 + (i as f64) * 0.01),
                coherence_score: Some(0.85),
                payoff_score: Some(0.85),
                semantic_complete: Some(true),
                ..Default::default()
            });
        }

        let target_count = 12;
        let all_ranked = deduplicate_and_rank_all_candidates(raw.clone(), 3600.0, target_count);
        assert_eq!(
            all_ranked.len(),
            25,
            "Full discovered pool of 25 candidates must be retained without truncation"
        );

        let truncated = deduplicate_and_balance_candidates(raw, 3600.0, target_count);
        assert_eq!(
            truncated.len(),
            12,
            "Legacy wrapper must return top 12 balanced candidates"
        );
        for i in 0..12 {
            assert_eq!(all_ranked[i].start, truncated[i].start);
            assert_eq!(all_ranked[i].hook, truncated[i].hook);
        }
    }

    #[test]
    fn test_multimodal_composite_scoring_weights_and_gating() {
        // High semantic (0.90), high visual (0.80), high audio (0.85), high temporal (0.90)
        let score_high = compute_multimodal_composite_score(0.90, 0.80, 0.85, 0.90);
        // Expected: 0.90*0.50 + 0.80*0.15 + 0.85*0.15 + 0.90*0.20 = 0.45 + 0.12 + 0.1275 + 0.18 = 0.8775
        assert!((score_high - 0.8775).abs() < 0.001);

        // Meaning Dominates Loudness Gate:
        // Weak semantic (0.30) with flashy visual (0.95), loud audio (0.95), fast temporal (0.90)
        // Raw composite would be: 0.30*0.50 + 0.95*0.15 + 0.95*0.15 + 0.90*0.20 = 0.15 + 0.1425 + 0.1425 + 0.18 = 0.615
        // But hard semantic gate caps score at 0.55
        let score_weak_semantic = compute_multimodal_composite_score(0.30, 0.95, 0.95, 0.90);
        assert!(
            score_weak_semantic <= 0.55,
            "Score {} must be capped at 0.55 due to weak semantic gate",
            score_weak_semantic
        );
    }

    #[test]
    fn test_sliding_coverage_windows_zero_uncovered_seconds() {
        // 75-minute podcast (4500.0s)
        let duration = 4500.0;
        let windows = calculate_sliding_coverage_windows(duration, 420.0, 90.0);
        assert!(!windows.is_empty());
        assert_eq!(windows[0].0, 0.0, "First window must start at 00:00");
        assert_eq!(
            windows.last().unwrap().1,
            duration,
            "Last window must end at exact podcast end"
        );

        // Verify zero uncovered gaps between consecutive windows
        for i in 1..windows.len() {
            let prev_end = windows[i - 1].1;
            let curr_start = windows[i].0;
            assert!(
                curr_start <= prev_end,
                "Gap detected between window {} and {}: {} > {}",
                i - 1,
                i,
                curr_start,
                prev_end
            );
        }
    }

    #[test]
    fn test_build_semantic_prompt_contains_r1_r2_r7_requirements() {
        let prompt = build_semantic_prompt("Test transcript", "HINDI", "devanagari");

        // R1: 4 Failure mechanisms & 4-step workflow
        assert!(prompt.contains("SEGMENT-FIRST HOOK DISCOVERY & REPAIR WORKFLOW"));
        assert!(prompt.contains("DELAY FAILURE"));
        assert!(prompt.contains("CONFUSION FAILURE"));
        assert!(prompt.contains("IRRELEVANCE FAILURE"));
        assert!(prompt.contains("DISINTEREST FAILURE"));
        assert!(prompt.contains("INTERVIEWER QUESTIONS AS FIRST-CLASS HOOKS"));
        assert!(prompt.contains("APPLY HOOK REPAIR STRATEGY"));

        // R2: 8-point closure criteria
        assert!(prompt.contains("8-POINT CLOSURE CRITERIA"));
        assert!(prompt.contains("Grammatical completion of the final sentence"));
        assert!(prompt.contains("Semantic completion of the current thought"));
        assert!(prompt.contains("Narrative completion"));
        assert!(prompt.contains("No unresolved continuation dependency"));
        assert!(prompt.contains("No obvious imminent payoff"));
        assert!(prompt.contains("Natural pause OR speaker turn"));
        assert!(prompt.contains("No dangling conjunction"));
        assert!(prompt.contains("No unfinished list or causal chain"));

        // JSON schema: 17 scoring fields in camelCase
        assert!(prompt.contains("hookTopicClarity"));
        assert!(prompt.contains("hookCuriosity"));
        assert!(prompt.contains("hookRelevance"));
        assert!(prompt.contains("hookContrast"));
        assert!(prompt.contains("hookConfusionPenalty"));
        assert!(prompt.contains("hookDelayPenalty"));
        assert!(prompt.contains("hookIrrelevancePenalty"));
        assert!(prompt.contains("hookDisinterestPenalty"));

        assert!(prompt.contains("continuationProbability"));
        assert!(prompt.contains("breathPauseRisk"));
        assert!(prompt.contains("midThoughtRisk"));
        assert!(prompt.contains("closureConfidence"));
        assert!(prompt.contains("answerComplete"));
        assert!(prompt.contains("storyCompleteness"));
        assert!(prompt.contains("payoffCompletion"));
        assert!(prompt.contains("endingNaturalness"));
        assert!(prompt.contains("endingType"));

        // R7: Multilingual & Hindi Devanagari examples
        assert!(prompt.contains("Primary Classification: HINDI"));
        assert!(prompt.contains("Dominant Script: devanagari"));
        assert!(prompt.contains("आपने अपनी जिंदगी का सबसे कठिन फैसला कब लिया?"));
    }

    #[test]
    fn test_parse_candidate_json_full_17_fields_camel_case() {
        let v10_json = r#"{
            "candidates": [
                {
                    "start": 12.5,
                    "end": 45.0,
                    "score": 0.94,
                    "hookText": "Why you never wake up refreshed",
                    "contextText": "Host asks about chronic fatigue syndrome",
                    "summary": "Guest explains sleep cycles and deep sleep architecture",
                    "startReason": "Immediate topic clarity on sleep deficit",
                    "endReason": "Guest completes actionable advice on sleep timing",
                    "payoffText": "Optimize your delta waves and you change your life.",
                    "semanticComplete": true,
                    "hookSpeaker": "Host",
                    "hookType": "interviewer_question",
                    "conversationType": "question_answer",
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
                    "answerEnd": 44.5,
                    "payoffEnd": 44.5
                }
            ]
        }"#;

        let result =
            parse_candidate_json(v10_json, 15.0).expect("Failed to parse full 17 fields camelCase");
        assert_eq!(result.len(), 1);

        let draft = &result[0];
        assert_eq!(draft.start, 12.5);
        assert_eq!(draft.end, 45.0);
        assert_eq!(draft.hook, "Why you never wake up refreshed");

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
    }

    #[test]
    fn test_parse_candidate_json_full_17_fields_nested_scores() {
        let nested_json = r#"{
            "candidates": [
                {
                    "clip_start": 10.0,
                    "clip_end": 42.0,
                    "hook_text": "The one thing nobody tells you about failure",
                    "semantic_complete": true,
                    "scores": {
                        "hook_topic_clarity": 0.92,
                        "hook_curiosity": 0.94,
                        "hook_relevance": 0.90,
                        "hook_contrast": 0.85,
                        "hook_confusion_penalty": 0.0,
                        "hook_delay_penalty": 0.0,
                        "hook_irrelevance_penalty": 0.0,
                        "hook_disinterest_penalty": 0.0,
                        "continuation_probability": 0.10,
                        "breath_pause_risk": 0.05,
                        "mid_thought_risk": 0.05,
                        "closure_confidence": 0.95,
                        "answer_complete": true,
                        "story_completeness": true,
                        "payoff_completion": true,
                        "ending_naturalness": 0.90,
                        "ending_type": "life_lesson",
                        "score": 0.93
                    }
                }
            ]
        }"#;

        let result = parse_candidate_json(nested_json, 15.0)
            .expect("Failed to parse nested scores 17 fields");
        assert_eq!(result.len(), 1);

        let draft = &result[0];
        assert_eq!(draft.hook_topic_clarity, Some(0.92));
        assert_eq!(draft.hook_curiosity, Some(0.94));
        assert_eq!(draft.continuation_probability, Some(0.10));
        assert_eq!(draft.closure_confidence, Some(0.95));
        assert_eq!(draft.answer_complete, Some(true));
        assert_eq!(draft.story_completeness, Some(true));
        assert_eq!(draft.payoff_completion, Some(true));
        assert_eq!(draft.ending_type.as_deref(), Some("life_lesson"));
    }

    #[test]
    fn test_composite_score_incorporates_positive_hook_and_closure() {
        let clean = CandidateDraft {
            score: 0.92,
            hook_topic_clarity: Some(0.95),
            hook_curiosity: Some(0.92),
            hook_relevance: Some(0.90),
            hook_contrast: Some(0.85),
            hook_confusion_penalty: Some(0.0),
            hook_delay_penalty: Some(0.0),
            hook_irrelevance_penalty: Some(0.0),
            hook_disinterest_penalty: Some(0.0),
            closure_confidence: Some(0.95),
            continuation_probability: Some(0.05),
            answer_complete: Some(true),
            story_completeness: Some(true),
            payoff_completion: Some(true),
            ending_naturalness: Some(0.92),
            ..Default::default()
        };

        let score = composite_score(&clean);
        assert!(
            score > 0.88,
            "Expected clean candidate score > 0.88, got {:.4}",
            score
        );
    }

    #[test]
    fn test_composite_score_penalties_reduce_ranking() {
        let clean = CandidateDraft {
            score: 0.90,
            hook_topic_clarity: Some(0.90),
            hook_curiosity: Some(0.90),
            hook_confusion_penalty: Some(0.0),
            hook_delay_penalty: Some(0.0),
            closure_confidence: Some(0.90),
            continuation_probability: Some(0.10),
            ..Default::default()
        };

        let delayed = CandidateDraft {
            score: 0.90,
            hook_topic_clarity: Some(0.90),
            hook_curiosity: Some(0.90),
            hook_confusion_penalty: Some(0.0),
            hook_delay_penalty: Some(0.50), // delayed intro penalty
            closure_confidence: Some(0.90),
            continuation_probability: Some(0.10),
            ..Default::default()
        };

        let confused = CandidateDraft {
            score: 0.90,
            hook_topic_clarity: Some(0.90),
            hook_curiosity: Some(0.90),
            hook_confusion_penalty: Some(0.40), // pronoun confusion penalty
            hook_delay_penalty: Some(0.0),
            closure_confidence: Some(0.90),
            continuation_probability: Some(0.10),
            ..Default::default()
        };

        let clean_score = composite_score(&clean);
        let delayed_score = composite_score(&delayed);
        let confused_score = composite_score(&confused);

        assert!(
            clean_score > delayed_score,
            "Clean ({:.3}) must rank higher than delayed ({:.3})",
            clean_score,
            delayed_score
        );
        assert!(
            clean_score > confused_score,
            "Clean ({:.3}) must rank higher than confused ({:.3})",
            clean_score,
            confused_score
        );
    }

    #[test]
    fn test_composite_score_hard_capping_on_severe_confusion_or_poor_clarity() {
        let severe_confusion = CandidateDraft {
            score: 0.95,
            hook_score: Some(0.95),
            coherence_score: Some(0.95),
            payoff_score: Some(0.95),
            hook_topic_clarity: Some(0.90),
            hook_curiosity: Some(0.90),
            hook_confusion_penalty: Some(0.55), // > 0.50 triggers cap
            ..Default::default()
        };
        assert!(
            composite_score(&severe_confusion) <= 0.50,
            "Must be capped at 0.50 on high confusion penalty"
        );

        let poor_clarity = CandidateDraft {
            score: 0.95,
            hook_score: Some(0.95),
            coherence_score: Some(0.95),
            payoff_score: Some(0.95),
            hook_topic_clarity: Some(0.38), // < 0.40 triggers cap
            ..Default::default()
        };
        assert!(
            composite_score(&poor_clarity) <= 0.50,
            "Must be capped at 0.50 on poor topic clarity"
        );
    }

    #[test]
    fn test_hard_rejection_low_topic_clarity() {
        let low_clarity_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 40.0,
                    "hookText": "So anyway...",
                    "hookTopicClarity": 0.20,
                    "semanticComplete": true,
                    "score": 0.90
                }
            ]
        }"#;

        let result = parse_candidate_json(low_clarity_json, 15.0).expect("Parse should succeed");
        assert_eq!(
            result.len(),
            0,
            "Candidate failing topic clarity (< 0.35) must be rejected"
        );
    }

    #[test]
    fn test_hard_rejection_high_confusion_penalty_without_curiosity() {
        let confusion_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 40.0,
                    "hookText": "He told me that was it.",
                    "hookConfusionPenalty": 0.75,
                    "hookCuriosity": 0.50,
                    "semanticComplete": true,
                    "score": 0.88
                },
                {
                    "start": 50.0,
                    "end": 85.0,
                    "hookText": "What if everything you knew was wrong?",
                    "hookConfusionPenalty": 0.0,
                    "hookCuriosity": 0.95,
                    "hookTopicClarity": 0.92,
                    "semanticComplete": true,
                    "score": 0.94
                }
            ]
        }"#;

        let result = parse_candidate_json(confusion_json, 15.0).expect("Parse should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].hook, "What if everything you knew was wrong?");
    }

    #[test]
    fn test_hard_rejection_excessive_delay_penalty() {
        let delay_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 40.0,
                    "hookText": "Like I said before, you know, um, so yeah...",
                    "hookDelayPenalty": 0.85,
                    "semanticComplete": true,
                    "score": 0.90
                }
            ]
        }"#;

        let result = parse_candidate_json(delay_json, 15.0).expect("Parse should succeed");
        assert_eq!(
            result.len(),
            0,
            "Candidate with extreme delay penalty (> 0.75) must be rejected"
        );
    }

    #[test]
    fn test_hard_rejection_mid_answer_missing_context() {
        let mid_answer_json = r#"{
            "candidates": [
                {
                    "start": 10.0,
                    "end": 40.0,
                    "hookText": "Because that's how we lost everything.",
                    "contextText": "",
                    "hookSpeaker": "Guest",
                    "hookConfusionPenalty": 0.45,
                    "semanticComplete": true,
                    "score": 0.88
                }
            ]
        }"#;

        let result = parse_candidate_json(mid_answer_json, 15.0).expect("Parse should succeed");
        assert_eq!(result.len(), 0, "Candidate starting mid-answer with missing context and confusion > 0.40 must be rejected");
    }

    #[test]
    fn test_interviewer_question_ranks_above_isolated_guest_statement() {
        let qa_candidate = CandidateDraft {
            start: 10.0,
            end: 45.0,
            score: 0.92,
            hook: "What was the hardest decision of your life?".to_string(),
            hook_speaker: Some("Host".to_string()),
            conversation_type: Some("question_answer".to_string()),
            question_hook_used: Some(true),
            hook_topic_clarity: Some(0.96),
            hook_curiosity: Some(0.95),
            interviewer_hook_strength: Some(0.96),
            answer_strength_score: Some(0.92),
            answer_complete: Some(true),
            closure_confidence: Some(0.95),
            ..Default::default()
        };

        let isolated_guest = CandidateDraft {
            start: 18.0,
            end: 45.0,
            score: 0.85,
            hook: "Leaving my company was terrifying.".to_string(),
            hook_speaker: Some("Guest".to_string()),
            hook_topic_clarity: Some(0.70),
            hook_curiosity: Some(0.75),
            hook_confusion_penalty: Some(0.20),
            closure_confidence: Some(0.85),
            ..Default::default()
        };

        let qa_score = composite_score(&qa_candidate);
        let guest_score = composite_score(&isolated_guest);

        assert!(
            qa_score > guest_score,
            "Interviewer question candidate ({:.3}) must rank above isolated guest ({:.3})",
            qa_score,
            guest_score
        );
    }

    #[test]
    fn test_hindi_hinglish_v10_candidate_parsing_and_scoring() {
        let hindi_v10_json = r#"{
            "candidates": [
                {
                    "start": 25.0,
                    "end": 62.0,
                    "hookText": "आपने अपनी जिंदगी का सबसे कठिन फैसला कब लिया?",
                    "contextText": "आपने अपनी जिंदगी का सबसे कठिन फैसला कब लिया?",
                    "payoffText": "उस एक फैसले ने मेरी पूरी जिंदगी बदल दी।",
                    "summary": "Guest shares the life-defining turning point of their career",
                    "language": "hi",
                    "script": "devanagari",
                    "hookSpeaker": "Host",
                    "hookType": "interviewer_question",
                    "conversationType": "question_answer",
                    "semanticComplete": true,
                    "hookTopicClarity": 0.96,
                    "hookCuriosity": 0.95,
                    "hookRelevance": 0.92,
                    "hookContrast": 0.88,
                    "hookConfusionPenalty": 0.0,
                    "hookDelayPenalty": 0.0,
                    "hookIrrelevancePenalty": 0.0,
                    "hookDisinterestPenalty": 0.0,
                    "continuationProbability": 0.08,
                    "breathPauseRisk": 0.10,
                    "midThoughtRisk": 0.05,
                    "closureConfidence": 0.96,
                    "answerComplete": true,
                    "storyCompleteness": true,
                    "payoffCompletion": true,
                    "endingNaturalness": 0.94,
                    "endingType": "life_lesson",
                    "score": 0.95
                }
            ]
        }"#;

        let result =
            parse_candidate_json(hindi_v10_json, 15.0).expect("Failed to parse Hindi v10 JSON");
        assert_eq!(result.len(), 1);

        let draft = &result[0];
        assert_eq!(draft.hook, "आपने अपनी जिंदगी का सबसे कठिन फैसला कब लिया?");
        assert_eq!(draft.language.as_deref(), Some("hi"));
        assert_eq!(draft.script.as_deref(), Some("devanagari"));
        assert_eq!(draft.hook_topic_clarity, Some(0.96));
        assert_eq!(draft.closure_confidence, Some(0.96));
        assert_eq!(draft.answer_complete, Some(true));

        let score = composite_score(draft);
        assert!(
            score >= 0.90,
            "Hindi candidate should receive strong composite score, got {:.3}",
            score
        );
    }

    #[test]
    fn test_align_hook_to_transcript_words_exact_and_fuzzy() {
        let words = vec![
            TranscriptWord {
                text: "She".to_string(),
                start: 7519.98,
                end: 7520.22,
                speaker: None,
            },
            TranscriptWord {
                text: "looks".to_string(),
                start: 7520.22,
                end: 7520.38,
                speaker: None,
            },
            TranscriptWord {
                text: "at".to_string(),
                start: 7520.38,
                end: 7520.54,
                speaker: None,
            },
            TranscriptWord {
                text: "him".to_string(),
                start: 7520.54,
                end: 7520.70,
                speaker: None,
            },
            TranscriptWord {
                text: "and".to_string(),
                start: 7520.70,
                end: 7520.78,
                speaker: None,
            },
            TranscriptWord {
                text: "says".to_string(),
                start: 7520.78,
                end: 7520.94,
                speaker: None,
            },
            TranscriptWord {
                text: "exactly".to_string(),
                start: 7520.94,
                end: 7521.50,
                speaker: None,
            },
            TranscriptWord {
                text: "the".to_string(),
                start: 7521.50,
                end: 7521.66,
                speaker: None,
            },
            TranscriptWord {
                text: "same".to_string(),
                start: 7521.66,
                end: 7521.90,
                speaker: None,
            },
            TranscriptWord {
                text: "thing.".to_string(),
                start: 7521.90,
                end: 7522.38,
                speaker: None,
            },
            TranscriptWord {
                text: "And".to_string(),
                start: 7522.78,
                end: 7523.34,
                speaker: None,
            },
            TranscriptWord {
                text: "this".to_string(),
                start: 7523.50,
                end: 7523.74,
                speaker: None,
            },
            TranscriptWord {
                text: "is".to_string(),
                start: 7523.74,
                end: 7523.98,
                speaker: None,
            },
            TranscriptWord {
                text: "where".to_string(),
                start: 7523.98,
                end: 7524.46,
                speaker: None,
            },
            TranscriptWord {
                text: "Krishna".to_string(),
                start: 7524.46,
                end: 7524.94,
                speaker: None,
            },
            TranscriptWord {
                text: "teaches".to_string(),
                start: 7524.94,
                end: 7525.34,
                speaker: None,
            },
            TranscriptWord {
                text: "us".to_string(),
                start: 7525.34,
                end: 7525.50,
                speaker: None,
            },
        ];

        let hook = "And this is where Krishna teaches us";
        let res = align_hook_to_transcript_words(hook, 7522.0, 7580.0, &words);
        assert!(res.is_some());
        let (s, _e, conf, s_idx, _e_idx) = res.unwrap();
        assert_eq!(s_idx, 10, "Should match 'And' at index 10");
        assert!(
            (s - 7522.78).abs() < 0.01,
            "Start should align to 7522.78s, got {:.2}",
            s
        );
        assert!(conf >= 0.90, "Confidence should be high");
    }

    #[test]
    fn test_evaluate_opening_hook_context_rules() {
        let config = HookPipelineConfig::default();

        // 1. Clean standalone hook
        let clean = evaluate_opening_hook_context(
            "How many people in the world do you completely trust?",
            false,
            &config,
        );
        assert!(clean.is_standalone);
        assert_eq!(clean.penalty, 0.0);
        assert!(!clean.hard_reject);

        // 2. First-person pronoun is valid
        let first_person =
            evaluate_opening_hook_context("I have 4,000 plus active clients.", false, &config);
        assert!(first_person.is_standalone);
        assert_eq!(first_person.penalty, 0.0);

        // 3. Demonstrative with predicate noun is valid
        let demo =
            evaluate_opening_hook_context("These are the three c's you must know.", false, &config);
        assert!(demo.is_standalone);
        assert_eq!(demo.penalty, 0.0);

        // 4. Proper noun antecedent in sentence is valid
        let named = evaluate_opening_hook_context(
            "The same Kamsa who had his sister imprisoned...",
            false,
            &config,
        );
        assert!(named.is_standalone);
        assert_eq!(named.penalty, 0.0);

        // 5. Continuation opener -> soft penalty, NO hard reject
        let continuation =
            evaluate_opening_hook_context("Now here is where the mystery is.", false, &config);
        assert!(continuation.has_continuation_opener);
        assert!((continuation.penalty - 0.04).abs() < 0.001);
        assert!(
            !continuation.hard_reject,
            "Continuation openers must NOT be hard rejected"
        );

        // 6. Ungrounded 3rd person pronoun -> hard reject
        let ungrounded = evaluate_opening_hook_context(
            "She looks at him and says exactly the same thing.",
            false,
            &config,
        );
        assert!(ungrounded.has_unresolved_reference);
        assert!(
            ungrounded.hard_reject,
            "Ungrounded third person pronoun must be hard rejected"
        );
        assert_eq!(ungrounded.unresolved_reference.as_deref(), Some("she"));

        // 7. Host question setup allows pronoun
        let qa_allowed =
            evaluate_opening_hook_context("She looks at him and says yes.", true, &config);
        assert!(
            !qa_allowed.has_unresolved_reference,
            "Host question setup must resolve pronoun context"
        );
        assert!(!qa_allowed.hard_reject);
    }

    #[test]
    fn test_window_coverage_duration_scaling_zero_uncovered_seconds() {
        let test_cases = vec![
            (600.0, 600.0, 120.0, 1),   // 10 min -> 1 window
            (1800.0, 480.0, 90.0, 5),   // 30 min (medium) -> 5 windows
            (3600.0, 600.0, 120.0, 8),  // 60 min (long) -> 8 windows
            (8538.0, 600.0, 120.0, 18), // 142.3 min (extra-long) -> 18 windows
        ];

        for (dur, win_sz, overlap, min_expected_wins) in test_cases {
            let windows = calculate_sliding_coverage_windows(dur, win_sz, overlap);
            assert!(
                windows.len() >= min_expected_wins,
                "Duration {:.1}s should produce at least {} windows, got {}",
                dur,
                min_expected_wins,
                windows.len()
            );

            assert_eq!(windows[0].0, 0.0, "First window must start at 0.0");
            assert_eq!(
                windows.last().unwrap().1,
                dur,
                "Last window must end at exact duration"
            );

            for i in 1..windows.len() {
                let prev_end = windows[i - 1].1;
                let curr_start = windows[i].0;
                assert!(
                    curr_start <= prev_end,
                    "Gap detected in duration {:.1}s between window {} [{:.1},{:.1}] and {} [{:.1},{:.1}]",
                    dur, i - 1, windows[i - 1].0, prev_end, i, curr_start, windows[i].1
                );
            }
        }
    }

    #[test]
    fn test_validate_window_candidate_bounds() {
        let valid_cand = CandidateDraft {
            start: 620.0,
            end: 665.0,
            ..Default::default()
        };
        assert!(validate_window_candidate_bounds(
            &valid_cand,
            600.0,
            1200.0,
            3600.0
        ));

        let out_of_bounds_cand = CandidateDraft {
            start: 120.0,
            end: 165.0,
            ..Default::default()
        };
        assert!(!validate_window_candidate_bounds(
            &out_of_bounds_cand,
            600.0,
            1200.0,
            3600.0
        ));

        let negative_dur_cand = CandidateDraft {
            start: 700.0,
            end: 650.0,
            ..Default::default()
        };
        assert!(!validate_window_candidate_bounds(
            &negative_dur_cand,
            600.0,
            1200.0,
            3600.0
        ));
    }

    #[test]
    fn test_deduplicate_window_candidates_overlap() {
        let c1 = CandidateDraft {
            start: 590.0,
            end: 640.0,
            score: 0.92,
            hook: "Identical viral moment".to_string(),
            ..Default::default()
        };
        let c2 = CandidateDraft {
            start: 592.0,
            end: 642.0,
            score: 0.85,
            hook: "Identical viral moment".to_string(),
            ..Default::default()
        };
        let c3 = CandidateDraft {
            start: 1200.0,
            end: 1250.0,
            score: 0.88,
            hook: "Distinct moment".to_string(),
            ..Default::default()
        };

        let deduped = deduplicate_window_candidates(vec![c1, c2, c3]);
        assert_eq!(
            deduped.len(),
            2,
            "Duplicate boundary candidate should be merged"
        );
        assert_eq!(
            deduped[0].score, 0.92,
            "Higher score candidate should be retained"
        );
    }

    #[test]
    fn test_adaptive_temporal_bins_selects_highest_score_per_bin() {
        // Create 2 bins across 1200s (bin size = 400s)
        // Bin 0: [0, 400] with high (0.95) and low (0.70)
        let b0_high = CandidateDraft {
            start: 100.0,
            end: 140.0,
            score: 0.95,
            hook_topic_clarity: Some(0.95),
            hook_curiosity: Some(0.95),
            closure_confidence: Some(0.95),
            ..Default::default()
        };
        let b0_low = CandidateDraft {
            start: 200.0,
            end: 240.0,
            score: 0.70,
            hook_topic_clarity: Some(0.70),
            hook_curiosity: Some(0.70),
            closure_confidence: Some(0.70),
            ..Default::default()
        };

        // Bin 1: [400, 800] with high (0.91) and low (0.68)
        let b1_high = CandidateDraft {
            start: 500.0,
            end: 540.0,
            score: 0.91,
            hook_topic_clarity: Some(0.91),
            hook_curiosity: Some(0.91),
            closure_confidence: Some(0.91),
            ..Default::default()
        };
        let b1_low = CandidateDraft {
            start: 600.0,
            end: 640.0,
            score: 0.68,
            hook_topic_clarity: Some(0.68),
            hook_curiosity: Some(0.68),
            closure_confidence: Some(0.68),
            ..Default::default()
        };

        // Bin 2: [800, 1200] with high (0.89)
        let b2_high = CandidateDraft {
            start: 900.0,
            end: 940.0,
            score: 0.89,
            hook_topic_clarity: Some(0.89),
            hook_curiosity: Some(0.89),
            closure_confidence: Some(0.89),
            ..Default::default()
        };

        let raw = vec![b0_high, b0_low, b1_high, b1_low, b2_high];
        let balanced = deduplicate_and_balance_candidates(raw, 1200.0, 3);

        assert_eq!(balanced.len(), 3);
        // Verify that the top score from each bin was selected (0.95, 0.91, 0.89), NOT the low scores (0.70, 0.68)
        let starts: Vec<f64> = balanced.iter().map(|c| c.start).collect();
        assert!(
            starts.contains(&100.0),
            "Must contain bin 0 high candidate (start 100.0)"
        );
        assert!(
            starts.contains(&500.0),
            "Must contain bin 1 high candidate (start 500.0)"
        );
        assert!(
            starts.contains(&900.0),
            "Must contain bin 2 high candidate (start 900.0)"
        );
    }

    #[test]
    fn test_build_window_semantic_prompt_absolute_timestamps() {
        let prompt = build_window_semantic_prompt(
            "Sample transcript",
            "ENGLISH",
            "latin",
            4800.0,
            5400.0,
            8,
            12,
            2,
            4,
        );

        assert!(prompt.contains("TIMELINE COVERAGE CONTEXT (WINDOW 9 OF 12)"));
        assert!(prompt.contains("[4800.0s to 5400.0s]"));
        assert!(prompt.contains(
            "CRITICAL GROUNDING INVARIANT: All timestamps in this transcript are ABSOLUTE seconds"
        ));
        assert!(prompt.contains("Generate 2 to 4 top viral candidates"));
    }

    #[test]
    fn test_align_payoff_to_transcript_words_exact_match() {
        let words = vec![
            TranscriptWord {
                start: 10.0,
                end: 10.4,
                text: "any".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 10.4,
                end: 10.8,
                text: "place".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 10.8,
                end: 11.0,
                text: "I".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 11.0,
                end: 11.5,
                text: "play,".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 11.8,
                end: 12.0,
                text: "I".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 12.0,
                end: 12.4,
                text: "show".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 12.4,
                end: 12.7,
                text: "my".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 12.7,
                end: 13.2,
                text: "level.".into(),
                speaker: None,
            },
        ];

        let result =
            align_payoff_to_transcript_words("any place I play I show my level", 5.0, 13.0, &words);
        assert!(result.is_some());
        let (s, e, conf, s_idx, e_idx) = result.unwrap();
        assert_eq!(s, 10.0);
        assert_eq!(e, 13.2);
        assert!(conf >= 0.80);
        assert_eq!(s_idx, 0);
        assert_eq!(e_idx, 7);
    }

    #[test]
    fn test_align_payoff_to_transcript_words_fallback() {
        let words = vec![
            TranscriptWord {
                start: 10.0,
                end: 10.5,
                text: "hello".into(),
                speaker: None,
            },
            TranscriptWord {
                start: 10.5,
                end: 11.0,
                text: "world".into(),
                speaker: None,
            },
        ];

        let result =
            align_payoff_to_transcript_words("something completely unmatched", 5.0, 10.9, &words);
        assert!(
            result.is_none(),
            "Unmatched payoff must return None and not invent a fallback"
        );
    }

    #[test]
    fn test_build_window_scoring_prompt_no_timestamp_contract() {
        let prompt = build_window_scoring_prompt(
            "Host: What was your breakthrough moment?\nGuest: It happened when I stopped doubting myself.",
            "ENGLISH",
            "latin",
            120.0,
            180.0,
            2,
            5,
        );

        assert!(prompt.contains("REZE WINDOW SCORING TASK"));
        assert!(prompt.contains("[120.0s to 180.0s]"));
        assert!(prompt.contains("STRICT GROUNDING & SCORING CONTRACT:"));
        assert!(prompt.contains("DO NOT emit timestamps"));
        assert!(prompt.contains("DO NOT emit fields named start, end, hookStart, hookEnd, payoffStart, payoffEnd, clip_start, or clip_end."));
        assert!(prompt.contains("The Rust engine computes all boundaries deterministically. Your sole job is quality and relevance scoring."));
        assert!(prompt.contains("highlight_score"));
        assert!(prompt.contains("hook_relevance"));
        assert!(prompt.contains("narrative_completeness"));
        assert!(prompt.contains("payoff_presence"));
        assert!(prompt.contains("engagement_signal"));
    }

    #[test]
    fn test_parse_window_score_json_valid() {
        let json_str = r#"{
            "highlight_score": 0.88,
            "hook_relevance": 0.92,
            "narrative_completeness": 0.85,
            "payoff_presence": 0.80,
            "engagement_signal": 0.90,
            "reasoning": "Very strong emotional turning point with clear resolution."
        }"#;

        let result = parse_window_score_json(json_str, 3, 180.0, 240.0, "openai", "gpt-4o-mini")
            .expect("Should parse valid window score JSON");

        assert_eq!(result.window_idx, 3);
        assert_eq!(result.window_start, 180.0);
        assert_eq!(result.window_end, 240.0);
        assert!((result.raw_score - 0.88).abs() < 1e-6);
        assert_eq!(result.hook_relevance, Some(0.92));
        assert_eq!(result.narrative_completeness, Some(0.85));
        assert_eq!(result.payoff_presence, Some(0.80));
        assert_eq!(result.engagement_signal, Some(0.90));
        assert_eq!(result.provider, "openai");
        assert_eq!(result.model, "gpt-4o-mini");
    }

    #[test]
    fn test_parse_window_score_json_with_think_tags_and_fences() {
        let raw_llm = r#"<think>
Evaluating the segment:
The guest discusses their hardest decision.
Score should be high, around 0.85.
</think>
```json
{
    "raw_score": 0.85,
    "hook_relevance": 0.90,
    "narrative_completeness": 0.80,
    "payoff_presence": 0.85,
    "engagement_signal": 0.88,
    "reasoning": "Compelling story segment with good payoff."
}
```"#;

        let result = parse_window_score_json(raw_llm, 0, 0.0, 60.0, "deepseek", "deepseek-r1")
            .expect("Should parse markdown fences and think tags");

        assert_eq!(result.window_idx, 0);
        assert!((result.raw_score - 0.85).abs() < 1e-6);
        assert_eq!(result.hook_relevance, Some(0.90));
        assert_eq!(result.narrative_completeness, Some(0.80));
        assert_eq!(result.payoff_presence, Some(0.85));
        assert_eq!(result.engagement_signal, Some(0.88));
        assert_eq!(result.provider, "deepseek");
        assert_eq!(result.model, "deepseek-r1");
    }

    #[test]
    fn test_parse_window_score_json_ignores_timestamps_with_warning() {
        let rogue_json = r#"{
            "highlight_score": 0.78,
            "start": 12.5,
            "end": 45.0,
            "hookStart": 12.5,
            "hookEnd": 16.0,
            "payoffStart": 40.0,
            "payoffEnd": 45.0,
            "hook_relevance": 0.82,
            "narrative_completeness": 0.75,
            "payoff_presence": 0.70,
            "engagement_signal": 0.80,
            "reasoning": "Solid moment despite timestamps."
        }"#;

        let result =
            parse_window_score_json(rogue_json, 1, 60.0, 120.0, "anthropic", "claude-3-5-sonnet")
                .expect("Should parse rogue JSON and ignore timestamps");

        assert_eq!(result.window_idx, 1);
        assert_eq!(result.window_start, 60.0);
        assert_eq!(result.window_end, 120.0);
        assert!((result.raw_score - 0.78).abs() < 1e-6);
        assert_eq!(result.hook_relevance, Some(0.82));
    }

    #[test]
    fn test_parse_window_score_json_malformed_returns_error() {
        let invalid_outputs = [
            "",
            "I cannot evaluate this segment because it has no substance.",
            "{ \"unrelated\": \"value\" }",
            "```json\n{ not valid json }\n```",
        ];

        for output in &invalid_outputs {
            let res = parse_window_score_json(output, 0, 0.0, 60.0, "openai", "gpt-4o");
            assert!(
                res.is_err(),
                "Expected error for invalid output: {}",
                output
            );
        }
    }

    #[test]
    fn test_normalize_provider_scores_small_sample_identity() {
        let mut scores = vec![
            WindowScoreResult {
                window_idx: 0,
                raw_score: 0.35,
                provider: "claude".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 1,
                raw_score: 0.85,
                provider: "claude".to_string(),
                ..Default::default()
            },
        ];

        // min_samples_for_zscore = 3, but we only have 2 samples
        normalize_provider_scores(&mut scores, 3);

        assert!((scores[0].raw_score - 0.35).abs() < 1e-6);
        assert!((scores[1].raw_score - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_provider_scores_zero_variance() {
        let mut scores = vec![
            WindowScoreResult {
                window_idx: 0,
                raw_score: 0.70,
                provider: "gemini".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 1,
                raw_score: 0.70,
                provider: "gemini".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 2,
                raw_score: 0.70,
                provider: "gemini".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 3,
                raw_score: 0.70,
                provider: "gemini".to_string(),
                ..Default::default()
            },
        ];

        normalize_provider_scores(&mut scores, 3);

        for s in &scores {
            assert!(!s.raw_score.is_nan(), "Score should not be NaN");
            assert!(!s.raw_score.is_infinite(), "Score should not be infinite");
            assert!(
                (s.raw_score - 0.70).abs() < 1e-6,
                "Score should remain clamped original value"
            );
        }
    }

    #[test]
    fn test_normalize_provider_scores_zscore_sigmoid() {
        let mut scores = vec![
            WindowScoreResult {
                window_idx: 0,
                raw_score: 0.10,
                provider: "openai".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 1,
                raw_score: 0.30,
                provider: "openai".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 2,
                raw_score: 0.50,
                provider: "openai".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 3,
                raw_score: 0.70,
                provider: "openai".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 4,
                raw_score: 0.90,
                provider: "openai".to_string(),
                ..Default::default()
            },
        ];

        normalize_provider_scores(&mut scores, 3);

        // Mean was 0.50, which corresponds to z=0, so sigmoid(0) == 0.50
        assert!(
            (scores[2].raw_score - 0.50).abs() < 1e-6,
            "Mean score must map to 0.50"
        );

        // Relative ordering strictly maintained
        for i in 0..scores.len() - 1 {
            assert!(
                scores[i].raw_score < scores[i + 1].raw_score,
                "Relative ordering must be strictly preserved: {} < {}",
                scores[i].raw_score,
                scores[i + 1].raw_score
            );
        }

        // Bounded in [0.0, 1.0]
        for s in &scores {
            assert!(s.raw_score >= 0.0 && s.raw_score <= 1.0);
        }
    }

    #[test]
    fn test_normalize_provider_scores_multi_provider_independent() {
        let mut scores = vec![
            // Claude: high-scoring distribution, mean = 0.85
            WindowScoreResult {
                window_idx: 0,
                raw_score: 0.80,
                provider: "claude".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 1,
                raw_score: 0.85,
                provider: "claude".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 2,
                raw_score: 0.90,
                provider: "claude".to_string(),
                ..Default::default()
            },
            // Gemini: low-scoring distribution, mean = 0.25
            WindowScoreResult {
                window_idx: 3,
                raw_score: 0.20,
                provider: "gemini".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 4,
                raw_score: 0.25,
                provider: "gemini".to_string(),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 5,
                raw_score: 0.30,
                provider: "gemini".to_string(),
                ..Default::default()
            },
        ];

        normalize_provider_scores(&mut scores, 3);

        // Claude mean (index 1) and Gemini mean (index 4) should both normalize to 0.50
        assert!(
            (scores[1].raw_score - 0.50).abs() < 1e-6,
            "Claude mean should normalize to 0.50"
        );
        assert!(
            (scores[4].raw_score - 0.50).abs() < 1e-6,
            "Gemini mean should normalize to 0.50"
        );

        // Within Claude, ordering preserved
        assert!(scores[0].raw_score < scores[1].raw_score);
        assert!(scores[1].raw_score < scores[2].raw_score);

        // Within Gemini, ordering preserved
        assert!(scores[3].raw_score < scores[4].raw_score);
        assert!(scores[4].raw_score < scores[5].raw_score);
    }

    #[test]
    fn test_gaussian_smooth_identity_single_window() {
        let single = vec![0.75];
        let smoothed = gaussian_smooth_scores(&single, 1.0);
        assert_eq!(smoothed.len(), 1);
        assert!((smoothed[0] - 0.75).abs() < 1e-6);

        let empty: Vec<f64> = vec![];
        let smoothed_empty = gaussian_smooth_scores(&empty, 1.0);
        assert!(smoothed_empty.is_empty());
    }

    #[test]
    fn test_gaussian_smooth_kernel_effect() {
        let values = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let smoothed = gaussian_smooth_scores(&values, 1.0);
        assert_eq!(smoothed.len(), values.len());

        // The peak at index 2 should be smoothed down (< 1.0) but remain the maximum
        assert!(smoothed[2] < 1.0);
        assert!(smoothed[2] > 0.3);

        // Neighboring values should be smoothed up (> 0.0)
        assert!(smoothed[1] > 0.0);
        assert!(smoothed[3] > 0.0);

        // Symmetry: index 1 and 3 should be equal; index 0 and 4 should be equal
        assert!((smoothed[1] - smoothed[3]).abs() < 1e-6);
        assert!((smoothed[0] - smoothed[4]).abs() < 1e-6);

        // Constant array preserves values exactly
        let constant = vec![0.6, 0.6, 0.6, 0.6];
        let smoothed_const = gaussian_smooth_scores(&constant, 1.5);
        for v in smoothed_const {
            assert!((v - 0.6).abs() < 1e-6);
        }
    }

    #[test]
    fn test_otsu_threshold_bimodal() {
        // Distinct bimodal distribution: low cluster [0.1, 0.2], high cluster [0.8, 0.9]
        let values = vec![0.1, 0.15, 0.2, 0.8, 0.85, 0.9];
        let thresh = otsu_threshold(&values);
        // Threshold should cleanly separate the two clusters (between 0.2 and 0.8)
        assert!(
            thresh > 0.2 && thresh < 0.8,
            "Threshold {} must be between clusters",
            thresh
        );
        assert!(
            (thresh - 0.5).abs() < 0.1,
            "Threshold {} should be close to midpoint 0.5",
            thresh
        );
    }

    #[test]
    fn test_otsu_threshold_uniform() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let thresh = otsu_threshold(&values);
        // Uniform distribution between 0.1 and 0.9 should have threshold near median (0.5)
        assert!(
            (thresh - 0.5).abs() < 0.05,
            "Threshold {} should be near median 0.5",
            thresh
        );

        // Identical values edge case
        let identical = vec![0.7, 0.7, 0.7];
        assert!((otsu_threshold(&identical) - 0.7).abs() < 1e-6);

        // Empty edge case
        assert_eq!(otsu_threshold(&[]), 0.0);
    }

    #[test]
    fn test_kadane_spans_contiguous_high() {
        let scores = vec![0.2, 0.8, 0.9, 0.85, 0.1];
        let window_starts = vec![0.0, 15.0, 30.0, 45.0, 60.0];
        let window_ends = vec![30.0, 45.0, 60.0, 75.0, 90.0];
        let threshold = 0.5;

        // Use default valley_drop_ratio (0.70) - no significant valley in this data
        let spans = kadane_score_spans(&scores, &window_starts, &window_ends, threshold, 0.70);
        assert_eq!(spans.len(), 1);
        // Spans from window 1 start (15.0) to window 3 end (75.0)
        assert!((spans[0].0 - 15.0).abs() < 1e-6);
        assert!((spans[0].1 - 75.0).abs() < 1e-6);
    }

    #[test]
    fn test_kadane_spans_separated_peaks() {
        let scores = vec![0.8, 0.85, 0.2, 0.1, 0.15, 0.9, 0.95];
        let window_starts = vec![0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0];
        let window_ends = vec![30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0];
        let threshold = 0.5;

        // Deep valley (0.85 -> 0.1 = 88% drop) exceeds 30% valley_drop_ratio
        let spans = kadane_score_spans(&scores, &window_starts, &window_ends, threshold, 0.70);
        assert_eq!(
            spans.len(),
            2,
            "Expected two separated spans, got {:?}",
            spans
        );

        // First peak: windows 0..=1 -> start 0.0, end 45.0
        assert!((spans[0].0 - 0.0).abs() < 1e-6);
        assert!((spans[0].1 - 45.0).abs() < 1e-6);

        // Second peak: windows 5..=6 -> start 75.0, end 120.0
        assert!((spans[1].0 - 75.0).abs() < 1e-6);
        assert!((spans[1].1 - 120.0).abs() < 1e-6);
    }

    #[test]
    fn test_kadane_spans_shallow_valley_merges_without_reset() {
        // Shallow valley in EXCESS terms that doesn't trigger valley reset (excess drop < 30%)
        // threshold=0.5, scores: peak_excess=0.35 (0.85), valley_excess=0.30 (0.80) = 14% drop < 30%
        let scores = vec![0.8, 0.85, 0.80, 0.82, 0.85, 0.9, 0.95];
        let window_starts = vec![0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0];
        let window_ends = vec![30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0];
        let threshold = 0.5;

        // With valley_drop_ratio=0.70, excess drop from 0.35 to 0.30 is 14% < 30% -> NO reset
        let spans = kadane_score_spans(&scores, &window_starts, &window_ends, threshold, 0.70);
        assert_eq!(
            spans.len(),
            1,
            "Shallow valley (excess drop 14%) should merge into one span with default ratio"
        );

        // With valley_drop_ratio=0.90, excess drop from 0.35 to 0.30 is 14% > 10% -> YES reset
        let spans_strict =
            kadane_score_spans(&scores, &window_starts, &window_ends, threshold, 0.90);
        assert_eq!(
            spans_strict.len(),
            2,
            "Strict ratio (0.90) should separate at shallow valley (excess drop 14% > 10%)"
        );
    }

    #[test]
    fn test_aggregate_scores_end_to_end() {
        let window_results = vec![
            WindowScoreResult {
                window_idx: 0,
                window_start: 0.0,
                window_end: 30.0,
                raw_score: 0.2,
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 1,
                window_start: 15.0,
                window_end: 45.0,
                raw_score: 0.85,
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 2,
                window_start: 30.0,
                window_end: 60.0,
                raw_score: 0.90,
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 3,
                window_start: 45.0,
                window_end: 75.0,
                raw_score: 0.15,
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 4,
                window_start: 60.0,
                window_end: 90.0,
                raw_score: 0.88,
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 5,
                window_start: 75.0,
                window_end: 105.0,
                raw_score: 0.92,
                ..Default::default()
            },
        ];

        let spans = aggregate_window_scores_to_spans(&window_results, 0.8, 0.4, 0.9, 0.70);
        assert!(!spans.is_empty(), "Expected spans to be detected");
        // Verify each span has valid positive duration
        for span in &spans {
            assert!(
                span.1 > span.0,
                "Span end must be after span start: {:?}",
                span
            );
            assert!(span.0 >= 0.0 && span.1 <= 105.0);
        }
    }

    #[test]
    fn test_aggregate_scores_empty_input() {
        let empty: Vec<WindowScoreResult> = vec![];
        let spans = aggregate_window_scores_to_spans(&empty, 1.0, 0.5, 1.0, 0.70);
        assert!(spans.is_empty(), "Empty input must return empty spans");
    }

    #[tokio::test]
    async fn test_spans_to_candidate_drafts_basic_conversion() {
        let transcript = NormalizedTranscript {
            language: "en".into(),
            duration: 60.0,
            speakers: vec!["Speaker 1".into()],
            words: vec![
                TranscriptWord {
                    start: 10.0,
                    end: 10.5,
                    text: "Welcome".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 10.6,
                    end: 11.0,
                    text: "to".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 11.1,
                    end: 11.5,
                    text: "the".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 11.6,
                    end: 12.0,
                    text: "podcast".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 12.1,
                    end: 12.5,
                    text: "today".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 12.6,
                    end: 13.0,
                    text: "we".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 13.1,
                    end: 13.5,
                    text: "discuss".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 13.6,
                    end: 14.0,
                    text: "artificial".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 14.1,
                    end: 14.5,
                    text: "intelligence".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 14.6,
                    end: 15.0,
                    text: "and".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 20.0,
                    end: 20.5,
                    text: "the".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 20.6,
                    end: 21.0,
                    text: "future".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 21.1,
                    end: 21.5,
                    text: "of".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 21.6,
                    end: 22.0,
                    text: "humanity".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 22.1,
                    end: 22.5,
                    text: "which".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 22.6,
                    end: 23.0,
                    text: "changes".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 23.1,
                    end: 23.5,
                    text: "everything".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 23.6,
                    end: 24.0,
                    text: "for".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 24.1,
                    end: 24.5,
                    text: "everyone".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 24.6,
                    end: 25.0,
                    text: "forever.".into(),
                    speaker: None,
                },
            ],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        let scores = vec![
            WindowScoreResult {
                window_idx: 0,
                window_start: 0.0,
                window_end: 20.0,
                raw_score: 0.85,
                hook_relevance: Some(0.92),
                narrative_completeness: Some(0.80),
                payoff_presence: Some(0.70),
                engagement_signal: Some(0.88),
                ..Default::default()
            },
            WindowScoreResult {
                window_idx: 1,
                window_start: 15.0,
                window_end: 35.0,
                raw_score: 0.94,
                hook_relevance: Some(0.75),
                narrative_completeness: Some(0.89),
                payoff_presence: Some(0.96),
                engagement_signal: Some(0.91),
                ..Default::default()
            },
        ];

        let spans = vec![(10.0, 26.0)];
        let config = WindowDiscoveryConfig {
            reze_targeted_extraction: false,
            ..Default::default()
        };
        let drafts = spans_to_candidate_drafts(
            &spans,
            &scores,
            &transcript,
            "English",
            "Latin",
            &config,
            None,
            None,
            None,
        )
        .await;

        assert_eq!(drafts.len(), 1, "Expected exactly 1 candidate draft");
        let d = &drafts[0];
        assert_eq!(d.start, 10.0);
        assert_eq!(d.end, 26.0);
        assert!(
            (d.score - 0.94).abs() < 1e-6,
            "Score should be max raw_score: got {}",
            d.score
        );
        assert_eq!(d.hook_score, Some(0.92));
        assert_eq!(d.coherence_score, Some(0.89));
        assert_eq!(d.payoff_score, Some(0.96));
        assert_eq!(d.curiosity_score, Some(0.91));
        assert_eq!(d.language.as_deref(), Some("english"));
        assert_eq!(d.script.as_deref(), Some("latin"));
        assert_eq!(d.semantic_complete, None);
        assert_eq!(d.closure_confidence, None);
        assert_eq!(d.continuation_probability, None);
        assert!(
            d.rationale.contains("REZE"),
            "Rationale should reference REZE: {}",
            d.rationale
        );
        assert!(
            d.rationale.contains("0.94"),
            "Rationale should reference score: {}",
            d.rationale
        );

        assert!(d.hook.starts_with("Welcome to the podcast"));
        assert_eq!(d.hook_start, Some(10.0));
        assert!(d.hook_end.is_some() && d.hook_end.unwrap() <= 15.0);

        assert!(d.payoff_text.is_some());
        let payoff = d.payoff_text.as_ref().unwrap();
        assert!(payoff.ends_with("forever."));
        assert_eq!(d.payoff_end, Some(25.0));
    }

    #[tokio::test]
    async fn test_spans_to_candidate_drafts_filters_degenerate_spans() {
        let transcript = NormalizedTranscript {
            language: "en".into(),
            duration: 60.0,
            speakers: vec![],
            words: vec![
                TranscriptWord {
                    start: 10.0,
                    end: 11.0,
                    text: "word1".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 11.5,
                    end: 12.5,
                    text: "word2".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 13.0,
                    end: 14.0,
                    text: "word3".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 14.5,
                    end: 15.5,
                    text: "word4".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 16.0,
                    end: 17.0,
                    text: "word5".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 17.5,
                    end: 18.5,
                    text: "word6".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 19.0,
                    end: 20.0,
                    text: "word7".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 20.5,
                    end: 21.5,
                    text: "word8".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 22.0,
                    end: 23.0,
                    text: "word9".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 23.5,
                    end: 24.5,
                    text: "word10".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 25.0,
                    end: 26.0,
                    text: "word11".into(),
                    speaker: None,
                },
            ],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        let scores = vec![WindowScoreResult {
            window_idx: 0,
            window_start: 0.0,
            window_end: 30.0,
            raw_score: 0.80,
            ..Default::default()
        }];

        let spans = vec![
            (10.0, 10.0), // Zero length -> discard
            (25.0, 15.0), // Inverted -> discard
            (10.0, 17.9), // 7.9s < 8.0s -> discard
            (10.0, 18.0), // Exactly 8.0s -> keep
            (10.0, 25.0), // 15.0s -> keep
        ];

        let config = WindowDiscoveryConfig {
            reze_targeted_extraction: false,
            ..Default::default()
        };
        let drafts = spans_to_candidate_drafts(
            &spans,
            &scores,
            &transcript,
            "en",
            "latin",
            &config,
            None,
            None,
            None,
        )
        .await;
        assert_eq!(
            drafts.len(),
            2,
            "Expected only non-degenerate spans (>= 8.0s)"
        );
        assert_eq!(drafts[0].start, 10.0);
        assert_eq!(drafts[0].end, 18.0);
        assert_eq!(drafts[1].start, 10.0);
        assert_eq!(drafts[1].end, 25.0);
    }

    #[tokio::test]
    async fn test_spans_to_candidate_drafts_verbatim_payoff_aligns_downstream() {
        let transcript = NormalizedTranscript {
            language: "en".into(),
            duration: 60.0,
            speakers: vec![],
            words: vec![
                TranscriptWord {
                    start: 10.0,
                    end: 10.5,
                    text: "The".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 10.6,
                    end: 11.0,
                    text: "journey".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 11.1,
                    end: 11.5,
                    text: "was".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 11.6,
                    end: 12.0,
                    text: "long".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 12.1,
                    end: 12.5,
                    text: "and".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 12.6,
                    end: 13.0,
                    text: "full".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 13.1,
                    end: 13.5,
                    text: "of".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 13.6,
                    end: 14.0,
                    text: "challenges".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 18.0,
                    end: 18.5,
                    text: "but".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 18.6,
                    end: 19.0,
                    text: "in".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 19.1,
                    end: 19.5,
                    text: "the".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 19.6,
                    end: 20.0,
                    text: "end".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 20.1,
                    end: 20.5,
                    text: "we".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 20.6,
                    end: 21.0,
                    text: "achieved".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 21.1,
                    end: 21.5,
                    text: "complete".into(),
                    speaker: None,
                },
                TranscriptWord {
                    start: 21.6,
                    end: 22.0,
                    text: "victory".into(),
                    speaker: None,
                },
            ],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        let scores = vec![WindowScoreResult {
            window_idx: 0,
            window_start: 10.0,
            window_end: 25.0,
            raw_score: 0.90,
            ..Default::default()
        }];

        let spans = vec![(10.0, 23.0)];
        let config = WindowDiscoveryConfig {
            reze_targeted_extraction: false,
            ..Default::default()
        };
        let drafts = spans_to_candidate_drafts(
            &spans,
            &scores,
            &transcript,
            "en",
            "latin",
            &config,
            None,
            None,
            None,
        )
        .await;

        assert_eq!(drafts.len(), 1);
        let draft = &drafts[0];
        assert!(
            draft.payoff_text.is_some(),
            "CandidateDraft must contain payoff_text"
        );

        let payoff_text = draft.payoff_text.as_ref().unwrap();
        // Downstream alignment call matching lib.rs line 660:
        let aligned = align_payoff_to_transcript_words(
            payoff_text,
            draft.start,
            draft.end,
            &transcript.words,
        );

        assert!(
            aligned.is_some(),
            "align_payoff_to_transcript_words must succeed for verbatim payoff"
        );
        let (_p_start, p_end, conf, _s_idx, _e_idx) = aligned.unwrap();

        assert!(
            conf >= 0.80,
            "Payoff alignment confidence must be >= 0.80, got {}",
            conf
        );
        // Downstream invariant: p_end locks the candidate's authoritative end boundary
        assert_eq!(
            p_end,
            draft.payoff_end.unwrap(),
            "Aligned payoff end must match draft.payoff_end"
        );
    }

    #[test]
    fn test_window_score_cache_insert_and_get() {
        let cache = WindowScoreCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let key = WindowScoreCache::compute_cache_key(
            "test_video_source_1",
            2,
            30.0,
            45.0,
            "This is a transcript excerpt for cache testing.",
            "gemini",
            "gemini-2.5-flash",
            "reze_v1",
        );
        assert!(key.starts_with("reze_score_"));
        if let Some(path) = reze_disk_cache_path(&key) {
            let _ = std::fs::remove_file(path);
        }
        assert!(cache.get(&key).is_none());

        let score_result = WindowScoreResult {
            window_idx: 2,
            window_start: 30.0,
            window_end: 45.0,
            raw_score: 0.88,
            hook_relevance: Some(0.92),
            narrative_completeness: Some(0.85),
            payoff_presence: Some(0.90),
            engagement_signal: Some(0.80),
            provider: "gemini".into(),
            model: "gemini-2.5-flash".into(),
            cache_key: key.clone(),
            prompt_version: "reze_v1".into(),
        };

        cache.insert(key.clone(), score_result.clone());

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());

        let retrieved = cache.get(&key).expect("Cached result should be found");
        assert_eq!(retrieved.window_idx, 2);
        assert_eq!(retrieved.window_start, 30.0);
        assert_eq!(retrieved.window_end, 45.0);
        assert_eq!(retrieved.raw_score, 0.88);
        assert_eq!(retrieved.hook_relevance, Some(0.92));
        assert_eq!(retrieved.narrative_completeness, Some(0.85));
        assert_eq!(retrieved.payoff_presence, Some(0.90));
        assert_eq!(retrieved.engagement_signal, Some(0.80));
        assert_eq!(retrieved.provider, "gemini");
        assert_eq!(retrieved.model, "gemini-2.5-flash");
        assert_eq!(retrieved.cache_key, key);
        assert_eq!(retrieved.prompt_version, "reze_v1");

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_window_score_cache_key_uniqueness() {
        let base_src = "video_source_alpha";
        let base_idx = 3;
        let base_start = 45.0;
        let base_end = 60.0;
        let base_text = "Here is an interesting story told by the guest.";
        let base_provider = "gemini";
        let base_model = "gemini-2.5-flash";
        let base_prompt_version = "reze_v1";

        let base_key = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            base_start,
            base_end,
            base_text,
            base_provider,
            base_model,
            base_prompt_version,
        );
        assert!(base_key.starts_with("reze_score_"));

        // 1. Alter source_identity
        let key_alt_src = WindowScoreCache::compute_cache_key(
            "video_source_beta",
            base_idx,
            base_start,
            base_end,
            base_text,
            base_provider,
            base_model,
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_src,
            "Altering source_identity must produce a distinct key"
        );

        // 2. Alter window_idx
        let key_alt_idx = WindowScoreCache::compute_cache_key(
            base_src,
            4,
            base_start,
            base_end,
            base_text,
            base_provider,
            base_model,
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_idx,
            "Altering window_idx must produce a distinct key"
        );

        // 3. Alter window_start
        let key_alt_start = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            46.0,
            base_end,
            base_text,
            base_provider,
            base_model,
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_start,
            "Altering window_start must produce a distinct key"
        );

        // 4. Alter window_end
        let key_alt_end = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            base_start,
            65.0,
            base_text,
            base_provider,
            base_model,
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_end,
            "Altering window_end must produce a distinct key"
        );

        // 5. Alter transcript_slice_text
        let key_alt_text = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            base_start,
            base_end,
            "Completely different text in window.",
            base_provider,
            base_model,
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_text,
            "Altering transcript_slice_text must produce a distinct key"
        );

        // 6. Alter provider
        let key_alt_provider = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            base_start,
            base_end,
            base_text,
            "openai",
            base_model,
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_provider,
            "Altering provider must produce a distinct key"
        );

        // 7. Alter model
        let key_alt_model = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            base_start,
            base_end,
            base_text,
            base_provider,
            "gpt-4o",
            base_prompt_version,
        );
        assert_ne!(
            base_key, key_alt_model,
            "Altering model must produce a distinct key"
        );

        // 8. Alter prompt_version
        let key_alt_version = WindowScoreCache::compute_cache_key(
            base_src,
            base_idx,
            base_start,
            base_end,
            base_text,
            base_provider,
            base_model,
            "reze_v2",
        );
        assert_ne!(
            base_key, key_alt_version,
            "Altering prompt_version must produce a distinct key"
        );

        // Mutual uniqueness across all 9 variants
        let keys = vec![
            base_key,
            key_alt_src,
            key_alt_idx,
            key_alt_start,
            key_alt_end,
            key_alt_text,
            key_alt_provider,
            key_alt_model,
            key_alt_version,
        ];
        let set: std::collections::HashSet<_> = keys.iter().collect();
        assert_eq!(
            set.len(),
            9,
            "All 9 parameter variants must produce unique keys"
        );
    }

    #[test]
    fn test_window_score_cache_thread_safe_concurrent_access() {
        let cache = WindowScoreCache::new();
        let num_threads = 10;
        let items_per_thread = 50;
        let mut handles = Vec::new();

        for t in 0..num_threads {
            let cache_clone = cache.clone();
            let handle = std::thread::spawn(move || {
                for i in 0..items_per_thread {
                    let idx = t * items_per_thread + i;
                    let key = WindowScoreCache::compute_cache_key(
                        "thread_test_source",
                        idx,
                        idx as f64 * 10.0,
                        idx as f64 * 10.0 + 15.0,
                        &format!("transcript text for item {}", idx),
                        "gemini",
                        "gemini-2.5-flash",
                        "reze_v1",
                    );

                    let score_result = WindowScoreResult {
                        window_idx: idx,
                        window_start: idx as f64 * 10.0,
                        window_end: idx as f64 * 10.0 + 15.0,
                        raw_score: (i as f64) / (items_per_thread as f64),
                        provider: "gemini".into(),
                        model: "gemini-2.5-flash".into(),
                        cache_key: key.clone(),
                        prompt_version: "reze_v1".into(),
                        ..Default::default()
                    };

                    cache_clone.insert(key.clone(), score_result);

                    let retrieved = cache_clone.get(&key);
                    assert!(
                        retrieved.is_some(),
                        "Concurrent read after insert must succeed"
                    );
                    assert_eq!(retrieved.unwrap().window_idx, idx);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Worker thread panicked or deadlocked");
        }

        assert_eq!(cache.len(), num_threads * items_per_thread);
        assert!(!cache.is_empty());

        for t in 0..num_threads {
            for i in 0..items_per_thread {
                let idx = t * items_per_thread + i;
                let key = WindowScoreCache::compute_cache_key(
                    "thread_test_source",
                    idx,
                    idx as f64 * 10.0,
                    idx as f64 * 10.0 + 15.0,
                    &format!("transcript text for item {}", idx),
                    "gemini",
                    "gemini-2.5-flash",
                    "reze_v1",
                );
                let item = cache.get(&key);
                assert!(
                    item.is_some(),
                    "Key for idx {} must be present in main cache",
                    idx
                );
                assert_eq!(item.unwrap().window_idx, idx);
            }
        }
    }

    #[test]
    fn test_global_score_cache_singleton() {
        let cache1 = get_global_score_cache();
        let cache2 = get_global_score_cache();
        assert!(
            std::ptr::eq(cache1, cache2),
            "get_global_score_cache must return the same singleton reference"
        );

        let key = "singleton_probe_key".to_string();
        let score = WindowScoreResult {
            window_idx: 0,
            window_start: 0.0,
            window_end: 30.0,
            raw_score: 0.88,
            provider: "gemini".into(),
            model: "gemini-2.5-flash".into(),
            cache_key: key.clone(),
            prompt_version: "v1.0".into(),
            ..Default::default()
        };
        cache1.insert(key.clone(), score);
        let retrieved = cache2.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().raw_score, 0.88);
    }

    #[test]
    fn test_timestamp_mode_is_default_and_untouched() {
        let config = WindowDiscoveryConfig::default();
        assert_eq!(
            config.discovery_mode,
            DiscoveryMode::TimestampGeneration,
            "Default discovery mode must be TimestampGeneration"
        );
    }

    #[tokio::test]
    async fn test_discover_candidates_reze_mode_selection_and_fallback() {
        let mut config = WindowDiscoveryConfig::default();
        config.discovery_mode = DiscoveryMode::WindowScoring;
        assert_eq!(config.discovery_mode, DiscoveryMode::WindowScoring);

        let transcript = NormalizedTranscript {
            language: "en".to_string(),
            duration: 20.0,
            speakers: vec!["Host".to_string()],
            words: vec![
                TranscriptWord {
                    text: "Welcome".to_string(),
                    start: 0.0,
                    end: 0.5,
                    speaker: Some("Host".to_string()),
                },
                TranscriptWord {
                    text: "to".to_string(),
                    start: 0.5,
                    end: 1.0,
                    speaker: Some("Host".to_string()),
                },
                TranscriptWord {
                    text: "the".to_string(),
                    start: 1.0,
                    end: 1.5,
                    speaker: Some("Host".to_string()),
                },
                TranscriptWord {
                    text: "show".to_string(),
                    start: 1.5,
                    end: 2.0,
                    speaker: Some("Host".to_string()),
                },
            ],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        // 1. With empty cache and dummy provider/api_key, REZE scoring fails and safely falls back
        // to timestamp generation (which attempts provider query and returns Err without panic).
        let res = discover_candidates_full_timeline(
            &transcript,
            "deepseek",
            "dummy_key_for_fallback_test",
            None,
            &config,
        )
        .await;

        assert!(
            res.is_err(),
            "Fallback path returns provider error on invalid key without panicking"
        );

        // 2. Verify direct REZE scoring succeeds when cache is pre-populated with non-degenerate scores
        let mut multi_config = config.clone();
        multi_config.reze_targeted_extraction = false;

        let long_transcript = NormalizedTranscript {
            language: "en".to_string(),
            duration: 240.0,
            speakers: vec!["Host".to_string()],
            words: (0..40)
                .map(|i| TranscriptWord {
                    text: format!("word_{i}"),
                    start: (i as f64) * 6.0,
                    end: (i as f64) * 6.0 + 3.0,
                    speaker: Some("Host".to_string()),
                })
                .collect(),
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        let local_cache = WindowScoreCache::new();
        let src_id = compute_transcript_source_identity(&long_transcript);
        let win_bounds = [(0, 0.0, 75.0, 0.15), (1, 60.0, 135.0, 0.45), (2, 120.0, 195.0, 0.88), (3, 180.0, 240.0, 0.94)];

        for (w_idx, w_start, w_end, w_score) in win_bounds {
            let win_text = format_transcript_window(&long_transcript.segments, &long_transcript.words, w_start, w_end);
            let key = WindowScoreCache::compute_cache_key(
                &src_id,
                w_idx,
                w_start,
                w_end,
                &win_text,
                "deepseek",
                "deepseek-chat",
                "v1.0",
            );
            let score = WindowScoreResult {
                window_idx: w_idx,
                window_start: w_start,
                window_end: w_end,
                raw_score: w_score,
                hook_relevance: Some(w_score * 0.9),
                payoff_presence: Some(w_score * 0.95),
                provider: "deepseek".into(),
                model: "deepseek-chat".into(),
                cache_key: key.clone(),
                prompt_version: "v1.0".into(),
                ..Default::default()
            };
            local_cache.insert(key, score);
        }

        let reze_res = discover_candidates_reze_scoring(
            &long_transcript,
            "deepseek",
            "dummy_key",
            None,
            &multi_config,
            Some(&local_cache),
        )
        .await;

        assert!(
            reze_res.is_ok(),
            "REZE scoring with cache hits must succeed without network: {:?}",
            reze_res.err()
        );
        let drafts = reze_res.unwrap();
        assert!(
            !drafts.is_empty(),
            "Drafts must be produced from cached span"
        );
    }

    #[test]
    fn test_resolve_effective_model_name_nvidia() {
        std::env::remove_var("NVIDIA_REZE_MODEL");

        // 1. Default model resolution for nvidia_diffusiongemma and nvidia alias
        assert_eq!(
            resolve_effective_model_name("nvidia_diffusiongemma", None),
            "google/diffusiongemma-26b-a4b-it"
        );
        assert_eq!(
            resolve_effective_model_name("nvidia", None),
            "google/diffusiongemma-26b-a4b-it"
        );

        // Case and whitespace trimming tolerance
        assert_eq!(
            resolve_effective_model_name("  NVIDIA_DIFFUSIONGEMMA  ", None),
            "google/diffusiongemma-26b-a4b-it"
        );
        assert_eq!(
            resolve_effective_model_name(" NVIDIA ", None),
            "google/diffusiongemma-26b-a4b-it"
        );

        // 2. Passed model_name override takes precedence
        assert_eq!(
            resolve_effective_model_name("nvidia_diffusiongemma", Some("meta/llama-3.1-70b-instruct")),
            "meta/llama-3.1-70b-instruct"
        );
        assert_eq!(
            resolve_effective_model_name("nvidia", Some("custom/override-model")),
            "custom/override-model"
        );
        assert_eq!(
            resolve_effective_model_name("nvidia_diffusiongemma", Some("  custom/trimmed  ")),
            "custom/trimmed"
        );
    }

    #[test]
    fn test_nvidia_reze_cache_key_isolation() {
        let src_id = "test_transcription_source_id_42";
        let window_idx = 3;
        let window_start = 45.0;
        let window_end = 75.0;
        let window_text = "This is a test transcript window text for cache isolation verification.";
        let prompt_version = "reze_v1";

        let nvidia_model = resolve_effective_model_name("nvidia_diffusiongemma", None);
        let openrouter_model = resolve_effective_model_name("openrouter", None);
        let deepseek_model = resolve_effective_model_name("deepseek", None);

        let nvidia_key = WindowScoreCache::compute_cache_key(
            src_id,
            window_idx,
            window_start,
            window_end,
            window_text,
            "nvidia_diffusiongemma",
            &nvidia_model,
            prompt_version,
        );
        let openrouter_key = WindowScoreCache::compute_cache_key(
            src_id,
            window_idx,
            window_start,
            window_end,
            window_text,
            "openrouter",
            &openrouter_model,
            prompt_version,
        );
        let deepseek_key = WindowScoreCache::compute_cache_key(
            src_id,
            window_idx,
            window_start,
            window_end,
            window_text,
            "deepseek",
            &deepseek_model,
            prompt_version,
        );

        // 1. Strict non-equality across providers for identical window inputs
        assert_ne!(
            nvidia_key, openrouter_key,
            "NVIDIA cache key must differ from OpenRouter cache key"
        );
        assert_ne!(
            nvidia_key, deepseek_key,
            "NVIDIA cache key must differ from DeepSeek cache key"
        );
        assert_ne!(
            openrouter_key, deepseek_key,
            "OpenRouter cache key must differ from DeepSeek cache key"
        );

        // Even with identical model string, provider isolation must hold
        let nvidia_forced_key = WindowScoreCache::compute_cache_key(
            src_id,
            window_idx,
            window_start,
            window_end,
            window_text,
            "nvidia_diffusiongemma",
            "same-model-string",
            prompt_version,
        );
        let openrouter_forced_key = WindowScoreCache::compute_cache_key(
            src_id,
            window_idx,
            window_start,
            window_end,
            window_text,
            "openrouter",
            "same-model-string",
            prompt_version,
        );
        assert_ne!(
            nvidia_forced_key, openrouter_forced_key,
            "Different providers with identical model string must produce distinct cache keys"
        );

        // 2. Verify on-disk file key format (reze_score_<16-hex-digit-hash>)
        assert!(
            nvidia_key.starts_with("reze_score_"),
            "Cache key must start with 'reze_score_'"
        );
        assert_eq!(
            nvidia_key.len(),
            "reze_score_".len() + 16,
            "Cache key must be 'reze_score_' followed by 16 hex characters"
        );
        let hex_suffix = &nvidia_key["reze_score_".len()..];
        assert!(
            hex_suffix.chars().all(|c| c.is_ascii_hexdigit()),
            "Key suffix must be lowercase hex string"
        );
    }

    #[test]
    fn test_nvidia_diffusiongemma_payload_construction() {
        let model = "google/diffusiongemma-26b-a4b-it";
        let prompt = "Analyze window virality and closed loops";
        let payload = build_nvidia_diffusiongemma_payload(model, prompt);

        assert_eq!(payload["model"], "google/diffusiongemma-26b-a4b-it");
        assert_eq!(payload["messages"].as_array().unwrap().len(), 1);
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(payload["messages"][0]["content"], prompt);
        assert_eq!(payload["max_tokens"], 4096);
        assert_eq!(payload["temperature"], 1.0);
        assert_eq!(payload["top_p"], 0.95);
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[tokio::test]
    async fn test_reze_scoring_fallback_to_deepseek_on_nvidia_failure() {
        let mut config = WindowDiscoveryConfig::default();
        config.discovery_mode = DiscoveryMode::WindowScoring;

        let transcript = NormalizedTranscript {
            language: "en".to_string(),
            duration: 20.0,
            speakers: vec!["Host".to_string()],
            words: vec![
                TranscriptWord {
                    text: "Welcome".to_string(),
                    start: 0.0,
                    end: 0.5,
                    speaker: Some("Host".to_string()),
                },
                TranscriptWord {
                    text: "to".to_string(),
                    start: 0.5,
                    end: 1.0,
                    speaker: Some("Host".to_string()),
                },
                TranscriptWord {
                    text: "the".to_string(),
                    start: 1.0,
                    end: 1.5,
                    speaker: Some("Host".to_string()),
                },
                TranscriptWord {
                    text: "show".to_string(),
                    start: 1.5,
                    end: 2.0,
                    speaker: Some("Host".to_string()),
                },
            ],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        // Call discover_candidates_full_timeline with WindowScoring, provider "nvidia_diffusiongemma"
        // and dummy key. Direct REZE scoring fails on invalid key/network, triggering fallback
        // to timestamp generation (which attempts provider query and returns Err without panic).
        let res = discover_candidates_full_timeline(
            &transcript,
            "nvidia_diffusiongemma",
            "dummy_nvidia_key",
            None,
            &config,
        )
        .await;

        assert!(
            res.is_err(),
            "Fallback to timestamp generation on NVIDIA failure should return Err without panicking"
        );

        // Also verify the "nvidia" provider alias triggers the same safe fallback
        let res_alias = discover_candidates_full_timeline(
            &transcript,
            "nvidia",
            "dummy_nvidia_key",
            None,
            &config,
        )
        .await;

        assert!(
            res_alias.is_err(),
            "Fallback to timestamp generation on NVIDIA alias failure should return Err without panicking"
        );
    }

    #[tokio::test]
    async fn test_reze_discrimination_guard_fires_on_degenerate_scores() {
        let transcript = NormalizedTranscript {
            language: "en".to_string(),
            duration: 240.0,
            speakers: vec!["Host".to_string()],
            words: (0..40)
                .map(|i| TranscriptWord {
                    text: format!("word_{i}"),
                    start: (i as f64) * 6.0,
                    end: (i as f64) * 6.0 + 3.0,
                    speaker: Some("Host".to_string()),
                })
                .collect(),
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        let mut config = WindowDiscoveryConfig::default();
        config.discovery_mode = DiscoveryMode::WindowScoring;
        config.reze_targeted_extraction = false;

        let local_cache = WindowScoreCache::new();
        let src_id = compute_transcript_source_identity(&transcript);

        // Pre-populate with degenerate flat scores: all 0.70 (distinct = 1, std = 0.0, range = 0.0)
        for (w_idx, w_start, w_end) in [(0, 0.0, 75.0), (1, 60.0, 135.0), (2, 120.0, 195.0), (3, 180.0, 240.0)] {
            let win_text = format_transcript_window(&transcript.segments, &transcript.words, w_start, w_end);
            let key = WindowScoreCache::compute_cache_key(
                &src_id,
                w_idx,
                w_start,
                w_end,
                &win_text,
                "deepseek",
                "deepseek-chat",
                "v1.0",
            );
            let score = WindowScoreResult {
                window_idx: w_idx,
                window_start: w_start,
                window_end: w_end,
                raw_score: 0.70,
                provider: "deepseek".into(),
                model: "deepseek-chat".into(),
                cache_key: key.clone(),
                prompt_version: "v1.0".into(),
                ..Default::default()
            };
            local_cache.insert(key, score);
        }

        let res = discover_candidates_reze_scoring(
            &transcript,
            "deepseek",
            "dummy_key",
            None,
            &config,
            Some(&local_cache),
        )
        .await;

        assert!(res.is_err(), "Degenerate flat scores must trigger discrimination guard Err");
        let err_msg = res.err().unwrap().to_string();
        assert!(
            err_msg.contains("degenerate signal"),
            "Error must indicate degenerate signal: {}",
            err_msg
        );
    }

    #[test]
    fn test_new_tau_formula_selects_otsu_term_over_floor() {
        let window_results = vec![
            WindowScoreResult { window_idx: 0, window_start: 0.0, window_end: 30.0, raw_score: 0.30, ..Default::default() },
            WindowScoreResult { window_idx: 1, window_start: 30.0, window_end: 60.0, raw_score: 0.75, ..Default::default() },
            WindowScoreResult { window_idx: 2, window_start: 60.0, window_end: 90.0, raw_score: 0.95, ..Default::default() },
            WindowScoreResult { window_idx: 3, window_start: 90.0, window_end: 120.0, raw_score: 0.90, ..Default::default() },
        ];

        let spans = aggregate_window_scores_to_spans(&window_results, 0.5, 0.35, 1.0, 0.70);
        assert!(!spans.is_empty(), "Spans must be detected under adaptive Otsu tau");
        assert!(spans[0].0 >= 30.0 && spans[0].1 <= 120.0);
    }

    #[tokio::test]
    async fn test_reze_long_form_restriction_redirects_short_videos() {
        let short_transcript = NormalizedTranscript {
            language: "en".to_string(),
            duration: 600.0, // 10 min <= 720.0s threshold
            speakers: vec!["Host".to_string()],
            words: vec![],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        let mut config = WindowDiscoveryConfig::default();
        config.discovery_mode = DiscoveryMode::WindowScoring;
        config.reze_long_form_only = true;

        let res = discover_candidates_full_timeline(
            &short_transcript,
            "deepseek",
            "dummy_key",
            None,
            &config,
        )
        .await;

        assert!(res.is_err(), "Short video redirects to timestamp generation which fails on dummy key");
    }

    #[test]
    fn test_second_pass_evaluator_populates_closure_fields_from_json() {
        let json_payload = r#"{
            "clip_score": 0.91,
            "hook_first_3s": 0.95,
            "payoff_lands": 0.88,
            "standalone_works": true,
            "endpoint_state": "PayoffComplete",
            "mid_thought_risk": 0.04,
            "answer_complete": true,
            "story_completeness": true,
            "reasoning": "Unforgettable viral lesson with decisive payoff punchline."
        }"#;

        let eval = parse_candidate_evaluation_json(json_payload)
            .expect("must parse candidate evaluation json");

        assert!((eval.clip_score - 0.91).abs() < 1e-4);
        assert!((eval.hook_first_3s - 0.95).abs() < 1e-4);
        assert!((eval.payoff_lands - 0.88).abs() < 1e-4);
        assert_eq!(eval.standalone_works, true);
        assert_eq!(eval.endpoint_state, Some(crate::models::EndpointState::PayoffComplete));
        assert!((eval.mid_thought_risk - 0.04).abs() < 1e-4);
        assert_eq!(eval.answer_complete, true);
        assert_eq!(eval.story_completeness, true);
        assert_eq!(eval.reasoning, "Unforgettable viral lesson with decisive payoff punchline.");
    }

    #[test]
    fn test_targeted_extraction_spans_merging_and_bounds_validation() {
        // 1. Spans overlapping by > 20% must merge
        let overlapping = vec![(0.0, 675.0), (660.0, 735.0)];
        let merged = merge_overlapping_spans(&overlapping, 0.20);
        assert_eq!(merged.len(), 1, "Spans overlapping > 20% must merge into 1 region");
        assert_eq!(merged[0].0, 0.0);
        assert_eq!(merged[0].1, 735.0);

        // 2. Disjoint spans must stay separate
        let disjoint = vec![(10.0, 50.0), (100.0, 160.0)];
        let merged_disjoint = merge_overlapping_spans(&disjoint, 0.20);
        assert_eq!(merged_disjoint.len(), 2, "Disjoint spans must remain separate");

        // 3. validate_window_candidate_bounds drops candidate outside parent region
        let in_bounds = CandidateDraft { start: 20.0, end: 45.0, ..Default::default() };
        let out_bounds = CandidateDraft { start: 120.0, end: 150.0, ..Default::default() };

        assert!(validate_window_candidate_bounds(&in_bounds, 10.0, 50.0, 200.0));
        assert!(!validate_window_candidate_bounds(&out_bounds, 10.0, 50.0, 200.0));
    }
}
