# Original User Request

## Initial Request — 2026-08-30T13:04:48Z

# AutoShorts — v10.x Hook Quality + True Semantic Closure Engine

Upgrade the **hook-selection, conversation-boundary, and semantic-closure logic** in AutoShorts so that every generated Short starts with a hook that provides immediate topic clarity and curiosity, and ends only when the speaker has genuinely completed their thought, answer, story, punchline, or lesson. The framing engine, captions, transcription, frontend, and all non-hook pipeline stages must remain untouched.

Working directory: `d:/College/Autoshorts 3.0`
Integrity mode: **demo** (spaCy/NLTK for continuation analysis allowed; reading tests before implementing allowed)

---

## Full Pipeline Path (Audit Required)

The team must audit and modify this exact path — no shortcuts:

```
LLM candidate (llm.rs)
  → CandidateDraft { start, end, continuation_probability, ... }  (models.rs)
  → snap_to_semantic_boundaries(words, start, end, duration)       (lib.rs:634)
  → draft.start / draft.end  ← FINAL TIMESTAMPS
  → render_flat_clip_for_candidate()                               (lib.rs:802)
  → FFmpeg render
```

The LLM recommends. `snap_to_semantic_boundaries` in `lib.rs` **enforces deterministically**.

---

## Context

Relevant source files:
- `autoshorts/src-tauri/src/llm.rs` — LLM prompt template, `CandidateDraft` parsing, re-ranking
- `autoshorts/src-tauri/src/lib.rs` — `snap_to_semantic_boundaries` (line 634), candidate pipeline (lines 521–543), render command (line 802)
- `autoshorts/src-tauri/src/models.rs` — `CandidateDraft` struct (line 159), `TranscriptWord` (line 138)
- `autoshorts/src-tauri/scripts/multimodal_hook_analyzer.py` — **READ-ONLY; do not modify**

Files that **must not be modified**:
- `speaker_tracker.py`, `media.rs` — framing/crop/scale/trajectory (v12.0, complete)
- `transcription.rs`, `youtube.rs`
- `db.rs` — do not add new DB columns or change schema
- Frontend (`autoshorts/src/`)

Existing test baselines (zero regressions):
- `python -m unittest test_active_speaker_suite.py` → 28/28
- `python -m unittest test_multimodal_hook_suite.py` → 6/6
- `cargo test` → 46/46
- `npm run build` → exit 0

Source videos for regression (use transcripts, not rendered clips):
- `d:/College/Autoshorts 3.0/Beat Emotional Fatigue_ Better Sleep & Clearer Mind.mp4`
- `d:/College/Autoshorts 3.0/Messi vs Ronaldo Fans: The Psychology Explained [rssDTc086bk].mp4`
- `d:/College/Autoshorts 3.0/clip-*.mp4`

---

## Requirements

### R1. Hook Quality Engine

**A. Topic Clarity + Curiosity Gate**
Every candidate opening must be evaluated against four failure mechanisms:
1. **DELAY** — generic intro, filler, or vague suspense before topic arrives (penalise)
2. **CONFUSION** — unexplained pronouns, dangling references, fragment mid-sentence, missing prior context (penalise or reject)
3. **IRRELEVANCE** — no discernible reason the viewer should care (penalise or reject)
4. **DISINTEREST** — no curiosity loop, no contrast, no question, no surprise (penalise)

**B. Segment-First Architecture**
The LLM must be instructed to:
1. Segment the transcript into topical conversation units
2. Identify question/answer relationships and hook/development/payoff structure within each unit
3. Evaluate only the opening 1–3 seconds of each unit for hook quality
4. Reject units whose first 1–3 seconds fail the topic clarity + curiosity gate
5. Never treat an interesting standalone sentence as equivalent to an effective short-form hook

**C. Interviewer Questions as First-Class Hooks**
A strong question followed by a resolved answer must be preferred over a guest statement requiring missing context.

**D. Hook Repair Strategy**
If a promising segment's opening is weak, attempt:
- Move start backward to the containing interviewer question
- Move start backward to the sentence that introduces the topic
- Move start forward past weak filler to a stronger standalone statement
- Reject if no valid opening exists
NEVER fabricate narration, reorder words, or paraphrase the speaker.

---

### R2. Semantic Closure Engine (LLM Layer)

Every candidate endpoint must carry a `continuation_probability` score. Breath/micro-pause alone must never be accepted as closure. Look-ahead must extend until answer/story/punchline/lesson resolves, or an unrelated new topical unit begins.

Valid closure requires all of:
1. Grammatical completion of the final sentence
2. Semantic completion of the current thought
3. Narrative completion (Q&A arc, story arc, lesson arc where applicable)
4. No unresolved continuation dependency
5. No obvious imminent payoff within look-ahead window
6. Natural pause OR speaker turn
7. No dangling conjunction in the final spoken word
8. No unfinished list or causal chain

---

### R3. New Scoring Dimensions in CandidateDraft (Backwards-Compatible)

Add to `CandidateDraft` in `models.rs`. All fields `Option<T>` with `#[serde(default)]`:

```rust
// Hook dimensions (LLM-supplied)
pub hook_topic_clarity:       Option<f64>,
pub hook_curiosity:           Option<f64>,
pub hook_relevance:           Option<f64>,
pub hook_contrast:            Option<f64>,
pub hook_confusion_penalty:   Option<f64>,
pub hook_delay_penalty:       Option<f64>,
pub hook_irrelevance_penalty: Option<f64>,
pub hook_disinterest_penalty: Option<f64>,

// Closure dimensions (LLM-supplied, enforced in Rust)
pub continuation_probability: Option<f64>,
pub breath_pause_risk:        Option<f64>,
pub mid_thought_risk:         Option<f64>,
pub closure_confidence:       Option<f64>,
pub answer_complete:          Option<bool>,  // bool sibling (existing answer_completeness is f64)
pub story_completeness:       Option<bool>,
pub payoff_completion:        Option<bool>,
pub ending_naturalness:       Option<f64>,
pub ending_type:              Option<String>,
```

Ending type values: `sentence_completion | answer_completion | story_resolution | punchline | life_lesson | speaker_turn | unrelated_topic`

---

### R4. Deterministic Closure Enforcement in `snap_to_semantic_boundaries` ← CRITICAL

This is the safety layer. The LLM recommends; this function enforces.

**Signature change:**
```rust
pub fn snap_to_semantic_boundaries(
    words: &[TranscriptWord],
    raw_start: f64,
    raw_end: f64,
    video_duration: f64,
    closure: Option<&ClosureSignals>,   // NEW — None = backwards compat
) -> (f64, f64)
```

`ClosureSignals` helper struct:
```rust
pub struct ClosureSignals {
    pub continuation_probability: Option<f64>,
    pub breath_pause_risk:        Option<f64>,
    pub mid_thought_risk:         Option<f64>,
    pub answer_complete:          Option<bool>,
    pub story_completeness:       Option<bool>,
    pub payoff_completion:        Option<bool>,
    pub closure_confidence:       Option<f64>,
    pub ending_type:              Option<String>,
}
```

Implement `ClosureSignals::from_draft(draft: &CandidateDraft) -> Self`.

**Enforcement rules (applied before existing terminator-finding logic):**

| Signal | Threshold | Action |
|---|---|---|
| `continuation_probability` | > 0.70 | **Must extend** — search forward for next terminator with adequate pause |
| `breath_pause_risk` | > 0.65 | **Must extend** — treat raw_end as breath pause, search forward |
| `mid_thought_risk` | > 0.65 | **Must extend** — treat raw_end as mid-thought, search forward |
| `answer_complete` | `false` | **Must extend** — keep extending (use `draft.answer_end` if available) |
| `story_completeness` | `false` | **Must extend** — keep extending toward payoff_end |
| `payoff_completion` | `false` | **Must extend** — keep extending toward payoff_end |
| `closure_confidence` | < 0.50 | Bias toward extension — weight terminator search to prefer later candidates |

When `closure` is `None`: fall back to existing logic unchanged.
When extension is required but no valid terminator exists within `max_guideline`: accept best available sentence terminator within bounds.

**Updated call site in `lib.rs` line 524:**
```rust
let (snapped_start, snapped_end) = snap_to_semantic_boundaries(
    &normalized.words,
    draft.start,
    draft.end,
    normalized.duration,
    Some(&ClosureSignals::from_draft(draft)),
);
```

---

### R5. Diagnostic Output (Stderr / Log)

Per candidate:
```
HOOK: <hook_text>
HOOK SPEAKER: Host | Guest
HOOK TYPE: <hook_type>
TOPIC CLARITY: <n>  CURIOSITY: <n>  RELEVANCE: <n>  CONTRAST: <n>
DELAY: PASS/FAIL  CONFUSION: PASS/FAIL  IRRELEVANCE: PASS/FAIL  DISINTEREST: PASS/FAIL
CONTEXT SUFFICIENCY: <bool>  QUESTION INCLUDED: <bool>
ANSWER COMPLETE: <bool>  STORY COMPLETE: <bool>  PAYOFF COMPLETE: <bool>
ENDING TYPE: <ending_type>
CONTINUATION_PROBABILITY: <n>
BREATH_PAUSE_RISK: <n>
SEMANTIC_CLOSURE_CONFIDENCE: <n>
START REASON: <reason>
END REASON: <reason>
DURATION: <n>s
```

For at least 3 candidates per run, also emit boundary trace:
```
RAW_START: <t>  FINAL_START: <t>
RAW_END: <t>    FINAL_END: <t>
LAST_SPOKEN_TEXT_BEFORE_RAW_END: <text>
NEXT_SPOKEN_TEXT: <text>
PAUSE_DURATION: <ms>ms
CONTINUATION_PROBABILITY: <n>
SEMANTIC_CLOSURE_CONFIDENCE: <n>
SNAP_ACTION: EXTENDED | ACCEPTED | REJECTED
SNAP_REASON: <explanation>
```

---

### R6. Hard Rejection

Hook failures (remove from ranking):
- Opening lacks topic clarity within first 1–3 seconds
- Opening contains unexplained pronoun or dangling reference without compensating hook strength
- Hook is vague suspense with no identifiable subject
- Hook starts mid-answer without sufficient prior context

Closure failures (enforced by `snap_to_semantic_boundaries`):
- Ending during unfinished phrase/clause AND no valid terminator exists within max_guideline
- `continuation_probability > 0.85` AND no valid extension possible → reject candidate entirely

---

### R7. Hindi / Hinglish Support

Apply all four hook mechanisms and all closure signals equally to Hindi, Hinglish, English, and multilingual conversations. Internal reasoning may use English; all output text must remain verbatim in the original spoken language.

---

### R8. New Regression Tests (`test_hook_closure_suite.py` — 25 minimum)

**Hook quality (8 tests):**
- Candidate with unexplained pronoun → `hook_confusion_penalty > 0`
- Candidate with filler before topic → `hook_delay_penalty > 0`
- No identifiable subject → rejected from ranking
- Strong interviewer Q + resolved answer scores above standalone guest statement
- Hindi/Hinglish hook correctly evaluated
- Guest statement mid-answer fails confusion check
- Segment-first: isolated interesting sentence without topic context not auto-selected
- Hook repair: weak opening moves back to interviewer question

**Semantic closure — LLM layer (6 tests):**
- Ending mid-sentence → `continuation_probability >= 0.70`
- Ending before punchline → `payoff_completion = false`
- Unresolved Q&A → `answer_complete = false`
- Genuine closure → `closure_confidence >= 0.80`
- Before unrelated new topic → accepted, not over-extended
- Duration > 50s at genuine closure → accepted

**Deterministic enforcement — `snap_to_semantic_boundaries` (8 tests):**
- `continuation_probability = 0.85`, raw_end mid-sentence → snapped_end > raw_end
- `breath_pause_risk = 0.80`, same speaker continues → snapped_end > raw_end
- `mid_thought_risk = 0.70` → snapped_end > raw_end
- `answer_complete = false`, answer_end in draft → snapped_end >= answer_end (or nearest terminator after)
- `payoff_completion = false` → snapped_end extends toward payoff_end
- `closure = None` → falls back to existing logic (no regression)
- `continuation_probability = 0.90` → final timestamp differs from raw_end (LLM premature endpoint cannot reach FFmpeg)
- All 4 existing Rust snap_to_semantic_boundaries tests continue passing

**Schema (3 tests):**
- All 17 new `CandidateDraft` fields present in JSON with correct types
- Old candidate JSON without new fields deserialises without error
- `ClosureSignals::from_draft` maps all fields including None defaults

---

## Acceptance Criteria

### Hook quality
- [ ] Candidate with unexplained pronoun/dangling conjunction → `hook_confusion_penalty > 0`, ranked below self-contained alternative
- [ ] Strong interviewer Q + resolved answer ranks ≥ equivalent guest statement requiring prior context
- [ ] Filler-opening candidate → `hook_delay_penalty > 0`
- [ ] All R5 diagnostic fields present per candidate in stderr/log

### Semantic closure (deterministic enforcement)
- [ ] Segment ending mid-sentence, same speaker continues within 2s → `continuation_probability >= 0.70` → `snap_to_semantic_boundaries` extends beyond raw_end
- [ ] Segment ending before punchline → `payoff_completion = false` → extended
- [ ] Unresolved Q&A → `answer_complete = false` → extended
- [ ] `closure_confidence >= 0.80`, duration > 50s → accepted, not truncated
- [ ] `closure_confidence >= 0.80`, just before unrelated topic → accepted, not over-extended
- [ ] `continuation_probability = 0.90` candidate **cannot reach FFmpeg render** with raw premature endpoint

### Schema
- [ ] All 17 new fields in `CandidateDraft` with `Option<T>` + `#[serde(default)]`
- [ ] Old candidates without new fields deserialise without error

### Regression (zero)
- [ ] `python -m unittest test_hook_closure_suite.py` — **25/25 pass**
- [ ] `python -m unittest test_active_speaker_suite.py` — **28/28 pass**
- [ ] `python -m unittest test_multimodal_hook_suite.py` — **6/6 pass**
- [ ] `cargo test` — **46/46 pass**
- [ ] `npm run build` — **exit 0**

## 2026-09-15T09:34:00Z

TASK — FORENSIC INSPECTION OF AUTOSHORTS 8.0 BEFORE NEXT FEATURE

PROJECT:
AutoShorts 8.0

CRITICAL:
AutoShorts 7.0 is FROZEN.
Do NOT modify AutoShorts 7.0.

This task is INSPECTION ONLY.

DO NOT IMPLEMENT ANY NEW FEATURE.
DO NOT REFACTOR.
DO NOT CLEAN UP.
DO NOT CHANGE PRODUCTION CODE.
DO NOT CHANGE TESTS.
DO NOT CHANGE THE BRAIN FILE.

Working directory: d:\College\Autoshorts 8.0
Integrity mode: benchmark

==================================================
OBJECTIVE
==================================================

Perform a complete forensic inspection of the current AutoShorts 8.0 codebase so
we have an accurate baseline before continuing development with Anti-Gravity.

The goal is to determine exactly:

1. what architectures currently exist;
2. where the three new 8.0 features are implemented;
3. how they integrate with the existing pipeline;
4. what per-clip information/status currently exists;
5. what is already tested and verified;
6. what files are currently modified;
7. what remains unimplemented from the planned roadmap.

Do NOT infer behavior from filenames alone.
Inspect the actual implementation.

==================================================
AUTHORITATIVE DOCUMENT
==================================================

Use:
AUTOSHORTS_8_0_BRAIN.md

Read the relevant sections, but verify the documented claims against the actual code.
The Brain is documentation, not proof.
If documentation and implementation disagree:
    report the disagreement.
Do NOT silently "fix" the documentation during this inspection.

==================================================
PROJECT BOUNDARY
==================================================

Inspect ONLY:
    AutoShorts 8.0

Explicitly verify that:
    AutoShorts 7.0
has not been modified.
Do not make any changes to 7.0.

==================================================
PART 1 — REPOSITORY STRUCTURE
==================================================

Inspect the 8.0 repository structure.
Identify:
- frontend;
- Rust/backend;
- Python analysis/tracking;
- test directories;
- scripts;
- render pipeline;
- Brain/documentation;
- diagnostic tools;
- temporary/scratch areas.
Report the important architecture boundaries.

==================================================
PART 2 — CURRENT END-TO-END PIPELINE
==================================================

Trace the actual current production flow from:
    source/import
        ↓
    download/import
        ↓
    transcription
        ↓
    semantic correction
        ↓
    candidate discovery
        ↓
    narrative/hook scoring
        ↓
    candidate selection
        ↓
    Hook & Ending Optimization 2.0
        ↓
    Smart Pacing
        ↓
    framing
        ↓
    Active Speaker
        ↓
    Single / DualFrame
        ↓
    conditional subject-tight framing
        ↓
    close-speaker isolation
        ↓
    Audio Intelligence
        ↓
    captions
        ↓
    render
        ↓
    final output

The order above is a hypothesis to verify, NOT an assumption.
Report the actual order.
If something occurs at a different stage, explain it.

==================================================
PART 3 — THREE COMPLETED 8.0 FEATURES
==================================================

Inspect the actual implementation of:
1. Smart Pacing
2. Hook & Ending Optimization 2.0
3. Audio Intelligence / Speech Quality

For EACH feature, report:
- main implementation files;
- main functions/classes;
- input;
- output;
- where it is invoked;
- where its result is stored;
- how the rest of the pipeline consumes its result;
- whether the feature can skip processing;
- how "actually applied" is represented;
- whether that state is currently available per clip.

==================================================
PART 4 — VISUAL PIPELINE
==================================================

Inspect the current visual architecture:
- Active Speaker;
- Ultralytics;
- Single Speaker;
- DualFrame;
- conditional subject-tight framing;
- close-speaker isolation;
- constrained tracking;
- isolation hysteresis;
- head-priority corridor;
- fallback;
- segment-aware geometry.

Confirm which pieces are sealed and where they live. Do not modify them.

==================================================
PART 5 — AUDIO PIPELINE
==================================================

Inspect:
- audio analysis;
- decision/gating;
- processing stages;
- Rust integration;
- render integration;
- kill-switch;
- exactly-once/idempotence;
- output loudness handling;
- skip behavior.

Confirm that Audio Intelligence is integrated as documented.

==================================================
PART 6 — TIMELINE ARCHITECTURE
==================================================

Trace all relevant time representations:
- source timeline;
- candidate timeline;
- Hook/Ending optimized timeline;
- Smart Pacing edit map;
- output timeline;
- caption timeline;
- framing timeline;
- audio timeline.

Identify every source→output mapping.
Report where conversions occur.
Look specifically for any place where source timestamps and output timestamps could be confused.

==================================================
PART 7 — BACKEND PER-CLIP METADATA
==================================================

Inspect the current backend result/status architecture:
- where a generated clip's metadata is represented;
- where Single Speaker / DualFrame status is stored;
- where render metadata is stored;
- where processing decisions are stored;
- whether Smart Pacing application status exists;
- whether Hook/Ending application status exists;
- whether Audio Intelligence application status exists.

Determine the canonical per-clip result object/path.

==================================================
PART 8 — TEST INVENTORY
==================================================

Inventory the current tests:
- Smart Pacing tests;
- Hook/Ending Optimization tests;
- Hook closure tests;
- Audio Intelligence tests;
- DualFrame tests;
- Active Speaker tests;
- caption tests;
- Ultralytics tests;
- sampling-memory tests;
- real-render tests;
- Rust/cargo tests;
- frontend/build tests.

Report current recorded/pass status if available from the repository.

==================================================
PART 9 — REAL-FOOTAGE ARTIFACTS
==================================================

Inspect existing real-footage validation artifacts and reports:
- which clips have been used;
- what features they exercised;
- where the artifacts/results are stored.

==================================================
PART 10 — B-ROLL STATUS
==================================================

Inspect whether ANY B-roll architecture already exists in 8.0:
- whether B-roll is implemented;
- whether there are stubs/placeholders;
- whether any generation model integration already exists;
- whether any UI/backend hooks exist.

==================================================
PART 11 — WORKTREE / DIFF AUDIT
==================================================

Inspect the complete current working-tree state:
- modified files;
- untracked files;
- newly created diagnostic/test files;
- generated artifacts;
- scratch files;
- documentation changes.

Separate:
1. intentional 8.0 feature work;
2. diagnostics;
3. scratch/temporary artifacts;
4. unrelated modifications.

==================================================
PART 12 — 7.0 FREEZE VERIFICATION
==================================================

Verify that AutoShorts 7.0 remains untouched.
Report the evidence available from the repository/filesystem.

==================================================
PART 13 — ROADMAP STATUS
==================================================

Classify these planned features as IMPLEMENTED, PARTIALLY IMPLEMENTED, or NOT IMPLEMENTED:
A. Smart Pacing
B. Hook & Ending Optimization 2.0
C. Audio Intelligence
D. Per-clip feature-application backend reporting
E. Multi-Speaker Reaction-Aware Editing
F. B-roll / Contextual Visual Insertion
G. Any other major feature already present

==================================================
PART 14 — PROBLEMS / DISCREPANCIES
==================================================

Report any:
- documentation/code mismatch;
- stale references;
- missing state propagation;
- dead code;
- suspicious integration;
- incomplete feature;
- duplicate implementation;
- timeline ambiguity;
- test/implementation mismatch.

Classify each finding: BLOCKING, NON-BLOCKING, or RECOMMENDATION.

==================================================
PART 15 — FINAL BASELINE
==================================================

Produce a concise final architecture map showing:
    input
      ↓
    candidate pipeline
      ↓
    retention pipeline
      ↓
    visual pipeline
      ↓
    audio pipeline
      ↓
    captions
      ↓
    render
      ↓
    backend result

Identify canonical files/functions for each major stage.

==================================================
FINAL REPORT FORMAT
==================================================

Return:
1. Repository structure
2. Actual end-to-end pipeline
3. Smart Pacing implementation
4. Hook & Ending Optimization implementation
5. Audio Intelligence implementation
6. Visual pipeline
7. Timeline architecture
8. Per-clip backend metadata architecture
9. Test inventory
10. Real-footage validation inventory
11. B-roll actual status
12. Worktree/diff audit
13. 7.0 freeze verification
14. Roadmap status
15. Problems/discrepancies
16. Final architecture map

Cite actual file paths and functions/classes for every important claim.
Do NOT modify any files.
