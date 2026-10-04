# AutoShorts 11.0 — Phase 1 Implementation Plan
# REZE-Style Candidate Discovery Upgrade + yolo11n.pt License Audit

**Created:** 2026-09-28  
**Workspace:** `d:\College\Autoshorts 11.0`  
**Phase 0 Status:** COMPLETE & VERIFIED (289 Rust tests PASS, 20 Python suites PASS, npm build PASS)

---

## Forensic Discovery Summary (read-only, verified against code)

### Current Discovery Flow
```
generate_candidates (lib.rs:556)
  → discover_candidates_full_timeline (llm.rs:829)
      → per-window build_window_semantic_prompt / build_semantic_prompt (timestamp-emission pattern)
      → query_provider_with_retry (each provider: claude, gemini, openai, openrouter, groq, local/ollama, deepseek)
      → parse_candidate_json → Vec<CandidateDraft>
  → hook alignment (align_hook_to_transcript_words, llm.rs:2049)
  → payoff alignment (align_payoff_to_transcript_words, llm.rs:2139)
  → PAYOFF ENDPOINT LOCK: lib.rs:668 draft.end = p_end  ← LOCKED INVARIANT, DO NOT TOUCH
  → snap_to_semantic_boundaries_with_hook_anchor (lib.rs:692)
  → analyze_multimodal_hook_signals (media.rs, calls multimodal_hook_analyzer.py)
  → deduplicate_and_rank_all_candidates (llm.rs:1511)
  → replace_candidates → DB
```

### Key Code Locations
| Item | Location |
|------|----------|
| Main Tauri command | `lib.rs:556` `generate_candidates` |
| Full-timeline engine | `llm.rs:829` `discover_candidates_full_timeline` |
| Window prompt builder | `llm.rs:243` `build_window_semantic_prompt` |
| Full prompt builder | `llm.rs:22` `build_semantic_prompt_core` |
| Provider dispatcher | `llm.rs:420` `query_provider_prompt_raw` |
| JSON parser | `llm.rs:1191` `parse_candidate_json` |
| Composite score | `llm.rs:1723` `composite_score` |
| WindowDiscoveryConfig | `models.rs:508` — 10 fields, no `discovery_mode` yet |
| Existing test count | **289 Rust tests** (35 in llm.rs, 31 in db.rs, rest across media/captions/lib) |

### What Does NOT Exist Yet
- No REZE scoring prompt
- No `WindowScoreResult` struct
- No `build_scoring_prompt` function
- No Kadane/Otsu aggregation
- No `discovery_mode` / feature flag in `WindowDiscoveryConfig`
- No score cache
- No provider score normalization
- No `score_windows_with_provider` function

### LOCKED INVARIANTS (must remain unchanged)
1. **Payoff endpoint**: lib.rs:704-708 — `draft.end = draft.payoff_end.unwrap()` after payoff alignment. Phase 1 scoring must NOT emit `start`, `end`, `hookStart`, `hookEnd`, `payoffStart`, `payoffEnd`.
2. **Adaptive two-person framing**: speaker_tracker.py — both-visible hard constraint.
3. **T7 captions**: no karaoke, chunk-size guardrails.
4. **Smart Pacing logic**: untouched.
5. **Original 9:16 behavior**: untouched.
6. **Audio enhancement defaults**: untouched.
7. **Rendering/loudness/A-V sync**: untouched.

---

## Implementation Tasks

### Task 1: Scoring Contract Design (Rust models.rs)
**File:** `autoshorts/src-tauri/src/models.rs`  
**TDD first:** Test structs in `mod tests` before production code.

Add to `models.rs`:
```rust
/// REZE-style per-window scoring result.
/// The LLM answers only a binary/ordinal relevance question per window.
/// NO timestamps emitted in scoring mode.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WindowScoreResult {
    /// Which window (0-indexed) this score applies to
    pub window_idx: usize,
    /// Window start time in seconds
    pub window_start: f64,
    /// Window end time in seconds  
    pub window_end: f64,
    /// Raw relevance score from LLM: 0.0 = not highlight, 1.0 = strong highlight
    /// Ordinal interpretation: 0.0 / 0.5 / 1.0 (binary/ternary) or continuous float
    pub raw_score: f64,
    /// Optional sub-scores for component aggregation
    pub hook_relevance: Option<f64>,
    pub narrative_completeness: Option<f64>,
    pub payoff_presence: Option<f64>,
    pub engagement_signal: Option<f64>,
    /// Provider that generated this score
    pub provider: String,
    /// Model name used
    pub model: String,
    /// Cache key for this score (source_hash + window_hash + provider + model + prompt_version)
    pub cache_key: String,
    /// Prompt version (for cache invalidation)
    pub prompt_version: String,
}

/// Feature flag / A-B control for candidate discovery mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryMode {
    /// Current timestamp-generation path (backward-compatible default)
    #[default]
    TimestampGeneration,
    /// REZE-style per-window scoring + deterministic aggregation
    WindowScoring,
}
```

Also add to `WindowDiscoveryConfig`:
```rust
/// Discovery mode: timestamp_generation (default, safe) or window_scoring (REZE)
pub discovery_mode: DiscoveryMode,
/// Prompt version string for cache invalidation (bump on any scoring prompt change)
pub scoring_prompt_version: String,
/// Gaussian smoothing sigma for REZE aggregation (default: 1.5)
pub score_smoothing_sigma: f64,
/// Otsu threshold multiplier (default: 1.0)
pub score_otsu_multiplier: f64,
/// Minimum window score to enter Kadane selection (default: 0.35)
pub score_kadane_min: f64,
```

**Tests required (BEFORE production code):**
1. `test_window_score_result_default_fields` — verify defaults compile
2. `test_discovery_mode_serde_roundtrip` — serialize/deserialize DiscoveryMode
3. `test_window_discovery_config_includes_mode` — verify new fields in default()

### Task 2: Scoring Prompt (llm.rs)
**File:** `autoshorts/src-tauri/src/llm.rs`  
**CRITICAL:** Do NOT parallelize llm.rs changes. Sequential, single-subagent.

Add `build_window_scoring_prompt`:
```rust
/// Build REZE-style per-window scoring prompt.
/// The LLM scores only binary/ordinal relevance — it MUST NOT emit start, end,
/// hookStart, hookEnd, payoffStart, payoffEnd, or any timestamps.
pub fn build_window_scoring_prompt(
    transcript_text: &str,
    lang_category: &str,
    window_start: f64,
    window_end: f64,
    window_idx: usize,
    total_windows: usize,
) -> String { ... }
```

Scoring prompt structure:
- Role: "You are a highlight relevance scorer for podcast content."
- Task: Score this transcript window's highlight potential 0.0–1.0
- STRICT PROHIBITION: "DO NOT emit any timestamps, clip boundaries, start/end times, hookStart, hookEnd, payoffStart, payoffEnd"
- Return JSON: `{"highlight_score": <0.0-1.0>, "hook_relevance": <0.0-1.0>, "narrative_completeness": <0.0-1.0>, "payoff_presence": <0.0-1.0>, "reasoning": "<brief>"}`
- Grounding: ONLY score content visible in this window

**Parser:** `parse_window_score_json(text: &str, window_idx: usize, window_start: f64, window_end: f64, provider: &str, model: &str) -> Result<WindowScoreResult>`

**Tests required (BEFORE production code):**
1. `test_build_window_scoring_prompt_no_timestamp_fields` — assert prompt does NOT contain "start", "end", "hookStart", "hookEnd", "payoffStart", "payoffEnd" as JSON field names
2. `test_parse_window_score_json_valid` — parse valid score JSON
3. `test_parse_window_score_json_rejects_timestamp_fields` — if LLM sneaks in start/end, log warning but still extract highlight_score
4. `test_parse_window_score_json_fallback_on_invalid` — malformed JSON → Err

### Task 3: Provider Score Normalization (llm.rs)
**File:** `autoshorts/src-tauri/src/llm.rs`

Each provider may score on different scales. Before aggregation, normalize scores per-provider.

```rust
/// Per-provider score normalization before global aggregation.
/// Uses z-score normalization when enough samples exist; otherwise identity.
pub fn normalize_provider_scores(
    scores: &mut Vec<WindowScoreResult>,
    min_samples_for_zscore: usize,  // default: 3
) { ... }

/// Score query for a single window using the scoring prompt.
pub async fn score_window_with_provider(
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    prompt: &str,
    window_idx: usize,
    window_start: f64,
    window_end: f64,
) -> Result<WindowScoreResult> { ... }
```

**Tests required:**
1. `test_normalize_provider_scores_identity_small_sample` — < min_samples → no change
2. `test_normalize_provider_scores_zscore_large_sample` — enough samples → mean-center + std-normalize
3. `test_normalize_provider_scores_handles_zero_std` — all same score → identity

### Task 4: Deterministic Score Aggregation — Kadane/Otsu (llm.rs)
**File:** `autoshorts/src-tauri/src/llm.rs`

```rust
/// REZE-style deterministic aggregation of window scores.
/// Pipeline:
///   1. Mean-center per provider (done upstream in normalize_provider_scores)
///   2. Gaussian smooth with sigma (default 1.5 window widths)
///   3. Otsu threshold to find the high-score boundary
///   4. Kadane's algorithm to select maximal-score contiguous regions
///   5. Return candidate spans as (start_sec, end_sec) pairs
pub fn aggregate_window_scores_to_spans(
    scores: &[WindowScoreResult],
    sigma: f64,
    kadane_min_score: f64,
    otsu_multiplier: f64,
) -> Vec<(f64, f64)> { ... }

/// Gaussian kernel for 1D smoothing of window scores.
fn gaussian_smooth_scores(values: &[f64], sigma: f64) -> Vec<f64> { ... }

/// Otsu's threshold on a 1D score array.
fn otsu_threshold(values: &[f64]) -> f64 { ... }

/// Kadane's maximum-subarray selection on score array with min_score floor.
fn kadane_score_spans(
    scores: &[f64],
    window_starts: &[f64],
    window_ends: &[f64],
    min_score: f64,
) -> Vec<(f64, f64)> { ... }
```

**Tests required (8 categories):**
1. `test_gaussian_smooth_identity_single_window` — single window unchanged
2. `test_gaussian_smooth_wider_kernel` — multiple windows, central window amplified
3. `test_otsu_threshold_bimodal` — clear high/low bimodal → threshold between peaks
4. `test_otsu_threshold_uniform` — all same score → some reasonable threshold
5. `test_kadane_spans_contiguous_high` — contiguous high-score windows → one span
6. `test_kadane_spans_gap_between` — two separate highs → two spans
7. `test_aggregate_scores_end_to_end` — full pipeline: normalize → smooth → otsu → kadane → spans
8. `test_aggregate_scores_empty_input` — empty scores → empty spans

### Task 5: Candidate Integration — REZE Spans → CandidateDraft (llm.rs)
**File:** `autoshorts/src-tauri/src/llm.rs`

Convert Kadane-selected spans into `CandidateDraft` objects compatible with the existing downstream pipeline (payoff alignment, boundary snapping, etc.).

```rust
/// Convert REZE-aggregated spans to CandidateDraft objects.
/// These drafts go through the SAME downstream pipeline as timestamp-generation drafts:
///   hook alignment → payoff alignment → payoff endpoint lock → snap_to_semantic_boundaries
/// The score field is set from the max window score in the span.
/// hook/payoff text fields are left None (filled by downstream alignment).
pub fn spans_to_candidate_drafts(
    spans: &[(f64, f64)],
    scores: &[WindowScoreResult],
    transcript: &NormalizedTranscript,
    lang_category: &str,
    script_type: &str,
) -> Vec<CandidateDraft> { ... }
```

**Tests required:**
1. `test_spans_to_drafts_basic` — two spans → two drafts with correct start/end/score
2. `test_spans_to_drafts_no_timestamp_fields_in_draft` — hook/payoff text None initially (downstream fills)

### Task 6: Score Cache (llm.rs)
**File:** `autoshorts/src-tauri/src/llm.rs`

Cache keyed on: `source_identity + transcript_window_hash + provider + model + prompt_version`

```rust
/// In-memory scoring cache (per-process lifetime).
/// Thread-safe: Arc<Mutex<HashMap>>
pub struct WindowScoreCache {
    inner: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, WindowScoreResult>>>,
}

impl WindowScoreCache {
    pub fn new() -> Self { ... }
    pub fn get(&self, key: &str) -> Option<WindowScoreResult> { ... }
    pub fn insert(&self, key: &str, result: WindowScoreResult) { ... }
    /// Build cache key from transcript window identity + provider + model + prompt version
    pub fn build_key(
        source_hash: &str,
        window_start: f64,
        window_end: f64,
        transcript_hash: &str,
        provider: &str,
        model: &str,
        prompt_version: &str,
    ) -> String { ... }
}
```

**Tests required:**
1. `test_score_cache_insert_and_get` — basic round-trip
2. `test_score_cache_key_uniqueness` — different provider/model → different key

### Task 7: Window Scoring Engine (llm.rs)
**File:** `autoshorts/src-tauri/src/llm.rs`

Add REZE-mode path to `discover_candidates_full_timeline` via feature flag.

```rust
/// REZE-style window scoring discovery engine.
/// Scores all windows in parallel (same concurrency as timestamp path),
/// normalizes per-provider, then deterministically aggregates.
/// Falls back to timestamp-generation if scoring fails for a window.
pub async fn discover_candidates_reze_scoring(
    transcript: &NormalizedTranscript,
    provider: &str,
    api_key: &str,
    model_name: Option<&str>,
    config: &WindowDiscoveryConfig,
    cache: &WindowScoreCache,
) -> Result<Vec<CandidateDraft>> { ... }
```

Modify `discover_candidates_full_timeline` to branch on `config.discovery_mode`:
```rust
match config.discovery_mode {
    DiscoveryMode::TimestampGeneration => {
        // Existing path — unchanged
    }
    DiscoveryMode::WindowScoring => {
        discover_candidates_reze_scoring(...).await
    }
}
```

**Tests required:**
1. `test_discover_reze_fallback_on_all_windows_fail` — all scoring calls fail → Err (no silent empty)
2. `test_timestamp_mode_unchanged_by_reze_code` — timestamp mode produces same result as before

### Task 8: Feature Flag in lib.rs (lib.rs — minimal change)
**File:** `autoshorts/src-tauri/src/lib.rs`

In `generate_candidates`, allow `discovery_mode` to come from env var for A-B testing:
```rust
let discovery_mode = std::env::var("AUTOSHORTS_DISCOVERY_MODE")
    .ok()
    .and_then(|v| match v.to_lowercase().as_str() {
        "window_scoring" | "reze" => Some(models::DiscoveryMode::WindowScoring),
        _ => None,
    })
    .unwrap_or_default(); // Default: TimestampGeneration

let discovery_config = models::WindowDiscoveryConfig {
    discovery_mode,
    ..models::WindowDiscoveryConfig::default()
};
```

**Default remains `TimestampGeneration`** — zero behavior change without the env var.

**Tests required:**
1. `test_generate_candidates_default_mode_is_timestamp` — env var absent → TimestampGeneration

### Task 9: Full Test Suite (minimum 8 scoring-specific tests, full regression)

**8 required scoring tests (may be spread across tasks 2-8 above):**
Already specified above per-task. Final count must be ≥ 8 across categories:
1. Scoring prompt no-timestamp contract ✓
2. Score JSON parsing valid ✓  
3. Provider normalization ✓
4. Gaussian smoothing ✓
5. Otsu threshold bimodal ✓
6. Kadane contiguous ✓
7. End-to-end aggregate ✓
8. Span to draft conversion ✓

**Full regression gates (all must pass before declaring Task 9 complete):**
```
cargo test -p autoshorts --lib -- (289+ tests, 0 failures)
python test_multimodal_acoustic_suite.py (10/10)
python test_path_resolution.py (4/4)
python test_framing_config_suite.py (2/2)
python test_geometry_scaling_suite.py (8/8)
[all other 20 Python suites]
npm run build (0 errors)
```

### Task 10: Real-Media A-B Validation

Run both discovery modes on real footage and measure:
- Raw candidates discovered (count)
- Survivors after quality gate (count)
- Payoff alignment success rate (%)
- HIT@k if ground truth exists
- End-to-end wall-clock time

Script: `scripts/reze_ab_validation.py` or integrated into the existing real-media invariant script.

Report metrics honestly — do NOT claim REZE beats timestamp-generation based only on QVHighlights literature.

### Task 11: License Audit (yolo11n.pt)

**Files to inspect:**
- `autoshorts/src-tauri/models/yolo11n.pt` (5.35 MB, present)
- `autoshorts/src-tauri/models/yolo11n-seg.pt` (5.9 MB, present)
- `autoshorts/src-tauri/models/face_detection_yunet_2023mar.onnx` (0.22 MB, present)
- No LICENSE or NOTICE file exists in models/

**Facts confirmed:**
- yolo11n.pt: Ultralytics YOLO11n, **AGPL-3.0** — copyleft, requires source disclosure for distributed binaries OR Ultralytics Enterprise License
- yolo11n-seg.pt: Same Ultralytics YOLO11, **AGPL-3.0**
- face_detection_yunet_2023mar.onnx: YuNet from OpenCV Zoo, **Apache-2.0** (permissive)

**Engineering implications to document:**
1. The shipped `yolo11n.pt` + `yolo11n-seg.pt` create AGPL-3.0 copyleft exposure for any distributed binary of AutoShorts
2. The AGPL network clause: if AutoShorts is offered as a network service, source code disclosure is required
3. Current state: dev-machine only (no sidecar bundling per F10), but this must be resolved before any release
4. Resolution options: (a) AGPL compliance (open-source entire app), (b) Ultralytics Enterprise License, (c) replace with non-AGPL alternative (e.g., YuNet for detection — already present — or YOLO-NAS from Deci/NVIDIA under Apache-2.0)
5. Do NOT add another model bundle as a workaround

**Produce:** `docs/license_audit/yolo11n_license_audit.md` with facts, open questions, and engineering options. NO invented legal conclusions.

### Task 12: Brain Update + Final Report

**Final report:** `autoshorts_11_phase_1_implementation_and_validation_report.md`  
**Brain update:** Append Section 42 to `AUTOSHORTS_11_0_BRAIN.md`

---

## Execution Model

**Sequential tasks using subagent-driven development:**

```
Task 1 (models.rs contracts) → validate → review
Task 2 (scoring prompt) → validate → review
Task 3 (provider normalization) → validate → review
Task 4 (Kadane/Otsu aggregation) → validate → review
Task 5 (candidate integration) → validate → review
Task 6 (cache) → validate → review
Task 7 (window scoring engine) → validate → review
Task 8 (feature flag lib.rs) → validate → review
Task 9 (regression suite) → validate → review
Task 10 (A-B validation) → validate → review
Task 11 (license audit) → produce report
Task 12 (brain update + final report)
```

**Never implement the next task until the previous task's review is clean.**

**DO NOT implement:**
- Phase 2 items: diarization, Nemotron, WhisperX diarization, OSNet re-ID, Smart Pacing classifier, CLIP crop scorer, T7 prosody model, new VLM, giant end-to-end video model
- Any change to payoff endpoint architecture
- Any change to adaptive two-person framing
- Parallelizing llm.rs changes

---

## Anti-Patterns to Avoid

1. ❌ Scoring prompt that emits any timestamp fields
2. ❌ Modifying the existing timestamp-generation path
3. ❌ Touching payoff endpoint logic (lib.rs:663-724)
4. ❌ Claiming REZE improvement from QVHighlights literature applies to AutoShorts
5. ❌ Declaring success based only on unit tests (real-media A-B required)
6. ❌ Adding model bundles as AGPL workaround
7. ❌ Inventing legal conclusions in the license audit
