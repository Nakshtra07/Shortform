# AutoShorts 11.0 Phase 1.1 Implementation & Validation Report

## Executive Summary

**Status: PASS** — Phase 1.1 successfully refines the REZE/WindowScoring candidate discovery system to detect distinct local highlight moments in long-form conversational videos (2+ hours), while preserving all AutoShorts invariants.

**Key Result**: On the 2.5-hour Jensen Huang interview:
- **Phase 1 Macro (600s windows)**: 7 candidates, 100% giant spans, 0% tIoU@0.30
- **Phase 1.1 Tiered + ValleyReset**: 22 candidates, 22.7% giant spans, 50% tIoU@0.30 (87% of Phase 1 Micro)

---

## Root Cause Analysis

### The Problem
Phase 1 demonstrated that REZE with 600s windows on 2.5-hour content produces **one giant candidate spanning the entire conversation** (or a few giant spans covering hours). The failure chain:

1. **600s windows** (10 min) each cover multiple conversational topics
2. **Window scores** derived from max-overlapping-candidates stay high throughout
3. **Global Otsu threshold** finds one threshold for entire video
4. **Standard Kadane** only resets when cumulative sum goes negative
5. **Sustained above-threshold scores** never trigger reset → ONE giant span

### Why Valley Detection on Raw Scores Failed
Initial valley reset tracked **global raw score peak**. With early high peak (0.95), all subsequent windows (0.85+) appeared as "valleys" (30% drop = 0.665), but threshold was 0.825 — valley detection never triggered without also triggering standard Kadane reset.

### Why Valley Detection on Excess Over Threshold Works
Tracking **excess over threshold** (score - τ) within each excursion:
- Peak excess = 0.125 (score 0.95, τ=0.825)
- 30% drop = 0.0875 excess = score 0.9125
- Normal conversational fluctuations (0.85→0.95) create detectable excess valleys
- Separates distinct local highlights even when both above threshold

---

## Design: Minimal Architecture Change

### Two Complementary Changes

#### 1. Local Excursion Peak Excess Valley Reset (Algorithm)
**File**: `autoshorts/src-tauri/src/llm.rs` — `kadane_score_spans()`

```rust
// Track peak EXCESS within current excursion (not global raw score)
excursion_peak_excess = excursion_peak_excess.max(y.max(0.0));

// Valley reset: excess drops from THIS excursion's peak excess
let valley_drop = excursion_peak_excess > 0.0 && current_excess < excursion_peak_excess * valley_drop_ratio;
```

- **Default `valley_drop_ratio = 0.70`** (30% excess drop forces new span)
- Preserves REZE contract: same inputs/outputs, configurable
- Only activates during active excursions (current_sum + y > 0)

#### 2. Tiered Window Sizing for Extra-Long Videos (Config)
**File**: `autoshorts/src-tauri/src/models.rs` — `WindowDiscoveryConfig`

```rust
// New tier: >2 hours (7200s) uses 90s windows with 15s overlap
extra_long_window_sec: 90.0,
extra_long_overlap_sec: 15.0,

fn window_and_overlap_for_duration(&self, duration: f64) -> (f64, f64) {
    if duration > 7200.0 {
        (self.extra_long_window_sec, self.extra_long_overlap_sec)  // >2h: 1.5-min windows
    } else if duration > 2700.0 {
        (self.long_window_sec, self.long_overlap_sec)              // 45m-2h: 10-min windows
    } else {
        (self.medium_window_sec, self.medium_overlap_sec)          // 12-45m: 8-min windows
    }
}
```

- **No arbitrary clip duration limits** — window size ≠ clip duration
- Clip duration still determined downstream by payoff alignment
- Finer analysis resolution enables local highlight separation

---

## Validation Results

### Real-Media A/B/C/D Comparison (Jensen Huang, 2.47 hours)

| Metric | Mode A (TimestampGen) | B1 (Ph1 Macro 600s) | B2 (Ph1 Micro 45s) | **B3 (Ph1.1 Tiered+Valley)** | B4 (Ph1.1 Micro+Valley) |
|--------|----------------------|---------------------|-------------------|----------------------------|------------------------|
| Raw Candidates | 56 | 7 | 54 | **22** | 36 |
| Giant Span Rate | 0% | 100% | 1.9% | **22.7%** | 5.6% |
| Mean Duration | 56s | 1099s | 91s | **244s** | 143s |
| tIoU ≥ 0.30 | 100% | 0% | 57.4% | **50.0%** | 41.7% |
| tIoU ≥ 0.50 | 100% | 0% | 25.9% | **27.3%** | 11.1% |
| HIT@1 | YES | NO | NO | **YES** | YES |
| HIT@5 | 100% | 0% | 40% | **80%** | 60% |
| Payoff Alignment | 100% | 100% | 100% | **100%** | 100% |

### Phase 1.1 Success Criteria

| # | Criterion | Target | Result | Status |
|---|-----------|--------|--------|--------|
| 1 | Separation improved vs Phase 1 Macro | More candidates | 7 → 22 | ✅ PASS |
| 2 | Giant span rate reduced | < Phase 1 Macro | 100% → 22.7% | ✅ PASS |
| 3 | tIoU@0.30 competitive with Phase 1 Micro | ≥ 80% of B2 | 57.4% → 50.0% (87%) | ✅ PASS |
| 4 | Payoff alignment high | ≥ 80% | 100% | ✅ PASS |
| 5 | Practical durations | < 5 min mean | 244s (< 300s) | ✅ PASS |

**Overall: ✅ PASS**

---

## Invariant Verification

| Invariant | Status | Evidence |
|-----------|--------|----------|
| Payoff Endpoint Lock | ✅ | candidate_end == payoff_end verified in all modes |
| No Timestamp Hallucination | ✅ | LLM emits ordinal scores only; Rust computes boundaries |
| No Arbitrary Duration Truncation | ✅ | Candidate duration naturally determined by payoff |
| Adaptive Two-Person Framing | ✅ | Unchanged — all DualFrame tests pass (55/55) |
| Zero Active-Speaker Follow in Adaptive | ✅ | Unchanged — Active Speaker tests pass (61/61) |
| Zero DualFrame in Adaptive Mode | ✅ | Unchanged |
| T7 Face-Safe Placement | ✅ | Unchanged — Caption QA passes (100/0) |
| Smart Pacing Monotonicity | ✅ | Unchanged — 55/55 tests pass |
| TimestampGeneration Default | ✅ | `AUTOSHORTS_DISCOVERY_MODE` unset → TimestampGeneration |
| WindowScoring Feature Flag | ✅ | Reversible via env var; safe fallback preserved |
| Cache Correctness | ✅ | WindowScoreCache key includes valley_drop_ratio via prompt_version |
| Multi-Provider Normalization | ✅ | `normalize_provider_scores` unchanged |
| Safe JSON Parsing | ✅ | All parsing tests pass |
| Verbatim Transcript Extraction | ✅ | `spans_to_candidate_drafts` uses exact transcript words |
| Deterministic Boundary Snapping | ✅ | All boundary tests pass (321/321 Rust tests) |

---

## Regression Testing

| Suite | Tests | Status |
|-------|-------|--------|
| Rust Unit & Integration | 321 | ✅ PASS (1 ignored) |
| Hook Closure (Python) | 47 | ✅ PASS |
| Smart Pacing | 55 | ✅ PASS |
| Hook & Ending Optimization | 29 | ✅ PASS |
| Active Speaker Framing | 61 | ✅ PASS |
| DualFrame Layout | 55 | ✅ PASS |
| Caption Templates | 14 | ✅ PASS |
| Caption QA | 100 checks | ✅ PASS (0 failures) |
| Applied Features | 30 | ✅ PASS |
| Audio Intelligence | 48 | ✅ PASS |
| Frontend Production Build | 1 | ✅ PASS |

---

## Remaining Limitations

1. **Simulation vs Real LLM Scoring**: Validation uses max-overlapping-candidate scores as proxy for LLM window scores. Real LLM scoring may produce different distributions.
2. **Valley Ratio Tuning**: Default 0.70 works for this corpus; may need adjustment for other content types.
3. **Micro Window Valley Reset**: Valley reset slightly reduces tIoU for 45s windows (57.4% → 41.7%) — over-segments already-resolved highlights. **Recommendation**: Use valley reset only for coarse windows (≥60s), not micro windows.
4. **Extra-Long Threshold**: 2-hour (7200s) threshold is heuristic; could be made configurable.

---

## Whether WindowScoring Should Remain Experimental

**RECOMMENDATION: Keep Experimental for Now**

**Reasons:**
1. **Not yet superior to TimestampGeneration**: Mode A (56 candidates, 100% tIoU) still outperforms B3 (22 candidates, 50% tIoU)
2. **Simulation gap**: Real LLM scoring behavior unvalidated at scale
3. **Valley ratio sensitivity**: Requires per-corpus tuning
4. **Micro window penalty**: Valley reset harms fine-grained discovery

**Path to Promotion:**
1. Real LLM scoring validation on 10+ long-form videos
2. Adaptive valley ratio (disable for windows < 60s)
3. Demonstrate consistent ≥80% tIoU@0.30 vs TimestampGeneration
4. A/B user study on rendered clip quality

---

## Files Modified

| File | Changes |
|------|---------|
| `autoshorts/src-tauri/src/llm.rs` | `kadane_score_spans()` — local excursion peak excess valley reset; `aggregate_window_scores_to_spans()` — added `valley_drop_ratio` param |
| `autoshorts/src-tauri/src/models.rs` | `WindowDiscoveryConfig` — added `extra_long_window_sec`, `extra_long_overlap_sec`, `score_valley_drop_ratio`; tiered `window_and_overlap_for_duration()` |
| `autoshorts/src-tauri/src/llm.rs` (tests) | Updated Kadane tests for excess-based valley logic; added `test_kadane_spans_shallow_valley_merges_without_reset` |
| `scratch/verify_phase_1_1_reze_ab.py` | New validation script with tiered windows + valley reset |

---

## Conclusion

Phase 1.1 delivers a **technically credible local-highlight discovery mechanism** for long-form media through two minimal, composable changes:

1. **Excess-based valley reset** in Kadane — separates sustained high-score regions into distinct local peaks
2. **Tiered window sizing** — 90s windows for >2h content provides sufficient analysis resolution

Both changes are **reversible, configurable, and preserve all invariants**. WindowScoring remains experimental pending real LLM validation, but the architecture now supports credible long-form discovery.