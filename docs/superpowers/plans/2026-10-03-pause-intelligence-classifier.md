# Pause Intelligence Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and wire the LightGBM multi-class pause classifier into Smart Pacing as an evidence-only signal with strict deterministic keep-wins safety invariants and a disabled-by-default feature flag.

**Architecture:** A hybrid architecture where Python Smart Pacing (`smart_pacing.py`) calls `pause_intelligence.py`'s scoring functions directly in-process with clean dependency fault-tolerance, while the Rust engine (`src/pause_intel.rs`) invokes `pause_intelligence.py` as a bounded sidecar (`proc_guard`, 60s timeout) with disk caching under `%APPDATA%/com.autoshorts.desktop/pause_intel_cache/`.

**Tech Stack:** Python 3.12, LightGBM, Silero VAD, NumPy, Rust 2021, Tauri, `proc_guard`, `serde_json`.

## Global Constraints

- **Keep-Wins-On-Conflict**: Deterministic safety rules in `smart_pacing.py` are strictly authoritative. The classifier never overrides a deterministic keep.
- **Scope Lock**: Zero modifications to `llm.rs`, `models.rs`, `db.rs` (schema), `speaker_tracker.py`, `speaker_intelligence.rs`, `captions.rs`, `boundary.rs`, `pacing.rs` (Rust), `vlm_scoring.rs`, `candidate_redundancy.rs`, `render_qa.rs`, `youtube.rs`, `transcription.rs`, `scene_intelligence.rs`, `t7_prosody.py`.
- **No Database Changes**: No new SQLite tables or schema migrations.
- **No Public Rust Module Exposure**: `pause_intel.rs` is `mod pause_intel;` (private), not `pub mod`.
- **No Frontend Modifications**: Zero changes under `autoshorts/src/`.
- **No Committed Trained Models**: Only scripts, schemas, and directories are committed.
- **No Fabricated Training Data**: Gaps $< 200$ cleanly outputs `"NO LABELS — classifier not trained"` and exits code 0.
- **openSMILE Exclusion**: Only Silero VAD + NumPy + LightGBM.
- **Removable Category Gate**: Only `breath_pause` and `waiting_pause` are removable.
- **Feature Flag Default**: `AUTOSHORTS_SMART_PACING_LEARNED` defaults to disabled (unset = disabled; `1/true/on` = enabled).

---

### Task 1: Dataset Labeling Pipeline (`scripts/pause_label.py`)

**Files:**
- Create: `autoshorts/src-tauri/scripts/pause_label.py`
- Create: `autoshorts/src-tauri/scripts/data/.gitkeep`

**Interfaces:**
- Consumes: `breath_detect.detect_gaps`, `pause_intelligence.build_gap_features`, `pause_intelligence.source_hash`, `pause_intelligence.features_to_row`.
- Produces: `scripts/data/pause_features_v1.jsonl` with `{sourceHash, gapIdx, srcStartSec, srcEndSec, features, label, labelSource, weakSupervision, timestamp}`.

- [ ] **Step 1: Write `autoshorts/src-tauri/scripts/pause_label.py`**

Implement the data labeling pipeline supporting:
- Ingestion of source video + `words.json` + optional `labels.jsonl`.
- Filtering gaps: reject $duration < 0.05$s or $duration > 5.0$s.
- Extracting 23 features via `pause_intelligence.build_gap_features`.
- Weak supervision mapping based on `smart_pacing.py` confident decisions.
- Single-video CLI mode and `--corpus` directory batch mode.

- [ ] **Step 2: Test `pause_label.py` on workspace sample**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/pause_label.py --help
```
Expected: Prints CLI help with `--labels`, `--out`, and `--corpus` options.

---

### Task 2: Multi-Class LightGBM Training Pipeline (`scripts/pause_train.py`)

**Files:**
- Create: `autoshorts/src-tauri/scripts/pause_train.py`
- Create: `autoshorts/src-tauri/scripts/models/.gitkeep`

**Interfaces:**
- Consumes: `scripts/data/pause_features_v1.jsonl`.
- Produces: `scripts/models/pause_classifier_v1.txt` + `scripts/models/pause_classifier_v1.meta.json`.

- [ ] **Step 1: Write `autoshorts/src-tauri/scripts/pause_train.py`**

Implement the training pipeline:
- Validates sample count: if $< 200$, prints `"NO LABELS — classifier not trained"` and exits 0.
- Groups records by `sourceHash` and applies a 20% disjoint per-video split.
- Configures 6-class LightGBM (`objective='multiclass'`, `num_class=6`, `metric=['multi_logloss', 'multi_error']`).
- Evaluates AUC, PR-AUC, removable-gate precision/recall, and per-class confusion matrix.
- Exports text model and JSON metadata with pinned `featureVersion: "1"`.

- [ ] **Step 2: Test `< 200` sample safety refusal guard**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/pause_train.py --data non_existent.jsonl
```
Expected: Outputs `NO LABELS — classifier not trained` and exits code 0.

---

### Task 3: Python Pause Intelligence Test Suite (`scripts/test_pause_intel_suite.py`)

**Files:**
- Create: `autoshorts/src-tauri/scripts/test_pause_intel_suite.py`

- [ ] **Step 1: Write `autoshorts/src-tauri/scripts/test_pause_intel_suite.py`**

Author 10+ comprehensive unit tests:
1. `test_feature_vector_dimension_and_order`
2. `test_acoustic_feature_calculations`
3. `test_labeling_duration_filters`
4. `test_weak_supervision_mapping`
5. `test_trainer_refuses_under_200_samples`
6. `test_trainer_disjoint_video_partition`
7. `test_model_export_and_meta_schema`
8. `test_model_scoring_removable_classes`
9. `test_feature_version_mismatch_refusal`
10. `test_predict_cli_json_payload`

- [ ] **Step 2: Run Python test suite**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" -m unittest autoshorts/src-tauri/scripts/test_pause_intel_suite.py
```
Expected: All 10+ tests pass with `OK`.

---

### Task 4: Rust Sidecar & Caching Module (`src/pause_intel.rs`)

**Files:**
- Create: `autoshorts/src-tauri/src/pause_intel.rs`

**Interfaces:**
- Produces: `pub struct PauseIntelGap`, `pub fn score_gaps(...) -> Vec<PauseIntelGap>`, `pub fn pause_intel_cache_dir() -> PathBuf`.

- [ ] **Step 1: Write `autoshorts/src-tauri/src/pause_intel.rs`**

Implement:
- `PauseIntelGap` struct with serde aliases for camelCase and snake_case.
- `pause_intel_cache_dir()` resolving `%APPDATA%/com.autoshorts.desktop/pause_intel_cache/`.
- Discovery of `pause_intelligence.py` and `pause_classifier_v1.txt`.
- Bounded execution via `crate::proc_guard::run_bounded` (60s timeout).
- Disk caching per `(source_hash, feature_version, start_ms, end_ms)`.
- Graceful fallback returning empty vector on missing model, timeout, or errors.

---

### Task 5: Smart Pacing Integration (`scripts/smart_pacing.py`)

**Files:**
- Modify: `autoshorts/src-tauri/scripts/smart_pacing.py:880-925`

**Interfaces:**
- Updates `_attach_learned_pause_evidence` to evaluate model predictions, mark `removable=True` when $P \ge 0.65$ for removable classes, and uphold deterministic keep-wins invariance.

- [ ] **Step 1: Update `smart_pacing.py`**

Refine `_attach_learned_pause_evidence`:
- Guard with clean `try...except` to fail soft if dependencies are missing.
- When $P(\text{removable}) \ge 0.65$ and `predictedClass in ('breath_pause', 'waiting_pause')`, mark `removable=True` with evidence `pause_intel`.
- Ensure deterministic safety rules in `detect_pacing_cuts` retain full authority.
- Log: `[PauseIntel] gaps_scored=<n> removable=<n> model=<version>`.

- [ ] **Step 2: Run Smart Pacing regression suites**

Run:
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" -m unittest autoshorts/src-tauri/scripts/test_smart_pacing_suite.py autoshorts/src-tauri/scripts/test_smart_pacing_2_suite.py
```
Expected: 55/55 and 108/108 tests pass.

---

### Task 6: Scoped Module & Flag Wiring (`src/lib.rs`)

**Files:**
- Modify: `autoshorts/src-tauri/src/lib.rs`

- [ ] **Step 1: Wire `pause_intel` module and flag helper in `lib.rs`**

- Declare private `mod pause_intel;`.
- Re-export `pub use pause_intel::{score_gaps, PauseIntelGap};`.
- Implement `pub fn pause_intel_enabled() -> bool`:
  ```rust
  pub fn pause_intel_enabled() -> bool {
      match std::env::var("AUTOSHORTS_SMART_PACING_LEARNED") {
          Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"),
          Err(_) => false,
      }
  }
  ```

---

### Task 7: Rust Integration Test Suite (`tests/pause_intel_suite.rs`)

**Files:**
- Create: `autoshorts/src-tauri/tests/pause_intel_suite.rs`

- [ ] **Step 1: Write `tests/pause_intel_suite.rs`**

Author 8 integration tests:
1. `test_missing_model_fallback`: Gracefully returns empty vector.
2. `test_flag_disabled_by_default`: Unset env var reports disabled and empty.
3. `test_flag_explicit_enable`: Enabled with "1" or "true".
4. `test_removable_above_threshold_detected`: $P \ge 0.65$ and removable class recognized.
5. `test_low_confidence_ignored`: $P < 0.50$ ignored.
6. `test_conflict_keep_wins`: Deterministic safety rules prevail.
7. `test_version_mismatch_fails_soft`: Mismatched feature version fails soft.
8. `test_serde_roundtrip_pause_intel_gap`: CamelCase and snake_case serialization.

- [ ] **Step 2: Run Rust integration tests**

Run:
```powershell
cargo test -p autoshorts --test pause_intel_suite
```
Expected: 8 passed; 0 failed.

---

### Task 8: End-to-End Verification & Acceptance Check

- [ ] **Step 1: Run complete Rust library suite**
```powershell
cargo test -p autoshorts --lib
```
Expected: 457 passed; 0 failed.

- [ ] **Step 2: Run complete Python test discovery**
```powershell
& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" -m unittest discover -s autoshorts/src-tauri/scripts -p "test_*.py"
```
Expected: All suites green with zero regressions.
