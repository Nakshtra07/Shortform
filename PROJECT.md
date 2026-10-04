# Project: AutoShorts 11.0 T7 Prosody Integration

## Architecture
AutoShorts 11.0 caption rendering pipeline integrates T7 prosodic boundary tagging as a flag-gated soft threshold modulator inside `captions.rs:segment_speech_chunks`, strictly preserving existing hard guardrails and baseline fallback behavior.

### High-Level Data Flow:
```
[Video / Audio Source] + [Deepgram Words JSON]
        │
        ▼
[Rust Sidecar: t7_prosody.rs] ─── (Checks AUTOSHORTS_T7_PROSODY)
        │
        ├── [Cache Hit Check: %APPDATA%/com.autoshorts.desktop/t7_cache/{hash}_{version}.json]
        │       └─ Yes ──► Return Vec<T7Boundary>
        │
        └── [Cache Miss / Corrupt]
                │
                ▼ (proc_guard::run_bounded 30s timeout)
        [Python Sidecar: scripts/t7_prosody.py --predict]
                ├── Loads scripts/models/t7_prosody_v1.txt
                │       └─ Missing/Corrupt ──► Return empty boundaries
                └── Extracts 8 prosodic features ──► Predicts pBoundary
                        │
                        ▼
                [Cached to %APPDATA% and returned to Rust]
        │
        ▼
[Caption Segmentation: src/captions.rs: segment_speech_chunks]
        ├── If t7_boundaries is Some and p_boundary >= 0.65:
        │       Reduce pause threshold 1.5s -> 0.9s for that gap
        ├── If t7_boundaries is None, empty, or p_boundary < 0.50:
        │       Keep pause threshold 1.5s (Byte-identical to baseline)
        └── Hard guardrails strictly enforced:
                - 1-3 word span protected
                - Never break on comma
                - Never break mid-sentence unless terminator present or gap > 2.5s
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | R1.1 Complete 8 Prosodic Features | Compute `pauseSec`, `endsSentence`, `endsClause`, `speakerChange`, `f0SlopePre`, `energyDropDb`, `vadSilenceFrac`, and `pBoundaryRule`. | M1 | ORIGINAL_REQUEST §R1 |
| 2 | R1.2 Label Mode Pipeline | `--label` mode ingesting source audio + transcript JSON + optional human annotations JSONL; outputs `scripts/data/t7_labels.jsonl` with weak label prior fallback. | M1 | ORIGINAL_REQUEST §R1 |
| 3 | R1.3 Predict Mode Sidecar | `--predict` mode performing inference via trained LightGBM model; graceful fallback to empty boundaries on missing/corrupt model. | M1 | ORIGINAL_REQUEST §R1 |
| 4 | R2.1 LightGBM Trainer & Per-Video Split | `scripts/t7_train.py` binary classifier training with early stopping on held-out per-video split to prevent speaker leakage. | M1 | ORIGINAL_REQUEST §R2 |
| 5 | R2.2 Safety Guard (<200 samples) | Refuse to train if labeled gaps < 200, outputting `NO LABELS — tagger not trained` and exit 0 without fabricating fake data. | M1 | ORIGINAL_REQUEST §R2 |
| 6 | R2.3 Model Artifacts & Metadata | Generate `scripts/models/t7_prosody_v1.txt` and `.meta.json` with featureVersion, threshold=0.65, classes, train stats (AUC, PR-AUC). | M1 | ORIGINAL_REQUEST §R2 |
| 7 | R3.1 Rust Sidecar Execution | `src/t7_prosody.rs` invoking `t7_prosody.py --predict` via `proc_guard::run_bounded` (30s timeout), typed `Vec<T7Boundary>`. | M2 | ORIGINAL_REQUEST §R3 |
| 8 | R3.2 Cache Management | SHA-256 source hash + model version cache under `%APPDATA%/com.autoshorts.desktop/t7_cache/`. Corrupt/missing cache handles gracefully. | M2 | ORIGINAL_REQUEST §R3 |
| 9 | R3.3 Graceful Degradation | Missing, failing, or timing-out model gracefully degrades to empty boundaries `vec![]` without panics. | M2 | ORIGINAL_REQUEST §R3 |
| 10 | R4.1 Caption Chunking Modulation | In `captions.rs:segment_speech_chunks`, accept optional `t7_boundaries`. If `p_boundary >= 0.65`, reduce pause threshold from 1.5s to 0.9s for that gap. | M2 | ORIGINAL_REQUEST §R4 |
| 11 | R4.2 Hard Guardrails Preservation | Never break inside 1–3 word span, never break on comma, never break mid-sentence unless terminator present or gap > 2.5s. | M2 | ORIGINAL_REQUEST §R4 |
| 12 | R4.3 Byte-Identical Fallback | When `t7_boundaries` is None, confidence < 0.50, or model missing, caption chunking output is identical to baseline. | M2 | ORIGINAL_REQUEST §R4 |
| 13 | R5.1 Scoped Module Wiring | `src/lib.rs`: `mod t7_prosody;`, `pub fn t7_prosody_enabled() -> bool`, re-export `T7Boundary` and `predict_t7_boundaries`. | M2 | ORIGINAL_REQUEST §R5 |
| 14 | R5.2 Feature Flag Gating | `AUTOSHORTS_T7_PROSODY` env var (default: enabled; `0/false/off` = disabled). | M2 | ORIGINAL_REQUEST §R5 |
| 15 | T.1 Python Unit Test Suite | `scripts/test_t7_prosody_suite.py` with 10+ (12) tests covering feature extraction, labeling, training, evaluation, and safe refusal on <200 samples. | M1 | ORIGINAL_REQUEST §Acceptance |
| 16 | T.2 Rust Integration Test Suite | `tests/t7_prosody_suite.rs` with 5+ tests (missing model fallback, threshold reduction at p>=0.65, boundary ignored at p<0.50, hard guardrails preserved, flag disabled path). | M2 | ORIGINAL_REQUEST §Acceptance |
| 17 | T.3 Regression & Baseline Suite | Existing 14 caption template tests, 45 captions.rs unit tests, and full crate `cargo test -p autoshorts --lib` passing with zero regressions. | M3 | ORIGINAL_REQUEST §Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Python ML Subsystem (R1, R2) | Extend `scripts/t7_prosody.py`, create `scripts/t7_train.py` with <200 guard, create `scripts/test_t7_prosody_suite.py` (22 tests) | Survey | DONE |
| M2 | Rust Engine & Modulation (R3, R4, R5) | Implement `src/t7_prosody.rs`, wire `src/lib.rs` (`render_flat_clip_for_candidate` @ line 2353), modulate `src/captions.rs` (`segment_speech_chunks` & `group_t7_phrase_units`), create `tests/t7_prosody_suite.rs` (8 tests) | M1 | DONE |
| M3 | Full E2E Acceptance & Regressions | Run all suites (`cargo test --test t7_prosody_suite` 8/8 PASS, python `test_t7_prosody_suite.py` 22/22 PASS, `cargo test -p autoshorts --lib` 471/471 PASS, `cargo test --test pause_intel_suite` 8/8 PASS, `npm run build` PASS, byte-identical fallback verified) | M1, M2 | DONE |

## Interface Contracts

### 1. Python CLI ↔ Sidecar Ingestion Contract
- **Predict Mode**:
  ```bash
  python scripts/t7_prosody.py <source_path> <start_ms> <end_ms> <words_json_path> --predict [--model <model_path>]
  ```
  - Standard output: JSON object:
    ```json
    {
      "modelScored": true,
      "featureVersion": "t7_prosody_v1",
      "boundaries": [
        { "gapIdx": 0, "pBoundary": 0.72, "isBoundary": 1, "features": { ... } }
      ]
    }
    ```
  - If model missing or error:
    ```json
    {
      "modelScored": false,
      "model": null,
      "boundaries": []
    }
    ```
- **Label Mode**:
  ```bash
  python scripts/t7_prosody.py <source_path> <start_ms> <end_ms> <words_json_path> --label [--annotations <annotations_jsonl>] [--out <out_path>]
  ```
- **Train Mode**:
  ```bash
  python scripts/t7_train.py [--data <labels_jsonl>] [--models_dir <dir>]
  ```
  - If < 200 samples: prints `NO LABELS — tagger not trained` and exits code 0.

### 2. Rust Sidecar ↔ Rust Captions Contract
- **Type**:
  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
  pub struct T7Boundary {
      #[serde(rename = "gapIdx", alias = "gap_idx")]
      pub gap_idx: usize,
      #[serde(rename = "pBoundary", alias = "p_boundary")]
      pub p_boundary: f64,
      #[serde(default)]
      pub features: Option<serde_json::Value>,
  }
  ```
- **Function**:
  ```rust
  pub fn predict_t7_boundaries(
      source_path: &Path,
      start_sec: f64,
      end_sec: f64,
      words: &[TranscriptWord],
  ) -> Vec<T7Boundary>;
  ```
- **Segmentation Signature**:
  ```rust
  pub fn segment_speech_chunks<'a>(
      words: &'a [(usize, &TranscriptWord)],
      max_words: usize,
      pause_threshold: f64,
      max_chunk_dur: f64,
      t7_boundaries: Option<&[T7Boundary]>,
  ) -> Vec<Vec<(usize, &'a TranscriptWord)>>;
  ```

## Code Layout
- `autoshorts/src-tauri/scripts/t7_prosody.py` (Owned by M1)
- `autoshorts/src-tauri/scripts/t7_train.py` (Owned by M1)
- `autoshorts/src-tauri/scripts/test_t7_prosody_suite.py` (Owned by M_TEST)
- `autoshorts/src-tauri/src/t7_prosody.rs` (Owned by M2)
- `autoshorts/src-tauri/src/lib.rs` (Owned by M2)
- `autoshorts/src-tauri/src/captions.rs` (Owned by M3)
- `autoshorts/src-tauri/tests/t7_prosody_suite.rs` (Owned by M_TEST)
