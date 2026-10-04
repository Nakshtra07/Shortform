# E2E Test Infra: AutoShorts 11.0 T7 Prosody Integration

## Test Philosophy
- Opaque-box, requirement-driven.
- Verifies full pipeline behavior against `ORIGINAL_REQUEST.md` constraints:
  - 10+ Python unit tests covering features, labeling, training guardrails, and predict mode.
  - 5+ Rust integration tests covering fallback on missing model, soft threshold modulation, low confidence ignore, hard guardrails preservation, and feature flag disabling.
  - Zero regressions across existing Rust library tests (457+ tests) and caption template tests (14 tests).
  - Byte-identical caption output when T7 is disabled or inactive.

## Feature Inventory & Test Mapping
| # | Feature | Source | Test Target | Tier |
|---|---------|--------|-------------|:----:|
| 1 | 8 Prosodic Features | ORIGINAL_REQUEST §R1 | `test_t7_prosody_suite.py:TestT7ProsodyFeaturePipeline` | Tier 1 |
| 2 | Label Mode & Weak Priors | ORIGINAL_REQUEST §R1 | `test_t7_prosody_suite.py:TestT7ProsodyLabeling` | Tier 1 |
| 3 | Predict Mode & Fallback | ORIGINAL_REQUEST §R1 | `test_t7_prosody_suite.py:TestT7ProsodyInference` | Tier 1 |
| 4 | LightGBM Trainer & Per-Video Split | ORIGINAL_REQUEST §R2 | `test_t7_prosody_suite.py:TestT7ProsodyTraining` | Tier 1 |
| 5 | <200 Samples Safe Refusal | ORIGINAL_REQUEST §R2 | `test_t7_prosody_suite.py:test_08_safe_refusal_under_200_samples` | Tier 2 |
| 6 | Missing Model Rust Fallback | ORIGINAL_REQUEST §R3, Acceptance | `tests/t7_prosody_suite.rs:test_missing_model_fallback` | Tier 1 |
| 7 | Threshold Reduction (p >= 0.65) | ORIGINAL_REQUEST §R4, Acceptance | `tests/t7_prosody_suite.rs:test_threshold_reduction_at_p_ge_065` | Tier 1 |
| 8 | Boundary Ignored (p < 0.50) | ORIGINAL_REQUEST §R4, Acceptance | `tests/t7_prosody_suite.rs:test_boundary_ignored_at_p_lt_050` | Tier 2 |
| 9 | Hard Guardrails Preserved | ORIGINAL_REQUEST §R4, Acceptance | `tests/t7_prosody_suite.rs:test_hard_guardrails_preserved` | Tier 2 |
| 10 | Flag Disabled Path | ORIGINAL_REQUEST §R3, R5, Acceptance | `tests/t7_prosody_suite.rs:test_flag_disabled_path` | Tier 1 |
| 11 | Byte-Identical Invariant | ORIGINAL_REQUEST §R4, Acceptance | `tests/t7_prosody_suite.rs:test_byte_identical_fallback` | Tier 4 |
| 12 | Regression: 14 Caption Tests | ORIGINAL_REQUEST §Acceptance | `scripts/test_caption_templates_suite.py` | Tier 4 |
| 13 | Regression: Full Rust Library | ORIGINAL_REQUEST §Acceptance | `cargo test -p autoshorts --lib` | Tier 4 |

## Test Architecture
- **Rust Integration Test Target**:
  - File: `autoshorts/src-tauri/tests/t7_prosody_suite.rs`
  - Runner: `cargo test -p autoshorts --test t7_prosody_suite`
- **Python Unit Test Target**:
  - File: `autoshorts/src-tauri/scripts/test_t7_prosody_suite.py`
  - Runner: `python -m unittest autoshorts/src-tauri/scripts/test_t7_prosody_suite.py`
- **Existing Regression Run**:
  - `cargo test -p autoshorts --lib`
  - `python -m unittest scripts/test_caption_templates_suite.py`
