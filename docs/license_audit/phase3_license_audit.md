# Software & Model License Audit — Phase 3 (Learned Signals)

**Date:** 2026-09-29
**Scope:** Phase 3 additions: Silero VAD, LightGBM, PySceneDetect, onnxruntime, librosa (already present), the trained pause classifier (in-repo artifact), and the T7/crop pipelines (in-repo code, no third-party models).
**Workspace:** `D:\College\Autoshorts 11.0`
**Verification method:** licenses read from the installed wheel metadata in the project venv (`*.dist-info/METADATA`), not assumed from memory. Atria research cross-checked.
**Disclaimer:** engineering technical audit; not legal advice.

---

## 1. Inventory & Verified Licenses

| Component | Version | Verified License | Source of verification | Commercial use |
|---|---|---|---|---|
| lightgbm | 4.7.0 | **MIT** (`License-Expression: MIT`) | wheel METADATA | Yes |
| scenedetect (PySceneDetect) | 0.7.1 | **BSD-3-Clause** (`License-Expression`) | wheel METADATA | Yes |
| silero-vad | 6.2.3 | **MIT** (classifier + LICENSE file) | wheel METADATA | Yes |
| silero-vad model weights (`silero_vad.onnx` / JIT, ~2 MB) | — | **MIT** (upstream `snakers4/silero-vad` repository distribution) | upstream repo policy | Yes |
| onnxruntime | 1.30.0 | **MIT** | wheel METADATA | Yes |
| librosa | 1.0.0 | ISC (pre-existing dependency) | upstream | Yes |
| numpy / scipy / scikit-learn | pre-existing | BSD-3 | upstream | Yes |
| Pause classifier (`pause_classifier_v1.txt`) | in-repo artifact | **AutoShorts own model** (trained from the project's own pipeline on workspace media) | n/a | Unrestricted |
| T7 prosody pipeline / crop scorer / scene engine glue | in-repo code | project code | n/a | Unrestricted |

## 2. Atria Compliance

- **openSMILE:** NOT used (Atria flags audEERING commercial-license risk). Feature extraction is numpy/librosa-family only. ✔
- **Silero VAD v6.x:** the exact Atria-recommended implementation (6.2.3, MIT, ~2 MB, CPU). ✔
- **LightGBM:** the Atria-preferred model class (Apache-2.0 upstream project; PyPI wheel is MIT-expression). ✔
- **PySceneDetect AdaptiveDetector:** the Atria-recommended scene detector (BSD-3, verified 0.7.1). ✔

## 3. Not Integrated (license posture preserved)

| Component | Status | Reason |
|---|---|---|
| openSMILE | NOT SUITABLE | Commercial audEERING licensing (Atria) |
| CLIP ViT-B/32 aesthetic head | Deferred | No rated crop dataset exists (data-blocked); no weights shipped |
| u2netp saliency | Deferred | Weights not bundled; advisory feature postponed with the learned head |
| Any VLM / large end-to-end model | Rejected by design | Atria + Phase 3 prompt prohibit |

## 4. Bundling Decision

| Component | Bundlable? | Condition |
|---|---|---|
| lightgbm / scenedetect / silero-vad / onnxruntime | **Yes** | Preserve MIT/BSD-3 notices in distributions |
| Silero VAD weights | **Yes** | MIT; bundle for offline use (2 MB) |
| Trained pause classifier | **Yes** | Own artifact |
| Learned crop head / T7 prosody model | n/a | Data-blocked; nothing to bundle yet |

## 5. Notes

- The Silero VAD model downloads on first use via the `silero-vad` package; for offline/air-gapped deployment, pre-seed the torch hub cache or bundle the ONNX file. Missing model = documented fallback (VAD features disabled; deterministic engine continues).
- All Phase 3 components are optional at runtime behind feature flags; no license obligation changes the render path.
