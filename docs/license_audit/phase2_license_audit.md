# Software & Model License Audit — Phase 2 (Speaker Intelligence)

**Date:** 2026-09-29
**Scope:** All Phase 2 models, runtimes, and services: OSNet Re-ID (`osnet_x1_0`) + torchreid, Deepgram diarization (nova-3, cloud API), pyannote fallback, WhisperX fallback, active-speaker fusion (in-repo, no third-party models).
**Workspace:** `D:\College\Autoshorts 11.0`
**Disclaimer:** This document is an engineering technical audit of upstream licenses as recorded during Phase 2 validation. It is not legal advice.

---

## 1. Inventory

| Component | Artifact / Source | How acquired | Local notice present? |
|---|---|---|---|
| Re-ID model code | `torchreid` 0.2.5 (PyPI wheel, MIT) | `pip install torchreid` into project venv | Yes — `torchreid-0.2.5.dist-info` records `License: MIT` + LICENSE file |
| Re-ID weights | `osnet_x1_0_imagenet.pth` (~11 MB) | Auto-downloaded by torchreid model zoo to `~/.cache/torch/checkpoints/` | No (cached artifact; not shipped in repo) — **VERIFIED downloaded and functional during Phase 2 e2e** (gallery of 34 embeddings extracted from real media; measured ~60–92 ms per crop on CPU, batch 16) |
| Diarization service | Deepgram `nova-3` (cloud API, `diarize=true`) | API key from `autoshorts/.env` | n/a (service) |
| Diarization fallback | `pyannote` (optional backend) | NOT installed in project venv | n/a (not bundled) |
| Diarization fallback | `whisperx` (optional backend) | NOT installed in project venv | n/a (not bundled) |
| Active-speaker fusion | `active_speaker_fusion.py`, in-repo | project code | n/a |

---

## 2. Component Analysis

### 2.1 torchreid + OSNet weights — **MIT**
- **Authoritative source:** PyPI `torchreid-0.2.5.dist-info/METADATA` states `License: MIT` (upstream repository: `KaiyangZhou/deep-person-reid`).
- **Implication:** Permissive. Commercial use, modification, and redistribution permitted with copyright notice preservation.
- **Weights:** `osnet_x1_0` ImageNet-pretrained weights are distributed through the torchreid model zoo under the same project. MIT applies to the project's code and released model artifacts.
- **Caveat (documented):** OSNet weights are downloaded at runtime from a public URL on first use. For offline/air-gapped operation, bundle the weights file with attribution, or accept a first-run network dependency. AutoShorts' current integration degrades gracefully when the model is unavailable (HAS_TORCHREID=False → Re-ID disabled → positional association), so bundling is optional.
- **Verdict: safe for commercial bundling (MIT), with notice preservation.**

### 2.2 Deepgram nova-3 diarization — **Commercial cloud service**
- **Nature:** Paid/ metered SaaS API. No weights are shipped; audio is uploaded for processing.
- **Implications:**
  - Requires an active API key and account (present in the developer environment only).
  - **NOT local/offline operation.** Audio leaves the machine — privacy/consent implications for user media must be surfaced in product documentation.
  - Usage cost and availability are governed by Deepgram's commercial terms (Master Service Agreement / data handling policy).
- **Fallback posture:** When the key is absent, invalid (verified: HTTP 401), or the network is unavailable, the sidecar exits non-zero and the Rust engine substitutes an empty diarization fallback (`model="fallback"`, zero fabricated evidence). The pipeline continues.
- **Verdict: EXPERIMENTAL/optional enhancement, not bundlable, not offline-capable. Product must treat cloud diarization as opt-in.**

### 2.3 pyannote fallback — **NOT bundled (gated weights)**
- pyannote *code* is MIT, but `pyannote/segmentation-3.0` *weights* are gated on HuggingFace (user must accept conditions; HF token required).
- Current status: not installed in the venv; the backend exists in `speaker_diarization.py` as a fallback chain member.
- **Verdict: keep FALLBACK ONLY. Do not bundle weights. Token requirements documented.**

### 2.4 WhisperX fallback — **NOT bundled**
- WhisperX (BSD-style license upstream) depends on faster-whisper models (CC-BY-NC variants for some sizes — non-commercial restriction risk for larger models) and pyannote segmentation (gated, see 2.3).
- Current status: not installed.
- **Verdict: FALLBACK ONLY; do not bundle; license-check per model size before any commercial activation.**

### 2.5 Active-speaker fusion / speaker mapping / cache — **project code**
- Pure in-repo Python/Rust; no third-party models; deterministic algorithms only.
- **Verdict: unrestricted.**

---

## 3. Runtime dependency licenses (new in Phase 2 venv)

| Package | Version | License | Notes |
|---|---|---|---|
| torchreid | 0.2.5 | MIT | Re-ID model factory |
| gdown | (transitive) | Apache-2.0 | torchreid dataset import path |
| tensorboard | (transitive) | Apache-2.0 | torchreid engine import path |

Pre-existing dependencies (torch, ultralytics [AGPL — see yolo11n audit], OpenCV [Apache-2.0]) are unchanged by Phase 2. Ultralytics remains under its documented AGPL-3.0 dual-license posture (see `yolo11n_license_audit.md`) — Phase 2 adds no new Ultralytics usage surface.

---

## 4. Commercial bundling decision

| Component | Bundlable? | Condition |
|---|---|---|
| OSNet Re-ID + torchreid | **Yes** | Preserve MIT notice; optionally bundle weights for offline use |
| Deepgram diarization | **No** (service) | Opt-in, account-gated, cloud-only |
| pyannote | **No** (weights gated) | Fallback only, user-supplied token |
| WhisperX | **No** | Model-license risk; fallback only |
| Fusion / mapping / cache | **Yes** | In-repo code |

**Phase 2 posture maintained:** every third-party model failure (missing, unreachable, unlicensed-for-use) must route to the deterministic in-repo fallback. Verified by the feature-flag and failure tests in the Phase 2 report.

---

## 5. Verification record (Phase 2 closeout)

- `torchreid` 0.2.5 wheel import verified in the project venv (Re-ID model loaded successfully on CPU).
- OSNet weights verified present at `~/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth` ("Successfully loaded imagenet pretrained weights").
- Deepgram live diarization verified on real media (90 s, 2 speakers, 12 segments, confidence 0.884; cold 40.38 s, cached 0.55 s).
- pyannote / whisperx remain NOT installed; the fallback chain members exist in `speaker_diarization.py` and are exercised only when a user-supplied HF token is present (documented FALLBACK ONLY posture unchanged).
- All Phase 2 failure modes verified to fail safely (`phase2_failure_tests.py`: 12/12).
