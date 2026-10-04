# Software & Model License Audit — Phase 4 (Advanced AI + Production Hardening)

**Date:** 2026-09-29
**Scope:** Phase 4 additions: OpenRouter VLM candidate scoring, PANNs reaction metadata, candidate redundancy detection, deterministic render QA, sidecar packaging readiness.
**Re-audited:** 2026-09-29 (post–DeepFilterNet2 removal). DeepFilterNet2 is REMOVED and is no longer in scope as a shipped component.
**Workspace:** `D:\College\Autoshorts 11.0`
**Verification method:** licenses read from the installed wheel metadata in the project venv (`*.dist-info/METADATA`), upstream repositories, and model cards. Atria research cross-checked.
**Disclaimer:** engineering technical audit; not legal advice.

---

## 1. Inventory & Verified Licenses

| Component | Version | Verified License | Source of verification | Commercial use | Bundled |
|---|---|---|---|---|---|
| **VLM (Qwen3.8 27B, cloud)** | `qwen/qwen3.8-27b:free` | **Provider-hosted (OpenRouter)** — weights not shipped | OpenRouter model registry | Yes (as a remote API) | **No — remote API only, no weights bundled or downloaded** |
| **DeepFilterNet2** | n/a | **REMOVED — not a component of this release** | n/a | n/a | **No** |
| **PANNs (CNN14-16k)** | 80M params | **MIT** (Qiuqiang Kong et al.) | PyPI `panns-inference` wheel / GitHub | Yes | No (lazy download) |
| **Sentence-Transformers (TF-IDF fallback)** | N/A | **MIT/Apache-2.0** | scikit-learn, scipy | Yes | Already present |
| **Render QA (deterministic)** | N/A | **Project code** | n/a | Unrestricted | n/a |
| **Candidate Redundancy (TF-IDF)** | N/A | **Project code** | n/a | Unrestricted | n/a |
| **Sidecar packaging (Tauri)** | N/A | **MIT** (Tauri) | Tauri crate metadata | Yes | Yes |

---

## 2. Atria Compliance

| Atria recommendation | Status | Notes |
|---|---|---|
| OpenRouter VLM candidate scoring (Qwen3.8 27B) | **Implemented** | Optional, flag-gated, remote API, cached per (source, candidate, model, prompt) |
| DeepFilterNet2 degraded-source enhancement | **REMOVED / INTENTIONALLY EXCLUDED** | Explored experimentally in Phase 4, then removed. No source, crate, wiring, or test remains. |
| PANNs Cnn14-16k reaction metadata | **Implemented** | Source-level pass, cached, hook intelligence integration |
| Candidate redundancy detection (transcript embeddings) | **Implemented** | TF-IDF cosine similarity, deterministic, config-driven |
| Deterministic render QA | **Implemented + PRODUCTION (mandatory)** | 13 real checks, post-render, report-only, no silent rewrite; gates clip acceptance |
| Sidecar packaging readiness | **Documented** | Tauri sidecar/externalBin strategy, lazy model download |
| No giant end-to-end video models | **Compliant** | Only specialized small models |
| No ML for loudness/A/V sync/caption geometry | **Compliant** | Deterministic paths unchanged |
| Payoff endpoint lock preserved | **Compliant** | LLM → transcript → payoff_end → candidate_end |
| Hard containment preserved | **Compliant** | No ML bypasses containment |

---

## 3. Model Details & Bundling Decisions

### VLM (Qwen3.8 27B via OpenRouter)
- **Deployment:** Remote API (`https://openrouter.ai`), model id `qwen/qwen3.8-27b:free`
- **Bundling:** **NOT BUNDLED — no weights are shipped or downloaded.** Inference happens entirely provider-side.
- **License exposure:** governed by the OpenRouter provider terms at call time, not by redistribution of weights in this project. No model weight is distributed with AutoShorts 11.0.
- **Cache:** only JSON score results, in `~/.cache/autoshorts/vlm/`
- **Fallback:** If the API key is missing, the network fails, or the model is unavailable → heuristic scoring, then the existing deterministic candidate pipeline. The deterministic path never depends on this call.
- **Feature flag:** `AUTOSHORTS_VLM_SCORING` (default OFF)

### DeepFilterNet2 — REMOVED
- **Status:** Explored/integrated experimentally during Phase 4, then intentionally removed. It is **not** a component of AutoShorts 11.0.
- **No code, dependency, FFmpeg wiring, feature flag, model/cache configuration, or test remains.**
- **No license obligation is carried by this release** because nothing is bundled, downloaded, or invoked.
- Audio Intelligence, Smart Pacing audio features, BS.1770 loudness processing, and the audio validation allowlist are unaffected.

### PANNs (CNN14-16k)
- **License:** MIT (verified via `panns-inference` PyPI wheel)
- **Model weights:** ~300 MB (CNN14-16k checkpoint)
- **Bundling:** **LAZY DOWNLOAD** — downloaded on first use, cached in `~/.cache/autoshorts/panns/`
- **Runtime:** `panns-inference` Python package (torch backend)
- **Source-level pass:** Runs once per source video, caches reaction events
- **Feature flag:** `AUTOSHORTS_PANNS_REACTIONS` (default OFF)

### Candidate Redundancy Detection
- **Implementation:** TF-IDF cosine similarity on transcript text (no external model)
- **Deterministic:** Pure Python/Rust, no ML randomness
- **Configurable:** Similarity threshold (default 0.85), min time gap (default 2.0s)
- **Feature flag:** `AUTOSHORTS_CANDIDATE_REDUNDANCY` (default OFF)

### Render QA
- **Implementation:** Deterministic FFmpeg/ffprobe checks
- **13 checks:** output exists, decodable, video stream, resolution, FPS/timing, A/V sync, loudness, face containment (crop bounds), crop validity, frame integrity, caption presence (band luma), silent audio, black frames
- **Report-only:** Never silently rewrites; reports critical/major/minor/info
- **Feature flag:** `AUTOSHORTS_RENDER_QA` (default OFF)

### Sidecar Packaging
- **Tauri sidecar/externalBin:** Python scripts registered as sidecars
- **Model download:** Lazy, with SHA256 verification, cached in OS cache dir
- **Python runtime:** Bundled via `tauri.conf.json` `externalBin` or user-provided via `AUTOSHORTS_PYTHON`
- **External binaries:** FFmpeg, ffprobe, yt-dlp via externalBin or PATH

---

## 4. Not Integrated (License/Architecture Posture Preserved)

| Component | Status | Reason |
|---|---|---|
| OpenSMILE | **NOT SUITABLE** | Commercial audEERING licensing (Atria flag) |
| Giant end-to-end video models | **Rejected** | Atria + Phase 4 prompt prohibit |
| ML for loudness/A/V sync/caption geometry | **Rejected** | Deterministic paths preserved |
| VLM as authoritative decision maker | **Rejected** | VLM is advisory only |
| PANNs as automatic editor | **Rejected** | Metadata only, hook intelligence integration |

---

## 5. Feature Flags Summary

| Flag | Default | Description |
|---|---|---|
| `AUTOSHORTS_VLM_SCORING` | OFF | OpenRouter VLM candidate scoring |
| `AUTOSHORTS_PANNS_REACTIONS` | OFF | PANNs reaction metadata |
| `AUTOSHORTS_CANDIDATE_REDUNDANCY` | OFF | Candidate redundancy detection |
| `AUTOSHORTS_RENDER_QA` | OFF | Deterministic render QA |
| `AUTOSHORTS_VLM_MODEL` | `qwen/qwen3.8-27b:free` | VLM model selection (OpenRouter model id) |
| `AUTOSHORTS_PANNS_MODEL` | `cnn14_16k` | PANNs model selection |
| `AUTOSHORTS_PANNS_THRESHOLD` | `0.5` | PANNs confidence threshold |

Note: `AUTOSHORTS_DEEPFILTER` no longer exists — DeepFilterNet2 was removed from the architecture.

---

## 6. Notes

- All Phase 4 components are **optional at runtime** behind feature flags; no license obligation changes the render path.
- Model downloads are **lazy, cached, and SHA256-verified**; missing model = documented fallback.
- The existing Ultralytics AGPL-3.0 issue (yolo11n.pt) remains unchanged from Phase 1.
- All Phase 3 data-blocked items (learned Smart Pacing, learned crop aesthetic, T7 prosodic model) remain untouched.
- Deterministic AutoShorts engine retains full authority; Phase 4 models are advisory signals only.