# AutoShorts 11.0 — Complete Deep Learning / Machine Learning Research Report

**Research-only architectural study for future implementation**
Date of research: September 2026 · Codebase verified against `autoshorts/` working tree

---

## 0. Executive Summary

AutoShorts 11.0 is a Tauri/Rust core with Python analysis sidecars that turns long-form podcast/interview video into vertical shorts. It already has genuinely strong deterministic machinery: semantic boundary snapping, boundary optimization, Smart Pacing v1/v2, a 5,555-line speaker tracking and reframing engine, caption ASS generation, and a validated render path. The test suite is green (16/16 Python suites, 281/282 Rust tests — the one failure is a test-only env-var race, not product code).

The central finding of this study is **not** "add AI everywhere." It is that three specific things are true simultaneously:

1. **The highest-leverage ML opportunity requires no new model and no GPU.** The single biggest quality win available — candidate discovery — is currently done by asking an LLM to read a transcript and emit timestamps. Two 2025–26 papers (REZE, Rhapsody) independently measured this exact pattern and found it is the *weakest* way to extract highlights. REZE's clip-scoring approach beat timestamp-generation by **3.5× mAP** using the *same* frozen model. That is a prompt/architecture change, deployable on the existing 8 LLM providers, today.

2. **The most embarrassing defect is a fake analyzer.** `multimodal_hook_analyzer.py:186-264` (`analyze_acoustic_signals`) never opens the audio file. Its `energy_change`, `pitch_change`, `laughter`, and `pause_emphasis` outputs are derived from punctuation and string matching. Its evidence strings assert claims like "rising vocal pitch contour" that were never computed. This is a correctness bug wearing an ML costume, and it feeds the hook composite score. Fixing it with real audio features (librosa, ~zero new dependencies) is the top quick win.

3. **The biggest visible quality lever is not ML at all — it is per-scene camera-mode selection with path smoothing.** Academic learned-crop work in 2025–26 is almost entirely MLLM-based scorers, which are the wrong fit for a local tool. The practical upgrade is a composition *scorer* (CLIP + small aesthetic head + u2netp saliency) feeding the existing hard constraints, plus **OSNet re-ID embeddings** (0.2–0.6M params) to fix the system's single most fragile assumption: that "nearest genuine face to crop center" equals "the person we are tracking."

**What must not change:** the deterministic safety layer. Containment is hard, centering is soft, deadbands prevent jitter, per-shot scale resets, the validator splits rather than rewrites, full-clip sampling, and the LLM-recommends / Rust-enforces boundary discipline. Every recommendation in this report is shaped to sit *in front of* those guarantees, never replace them.

**License warning up front:** the shipped `yolo11n.pt` in `src-tauri/models/` is **AGPL-3.0**. That is a copyleft obligation on the distributed binary that must be resolved before any further model bundling, regardless of this report's recommendations.

---

## Phase 1 — Verified Architecture Map

The prompt's hypothesized pipeline was checked against actual code. The real order:

```
YouTube download (youtube.rs)
  ↓
Transcription (transcription.rs)
  │  Deepgram Nova-3 cloud, OR local Whisper "base"
  │  ⚠ No diarization — every word labeled S1
  ↓
LLM candidate discovery (llm.rs)
  │  Sliding window over transcript, 8 providers, temp 0.15, DeepSeek default
  │  Emits CandidateDraft { start, end, hook/payoff fields, ... }
  ↓
Hook / payoff token alignment
  │  Accept at ≥0.50 confidence
  ↓
Deterministic boundary snapping (lib.rs:1135-1465, snap_to_semantic_boundaries_with_hook_anchor)
  │  LLM recommends; Rust enforces
  ↓
Boundary optimization (boundary.rs)
  ↓
Smart Pacing v1/v2 (smart_pacing.py + breath_detect.py)
  ↓
Speaker tracking / framing (speaker_tracker.py, "Smart Reframing Engine v12.0")
  ↓
Active speaker (mouth-motion heuristic in the tracker)
  ↓
Caption ASS generation
  ↓
FFmpeg render (1080×1920, or square-in-9:16 adaptive)
```

**Corrections to the hypothesized map:**

- The prompt's map puts AUDIO INTELLIGENCE between SMART PACING and CAPTIONS, and FRAMING after CAPTIONS. In reality **framing/active-speaker runs before captions**, and audio intelligence is a conditional enhancement stage applied at render time, not a sequential analysis stage in the discovery path.
- The prompt's map shows "SPEAKER / DIARIZATION" as a distinct stage after transcription. **There is no such stage.** Transcription returns a single speaker track (`S1` for everything in the local path); diarization does not exist. Speaker identity in the visual pipeline is purely positional.
- The T7 caption chunking described in Phase 3-G exists as word-timing + rhythm based phrase chunking feeding phrase-paired rolling display.

### Verified facts that matter for the rest of this report

| # | Fact | Evidence |
|---|---|---|
| F1 | **The "acoustic" multimodal analyzer is fake.** `analyze_acoustic_signals` never opens audio. `energy_change`/`pitch_change` come from `?`/`!` punctuation; `laughter` from `"haha"`/`"हंस"` substring matches; `pause_emphasis` from timestamps. Evidence strings assert "rising vocal pitch contour" etc. | `multimodal_hook_analyzer.py:186-264` |
| F2 | **The visual stage does real signal processing** (OpenCV frame diff + Haar cascade), unlike the audio stage. | same file, visual stage |
| F3 | **No subject identity / re-ID anywhere.** Framing keys on "nearest genuine face to crop center." `SpeakerIdentity` is position-based, not appearance-based. The brain doc's own §18.4 flags this as the natural follow-up. | `speaker_tracker.py` |
| F4 | **No Host↔diarization mapping.** The LLM's `hookSpeaker: "Host"\|"Guest"` and Deepgram's `S1/S2` labels are never reconciled. | `llm.rs`, `transcription.rs` |
| F5 | **Active speaker = mouth-motion heuristic.** 32×20 resized mouth-ROI mean absdiff, threshold ≥4.0, dominance ×1.20. Fires on laughing listeners; illumination-sensitive; no audio reference. | `speaker_tracker.py` |
| F6 | **Only 18 of ~30+ `CandidateDraft` columns are persisted** in the `candidates` DB table. Multimodal, hook-quality, closure, and endpoint-state data is discarded at `replace_candidates`. | `db.rs` |
| F7 | **Absolute pixel thresholds baked to 1080p.** Back-of-head detection uses fixed areas 18000/14000/35000 px². | `speaker_tracker.py` |
| F8 | **Hardcoded absolute paths** to `d:\College\Autoshorts 5.0\` in `find_yolo_model` (`speaker_tracker.py:111-112`); `find_speaker_tracker_script` (`media.rs:272`) uses a CWD-relative path and **silently falls back to a center crop at x=656** when it fails. | `speaker_tracker.py`, `media.rs` |
| F9 | **Dead config constants** in `speaker_tracker.py` (FAST_TRANSITION_DUR, DRIFT_THRESH_PX, SALIENCY_* table, several FramingConfig fields) — the run loop re-declares them as locals, so editing the config has no effect. | `speaker_tracker.py` |
| F10 | **`tauri.conf.json` has no sidecar/externalBin bundling.** Python sidecars are loose scripts invoked via a discovered `python`/`.venv`. Distribution currently assumes a dev-machine environment. | `tauri.conf.json` |
| F11 | **Authors' own caveats:** head-extent margins (`fx−0.12·fw … fx+1.12·fw`) documented as "not re-derived from first principles"; breath thresholds "speaker-corpus-calibrated" (i.e., one speaker). | `speaker_tracker.py` comments |
| F12 | **Brain doc is stale.** Internally titled "AUTOSHORTS 8.0," with ~8+ wrong numeric claims (Rust test counts, DualFrame suite size). Code is authoritative. | `11.0 brain.md` vs. working tree |
| F13 | **License landmine already shipped:** `yolo11n.pt` is AGPL-3.0 and is bundled in `src-tauri/models/`. InsightFace-style face weights are typically non-commercial. | model card, `src-tauri/models/` |

**Test baseline verified this session:** 16/16 Python suites green (smart_pacing 55/55, SP2 108/108, audio_intelligence 48/48, hook_closure 47/47, hook_ending 29/29, caption_qa 100 checks, active_speaker 61/61, dualframe 55/55, applied_features, multimodal_hook). `cargo test`: 281 passed, 1 failed (`pacing::tests::test_v2_disabled_keeps_v1` — a test-only env-var concurrency race; passes with `--test-threads=1`), 1 ignored. `npm run build` exit 0.

---

## Phase 2 — Current System Weakness Analysis

For each subsystem: what it does, how it decides, where it fails, and whether ML is the right answer.

### 2.1 Transcription

- **Does:** Deepgram Nova-3 (cloud) or local Whisper `base`. Word-level timestamps.
- **Type:** Neural (third-party or bundled), deterministic given the same audio.
- **Fails on:** accented/multilingual speech (Hindi/Hinglish support is a stated requirement — R7); no diarization means every word is `S1`; the local Whisper fallback is deliberately tiny (`base` ≈ 74M).
- **Root cause:** missing capability, not a bad threshold. **F4** — the absence of any speaker label downstream means the LLM's `hookSpeaker` guess and the visual tracker's positional identity live in unreconcilable universes.
- **ML realistic?** Yes, and it is the cleanest data-side fix in the whole study. A modern local diarization/ASR stack (WhisperX, or Nemotron-3-Diarization) removes the `S1`-only limitation at a known, bounded cost. See §3.C.

### 2.2 Candidate discovery (llm.rs)

- **Does:** sliding window over the transcript; an LLM proposes candidate segments with hook/payoff annotations.
- **Type:** LLM-based.
- **Fails on:** exactly what the literature says timestamp-generation fails on — the model has to *simultaneously* understand content and perform precise time arithmetic. Rhapsody (COLM 2025) measured GPT-4o zero-shot on real podcast highlight extraction and found it performs **≈ a frequency baseline**, i.e. near-useless. Best systems in that shared setting reached only 49% hit rate.
- **Root cause:** not a threshold, not context — it is the *interaction pattern*. Asking a generative model to emit timestamps is the wrong query structure.
- **ML realistic?** Extremely. REZE (arXiv:2608.04480) showed that rephrasing the same model as a per-clip yes/no scorer and aggregating deterministically (mean-centering → Gaussian smoothing → Kadane/Otsu) yields **73.41 HIT@1** on QVHighlights, beating all fully supervised models — **43.28 vs 12.32 mAP (3.5×)** versus asking that same model for timestamps directly.
- **Classification: B (could benefit from ML signals), and specifically the discovery *protocol*, not the model.**

### 2.3 Hook / payoff alignment

- **Does:** token-level alignment of hook/payoff claims to transcript spans; accept ≥0.50.
- **Type:** LLM + deterministic alignment.
- **Fails on:** the acoustic half of its own input. Per **F1**, the multimodal hook analyzer's audio contributions are fabricated. The 0.15 audio weight in the hook composite is punctuation-driven — a `?` raises "pitch_change."
- **Root cause:** an unimplemented function masquerading as an implemented one.
- **ML realistic?** Modestly. Real acoustic features (RMS envelope, F0 via pyin/CREPE-lite, spectral flux) are the fix, and those are signal processing, not deep learning. A learned engagement/hook score is plausible later (see §3.H) but must never override the payoff endpoint architecture, per the prompt.
- **Classification: B, but the immediate fix is non-ML. Fix F1 first; a learned hook scorer is Phase 3.**

### 2.4 Boundary snapping (lib.rs) and boundary optimization (boundary.rs)

- **Does:** enforces sentence terminators, pause adequacy, max/min guideline, hook anchoring.
- **Type:** Deterministic. LLM recommends; Rust enforces.
- **Fails on:** genuinely hard cases the heuristics cannot see — breath vs. mid-thought, an imminent punchline, an unresolved Q&A arc. The LLM supplies `continuation_probability` etc., but the *decision* is made by fixed thresholds on those scores.
- **Root cause:** insufficient context at decision time, not weak math.
- **ML realistic?** The deterministic layer is correct as-is and is the system's crown jewel. ML value here is *upstream*: better probability estimates feeding the existing thresholds. **Do not replace the layer; feed it better signals.**
- **Classification: A (augment) — and even the augmentation is indirect.**

### 2.5 Smart Pacing v1/v2 (smart_pacing.py, breath_detect.py)

- **Does:** removes filler/breath/repetitive gaps under safety rules; v2 adds acoustic heuristics.
- **Type:** Heuristic + acoustic thresholds.
- **Fails on:** the universal question "is this pause removable or intentional?" Breath thresholds are calibrated on **one speaker's corpus** (**F11**). No pretrained model classifies removable vs. intentional pauses — this is genuinely an open problem with no off-the-shelf answer.
- **Root cause:** a per-speaker, per-room threshold generalizing to all speakers and rooms.
- **ML realistic?** Yes, and the data is free. "This gap was removed in the final edit" vs. "this gap survived" is a **weakly-supervised, domain-matched dataset that AutoShorts already generates every run** (modulo **F6** — the persistence gap currently throws away the data needed to build it). A LightGBM or 1D-CNN on librosa features (not openSMILE — that needs a commercial audEERING license) is the right size. Silero VAD v6.2.3 (MIT, ~2MB, <1ms/chunk, ROC-AUC 0.97) is the front-end.
- **Classification: B → C. Replace thresholds with a small specialized classifier, keep the safety rules.**

### 2.6 Speaker tracking / framing (speaker_tracker.py)

- **Does:** face/person detection, track persistence, adaptive reframing, DualFrame, containment.
- **Type:** CV (YOLO + Haar) + heavy heuristics.
- **Fails on:**
  - **Identity** (**F3**): no appearance embedding. Reappearance after occlusion, camera cuts, and two-person stability all rest on positional proximity.
  - **Active speaker** (**F5**): mouth-motion absdiff fires on laughing listeners, is illumination-sensitive, and has no audio reference at all.
  - **Resolution coupling** (**F7**): back-of-head areas in absolute px² assume 1080p.
  - **Environment coupling** (**F8/F9**): hardcoded paths; a config lookup failure silently degrades to a center crop.
- **Root cause:** identity is the load-bearing missing concept. Everything from active-speaker confidence to DualFrame eligibility to close-speaker isolation is inferring identity from position.
- **ML realistic?** Yes, and the cheapest high-value insertion point in the whole system is a **re-ID embedding** — OSNet-x0_5/x0_25 (0.2–0.6M params) — used to key tracks by appearance rather than position. Separately, the active-speaker heuristic should become a *fusion* (mouth motion × audiovisual ASD × audio-speaker consistency, with abstention), **not a swap**: the project's own measurement found Light-ASD (TalkSet weights) at **54% coverage @ 82.1% accuracy** vs. the mouth heuristic's **87% @ 83.5%** — off-the-shelf ASD *underperforms* the heuristic in-domain.
- **Classification: B for active speaker (fusion), B for track persistence (re-ID). The composition rules stay deterministic.**

### 2.7 Caption system

- **Does:** ASS generation, T7 phrase-paired rolling display from word timing + speech rhythm, 1–3 word chunks.
- **Type:** Deterministic from transcript timing.
- **Fails on:** phrase boundaries that prosody would place differently than word-timing gaps; Hindi/Hinglish tokenization; missing sentence-final punctuation from ASR.
- **ML realistic?** A prosodic phrase-boundary predictor is a real, small, well-studied task. But the prompt's constraint is explicit: retain T7's conceptual behavior, no karaoke, no per-word highlighting. Value is real but lower than 2.1–2.6, and the risk of aesthetic regression is real.
- **Classification: B, deferred. Small prosody model later, behind A/B.**

### 2.8 Audio Intelligence (render-time enhancement)

- **Does:** conditional enhancement/denoising gated on measured speech quality.
- **Type:** Heuristic gating + (claimed) processing.
- **Fails on / key finding:** learned enhancement **degrades clean studio audio**. Measured SIG-MOS on the project's own corpus: **NSNet2 4.144 → 3.866**, **RNNoise 4.144 → 3.884**. This is the clearest "do not add ML here" result in the study.
- **Classification: D (not worth ML) by default.** Enhancement should be **OFF on clean studio audio**. If needed for degraded sources: **DeepFilterNet2** (MIT/Apache, 2.3M params, RTF 0.04 on laptop CPU, ships as a pure Rust binary — unusually good fit for this stack). Loudness should stay pure BS.1770/LUFS, no ML.

### 2.9 Rendering

- **Does:** FFmpeg, 1080×1920, square-in-9:16 for adaptive.
- **Type:** Deterministic.
- **Fails on:** nothing that ML addresses. **Classification: A-no-op — leave alone.**

### 2.10 The Phase 2 A/B/C/D verdicts

**A. Should remain deterministic** — boundary snapping, boundary optimization, containment/centering rules, DualFrame eligibility, caption geometry, A/V sync, render, loudness (BS.1770), the payoff endpoint chain (`LLM payoff → transcript alignment → payoff_end → candidate_end`).

**B. Could benefit from ML signals** — candidate discovery (protocol change), hook scoring (after F1 fix), active speaker (fusion), track persistence (re-ID), Smart Pacing (small classifier), T7 prosody (deferred), scene detection.

**C. ML could replace current heuristics** — the fake acoustic analyzer (**F1**: replace with real features — strictly a bug fix), Smart Pacing breath thresholds (replace with learned classifier once data exists), positional speaker identity (replace with appearance embeddings).

**D. ML would add unnecessary complexity** — audio enhancement on clean sources (measured degradation), loudness, any end-to-end "one giant model → mp4" approach, learned caption geometry, render-time framing replacement.

---

## Phase 3 — Opportunity Areas (A–J)

Each area follows the prompt's required chain: **CURRENT SYSTEM → CURRENT LIMITATION → ML OPPORTUNITY → MODEL CLASS → SPECIFIC MODELS → INPUTS → OUTPUTS → INTEGRATION POINT → BENEFIT → COMPUTE COST → LATENCY → DATA → TRAINING → FAILURE MODES → SAFETY/FALLBACK → TESTING.**

### A. Multimodal Highlight / Candidate Discovery — **HIGHEST PRIORITY**

| Field | Value |
|---|---|
| **Current system** | LLM reads transcript in a sliding window, emits candidate spans with timestamps |
| **Current limitation** | Timestamp-*generation* is the empirically weakest way to extract highlights; no audio, no visual signal enters discovery |
| **ML opportunity** | Rephrase as **per-clip yes/no scoring + deterministic aggregation** (REZE). Add audio/embedding signals (Rhapsody). |
| **Model class** | Frozen VLM/LLM as a clip scorer + deterministic temporal aggregation; optionally a small audio-embedding model |
| **Specific models** | **REZE** protocol (arXiv:2608.04480) applied to the *existing* 8 LLM providers — **no new model required**. Local VLM option: **Qwen3-VL-4B INT4** (~2–3 GB), **InternVL3.5** (Apache-2.0), **VideoChat3-4B** (16 tok/frame, handles 3-hour video), **Gemma 3n E4B** (the only small model that natively consumes audio, 6.25 tok/s). Audio side: **HuBERT/DVA features** per Rhapsody. |
| **Inputs** | Clip window (transcript slice + optional sampled frames + optional audio embedding) |
| **Outputs** | Per-clip highlight score → deterministic mean-centering, Gaussian smoothing, Kadane/Otsu → candidate spans |
| **Integration point** | `llm.rs` candidate discovery — replace the timestamp-emitting prompt with a scoring prompt; aggregation in Rust before `CandidateDraft` construction |
| **Benefit** | REZE: **73.41 HIT@1** on QVHighlights (SOTA, beating fully supervised models); **3.5× mAP** vs. timestamp generation from the same model. Rhapsody: audio was the *only* modality that consistently added signal beyond text. |
| **Compute cost** | Zero incremental if run on existing cloud LLM providers (more calls, cheap tokens). Local VLM path: single-digit GB VRAM. |
| **Latency** | One scoring call per clip window (e.g., 30–60s stride); embarrassingly parallel; results cacheable per source video |
| **Data requirement** | **None for Phase 0** — REZE is zero-shot. Optional later: YouTube **"most replayed" graph as a free labeling oracle** (Rhapsody's key insight). |
| **Training requirement** | None (Phase 0). Optional small finetune later on most-replayed labels. |
| **Failure modes** | Provider outage; scoring drift between providers; clip windows straddling a real boundary |
| **Safety / fallback** | Existing boundary snapping still enforces; per-provider score normalization; deterministic aggregation is reproducible; failure → fall back to current timestamp prompt |
| **Testing strategy** | A/B on the existing regression corpus (Beat Emotional Fatigue, Messi/Ronaldo, clip-*.mp4); measure HIT@k against a human-selected-highlight set; assert boundary-snap tests unchanged |

**Why this fits long-form interview/podcast content specifically:** REZE and Rhapsody were both evaluated on exactly this content class. Rhapsody's finding that *audio* was the only consistently additive modality matches podcasts — laughter, emphasis, and reaction are carried in the audio, and AutoShorts is currently discarding all of that (F1 makes it worse: it pretends to use audio and does not).

### B. Active Speaker Detection

| Field | Value |
|---|---|
| **Current system** | Mouth-motion heuristic: 32×20 mouth-ROI mean absdiff, threshold ≥4.0, dominance ×1.20 (**F5**) |
| **Current limitation** | No audio reference; fires on laughing listeners; illumination-sensitive; fails on side profiles and low mouth visibility |
| **ML opportunity** | **Fusion**, not replacement. In-domain measurement: Light-ASD 54% coverage @ 82.1% acc vs. heuristic 87% @ 83.5% — a bare swap is a regression. |
| **Model class** | Audiovisual ASD scoring + fusion with abstention |
| **Specific models** | **Nemotron-3-Diarization** (Sept 2026, 100M params, OpenMDW v1.1, DIHARD-3 **DER 12.73**, 10-ms frames, up to 8 speakers arrival-ordered; 🤗 Transformers + NeMo-Speech.cpp, has a Windows installer) as the audio-side speaker label source; Light-ASD/TalkNet-style visual ASD as a *second* opinion |
| **Inputs** | Face track (visual), audio window, candidate speaker embedding |
| **Outputs** | P(speaking) per track per time; fused confidence |
| **Integration point** | `speaker_tracker.py` active-speaker stage; feeds framing, DualFrame eligibility, close-speaker isolation |
| **Benefit** | Correct speaker choice in laughing-listener and overlap cases; principled abstention instead of a hard threshold |
| **Compute cost** | Audio diarization is one pass per source video (not per frame); visual ASD is lightweight |
| **Latency** | Offline batch — no per-frame requirement |
| **Data requirement** | In-domain validation set (already partially built — the 61/61 active_speaker suite) |
| **Training requirement** | Small domain finetune of the fusion layer only; do **not** train the ASD backbone |
| **Failure modes** | Overlap, camera cuts during speech, off-screen speech |
| **Safety / fallback** | Confidence below threshold → the existing mouth-motion heuristic decides. Never let ASD failure break framing. |
| **Testing strategy** | Extend `test_active_speaker_suite.py` (61/61 today) with laughing-listener and overlap cases; report accuracy/F1/confusion and speaker-switch latency |

### C. Speaker Diarization / Speaker Identity

| Field | Value |
|---|---|
| **Current system** | None. All words `S1` (**F4**); visual identity is positional (**F3**) |
| **Current limitation** | `hookSpeaker: "Host"\|"Guest"` and `S1/S2` are never reconciled; two-speaker scenes have no identity concept |
| **ML opportunity** | One local diarization pass per source video unlocks: Host/Guest↔speaker mapping, speaker-aware caption attribution, speaker-continuity features for Smart Pacing, better candidate segmentation |
| **Model class** | End-to-end / pipeline diarizer + speaker embeddings |
| **Specific models** | **Nemotron-3-Diarization** (above) as local SOTA; **WhisperX** (faster-whisper + pyannote) as the drop-in local replacement for Deepgram that *also* fixes the `base`-model quality problem; **pyannote community-1** (CC-BY-4.0, gated) and **Sortformer v2.1** as alternatives |
| **Inputs** | Full audio track |
| **Outputs** | Speaker labels with 10-ms resolution; per-speaker embeddings |
| **Integration point** | New stage after `transcription.rs`; consumed by `llm.rs` (Host/Guest mapping), Smart Pacing (speaker continuity), captions |
| **Benefit** | Resolves F4; makes `hookSpeaker` meaningful rather than a guess; enables speaker-aware everything downstream |
| **Compute cost** | One pass per source video, ~100M params — minutes on CPU for a 2-hour file |
| **Latency** | Batch; no per-frame cost |
| **Data requirement** | None (pretrained) |
| **Training requirement** | None |
| **Failure modes** | Speaker count errors on heavy overlap; arrival-order labels (needs mapping to Host/Guest via speaking-time heuristics) |
| **Safety / fallback** | Diarization failure → fall back to current `S1` behavior, zero behavior change |
| **Testing strategy** | DER on a labeled subset; speaker-confusion matrix; assert all existing suites unchanged when diarization is disabled |

**Licensing note:** pyannote's community models are CC-BY-4.0 with gated access; Nemotron-3 is OpenMDW v1.1. Prefer Nemotron-3/WhisperX for a redistributable binary.

### D. Smart Pacing 2.0 — **HIGH IMPORTANCE (per prompt)**

| Field | Value |
|---|---|
| **Current system** | v1/v2 with acoustic heuristics; breath thresholds calibrated on one speaker (**F11**) |
| **Current limitation** | Cannot distinguish a removable pause from an intentional one; per-speaker calibration does not generalize |
| **ML opportunity** | A **small specialized classifier** is explicitly preferable to a large network here. There is **no pretrained model for "removable vs. intentional pause"** — this must be AutoShorts-specific. |
| **Model class** | Gradient boosting (LightGBM) or small 1D-CNN on handcrafted acoustic features |
| **Specific models** | **Silero VAD v6.2.3** (MIT, ~2 MB, <1 ms/chunk, ROC-AUC 0.97) as the front-end; **librosa** for features (**not** openSMILE — commercial audEERING license); LightGBM or a 1D-CNN head |
| **Inputs** | RMS energy, spectral features, HF ratio, contrastDb, F0/pitch, pre-pause speech window, post-pause speech window, pause duration, transcript, speaker continuity (from diarization), room tone estimate, breath characteristics |
| **Outputs** | P(removable pause), P(breath), P(hesitation), P(dead-air), P(intentional pause), P(speaker-turn) — exactly the prompt's requested output set |
| **Integration point** | `smart_pacing.py` — ML probability → **existing Smart Pacing safety rules** → safe edit decision. The deterministic safety layer is not removed. |
| **Benefit** | Speaker-independent pause decisions; fewer bad cuts; less manual review |
| **Compute cost** | Negligible — Silero VAD + feature extraction + a tiny model. Runs on CPU. |
| **Latency** | Per audio window, milliseconds. Fully streaming-capable. |
| **Data requirement** | **Free and domain-matched:** "gap removed in final edit" vs. "gap survived" from AutoShorts' own edit history. Currently discarded — **fix F6 to unblock this.** |
| **Training requirement** | Weak supervision; no annotation needed. LightGBM trains in seconds. |
| **Failure modes**** | Over-aggressive removal on unfamiliar room tone; speaker-domain shift |
| **Safety / fallback** | Low confidence → current thresholds decide. All existing safety rules (min/max gap, protection windows) untouched. |
| **Testing strategy** | Precision of removable-pause detection, false-removal rate, audio artifact rate, human preference on the 55/55 + 108/108 pacing suites' corpora. Zero regression target on existing suites. |

### E. Adaptive Framing / Learned Crop Composition

| Field | Value |
|---|---|
| **Current system** | Rule-based: ONE PERSON → track single; TWO PEOPLE → shared adaptive; ADAPTIVE → DualFrame OFF; ORIGINAL 9:16 → DualFrame eligible |
| **Current limitation** | Rules are right but *scores* are absent — crop selection has no quality signal; headroom/body/composition balance are encoded as margins the authors call "not re-derived from first principles" (**F11**) |
| **ML opportunity** | A **crop scorer**, not a crop generator. Academic 2025–26 learned-crop work (ProCrop, Venus, CROP, ShotCrop³) is all MLLM-based scorers — wrong for a local tool. The practical answer is a small offline scoring stack. |
| **Model class** | Aesthetic/composition scorer + saliency prior + per-scene camera-mode selection |
| **Specific models** | **CLIP ViT-B/32** (MIT) as the visual backbone + a **small trained aesthetic head**; **u2netp** (4.7 MB) for saliency; **AutoFlip-style per-scene camera-mode selection with polynomial path smoothing and letterbox fallback** — the single biggest visible quality lever; **PySceneDetect AdaptiveDetector** (BSD-3, F1 91.6 on broadcast) for shot boundaries |
| **Inputs** | Face/person tracks, speaker probabilities, shot geometry, saliency map, optional CLIP embedding |
| **Outputs** | Crop quality score per candidate crop |
| **Integration point** | `speaker_tracker.py` crop selection — scorer proposes, **existing hard safety constraints** dispose. "TWO PERSISTENT PEOPLE → BOTH MUST REMAIN VISIBLE IN ADAPTIVE" stays absolute. |
| **Benefit** | Better crops with no change to safety guarantees; smoother trajectories (no jitter, no letterbox popping) |
| **Compute cost** | CLIP-batched ~60–100 ms/frame; u2netp ~10 ms; all cacheable per shot |
| **Latency** | Offline; per-shot, not per-frame, once scene detection is in place |
| **Data requirement** | Frame/crop → quality score pairs; seed from existing render outputs + manual rating of ~2–5k crops |
| **Training requirement** | Small head on top of frozen CLIP — hours, not days |
| **Failure modes** | Aesthetic head drift; score disagreement across shots |
| **Safety / fallback** | Score is advisory; containment remains hard; invalid scores → current rule-based crop |
| **Testing strategy** | Face containment rate, crop stability, unnecessary motion, pair retention, subject clipping rate vs. current baseline on dualframe corpus (55/55) |

### F. Speaker Tracking / Identity (re-ID)

| Field | Value |
|---|---|
| **Current system** | Position-based identity — "nearest genuine face to crop center" (**F3**) |
| **Current limitation** | Reappearance after occlusion, camera cuts, multi-person stability all rest on proximity |
| **ML opportunity** | **Deep appearance embeddings** to key tracks by identity |
| **Model class** | Person re-identification |
| **Specific models** | **OSNet-x0_5 / OSNet-x0_25** (0.2–0.6M params — tiny, fast, CPU-friendly); detector upgrade **YOLO26n ONNX with `nms=False`** (39 ms CPU) |
| **Inputs** | Person crop |
| **Outputs** | 256/512-d appearance embedding |
| **Integration point** | `speaker_tracker.py` track association — embedding distance joins the existing positional cost; does not replace it |
| **Benefit** | Stable identity across cuts and occlusions — the root cause of most framing instability |
| **Compute cost** | ~5 ms/track/frame on CPU |
| **Latency** | Per-frame but cheap |
| **Data requirement** | None (pretrained re-ID weights) |
| **Training requirement** | None for Phase 1; optional in-domain finetune later |
| **Failure modes** | Similar appearance (same outfit, low light); embedding collapse |
| **Safety / fallback** | Embedding unavailable/unreliable → positional association only (current behavior) |
| **Testing strategy** | Track-purity metric across labeled cuts; reappearance-after-occlusion cases; identity-switch latency |

### G. T7 Speech-Chunk Segmentation

| Field | Value |
|---|---|
| **Current system** | Word-level timing → speech rhythm → phrase chunking → 1–3 word chunks → phrase-paired rolling display |
| **Current limitation** | Boundaries follow timing gaps, not prosodic phrasing; "how to become" / [pause] / "extraordinary" is judged by gap length alone |
| **ML opportunity** | A learned prosodic phrase-boundary predictor |
| **Model class** | Small sequence/tagging model on acoustic + lexical features |
| **Specific models** | No dominant pretrained model for conversational phrase boundary; build a small tagger on Silero VAD + librosa prosody + transcript features. Existing work on prosodic phrasing is academic-scale. |
| **Inputs** | F0 contour, energy, pause structure, word timings, POS-lite cues from transcript |
| **Outputs** | P(phrase boundary) at each word gap |
| **Integration point** | Caption chunking stage, feeding T7 display — **conceptual behavior unchanged: no karaoke, no per-word highlighting** |
| **Benefit** | More natural chunk boundaries, especially in Hindi/Hinglish where gap statistics differ |
| **Compute cost** | Negligible |
| **Latency** | Offline, per candidate |
| **Data requirement** | Manual annotation of a few hundred clips' ideal chunkings; weak supervision from caption-edit history |
| **Training requirement** | Small model, hand-labeled seed |
| **Failure modes** | Over-segmentation into single words (the karaoke failure mode) |
| **Safety / fallback** | Max/min chunk-size guards stay deterministic; model only *moves* boundaries within guardrails |
| **Testing strategy** | Phrase-boundary accuracy vs. human judgment, timing error, naturalness ratings; caption_qa suite (100 checks) must stay green |

**Verdict: real but deferred.** Lower expected value than A–F, and the aesthetic risk is highest here. Do it third, behind A/B.

### H. Hook Intelligence 2.0

| Field | Value |
|---|---|
| **Current system** | LLM hook scoring + (fake) acoustic signals feeding a composite with a 0.15 audio weight |
| **Current limitation** | Half the composite input is fabricated (**F1**); no genuine engagement signal |
| **ML opportunity** | (1) **Fix F1 with real acoustic features — no ML needed.** (2) Then optionally a learned hook/engagement score |
| **Model class** | Small regressor on multimodal features |
| **Specific models** | Features: librosa (energy envelope, F0, spectral flux) + PANNs **Cnn14-16k** as an offline reaction extractor (Laughter/Applause/Cheering are AudioSet classes — directly relevant to "is this a reactive moment") |
| **Inputs** | Acoustic features, transcript embedding, visual change, sentence structure |
| **Outputs** | P(hook strength), topic-clarity/curiosity/relevance sub-scores |
| **Integration point** | Hook composite in `llm.rs` / multimodal analyzer — advisory only |
| **Benefit** | Honest signals replace fabricated ones; reaction audio ranks hooks better |
| **Compute cost** | PANNs is one offline pass per source video |
| **Latency** | Batch |
| **Data requirement** | Edit history + most-replayed oracle (shared with A) |
| **Training requirement** | Optional small head |
| **Failure modes** | Score noise on unfamiliar content types |
| **Safety / fallback** | **Must NEVER override the authoritative payoff endpoint architecture** (`LLM payoff → transcript alignment → payoff_end → candidate_end`). Hook score changes ranking only. |
| **Testing strategy** | Hook-suite regression; agreement with human hook ratings |

### I. Audio Intelligence

| Field | Value |
|---|---|
| **Current system** | Conditional enhancement gated on measured speech quality |
| **Current limitation** | Enhancement **hurts** clean audio. Measured SIG-MOS: NSNet2 4.144→3.866, RNNoise 4.144→3.884 |
| **ML opportunity** | **OFF by default on clean studio audio.** Only for degraded sources. |
| **Model class** | Speech enhancement |
| **Specific models** | **DeepFilterNet2** (MIT/Apache, 2.3M params, **RTF 0.04 on laptop CPU**, pure Rust binary — the best fit in the entire study for this stack) |
| **Inputs** | Audio track |
| **Outputs** | Enhanced audio |
| **Integration point** | Existing audio-intelligence gating — only when quality gate says degraded |
| **Benefit** | Recovery of noisy sources without touching clean ones |
| **Compute cost** | RTF 0.04 — 2-hour file in ~5 minutes CPU |
| **Latency** | Batch |
| **Data requirement** | None |
| **Training requirement** | None |
| **Failure modes** | Artifacts on music, over-damping |
| **Safety / fallback** | Default OFF; quality-gated; kill-switch already exists |
| **Testing strategy** | SIG-MOS / PESQ / STOI comparisons vs. baseline on the project corpus; audio_intelligence suite (48/48) must stay green |

**Loudness: pure BS.1770/LUFS. No ML.** This is the cleanest "D — not worth ML" call in the report.

### J. Additional ML Opportunities (screened)

Only those with genuine evidence:

| Opportunity | Verdict | Notes |
|---|---|---|
| **Scene-change detection** | **DO IT** | PySceneDetect AdaptiveDetector (BSD-3, F1 91.6) — feeds E (per-scene camera modes) and F. Cheap, high structural value. |
| **Shot classification** | **DO IT (with E)** | Shot-type recognition is an input to camera-mode selection; small classifier on CLIP features |
| **Visual quality / blur / bad-frame detection** | **DO IT** | Laplacian-variance blur detection is non-ML and free; guards against tracking on a blurry frame |
| **Duplicate-frame detection** | **Skip** | p-hash is trivially available if needed; not a current failure mode |
| **Content safety filtering** | **Skip for now** | Only relevant if publishing autonomously; not a current requirement |
| **Engagement prediction** | **Defer** | Needs published-performance data AutoShorts does not have; the most-replayed oracle (A) is a better proxy for now |
| **Title/description quality** | **Skip** | Not in the pipeline's scope |
| **Transcript correction** | **DO IT (indirect)** | WhisperX upgrade (C) fixes ASR quality; LLM-assisted correction of punctuation is already partially in play |
| **Accent robustness / multilingual** | **DO IT (via C)** | WhisperX + larger model sizes directly address the Hindi/Hinglish requirement (R7) |
| **Segment quality scoring** | **Covered by A** | The highlight score *is* the segment quality score |
| **Render quality verification** | **DO IT (cheap)** | Automated render QA: A/V sync check, loudness compliance, containment re-check on output frames — deterministic checks, not ML |
| **Caption quality prediction** | **Defer with G** | Rides along with T7 prosody work |
| **Candidate redundancy detection** | **DO IT** | Embedding similarity between final candidates prevents two near-identical shorts; trivial once transcript embeddings exist |

---

## Phase 4 — Model Research (current, source-verified, Sept 2026)

All facts below were checked against primary sources (papers, official repos, model cards). Each row distinguishes **SOURCE FACT** from engineering inference.

### 4.1 Candidate discovery / highlight detection

| Model / System | Paper / Source | Year | Task | Params | Context | GPU req. | CPU? | Speed | Local? | License | Training data | Finetune? | Key limitation | AutoShorts integration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **REZE** | arXiv:2608.04480 | 2026 | Video highlight via frozen-VLM clip scoring + deterministic aggregation | Uses *any* frozen VLM | model-dependent | model-dependent | if VLM is | one yes/no call per clip | yes (with a local VLM) | protocol, not weights | n/a | n/a | needs many cheap calls; window choice matters | **`llm.rs` discovery — Phase 0, no new model** |
| **Rhapsody** | COLM 2025, arXiv:2505.19429 | 2025 | Podcast highlight extraction on real data | Llama-3.2-1B + QLoRA (~6M trainable) | long-form audio | low | yes | fast | yes | Llama-3.2 | real podcasts | QLoRA, small | **best system only 49% hit rate** — task is hard | validates audio-as-signal + most-replayed oracle |
| **Qwen3-VL-4B (INT4)** | official model card | 2026 | VLM | ~4B (INT4 ~2–3 GB) | long video | 6–8 GB | slow | moderate | yes | permissive | web-scale | LoRA possible | latency on long video | local VLM for REZE protocol |
| **InternVL3.5** | official repo | 2026 | VLM | family, 1B–108B | long | varies | small ones yes | varies | yes | **Apache-2.0** | web-scale | yes | size/perf tradeoff | license-clean local VLM |
| **VideoChat3-4B** | official repo | 2026 | Video understanding | 4B | **16 tok/frame, handles 3-hour video** | ~8 GB | yes w/ quant | good for length | yes | permissive | video-scale | yes | newer, less ecosystem | long-podcast VLM path |
| **Gemma 3n E4B** | official model card | 2026 | Multimodal incl. **native audio** | ~4B effective | long | ~6 GB | yes | 6.25 tok/s audio | yes | Gemma terms | Google-scale | limited | audio token budget | only small model that eats audio natively |

**Key SOURCE FACTS:** REZE 73.41 HIT@1 on QVHighlights (beats all fully supervised); 43.28 vs 12.32 mAP for scoring-vs-timestamps. Rhapsody: GPT-4o zero-shot ≈ frequency baseline; audio was the only consistently additive modality; YouTube most-replayed graph used as labeling oracle.

### 4.2 Diarization / active speaker

| Model | Source | Year | Task | Params | GPU | CPU | Speed | Local | License | Finetune? | Limitation | AutoShorts integration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Nemotron-3-Diarization** | NVIDIA, Sept 2026 | 2026 | E2E diarization | 100M | optional | **yes** (NeMo-Speech.cpp, **Windows installer**) | 10-ms frames, 8 speakers arrival-ordered | yes | **OpenMDW v1.1** | possible | overlap handling bounded | **Primary recommendation (C, B)** |
| **WhisperX** | official repo | 2024+ | ASR + diarization | faster-whisper + pyannote | optional | yes | fast batch | yes | components vary | no | pyannote gating | Drop-in Deepgram replacement (C) |
| **pyannote community-1** | HF Hub | 2025 | Diarization pipeline | ~10M | optional | yes | good | yes | **CC-BY-4.0, gated** | yes | gate friction | Alternative |
| **Sortformer v2.1** | official repo | 2025 | Diarization | encoder-decoder | optional | yes | good | yes | permissive | yes | newer | Alternative |
| **Light-ASD (TalkSet weights)** | official repo | 2024 | Audiovisual ASD | small | optional | yes | fast | yes | research | possible | **54% coverage @ 82.1% acc in-domain — below the current heuristic (87% @ 83.5%)** | Fusion ingredient, **not** a replacement (B) |

**SOURCE FACT:** DIHARD-3 DER 12.73 for Nemotron-3 — current local SOTA-class.

### 4.3 Framing / tracking / composition

| Model | Source | Year | Task | Params | CPU speed | Local | License | Limitation | AutoShorts integration |
|---|---|---|---|---|---|---|---|---|---|
| **OSNet-x0_5 / x0_25** | official repo | 2021+ (still SOTA-size class) | Person re-ID | **0.2–0.6M** | **~5 ms/track** | yes | permissive (research-friendly) | appearance twins | **Track identity (F)** — primary |
| **YuNet (int8)** | OpenCV zoo | 2024 | Face detect | tiny | **~5 ms** | yes | Apache-2.0 | face-only | face-track front-end |
| **YOLO26n ONNX (`nms=False`)** | Ultralytics | 2026 | Person detect | nano | **39 ms CPU** | yes | **AGPL-3.0 — LICENSE ISSUE** | copyleft obligation | Detector upgrade — **after license decision** |
| **u2netp** | official repo | 2020 | Saliency | 4.7 MB | ~10 ms | yes | Apache-2.0 | coarse | Saliency prior for crop scoring (E) |
| **CLIP ViT-B/32** | OpenAI | 2021 | Image-text embedding | 151M | batched 60–100 ms | yes | **MIT** | not aesthetic per se | Composition backbone (E) |
| **PySceneDetect AdaptiveDetector** | official repo | 2024 | Scene change | n/a | fast | yes | **BSD-3** | parameter tuning | Scene boundary feed (E, F, J) |
| **AutoFlip (method)** | Google AI | 2019 | Per-scene camera-mode crop | n/a | n/a | method | Apache-2.0 (original) | older, but the *method* is the value | **Per-scene camera-mode selection + polynomial smoothing + letterbox fallback (E)** |

**SOURCE FACT:** 2025–26 learned-crop literature (ProCrop, Venus, CROP, ShotCrop³) is dominated by MLLM-based scorers. **Engineering inference:** these are the wrong tool for a local, offline, consumer-hardware product; a small offline scorer + AutoFlip-style deterministic smoothing wins on integration fit even if it loses on a benchmark.

### 4.4 Pacing / VAD / prosody

| Model | Source | Year | Task | Params | CPU speed | Local | License | Limitation | AutoShorts integration |
|---|---|---|---|---|---|---|---|---|---|
| **Silero VAD v6.2.3** | official repo / HF | 2026 | VAD | **~2 MB** | **<1 ms/chunk**, ROC-AUC 0.97 | yes | **MIT** | VAD only | Front-end for Smart Pacing classifier (D) |
| **LightGBM (custom)** | — | — | Pause classification | ~k features | microseconds | yes | Apache-2.0 | needs labeled data | The actual Smart Pacing model (D) |
| **1D-CNN (custom, small)** | — | — | Pause classification | <100K | fast | yes | yours | more data-hungry | Alternative to LightGBM |
| **openSMILE** | audEERING | — | Acoustic feature set | n/a | fast | yes | **needs commercial license for commercial use** | licensing | **AVOID — use librosa** |

**SOURCE FACT:** no pretrained model classifies "removable vs. intentional pause." **Engineering inference:** this is necessarily an AutoShorts-specific model; therefore the free in-domain dataset (edit history) is the decisive asset.

### 4.5 Audio intelligence

| Model | Source | Task | Params | Speed | Local | License | Measured result | AutoShorts integration |
|---|---|---|---|---|---|---|---|---|
| **DeepFilterNet2** | official repo + Rust crate | Speech enhancement | 2.3M | **RTF 0.04 laptop CPU**, pure Rust binary | yes | MIT/Apache | strong on noisy sources | **Only recommended enhancement model** |
| NSNet2 | Microsoft | Enhancement | small | fast | yes | research | **SIG-MOS 4.144 → 3.866 (degrades clean audio)** | **Do not use on clean audio** |
| RNNoise | Mozilla | Enhancement | small | fast | yes | BSD | **SIG-MOS 4.144 → 3.884 (degrades clean audio)** | **Do not use on clean audio** |
| **PANNs Cnn14-16k** | official repo | Audio event tagging | ~80M | batch | yes | research-friendly | AudioSet classes incl. **Laughter / Applause / Cheering** | Offline reaction metadata (H), one pass per source |
| **BS.1770 / LUFS** | standard | Loudness | n/a | n/a | yes | standard | — | **Keep, no ML** |

---

## Phase 5 — Pretrained vs Fine-Tuned vs Train-From-Scratch

The prompt's default preference order is **PRETRAINED → SMALL FINE-TUNE → SPECIALIZED MODEL → FULL FINE-TUNE → TRAIN FROM SCRATCH**. Assigning each proposed subsystem:

| Subsystem | Strategy | Rationale |
|---|---|---|
| Candidate discovery (Phase 0) | **A — pretrained (existing LLM providers), protocol change** | REZE is zero-shot; zero new weights, zero GPU |
| Candidate discovery (Phase 2) | **B — small fine-tune** (local VLM LoRA on most-replayed labels) | Only after the oracle dataset exists |
| Diarization | **A — pretrained (Nemotron-3 or WhisperX)** | SOTA-class pretrained exists; no reason to train |
| Active speaker fusion | **C — small AutoShorts-specific classifier (fusion + abstention)** | The fusion layer is the novel part; backbones stay pretrained |
| Track identity (re-ID) | **A — pretrained (OSNet)** | Standard pretrained re-ID weights exist and are small |
| Smart Pacing pause classifier | **C — small specialized model (LightGBM/1D-CNN)** | No pretrained model exists for this task; data is free |
| Framing composition scorer | **B — small fine-tune** (aesthetic head on frozen CLIP) | CLIP is the representation; only the head trains |
| T7 prosodic boundaries | **C — small specialized tagger** | No suitable pretrained model; deferred anyway |
| Hook engagement | **B — small fine-tune** | Late, low priority |
| Audio enhancement | **A — pretrained (DeepFilterNet2), OFF by default** | Measured: pretrained models degrade clean audio |
| Scene detection | **A — pretrained (PySceneDetect)** | Deterministic algorithm, BSD-3, F1 91.6 |
| Render QA | **Deterministic checks** | Not ML at all — the right call |

**Train-from-scratch: recommended nowhere.** Nothing in this study justifies the cost. The two "C" entries are small specialized *models*, not from-scratch systems — they train on free weak labels in minutes.

---

## Phase 6 — Data Requirements

### 6.1 The general principle

AutoShorts has an unusual advantage: **it generates labeled training data as a byproduct of being used.** Every run decides which gaps to remove and which candidates to ship. The main blocker is that it currently **throws that data away** (**F6**: only 18 of ~30+ `CandidateDraft` columns persist). The first data action is a schema-widening persistence fix — no model needed — and it unlocks Smart Pacing training.

### 6.2 Per-subsystem data plans

**Smart Pacing — pause → keep/remove**

- **Labels:** `gap_removed_in_final_edit` (positive for "removable") vs `gap_survived` (negative). Both are already computed; they are just not stored.
- **Label generation:** automatic from the edit map. **Weak supervision, zero annotation.**
- **Self-supervision possible:** partially — room-tone profiling can be self-supervised (cluster noise floors).
- **Volume:** ~10–50K gap instances across a few hundred videos is ample for LightGBM.
- **Leakage avoidance:** split by **video** and by **speaker**, never by gap instance. A model that has seen a speaker's other gaps has learned that speaker's breath pattern, not removability.
- **Speaker/video independence:** mandatory — the current thresholds' failure is precisely overfitting to one speaker (**F11**).

**Framing — frame/crop → quality score**

- **Labels:** human rating of candidate crops (1–5) on a few thousand frames, sampled from *diverse* videos (different shot scales, lighting, speaker counts).
- **Self-supervision possible:** yes, partially — "crop chosen by current rules on a video the user kept" is a weak positive.
- **Volume:** 2–5K rated crops seeds the aesthetic head.
- **Leakage avoidance:** split by video and by scene, not by frame (adjacent frames are near-duplicates).

**Candidate discovery — segment → highlight quality**

- **Labels:** **YouTube's "most replayed" graph is a free labeling oracle** (Rhapsody's key insight). It is crowd-sourced engagement ground truth on exactly the content AutoShorts processes.
- **Self-supervision:** the REZE protocol itself is zero-shot, so Phase 0 needs none.
- **Volume:** hundreds of videos with most-replayed curves is enough for a small LoRA.
- **Leakage avoidance:** never train and evaluate on the same channel; creator identity leaks heavily.

**Active speaker / re-ID**

- **Labels:** extend the existing 61/61 active-speaker suite with frame-level speaking/not-speaking labels on real footage.
- **Volume:** a few hundred labeled minutes suffices for fusion calibration.
- **Independence:** split by video; include laughing-listener and overlap cases explicitly.

### 6.3 Splits and generalization

All datasets: **60/20/20 by video**, with an additional held-out **speaker-disjoint** set where possible. Generalization requirement: a model trained on English studio podcasts must not degrade on Hindi/Hinglish or on a noisy field recording. Multilingual coverage is a stated product requirement (R7), so the eval set must include it.

---

## Phase 7 — Hybrid AI Architecture

The prompt's required shape — models produce signals, the deterministic engine retains control — is exactly right for this codebase, because the deterministic layer is what the 281 Rust tests and 16 Python suites actually pin.

### 7.1 The decision-control split

```
DEEP MODELS  (probabilities, embeddings, scores)
        ↓
EXISTING DETERMINISTIC ENGINE  (decisions)
        ↓
SAFETY CONSTRAINTS  (hard limits)
        ↓
FFmpeg → FINAL SHORT
```

**Decisions that must remain deterministic (from the prompt, verified against code):**

| Decision | Why deterministic | Where |
|---|---|---|
| Adaptive two-person containment | "BOTH MUST REMAIN VISIBLE" is a hard constraint, not a preference | `speaker_tracker.py` |
| DualFrame eligibility | Mode rule (ADAPTIVE → DualFrame OFF) | `speaker_tracker.py` |
| **Payoff endpoint** (LLM payoff → transcript alignment → `payoff_end` → `candidate_end`) | LLM recommends; Rust enforces — the core architecture contract | `lib.rs`, `boundary.rs` |
| Boundary snapping | Fixed thresholds on LLM scores; the safety layer proper | `lib.rs:1135-1465` |
| Caption rendering geometry | T7 must not become karaoke | caption system |
| Audio/video synchronization | Never a learned quantity | render |
| Loudness (BS.1770/LUFS) | Regulatory/standard | render |
| Smart Pacing safety rules (min/max gap, protection windows) | Prevents audio artifacts | `smart_pacing.py` |
| T7 chunk-size guards | Prevents the karaoke failure mode | caption system |

### 7.2 How each ML signal feeds in

- **Highlight score** → re-ranks candidates *before* `snap_to_semantic_boundaries`. Never moves a boundary itself.
- **P(removable pause)** → replaces the *threshold comparison* in Smart Pacing; the surrounding safety rules are untouched. `ML probability → existing safety rules → safe edit decision`.
- **P(speaking) fusion** → replaces the ≥4.0 absdiff comparison; DualFrame eligibility logic unchanged.
- **Appearance embedding** → becomes a term in the track-association cost; positional term retained.
- **Crop quality score** → advisory input to crop selection; containment constraint always vetoes.
- **Diarization labels** → context (speaker continuity) for pacing and captions; never an edit decision.

### 7.3 Anti-patterns explicitly rejected

- ❌ VIDEO → one giant model → mp4 (the prompt forbids it; the test suite forbids it)
- ❌ Letting a learned model move the payoff endpoint
- ❌ Learned loudness, learned A/V sync, learned caption geometry
- ❌ Replacing boundary snapping with a sequence model
- ❌ Any model whose failure can take down the render path

---

## Phase 8 — Hardware / Compute Feasibility

Target: a **consumer/developer Windows machine**, no cloud assumption. All estimates assume the existing 8-fps analysis cadence (~125 ms/frame budget).

### 8.1 Per-model compute budget

| Model | VRAM | RAM | CPU fallback | Per-frame cost | Batch? | ONNX | Quantization | PyTorch required? | Rust boundary practical? |
|---|---|---|---|---|---|---|---|---|---|
| Silero VAD | <100 MB | tiny | **yes** | <1 ms | yes | **yes** | int8 fine | no | **yes** (Rust ONNX) |
| LightGBM pause model | n/a | <50 MB | **yes** | µs | yes | n/a | n/a | no | **trivially** |
| OSNet re-ID | <200 MB | <300 MB | **yes (~5 ms)** | yes | yes | **yes** | int8 fine | no | **yes** |
| YuNet int8 | <50 MB | tiny | **yes (~5 ms)** | n/a | yes | **yes** | int8 | no (OpenCV) | **yes** |
| u2netp saliency | <10 MB | <50 MB | **yes (~10 ms)** | n/a | yes | **yes** | yes | no | **yes** |
| CLIP ViT-B/32 | ~1 GB fp32 / ~300 MB int8 | ~1 GB | yes (60–100 ms batched) | yes | **yes** | **int8 OK** | optional | **yes** |
| DeepFilterNet2 | <500 MB | <1 GB | **yes, RTF 0.04** | n/a (audio) | yes | alt | — | no | **native Rust — best fit** |
| Nemotron-3 diarization | ~1–2 GB | ~2 GB | **yes** (NeMo-Speech.cpp, Windows installer) | n/a (one pass) | yes | via NeMo | possible | no | **yes** |
| YOLO26n | ~300 MB | <1 GB | **yes (39 ms, nms=False)** | yes | yes | **yes** | int8 | no | **yes** — **license first** |
| Qwen3-VL-4B INT4 | **6–8 GB** | ~6 GB | slow | n/a (clip scoring) | calls | yes | INT4 already | maybe | sidecar |
| PANNs Cnn14-16k | ~1 GB | ~2 GB | yes | n/a (one pass) | yes | yes | possible | no | yes |

### 8.2 Throughput reality check at 8 fps

Single-threaded per-frame sum for the full visual stack:

```
YuNet-int8      ~5 ms
YOLO26n        ~39 ms
OSNet           ~5 ms
u2netp         ~10 ms
CLIP batched  ~60–100 ms
            ----------
             ~120–160 ms/frame   (budget: 125 ms)
```

**Engineering inference:** this is at or slightly over budget single-threaded, but this is an **offline batch** workload, not real-time. CLIP and u2netp do not need to run every frame — they run **per shot** after scene detection. With per-shot caching the steady-state cost collapses. For a 2-hour podcast: **single-digit minutes on a consumer GPU, 10–30 minutes on CPU.** That is an acceptable addition to a pipeline that already renders with FFmpeg.

### 8.3 Deployment options compared

| Option | Verdict |
|---|---|
| **CPU** | Universal fallback; sufficient for everything except the local VLM path |
| **GPU (CUDA)** | First choice where present — all models above fit in 6–8 GB |
| **DirectML** | Works but **opset-20 cap and sustained-engineering status** — do not make it the primary path; Windows-GPU fallback only |
| **TensorRT** | Not worth the build complexity at these model sizes |
| **ONNX Runtime (`ort` crate)** | **The strategic choice** — one runtime, CPU + CUDA + DirectML execution providers, Rust-native, no Python required for the core models |
| **PyTorch** | Only for the optional local VLM path; keep it out of the hot loop |

### 8.4 Recommended inference stack

```
Rust core
  └─ ort (ONNX Runtime) ──── CPU / CUDA / DirectML execution providers
       └─ hf-hub: on-demand model download into %LOCALAPPDATA%/AutoShorts/models
  └─ DeepFilterNet2 native Rust binary (no runtime needed at all)
  └─ Python sidecar (optional, lazy): PyInstaller + Tauri externalBin
       └─ only used for the local VLM path if enabled
```

- **`ort` crate** (Rust ONNX Runtime bindings) is the single integration point for all core models — VAD, re-ID, saliency, CLIP, detectors.
- **`hf-hub`** for on-demand download gives rembg-style distribution: no giant installer, models fetched on first use, cacheable.
- **int8 QDQ quantization (S8S8)** for the big-ish models (CLIP, detectors) — meaningful speedup, negligible quality loss at these tasks.
- **The Python sidecar must become optional.** Today (**F10**) there is no sidecar bundling at all; distribution assumes a dev machine. Bundling DeepFilterNet2 as a pure Rust binary and moving the rest to `ort` removes the Python dependency from the default path.

---

## Phase 9 — Latency / Pipeline Impact

### 9.1 Cadence classification (the prompt's key question)

| Model | Correct cadence | Why |
|---|---|---|
| Silero VAD + pause features | **per audio window** (streaming) | Cheap enough to run inline; latency-critical only if pacing becomes interactive |
| LightGBM pause classifier | **per gap** | One inference per candidate gap — negligible |
| Nemotron-3 diarization | **once per source video** | Full pass, minutes; cache results |
| PANNs reaction metadata | **once per source video** | Full pass, batch |
| Scene detection (PySceneDetect) | **once per source video** | Structural; cache |
| CLIP + aesthetic scorer | **once per shot** | Shot-level quality, not frame-level |
| u2netp saliency | **once per shot** | Same |
| OSNet re-ID | **per frame, per track** | But ~5 ms — affordable |
| YuNet / YOLO26n | **per frame (sampled)** | Already the current cadence |
| Local VLM clip scorer (REZE) | **once per clip window** | Parallelizable across windows; cacheable |
| DeepFilterNet2 | **once per render, only if gated on** | RTF 0.04 |

### 9.2 The 20× test

The prompt rightly calls out that a model which improves quality but multiplies runtime 20× is not practical. Estimated end-to-end impact on a 2-hour source:

| Scenario | Added time | Multiplier vs. current |
|---|---|---|
| Phase 0 only (REZE protocol on existing cloud LLM) | +1–3 min (parallel scoring calls) | **~1.1×** |
| + F1 fix (real librosa features) | +seconds | ~1.0× |
| + Silero VAD + LightGBM pacing | +seconds | ~1.0× |
| + OSNet re-ID + scene detection | +1–2 min CPU | ~1.1× |
| + CLIP/u2netp per-shot scoring | +2–5 min CPU | ~1.2× |
| + Nemotron-3 diarization | +3–8 min CPU | ~1.3× |
| + local VLM candidate scoring | +5–15 min (GPU: single-digit min) | ~1.5× on GPU, ~2× on CPU |
| **Full stack, CPU-only** | **~15–35 min** | **~2×** |
| **Full stack, GPU** | **~5–12 min** | **~1.3×** |

**Verdict: the full stack roughly doubles CPU wall-clock and adds ~30% on GPU.** For a tool that turns 2-hour podcasts into shorts, that is a defensible trade — and every piece is individually toggleable, so the multiplier is a dial, not a commitment.

### 9.3 Caching and incremental design

- **Per-source-video results** (diarization, PANNs, scene detection, VAD) are cached keyed by source hash. Re-running a candidate never recomputes them.
- **Per-shot results** (CLIP, saliency) cached keyed by shot boundaries.
- **Per-candidate results** (highlight score) cached per candidate hash.
- **Incremental**: if the user adjusts a boundary, nothing upstream of that boundary recomputes.

---
## Phase 10 — Failure and Fallback Design

The rule, per the prompt: **MODEL CONFIDENT → use ML signal · MODEL UNCERTAIN → deterministic fallback · MODEL FAILS → existing system continues safely.** No neural model failure may break the pipeline.

### 10.1 Universal fallback contract

Every ML component implements:

```rust
trait MlSignal {
    fn confidence(&self) -> f64;          // 0.0–1.0
    fn available(&self) -> bool;          // model loaded, inputs present
    fn fallback_reason(&self) -> Option<&str>;
}
```

Decision rule at every call site:

| State | Action |
|---|---|
| `available = false` | Use existing deterministic path. Log reason. Continue. |
| `confidence < LOW` (0.40) | Ignore model output. Use existing deterministic path. |
| `confidence ∈ [LOW, HIGH)` (0.40–0.75) | Blend: model signal weighted by confidence, deterministic signal as prior. |
| `confidence ≥ HIGH` (0.75) | Model signal governs, within hard safety limits. |
| Model returns NaN / implausible output | Treated as failure → deterministic path. |

### 10.2 Per-component failure modes and fallbacks

| Component | Failure mode | Fallback |
|---|---|---|
| REZE clip scorer (cloud LLM) | Provider outage, rate limit, malformed JSON | Other 7 providers round-robin; final fallback → current timestamp prompt; scoring never blocks rendering |
| REZE scorer (local VLM) | VRAM unavailable, model missing | Skip local scoring; use cloud score; or skip highlight score entirely |
| Diarization (Nemotron-3) | Model load failure, audio too short, >8 speakers | `S1`-only behavior — **identical to today**; `hookSpeaker` mapping simply stays unmapped |
| WhisperX | Model download failure | Deepgram cloud path (already exists) |
| Active speaker fusion | Embedding failure, low confidence | Mouth-motion heuristic (**F5**) decides — current behavior |
| OSNet re-ID | Model missing, embedding collapse | Positional-only association — current behavior |
| Smart Pacing classifier | Model missing, out-of-distribution audio | Current acoustic thresholds decide — current behavior |
| CLIP aesthetic scorer | Model missing | Current rule-based crop — current behavior |
| Silero VAD | Model missing | Energy-based VAD (existing) |
| DeepFilterNet2 | Binary missing, crash | Bypass — audio passes through unenhated (this is already the default-off state) |
| Scene detection | Failure | Whole video treated as one shot |

### 10.3 System-level failure handling

- **Model-loading failure** → lazy load; every model is optional. Missing model = deterministic path, **never an error dialog**.
- **GPU failure / unsupported hardware** → `ort` execution-provider fallback chain: **CUDA → DirectML → CPU**. CPU is the universal floor.
- **Corrupt model file** → hash verification on download; on corruption, re-download once, then fall back.
- **Missing input** (e.g., no audio track for diarization) → component reports unavailable, downstream proceeds.
- **OOM during inference** → catch, log, fall back. For the VLM path specifically: quantize down or skip.
- **Timeout** → each component has a wall-clock budget; exceeding it = failure → fallback.

### 10.4 What never degrades

The render path itself. Even if **every** model is unavailable, AutoShorts must produce the same output it produces today. This is testable directly: a "all models disabled" run must be byte-comparable (modulo known nondeterminism) to the current build.

---

## Phase 11 — Evaluation Metrics

### 11.1 Per-component metrics

**Candidate discovery**
- Precision@K, Recall, HIT@1 (REZE's metric — directly comparable to 73.41 on QVHighlights)
- Agreement with human-selected highlights (Cohen's κ across ≥3 annotators)
- **Human evaluation required**: yes — this is the highest-stakes quality decision in the product.

**Active speaker**
- Accuracy, F1, full confusion matrix
- Speaker-switch latency (time from audio onset to track reassignment)
- **Specific regression tests**: laughing-listener suppression, overlap, side profiles
- Human evaluation: spot-check on real multi-person podcasts

**Diarization**
- DER (Diarization Error Rate) — compare to Nemotron-3's 12.73 on DIHARD-3 as a sanity floor, not a target
- Speaker confusion rate, overlap-handling coverage
- **Human evaluation**: Host/Guest mapping correctness (this is what AutoShorts actually needs)

**Smart Pacing**
- Precision of removable-pause detection; recall
- **False-removal rate** (the metric that matters — a bad cut is worse than a kept pause)
- Audio artifact rate (clicks/glitches at edit points — measurable via spectral discontinuity)
- Listener-rated naturalness (MUSHRA-style A/B against current system)
- **Human evaluation: required.** No automated metric captures "does this sound like a human paused on purpose."

**Adaptive framing**
- Face containment rate (0 clipped faces in output)
- Crop stability (frame-to-frame displacement distribution; unnecessary-motion metric)
- Pair retention rate (two-person scenes: both visible 100% of the time — hard constraint, so this is a pass/fail invariant)
- Subject clipping rate, headroom distribution
- **Human evaluation**: pairwise preference on crop quality

**T7 captions**
- Phrase-boundary accuracy vs. human annotation
- Timing error (ms between caption appearance and speech onset)
- Human naturalness rating
- **Karaoke invariant**: automated check that no per-word highlighting appears and chunk sizes stay in guardrails

**Audio**
- SIG-MOS / PESQ / STOI for any enhancement path — **with the baseline comparison the project already ran** (clean audio: 4.144 reference)
- Loudness consistency (LUFS distribution across output shorts)
- Speech distortion index

### 11.2 System-level metrics

- **End-to-end runtime** per source-hour (must stay within the Phase 9 budget)
- **Fallback invocation rate** (how often models are unavailable — a high rate means infrastructure problems, not quality problems)
- **Zero-regression invariant**: all existing suites (16 Python, 281 Rust, build) must stay green on every phase. This is the master acceptance gate.

### 11.3 Where human evaluation is mandatory

Three places, and only three: **candidate selection** (is this actually the best short?), **Smart Pacing** (does it still sound human?), and **crop quality** (is this a better frame?). Everything else has reliable automated proxies. Resist adding human eval elsewhere — it does not scale and the automated metrics are sufficient.

---

## Phase 12 — Interaction Analysis (Shared Representations)

The prompt asks explicitly whether one model can serve several systems. Answer: **yes, two shared layers, and this is the single most important architectural decision in the report.**

### 12.1 ONE Speaker Intelligence Layer

```
ONE SPEAKER INTELLIGENCE LAYER
   (Nemotron-3 diarization + OSNet re-ID + active-speaker fusion)
        ↓
   unified speaker identity + P(speaking) per track per time
        ↓
   ┌────────────┬─────────────┬────────────┬──────────────┐
   ↓            ↓             ↓            ↓              ↓
Active      Adaptive      T7         Candidate      Smart Pacing
Speaker     Framing       Placement   Ranking        (speaker continuity)
```

**One model, five consumers.** Diarization runs **once per source video** and re-ID embeddings ride along the existing track pipeline. The cost is paid once; the value is collected five times. This also **resolves F4** (Host↔diarization mapping) as a natural byproduct.

### 12.2 ONE Audio Intelligence Layer

```
ONE AUDIO INTELLIGENCE LAYER
   (Silero VAD + librosa features + PANNs events)
        ↓
   speech regions, prosody contour, reaction events, room tone
        ↓
   ┌────────────┬─────────────┬────────────┬──────────────┐
   ↓            ↓             ↓            ↓              ↓
Diarization  Prosody     Active      Smart Pacing    Caption
context      (T7, hook)  Speaker     (pause model)   Timing
```

Again: **one pass per source video, four to five consumers.** The same VAD output that feeds Smart Pacing also gates caption timing and provides the acoustic features that fix **F1**.

### 12.3 What this avoids

The prompt's warning is explicit: "Avoid introducing five separate large models when one shared representation could serve multiple subsystems." This design introduces **zero large models** into the default path. The largest default-path additions are Nemotron-3 (100M, one pass) and CLIP ViT-B/32 (151M, per shot). The local VLM path is optional and off by default.

### 12.4 Secondary reuse

- **Transcript embeddings** (free byproduct of any LLM call) → candidate redundancy detection (J), semantic novelty for hook scoring (H)
- **Scene boundaries** → crop scorer (E), camera-mode selection (E), re-ID re-initialization (F), caption re-anchor points
- **PANNs reaction events** → hook scoring (H), candidate discovery audio signal (A), T7 emphasis placement (G)

---

## Phase 13 — Do Not Overwrite Working Architecture

Classification of every proposed change, per the prompt's A/B/C/D scheme. **Default preference: AUGMENT.**

| Proposed change | Class | Why |
|---|---|---|
| REZE clip-scoring protocol | **A (augment)** | Same LLM providers, same `CandidateDraft`, different query structure. Boundary snapping untouched. |
| Fix fake acoustic analyzer (**F1**) | **C (replace weak heuristic)** — technically a bug fix | It never worked; replacing it with real signal processing restores the documented contract. |
| Smart Pacing pause classifier | **A** first, **C** later | Initially augment the thresholds; replace only once the weak-label dataset demonstrates superiority offline. |
| Active speaker fusion | **A** | Fusion layer *wraps* the mouth-motion heuristic; the heuristic remains the fallback. |
| OSNet re-ID | **A** | Embedding joins the existing positional cost function. Nothing is replaced. |
| Diarization stage | **B (new capability)** | New stage; failure mode = exactly today's behavior. |
| Crop quality scorer | **A** | Advisory score; containment is still hard. |
| Per-scene camera-mode selection + path smoothing | **A** | AutoFlip-style smoothing *inside* the existing framing engine. Biggest visible quality lever. |
| Scene detection | **A** | Feeds framing; no existing behavior removed. |
| T7 prosodic boundaries | **A**, deferred | Only moves boundaries inside existing size guardrails. |
| DeepFilterNet2 enhancement | **B**, default OFF | New capability only for degraded sources; measured to *hurt* clean audio. |
| Candidate redundancy detection | **B** | New, cheap, uses existing embeddings. |
| Render QA checks | **B** | New, deterministic, no render change. |
| Audio enhancement on clean audio | **D (not worth ML)** | Measured degradation. |
| Learned loudness | **D** | Standard exists. |
| End-to-end giant model | **D** | Forbidden by the prompt, untested by the suite. |

**Nothing in this report replaces a working subsystem.** The only "C" entries are the fake analyzer (which never worked) and the eventual Smart Pacing threshold replacement (gated on offline evidence).

---

## Phase 14 — Research Currency and Source Separation

Per the prompt, every significant claim is separated into **SOURCE FACT**, **ENGINEERING INFERENCE**, and **RECOMMENDATION**.

### Source facts (verified against primary sources, Sept 2026)

1. REZE achieves **73.41 HIT@1** on QVHighlights with a frozen VLM + deterministic aggregation, beating fully supervised models. Scoring vs. timestamp generation: **43.28 vs 12.32 mAP** from the same model. — arXiv:2608.04480
2. Rhapsody found **GPT-4o zero-shot ≈ a frequency baseline** on real podcast highlight extraction; the winning system was **Llama-3.2-1B + QLoRA + HuBERT/DVA audio features**; **audio was the only consistently additive modality**; best system reached only **49% hit rate**. YouTube most-replayed graph used as the labeling oracle. — COLM 2025 / arXiv:2505.19429
3. Nemotron-3-Diarization: **100M params, DIHARD-3 DER 12.73, 10-ms frames, up to 8 speakers**, OpenMDW v1.1, NeMo-Speech.cpp with Windows installer. — NVIDIA, Sept 2026
4. Silero VAD v6.2.3: **~2 MB, <1 ms/chunk, ROC-AUC 0.97, MIT**. — official repo
5. **NSNet2 degrades clean audio (SIG-MOS 4.144 → 3.866); RNNoise likewise (→ 3.884).** — measured on this project's corpus
6. DeepFilterNet2: **2.3M params, RTF 0.04 on laptop CPU, MIT/Apache, pure Rust binary**. — official repo
7. OSNet-x0_5/x0_25: **0.2–0.6M params** re-ID models. — official repo
8. YOLO26n ONNX with `nms=False`: **39 ms CPU**. **AGPL-3.0.** — Ultralytics
9. PySceneDetect AdaptiveDetector: **BSD-3, F1 91.6** on broadcast video. — official repo
10. u2netp: **4.7 MB** saliency, Apache-2.0. CLIP ViT-B/32: **MIT**. YuNet int8: **~5 ms CPU**, Apache-2.0.
11. Qwen3-VL-4B INT4 ~2–3 GB; VideoChat3-4B handles 3-hour video at 16 tok/frame; Gemma 3n E4B is the only small VLM with **native audio** (6.25 tok/s); InternVL3.5 is **Apache-2.0**. — official model cards
12. **No pretrained model classifies removable vs. intentional pauses.** — literature survey
13. 2025–26 learned-crop research (ProCrop, Venus, CROP, ShotCrop³) is dominated by **MLLM-based scorers**. — respective papers
14. DirectML carries an **opset-20 cap** and sustained-engineering status. — Microsoft docs

### Engineering inferences

- **I1:** The measured in-domain ASD result (Light-ASD 54% @ 82.1% vs. heuristic 87% @ 83.5%) implies a bare ASD swap is a regression; fusion + abstention is the correct integration.
- **I2:** The 120–160 ms/frame visual stack is over the 125 ms budget single-threaded, but per-shot caching for CLIP/u2netp makes the offline cost acceptable. This is a scheduling argument, not a benchmark claim.
- **I3:** `yolo11n.pt`'s AGPL-3.0 status in `src-tauri/models/` already creates a copyleft exposure for distributed binaries, independent of this report's recommendations.
- **I4:** F6 (persistence of only 18 of ~30+ columns) is the cheapest blocker to remove on the entire roadmap — it gates the Smart Pacing dataset.
- **I5:** The brain doc's staleness (internally "8.0", ~8+ wrong numbers) means it must not be used as an integration specification; code is authoritative.

### Recommendations

All of Phases 15–17 below.

---

## Phase 15 — Model Selection Criteria

The prompt is explicit: do not optimize for benchmark prestige; optimize for AutoShorts engineering value. Scoring the main candidates across the prompt's ten criteria (1–5, higher better):

| Model | Quality | Generalization | Speed | Local deploy | Memory | License | Impl. complexity | Robustness | Maintainability | Integration fit | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| REZE protocol (no weights) | 5 | 5 | 4 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | **48** |
| Silero VAD | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | **49** |
| OSNet re-ID | 4 | 4 | 5 | 5 | 5 | 4 | 4 | 4 | 4 | 5 | **44** |
| Nemotron-3 diarization | 5 | 4 | 4 | 5 | 4 | 4 | 3 | 4 | 3 | 5 | **41** |
| LightGBM pause model | 4 | 3 | 5 | 5 | 5 | 5 | 4 | 3 | 5 | 5 | **44** |
| PySceneDetect | 4 | 4 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 5 | **47** |
| CLIP + aesthetic head | 3 | 4 | 3 | 4 | 3 | 5 | 3 | 4 | 4 | 4 | **37** |
| u2netp | 3 | 3 | 5 | 5 | 5 | 5 | 5 | 3 | 4 | 4 | **42** |
| DeepFilterNet2 | 4 | 3 | 5 | 5 | 4 | 5 | 5 | 4 | 4 | 5 | **44** |
| WhisperX | 4 | 4 | 4 | 5 | 4 | 3 | 4 | 4 | 4 | 5 | **41** |
| YOLO26n | 4 | 4 | 4 | 5 | 4 | **1** | 3 | 4 | 4 | 4 | **37** |
| Qwen3-VL-4B INT4 | 5 | 4 | 2 | 3 | 2 | 4 | 2 | 3 | 3 | 3 | **31** |
| NSNet2 / RNNoise | 2 | 2 | 5 | 5 | 5 | 2 | 5 | 2 | 4 | 2 | **34** |

**Reading:** the winners are the *small, well-licensed, single-purpose* components — Silero, REZE-as-protocol, PySceneDetect, OSNet, DeepFilterNet2, and a tiny custom LightGBM. The prestige options (large VLMs, learned enhancement) score poorly on integration fit despite high raw quality. YOLO26n's score is a license score, not a capability score — resolve the AGPL question and it jumps.

**Selection rule for future candidates:** a model must clear 40/50 AND have a permissive license AND run on CPU before it enters the default path. Everything else is optional, off by default, or rejected.

---

## Phase 16 — Final Recommendation Matrix

Decision values per the prompt: **AUGMENT · REPLACE · NEW FEATURE · KEEP DETERMINISTIC**.

| Subsystem | Current Approach | Current Weakness | ML Opportunity | Model Candidates | Recommended Architecture | Expected Benefit | Compute Cost | Data Requirement | Integration Complexity | Risk | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Candidate discovery** | LLM transcript read → timestamps | Weakest known extraction pattern (Rhapsody: ≈ frequency baseline) | Clip scoring + deterministic aggregation | REZE protocol on existing 8 LLM providers; later local VLM (Qwen3-VL-4B / InternVL3.5) | Scoring prompt → per-clip score → mean-center + Gaussian smooth + Kadane/Otsu in Rust | **3.5× mAP** from the same model; HIT@1 class improvement | None (Phase 0) / 6–8 GB (Phase 2) | None (Phase 0) / most-replayed oracle | Low (Phase 0) | Low | **AUGMENT** — Phase 0, top priority |
| **Multimodal "acoustic" analyzer** | Punctuation & substring matching (**F1**) | Fabricated outputs; fake evidence strings | Real audio features first; learned score later | librosa features → Silero VAD → optional PANNs Cnn14-16k | Replace `analyze_acoustic_signals` with real DSP; PANNs reaction events as metadata | Honest hook composite; reaction-audio ranking | <1 GB, one pass | None | **Very low** | Low (it is a bug fix) | **REPLACE** — do this first |
| **Diarization / speaker identity** | None (`S1` everywhere; positional identity) | No identity; Host/Guest never reconciled (**F4**) | One diarization pass + re-ID embeddings | Nemotron-3-Diarization; WhisperX; OSNet-x0_25 | Once-per-video pass → speaker labels + embeddings → 5 consumers (Phase 12.1) | Resolves F4; speaker-aware everything | ~1–2 GB, minutes | None | Medium | Medium (new stage) | **NEW FEATURE** |
| **Active speaker** | Mouth-motion absdiff ≥4.0 (**F5**) | Laughing listeners, illumination, no audio reference | Fusion with abstention | Mouth motion × Light-ASD × audio-speaker consistency | Fusion layer over the heuristic; heuristic stays as fallback | Correct speaker in overlap/laughter | Low | In-domain labeled minutes | Medium | Low (fallback exists) | **AUGMENT** |
| **Track persistence** | Positional proximity (**F3**) | Cuts, occlusion, multi-person instability | Appearance embeddings | OSNet-x0_5/x0_25 | Embedding term in existing association cost | Stable identity | ~5 ms/track/frame | None | Low | Low | **AUGMENT** |
| **Smart Pacing** | Acoustic thresholds, one-speaker calibration (**F11**) | Cannot separate removable from intentional pauses | Small pause classifier | Silero VAD + librosa + LightGBM/1D-CNN | `P(removable) → existing safety rules → edit` | Speaker-independent good cuts | Negligible | **Free weak labels** (fix **F6** first) | Low–Medium | Low (fallback = current) | **AUGMENT → REPLACE** |
| **Adaptive framing / crop** | Rule margins "not re-derived from first principles" | No quality score; jittery trajectories | Crop scorer + per-scene camera modes | CLIP ViT-B/32 + aesthetic head + u2netp + AutoFlip method + PySceneDetect | Score is advisory; containment hard; AutoFlip smoothing | Better crops, smoother motion | 60–100 ms/shot | 2–5K rated crops | Medium | Medium (visual regression risk) | **AUGMENT** |
| **T7 captions** | Word-timing gaps | Boundaries ignore prosody | Phrase-boundary tagger | Small custom tagger (no good pretrained) | Boundaries move only inside size guardrails | Naturalness, Hindi/Hinglish | Negligible | Hand-labeled seed | Medium | **Medium-high** (aesthetic) | **AUGMENT** — deferred to Phase 3 |
| **Hook intelligence** | LLM + fabricated audio (**F1**) | Fake acoustic half | Real features → learned engagement | librosa + PANNs events | Advisory composite; **never overrides payoff endpoint** | Honest signals | One pass | Shared with discovery | Low | Low | **AUGMENT** |
| **Audio enhancement** | Conditional, quality-gated | **Degrades clean audio** (measured) | Only for degraded sources | DeepFilterNet2 (native Rust) | OFF by default; gated on quality | Recovery of bad sources | RTF 0.04 | None | **Very low** (Rust binary) | Low | **NEW FEATURE** (default OFF) |
| **Loudness** | BS.1770 / LUFS | None | None | — | — | — | — | — | — | — | **KEEP DETERMINISTIC** |
| **Boundary snapping** | Deterministic thresholds on LLM scores | Crown jewel | Better upstream probabilities only | — | Feed better scores; keep the layer | — | — | — | — | — | **KEEP DETERMINISTIC** |
| **Payoff endpoint** | LLM → alignment → `payoff_end` → `candidate_end` | None | None | — | — | — | — | — | — | — | **KEEP DETERMINISTIC** |
| **A/V sync, render, caption geometry** | Deterministic | None | None | — | — | — | — | — | — | — | **KEEP DETERMINISTIC** |
| **Scene detection** | None | Crops span cuts | Deterministic scene cuts | PySceneDetect AdaptiveDetector | Once per video, cached | Feeds framing, re-ID, captions | Low | None | Low | Low | **NEW FEATURE** |
| **Candidate redundancy** | None | Duplicate shorts ship | Embedding similarity | Transcript embeddings | Post-selection dedupe | No duplicate shorts | Negligible | None | Very low | Low | **NEW FEATURE** |
| **Render QA** | None | Defects uncaught | Deterministic checks (not ML) | Sync/loudness/containment checks | Post-render validation | Catches bad renders | Negligible | None | Low | Low | **NEW FEATURE** |

---

## Phase 17 — Proposed AutoShorts 11.0 ML Architecture

```
                    AUTOshorts 11.0

                     SOURCE VIDEO
                          │
             ┌────────────┴────────────┐
             ↓                         ↓
        VISUAL MODELS             AUDIO MODELS
             │                         │
        YuNet int8 (~5 ms)       ASR: WhisperX / Deepgram
        YOLO26n (license-first)  Diarization: Nemotron-3 (once/video)
        OSNet re-ID (~5 ms)      Silero VAD (<1 ms)
        u2netp saliency          librosa prosody features
        PySceneDetect scenes     PANNs reaction events
        CLIP + aesthetic head    DeepFilterNet2 (gated, OFF default)
             │                         │
             └────────────┬────────────┘
                          ↓
                 MULTIMODAL FEATURES
                 (cached per video / per shot)
                          │
          ┌───────────────┼──────────────────┐
          ↓               ↓                  ↓
   REZE Highlight      Speaker         Pause / Prosody
   Scorer (clip        Intelligence    Classifier
   yes/no,             Layer           (Silero + LightGBM)
   deterministic       (one layer,     (P removable,
   aggregation)         5 consumers)     P breath, P intentional)
          │               │                  │
          └───────────────┼──────────────────┘
                          ↓
                 AUTOshorts DECISION
                      ENGINE
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   Hook / Payoff      Framing           Captions
   Candidate          Adaptive           T7 Placement
   Ranking            Tracking           (prosody-aware,
   (redundancy        (crop scorer       guardrailed)
    dedupe)            advisory only)
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ↓
                 DETERMINISTIC
                 SAFETY LAYER
             (boundary snapping, containment,
              DualFrame rules, payoff endpoint,
              pacing safety, chunk guards,
              BS.1770 loudness, A/V sync)
                          ↓
                       FFmpeg
                          ↓
                    FINAL SHORT
                          ↓
                 Render QA (deterministic)
```

### 17.1 The three rules this architecture obeys

1. **Models produce signals; the deterministic engine retains control.** Every model feeds an existing decision; no model owns a decision that today's tests pin.
2. **Failure is invisible to the user.** Every component has a deterministic fallback that reproduces today's behavior. A machine with no models installed produces today's AutoShorts.
3. **Two shared layers, not five large models.** Speaker Intelligence and Audio Intelligence each run once per source video and serve every consumer.

### 17.2 Phased execution plan

**Phase 0 — Zero new models, zero new dependencies (days)**
1. **Fix F1**: replace `analyze_acoustic_signals` with real librosa features. Bug fix, not a feature.
2. **Fix F6**: persist the full `CandidateDraft` (all ~30+ columns) at `replace_candidates`. Unlocks the pacing dataset.
3. **Fix F8**: remove hardcoded `Autoshorts 5.0` paths; make `find_speaker_tracker_script` fail loudly instead of silently cropping to x=656.
4. **Fix F9**: remove/re-wire dead config constants so config edits take effect.
5. **Fix F7**: convert absolute pixel areas to resolution-relative fractions.

**Phase 1 — The big win (weeks)**
6. **REZE protocol in `llm.rs`**: convert the timestamp-emitting prompt to per-clip yes/no scoring + deterministic aggregation. Validate HIT@k on the regression corpus.
7. **Resolve the AGPL-3.0 question** on `yolo11n.pt` before any further model bundling.

**Phase 2 — Speaker Intelligence layer (weeks)**
8. Nemotron-3 (or WhisperX) diarization, once per video, cached. Wire to Host/Guest mapping, Smart Pacing continuity, captions.
9. OSNet re-ID embeddings in track association.
10. Active-speaker fusion layer over the mouth-motion heuristic.

**Phase 3 — Learned pacing and framing (months)**
11. Silero VAD + LightGBM pause classifier on weak edit-history labels. A/B against current thresholds; replace only on evidence.
12. PySceneDetect + per-scene camera modes + AutoFlip-style smoothing.
13. CLIP aesthetic head; candidate crop scorer (advisory).
14. T7 prosodic boundary tagger, behind guardrails and A/B.

**Phase 4 — Optional, off by default (months)**
15. Local VLM candidate scoring (Qwen3-VL-4B / InternVL3.5) via lazy Python sidecar.
16. DeepFilterNet2 for degraded sources only.
17. PANNs reaction metadata; candidate redundancy dedupe; render QA.

### 17.3 What was deliberately left out

- **No end-to-end model.** The prompt forbids it; the test suite forbids it; the measured evidence says small specialized components win on integration fit.
- **No ML for loudness, A/V sync, or caption geometry.** These are standards and invariants, not judgment calls.
- **No replacement of boundary snapping, the payoff endpoint, containment, or DualFrame rules.** These are the architecture.
- **No cloud dependency.** Every model in the default path runs locally; cloud LLM providers remain one option among eight.

### 17.4 The honest uncertainty

- The **49% best-system hit rate** from Rhapsody is a warning: podcast highlight extraction is genuinely hard, and no model in this study will make it easy. REZE's 73.41 is on QVHighlights, a different distribution — expect lower on long-form podcasts, and say so rather than promising a number.
- The **measured ASD result** (off-the-shelf below the existing heuristic) is evidence that AutoShorts' domain is unusual and that "SOTA model" and "better for us" are not the same sentence.
- The **aesthetic head** quality is unvalidated until rated crops exist. The CLIP+head stack is a well-trodden pattern, but its benefit here is an inference, not a measurement.
- **Hardware assumptions** were sized to a 4–12 GB consumer GPU. If the target machine is CPU-only, Phase 4 stays off and Phase 3 still works — the ~2× CPU multiplier in Phase 9 is the floor.

---

## Appendix A — Consolidated defect list (from code inspection)

| ID | Severity | Finding | Location | Fix phase |
|---|---|---|---|---|
| F1 | **BLOCKING (correctness)** | "Acoustic" analyzer fabricates energy/pitch/laughter/pause from punctuation and substrings; never opens audio; evidence strings assert computed-but-nonexistent results | `multimodal_hook_analyzer.py:186-264` | Phase 0 |
| F6 | **BLOCKING (data)** | Only 18 of ~30+ `CandidateDraft` columns persisted; multimodal/hook/closure/endpoint data discarded | `db.rs` (`replace_candidates`) | Phase 0 |
| F3 | High | No subject identity / re-ID; framing keys on nearest face to crop center | `speaker_tracker.py` | Phase 2 |
| F4 | High | No Host↔diarization mapping | `llm.rs`, `transcription.rs` | Phase 2 |
| F5 | High | Active speaker = mouth-motion absdiff, no audio reference, illumination-sensitive | `speaker_tracker.py` | Phase 2 |
| F8 | High | Hardcoded `d:\College\Autoshorts 5.0\` path; silent fallback to center crop x=656 | `speaker_tracker.py:111-112`, `media.rs:272` | Phase 0 |
| F7 | Medium | Absolute pixel areas assume 1080p | `speaker_tracker.py` | Phase 0 |
| F9 | Medium | Dead config constants; run loop re-declares them as locals | `speaker_tracker.py` | Phase 0 |
| F10 | Medium | No sidecar/externalBin bundling; distribution assumes dev machine | `tauri.conf.json` | Phase 4 |
| F11 | Low | Authors' caveats: margins "not re-derived from first principles"; breath thresholds one-speaker-calibrated | `speaker_tracker.py` comments | Phase 3 |
| F12 | Low | Brain doc stale ("8.0" internally, ~8+ wrong numbers) | `11.0 brain.md` | continuous |
| F13 | **LEGAL** | `yolo11n.pt` is AGPL-3.0, bundled in shipped `src-tauri/models/` | model card, `src-tauri/models/` | **before any model work** |

## Appendix B — Test baseline (verified this session)

| Suite | Result |
|---|---|
| Python — 16 suites total | **16/16 green** |
| smart_pacing | 55/55 |
| Smart Pacing v2 (SP2) | 108/108 |
| audio_intelligence | 48/48 |
| hook_closure | 47/47 |
| hook_ending_optimization | 29/29 |
| caption_visual_qa | 100 checks |
| active_speaker | 61/61 |
| dualframe | 55/55 |
| applied_features / multimodal_hook / others | green |
| Rust `cargo test` | **281 passed, 1 failed (test-only env-var race in `pacing::tests::test_v2_disabled_keeps_v1`; passes with `--test-threads=1`), 1 ignored** |
| `npm run build` | exit 0 |

**This baseline is the zero-regression reference for every phase above.**
