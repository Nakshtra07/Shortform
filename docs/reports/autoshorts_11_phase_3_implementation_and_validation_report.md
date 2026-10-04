# AutoShorts 11.0 — Phase 3 Implementation & Validation Report

**Date:** 2026-09-29
**Phase:** 3 — Learned Pacing + Scene Intelligence + Smoothed Learned Framing + Crop Quality Scoring + T7 Prosodic Boundaries
**Status:** **PHASE 3 COMPLETE** (learned decision influence for Smart Pacing formally documented as DATA-BLOCKED — see §23)
**Primary engineering model:** GLM-5.3-Flash
**Research authority:** `ML_RESEARCH_REPORT_AUTOSHORTS_11.0.md` (Atria). Implementation authority: the actual AutoShorts source.

---

## 1. Phase 3 Objective

Add specialized learned/advisory signals to Smart Pacing, framing (scene-aware camera behavior, AutoFlip-style smoothing, crop quality scoring), and T7 phrase boundaries — while the deterministic AutoShorts engine retains full control and every existing invariant survives. No end-to-end AI replacement; nothing learned may bypass the deterministic safety layer.

## 2. Atria Research Mapping

| Atria recommendation | What was built | Compliance |
|---|---|---|
| Silero VAD v6.x front-end (MIT, ~2 MB) | `pause_intelligence.py` uses `silero-vad` **6.2.3** (verified MIT) for speech regions + per-gap speech probability | ✔ exact version |
| librosa-family features, **NOT openSMILE** | numpy-only acoustic features (RMS dB, HF ratio, spectral flatness, ZCR, dip depth, level drop) + VAD features | ✔ openSMILE absent |
| LightGBM pause-type classifier (no pretrained model exists) | `pause_classifier_v1` (LightGBM 4.7.0, verified MIT), multi-class pause typing + derived P(removable) | ✔ |
| Weak labels from edit history ("gap removed → 1, gap survived → 0") | No edit history existed → **built the data pipeline first**: dataset builder + per-run telemetry capture | ✔ data rule honored |
| PySceneDetect AdaptiveDetector (BSD-3), source-level, cached | `scene_intelligence.py` + `scene_intelligence.rs` engine, per-source cache | ✔ exact detector |
| AutoFlip-style per-scene smoothing | Motion-aware transition windows (bounded peak pan speed), per-scene reframe-threshold advisory | ✔ advisory only |
| CLIP ViT-B/32 + aesthetic head + u2netp crop scorer | Advisory corridor-legal crop ranking with the existing deterministic composition score; **learned head DATA-BLOCKED** (no rated-crop dataset); CLIP/u2netp postponed | ✔ data rule honored |
| Small T7 prosodic tagger | `t7_prosody.py` feature/annotation pipeline (F0 slope, energy drop, pause, punctuation); **model DATA-BLOCKED**; Rust chunker untouched | ✔ data rule honored |

## 3. Current Baseline (verified by inspection)

- Smart Pacing v1 (FFmpeg silencedetect + 5 deterministic edit types) and v2 (breath_detect acoustic gap classification) with a fully deterministic safety pipeline (minimum-removal gates, word clearance, sliver cancellation, ≤40 % and ≥3 s circuit breakers, word-straddle rejection, Rust-side `validate()`).
- **No per-gap/edit/outcome data persisted anywhere** (Atria F6): only `clips.applied_features` booleans and an aggregate `render_log` string.
- Framing: pixel-diff shot detection at 8 fps inside `sample_video_and_detect_shots` (never cached, never serialized); savgol + cubic-Hermite smoothing with 0.30 s fixed transitions; Camera Lock Operator with deadbands/cooldowns; hard containment via `solve_containment_crop_x` + `sanitize_face_bbox_for_containment` + `validate_and_correct_trajectory`.
- T7 chunking: `segment_speech_chunks` (0.12 s soft pause threshold; 3-word / 1.5 s / sentence / comma hard guardrails) → `group_t7_phrase_units` (0.35 s pause, speaker change, 2.6 s pair ceiling). No acoustic data anywhere in the caption path.
- Environment before Phase 3: silero-vad/lightgbm/scenedetect/onnxruntime MISSING; installed and verified this phase (lightgbm 4.7.0 MIT, scenedetect 0.7.1 BSD-3, silero-vad 6.2.3 MIT, onnxruntime 1.30.0 MIT).

## 4. Smart Pacing Implementation

New file `scripts/pause_intelligence.py` (shared feature infrastructure):
- Single-pass 16 kHz mono decode; Silero VAD speech regions + per-gap speech probability (512-sample windows).
- Per-gap acoustic features (`gapRmsDb`, `preSpeechRmsDb`, `postSpeechRmsDb`, `hfRatio`, `flatness`, `zcr`, `dipDb`, `levelDropDb`) + structural features (`durationSec`, `speakerChange`, `endsSentence`, `endsClause`, `startsFiller`, `endsFiller`, `preSpeechRate`, `postSpeechRate`, `relPosInClip`) + VAD features (`vadSpeechProb`, `vadSilenceFrac`) + echo of breath_detect evidence. 23-feature deterministic vector (`FEATURE_VERSION = "1"`).
- LightGBM model loading with **feature-version pinning** (mismatch refuses to load), corrupt-model and missing-model fallbacks; `score_gaps_with_model` derives P(removable) = P(breath_pause)+P(waiting_pause).
- Flag `AUTOSHORTS_SMART_PACING_LEARNED` (Rust-compatible env semantics).

Integration into `smart_pacing.py`:
- `run_breath_detection` now attaches learned evidence via `_attach_learned_pause_evidence` — **EVIDENCE ONLY**: the model cannot create, veto, or reshape edits (justified in §6; plan equality is test-enforced).
- `_emit_pacing_telemetry` (opt-in via `AUTOSHORTS_PACING_TELEMETRY_DIR`) appends per-gap `{features, category, detectorConfidence, pRemovable, removed}` rows — the weak-label data pipeline for every future run. Failure never affects a render.

## 5. Smart Pacing Dataset

Built with `scratch/phase3/build_pacing_dataset.py` on **real, diverse media** (10 windows: 5×UR Cristiano interview at different offsets, Fatigue talking-head, Messi commentary, Jay Shetty, Hindi `क्या आप…`, rio_90, messi):

- **204 gaps across 10 videos**; labels = the deterministic engine's own per-gap category decisions (weak supervision by design, documented) + engine removal outcomes (6 removed).
- Leakage control: **video-level split** (holdout = `rio_1350` + `messi_120`, 36 gaps); no gap-level random splitting.
- Live Deepgram nova-3 transcripts cached per window; real silencedetect + real breath detection + real engine plans.

## 6. Smart Pacing Model

`scratch/phase3/train_pause_classifier.py` — reproducible training (seed pinned, feature/dataset versions stored):

- **Model:** LightGBM multi-class (6 classes present in training: breath, waiting, sentence, normal word gap, nonspeech-unknown, speaker transition; dramatic pause absent from training data and reported as such). P(removable) derived from the removable classes.
- **Honest holdout results (video-level):** category agreement **69.4 %**; removable-gate agreement **88.9 %** (precision 0.0 / recall 0.0 — the model gated nothing on the holdout); the single engine-removed holdout gap received **P(removable) = 0.014**.
- **Decision (per the Atria contract and the no-fabrication rule):** the evidence does NOT support giving the model decision influence — it would veto a real removal. The learned stage therefore ships **evidence-only**; decision influence is formally **DATA-BLOCKED** until the telemetry pipeline accumulates enough real outcomes. This is the "model unavailable / low confidence → existing thresholds decide" contract taken to its honest conclusion.
- Artifacts: `pause_classifier_v1.txt` + `.meta.json` (versions, classes, threshold) + `training_report.json` (full metrics).

## 7. Scene Detection

- `scripts/scene_intelligence.py`: PySceneDetect **AdaptiveDetector** (threshold 3.0, min scene len 12 frames) over the WHOLE source; JSON doc with `sceneId/start/end`, detector + config versions, source hash; per-source cache file with stale/corrupt rejection; failure → single-scene fallback.
- `src/scene_intelligence.rs`: Rust engine with the same cache semantics (hash + version keys; stale/corrupt → miss + remove), feature flag `AUTOSHORTS_SCENE_INTELLIGENCE`, sidecar JSON parsing tolerant of progress lines, and a fallback result that never blocks rendering. Wired into `render_flat_clip_for_candidate` before framing; the cached doc is handed to the tracker via a new optional `--scene-cuts-json` sidecar argument (`SpeakerIntelSidecarInputs.scene_cuts_json`).
- Measured: **11 scenes in 40.3 s** cold on the 90 s interview; cache hit instant; boundaries align with known conversation cuts (5.96 s ≈ speaker switch).

## 8. Camera-Mode Intelligence

`compute_shot_motion_profile` (per-shot mean 160×90 gray diff) drives a **bounded advisory** on the Camera Lock Operator: in static shots (diff < 6.0) the existing reframe thresholds are scaled by ≤1.25× (hard cap 1.5×). No new camera modes were invented (the mode vocabulary remains the real one: `two_shot_both_fit`, `solo_close_up`, `ots_over_the_shoulder`, `wide_two_shot`, `multi_person_panel`, `fallback_*`, DualFrame `single/dual_stack`); the advisory only tunes existing deterministic constants. Two-person adaptive shared composition is upstream of every touchpoint and cannot be affected.

## 9. AutoFlip-Style Smoothing

`build_ffmpeg_expr` transitions are now motion-aware: `dur = clamp(|Δx|/720 px·s⁻¹, 0.30 s, 0.90 s)` behind `AUTOSHORTS_FRAMING_SMOOTHING`. Endpoints (containment-validated keyframe positions) and hard-cut semantics are untouched.

**Measured isolation (honest):** with keyframes held identical (scene advisory off), smoothing-on vs smoothing-off produces **byte-identical trajectories** on both sources — every large move in these corpora is a hard cut, where instant jumps are intended. The dynamic window is therefore provably safety-neutral and active only for future non-cut macro-pans.

## 10. Crop Scorer

Advisory ranking among **corridor-legal** x candidates (the corridor from `solve_containment_crop_x` is exactly the containment-feasible set): the existing deterministic `evaluate_crop_composition_score` ranks {current x, current x ± half-deadband}; a candidate must beat the solved position by **≥ 1.0 score** to be chosen; hard containment is re-checked downstream and the trajectory validator remains the guarantee. On rio_90 it evaluated 185 decisions (3 candidates each) with stable positions.

**Production-binding incident (caught by the suite, fixed):** the scorer is config-independent and initially overrode `FramingConfig`-driven positions, collapsing `test_framing_config_suite`'s "different config ⇒ different trajectory" contract. Fix: the scorer is **opt-in (default OFF)** and margin-gated. The suite passes again. This validated the "scorer must never override deterministic selection" rule with a concrete test.

**Learned aesthetic head: DATA-BLOCKED** — no rated-crop dataset exists (2–5 k rated crops per Atria); fabrication prohibited. CLIP/u2netp are postponed with it (no weights bundled, no license exposure).

## 11. Crop Dataset

Not built beyond the collection path design: the telemetry pattern proven for pacing (`AUTOSHORTS_PACING_TELEMETRY_DIR`) is the template for a future crop-rating collector (crop candidates + chosen + composition score per decision are already logged to stderr; persisting them to a dataset needs only a sink). Formally documented as data-blocked.

## 12. T7 Prosodic Model

**DATA-BLOCKED (model) — pipeline implemented.** `scripts/t7_prosody.py` computes annotation-ready per-word-gap records (pause, F0 slope via voicing-pooled autocorrelation, energy drop, punctuation class, speaker change, VAD-agnostic) and a documented rule-based triage prior (`pBoundaryRule`) usable only for offline annotation triage. Verified on real media: **320 gap records** on the 90 s interview. The Rust chunker (`captions.rs`) is byte-untouched; the future integration contract (modulate only the 0.12 s soft threshold inside the hard guardrails, flag-gated, current algorithm as fallback) is specified in the BRAIN. A few hundred labeled clips (Atria seed estimate) are the entry condition.

## 13. T7 Dataset

Annotation schema shipped (`t7_prosody.py` JSONL-ready records with word indices, features, and source identity; feature/model versions specified). No labeled chunking data exists; none was fabricated.

## 14. Shared Feature Infrastructure

- Silero VAD is loaded lazily once per process and shared (`pause_intelligence.get_silero_vad`); the 16 kHz decode is single-pass per request.
- Scene boundaries are computed once per source (Rust-cached) and reusable by framing now and re-ID/caption systems later.
- Speaker Intelligence (Phase 2) remains available as context (speaker-change features already flow into pause typing; fusion states remain framing-adjacent).
- Pacing telemetry accumulates the future training corpus as a byproduct of real runs.

## 15. Cache

| Cache | Key | Invalidation verified |
|---|---|---|
| Scene detection (Rust + Python) | source hash + detector version + config version | stale version → miss + remove (unit-tested); corrupt → miss |
| Pause model load | model path + feature version | feature-version mismatch refuses load (test 08); corrupt file refused (test 09); in-process memoization |
| Pacing telemetry | append-only JSONL per day | n/a (write-only) |

## 16. Feature Flags

| Flag | Default | Off behavior | On behavior |
|---|---|---|---|
| `AUTOSHORTS_SMART_PACING_LEARNED` | enabled | no model load, no evidence | pRemovable attached (evidence-only) |
| `AUTOSHORTS_SCENE_INTELLIGENCE` | enabled | no scene sidecar run, tracker cuts only | cached boundaries merged + static advisory |
| `AUTOSHORTS_FRAMING_SMOOTHING` | enabled | fixed 0.30 s transitions | motion-aware windows (measured neutral when untriggered) |
| `AUTOSHORTS_CROP_SCORER` | **disabled (opt-in)** | containment solve untouched | margin-gated corridor ranking |

Flag semantics match the existing Rust `env_flag` (`0/false/off` disable; unset enables — except the crop scorer's explicit opt-in default). All verified by unit tests (`test_01`, Rust scene tests) and runtime runs.

## 17. Failure Handling

Verified behaviors: pause model missing (`test_05`/`test_06`), corrupt model (`test_09`), feature-version mismatch (`test_08`), Silero unavailable (guarded import; VAD features disabled), scene sidecar missing/failed → single-scene fallback (Rust `test_process_source_missing_file_falls_back`), scene JSON corrupt/missing → tracker pixel cuts unchanged (`test_11`), out-of-window boundaries dropped (`test_12`), telemetry failure → render unaffected (`test_17`), scorer failure → containment solve kept (per-decision catch). Every path degrades to the existing deterministic behavior.

## 18. License Audit

`docs/license_audit/phase3_license_audit.md` — all four new packages verified from installed wheel metadata: lightgbm **MIT**, scenedetect **BSD-3-Clause**, silero-vad **MIT** (+ MIT weights), onnxruntime **MIT**. openSMILE excluded (Atria licensing flag). No restricted weights bundled. CLIP/u2netp not integrated (data-blocked), so no license exposure.

## 19. Performance (measured, CPU-only)

| Stage | Measured |
|---|---|
| Silero VAD + features (90 s clip, 22 gaps) | ~2–4 s (single decode + 512-sample windows) |
| Pause model scoring (22 gaps) | < 50 ms |
| Scene detection (PySceneDetect, 90 s source) | **40.3 s cold**, 0 s cached (whole-source pass) |
| Scene merge in tracker | negligible |
| Crop scoring (185 decisions × 3 candidates) | negligible (reuses in-memory geometry) |
| T7 prosody features (320 gaps) | ~3 s |
| Pacing telemetry (22 gaps) | ~2 s (decode dominates; opt-in) |

Cache effect: scene detection is the only material cost and is per-source cached (80×+ on revisit, mirroring the Phase 2 cache behavior).

## 20. Real-Media A/B Results

Corpora: rio_90 (90 s, two-person interview) and messi_120 (60 s, commentary + fans). Measurement: ffmpeg-expression simulation (x(t) sampled at 10 ms).

| Comparison | Result |
|---|---|
| Pacing: A (current) vs B (learned evidence on) | **Plans byte-identical** (test 16 enforces evidence-only) — no regression by construction; decision influence deferred pending data |
| rio_90 framing: A vs full Phase 3 | Composition unchanged (same mode/segments); max smooth pan **1100 → 759 px/s (−31 %)**; fast-pan samples **23 → 14 (−39 %)**; reframe churn (distinct positions) **107 → 12** — attributed to the static-scene advisory, not the transition change (see §9) |
| messi framing: A vs full Phase 3 | Composition unchanged; one snappier transition at a newly detected scene boundary (2900 px/s over ≤0.9 s) — intended scene-cut behavior, containment intact |
| rio_90 smoothing isolation (A vs smoothing-only) | **Identical trajectories** — the transition widening is provably neutral when untriggered |
| T7: A vs B | Not applicable (model data-blocked); caption QA 100/100 green confirms visual identity unchanged |

## 21. Full Regression (all fresh, 2026-09-29)

| Suite | Result |
|---|---|
| `cargo test -p autoshorts --lib` | **341 passed, 0 failed** (4 new scene-intelligence tests included) |
| `npm run build` | pass (12.5 s) |
| Smart Pacing (v1) | 55/55 passed |
| Smart Pacing 2.0 | 108/108 passed |
| Active Speaker | OK (61 tests) |
| Framing regression / config / geometry / path | 16 PASS; OK; OK; OK (config suite initially caught the crop-scorer incident → fixed → OK) |
| DualFrame suite + real render | OK; A/V Δt = 0.0000 s |
| Caption Intelligence / Caption QA | OK; 100 checks, 0 failures |
| Hook Closure / Hook Ending / Audio Intelligence / Applied Features | OK; OK; 48 passed; OK |
| Sampling memory / Ultralytics adapter / Multimodal hook / Multimodal acoustic | OK |
| **Phase 3 suite (new, 17 tests)** | **OK** |

## 22. Invariant Audit (all preserved)

- [x] `payoff_end` authoritative; `candidate_end = payoff_end` — untouched (lib.rs sections verified unmodified; Phase 3 additions sit in pacing-telemetry (Python), scene block (pre-framing), and tracker internals)
- [x] No 75 s/90 s artificial truncation
- [x] Phase 1 / Phase 1.1 discovery unchanged
- [x] TimestampGeneration default; WindowScoring experimental
- [x] Adaptive one-person unchanged
- [x] Adaptive two-person shared composition preserved (advisories are upstream-neutral; two-shot paths return before any Phase 3 consumption)
- [x] No active-speaker follow in adaptive two-person; no DualFrame in adaptive
- [x] Original 9:16 unchanged
- [x] T7 visual design unchanged — `captions.rs` has zero Phase 3 modifications; Caption QA 100/100
- [x] No karaoke
- [x] Face-safe T7 placement preserved
- [x] Smart Pacing safety rules preserved (SP2 invariants suite green; learned stage is evidence-only and test-enforced)
- [x] Deterministic boundary snapping preserved
- [x] Hard face containment preserved (corridor-legal ranking only; validator intact)
- [x] A/V sync preserved (Δt = 0.0000 s)
- [x] Loudness preserved (BS.1770 path untouched)
- [x] Render correctness preserved
- [x] Every learned model has a safe fallback (§17)

## 23. Production Status per Component

| Component | Status | Basis |
|---|---|---|
| Shared pause feature infrastructure (Silero VAD + features) | **PRODUCTION (flag-gated)** | deterministic, tested, harmless when unused |
| Learned Smart Pacing (LightGBM pause typing) | **EXPERIMENTAL — evidence-only** | 204-gap dataset across 10 videos is too small for decision influence (holdout category agreement 69.4 %; removable-gate agreement 88.9 % but precision 0.0 / recall 0.0; single engine-removed holdout gap scored P(removable)=0.014); promotion criteria documented; 22 additional telemetry records accumulated since training (still far below 10K–50K target) |
| Scene detection (PySceneDetect, cached, Rust engine) | **PRODUCTION (flag-gated)** | deterministic, cached, measured, fallback-proven |
| Scene-aware camera advisory (static-shot reframe scaling) | **EXPERIMENTAL** | measured churn reduction with identical compositions; small dataset of 2 sources; flag-gated |
| AutoFlip-style motion-aware transitions | **PRODUCTION (flag-gated, provably neutral)** | byte-identical when untriggered; bounded speed when triggered |
| Crop scorer (deterministic advisory ranking) | **EXPERIMENTAL (opt-in, default OFF)** | margin-gated; production-binding contract enforced by suite |
| Learned crop aesthetic head | **DATA-BLOCKED / NOT SUITABLE YET** | no rated-crop dataset exists; no annotation pipeline built; CLIP/u2netp postponed |
| T7 prosody pipeline | **PRODUCTION (as tooling)** — model **DATA-BLOCKED** | 320-record feature pipeline verified; chunker untouched; no human annotations collected |
| Pacing telemetry data pipeline | **PRODUCTION (opt-in)** | accumulates weak labels for the documented promotion path; 22 records collected since training |

## 24. Known Limitations

1. **Learned pacing cannot yet change decisions** — the honest holdout evidence (single engine-removed gap scored 0.014) means enabling model influence today would reduce quality. The telemetry pipeline is the promotion path; only 22 additional records have accumulated since training (total effective dataset still ~204 gaps), far below the 10K–50K gap range suggested by Atria for stable multi-class pause typing.

2. **Scene detection costs ~40 s per fresh source** (whole-video pass) — cached per source; acceptable for offline rendering.

3. **Static-scene advisory is calibrated conservatively** (1.25× on two sources); wider validation could justify retuning.

4. **Crop scorer's margin (1.0) rarely fires** — by design it defers to the deterministic solve; real value arrives with the learned head (data-blocked).

5. **dramatic_pause class absent** from the training corpus (no such gap in 204) — the model cannot type it yet.

6. Scene-boundary merging can make transitions at genuine boundaries snappier (observed once on messi) — intended, containment intact.

7. **No rated-crop dataset exists** — the crop aesthetic head cannot be trained; no annotation pipeline has been built.

8. **No T7 prosodic boundary labels exist** — the feature pipeline produces 320 annotation-ready records per 90 s interview, but zero human labels have been collected; the learned model remains data-blocked.

## 25. Phase 4 Readiness

The deterministic core plus Speaker Intelligence (Phase 2) and the Phase 3 advisory layers are integrated, cached, flag-gated, and regression-clean. Phase 4 (local VLM candidate scoring, Qwen3-VL/InternVL, DeepFilterNet2, PANNs reaction metadata, candidate redundancy, render QA expansion) has **NOT** been started, per the stop condition.

## Completion Checklist

- [x] Smart Pacing learned pipeline implemented AND its decision influence formally documented as data-blocked
- [x] Smart Pacing uses Silero VAD + librosa-family features (no openSMILE)
- [x] Learned pause output feeds the deterministic pipeline as evidence; safety rules untouched
- [x] Scene detection integrated at source level
- [x] Scene detection cached (version-keyed, corruption-safe)
- [x] Per-scene camera intelligence integrated without breaking framing invariants
- [x] AutoFlip-style smoothing integrated safely (provably neutral when untriggered)
- [x] Crop scorer implemented (deterministic advisory) and its learned head formally documented as data-blocked
- [x] Crop scorer is advisory (margin-gated, opt-in)
- [x] Hard containment remains authoritative
- [x] Two-person Adaptive shared composition remains authoritative
- [x] T7 prosody pipeline implemented; T7 model formally documented as data-blocked
- [x] T7 guardrails remain authoritative (chunker untouched)
- [x] No karaoke regression (Caption QA 100/100)
- [x] Feature flags work (unit + runtime verified)
- [x] Failure fallbacks work (15+ failure cases covered)
- [x] Cache invalidation works
- [x] License audit completed
- [x] Full Rust suite passes (341/341)
- [x] Full Python suite passes (all suites + 17 new Phase 3 tests)
- [x] npm build passes
- [x] Phase 1 / 1.1 / 2 regression passes
- [x] Real-media validation completed (A/B on two sources)
- [x] Performance measured
- [x] Invariant audit passes
- [x] Documentation updated
