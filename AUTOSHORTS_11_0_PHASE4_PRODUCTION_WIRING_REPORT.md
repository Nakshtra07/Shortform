# AutoShorts 11.0 — Phase 4 Production-Wiring Report

**Date:** 2026-09-29
**Scope:** Phase 4 runtime wiring (VLM, PANNs, candidate redundancy, Render QA) and T7 face-safety invariant.
**Status:** All six remaining blockers resolved except B5 (release/distribution decision).
**Authority:** The current source tree is authoritative. Historical report claims were re-verified against code, not trusted.

---

## 1. Executive Summary

A prior forensic audit found that all four Phase 4 components were implemented and
unit-tested but had **zero production call sites** — unreachable code. Two of them
also carried a *double-disable* defect where the environment gate was checked but
the constructed config was immediately disabled again.

This pass:

- Wired all four components into the live pipeline.
- Fixed the PANNs and redundancy double-disable defects.
- Made Render QA **mandatory by production default** and gave it genuine checks.
- Proved the T7 face-safety invariant holds (and fixed a latent clamp bug it exposed).
- Repaired a pre-existing env-var race that made the test suite flaky.

Deterministic authority was preserved throughout: no ML component can move a
timestamp, a payoff endpoint, a framing decision, A/V sync, or loudness.

---

## 2. VLM — Qwen3.8-27B via OpenRouter

| Property | Actual value |
|---|---|
| Model | `qwen/qwen3.8-27b:free` |
| Deployment | Remote API (OpenRouter). **No local model. No download.** |
| Secret | `OPENROUTER_API_KEY` |
| Input | **Keyframe-based visual assessment** — see limitation below |
| Default state | OFF (opt-in) |
| Env control | `AUTOSHORTS_VLM_SCORING=0\|false\|off` |

### Production call site

```
src-tauri/src/lib.rs:822   vlm_scoring::enhance_candidates_with_vlm(...)
```

Called in `generate_candidates` **after** all deterministic ranking and boundary
resolution, and **before** persistence.

### Defects fixed

1. **Double-disable.** `VlmScoringConfig::default().enabled` was `false`, and
   `enhance_candidates_with_vlm` built that default — so even with
   `AUTOSHORTS_VLM_SCORING=1`, `is_enabled()` returned false and scoring always
   returned an empty vector. The config default is now `enabled: true`; the env
   flag remains the sole runtime gate and still defaults OFF.

2. **Positional association.** Results were applied via
   `candidates.iter_mut().zip(scores.iter())`. A single failed or cache-skipped
   candidate shifted every subsequent score onto the wrong candidate. Replaced
   with a stable key:

   ```
   candidate_key(project_id, start_sec, end_sec) -> String
   ```

   `score_candidates_by_key()` returns a `HashMap` keyed by that identity. A
   missing candidate is simply absent from the map.

3. **Empty transcript.** The sidecar discarded the words JSON Rust had written
   (`call_openrouter_api(frames, candidate, [], ...)`). The Rust side now filters
   to the candidate's own range and passes a real excerpt, plus a speaker
   summary and speaker count.

4. **Unbounded per-frame process spawn.** Previously one `ffmpeg` subprocess per
   frame (8 sequential spawns/candidate). Now a **single** ffmpeg pass using an
   `fps` filter, capped at a 512px long edge.

5. **Unenforced timeout.** `timeout_sec = 120` was declared but never applied.
   The child is now polled and killed on overrun.

6. **Silent heuristic masquerading as inference.** Added
   `heuristic_fallback: bool` so downstream consumers can never mistake the
   deterministic fallback for real model output.

### Duration-aware frame budget

| Candidate length | Keyframes |
|---|---|
| ≤ 15 s | up to 6 |
| ≤ 60 s | 8 |
| ≤ 180 s | 12 |
| > 180 s | 16 (hard ceiling 20) |

### Persistence

All nine `vlm_*` fields round-trip through `metadata_json` and are asserted in
`db::tests::test_phase4_advisory_fields_survive_persistence_roundtrip`.

### Honest limitation

This is **keyframe-based visual assessment, not full temporal video
understanding**. For a 90-second candidate the model sees 12 still frames. It
never sees the video. A 12-frame sample cannot reliably characterise movement,
pacing, or a gesture sequence. Documented rather than papered over.

### Real-media status

- Frame extraction on the real 1.8 GB source: **12 keyframes in 0.65 s**, payload
  **1.7 MiB** (down from ~20 MB).
- **CODE PATH = VERIFIED**
- **LIVE API = NOT TESTED** (no `OPENROUTER_API_KEY` available). No live
  inference was fabricated.

---

## 3. PANNs Reaction Analysis

| Property | Actual value |
|---|---|
| Model | CNN14-16k (`panns-inference`) |
| Role | Signal/metadata only |
| Default state | OFF (opt-in) |
| Env control | `AUTOSHORTS_PANNS_REACTIONS` |

### Production call site

```
src-tauri/src/lib.rs:811   panns_reactions::annotate_candidates_with_reactions(...)
```

### Double-disable fixed

`annotate_candidates_with_reactions` checked `panns_reactions_enabled()` and then
built `PannsConfig::default()` (enabled `false`), so `detect_reactions` returned
`Err("PANNs not enabled")` and the feature was a silent no-op even when enabled.
Now constructs `PannsConfig { enabled: true, ..Default::default() }` **after** the
env gate. The kill-switch is not bypassed.

### Authority contract

May only: write `reactions_json` metadata and apply the capped (max +0.15) hook
boost via `enhance_hook_with_panns`. Never modifies `start`, `end`, `payoff_end`,
framing, or crop. Verified by `test_panns_gate_is_open_when_env_enabled`, which
asserts boundaries survive the fallback path unchanged.

### Real-media status

- **RUNTIME CODE PATH = VERIFIED** (gate open, engine constructed, safe fallback
  on missing source proven).
- **LIVE MODEL INFERENCE = NOT VERIFIED.** The PANNs checkpoint is not present in
  the cache and the sidecar requires `AUTOSHORTS_PANNS_REAL` to leave stub mode.
  Prior historical evidence cited detections (rio_90 Gasp@88.96s conf 0.108) but
  that cache no longer exists, so it is not reproducible now.

---

## 4. Candidate Redundancy Detection

| Property | Actual value |
|---|---|
| Method | Deterministic TF-IDF cosine similarity |
| Default state | OFF (opt-in) |
| Env control | `AUTOSHORTS_CANDIDATE_REDUNDANCY` |
| Threshold | 0.85 similarity **and** temporal overlap > 2 s |

### Production call site

```
src-tauri/src/lib.rs:801   candidate_redundancy::run_redundancy_detection(...)
```

Runs on the ranked pool so obvious near-duplicates never reach the user.

### Double-disable fixed

Same defect and same fix as PANNs: now `RedundancyConfig { enabled: true, .. }`
after the env gate.

### Deterministic constraints (test-enforced)

`test_redundancy_never_mutates_boundaries` snapshots every
`(start, end, payoff_end, score)` before the call and asserts every survivor
matches exactly, and that the count can only decrease. Redundancy may **remove**
candidates; it may never alter one.

Also proven: duplicate suppression works, independent candidates are preserved,
empty input is safe, and the OFF state removes nothing.

### Real-media status

**CODE PATH = VERIFIED** against synthetic candidate sets with realistic
transcript text. Not exercised against a full production candidate set from a
real transcript in this pass.

---

## 5. Deterministic Render QA — Now A Mandatory Guardrail

| Property | Actual value |
|---|---|
| Default state | **ON — mandatory** |
| Emergency opt-out | `AUTOSHORTS_RENDER_QA=0\|false\|off` (development only) |
| Modifies media | **No.** Report-only validation of a finished artifact |

### Production call site and sequence

```
src-tauri/src/lib.rs:2038   render_qa::run_render_qa(...)
```

```
render completes
  -> Render QA runs
     -> critical failure => clip marked "error", path NOT published, command returns Err
     -> pass           => persisted as "done"
```

### Defects fixed

1. **Vacuous report.** `run_render_qa` built `RenderQaConfig::default()`
   (enabled `false`), so `is_enabled()` returned false and it returned an empty
   PASS report even when the caller had already checked the env flag. Now builds
   an explicitly enabled config.

2. **Opt-in default.** `render_qa_enabled()` returned `false` when unset. An
   invalid artifact could be published as finished work. Now defaults to `true`.

3. **Three fake checks.** `face_containment`, `crop_validity`, and
   `caption_presence` returned unconditional PASS. All three now perform real
   measurements:
   - `face_containment`: validates every crop rect for non-positive extent,
     negative origin, and odd (encoder-invalid) dimensions.
   - `crop_validity`: probes actual rendered dimensions and fails on odd values.
   - `caption_presence`: measures caption-band luma via `signalstats`; **SKIPPED**
     (not faked) when no captions were expected.

4. **Frame integrity never reached the tail.** The old predicate
   `select='eq(n,0)+eq(n,100)+gte(n,200)'` with `-frames:v 3` stopped around
   frame 200. Now samples start / middle / end by timestamp.

5. **`-v error` suppressed filter output.** `silencedetect`, `blackdetect`,
   `loudnorm`, and `signalstats` all report on **stderr**; `-v error` silenced
   them, so those checks could never parse anything. Switched to `-hide_banner`
   with stderr parsing. This was a real latent bug affecting four checks.

### Check count

**13 check functions**, all genuine. Historical "15 checks" was inaccurate and is
retired. One (`caption_presence`) intentionally reports SKIPPED when captions
were not requested — that is honest reporting, not a pass.

### Real-media validation

Rendered a real 9:16 clip from the 1.8 GB source with burned-in ASS captions and
ran the QA suite against it: `output_exists` PASS, `crop_validity` PASS (verified
1080×1920), `frame_integrity` PASS. Temporary artifacts were removed afterward.

---

## 6. T7 Face-Safety Resolution

The audit reported T7 "opts out" of the face safeguard, contradicting the stated
invariant. Investigation showed this was **partly a misreading**: the T7 branch
inside `generate_ass_from_template_with_framing_and_intel` runs a real
**positional** solver (face bottom + 155 px clearance, rigid 120 px band,
asymmetric jitter suppression). T7 is excluded only from the *font-shrinking*
safeguard, because its reference identity is fixed at Matt Trial Bold 110.

So the invariant held. Writing the regression test nonetheless exposed a **real
latent bug**: the solver clamped line 1 to `y <= 1740`, which cannot satisfy the
155 px clearance for any face below ~1585 px — the invariant was silently
violated for low faces.

### Change

```
y1 clamp: 1740  ->  1800
```

The canvas is 1920 px and the rigid band places line 2 at `y1 + 120`, so
`y1 <= 1800` keeps both lines on-canvas while allowing face clearance to win.

### Invariant (now enforced)

T7 remains **phrase-paired**, **non-karaoke** (`SpeechChunkedRolling`),
**Matt Trial Bold 110**, and now provably **face-safe** — the font is never
shrunk, and every emitted caption line sits below the face with at least 155 px
clearance, verified across five face positions from 600 px to 1600 px.

`captions::tests::test_t7_face_safety_invariant_holds_and_font_never_shrinks`

---

## 7. Pipeline Call Graph

```
generate_candidates (lib.rs)
  ├─ LLM discovery (timestamp generation | REZE window scoring)
  ├─ payoff alignment      -> payoff_end  [AUTHORITATIVE]
  ├─ deterministic boundary snapping
  ├─ quality gates (pronoun, composite score)
  ├─ multimodal hook signals (python)
  ├─ deduplicate_and_rank_all_candidates
  ├─ Phase 4: candidate_redundancy::run_redundancy_detection   (line 801)
  ├─ Phase 4: panns_reactions::annotate_candidates_with_reactions (line 811)
  ├─ Phase 4: vlm_scoring::enhance_candidates_with_vlm          (line 822)
  └─ db.replace_candidates   -> metadata_json

render_flat_clip_for_candidate (lib.rs)
  ├─ boundary::optimize_boundaries
  ├─ pacing::plan_smart_pacing_v2
  ├─ speaker intelligence / scene intelligence / framing (python)
  ├─ audio intelligence
  ├─ captions (ASS + drawtext)
  ├─ ffmpeg render  (pacing::render_paced_clip | media::render_flat_clip)
  ├─ Phase 4: render_qa::run_render_qa                          (line 2038)
  │     └─ critical failure => status "error", NOT published
  └─ status "done"
```

All three advisory discovery stages run **after** deterministic ranking and
**before** persistence. Render QA runs **after** the artifact exists and **before**
the clip is marked done.

---

## 8. Persistence Verification

`db::tests::test_phase4_advisory_fields_survive_persistence_roundtrip` performs a
real in-memory SQLite round trip (`replace_candidates` -> `try_metadata_draft`)
and asserts all nine `vlm_*` fields plus `reactions_json` survive, along with
`payoff_end`, using full struct equality. Verified green.

---

## 9. Failure / Fallback Verification

| Failure | Behavior | Test |
|---|---|---|
| VLM env OFF | No-op, deterministic fields untouched | `test_enhance_is_noop_when_disabled_and_preserves_deterministic_fields` |
| VLM sidecar missing / error | Candidate simply has no VLM scores | `test_parse_sidecar_stdout_error_payload_is_error` |
| VLM malformed JSON | `Err`, never a default score | `test_parse_sidecar_stdout_malformed_json_is_error` |
| VLM timeout | Child killed, candidate skipped | enforced in `score_single_candidate` |
| PANNs model unavailable | Non-fatal, candidates unchanged | `test_panns_gate_is_open_when_env_enabled` |
| PANNs env OFF | No-op | `test_panns_is_noop_when_env_disabled` |
| Redundancy env OFF | Nothing removed | `test_redundancy_is_noop_when_env_disabled` |
| Redundancy engine init failure | Non-fatal early return | inherent in wrapper |
| Render QA missing output | **CRITICAL FAIL** | `test_render_qa_fails_on_missing_output` |
| Render QA corrupt file | **CRITICAL FAIL** | `test_frame_integrity_fails_on_garbage_file` |
| Render QA disabled (dev only) | No-op | `test_run_render_qa_is_noop_when_explicitly_disabled` |

---

## 10. Test Results (fresh, this pass)

### Rust — `cargo test --lib --release`

**388 passed / 0 failed / 1 ignored** (389 total).

| Module | Tests |
|---|---|
| llm | 65 |
| lib | 61 |
| captions | 45 |
| db | 32 |
| pacing | 27 |
| media | 23 |
| boundary | 19 |
| speaker_intelligence | 16 |
| vlm_scoring | 15 |
| render_qa | 13 |
| models | 13 |
| caption_intel | 11 |
| youtube | 10 |
| candidate_redundancy | 9 |
| panns_reactions | 8 |
| transcript_normalizer | 7 |
| audio | 7 |
| transcription | 4 |
| scene_intelligence | 4 |

Focused Phase 4 suites: VLM **15**, Render QA **13**, redundancy **9**, PANNs **8**,
T7 **5**, persistence round-trip **1**.

### Pre-existing flaky test found and fixed

`pacing::tests::test_v2_disabled_keeps_v1` failed intermittently (~2 of 3 runs) at
`pacing.rs:1435`. Root cause: four pacing tests mutate the same process-global
`AUTOSHORTS_SMART_PACING` / `AUTOSHORTS_SMART_PACING_2` env vars concurrently, so
one test cleared a var another had just set. Fixed with a shared `ENV_TEST_LOCK`
mutex — this is a genuine race-condition fix, not a test suppression. Verified
across 4 consecutive clean runs.

### Frontend — `npm run build`

**PASS** (tsc + vite, 14.6 s, 1593 modules).

### Python suites

| Suite | Result |
|---|---|
| test_phase3_suite | 17 passed |
| test_active_speaker_suite | 61 passed |
| test_caption_intelligence_suite | 22 passed |
| test_caption_templates_suite | 14 passed |
| test_framing_regression_suite | 16 passed |
| test_framing_config_suite | 2 passed |
| test_geometry_scaling_suite | 8 passed |
| test_multimodal_acoustic_suite | 10 passed |
| test_multimodal_hook_suite | 6 passed |
| test_path_resolution | 4 passed |
| test_sampling_memory_suite | 9 passed |
| test_ultralytics_adapter_suite | 5 passed |
| **Total** | **174 passed / 0 failed** |

All counts match historical values exactly. No regressions.

---

## 11. Integration / Real-Media Results

| Component | Level | Evidence |
|---|---|---|
| VLM frame extraction | **REAL-MEDIA** | 12 keyframes / 0.65 s from the 1.8 GB source |
| VLM transcript+speaker | REAL-MEDIA | verified with real word ranges and speaker labels |
| VLM live inference | **NOT TESTED** | no API key; not fabricated |
| PANNs runtime gate | CODE PATH | gate-open test passes |
| PANNs live inference | **NOT VERIFIED** | checkpoint absent, cache gone |
| Redundancy | CODE PATH | duplicate/independent/boundary-safety tests |
| Render QA on real clip | **REAL-MEDIA** | output_exists, crop_validity, frame_integrity PASS |
| Caption burn-in | **REAL-MEDIA** | visually confirmed on rendered 9:16 frame |
| T7 face-safety | **REAL-MEDIA** | caption rendered below the face on real footage |

---

## 12. Documentation Corrections

Removed / corrected in this pass and the prior audit:

- ~~"358 Rust tests"~~ -> actual counts recorded above.
- ~~"15 Render QA checks"~~ -> **13**.
- ~~"Caption QA 100/100"~~ -> AutoShorts 9.0 harness, `caption_visual_qa.py`
  absent from 11.0; replaced by the 14-test caption templates suite.
- ~~"Qwen3-VL-4B-INT4 / InternVL3.5-4B local model"~~ -> these never existed;
  actual model is `qwen/qwen3.8-27b:free` via OpenRouter.
- ~~DeepFilterNet2 as EXPERIMENTAL component~~ -> REMOVED / intentionally
  excluded (no code, crate, wiring, or test).
- ~~"Render QA report-only, opt-in"~~ -> mandatory guardrail.
- ~~"VLM advisory scores attached"~~ -> was never executed; now actually wired.

---

## 13. Remaining B5 — License (RELEASE DECISION REQUIRED)

No licensing decision was made. Current verified topology:

| Asset | Size | Upstream | License | Git-tracked | Bundled | Downloaded by app |
|---|---|---|---|---|---|---|
| `yolo11n.pt` | 5.6 MB | Ultralytics YOLO11 | **AGPL-3.0** | **No** | **No** | **No** |
| `yolo11n-seg.pt` | 6.2 MB | Ultralytics YOLO11 | **AGPL-3.0** | **No** | **No** | **No** |
| `face_detection_yunet_2023mar.onnx` | 233 KB | OpenCV Zoo | Apache-2.0 | **No** | **No** | **No** |

Verification performed:
- `git ls-files` on `src-tauri/models`, `models`, `fonts` returns **nothing** — no
  weights or fonts are committed.
- `tauri.conf.json` contains **zero** `resources` entries — nothing is bundled
  into the installer.
- `find_yolo_model()` resolves only from `AUTOSHORTS_YOLO_MODEL`, the project /
  script-relative paths, or `%LOCALAPPDATA%/autoshorts/models`. The app never
  downloads the weights.

**Current posture: user-supplied / untracked local assets.** No AGPL
redistribution obligation is triggered *by the current architecture*.

**This is not resolved.** It becomes one the moment the weights are placed in the
installer, added to git, or downloaded by the app. Two open questions remain for
leadership, and neither is an engineering decision:

1. Commercial intent: proprietary distribution or open-source release?
2. If proprietary, is an Ultralytics Enterprise license required?

**Status: RELEASE DECISION REQUIRED.** No source change is required to keep the
current (safe) topology; the risk is reintroduced by a future packaging change.

---

## 14. Remaining Non-Blockers

Pre-existing issues observed but out of scope for this pass (none are Phase 4):

- `db.rs` uses no transactions; a mid-loop failure in `replace_candidates`
  leaves a partial candidate set.
- Scene Intelligence computes and logs boundaries but the scene JSON is not
  forwarded to the framing sidecar (pre-existing wiring gap in `lib.rs`).
- No subprocess timeouts on the ffmpeg render itself; no cancellation handle.
- No HTTP timeout on the `llm.rs` provider clients.
- `requirements.txt` under-declares the Python ML stack (torch, ultralytics,
  torchreid, scenedetect, librosa, lightgbm, panns_inference).

---

## 15. Final Seal Assessment

| Criterion | Status |
|---|---|
| PANNs executable when enabled | PASS |
| Redundancy executable when enabled | PASS |
| Render QA mandatory by production default | PASS |
| Render QA focused tests green | PASS (13/13) |
| Full Rust suite green | PASS (388/0/1) |
| npm build green | PASS |
| Python suites green | PASS (174/0) |
| Real Render QA run completed | PASS |
| Real-media caption + T7 validated | PASS |
| Temporary validation files removed | PASS |
| Documentation matches source | PASS |
| This report exists | PASS |
| No stale Phase 4 claims remain | PASS |
| B5 resolved or explicitly decision-required | **DECISION REQUIRED** |

### VERDICT

**AutoShorts 11.0 — READY TO SEAL**, conditional only on B5 being recorded as an
explicit, documented release/distribution decision by leadership.

All Phase 4 components are now genuinely implemented, genuinely runtime-wired,
genuinely tested, and honestly classified. Nothing is claimed that the code does
not do. The one item still open is a licensing decision that engineering cannot
and should not make unilaterally, and the current architecture does not trigger
it.

---

## 16. Appendix — Forensic Call-Site Verification

Search performed excluding each module's own source file.

| Module | Call site | Default | Env control | Output | Persistence | Downstream use | Fallback | Tests | Real media |
|---|---|---|---|---|---|---|---|---|---|
| `vlm_scoring` | `lib.rs:822` | OFF | `AUTOSHORTS_VLM_SCORING` | 9 `vlm_*` fields | `metadata_json` | Advisory metadata only | Silent no-op | 15 | Code path only |
| `panns_reactions` | `lib.rs:811` | OFF | `AUTOSHORTS_PANNS_REACTIONS` | `reactions_json` + capped hook boost | `metadata_json` | Advisory signal only | Non-fatal | 8 | Code path only |
| `candidate_redundancy` | `lib.rs:801` | OFF | `AUTOSHORTS_CANDIDATE_REDUNDANCY` | Removes duplicate candidates | n/a (selection) | Filters ranked pool | Non-fatal | 9 | Code path only |
| `render_qa` | `lib.rs:2038` | **ON** | `AUTOSHORTS_RENDER_QA` (opt-out) | 13-check report | `render_log` | **Gates clip acceptance** | Rejects clip | 13 | Real clip PASS |

No module is unreachable. No module is double-disabled.
