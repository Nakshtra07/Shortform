# AutoShorts 9.0 — Smart-Framing Regression Diagnosis & Fix: Final Report

Evidence clips: `clip-13_flat.mp4` (project `beade84f` clip-02) and `clip-12_flat.mp4`
(project `beade84f` clip-01). Source master:
`C:/Users/naksh/Downloads/AutoShorts_ngvOyccUzzY.mp4` (3840×2160, 25 fps, 6779.854 s).
**Timing domains:** source window boundaries in project metadata are milliseconds
(289880–359605 ms, 3212335–3287070 ms); the engine's internal trajectory is
source-domain seconds; probe times are seconds relative to the clip start. The
final MP4 is ground truth: every fix claim below is backed by a re-render from the
source master plus a dense face probe over the rendered output, not by unit tests
alone.

---

## 1. Files created

**Regression coverage**
- `autoshorts/src-tauri/scripts/test_framing_regression_suite.py` — framing/temporal
  regression suite, 16 checks (IDs `FR-*`), report written to
  `autoshorts/tmp/qa_suite/reports/framing/framing_regression_report.json`
  (env override `AUTOSHORTS_FRAMING_REPORT`).

**Forensic harnesses (all re-usable, all under `scratch/forensics/`)**
- `capture_kf.py` — captures raw / optimized / validated keyframes + per-sample
  validation detail for a window. Usage: `python capture_kf.py <A|B> <outdir>`.
- `capture_diag.py` — engine diagnostics dump (`repro/diag_A.json`, `diag_B.json`).
- `clip_face_forensics.py` — dense YuNet probe of any MP4; `probe(path, times, tag)`
  returns per-time face boxes (output-domain px), confidence, containment, margin.
- `eval_ffmpeg_expr.py` — numerical evaluator for the piecewise `if(lt(t,…))`
  crop expressions; `ev(expr, t)` returns crop-x at time t.
- `render_and_probe.py` — renders a plan JSON from the source master and probes it.
- `probe_fixed.py` / `probe3.py` — defect-A/B probe scripts over the fixed renders.
- `src_a_probe.py`, `src_zoom_probe.py` — full-res (3840px) source-face probes.

**Evidence artifacts (`scratch/forensics/repro/`)**
- `runA_clip13_289880_359605.stdout.json`, `runB_clip12_3212335_3287070.stdout.json`
  — **original** (pre-fix) engine plans for the two evidence windows.
- `kf2/kf_A.json`, `kf2/kf_B.json` — keyframe capture under fixes 1–3 (isolated the
  residual defect-A bug).
- `kf3/kf_A.json` — keyframe capture under all four fixes (this report's "after").
- `fix3A_render.mp4`, `fix3B_render.mp4` — re-renders from the fixed plans.
- `probe3.py` output, `probe_fixed_out.txt`, `capA/capB/fixA/fix2A/fix2B` logs.

**QA**
- `tmp/qa_suite/caption_qa_final.log`, `tmp/qa_suite/caption_qa_rerun.log`,
  `validation_report_caption_qa.md`.

## 2. Files modified

Only one production file was modified: **`autoshorts/src-tauri/scripts/speaker_tracker.py`**
(5170 lines). Four surgical changes:

| # | Fix | Lines | What changed |
|---|-----|-------|--------------|
| 1 | Camera-lock hard-transition flag | 4910–4995 | `hard_transition` computed per sub-interval (shot cut, corridor exit/entry, reframe) and OR-ed into the keyframe tuple: `raw_keyframes.append((sub_abs, clamped_x, is_cut_event or hard_transition))` (line 4995). Propagated through `is_cut` at 5007/5022. |
| 2 | Validator recovery target | 2546 | `primary_face = nearest_genuine_face(genuine, kf_x, eff_crop_w)` — the validator now recovers the genuine face **nearest the current crop center**, not the most prominent detection; `kf_t, kf_x, is_cut = corrected[bki]` (2544) is read before selection so the reference is the active crop. |
| 3 | Degenerate face-box sanitizer | 2369, 2549 | `sanitize_face_bbox_for_containment(...)` applied at the classify site (2369) and in the validator (2549), **before** scale estimation. Caps face width at `crop_w_baseline*(1-2*SAFETY_MARGIN_RATIO)/1.24` = 823 source px, center preserved. |
| 4 | Keyframe split (defect-A root cause) | 2508–2511, 2558–2587, 2604 | See §6. On a mid-interval containment violation the original keyframe is preserved and a new hard-cut keyframe is **inserted** at the first violating sample, instead of rewriting the keyframe in place. New diag counter `n_insertions` (2604). |

Also modified (test-only): `autoshorts/src-tauri/scripts/test_framing_regression_suite.py`
— updated `test_recovery_follows_camera_center_not_prominence` (line 126) to the
new split semantics and added `test_mid_interval_subject_change_splits_not_rewrites`
(line 155).

`build_ffmpeg_expr` (2598+) and `build_balanced_binary_tree` were read, proven
correct, and **left untouched**.

## 3. Framing architecture discovered

Pipeline invoked from Rust at `media.rs:938` (`detect_speaker_crop_params`):
`python speaker_tracker.py <src> <start_ms> <end_ms> <crop_w> <max_x> <default_x>
[transcript.json]`, with crop_w=1215, max_x=2625, default_x=1312.

`run()` (speaker_tracker.py:5041–5088):
1. **Camera-lock operator** (4788–4995) — walks the window, classifies each
   sub-interval (0.50 s granularity) with `classify_and_frame_subject`, picks a
   subject, solves a containment crop, and emits one keyframe per sub-interval.
2. **Dedup** (5046) — drops a keyframe when |Δx| < 3 and it is not a cut.
3. **`optimize_camera_trajectory`** (2471) — segments on cuts; Savitzky-Golay
   smoothing only inside segments of length ≥ 4 with ≥ 4 samples. (A no-op for
   defect-A's window after fix 1, where every keyframe is a cut.)
4. **`validate_and_correct_trajectory`** (2503) — re-checks containment on
   `all_samples` (full-res faces, sampled every 0.12 s) and corrects.
5. **`build_ffmpeg_expr`** (2598) — emits the piecewise FFmpeg expression:
   value `x_i` applies over `[t_i, t_{i+1})`; `is_cut` or |Δx| < 1 → instant step;
   otherwise a cubic-Hermite smoothstep over the last `TRANSITION_DUR` of the
   interval; assembled into a balanced binary tree of `if(lt(t,…))` terms.

Production render filter (`media.rs:1326/1385`):
`crop=w=1214:h=2160:x='<expr>':y='<expr>',scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int`.

Key containment helpers: `nearest_genuine_face` (495) — sort key
`(-abs(center-crop_cx), conf*frontality*area)`; `sanitize_face_bbox_for_containment`
(515); `compute_subject_safe_box` (547) — head 1.24×fw, shoulders 2.2×fw;
`solve_containment_crop_x` — clamps the crop so the safe box fits with 8% margins;
`classify_shot_and_estimate_scale` (575).

## 4. Temporal architecture relevant to framing

- The trajectory lives entirely in **source-domain seconds**; the Rust layer
  converts `start_ms`/`end_ms` once at shell-out.
- Keyframe `i` governs `[t_i, t_{i+1})`. This invariant is what the pre-fix
  validator violated.
- The engine emits **two timelines**: the framing trajectory and the Smart-Pacing
  removal plan. `FR-PACING-MAP` proves the source→output time map is monotonic, so
  pacing removals never desync framing keyframes.
- DualFrame shot resets (`DualFrame shot_reset t=32.88 / 60.24` in the window-A
  run log) force a fresh tracking pass; `FR-SHOT-RESET` proves scale state resets
  at cuts.

---

## 5. Clip 02 diagnosis (`clip-13_flat.mp4`, source 289.880–359.605 s)

**Symptom (ground truth = final MP4):** for roughly 11 of the clip's 69.7 s the
crop points at empty background while the active speaker is speaking.

**Diagnosis.** A dense YuNet probe of the re-render under fixes 1–3 but **before
fix 4** (`repro/fixA_render.mp4`) found **no face in the crop** continuously over
relative t = 1.5–7.25 s and t = 9.0–12.25 s, plus isolated samples at t = 30 and 69.
(The fully pre-fix original clip shows the same empty-crop intervals.) A full-res
source probe (`src_a_probe.py`) confirmed the speaker's face is present at source
x ≈ 2300 throughout those times — the crop was simply aimed elsewhere.

Tracing the pipeline (`repro/kf2/kf_A.json`): the raw camera-lock keyframes are
correct — rel 1.5 → x=1944.15 (crop [1944, 3158], contains the speaker at 2300)
and rel 9 → x=1021.04 (crop contains the faces actually present at rel 9–12.5).
`optimized_kf` was byte-identical to `raw_kf` (fix 1 had already removed the
Savitzky-Golay smearing; all 12 keyframes carry `is_cut=True`). The corruption
appeared **exactly at `validate_and_correct_trajectory`**: `validated_kf` overwrote
rel 1.5: 1944.15 → **808.18**, and rel 9: 1021.04 → **1971.77**.

## 6. Clip 02 root cause

`validate_and_correct_trajectory` rewrote keyframes **backwards in time**.

Per-sample trace (`kf2/kf_A.json` `validation_detail`): at rel 7.32 the speaker
(center 2518) is still inside crop [1944, 3158]. At **rel 7.56–8.88 the only
genuine face detected jumps to center 1339.9 → 1552.8** (conf 0.89–0.92) — a
second person (S2) enters while the speaker at ≈2300 is momentarily undetected.
`is_any_face_contained` becomes false, and the old code executed:

```python
corrected[bki] = (kf_t, float(np.clip(target_crop_x, 0.0, eff_max_x)), is_cut)
```

Since `bki` is the keyframe governing the interval containing the violating sample,
this re-centers the keyframe **for its whole interval**, including the part
before the violation. rel 1.5–7.5 was correctly framed and got broken. The mirror
case follows at rel ≈12.5: the face reappears at 2300 (outside crop [1021, 2177]),
the validator re-centers the rel-9 keyframe to 1971.77, and rel 9–12.5 — where the
visible faces were at 1300–1500 — gets broken.

A second, independent bug compounds it: `is_any_face_contained` tests **any**
genuine face and the recovery targets `nearest_genuine_face`, so the keyframe is
stolen by whichever face appears — a listener entering the frame can hijack a
keyframe built for the speaker. (The value 808.18 equals raw keyframe 4's value
only because both are the same S2 face — not cross-keyframe value bleed.)

**Source location:** `speaker_tracker.py:validate_and_correct_trajectory`, the
in-place `corrected[bki] = ...` assignment (pre-fix line 2556).

## 7. Clip 02 surgical fix

Fix 4 (speaker_tracker.py:2508–2587) replaces the in-place rewrite with
**keyframe splitting**:

- On a violation at sample `t_v` for governing keyframe `bki`:
  - **Case 1** — `|abs_t - kf_t| ≤ 0.001`: the keyframe is wrong from its own
    start → keep the old in-place correction (2559–2566).
  - **Case 2** — otherwise: leave `corrected[bki]` **intact** and **insert** a new
    keyframe `(t_first_violation, clipped_target_x, is_cut=True)` at the sorted
    position (2575–2585). Semantics: the original crop was valid up to `t_v`; from
    `t_v` a different crop is needed. Because every keyframe is `is_cut` and
    `build_ffmpeg_expr` treats `is_cut` as a hard interval boundary, the inserted
    keyframe produces an instant step — no smoothing artifact.
- **Persistence guard** (2575): the insertion fires only after ≥2 *consecutive*
  violating samples, so a single-frame detection dropout no longer churns the
  trajectory (this is exactly the rel-9 case, where the very next sample is
  contained again).
- **De-duplication guard** (2577): skip insertion if the insertion time is within
  0.25 s of any existing keyframe.
- `corrected` grows during the sample loop; the per-sample `bki` scan re-scans the
  list, so inserted keyframes are validated against later samples automatically.

**Why this fixes the root cause:** the correction now starts at the earliest time
the crop actually became wrong, instead of being back-dated to the keyframe's own
time. Nothing that was ever correct is rewritten.

## 8. Clip 02 regression test

`test_framing_regression_suite.py:155`
`test_mid_interval_subject_change_splits_not_rewrites` — a synthetic all-cut
keyframe `(101.5, 1944.0, True)` that correctly frames a speaker at 2300; the
sample stream contains the speaker for rel 1.5–7.0 and only a foreign face at
1400 for rel 7.5–9.0. Asserts:

1. the original keyframe `(101.5, 1944.0, True)` is present **unchanged** in the
   output (proves no backward-in-time corruption);
2. exactly one new hard-cut keyframe is inserted within 0.26 s of rel 7.5
   (proves the split lands at the subject change, not at the keyframe start);
3. the inserted crop contains the face actually visible then (proves the
   recovery targets the right subject);
4. `info["n_insertions"] > 0`.

Check ID `FR-SPLIT-NOT-REWRITE`, status **PASS**, timing domain source, evidence
`validate_and_correct_trajectory split`.

## 9. Clip 01 diagnosis (`clip-12_flat.mp4`, source 3212.335–3287.070 s)

**Symptom (ground truth = final MP4):** the source footage itself contains a
camera zoom; in the generated clip the engine's own crop **over-compounds** that
zoom, producing a face-clipping crop and continual jitter over t ≈ 30–48 s.
Probe of the original clip: at t = 30.5 the top-confidence face (0.71) is cut
(−42 px); at t = 47.5 the **only** detected face (0.60) is cut (−52 px).

**Diagnosis.** Evaluating the original plan (`runB_clip12_*.stdout.json`) with
`eval_ffmpeg_expr.ev` over t = 30–47.5 s gives crop-x
`1937, 1977, 1992, 1864, 1948, 1940, 1922, 1934, 1951, 1903` — the crop never
settles, and the expression contains an explicit interpolated pan
`(1937+(1977-1937)*(((t-30.06)/0.3)*((t-30.06)/0.3)*(3-2*((t-30.06)/0.3))))`,
i.e. a smoothstep blend across t = 30.06–30.36. The engine was panning on top of
the source's own zoom. Three causes: (a) camera-lock keyframes without the cut
flag let `build_ffmpeg_expr` interpolate between re-centerings; (b) the validator
latched onto the most prominent detection rather than the tracked subject;
(c) degenerate oversized merged detector boxes (e.g. a 1210 source-px-wide group
box, vs. the 823 px containment cap) pushed `solve_containment_crop_x` toward
crops that clipped faces at the source-zoom peak.

## 10. Clip 01 root cause

- **Where:** camera-lock transition emission (4904–5022); validator
  `primary_face` selection (2546); missing sanitizer before containment solve.
- **Why:** without the cut flag, `build_ffmpeg_expr` smoothly pans between
  successive re-centerings (its `TRANSITION_DUR` smoothstep), adding synthetic
  camera motion on top of the source zoom; prominence-based recovery let a merged
  bystander group box drag the crop; and an uncapped 1210 px "face" box makes the
  containment solver chase a box wider than any real face, so the solved crop
  clips the actual face.

## 11. Clip 01 surgical fix

- **Fix 1** (4910–4995): the `hard_transition` flag marks every reframe/cut/corridor
  change as a hard step, so handoffs become instant cuts — no interpolated pan is
  ever synthesized. (`FR-HARD-EXPR` proves the emitted expression contains no
  `(3-2*u)` smoothstep across a flagged handoff.)
- **Fix 2** (2546): recovery follows `nearest_genuine_face` to the current crop
  center, keeping the crop anchored on the subject the camera was already tracking.
- **Fix 3** (2369, 2549): `sanitize_face_bbox_for_containment` caps degenerate
  boxes at 823 source px (center preserved) before the containment solve and before
  scale estimation. The bound `crop_w*(1-2*0.08)/1.24` derives from existing
  constants (`SAFETY_MARGIN_RATIO`, the 1.24 head multiplier), not an invented
  number. (`FR-ZOOM-CAP` proves capping; `FR-ZOOM-NOTOUCH` proves normal boxes are
  untouched; `FR-ZOOM-CORRIDOR` proves the corridor stays non-degenerate.)

## 12. Clip 01 regression test

- `FR-RECOVERY-TARGET` (test at line 126): crop deliberately placed beyond both
  speakers so no face is contained; the recovery must follow the speaker nearest
  the camera center (center 2527), not the larger, slightly more prominent speaker
  at 1322. **PASS** — the inserted keyframe's crop contains the tracked speaker
  and excludes the prominent one.
- `FR-HARD-EXPR` (line 100): a flagged handoff must emit an instant step with no
  smoothstep blend term. **PASS.**
- `FR-ZOOM-CAP` / `FR-ZOOM-NOTOUCH` / `FR-ZOOM-CORRIDOR` (TestFix3SourceZoomFaceSanitizer):
  the evidenced 1210 px merged box is capped to 823 px with its center preserved;
  normal boxes are byte-identical; the containment corridor stays legal. **PASS.**
- `FR-HANDOFF-SMOOTH`: handoff keyframes are preserved exactly through
  `optimize_camera_trajectory` (no smoothing into an empty pan). **PASS.**

## 13. Before/after evidence

All numbers are source-domain unless noted. Renders produced from the source
master with the production crop filter; probes are dense YuNet passes over the
rendered 1080×1920 output.

### Clip 02 (defect A) — keyframe level (`repro/kf2/kf_A.json` before, `repro/kf3/kf_A.json` after)

| rel t | raw camera-lock x | validated x BEFORE | validated x AFTER |
|-------|------------------|--------------------|-------------------|
| 1.50  | 1944.15 | **808.18** (corrupted) | **1944.15** (preserved) |
| 7.56  | —       | —                   | **808.18** (inserted, cut) |
| 9.00  | 1021.04 | **1971.77** (corrupted) | **1021.04** (preserved) |
| 12.60 | —       | —                   | **1971.77** (inserted, cut) |
| 30.00 | 808.18  | **668.97** (corrupted) | 1958.04 (in-place correction — wrong from its own start) |
| 30.36 | —       | —                   | 729.45 (inserted, cut) |
| 31.00 | 1924.06 | **729.16** (corrupted) | **1924.06** (preserved) |
| 65.24 | 815.78  | **1969.64** (corrupted) | **815.78** (preserved) |

Before the fix, 5 of 12 keyframes were corrupted by the validator; after the fix
all 12 raw values are preserved verbatim, and the validator adds 3 insertions
(7.56, 12.60, 30.36) plus 1 in-place correction (rel 30, wrong from its own start).

### Clip 02 — render probe (ground truth)

`fixA_render.mp4` (fixes 1–3, before fix 4) vs `fix3A_render.mp4` (all fixes), dense probe:

- **no-face times: [1.5–7.25, 9.0–12.25, 30, 69] → [7.5, 9.25, 12.5].**
  The three residual samples are single-frame detector dropouts, not empty-crop
  framing: at each of 7.5/9.25/12.5 the adjacent samples (±0.25 s) show exactly one
  face, contained, at the same position (e.g. t=7.25 box [324..670] conf 0.90
  contained; t=7.50 no face; t=7.75 box [328..675] conf 0.89 contained).
- **face-cut times: [40] → [40]** — but t=40 is a low-confidence (0.50–0.61)
  secondary box partially off the left edge of the frame; the primary speaker box
  (conf 0.70, [354..854] in 1080 px output) is contained with 179–237 px margin at
  every sampled time from 39.5 to 40.5.
- The crop now contains a genuine face at every sampled time where the detector
  fires. Continuous empty-crop framing is eliminated.

### Clip 01 (defect B) — trajectory level

Evaluating the emitted crop expression at the probe times (`eval_ffmpeg_expr.ev`):

| t | 30.0 | 30.5 | 31.0 | 33.0 | 35.0 | 38.0 | 40.0 | 43.0 | 45.0 | 47.5 |
|---|------|------|------|------|------|------|------|------|------|------|
| BEFORE x | 1937 | 1977 | 1992 | 1864 | 1948 | 1940 | 1922 | 1934 | 1951 | 1903 |
| AFTER x  | 1974 | 1677 | 1677 | 1677 | 1677 | 1677 | 1677 | 1677 | 1677 | 1677 |

The fixed expression is
`if(lt(t,30.36),if(lt(t,3.36),2178,1974),if(lt(t,49.08),1677,1970))` — three hard
steps, **zero smoothstep blend terms**, crop locked at 1677 for 18.7 s. The
original expression contained an explicit blend across t = 30.06–30.36 and never
settled (Δ up to 129 px of jitter).

Re-capturing window B under all four fixes (`repro/kf3/kf_B.json`) reproduces the
plan **byte-identically** (`raw_kf=4, validated_kf=4, corr=0`, same expression),
confirming the clip-01 "after" numbers above are the final-code behavior, not an
intermediate state.

### Clip 01 — render probe (ground truth)

`clip-12_flat.mp4` (before) vs `fix3B_render.mp4` (after):

- Before: t=47.5 the **only** detected face (conf 0.60) is cut −52 px; t=30.5 the
  top face (0.71) is cut −42 px.
- After: the highest-confidence face is fully contained at t = 35/38/40/43 with
  margins 259/107/57/279 px; no-face times: **[]**.

## 14. Caption QA regression result

`test_caption_qa_suite.py` re-run end-to-end (drives the real production code
paths — caption templates, caption intelligence, ASS, ffmpeg renders, SQLite
artifacts, OpenCV pixel verification):

```
[A] Template isolation:        43 PASS /  0 FAIL / 4 SKIP
[B] Real-clip forensic audit:  32 PASS /  0 FAIL / 3 SKIP
[C] Real renders r1–r4:        21, 21, 23, 20 PASS / 0 FAIL each
[D] Visual QA on rendered MP4: 5, 6, 6, 6 PASS / 0 FAIL each
[E] Regression guard:          11/11 protected files PASS
TOTAL: 212 checks, 0 failures   EXIT=0
```

Log: `tmp/qa_suite/caption_qa_final.log`; full report:
`validation_report_caption_qa.md`.

## 15. Framing / pacing / render test result

`test_framing_regression_suite.py`: **16 PASS / 0 FAIL / 0 SKIPPED / 0 NOT VERIFIED**
(report `autoshorts/tmp/qa_suite/reports/framing/framing_regression_report.json`).
Mapped to the required diagnostics A–J:

| Diag | Checks | Result |
|------|--------|--------|
| A. Active-speaker / framing-target agreement | FR-RECOVERY-TARGET, FR-SPLIT-NOT-REWRITE | PASS |
| B. Target validity | FR-NO-FALSE-CORRECTION, FR-TARGET-LOSS | PASS |
| C. Target handoff correctness | FR-HANDOFF-SMOOTH, FR-HARD-EXPR, FR-REACQUIRE | PASS |
| D. Crop containment | FR-CROP-CONTAIN | PASS (crop_x=1692, safe box inside [x, x+1215] with 8% margins) |
| E. Full-face containment | FR-FACE-CONTAIN (widths 100..820 all contained), FR-ZOOM-CAP, FR-ZOOM-CORRIDOR, FR-ZOOM-NOTOUCH | PASS |
| F. Crop trajectory after target changes | FR-SPLIT-NOT-REWRITE + kf3 capture (values above) | PASS |
| G. Source-camera zoom response | FR-ZOOM-CAP / FR-ZOOM-CORRIDOR / FR-ZOOM-NOTOUCH | PASS |
| H. Smart-Pacing temporal mapping | FR-PACING-MAP (source→output map monotonic) | PASS |
| I. Framing state across shot boundaries | FR-SHOT-RESET, FR-SEGMENT-TILING | PASS |
| J. Framing state across target loss/reacquisition | FR-TARGET-LOSS, FR-REACQUIRE | PASS |

Other suites (`.venv/Scripts/python.exe`): active-speaker **58/58**, smart-pacing
**55/0**, applied-features **30/0**, hook-closure **47/0**, audio-intelligence
**48/0**.

Render verification is the load-bearing evidence: §13's probes run on real MP4s
produced from the source master with the production crop filter.

## 16. Build result

- `cargo test` (from `autoshorts/src-tauri`): **261 passed, 0 failed, 1 ignored**
  (262 tests).
- `cargo check --bin autoshorts`: **exit 0** (7 pre-existing warnings, unchanged
  in count and content).
- `npm run build`: **exit 0** — `dist/index.html` 0.41 kB,
  `dist/assets/index-DAJtYOU4.css` 30.27 kB, `dist/assets/index-D8TVtgDJ.js`
  254.86 kB, built in 5.95 s.

## 17. Protected systems confirmed unchanged

The caption-QA regression guard (Part 11) byte-compares the protected pipeline
sources; **all 11 PASS**: `audio.rs`, `boundary.rs`, `captions.rs`,
`caption_intel.rs`, `media.rs`, `pacing.rs`, `transcription.rs`,
`transcript_normalizer.rs`, `youtube.rs`, `llm.rs`, `main.tsx`.

Production-code changes were confined to `speaker_tracker.py` (framing engine
only). The Rust shell-out contract (`media.rs:938`) — argv shape, crop parameters
(1215/2625/1312), and the emitted plan schema — is unchanged, so the Tauri
pipeline consumes the fixed engine without any Rust modification. The emitted
crop schema (`mode`/`x`/`y`/`w`/`h`/`segments`) is identical.

## 18. Remaining limitations

1. **No visual (human-eye) frame inspection was performed.** All frame-level
   verification is numeric: YuNet face-box coordinates, containment margins, and
   crop-expression evaluation on real rendered MP4s. No claim in this report rests
   on a subjective visual check.
2. **Clip-01 residual edge-clipped detections.** Over t ≈ 30.5–48.5 the probe still
   flags boxes that start up to 65 px (≈6% of the 1080 px width) outside the left
   edge — but only among 2–3 **simultaneous overlapping low-confidence boxes** (0.50–0.74)
   that are 400–610 px wide in the 1080 px output, versus ≈330 px for a genuine
   single face (cf. clip-02's boxes at 0.31–0.36 of frame width). These are merged
   duplicates of a bystander group spanning more than the 1214 px source crop.
   Containing the entire group is geometrically impossible without zooming out
   below the shot's intended scale; the engine deliberately contains its own
   detected subject box. The primary (highest-confidence) face is contained at
   every probed time in that window except t = 30.5/33.0/45.0/47.5, where the
   clipped box is a merged group, not the tracked speaker. This is reported as a
   defensible residual, not silently fixed by widening the crop.
3. **Single-frame detector dropouts** remain visible in probes as isolated
   "no face" samples (clip-02 t = 7.5/9.25/12.5). The persistence guard (≥2
   consecutive violations) prevents these from perturbing the trajectory; they are
   detector noise, not framing defects.
4. **No subject-identity tracking in the validator.** Fix 4 prevents retroactive
   corruption and requires violations to persist, but the validator still keys on
   "nearest genuine face to the crop center" rather than a tracked subject ID.
   Identity-level plumbing (e.g. carrying the BoT-SORT track id into
   `all_samples`) would be a larger change and is the natural follow-up if a
   future case shows two persistent, co-visible speakers confusing the split.
5. **Validation set = two windows.** The fixes are proven on the two evidenced
   windows plus synthetic unit tests. Full-portfolio re-rendering of every
   generated clip was out of scope.
6. **Timing domains.** Window times in project metadata are milliseconds; the
   engine works in source seconds; renders use `-ss` in seconds after `/1000`.
   Mixing these silently produces empty output (the source is only 6779.854 s, so
   e.g. `-ss 289882` yields nothing). Every number in this report was produced
   with the correct domain.
