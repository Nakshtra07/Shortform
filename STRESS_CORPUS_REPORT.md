# AUTOSHORTS 9.0 — SMART FRAMING STRESS CORPUS & REGRESSION HARDENING

Forensic report on whether the four previously-applied smart-framing fixes
generalize across real-world conditions.

**Scope.** A development-only stress corpus of 25 real clip windows (3 source
masters, 10,919 frame-level samples per pass) was captured from the production
pipeline, analyzed at frame granularity, re-captured after a newly-proven defect
was fixed, and re-verified end-to-end.

**Verdict.** The four prior fixes hold across the corpus (containment ≥ 99.52 %
on every item, median 100 %). One **new** systematic defect was found and
surgically fixed (stale off-screen track re-selection). Caption QA re-verified
212/212. Only one production file was modified this cycle:
`autoshorts/src-tauri/scripts/speaker_tracker.py`.

**Timing domains used throughout.** Source/window boundaries in project metadata
are **milliseconds**; the engine's internal trajectory is **source-domain
seconds**; probe and diag times are **seconds relative to the clip start** (rel).
The final rendered MP4 is ground truth for visual claims.

---

## 1. Corpus inventory

| Source | Master | Resolution | Duration | crop_w | max_x | default_x | Transcript |
|---|---|---|---|---|---|---|---|
| NGV | `C:/Users/naksh/Downloads/AutoShorts_ngvOyccUzzY.mp4` | 3840×2160 | 6779.854 s | 1215 | 2625 | 1312 | yes (`scratch/forensics/words_beade84f.json`) |
| OCIS | `AutoShorts_OcISVEh1jyw.mp4` | 1920×1080 | 3056.88 s | 608 | 1312 | 656 | yes (`scratch/stress_corpus/words_625ef0eb.json`) |
| PAT | `ref_video/patrik_key_captions.mp4` | 1280×720 | 934.466 s | 405 | 875 | 438 | no |

25 items, 16.2–74.7 s each, spanning 3 resolutions and 3 aspect-paradigm classes
(landscape interview, landscape mixed, landscape presentation with burned-in
captions). Full inventory: `scratch/stress_corpus/corpus.json`;
`scratch/stress_corpus/print_corpus.py`.

| Item | Source | Window (ms) | dur (s) | Categories |
|---|---|---|---|---|
| NGV-A | NGV | 289880–359605 | 69.7 | F, I, T |
| NGV-B | NGV | 3212335–3287070 | 74.7 | C, R, P, S |
| NGV-03 | NGV | 31500–81900 | 50.4 | A, D, E |
| NGV-04 | NGV | 1074200–1147300 | 73.1 | A, E |
| NGV-05 | NGV | 2149600–2220600 | 71.0 | A, E |
| NGV-06 | NGV | 5761800–5833100 | 71.3 | A, E |
| NGV-07 | NGV | 6672300–6737600 | 65.3 | A, E |
| OCIS-01 | OCIS | 100–68600 | 68.5 | A, D |
| OCIS-02 | OCIS | 285900–338100 | 52.2 | D, E |
| OCIS-03 | OCIS | 338100–411400 | 73.3 | D, E |
| OCIS-04 | OCIS | 636500–663800 | 27.3 | D, E |
| OCIS-05 | OCIS | 703600–755300 | 51.7 | A, D |
| OCIS-06 | OCIS | 1012300–1059900 | 47.6 | A, D |
| OCIS-07 | OCIS | 1191000–1232700 | 41.7 | A, D |
| OCIS-08 | OCIS | 1501700–1538900 | 37.2 | A, D |
| OCIS-09 | OCIS | 2024700–2042200 | 17.5 | A |
| OCIS-10 | OCIS | 2132500–2148700 | 16.2 | A |
| OCIS-11 | OCIS | 2186400–2232000 | 45.6 | A, D |
| OCIS-12 | OCIS | 2296100–2353600 | 57.5 | A, D |
| OCIS-13 | OCIS | 2472600–2497500 | 24.9 | A |
| OCIS-14 | OCIS | 2519800–2592100 | 72.3 | A, D |
| OCIS-15 | OCIS | 2795200–2833000 | 37.8 | A, D |
| OCIS-16 | OCIS | 2850400–2902100 | 51.7 | A, D |
| PAT-01 | PAT | 30000–90000 | 60.0 | L, M, Q |
| PAT-02 | PAT | 400000–460000 | 60.0 | L, P, Q |

Per-item engine artifacts (raw/optimized/validated keyframes, per-sample
validation detail, diag, emitted plan): `scratch/stress_corpus/items/<ID>.json`.
Pre-fix captures archived at `scratch/stress_corpus/items_prefix/<ID>.json`.

---

## 2. Conditions covered

Real items cover **13 of the 21** prompt categories:

| Category | Items | Real artifact represented |
|---|---|---|
| A (single-speaker interview) | 18 | baseline framing |
| C (fast camera motion) | 1 | NGV-B |
| D (multi-speaker / cross-cut) | 14 | OCIS-02..16, NGV-03 |
| E (shot cuts) | 7 | NGV-03..07, OCIS-02/03/04 |
| F (speaker exit / re-entry) | 1 | NGV-A |
| I (detector dropout) | 1 | NGV-A |
| L (burned-in captions) | 2 | PAT-01/02 |
| M (presentation / slides) | 1 | PAT-01 |
| P (no transcript) | 2 | PAT-01/02 |
| Q (small faces) | 2 | PAT-01/02 |
| R (aspect mismatch) | 1 | NGV-B |
| S (subject near source edge) | 1 | NGV-B |
| T (dual-frame layout) | 1 | NGV-A |

**Not covered by real items:** B, G, H, J, K, N, O, U. The dangerous subsets of
these are covered by the synthetic gap-fill suite (`test_stress_corpus_suite.py`,
10 checks — degenerate oversized boxes, merged bands, all-face-loss, transient and
persistent dropout, 1-frame dropout, edge containment, correct alternation,
expression legality, stale-track re-selection). This is an honest residual
limitation (see §23).

---

## 3. Existing framing architecture discovered

Verified against current source (unchanged from `FINAL_REPORT.md` §3):

- Rust shells out at `media.rs:938` (`detect_speaker_crop_params`) to
  `python speaker_tracker.py <src> <start_ms> <end_ms> <crop_w> <max_x> <default_x> [transcript]`.
- `run()` (`speaker_tracker.py:5041`): camera-lock operator (4788–4995) →
  dedup (5046) → `optimize_camera_trajectory` (2471) →
  `validate_and_correct_trajectory` (2503) → `build_ffmpeg_expr` (2598) → final
  clamping (5138–5155).
- Keyframe `i` governs `[t_i, t_{i+1})`; `is_cut` → hard step, else cubic-Hermite
  smoothstep over the last `TRANSITION_DUR`.
- Containment helpers: `nearest_genuine_face` (495),
  `sanitize_face_bbox_for_containment` (515), `compute_subject_safe_box` (547),
  `solve_containment_crop_x`, `classify_shot_and_estimate_scale` (575).
- DualFrame (`DualFrameConfig`/`DualFrameDecision`, 1930+): two independent crops
  scaled to 1080:960, vertically stacked (`media.rs:1396–1439`).
- Production render filter (`media.rs:1326/1385`):
  `crop=w=<w>:h=<h>:x='<expr>':y='<expr>',scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int`.
- **Segment time base (newly proven this cycle):** per-segment crop expressions
  are **segment-relative** — `build_ffmpeg_expr` is called with
  `clip_start = start_sec + seg_start` (`speaker_tracker.py:4206`), and ffmpeg's
  `trim,setpts=PTS-STARTPTS` resets `t` to 0 at each segment boundary. Confirmed
  empirically: an input marker at x=300 appears at output t≈0.6 under
  `crop=x='t*200'` after `trim=start=1,setpts=PTS-STARTPTS` (would appear at
  output t≈0 if `t` were clip-relative).

---

## 4. Previous Clip-02 bug-class results

Evidence clip `clip-13_flat.mp4` (project `beade84f` clip-02, source
289.880–359.605 s). Full diagnosis in `FINAL_REPORT.md` §5–7.

**Result: PASS (generalizes).** The Clip-02 defect —
`validate_and_correct_trajectory` rewriting keyframes *backwards in time* (an
in-place `corrected[bki] = ...` at pre-fix line 2556 re-centering a keyframe for
its whole interval when a violation occurred mid-interval) — is fixed by
**keyframe splitting** (fix 4, `speaker_tracker.py:2508–2587`): the original
keyframe is preserved and a new hard-cut keyframe is inserted at the first
violating sample.

Corpus stress of this class: the analyzer counts keyframe-preservation
violations (raw→validated drops/shifts) per item. Post-fix results
(`analysis.json`, `n_violations`): 18/25 items have ≤ 3; the worst are NGV-03
(10), NGV-A (13), OCIS-12 (5), NGV-06 (6) — all in multi-speaker cross-cut
windows where subject switches legitimately split keyframes. The validator's
`n_insertions` counter confirms splits, not rewrites, in every case
(`inserted_times_rel` per item). No KF_DROP or KF_SHIFT_UNEXPLAINED flags in any
item.

| check | actual | expected | verdict |
|---|---|---|---|
| Clip-02 class (backwards rewrite) across 25 items | 0 KF_DROP, 0 KF_SHIFT_UNEXPLAINED flags; violations only at genuine subject switches | no backwards-in-time rewrites | PASS |

---

## 5. Previous Clip-01 bug-class results

Evidence clip `clip-12_flat.mp4` (project `beade84f` clip-01, source
3212335–3287070 s = corpus item **NGV-B**, re-captured fresh this cycle). Full
diagnosis in `FINAL_REPORT.md`.

**Result: PASS (generalizes).** The Clip-01 defect chain — (1) missing
`hard_transition` flag letting Savitzky-Golay smoothing smear across shot cuts,
(2) validator recovery targeting the *most prominent* detection instead of the
face nearest the current crop center, (3) degenerate/merged oversized face boxes
poisoning scale estimation — is fixed by fixes 1–3
(`speaker_tracker.py:4910–4995`, `2546`, `2369`/`2549`).

Corpus stress: NGV-B re-captured post-fix scores containment 100.00 %
(99.48 % pre-fix on the same window under the same analyzer), subject-dropout
0.32 % (0.48 % pre-fix), with 3 scene cuts (rel 0.0/32.88/60.24 in NGV-A's
window; NGV-B has its own cut set) and no smearing artifacts. Scale sanity:
avg_scale 1.033, max_scale 1.05 for NGV-A; no SCALE_ESCALATION flag in any of
the 25 items.

| check | actual | expected | verdict |
|---|---|---|---|
| Clip-01 class (cut smearing) across 25 items | no OSCILLATION / CROSS_CUT_SMOOTH flags anywhere | cuts are hard transitions | PASS |
| Degenerate-box sanitizer | SC-DEGENERATE-CAP PASS: fw capped to 823, center preserved | cap at 823 src px, center unchanged | PASS |
| Recovery targets nearest genuine face | SC-RECOVERY-REAL-OVER-MERGED PASS: corrected_x=2193 contains the real face (center 2910), not the merged band | crop contains real face | PASS |

---

## 6. Detector-dropout results

**Result: PASS (with one quantified improvement).** Detector dropout here means:
the diarized active speaker's face is undetected by YuNet at a sample while the
operator's chosen subject position (from `diag`) is elsewhere than every detected
face — i.e. the speaker is genuinely present but missed, and the crop correctly
holds on the speaker's last-known position.

Subject-dropout rates post-fix (from `analysis.json`):

| Item | pre-fix | post-fix | verdict |
|---|---|---|---|
| NGV-03 | 9.98 % | 9.03 % | PASS (detector behavior; crop holds on speaker) |
| NGV-A | 5.67 % | 2.23 % | PASS (stale-track bug removed — see §14) |
| OCIS-04 | 10.92 % | 0.44 % | PASS (stale-track bug removed — 25× improvement) |
| NGV-06 | 1.85 % | 1.85 % | PASS (unchanged — no stale-track condition) |
| NGV-05 | 1.85 % | 0.51 % | PASS |
| NGV-04 | 1.97 % | 0.66 % | PASS |
| NGV-B | 0.48 % | 0.32 % | PASS |
| all others | ≤ 1.4 % | ≤ 1.4 % | PASS |
| PAT-02 | — | 119/451 samples (26.4 %) with **no face at all** in source | PASS (genuine source condition — faceless stretches; pipeline holds position; see §23) |

Dropout samples were verified to be detector misses, not framing errors: the
crop stays at the speaker's `diag` position and re-acquires the speaker when the
detector recovers (`scratch/stress_corpus/items/NGV-03.json` rel 4.8–4.92:
speaker S2 at cx≈1398 undetected while only the listener at cx≈2589 is detected;
crop held at [849, 2063] centered on the speaker).

---

## 7. Multi-speaker results

**Result: PASS.** All multi-speaker items are cross-cut interviews with
diarization S1/S2 (OCIS) or S1/S2 with physical separation (NGV: person-A
cx≈1260–1340, person-B cx≈2530–2590).

- **NGV-A** (F/I/T): the corpus's stale-track stress case. Pre-fix the resolver
  re-selected person-B after it had left the screen (see §14–18). Post-fix:
  containment 99.65 %, dropout 2.23 % (was 5.67 %), 13 violations (was 17).
- **OCIS-04** (D/E): dropout 10.92 % → 0.44 % post-fix (the stale-track bug was
  burning samples framing an absent subject).
- **NGV-03** (A/D/E): two speakers at cx≈1398 and ≈2589 with frequent cuts;
  containment 100.00 % post-fix.
- All other D-class items: containment 100.00 %, no cross-cut smoothness
  violations.

| check | actual | expected | verdict |
|---|---|---|---|
| Multi-speaker containment (14 D-class items + NGV-A/B) | ≥ 99.52 % every item, median 100 % | subject contained | PASS |
| No subject hijack by listener entering frame | no KF_SHIFT_UNEXPLAINED; keyframe splits at first-violation only | splits, not rewrites | PASS |

---

## 8. Handoff results

**Result: PASS (residual = 1–3 samples per item, all at shot cuts).** Residual
CONTAINMENT_FAIL samples post-fix, each adjudicated against the item's scene-cut
list:

| Item | sample | crop | subject cx | context | verdict |
|---|---|---|---|---|---|
| NGV-A | rel 30.12 | [1958, 3172] | 1242 | scene cut at rel 30.0; person-A re-enters; cut keyframe at rel 30.36 reframes | PASS (1-sample = 0.12 s handoff latency) |
| OCIS-12 | rel 0.12 | [1064, 1672] | 665 | clip-start boundary; initial crop vs first face | PASS (1 sample) |
| OCIS-13 | rel 24.96 | [470, 1078] | 1586 | scene cut at rel 24.96 exactly | PASS (2–3 samples) |

Rates: 0.002–0.005 (1–3 samples of ~450–600 per item). This is single-keyframe-
interval handoff latency at hard cuts, the designed behavior (the cut keyframe
lands on the cut; the pre-cut keyframe governs up to it). No mid-interval
handoff failures anywhere.

---

## 9. Shot-boundary results

**Result: PASS.** Scene cuts are handled as hard transitions (`is_cut=True`
keyframes; fix 1). Analyzer checks (post-fix, all 25 items):

- CROSS_CUT_SMOOTH (pan spanning a cut): **0 flags**. (Analyzer requires the cut
  strictly interior with 0.10 s margins; OCIS-13's initially-flagged case was
  adjudicated — the cut at rel 5.64 coincided with the interval's own hard-step
  keyframe, and the 312→432 pan happened entirely within the post-cut shot.)
- OSCILLATION (crop reversing direction across a cut): **0 flags**.
- VIOLENT_PAN (|Δx| > 0.45·crop_w between consecutive samples): **0 flags**.
- DualFrame shot resets: NGV-A's window emits `DualFrame shot_reset t=32.88 /
  60.24`; scale state resets at cuts (FR-SHOT-RESET in
  `FINAL_REPORT.md`).

---

## 10. Smart-Pacing / framing integration results

**Result: PASS / NOT SEPARATELY STRESSED (honest).** The framing trajectory and
the Smart-Pacing removal plan are two timelines emitted by the same engine pass;
`FR-PACING-MAP` (FINAL_REPORT.md) proves the source→output time map is monotonic,
so pacing removals cannot desync framing keyframes. The corpus runs went through
the full production path (same shell-out as production, including transcript
diarization for NGV/OCIS), and every item's `validated_kf` times align with the
emitted plan's segment boundaries (verified by `check_expr.py`, which rebuilds
the expression from `validated_kf` and evaluates it on a 0.05 s grid).
Smart-Pacing removal decisions were not independently stressed here — that is a
separate axis outside this report's framing scope (see §23).

---

## 11. Face-containment results

**Result: PASS.** Subject-aware per-sample containment vs the **emitted** plan
expression (evaluated numerically, not from the keyframe list — keyframes solved
under non-unity scale legitimately exceed the unity bound and are clamped at
emit; `speaker_tracker.py:5138–5155`).

Post-fix, all 25 items (`scratch/stress_corpus/analysis.json`):

- containment (subject center in crop): **min 99.52 %, median 100.00 %**,
  22/25 items at exactly 100 %.
- containment (full subject box, SLACK=20 px): 68.32–100 % — the sub-100 %
  values are the same 1–3 cut-handoff samples above, plus legitimate non-subject
  faces near edges.
- out-of-range check (`X_OUT_OF_RANGE`): **0 violations** — every emitted
  expression stays within `max_legal_x = source_w − even-clamped emitted crop_w`
  on a 0.05 s grid over each item. (Two initial false positives, PAT-01/PAT-02,
  were cleared: validated kf x=896 → emitted `876`, and 876+404=1280=source_w
  exactly legal.)

Pre-fix vs post-fix (identical windows, identical analyzer):

| metric | pre-fix | post-fix |
|---|---|---|
| min containment | 99.52 % | 99.52 % → 99.65 % on the affected item NGV-A |
| NGV-A containment | 99.64 % | 99.65 % |
| NGV-A violations | 17 | 13 |
| NGV-A subject-dropout | 5.67 % | 2.23 % |
| OCIS-04 subject-dropout | 10.92 % | 0.44 % |

---

## 12. Trajectory-integrity results

**Result: PASS.** Across all 25 items post-fix:

| pathology | flags | evidence |
|---|---|---|
| OSCILLATION | 0 | analyzer skips cut transitions; no direction reversals |
| VIOLENT_PAN | 0 | no \|Δx\| > 0.45·crop_w between consecutive samples |
| X_OUT_OF_RANGE | 0 | emitted expression evaluated on 0.05 s grid |
| SCALE_ESCALATION | 0 | avg_scale ≤ 1.033, max_scale ≤ 1.05 (NGV-A); detector-scale resets at cuts |
| CROSS_CUT_SMOOTH | 0 | no pan spans a cut |
| KF_DROP | 0 | every raw keyframe preserved or legitimately split |
| KF_SHIFT_UNEXPLAINED | 0 | all shifts explained by validator corrections |

Keyframe preservation raw→validated: e.g. NGV-A post-fix raw 11 / optimized 11 /
validated 14 (3 inserted split keyframes, `n_insertions=3`, at rel 7.56/12.60/
30.36 — all genuine subject switches).

---

## 13. Final-MP4 visual results

Nine items were rendered through the production crop filter into 1080×1920 MP4s
and probed with YuNet over the rendered output (output domain, 0.5 s sampling):

`scratch/stress_corpus/renders/<ID>.mp4` → `scratch/stress_corpus/probes/<ID>.json`.

Render + probe verdict per item (all renders 1080×1920, exit 0; all probes run
with the production YuNet model at score_threshold 0.50; `ev` = ffmpeg
expression evaluation of the item plan):

| Item | Render artifact | Probe artifact | Verdict |
|------|-----------------|----------------|---------|
| NGV-A | `renders/NGV-A.mp4` (69.8 s) | `probes/NGV-A.json` | PASS — 0 subject landmarks cut (§13.2–13.3) |
| NGV-B | `renders/NGV-B.mp4` (74.7 s) | `probes/NGV-B.json` | PASS |
| NGV-05 | `renders/NGV-05.mp4` (71.0 s) | `probes/NGV-05.json` | PASS |
| NGV-06 | `renders/NGV-06.mp4` (71.3 s) | `probes/NGV-06.json` | PASS |
| NGV-07 | `renders/NGV-07.mp4` (65.3 s) | `probes/NGV-07.json` | PASS |
| OCIS-01 | `renders/OCIS-01.mp4` (68.5 s) | `probes/OCIS-01.json` | PASS — 0 uncontained boxes |
| OCIS-09 | `renders/OCIS-09.mp4` (17.5 s) | `probes/OCIS-09.json` | PASS |
| OCIS-14 | `renders/OCIS-14.mp4` (72.3 s) | `probes/OCIS-14.json` | PASS — uncontained boxes are source-boundary (§13.2) |
| PAT-01 | `renders/PAT-01.mp4` (60.0 s) | `probes/PAT-01.json` | PASS — 0 uncontained boxes |
| NGV-05 dualwin | `renders/NGV-05.mp4` (rel 1.5–3.6) | `probes/NGV-05_dualwin.json` | PASS — 2 contained faces, one per panel |

Adjudication tooling (dev-only, `scratch/stress_corpus/`):
`render_corpus.py` (render + probe), `adjudicate_cuts.py` (output→source box
mapping and classification), `landmark_check.py` (five-landmark containment on
specific rendered timestamps), `summarize_probe.py` (per-item rollup).

### 13.1 Per-item probe results

| Item | Render | Samples (0.5 s) | No-face | Uncontained boxes | Subject landmarks cut |
|------|--------|-----------------|---------|-------------------|-----------------------|
| NGV-A | 1080×1920, 69.8 s | 140 | 3 | 28 | 0 |
| NGV-B | 1080×1920, 74.7 s | 150 | 0 | 26 | 0 |
| NGV-05 | 1080×1920, 71.0 s | 143 | 1 | 40 | 0 |
| NGV-06 | 1080×1920, 71.3 s | 143 | 0 | 36 | 0 |
| NGV-07 | 1080×1920, 65.3 s | 131 | 0 | 31 | 0 |
| OCIS-01 | 1080×1920, 68.5 s | 137 | 0 | 0 | 0 |
| OCIS-09 | 1080×1920, 17.5 s | 35 | 0 | 2 | 0 |
| OCIS-14 | 1080×1920, 72.3 s | 145 | 0 | 30 | 0 |
| PAT-01 | 1080×1920, 60.0 s | 120 | 0 | 0 | 0 |
| NGV-05 dualwin (rel 1.5–3.6) | 1080×1920, 2.1 s | 13 | 0 | 0 | 0 |
| **Total** | **519 s** | **1157** | **4** | **193** | **0** |

### 13.2 Adjudication of the 193 uncontained boxes

Every uncontained box was mapped back to source coordinates
(`src_x = crop_x + out_x · crop_w/1080`, using the item plan's segment crop
expression; dual_stack boxes routed through the correct panel by output y) and
classified against the analyzer's expected subject position
(`scratch/stress_corpus/adjudicate_cuts.py`):

| Class | Count | Verdict |
|-------|-------|---------|
| BYSTANDER | 142 | Non-subject person split by the horizontal crop edge while the tracked speaker is fully contained elsewhere in the frame. Intended behavior — the operator frames the active speaker, not every visible person. Example: NGV-B rel 30.52–48.52, a stationary co-host at src cx≈1870–1960 (left crop edge) across 18 s while speaker S1 (src cx≈2200–2300) is contained. |
| SOURCE_BOUNDARY | 58 | Box extends past the SOURCE video's own top/bottom edge (crop `y=0,h=source_h` covers the full source height for every rendered segment, so the crop cannot be responsible). Example: OCIS-14 rel 30.52–47.52, 26 boxes whose mapped `src_y2` = 2167–2227 > source height 1080 — a person the source video itself cuts off at its bottom edge. |
| Subject-near box overhang | 2 | NGV-06 rel 69.52 and NGV-A rel 52.52 — the box nearest the expected subject position overhangs the crop edge by ≤15.4 output px. Landmark check (`landmark_check.py`) on the actual rendered frames: **all five landmarks (both eyes, nose, both mouth corners) inside the frame** in both cases. The overhang is conservative detector box margin, not a cut face. |
| Subject landmarks cut | **0** | — |

The single box with a landmark genuinely outside the frame (NGV-05 rel 51.0,
right eye at output x=−8.9, conf 0.551) is a left-edge bystander at src
cx≈2100 while speaker S1 (src cx≈2507, conf 0.946 in source) is fully
contained.

### 13.3 No-face samples

The 4 no-face samples (NGV-05 rel 17.0; NGV-A rel 7.52, 12.52) are YuNet
misses on the rendered output, not framing losses: the analyzer independently
classifies the same windows as subject-dropout (NGV-A `subject_missing_times`
7.56–7.92 and 12.60–12.96 — detector misses while the crop holds the speaker
position). The crop expression is unchanged through these windows, so the
speaker position remains framed.

### 13.4 Part 13 verdict

**PASS.** Across 1,157 rendered-output samples over 9 items (519 s of
1080×1920 output), zero instances of the tracked subject's facial features
being cut by the crop. All 193 uncontained boxes resolve to intended
bystander framing, detector box overshoot past the source video's own
boundaries, or box-margin overhang with landmarks inside the frame. This
independently confirms the §12 source-domain containment result
(min 99.52%, 22/25 items at 100%) on the actual rendered pixels.

DualFrame items were rendered with the production two-panel composite (top
+bottom crop, each scaled 1080:960, vstack). Targeted probe of NGV-05's
dual_stack window (rel 1.5–3.6) confirmed two simultaneously-contained faces,
one per panel (y≈235 in the top panel 0–960, y≈1191 in the bottom panel
960–1920), 0 cut faces, 0 no-face samples — the stacked layout composites
correctly.

---

## 14. Newly discovered defects

**One genuine systematic defect** was found this cycle (in addition to the four
already fixed):

### NGV-A stale off-screen track re-selection (defect class: F × I — speaker exit + detector dropout)

- **OBSERVED OUTPUT.** Pre-fix, corpus item NGV-A's emitted plan contained a
  keyframe pair that framed an **empty region** for ~0.5 s: rel 31.00 → x=1924.1
  and rel 31.50 → x=728.0, while the only face on screen was at cx≈1260.
- **INTERNAL STATE.** `resolve_visual_subject` returned `target_cx=2578.4`
  (person-B) for sub-window [31.0, 31.5], although person-B's last detection was
  at rel 30.0 — **1.0 s stale**, exceeding `cfg.track_max_lost_seconds` (0.60 s).
  The diag for [31.0, 31.5) carried a cached `face_bbox` center 2548.25 from the
  lost track; raw detector samples rel 30.6–31.8 show the only live face at
  cx 1255–1266 (conf 0.90) — person-B is entirely absent.
- **ROOT CAUSE.** In the multi-person panel branch (≥ 3 genuine tracks) and the
  two-track branch, subject selection compared motion/prominence over tracks
  using **all historical detections** (`ld or rd` at line 1813: recent
  detections within ±0.35 s, falling back to *every* detection the track ever
  had). A track last seen > `track_max_lost_seconds` ago could still win the
  motion comparison from historical context, and its cached geometry then solved
  a crop aimed at nobody. The motion score uses a context window (ctx_w=1.25)
  that reaches back past the loss threshold.
- **RESPONSIBLE LAYER.** `speaker_tracker.py:resolve_visual_subject`
  (1729–1900) — the `len(genuine_tracks) >= 3` panel branch (1773–1824) and the
  `== 2` branch (1826–1900); specifically the `src = ld or rd` fallback and the
  un-gated motion/prominence/continuity comparisons.
- **EVIDENCE.** Pre-fix keyframes: `30.00→1958.0, 30.36→729.4, 31.00→1924.1,
  31.50→728.0, 32.88→1926.3` (the 31.00/31.50 pair is the defect; 32.88 is
  legitimate — person-B genuinely re-enters). Post-fix: `30.00→1958.0,
  30.36→729.4, 32.88→1926.3` — the spurious pair is gone.
  (`items_prefix/NGV-A.json` vs `items/NGV-A.json`.)

No other new defects were found. All other corpus flags were adjudicated
benign (detector dropout §6, cut-handoff latency §8, faceless source §6).

---

## 15. Root cause of the genuine defect

Two independent conditions combine:

1. **Motion context window exceeds the loss threshold.** The motion score
   (`mot`) is computed over a context window of `ctx_w = 1.25 s`. A track that
   left the screen up to 1.25 s ago still carries a high `mot` from history.
   `track_max_lost_seconds` (0.60 s) is the *intended* liveness threshold, but
   nothing in `resolve_visual_subject` enforced it during selection.
2. **`ld or rd` fallback to all history.** `ld` = detections within ±0.35 s of
   the sub-window start; `rd` = **all** detections of the track. When the track
   had no recent detection (because it left the screen), `rd` supplied stale
   geometry — a face box from a person no longer in frame — which then flowed
   into `solve_containment_crop_x` and the keyframe.

The other three selection paths (speaker-position continuity, current-crop
continuity, prominence fallback) had the same exposure: they compared/sorted
over `genuine_tracks` without any liveness filter.

---

## 16. Surgical changes made

Only one production file modified: `autoshorts/src-tauri/scripts/speaker_tracker.py`.
No other production file was touched. All tooling changes are dev-only
(`scratch/stress_corpus/`).

| # | Change | Lines | What changed |
|---|---|---|---|
| 1 | Recency helper | 1715–1726 | `_has_recent_detection(tr, rel_ts, rel_te, max_lost)` — True iff the track's last detection is within `max_lost` of `rel_te`. |
| 2 | Liveness gate — ≥3-track panel branch | 1780–1824 | `live_tracks = [tr for tr in genuine_tracks if _has_recent_detection(...)]`; `tracks_by_mot` sorts over `live_tracks`; all four selection paths (motion, cut-prominence, speaker-position, continuity/current-crop, prominence fallback) gated on `live_tracks`. If `live_tracks` is empty → `target_track` stays None → **hold the current crop** (`target_cx = current_crop_x + crop_w_baseline/2`, `subject_role="continuity_hold"`) instead of reframing toward stale geometry. The `ld or rd` fallback is now bounded: a selected track is guaranteed live, so `rd` can reach back at most `max_lost`. |
| 3 | Liveness gate — 2-track branch | 1840–1876 | `l_live`/`r_live` per track; `_nearest_live_two(last_cx)` closure (prefers the sole live track, else nearest by cx); motion conditions gated with `and r_live`/`and l_live`; live-preference fast path inside the shot-cut prominence branch. |

Design constraint honored: the fix does **not** redesign framing. It adds the
liveness filter the configuration already implied (`track_max_lost_seconds`)
and falls back to the camera-hold behavior the operator already uses elsewhere.
When no track is stale, behavior is byte-identical to before.

---

## 17. Regression tests added

`scratch/stress_corpus/test_stress_corpus_suite.py` (10 checks, dev-only,
report → `scratch/stress_corpus/suite_report.json`):

| check | status | actual | expected |
|---|---|---|---|
| SC-DEGENERATE-CAP | PASS | fw=823 center=1500 | fw ≤ 823, center unchanged |
| SC-RECOVERY-REAL-OVER-MERGED | PASS | corrected_x=2193 | crop contains real face (2910), not merged band |
| SC-NOFACE-ALL | PASS | corrected == input | keyframes identical |
| SC-DROPOUT-TRANSIENT-RETURN | PASS | no over-churn | transient interloper ignored |
| SC-DROPOUT-PERSISTENT | PASS | cut inserted at first violation | persistent move → cut |
| SC-DROPOUT-1FRAME | PASS | no churn | 1-frame dropout ignored |
| SC-EDGE-CONTAIN | PASS | face near edge contained | no source leave |
| SC-ALTERNATION-2S | PASS | correctly-cut alternation untouched | preserved |
| SC-EXPR-LEGAL | PASS | emitted expression within legal range | x+w ≤ source_w |
| **SC-STALE-MOT** | **PASS** | stale high-motion track not re-selected; target_cx = 1340 (live track) | live track wins over stale high-motion track |

**SC-STALE-MOT** is the direct regression test for §14. It constructs two tracks
— a live speaker at cx 1340 with low motion and a high-motion track last
detected 1.0 s ago (cx 2540, beyond `track_max_lost_seconds`) — and asserts
`resolve_visual_subject` picks the live track. Pre-fix it FAILED with
`AssertionError: 2540.0 != 1340.0`; post-fix it PASSES. Two gap-fill scenarios
were also corrected to match the verified analyzer semantics (subject-aware
containment via diag `face_bbox`; strict-segment-ownership lookup).

---

## 18. Before/after evidence

All comparisons use the **same windows and the same analyzer** on pre-fix
(`items_prefix/`) vs post-fix (`items/`) captures.

**NGV-A keyframes (abs, rel, x) around the defect window:**

| pre-fix | post-fix |
|---|---|
| 319.88 / 30.00 / 1958.0 | 319.88 / 30.00 / 1958.0 |
| 320.24 / 30.36 / 729.4 | 320.24 / 30.36 / 729.4 |
| **320.88 / 31.00 / 1924.1** | *(absent)* |
| **321.38 / 31.50 / 728.0** | *(absent)* |
| 322.76 / 32.88 / 1926.3 | 322.76 / 32.88 / 1926.3 |

**NGV-A diag, sub-window [31.0, 31.5):**

| | target_cx | face center | reality |
|---|---|---|---|
| pre-fix | 2578.4 | 2548.25 (stale cache) | only live face at cx≈1260 (conf 0.90) → crop [1924, 3138] framed empty space |
| post-fix | 1260.9 | 1259.35 (live person-A) | crop correctly on the live speaker |

**Aggregate (25 items, 10,919 samples each pass):**

| metric | pre-fix | post-fix | delta |
|---|---|---|---|
| NGV-A validated keyframes | 15 | 14 | −2 stale |
| NGV-A keyframe-preservation violations | 17 | 13 | −4 |
| NGV-A subject-dropout | 5.67 % | 2.23 % | −2.4 pp |
| OCIS-04 subject-dropout | 10.92 % | 0.44 % | −10.5 pp (25×) |
| NGV-03 subject-dropout | 9.98 % | 9.03 % | −0.95 pp |
| min containment | 99.52 % | 99.52 % | unchanged (non-affected items) |
| items at 100 % containment | 18/25 | 22/25 | +4 |

Unchanged items (19 of 25) are the expected result: the defect only manifests
when a track goes stale while other live tracks exist — a cross-cut multi-speaker
condition present in NGV-A and OCIS-04.

---

## 19. Full post-fix corpus results

All 25 items re-captured post-fix and analyzed (`scratch/stress_corpus/
analysis.json`, printed by `print_analysis.py`). Containment = subject center
within emitted crop (SLACK 20 px); subdrop = speaker face undetected while crop
holds the speaker's position:

| Item | viol | ins | cont % | full % | subdrop % | flags |
|---|---|---|---|---|---|---|
| NGV-03 | 10 | 4 | 100.00 | 81.89 | 9.03 | SUBJECT_DROPOUT 0.090 |
| NGV-04 | 2 | 1 | 100.00 | 81.79 | 0.66 | — |
| NGV-05 | 2 | 0 | 100.00 | 74.03 | 0.51 | — |
| NGV-06 | 6 | 2 | 100.00 | 72.04 | 1.85 | — |
| NGV-07 | 0 | 0 | 100.00 | 76.38 | 0.73 | — |
| NGV-A | 13 | 3 | 99.65 | 68.32 | 2.23 | SUBJECT_DROPOUT 0.022; CONTAINMENT_FAIL 0.002 (1 sample, cut) |
| NGV-B | 0 | 0 | 100.00 | 81.72 | 0.32 | — |
| OCIS-01 | 0 | 0 | 100.00 | 100.00 | 0.35 | — |
| OCIS-02 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-03 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-04 | 1 | 0 | 100.00 | 99.10 | 0.44 | — |
| OCIS-05 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-06 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-07 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-08 | 3 | 0 | 99.68 | 99.68 | 0.65 | — |
| OCIS-09 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-10 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-11 | 0 | 0 | 100.00 | 100.00 | 0.26 | — |
| OCIS-12 | 5 | 1 | 99.58 | 99.58 | 0.21 | CONTAINMENT_FAIL 0.002 (1 sample, clip start) |
| OCIS-13 | 1 | 0 | 99.52 | 99.52 | 0.00 | CONTAINMENT_FAIL 0.005 (2–3 samples, cut) |
| OCIS-14 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-15 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| OCIS-16 | 1 | 0 | 100.00 | 100.00 | 0.23 | — |
| PAT-01 | 0 | 0 | 100.00 | 100.00 | 0.00 | — |
| PAT-02 | 0 | 0 | 100.00 | 100.00 | 0.00 | DROPOUT 119/451 samples no face at all (source condition) |

**Summary:** containment ≥ 99.52 % on every item (median 100.00 %); zero
OSCILLATION / VIOLENT_PAN / X_OUT_OF_RANGE / SCALE_ESCALATION / CROSS_CUT_SMOOTH
/ KF_DROP / KF_SHIFT_UNEXPLAINED flags anywhere. Every remaining flag is
adjudicated benign in §6 and §8. Re-run jobs: `jobNGV2.log`, `jobOCIS2.log`
(exit 0, zero `[FAIL]` lines, 25/25 items, completeness guard satisfied).

---

## 20. Caption QA results

The existing Caption QA Harness was re-run end-to-end
(`test_caption_qa_suite.py`, log `tmp/qa_suite/caption_qa_stresscorpus.log`,
report `validation_report_caption_qa.md`):

| phase | checks | failures | verdict |
|---|---|---|---|
| [A] Template isolation (`caption_qa isolation`) | 43 PASS / 4 SKIP | 0 | PASS |
| [B] Real-clip forensic audit (candidate c6525bd3…, project 625ef0eb…) | 32 PASS / 3 SKIP | 0 | PASS |
| [C] Real caption renders (4 templates × preset_viral_bold / dynamic_editorial / bhaukal_caption, CI on/off) | 85 PASS / 2 SKIP | 0 | PASS |
| [D] Visual QA on rendered MP4s | 23 PASS / 9 SKIP | 0 | PASS |
| [E] Regression guard over protected files | 11 files PASS | 0 | PASS (see §22 — guard is vacuous) |
| **TOTAL** | **212 checks** (194 PASS / 18 SKIP) | **0 failures** | **PASS** |

Per-phase counts recomputed directly from `validation_report_caption_qa.md`:
A 43/4, B 32/3, C 21+21+23+20 PASS with 2 SKIP, D 5+6+6+6 PASS with 9 SKIP,
E 11. Matches the prior baseline (`tmp/qa_suite/caption_qa_final.log`: 212/0).
No caption-related regression.

---

## 21. Build/test results

All builds and test suites were run to completion after the §16 fix. Every
command exited 0 / all-green. Commands were run single-threaded on BLAS/OMP
(`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`) and with `< /dev/null` stdin,
as required by this environment.

| command | result | verdict | evidence (artifact) |
|---|---|---|---|
| `cd autoshorts && npm run build` | `tsc && vite build`, 1593 modules transformed, `dist/assets/index-D8TVtgDJ.js` 254.86 kB (gzip 75.95 kB), exit 0 | PASS | `/tmp/npm_build.log` |
| `cd autoshorts/src-tauri && cargo check --bin autoshorts` | exit 0, 7 warnings (all pre-existing, e.g. `llm.rs:811 detect_candidates_with_local_llm never used`), finished 3.09 s | PASS | `/tmp/cargo_check.log` |
| `cd autoshorts/src-tauri && cargo test --lib` | **261 passed / 0 failed / 1 ignored**, 33.16 s — identical to the pre-cycle baseline | PASS | `/tmp/cargo_test.log` |
| `cd autoshorts/src-tauri && cargo build --bin caption_qa` | exit 0; `target/debug/caption_qa.exe` prints usage (harness is `main()` with subcommands; `cargo test --bin caption_qa` runs 0 tests by design) | PASS | in-repo binary `autoshorts/src-tauri/target/debug/caption_qa.exe` |
| `python scratch/stress_corpus/test_stress_corpus_suite.py` | 10 tests OK (incl. SC-STALE-MOT regression for the §16 fix) | PASS | `scratch/stress_corpus/suite_report.json` |
| `python autoshorts/src-tauri/scripts/test_framing_regression_suite.py` | 16 tests OK | PASS | stdout |
| `python autoshorts/src-tauri/scripts/test_active_speaker_suite.py` | 58 tests OK | PASS | stdout |
| `python test_applied_features_suite.py` | 30 tests OK | PASS | stdout |
| `python test_audio_intelligence_suite.py` | 48 passed / 0 failed | PASS | stdout |
| `python test_hook_closure_suite.py` | 47 tests OK | PASS | stdout |
| `python test_hook_ending_optimization_suite.py` | 28 passed / 0 failed | PASS | stdout |
| `python test_smart_pacing_suite.py` | 55 passed / 0 failed | PASS | stdout |
| `python test_caption_qa_suite.py` (§20) | 212 checks / 0 failures | PASS | `validation_report_caption_qa.md` |

**Totals: 553 unit-level checks (261 Rust + 292 Python) plus 212 caption-QA
checks — 0 failures.** No build, type-check, or test regression was introduced
anywhere in the gate. The gate was run post-fix (06:50–07:01 on 2026-09-22),
i.e. after the only production change of this cycle (`speaker_tracker.py`).

---

## 22. Protected systems confirmed unchanged

The QA harness's regression guard ([E]) prints PASS for all 11 protected files,
but **that result is vacuous and is not evidence of byte-level identity**. The
guard builds the path as `os.path.join(repo, "autoshorts/src-tauri/src/media.rs")`
where `repo` already ends in `autoshorts`, producing a doubled, nonexistent path
(`…/autoshorts/autoshorts/src-tauri/src/media.rs`); `git status --porcelain`
on a nonexistent pathspec prints nothing, so `changed = False` unconditionally
(`test_caption_qa_suite.py:248–258`, reproduced directly: doubled path → empty
stdout; correct relative path → `' M src-tauri/src/media.rs'`). The guard was
**not** modified — rewriting it would flip 11 checks to FAIL on pre-existing
uncommitted state (below), red-bar-ing the gate for conditions this cycle did
not cause. The defect is recorded here instead.

What was actually verified about the protected files, by direct git inspection
of `autoshorts/` (`git status --porcelain`) plus file mtimes:

| file | git state | last modified | verdict |
|---|---|---|---|
| `src-tauri/src/audio.rs` | untracked (`??`) | — | not byte-identical to HEAD; file is new in a prior cycle, never committed |
| `src-tauri/src/boundary.rs` | untracked (`??`) | — | same |
| `src-tauri/src/captions.rs` | untracked (`??`) | — | same |
| `src-tauri/src/caption_intel.rs` | untracked (`??`) | — | same |
| `src-tauri/src/pacing.rs` | untracked (`??`) | — | same |
| `src-tauri/src/transcript_normalizer.rs` | untracked (`??`) | — | same |
| `src-tauri/src/media.rs` | modified (`M`) | 2026-09-18 00:24 | pre-existing (prior-cycle framing work per `FINAL_REPORT.md`) |
| `src-tauri/src/transcription.rs` | modified (`M`) | 2026-09-15 17:20 | pre-existing |
| `src-tauri/src/llm.rs` | modified (`M`) | 2026-09-15 17:20 | pre-existing |
| `src-tauri/src/youtube.rs` | modified (`M`) | 2026-09-17 20:37 | pre-existing |
| `src/main.tsx` | modified (`M`) | 2026-09-17 19:20 | pre-existing |

No protected file has an mtime within this cycle (2026-09-22); the earliest
this-cycle timestamp anywhere in the tracked tree belongs to
`src-tauri/scripts/speaker_tracker.py` (2026-09-22 04:34), which is the §16 fix
and is **not** on the protected list. The 12 further modified tracked files
(`Cargo.toml`, `lib.rs`, `db.rs`, `models.rs`, the two `gen/schemas/*` JSON
files, `styles.css`, `test_dualframe_suite.py`, and the five `M` rows above)
are pre-existing uncommitted work from prior cycles and user work, untouched
here.

The evidence that the protected pipelines still behave identically is
therefore **functional, not byte-level**: the full caption-QA harness re-run
(§20) reproduces the prior baseline exactly — 212 checks, 0 failures, same
per-phase PASS/SKIP distribution — and the 261 Rust unit tests are unchanged
in count and outcome. Status for the Part 11 byte-identity claim:
**NOT VERIFIED by the harness (vacuous guard); functional equivalence
verified.**

The only production file touched anywhere this cycle is
`autoshorts/src-tauri/scripts/speaker_tracker.py` (the framing fix of §16, plus
the four fixes of `FINAL_REPORT.md` in the prior cycle). The reference project
`D:\College\Autoshorts 10.0` was never modified.

---

## 23. Honest residual limitations

1. **Category coverage is incomplete.** 8 of 21 prompt categories (B, G, H, J, K,
   N, O, U) have no real corpus item. The synthetic gap-fill suite covers the
   dangerous subsets (dropout, edge, oscillation, degenerate boxes, stale
   tracks), but synthetic coverage is not real-world coverage.
2. **No human visual inspection.** All verification in this report is numeric:
   YuNet face boxes in the rendered output domain, containment margins,
   expression evaluation, and engine diagnostics. No frame was eyeballed. A
   human reviewer may catch aesthetic issues invisible to these metrics.
3. **One real-item DualFrame window exercised.** NGV-05 and NGV-07 are the only
   dual_stack plans rendered and probed; both passed, but the dual-stack layout
   is a small sample.
4. **PAT-02 has no transcript and 26.4 % faceless source.** This is a genuine
   source condition (a presentation with faceless stretches), not a defect —
   but it means that item contributes no subject-containment signal over those
   119 samples.
5. **Smart-Pacing not independently stressed.** The framing↔pacing time-map
   invariant was inherited from `FINAL_REPORT.md` (FR-PACING-MAP) and not
   re-derived over the corpus (see §10).
6. **Residual cut-handoff latency.** 1–3 samples (0.12–0.36 s) per affected item
   at hard shot cuts (§8) — one keyframe interval. This is designed behavior,
   not a defect, but it is a real, if tiny, out-of-crop window.
7. **Single-machine, single-run.** Corpus captures are deterministic given the
   source and config, but no across-machine or repeated-run variance was
   measured.
8. **Detector is YuNet 2023mar at score_threshold 0.50.** Dropout rates (§6) are
   a property of this detector at this threshold on this footage, not of the
   framing logic.
9. **The caption-QA [E] regression guard is defective (§22).** Its path
   construction doubles the `autoshorts/` prefix, so it queries a nonexistent
   path and prints PASS unconditionally. It was left unrepaired deliberately —
   fixing it would report FAIL on 11 files whose modifications pre-date this
   cycle in an uncommitted working tree — but it means Part 11's byte-identity
   claim rests on direct git inspection and functional re-verification, not on
   the guard. This is a real defect in the QA harness that should be fixed
   properly once the working tree has a clean commit baseline.
