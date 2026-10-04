# Final Report — Adaptive Framing: Close-Pair Geometry Calibration & Real-Footage Verification

> This report **supersedes** the earlier `FINAL_REPORT_CLOSEPAIR.md` (pre-calibration
> version, which documented the retired raw-span threshold and the "no real close pair
> fits" limitation). Every number below was produced by executing the **committed**
> production pipeline against real footage. No simulated geometry except where
> explicitly labelled (unit test).

**Calibration date:** 2026-09-23
**Implementation under test:** `autoshorts/src-tauri/scripts/speaker_tracker.py`
(committed Adaptive Framing path), `autoshorts/src-tauri/src/media.rs` render path.

---

## 0. Verdict summary

| # | Required check | Result |
|---|----------------|--------|
| 1 | Reference close pair classified as CLOSE and held together | **PASS** — 20/20 sub-intervals `two_shot_both_fit`, both heads in crop |
| 2 | Close-pair render: both people together, DualFrame OFF, no split | **PASS** — 2 faces in 12/12 sampled frames, 1 segment, `single` mode |
| 3 | Far pair classified as FAR and isolated to active speaker | **PASS** — 0/23 two-person sub-intervals fit; 1 face in 12/12 frames |
| 4 | Original 9:16 behaviour unchanged | **PASS** — DualFrame still engages; close pair isolates speaker as before |
| 5 | Criterion is general composition geometry, not video-specific | **PASS** — single ratio `head_span ≤ renderable_w`, no raw-pixel constant |
| 6 | Full regression (10 Python suites + Rust lib) | **PASS** — 586 Python checks + 272 Rust tests, 0 failures |
| 7 | Borderline close-pair cases on other footage | **LIMITATION** — see §19 |
| 8 | Seam-correlation as a split detector | **LIMITATION** — supportive only, see §19 |

Final success condition (§19 of the prompt): **MET.** Reference-like close pair → both
people together, DualFrame OFF. Clearly far pair → active speaker, DualFrame OFF.
Original 9:16 → existing behaviour unchanged.

---

## 1. Previous geometry/threshold rule

The Adaptive mode gated its joint two-shot on a **raw face-span threshold**:

- `TWO_SHOT_MAX_SPAN_ADAPTIVE = 0.92` (ratio of the 608 px crop baseline)
  → absolute span bound **559.4 px** at 1080p.
- The span measured was the **outer face-box edge span** (`face_span`): left face's
  left edge → right face's right edge.
- `crop_w_baseline = 608 px` (1080 × 9/16), `default_x = 656`, `max_x = 1312`.
- Original 9:16 bound `TWO_SHOT_MAX_SPAN = 0.65` → 395.2 px (unchanged, still in force
  for the Original mode).

Two structural problems: (a) the threshold is a **ratio of the crop width**, so it is
resolution-dependent in effect, and (b) it measures the wrong quantity — outer face-box
edges, not the composition that actually has to fit inside a renderable crop.

## 2. Exact reference video used

`My thoughts on my 2023_24 season and the truth about my future. Talk with my friend Rio (PART 1) - UR · Cristiano (1080p, h264).mp4`
— 1920×1080, 25 fps, 1482.36 s. The same video that supplied the reference screenshot
for the desired close-pair framing.

## 3. Exact reference frame/segment

**Window 452.6 s → 462.2 s (9.6 s)** — selected by a global two-track scan over the full
1482 s (177 shots, 38 real two-person windows;
`scratch/closepair_trackscan.py` → `scratch/closepair_trackscan.json`). It is the closest
stable, genuine two-person composition in the entire video: two real people tracked
continuously for the whole 9.6 s, one speaking, side by side — matching the supplied
reference screenshot.

## 4. Actual measured geometry of the reference

Real production detections (YOLO face detector, BoT-SORT tracks, 80/83 detections over
the window; `scratch/closepair_geometry_ref.json`):

| Quantity | Value |
|---|---|
| Left face bbox | [649.2, 350.9, 74.6, 98.4] px, conf 0.887 |
| Right face bbox | [1165.3, 343.8, 70.8, 103.3] px, conf 0.927 |
| Left subject cx / right subject cx | 709.1 / 1209.6 px |
| **Outer face-box span (`face_span`)** | **566.4 px** (constant across all 20 sub-intervals) |
| Centre distance | 500.5 px |
| Pair midpoint (tracked centres) | 959.4 px |
| **Head extents** (safe-box model `fx − 0.12·fw` … `fx + 1.12·fw`) | **640.2 → 1244.6 px** |
| **Head span** | **604.4 px** (head centre 942.4) |
| Union safe box (incl. shoulders) | 604.4 → 1278.6 px, **width 674.2 px**, centre 941.5 |

Scale sweep on the real geometry (`fit_table`, `closepair_geometry_ref.json`): the union
box fits at **no** scale in [0.80, 1.10] (`union_fits: false` everywhere, slack −35.8 px
at scale 0.80 worsening to −210.6 px at 1.10), while the **head extents fit at every
scale ≥ 0.80** down to 608 px (head span 604.4 ≤ 608, slack +3.6 px at scale 1.00).

## 5. Why the previous rule classified it differently from the desired visual definition

The desired visual definition (from the reference screenshot): both people held in one
9:16 composition. The old rule said **no**:

- `face_span 566.4 > 559.4` (the 0.92 bound) → missed by **7 px** → classified
  `wide_two_shot`, role `speaker`.
- Consequence measured in Original mode (`scratch/closepair_measure_original_452.json`):
  20/20 sub-intervals `wide_two_shot`/`speaker`, crop oscillates between the two
  speakers (crop centre 681.9 ↔ 1181.3, target_cx 685…1223) — i.e. it frames the active
  speaker and cuts the other person out. Render: **1 face in 11/12 frames.**

The rule failed on the reference for two reasons:

1. **Wrong span quantity.** Outer face-box edges under-measure what must fit: the
   face box crops hair and head silhouette, and the decision needs the *head extents*
   (the visually non-clippable region). The reference's head span (604.4) is 38 px wider
   than its face span (566.4) — so the "distance" the rule measured was not the
   composition width that actually has to be contained.
2. **Wrong reference width.** The bound was `0.92 × crop_w_baseline` — an arbitrary
   safety ratio against the *crop*, not against anything geometric. The correct
   reference width is the **widest renderable 9:16 crop**, which is a property of the
   source, not a tunable ratio.

## 6. Investigation of whether face span alone is the correct abstraction

**Investigated and rejected.** Three candidate abstractions were measured against the
reference geometry:

| Candidate | Reference value | Fits 608 px? | Verdict |
|---|---|---|---|
| Outer face-box span | 566.4 px | yes (but by accident) | Wrong quantity — under-measures heads |
| **Head extents** | **604.4 px** | **yes (+3.6 px)** | **Correct — the visually non-clippable region** |
| Union safe box (heads + shoulders) | 674.2 px | **no** (slack −66.1 px at scale 1.0) | Too conservative — rejects the reference itself |

The union safe box is 66 px **wider** than the renderable crop. A union-based criterion
would classify the reference pair as FAR — the exact opposite error. A two-shot that
clips a few px of outer shoulder is a normal two-shot; a two-shot that clips a head is
not. Therefore the correct abstraction is the **head extents**, and the fit test is
`head_span ≤ renderable_w`.

## 7. Final close-pair geometry criterion

Implemented in `_pair_composition_fits` (`speaker_tracker.py:580`), used by
`resolve_visual_subject` and the position solve:

```
renderable_w = min(source_h * 9/16, source_w)          # 608 px at 1080p
head_left    = fx − 0.12 * fw                           # from compute_subject_safe_box
head_right   = fx + 1.12 * fw
head_span    = head_right − head_left

close pair  ⟺  head_span ≤ renderable_w
needed_scale = min(1.10, renderable_w / head_span)      # CEILING, not a floor
```

Supporting committed changes:

- **Two-shot scale band** in `classify_shot_and_estimate_scale`: under Adaptive the
  `two_shot_both_fit` band is `[1.00, needed_scale]` (was the fixed `[0.88, 1.10]`).
  `ideal_scale` is clamped into `[1.00, needed_scale]`. The clamp is essential: the
  "wide" shot class proposes scale 0.95, which would open a 640 px crop — wider than
  the renderable 608.
- **Tight-pair head-centre override** in the position solve: when
  `head_span + 2·eff_crop_w·SAFETY_MARGIN_RATIO > eff_crop_w` (the reference case), the
  crop is centred on the pair **head centre** rather than the union-box centre. With
  asymmetric shoulder overhang the union centre can push a head outside the crop. On
  the reference the two centres differ by only 0.9 px, but the override is what
  guarantees the symmetric head margins. `solve_containment_crop_x` is untouched.
- **Plan emission:** `final_crop_w = crop_w_baseline` **always** (608 px). A widening
  path was implemented, proven invalid, and reverted (see §17).
- Constants: `SAFETY_MARGIN_RATIO = 0.08`, `MIN_SCALE = 0.80`, `MAX_SCALE = 1.35`,
  `TWO_SHOT_MAX_SPAN = 0.65` (Original mode only). `TWO_SHOT_MAX_SPAN_ADAPTIVE` is
  **deleted**.

For the reference: `needed_scale = 608 / 604.4 = 1.0059`, band `[1.00, 1.0059]` → scale
**1.00**, crop 608 px at x = 638.4 (crop centre 942.4 = pair head centre). The 608-wide
crop spans 638.4 → 1246.4: contains both heads (640.2 ≥ 638.4, 1244.6 ≤ 1246.4) with
~1.8 px symmetric margin, contains both faces fully, and clips only ~33 px of each outer
shoulder — a normal two-shot.

## 8. Why the criterion is general and not video-specific

- **No raw-pixel threshold.** The comparison is one length against another length derived
  from the source's own dimensions. Nothing in it is a tuned constant.
- **Resolution independent.** Both `head_span` and `renderable_w` scale with the source;
  the truth value is invariant to resolution. (Verified on a second video, 3840×2160 —
  the Adaptive windows table in Brain §33.4.)
- **Subject-size and camera-distance independent.** `head_span` is measured in source
  pixels from detections; a closer camera makes both heads and the span larger together,
  and the criterion still asks exactly "do both heads fit in the widest portrait crop?"
- **No timestamp, no shot-ID, no per-video tuning anywhere in the path.** The reference
  window is used only to *calibrate the abstraction*, never to branch the code.
- The only free parameters are the head-extent margins (0.12·fw / 1.12·fw), which are
  part of the pre-existing safe-box model shared with all other framing decisions — and
  the 1.10 scale ceiling, which caps zoom and is orthogonal to the close/far decision.

## 9. Close-pair real render evidence — **PASS**

Production render path (`closepair_render.exe` → real `speaker_tracker` → real FFmpeg),
not a diagnostic branch. `tmp/closepair_renders/harness_run3.log`:

- Classification: **20/20** sub-intervals `two_shot_both_fit`, role `two_shot`, scale
  1.00, crop_w 608, x = 638.4, crop centre 942.4 = pair midpoint, `pair_needed_w` 604.0.
  (`scratch/closepair_measure_adaptive_452.json`)
- Emitted plan: `single`, **1 segment**, `x=638, y=0, w=608, h=1080`, DualFrame off.
- FFmpeg filter: `crop=w=608:h=1080:x='638':y='0',scale=1080:1920:flags=lanczos+…` —
  one crop, one segment, **no split / no vstack / no concat**.
- Pixel verification (`scratch/closepair_verify_render.py` →
  `tmp/closepair_renders/pixel_verify.json`): 1080×1920, 25 fps, **2 faces in 12/12
  sampled frames** (`frac_frames_both = 1.0`), face x positions ≈ 0.08 / 0.92 of width —
  both people, one composition, for the entire clip.
- Active-speaker rule holds: the speaker **never** overrides a valid close-pair
  composition — both people are kept even though only one is speaking.

## 10. Far-pair real render evidence — **PASS**

Control window **977.6 s → 1001.3 s** (23.7 s), a genuinely far two-person segment.

- Classification (`scratch/closepair_measure_adaptive_977.json`): **0/23** two-person
  sub-intervals fit — `face_span 954.9`, head/needed width **1032.0 > 608** →
  `multi_person_panel` / `wide_two_shot`, role `speaker`. Scale 1.05, crop follows the
  active speaker.
- Emitted plan: `single`, 1 segment, speaker-following trajectory, DualFrame off.
- Pixel verification: 1080×1920, **1 face in 12/12 frames** (`frac_frames_one = 1.0`) —
  active-speaker isolation, no split-screen.

## 11. Original 9:16 regression evidence — **PASS**

Same two windows, `AUTOSHORTS_FRAMING_MODE=original` (i.e. the existing 9:16 behaviour):

- **Close pair, Original:** 20/20 `wide_two_shot` / `speaker` — speaker isolation, crop
  centre 681.9 ↔ 1181.3 vs pair midpoint 959.4; plan `x=if(lt(t,2.5),859,if(lt(t,7.5),393,861))`.
  Pixel: 1 face in 11/12 frames. DualFrame remains eligible. **Unchanged behaviour.**
- **Far pair, Original:** `dual_stack` — **3 segments** (1 dual-stack, enters t=14.22,
  exits t=23.72), vstack filter in the emitted `-filter_complex`. Pixel: up to 2 faces.
  **DualFrame still engages in Original mode exactly as before.**
- The Original-mode span bound (`TWO_SHOT_MAX_SPAN = 0.65`) and the entire Original code
  path are untouched; `solve_containment_crop_x` is untouched.

## 12. DualFrame evidence — **PASS**

The full 2×2 matrix was rendered through the real harness:

| Window | Mode | DualFrame | Segments | Faces in render |
|---|---|---|---|---|
| Close pair | Adaptive | **OFF** | 1 (`single`) | 2 in 12/12 |
| Close pair | Original | eligible (single here) | 1 | 1 in 11/12 |
| Far pair | Adaptive | **OFF** | 1 (`single`) | 1 in 12/12 |
| Far pair | Original | **ON** (`dual_stack`) | 3 | up to 2 |

The Adaptive-mode rows are the success condition: DualFrame off in both cases, with the
composition difference expressed purely through crop geometry (joint two-shot vs
speaker isolation). The Original rows prove no regression in the existing DualFrame
behaviour.

## 13. Smart Pacing v1/v2 regression evidence — **PASS**

- `test_smart_pacing_suite.py`: **55 checks, 0 failures.**
- `test_smart_pacing_2_suite.py`: **108 checks, 0 failures.**
- Real-render telemetry on all four renders: `[Smart Pacing 2.0] no v2 edits found —
  falling back to the v1 plan` — the v1/v2 fallback chain is exercised on the real
  footage and behaves as before. The framing change touches crop geometry only; the
  pacing pipeline consumes `LayoutSegment` generically.

## 14. Audio / caption / framing regression evidence — **PASS**

- `test_audio_intelligence_suite.py`: **48 checks, 0 failures.**
- `test_caption_qa_suite.py`: **138 checks, 0 failures.**
- `test_active_speaker_suite.py`: **61/61** — including the rewritten `test_10c` for the
  head-span criterion (synthetic pair: head span 579.2 ≤ 608, needed scale 1.0497;
  assertions on head containment `head_left 630.4`, `head_right 1209.6`, needed scale in
  [1.0, 1.10], `target_cx 920` vs Original `target_cx 1160`). `test_10b`/`test_10d`
  unchanged and passing.
- `test_dualframe_suite.py`: **55/55.** `test_framing_regression_suite.py`: **16/16.**
- `test_hook_closure_suite.py` 47/0, `test_hook_ending_optimization_suite.py` 28/0,
  `test_applied_features_suite.py` 30/0.
- Rust: `cargo test --lib --release` → **272 passed, 0 failed, 1 ignored** (~75 s).
- All four real renders carry real ASS subtitle burns (`closepair_adaptive.ass`, etc.)
  produced by the real caption pipeline.

## 15. Exact commands actually executed

```bash
# 0. Environment
PYTHON="D:/College/Autoshorts 10.0/.venv/Scripts/python.exe"
VIDEO="D:/College/Autoshorts 10.0/My thoughts on my 2023_24 season and the truth about my future. Talk with my friend Rio (PART 1) - UR · Cristiano (1080p, h264).mp4"

# 1. Global scan for real two-person windows (whole 1482 s video)
$PYTHON scratch/closepair_trackscan.py "$VIDEO"

# 2. Reference + far geometry dump (safe boxes, head extents, scale sweep)
$PYTHON scratch/closepair_geometry.py "$VIDEO" 452.6 462.2 scratch/closepair_geometry_ref.json
$PYTHON scratch/closepair_geometry.py "$VIDEO" 977.6 1001.3 scratch/closepair_geometry_far.json

# 3. Authoritative per-sub-interval measurement via the real production entry point
$PYTHON scratch/closepair_measure.py 452.6 462.2 adaptive scratch/closepair_measure_adaptive_452.json
$PYTHON scratch/closepair_measure.py 452.6 462.2 original scratch/closepair_measure_original_452.json
$PYTHON scratch/closepair_measure.py 977.6 1001.3 adaptive scratch/closepair_measure_adaptive_977.json

# 4. Plan JSON straight from the CLI (proves the emitted plan, not just instrumentation)
AUTOSHORTS_FRAMING_MODE=adaptive $PYTHON autoshorts/src-tauri/scripts/speaker_tracker.py \
  "$VIDEO" 452600 462200 608 1312 656
# -> {"mode":"single","x":"638","y":"0","w":"608","h":"1080", 1 segment, DualFrame off, avg_scale 1.000, corr 0}

# 5. Honest renders: close + far, adaptive + original (production binary, real tracker)
#    CRITICAL: MUST run with CWD = autoshorts/ — find_speaker_tracker_script (media.rs:272)
#    resolves the dev path src-tauri/scripts/speaker_tracker.py relative to CWD; from the
#    repo root it returns None and every case silently falls back to centre crop x=656.
cd autoshorts
AUTOSHORTS_PYTHON="$PYTHON" src-tauri/target/release/closepair_render.exe \
  452.6 462.2 977.6 1001.3     # -> 4 PASS / 0 FAIL, 0 fallbacks
cd ..

# 6. Pixel-level verification of the four rendered MP4s
$PYTHON scratch/closepair_verify_render.py   # -> tmp/closepair_renders/pixel_verify.json

# 7. Full regression suites
$PYTHON test_active_speaker_suite.py                                   # 61/61
$PYTHON test_dualframe_suite.py                                        # 55/55
$PYTHON test_framing_regression_suite.py                               # 16/16
$PYTHON test_caption_qa_suite.py                                       # 138/0
$PYTHON test_smart_pacing_suite.py                                     # 55/0
$PYTHON test_smart_pacing_2_suite.py                                   # 108/0
$PYTHON test_hook_closure_suite.py                                     # 47/0
$PYTHON test_hook_ending_optimization_suite.py                         # 28/0
$PYTHON test_audio_intelligence_suite.py                               # 48/0
$PYTHON test_applied_features_suite.py                                 # 30/0
(cd autoshorts && cargo test --lib --release)                          # 272 passed / 0 failed / 1 ignored
```

Note on §15 item 7: `caption_visual_qa.py` is an argparse CLI tool (args: mp4, ass,
template_json), not a self-running suite; it was not part of this regression.

## 16. Exact defects discovered

**D1 — Retired threshold misclassified the reference (real defect, fixed).**
`TWO_SHOT_MAX_SPAN_ADAPTIVE = 0.92` (559.4 px) rejected the reference pair by **7 px**
(face span 566.4). Root cause: the bound was a tuned ratio of the crop, compared against
outer face-box edges — neither the right width nor the right span (§5, §6). The same
rule is also effectively resolution-dependent.

**D2 — Intermittent harness fallback (diagnostic only, NOT a code defect).**
The first render runs intermittently produced centre-crop plans (`x=656`) with no error
text. Root cause: `find_speaker_tracker_script` (`media.rs:272`) resolves a CWD-relative
dev path; running the harness from the repo root made it return `None` and every case
silently fell back. **No production code was changed for this** — the fix is the CWD
requirement documented in §15 item 5. Recorded because it would otherwise look like a
flaky pipeline.

**D3 — First-sub-interval transient 0.95 / 640 px (real defect, fixed).**
After the criterion rewrite, the first 0.5 s sub-interval still emitted scale 0.95 and
crop width 640 — the "wide" shot class's ideal scale leaking past the new band. Root
cause: the two-shot band was applied to the *min/max* but the *ideal* scale was not
clamped into it, so a sub-threshold proposal survived (§17, fix F2).

## 17. Exact fixes made

1. **F1 — Criterion rewrite.** Deleted `TWO_SHOT_MAX_SPAN_ADAPTIVE`; added
   `_pair_composition_fits` (`speaker_tracker.py:580`) comparing the pair's **head span**
   against `renderable_w = min(source_h·9/16, source_w)`, returning
   `(fits, needed_scale)` with `needed_scale` as a ceiling. Wired into
   `resolve_visual_subject` (replacing the span gate) and the position solve.
2. **F2 — Ideal-scale clamp.** In `classify_shot_and_estimate_scale` the Adaptive
   two-shot band is `[1.00, needed_scale]` and `ideal_scale` is clamped into it,
   eliminating the D3 transient (0.95/640 would open a crop wider than the renderable
   608 px).
3. **F3 — Plan-emission widening reverted.** An earlier path emitted a wider crop
   (720 px) for close pairs. Proven geometrically invalid: `build_layout_filtergraph`
   (`media.rs:1332-1333`) renders single segments as `crop=w:h:x:y` then a fixed
   `scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int` with **no aspect
   correction** — the widest true-9:16 crop of 1920×1080 is 607.5 → 608 px, a 720×1080
   crop stretches anisotropically (sx 1.5 vs sy 1.7778, 18.5%), and a 720-wide 9:16 crop
   needs height 1280 > 1080. Reverted to `final_crop_w = crop_w_baseline` always, with a
   code comment recording why wider is not renderable.
4. **F4 — Tight-pair head-centre override.** In the position solve, when
   `head_span + 2·eff_crop_w·SAFETY_MARGIN_RATIO > eff_crop_w`, centre on the pair head
   centre instead of the union-box centre. `solve_containment_crop_x` untouched.
5. **F5 — `test_10c` rewritten** for the head-span criterion (synthetic pair, assertions
   on head containment, needed-scale band, and `target_cx` 920 vs Original 1160).
   `test_10b`/`test_10d` unchanged.

## 18. Brain documentation changes

`AUTOSHORTS_10_0_BRAIN.md`:

- §33.3/§33.4/§33.5/§33.6 rewritten for the calibrated Adaptive architecture
  (criterion, scale band, render-width invariant, position solve).
- **New §33.7 "Close-Pair Geometry Calibration (2026-09-23)"**: reference geometry, the
  criterion, the §7 crop-width investigation, the tight-pair override, and the four
  verification bullets (close/far × adaptive/original).
- Test-count references for `test_active_speaker_suite` updated 58 → 61 (lines 123, 173,
  577, 960, 962, 1109, 1188).
- §32 (Smart Pacing 2.0) reviewed and left unchanged — already accurate.

## 19. Remaining limitations

1. **Seam-correlation metric is supportive, not decisive.** The pixel verifier's
   `seam_corr` (correlation across the vertical centre line) was computed for all four
   renders (close-adaptive 0.126/0.07 mean/min; far-adaptive 0.459/0.334;
   far-original 0.283/−0.025; close-original 0.199/0.031). A split-screen render does
   not reliably produce a *low* centre-line correlation, so this number is reported as
   context only. The authoritative "no split" evidence is the emitted plan JSON and the
   FFmpeg filter string (1 segment, `crop`+`scale`, no `vstack`/`split`/`concat`) —
   which is exactly what §9–§12 cite.
2. **Head margins are tight by construction.** At scale 1.00 the reference's heads are
   contained with ~1.8 px symmetric margin, and ~33 px of each outer shoulder is
   clipped. This is a normal two-shot, but there is no slack for a pair whose head span
   is *just* under 608 — the render will be correct but visually unforgiving at the
   extremes.
3. **Head-extent margins are a heuristic.** `head_left = fx − 0.12·fw`,
   `head_right = fx + 1.12·fw` come from the pre-existing safe-box model. They encode an
   assumption about how much of the head lies outside the detected face box. They were
   not re-derived from first principles here; they are shared with all other framing
   decisions, which is why the criterion behaves consistently.
4. **Real-footage end-to-end coverage is limited.** Verified: one real close-pair window
   (452.6–462.2 s) and one real far window (977.6–1001.3 s) on the reference video,
   plus the Adaptive windows of a second video (3840×2160, Brain §33.4 table) at the
   classification level. Other real close pairs in the 38-window scan were not rendered
   end-to-end.
5. **Borderline cases not exercised on real footage.** No real window in the scan has a
   head span just under the renderable width, so the near-boundary behaviour of the
   criterion (needed_scale → 1.10 clamp, head-centre override engaging) is verified only
   on synthetic geometry (`test_10c`, head span 579.2). **LIMITATION.**
6. **`caption_visual_qa.py` not re-run** — it is an argparse CLI requiring mp4/ass/
   template_json arguments, not a suite; caption regression is instead covered by
   `test_caption_qa_suite.py` (138/0) and by the real ASS burns on all four renders.
7. **Harness CWD fragility (process, not product).** The dev-path script discovery in
   `media.rs:272` silently falls back to centre crop when run from the wrong CWD (§16
   D2). This affects only the local dev harness, not shipped behaviour, but it will
   silently produce false results for anyone who runs it from the repo root.

---

## Conclusion

The close-pair decision is now a **single geometric comparison** — do both subjects' head
extents fit inside the widest 9:16 crop renderable from this source — calibrated against
the supplied reference frame and verified end-to-end on real footage through the
production render path. The retired 0.92 raw-span threshold misclassified that same
reference by 7 px because it measured outer face-box edges against an arbitrary ratio of
the crop; the union-safe-box alternative would have rejected the reference outright
(674.2 px > 608 px). All three success conditions hold: the close pair is held together
with DualFrame off (2 faces in 12/12 frames), the far pair isolates the active speaker
with DualFrame off (1 face in 12/12 frames), and the Original 9:16 behaviour is
unchanged (DualFrame still engages, speaker isolation intact). Full regression is green:
586 Python checks and 272 Rust tests, zero failures. Remaining limitations are documented
in §19 and are inherent to the coverage available, not to the criterion's correctness.
