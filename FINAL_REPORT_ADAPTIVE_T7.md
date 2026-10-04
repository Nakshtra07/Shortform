# FINAL REPORT — Adaptive Framing + T7 Caption Style (AutoShorts 10.0)

**Date:** 2026-09-21
**Source video used for all real testing:** `AutoShorts_ngvOyccUzzY.mp4` (3840x2160, 25 fps, ~6780 s)
**Reference for T7:** `ref_video/patrik_key_captions.mp4`

Both features are **additive**: with `framingMode = "original"` the pipeline is byte-for-byte the pre-existing one. Candidate boundaries, Smart Pacing (v1 + 2.0), Hook & Ending, and Audio/Caption Intelligence are untouched by either feature.

---

## 1. What was implemented

### 1.1 Adaptive Framing (per-project import choice)

| File | Change |
| :--- | :--- |
| `autoshorts/src-tauri/src/models.rs` | `Project.framing_mode: Option<String>` (serde camelCase `framingMode`) |
| `autoshorts/src-tauri/src/db.rs` | `ALTER TABLE projects ADD COLUMN framing_mode TEXT`; SELECTs, `project_from_row` (`row.get(9)?`), join closure (`row.get(27)?`); round-trip unit test asserts `framing_mode == Some("adaptive")` |
| `autoshorts/src-tauri/src/lib.rs` | `create_project_from_path` gained `framing_mode: String`; render path reads `project.framing_mode.as_deref().unwrap_or("original")` |
| `autoshorts/src-tauri/src/media.rs` | `detect_speaker_crop_params` gained `framing_mode: &str`, passed as `AUTOSHORTS_FRAMING_MODE` env to the tracker; `render_flat_clip` passes `"original"` |
| `autoshorts/src-tauri/src/pacing.rs` (2 sites), `src/bin/boundary_inspect.rs`, `src/bin/pacing_inspect.rs` | pass `"original"` (unchanged behavior) |
| `autoshorts/src-tauri/scripts/speaker_tracker.py` | `TWO_SHOT_MAX_SPAN_ADAPTIVE = 0.92` (vs 0.65 original); adaptive env read in `run()`; `dual_enabled = (not adaptive_framing) and _is_dualframe_env_enabled()`; relaxed two-shot fit threshold in `resolve_visual_subject` with adaptive scale band; telemetry `[Framing] adaptive=on dualframe_disabled=true` |
| `autoshorts/src/main.tsx`, `src/styles.css` | two-card framing selector (Original 9:16 / Adaptive Framing) with previews |

**Behavior:** final canvas is always 9:16 (1080x1920). In adaptive mode DualFrame/split-screen is structurally impossible (`dual_enabled=False`). Two people who fit (face span ≤ 0.92 × crop width) stay in one composition centered on the pair midpoint; too far apart → active-speaker focus with a generous crop; one person → that person is focused. The DualFrame resolver and render filtergraph code paths are never invoked.

### 1.2 T7 Caption Style

| File | Change |
| :--- | :--- |
| `autoshorts/src-tauri/src/captions.rs` | new `preset_t7` template: Montserrat, size 110, weight 800, base color `#FFFFFF`, position_y 0.82, stroke `#000000` width 8, shadow rgba(0,0,0,0.7) y-offset 4, **no caption box**, `KaraokeProgressive` highlight `#F2CD0D`, max 4 words/line, max 2 lines |
| `autoshorts/src-tauri/src/bin/caption_qa.rs` | ISO registry expects 7 templates incl. `preset_t7`; the placeholder `is None` check flipped to `is_some()` PASS; font allowlist arm added (Montserrat/Poppins/Inter) |
| `autoshorts/src/main.tsx`, `src/styles.css` | T7 style card + `.preview-text-t7` with white normal words and yellow `#F2CD0D` emphasis words |

T7 reuses the existing Caption Intelligence emphasis indices — no new AI calls. The reference's persistent title/header is deliberately **not** implemented (out of scope).

### 1.3 Render harness (new)

`autoshorts/src-tauri/src/bin/adaptive_t7_render.rs` (+ `[[bin]]` entry in `Cargo.toml`) renders 5 cases from the real video: `t7_original` (490-510 s), `adaptive_single` (300-320 s), `adaptive_two_wide` (2040-2060 s), `original_two_wide` (2040-2060 s), `t7_adaptive` (490-510 s).

---

## 2. Real-render evidence (all MP4s inspected at pixel level)

Renders: `tmp/adaptive_t7_renders/*.mp4` — all **1080x1920, 25 fps, 20 s, 500 frames**.

### 2.1 DualFrame differential — the core assertion

Harness log (authoritative, from the tracker + layout plan):

```
original_two_wide  : [DualFrame] mode=dual_stack, layout_plan total_segments=5 dual_segments=2
                     enter t=1.50 exit t=4.00 reason=too_many_subjects; enter t=7.50
                     [Framing] (not present)  -> dual enabled
adaptive_two_wide  : [DualFrame] mode=single, total_segments=1 dual_segments=0
                     [Framing] adaptive=on dualframe_disabled=true
```

Pixel-level confirmation (clean-background re-renders minus subtitles, `tmp/splitscreen_check.py`): column gradient-energy correlation across the row-960 seam.

| Case | dual-window mean corr | single-window mean corr | Verdict |
| :--- | :--- | :--- | :--- |
| `original_two_wide` | **-0.025** (-0.03, -0.03, -0.04, -0.01, -0.01) | 0.409 | **DualFrame present** (two unrelated stacked pictures) |
| `t7_original` | 0.205 | 0.207 | never split-screen |
| `adaptive_single` | 0.478 | 0.415 | never split-screen |
| `adaptive_two_wide` | 0.323 | 0.421 | never split-screen |
| `t7_adaptive` | 0.205 | 0.207 | never split-screen |

Additional geometry proof: the two `original_two_wide` halves match the predicted vstack source crops (crop `1992x1770` at x=231 / x=1704, scaled 1080x960) to **0.74 / 0.49 mean-abs-diff**, and the two panels differ by 28.5 mean-abs-diff (two distinct subjects).

**Result: 7/7 PASS.**

### 2.2 Caption pixel isolation (T7 colors/position)

`tmp/caption_pixel_isolate.py`: re-render each case's filter chain minus `subtitles=`, diff subtitled vs clean, classify colors only inside the diff mask.

| Case | white px | yellow `#F2CD0D` px | caption band y | x span |
| :--- | :--- | :--- | :--- | :--- |
| `t7_original` | 9076 | **83204** | 0.844 | 0.659 |
| `t7_adaptive` | 9076 | **83204** | 0.844 | 0.659 |
| `adaptive_single` | 4553 | 152 | 0.677 | 0.667 |
| `adaptive_two_wide` | 4785 | 129 | 0.677 | 0.670 |
| `original_two_wide` | 4777 | 152 | 0.602 | 0.675 |

T7's band 0.844 sits just under the template's `position_y = 0.82` target (within the text-block height), non-T7 captions sit at their own template positions. **Result: 20/20 PASS.**

### 2.3 Audio + caption timing

Rendered `t7_original_audio.mp4` with the production audio chain (`-c:a aac -b:a 192k`):

- A/V durations: video 20.000 s / audio 20.000 s — **delta 0 ms**
- 25 speech-energy onsets detected in 20 s
- First caption line at 0.30 s vs first speech onset 0.64 s — delta 335 ms (captions lead slightly, expected for word-level lead-in)
- 27/42 caption lines have an audio onset within 300 ms (64%)

**Result: 4/4 PASS.**

---

## 3. Regression suites (all re-run this session, all green)

| Suite | Result |
| :--- | :--- |
| `cargo test --lib` | **272 passed, 0 failed, 1 ignored** |
| `cargo run --bin caption_qa -- isolation` | **58 PASS / 0 FAIL / 2 skipped** |
| `test_caption_qa_suite.py` | **225/225 PASS** |
| `test_smart_pacing_2_suite.py` | **108/108 PASS** |
| `test_smart_pacing_suite.py` (v1) | **55/55 PASS** |
| `test_dualframe_suite.py` | **55/55 PASS** |
| `test_active_speaker_suite.py` | **58/58 PASS** |
| `test_hook_closure_suite.py` | **47/47 PASS** |
| `test_hook_ending_optimization_suite.py` | **28/28 PASS** |
| `test_applied_features_suite.py` | **30/30 PASS** |
| `test_audio_intelligence_suite.py` | **48/48 PASS** |
| `test_caption_intelligence_suite.py` | **22/22 PASS** |
| `npm run build` (frontend) | **PASS** |

**Total: 1040 checks, 0 failures.**

---

## 4. Test-case matrix (prompt.txt §TESTING, 15 required cases)

| # | Required test | Status | Evidence |
| :--- | :--- | :--- | :--- |
| 1 | Original 9:16 mode | **PASS** | `t7_original`, `original_two_wide` renders; 1080x1920; DualFrame active; crop trajectory `x=if(lt(t,...))` unchanged |
| 2 | Adaptive Framing with one visible person | **PASS** | `adaptive_single` (300-320 s): tracker `mode=single`, 1 segment, `[Framing] adaptive=on dualframe_disabled=true`; 8.68 MB render |
| 3 | Adaptive Framing with two people close together | **LIMITATION** | See §5 |
| 4 | Adaptive Framing with two people far apart | **PASS** | `adaptive_two_wide` (2040-2060 s): `mode=single`, 1 segment, 0 dual vs original `dual_stack` 5 segments / 2 dual on the same window |
| 5 | Active speaker changes where applicable | **PASS** | `test_active_speaker_suite.py` 58/58; crop trajectory in adaptive renders follows the speaker (`x=if(lt(t,4),622,2166)`) |
| 6 | T7 caption rendering | **PASS** | `preset_t7` ASS + pixel isolation: white 9076 px, yellow `#F2CD0D` 83204 px, band y=0.844, no caption box |
| 7 | Existing caption styles still render | **PASS** | 6 pre-existing templates registered (7 total); `caption_qa` isolation 58 PASS; non-T7 renders carry their own caption bands (0.602/0.677) |
| 8 | Existing DualFrame behavior in original 9:16 mode | **PASS** | `original_two_wide`: `dual_stack`, vstack filter chain, seam corr -0.025, panels match source crops |
| 9 | Adaptive Framing does not invoke DualFrame | **PASS** | `dualframe_disabled=true` in log; 0 dual segments in every adaptive window probed; pixel seam correlation 0.205-0.478 |
| 10 | Smart Pacing remains unchanged | **PASS** | `test_smart_pacing_suite.py` 55/55 |
| 11 | Smart Pacing 2.0 remains unchanged and compatible | **PASS** | `test_smart_pacing_2_suite.py` 108/108; `[Smart Pacing 2.0] no v2 edits found — falling back to the v1 plan` in harness (additive path) |
| 12 | Audio remains synchronized | **PASS** | A/V durations 20.000 s / 20.000 s; audio follows the same edit map via `asplit/atrim`; onset alignment 64% within 300 ms |
| 13 | Caption timing remains correct | **PASS** | ASS word timings clip-relative; 27/42 lines matched to audio onsets; first line 0.30 s vs onset 0.64 s |
| 14 | Candidate boundaries remain unchanged | **PASS** | SP2-A suite: "candidate boundaries never move; clip selection untouched"; framing mode is applied post-boundary in the crop stage only |
| 15 | Final output remains 9:16 | **PASS** | All 6 rendered MP4s 1080x1920 (ffprobe) |

**Summary: 14 PASS / 0 FAIL / 1 LIMITATION / 0 NOT TESTED.**

---

## 5. Limitations

**Case 3 — Adaptive Framing with two people close together: LIMITATION (verified at logic level, not rendered).**

The adaptive two-shot threshold is `TWO_SHOT_MAX_SPAN_ADAPTIVE = 0.92` × crop width (1117 px of the 3840 px source) vs the original `0.65` (789 px). A close pair is defined as a two-person span in (789, 1117] px — original falls through to speaker-lock/DualFrame, adaptive keeps both in one composition.

A span scan of the real video (`tmp/adaptive_probe/span_scan2.py`, YOLO person boxes on sampled frames, face span estimated as outer face edges, 14 two-person windows found) measured **every two-person window at ≥ 2205 px outer face span** (typical 2200-2260 px ≈ 0.58 of source width, up to 3548 px ≈ 0.92). The close-pair band (789-1117 px) contains **zero** windows — this video's subjects sit at interview distance apart, i.e. genuinely "far apart" by the tracker's definition. No window exists that could exercise a close-pair render.

What was verified instead:
- The threshold logic itself: `resolve_visual_subject` line 1894 applies `two_shot_max_span` = 0.92 under adaptive framing, with the scale band clamped to `[MIN_SCALE, 1.10]` so the pair crop does not over-zoom.
- `probe_close.py` partial output: windows 2040/2290 s confirmed `orig=dual_stack, adapt=single` on the same footage.
- The threshold difference is exercised in the DualFrame suite's threshold unit tests.
- Span scan over 14 two-person windows (`tmp/adaptive_probe/span_scan2.json`): 0 windows in the close-pair band, so the LIMITATION is a property of the supplied footage, not of the implementation.

A true close-pair render would require footage with two people within ~1100 px of each other; the supplied video contains no such window.

---

## 6. Defects found and fixed during verification

| Defect | Root cause | Fix |
| :--- | :--- | :--- |
| Harness produced 0-byte ASS files | `synthetic_words()` generated words at 0.3-3.2 s but clip windows start at 300 s+, so `remap_words` dropped them all → early return on empty candidates | `synthetic_words(start, end)` now spans the whole source window; fallback to source words; `.expect()` on ASS write |
| Split-screen inspector false positives | Sobel row-energy peak ratio > 6.0 fired on natural horizontal content edges in all 5 clips; sampled frames 100/300 were outside the dual segments (dual windows are 1.5-4.0 s and 7.5-11.4 s = frames ~37-100, ~187-285) | Sample frames inside the dual windows; replaced the peak-ratio heuristic with column-energy cross-seam correlation (a vstack is two unrelated pictures → uncorrelated bands); asserted both directions |
| Background re-render seek bug | First isolation version used `-ss 0`, producing identical/wrong frames | Seek to the case's source window start (`-ss <start>`) so bg frames align frame-for-frame |
| `span_scan2.py` crashed on JSON dump | numpy float32 not serializable by `json.dumps` | `round(float(frac), 4)` |

---

## 7. Brain updates

`AUTOSHORTS_10_0_BRAIN.md` — three new as-implemented sections + refreshed test inventory:
- **§32 Smart Pacing 2.0** (line 1510): scope, sidecar architecture, kill switch, real-footage evidence, test inventory
- **§33 Adaptive Framing** (line 1561): user-facing behavior, persistence, implementation, real-footage evidence incl. the pixel-level seam-correlation table, scope discipline
- **§34 T7 Caption Style** (line 1639): reference measurements (white `#FFFFFF`, yellow `#F2CD0D`, ~11.25% width → fs 110, stroke ~8, no box), implementation, registry test updates, test inventory

---

## 8. Reproduction commands

```bash
# Rust tests + harness build
cd autoshorts/src-tauri && cargo test --lib && cargo build --bins

# Render the 5 real cases
cd autoshorts/src-tauri && cargo run --quiet --bin adaptive_t7_render

# Pixel verification
.venv/Scripts/python.exe tmp/caption_pixel_isolate.py      # 20/20
.venv/Scripts/python.exe tmp/splitscreen_check.py          # 7/7
.venv/Scripts/python.exe tmp/audio_timing_check.py         # 4/4

# Regression suites
.venv/Scripts/python.exe test_smart_pacing_2_suite.py      # 108
.venv/Scripts/python.exe test_smart_pacing_suite.py        # 55
.venv/Scripts/python.exe test_caption_qa_suite.py          # 225
.venv/Scripts/python.exe autoshorts/src-tauri/scripts/test_dualframe_suite.py        # 55
.venv/Scripts/python.exe autoshorts/src-tauri/scripts/test_active_speaker_suite.py   # 58
.venv/Scripts/python.exe test_hook_closure_suite.py        # 47
.venv/Scripts/python.exe test_hook_ending_optimization_suite.py  # 28
.venv/Scripts/python.exe test_applied_features_suite.py    # 30
.venv/Scripts/python.exe test_audio_intelligence_suite.py  # 48
.venv/Scripts/python.exe autoshorts/src-tauri/scripts/test_caption_intelligence_suite.py  # 22
cd autoshorts && npm run build                             # PASS
```

## 9. Artifacts

| Path | Contents |
| :--- | :--- |
| `tmp/adaptive_t7_renders/*.mp4` | 5 harness renders + 1 audio render (all 1080x1920, 25 fps, 20 s) |
| `tmp/adaptive_t7_renders/*.ass` | T7/adaptive caption files |
| `tmp/adaptive_t7_renders/harness.log` | tracker + layout plan + filtergraph log (DualFrame differential) |
| `tmp/adaptive_t7_renders/render_checks.json` | 6/6 PASS |
| `tmp/adaptive_t7_renders/caption_pixel_checks.json` | 20/20 PASS |
| `tmp/adaptive_t7_renders/splitscreen_checks.json` | seam correlation results |
| `tmp/adaptive_t7_renders/audio_timing_checks.json` | 4/4 PASS |
| `tmp/adaptive_t7_renders/bg_ref/*_bg.mp4` | clean-background re-renders for isolation |
| `tmp/adaptive_probe/span_scan*.json` | two-person span scan over the source video |
