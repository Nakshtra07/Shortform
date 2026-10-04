# DualFrame Visual-Layout Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the DualFrame visual-layout capability in AutoShorts 6.0 (`d:/College/Autoshorts 6.0`), allowing two simultaneously valid visual subjects in the same shot to receive independent 9:8 crops vertically composed into a 1080x1920 frame while preserving existing 5.0 single framing, continuous audio stream encoded once, and single-pass subtitles.

**Architecture:** A dedicated `DualFrameLayoutResolver` is inserted post-VSR in `speaker_tracker.py` to evaluate track persistence, VSR prominence, visibility, and crop-ability with anti-flicker hysteresis. It generates independent top and bottom 9:8 crop trajectories (left subject $\rightarrow$ top panel, right subject $\rightarrow$ bottom panel, locked by track ID) and outputs a structured temporal layout plan. `media.rs` consumes the plan and builds a valid multi-branch FFmpeg complex filtergraph (`split` $\rightarrow$ `trim` $\rightarrow$ `crop` $\rightarrow$ `scale` $\rightarrow$ `vstack` $\rightarrow$ `concat` $\rightarrow$ `subtitles`) with continuous audio from `0:a` encoded once.

**Tech Stack:** Python 3.12 (NumPy, OpenCV, Ultralytics YOLO11n, BoT-SORT), Rust 1.80+ (Tauri, Tokio, Serde, FFmpeg 6.0+).

## Global Constraints
- `d:/College/Autoshorts 5.0` MUST remain untouched. All changes inside `d:/College/Autoshorts 6.0`.
- Do NOT replace or modify YOLO11n, BoT-SORT, YuNet, VSR, diarization, single-mode framing solver, hook pipeline, or duplicate-render protection.
- DualFrame triggers ONLY for exactly TWO relevant persistent subjects. 3+ subjects or 1 subject remain in SINGLE mode.
- Top panel: 1080x960, Bottom panel: 1080x960, Composed output: 1080x1920 (9:16). Panel aspect ratio strictly 9:8 (`crop_w / crop_h = 1.125`).
- Subtitles remain a single full-frame overlay. Audio remains one continuous stream encoded once. Real A/V synchronization verification is the acceptance criterion ($|\Delta t| < 0.05\text{s}$).
- DualFrame is toggleable via `AUTOSHORTS_DUALFRAME_ENABLED` (default `"true"`).
- FFmpeg input streams MUST be explicitly split (`[0:v]split=K...`) when feeding multiple temporal branches.
- Test commands in Windows/.venv environment use direct script execution: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`.

---

### Task 1: Python DualFrame Configuration & Layout Resolver State Machine

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py`
- Test: `autoshorts/src-tauri/scripts/test_dualframe_suite.py`

**Interfaces:**
- Consumes: `genuine_tracks`, `shot_tracks`, `rel_ts`, `rel_te`, `is_shot_cut` from `speaker_tracker.py`.
- Produces: `DualFrameConfig`, `DualFrameLayoutResolver`, returning layout mode (`single` or `dual_stack`), selected top/bottom track IDs, and slot stability locks.

- [ ] **Step 1: Write the failing tests for DualFrame eligibility, hysteresis, and slot locking**
Create `autoshorts/src-tauri/scripts/test_dualframe_suite.py` with tests verifying:
1. 1 subject $\rightarrow$ `single`.
2. 2 subjects for 1 frame ($< 0.60\text{s}$) $\rightarrow$ remains `single` (anti-flicker entry condition).
3. 2 persistent subjects ($\ge 0.60\text{s}$) $\rightarrow$ enters `dual_stack`.
4. Left-to-right slot assignment: left subject locked to TOP panel, right subject locked to BOTTOM panel.
5. Speaker turn switch while both remain visible $\rightarrow$ slots DO NOT swap (slot stability).
6. 1 subject briefly occluded ($< 0.75\text{s}$) $\rightarrow$ remains `dual_stack`.
7. 1 subject absent for $> 0.75\text{s}$ $\rightarrow$ exits to `single`.
8. Camera cut $\rightarrow$ complete state reset.
9. 3 subjects $\rightarrow$ `single`.

- [ ] **Step 2: Run tests to verify they fail**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`
Expected: FAIL (modules/classes not yet defined).

- [ ] **Step 3: Implement `DualFrameConfig` and `DualFrameLayoutResolver`**
In `autoshorts/src-tauri/scripts/speaker_tracker.py`:
- Define `@dataclass(frozen=True) class DualFrameConfig`.
- Implement `class DualFrameLayoutResolver`:
  - `is_subject_eligible(track, source_w, source_h)` using track duration, VSR prominence, and geometry.
  - `update(rel_t, active_tracks, is_shot_cut)` tracking persistence, hysteresis, left-to-right entry slot assignment, and slot identity locking.
  - `reset()` for camera cuts.

- [ ] **Step 4: Run tests to verify they pass**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add autoshorts/src-tauri/scripts/speaker_tracker.py autoshorts/src-tauri/scripts/test_dualframe_suite.py
git commit -m "feat(dualframe): add DualFrameConfig and DualFrameLayoutResolver with anti-flicker state machine"
```

---

### Task 2: Independent 9:8 Panel Geometry & Dual Trajectory Generation

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py`
- Test: `autoshorts/src-tauri/scripts/test_dualframe_suite.py`

**Interfaces:**
- Consumes: Track data for `top_track` and `bottom_track`, `source_w`, `source_h`, `start_sec`.
- Produces: `generate_panel_crop(track, rel_t, source_w, source_h, panel_role)` and `solve_dual_frame_trajectories(...)` returning independent `CropRectExpr` for top and bottom panels.

- [ ] **Step 1: Write failing tests for independent 9:8 panel framing**
In `test_dualframe_suite.py`, add tests verifying:
1. `crop_w / crop_h` strictly equals $9.0 / 8.0 = 1.125$ for all generated panel crops.
2. Independent crops: top subject at $x=200$ and bottom subject at $x=1400$ produce independent $x$ trajectories centered on each subject.
3. Scale varies dynamically according to subject bounds, clamped within source bounds ($0 \le x \le W - \text{crop\_w}$, $0 \le y \le H - \text{crop\_h}$).
4. Zero non-uniform stretching.

- [ ] **Step 2: Run tests to verify they fail**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`
Expected: FAIL.

- [ ] **Step 3: Implement panel crop solver and trajectory generator**
In `speaker_tracker.py`:
- Add `compute_panel_crop_dimensions(subject_box, source_w, source_h) -> (crop_w, crop_h)`.
- Enforce $9:8$ aspect ratio (`crop_w = round(crop_h * 1.125)` with even pixel clamping).
- Position $x$ around subject smoothed $cx$ and $y$ around eye-line/headroom.
- Implement `solve_dual_frame_trajectories(...)` generating independent position keyframes and FFmpeg expressions for top and bottom panels.

- [ ] **Step 4: Run tests to verify they pass**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add autoshorts/src-tauri/scripts/speaker_tracker.py autoshorts/src-tauri/scripts/test_dualframe_suite.py
git commit -m "feat(dualframe): implement independent 9:8 panel crop trajectory solver"
```

---

### Task 3: Python Full Layout Plan JSON Output & Telemetry

**Files:**
- Modify: `autoshorts/src-tauri/scripts/speaker_tracker.py`
- Test: `autoshorts/src-tauri/scripts/test_dualframe_suite.py`

**Interfaces:**
- Consumes: Segments resolved by `DualFrameLayoutResolver` and single/dual crop solvers.
- Produces: JSON payload written to `stdout` containing `mode`, `x`, `y`, `w`, `h`, `segments`, and telemetry lines written to `stderr` prefixed with `[DualFrame]`.

- [ ] **Step 1: Write failing tests for layout plan JSON serialization and telemetry**
In `test_dualframe_suite.py`, add tests verifying:
1. Output JSON schema matches `{"mode": "...", "x": "...", "y": "...", "w": "...", "h": "...", "segments": [...]}`.
2. When dual segments exist, each segment contains `mode: "dual_stack"`, `start`, `end`, `top_crop`, `bottom_crop`.
3. When `AUTOSHORTS_DUALFRAME_ENABLED=false`, DualFrame is bypassed and output is 100% `single`.
4. Telemetry logs contain `[DualFrame] mode=...`, `[DualFrame] enter ...`, `[DualFrame] exit ...`.

- [ ] **Step 2: Run tests to verify they fail**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`
Expected: FAIL.

- [ ] **Step 3: Integrate layout plan into `run()` and format output JSON**
In `speaker_tracker.py`:
- Integrate `DualFrameLayoutResolver` into the shot iteration loop of `run()`.
- Segment continuous runs of `single` and `dual_stack`.
- Emit structured `[DualFrame]` logs to `sys.stderr`.
- Output the extended JSON with `segments` and backward-compatible fallback fields.

- [ ] **Step 4: Run tests to verify they pass**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" "d:\College\Autoshorts 6.0\autoshorts\src-tauri\scripts\test_dualframe_suite.py"`
Expected: PASS.

- [ ] **Step 5: Verify existing active speaker and ultralytics tests remain 100% passing**
Run:
`& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_active_speaker_suite.py`
`& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_ultralytics_adapter_suite.py`
Expected: 58/58 and 5/5 PASS.

- [ ] **Step 6: Commit**
```bash
git add autoshorts/src-tauri/scripts/speaker_tracker.py autoshorts/src-tauri/scripts/test_dualframe_suite.py
git commit -m "feat(dualframe): output full layout plan JSON and structured runtime telemetry"
```

---

### Task 4: Rust `media.rs` Data Contract Deserialization & Telemetry Forwarding

**Files:**
- Modify: `autoshorts/src-tauri/src/media.rs`
- Test: `autoshorts/src-tauri/src/media.rs` (unit tests)

**Interfaces:**
- Consumes: Python stdout JSON string.
- Produces: `SmartFramingPlan`, `LayoutSegment`, `CropRectExpr` in Rust, forwarding `[DualFrame]` telemetry lines to stdout.

- [ ] **Step 1: Write Rust failing unit test for `SmartFramingPlan` deserialization**
In `autoshorts/src-tauri/src/media.rs`:
Add test `test_parse_smart_framing_plan_with_dual_segments` verifying deserialization of:
1. Pure legacy single crop: `{"x": "100", "y": "0", "w": "608", "h": "1080"}`.
2. Plan with `Single` and `DualStack` segments.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test --lib media::tests::test_parse_smart_framing_plan_with_dual_segments`
Expected: FAIL (types not defined).

- [ ] **Step 3: Define Rust types and update `detect_speaker_crop_params`**
In `autoshorts/src-tauri/src/media.rs`:
- Define `CropRectExpr`, `LayoutSegment`, `SmartFramingPlan`.
- Update `detect_speaker_crop_params` to intercept and forward lines starting with `[DualFrame]` to `println!`.
- Return `SmartFramingPlan` containing parsed segments and fallback dimensions.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test --lib media::tests::test_parse_smart_framing_plan_with_dual_segments`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add autoshorts/src-tauri/src/media.rs
git commit -m "feat(dualframe): parse SmartFramingPlan and forward DualFrame telemetry in media.rs"
```

---

### Task 5: Rust `media.rs` Multi-Branch FFmpeg Filtergraph Construction

**Files:**
- Modify: `autoshorts/src-tauri/src/media.rs`
- Test: `autoshorts/src-tauri/src/media.rs` (unit tests)

**Interfaces:**
- Consumes: `SmartFramingPlan`, `source_path`, `start_sec`, `end_sec`, `ass_subtitle_path`.
- Produces: Valid multi-branch FFmpeg filtergraph string and `-vf` arguments in `render_flat_clip`.

- [ ] **Step 1: Write failing unit test for filtergraph string generation**
In `autoshorts/src-tauri/src/media.rs`:
Add `test_build_dualframe_ffmpeg_filtergraph` verifying:
1. 1 Single segment $\rightarrow$ produces exact legacy filter string `crop=...,scale=1080:1920`.
2. Multi-segment plan with 1 Single and 1 DualStack $\rightarrow$
   Calculates total branches: 1 for Single + 2 for DualStack = 3 branches.
   Generates `[0:v]split=3[b0][b1][b2];`
   Applies `trim` and `crop` to `b0` (scale 1080x1920), `b1` (scale 1080x960), `b2` (scale 1080x960), `vstack=inputs=2`, `concat=n=2:v=1:a=0[v_composed]`.
   Applies subtitles to `[v_composed]`.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test --lib media::tests::test_build_dualframe_ffmpeg_filtergraph`
Expected: FAIL.

- [ ] **Step 3: Implement `build_layout_filtergraph` in `media.rs`**
In `media.rs`:
- Implement helper function `build_layout_filtergraph(plan: &SmartFramingPlan, duration_sec: f64, ass_path: Option<&Path>, drawtext: Option<&str>) -> String`.
- Correctly split source video stream `[0:v]split=K...`.
- Wire `Single` and `DualStack` branches into `concat=n=N:v=1:a=0`.
- Append subtitle overlay to the concatenated 1080x1920 stream.
- Update `render_flat_clip` to invoke `build_layout_filtergraph`.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test --lib media::tests::test_build_dualframe_ffmpeg_filtergraph`
Expected: PASS.

- [ ] **Step 5: Run all 100+ cargo tests**
Run: `cargo test --lib`
Expected: All tests PASS.

- [ ] **Step 6: Commit**
```bash
git add autoshorts/src-tauri/src/media.rs
git commit -m "feat(dualframe): implement valid multi-branch FFmpeg filtergraph builder"
```

---

### Task 6: Real-Video Render, A/V Duration & Synchronization Verification

**Files:**
- Create: `autoshorts/src-tauri/scripts/test_dualframe_real_render.py`
- Test: Real video clip render with layout transitions.

**Interfaces:**
- Consumes: Test video with 2 visible speakers (e.g. `Messi vs Ronaldo Fans` or local clip).
- Produces: Rendered 1080x1920 MP4 file, ffprobe stream inspection, duration comparison report.

- [ ] **Step 1: Write automated end-to-end real render test script**
Create `autoshorts/src-tauri/scripts/test_dualframe_real_render.py`:
- Runs real clip with two speakers visible.
- Confirms `[DualFrame] enter` and `[DualFrame] mode=dual_stack` appear in output telemetry.
- Renders clip using `media.rs` / FFmpeg.
- Probes output with `ffprobe`:
  - Video stream width = 1080, height = 1920.
  - Video duration matches audio duration with $|\Delta t| < 0.05\text{s}$.
  - Audio stream exists, sample rate $\ge 44100\text{Hz}$, channels $\ge 1$.
  - Subtitles rendered without errors.

- [ ] **Step 2: Run real render test**
Run: `& "d:\College\Autoshorts 6.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_dualframe_real_render.py`
Expected: PASS with measured A/V drift $< 0.05\text{s}$.

- [ ] **Step 3: Commit test script and verification output**
```bash
git add autoshorts/src-tauri/scripts/test_dualframe_real_render.py
git commit -m "test(dualframe): add real-video rendering and A/V duration synchronization test"
```

---

### Task 7: Full System Regression Verification & Final Report

**Files:**
- Create: `validation_report_dualframe.md`
- Verify: `d:/College/Autoshorts 5.0` untouched.

- [ ] **Step 1: Run all Python test suites**
- `test_dualframe_suite.py`
- `test_active_speaker_suite.py`
- `test_ultralytics_adapter_suite.py`
- `test_multimodal_hook_suite.py`
- `test_hook_closure_suite.py`

- [ ] **Step 2: Run all Rust tests**
- `cargo test --lib`

- [ ] **Step 3: Verify frontend build**
- `npm run build` in `autoshorts`

- [ ] **Step 4: Verify `d:/College/Autoshorts 5.0` is untouched**
Check git status / file timestamps in `d:/College/Autoshorts 5.0`.

- [ ] **Step 5: Document results in `validation_report_dualframe.md` and commit**
```bash
git add validation_report_dualframe.md
git commit -m "docs: add DualFrame verification report"
```
