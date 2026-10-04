# AutoShorts 6.0 — DualFrame Architecture Design

**Date:** 2026-09-06  
**Status:** Approved Architecture Draft  
**Scope:** `d:/College/Autoshorts 6.0` (`d:/College/Autoshorts 5.0` remains untouched)

---

## 1. Overview & Core Concept

### 1.1 The DualFrame Concept
In AutoShorts 5.0, when two people appear in the same camera shot, the system either forces both people into a single, uncomfortably wide 9:16 crop, or frames one person while cutting the other out.

AutoShorts 6.0 introduces **DualFrame**:
When **two relevant people** are simultaneously visible in the same camera shot and both remain valid visual subjects, AutoShorts does not force both people into one narrow crop. Instead:
1. Detects both subjects through the existing CV tracking pipeline.
2. Evaluates both subjects via the Visual Subject Resolver (VSR).
3. The **DualFrame Layout Resolver** determines that a dual layout is appropriate.
4. Generates an **independent crop trajectory** for Subject A and an **independent crop trajectory** for Subject B.
5. Places the left-origin subject in the **TOP** half of the 9:16 frame ($1080 \times 960$).
6. Places the right-origin subject in the **BOTTOM** half of the 9:16 frame ($1080 \times 960$).
7. Composes the two panels vertically into the full $1080 \times 1920$ output.
8. Automatically returns to the existing single-person framing behavior when only one relevant subject remains or when the scene cut warrants it.

```
SOURCE 16:9
+---------------------------------------+
|                                       |
|          PERSON A       PERSON B      |
|                                       |
+---------------------------------------+
                    ↓
DUALFRAME 9:16 (1080 x 1920)
+---------------------------------------+
|                                       |
|               PERSON A                |
|            (Top: 1080x960)            |
|                                       |
+---------------------------------------+
|                                       |
|               PERSON B                |
|          (Bottom: 1080x960)           |
|                                       |
+---------------------------------------+
                    ↓ (Person B exits or shot cut)
SINGLE 9:16 (1080 x 1920)
+---------------------------------------+
|                                       |
|               PERSON A                |
|         Existing normal crop          |
|                                       |
+---------------------------------------+
```

### 1.2 Critical Architectural Invariants
1. **Preserve Existing CV Pipeline**:
   - Ultralytics YOLO11n (`yolo11n.pt`) detection (`classes=[0]`).
   - BoT-SORT multi-object tracking with native `sparseOptFlow` GMC.
   - YuNet ONNX facial landmark & mouth motion analysis.
   - Visual Subject Resolver (VSR) multi-factor prominence scoring ($P$).
   - Existing single-mode framing solver and camera lock operator.
   - Diarization and active speaker analysis.
2. **Additional Layout Mode Only**:
   - DualFrame is an additive layout mode (`dual_stack`), not a replacement for single framing (`single`).
   - Existing single-mode behavior must remain 100% equivalent to AutoShorts 5.0 when DualFrame is inactive.
3. **Strict Scope Limit**:
   - Exactly **TWO** relevant subjects. No 3-panel or N-panel layouts are introduced.
4. **Single-Pass Captions**:
   - Kinetic ASS subtitles are burned onto the composed 9:16 canvas at the final step, never duplicated per panel.
5. **Continuous Audio**:
   - Audio is never cut, sliced, or re-segmented; continuous synchronization is maintained.

---

## 2. Layout Resolver & Temporal State Machine

### 2.1 Pipeline Flow
The Layout Resolver sits directly after VSR and before crop trajectory generation:
```
CV TRACKS (YOLO + BoT-SORT / Native + YuNet)
     ↓
VISUAL SUBJECT RESOLVER (VSR)
     ↓
DUALFRAME LAYOUT RESOLVER
     ├── SINGLE (1 eligible subject, 3+ subjects, or dual criteria unmet)
     │      ↓
     │   Standard 9:16 Single Framing Path
     │
     └── DUAL_STACK (exactly 2 eligible persistent subjects)
            ↓
         Independent Crop Trajectory A (9:8 aspect ratio)
         Independent Crop Trajectory B (9:8 aspect ratio)
            ↓
         Vertical Stack Composition (1080x1920)
```

The Layout Resolver decides:
> *"Should this temporal segment use one viewport (`single`) or two independent viewports (`dual_stack`)?"*

It does **not** decide who is speaking, and does **not** replace the VSR.

### 2.2 DualFrame Eligibility Criteria
A subject track is eligible for DualFrame if and only if:
1. **Track Persistence & Stability**: The track has accumulated stable detections within the current shot ($\ge 0.50\text{s}$ track age, low velocity jitter).
2. **VSR Relevance & Prominence**: The subject is confirmed relevant by the existing VSR (visual prominence score $P$ meets baseline prominence, face or head-corridor tracked, not a distant background passerby).
3. **Sufficient Visibility**: The subject is not currently occluded or off-screen, with valid upper-body / head geometry.
4. **Crop Feasibility**: The subject's center and bounds allow a valid 9:8 panel crop inside the source dimensions.

### 2.3 Anti-Flicker & Hysteresis State Machine
To eliminate flickering between `single` and `dual_stack`:
- **Dual-Entry Persistence**: Exactly two eligible subjects must remain continuously valid for `dual_entry_persistence_sec` ($\ge 0.60\text{s}$, or $\ge 5$ consecutive analysis frames).
- **Dual-Exit Persistence**: If one subject becomes temporarily occluded or missed, DualFrame does not immediately collapse to `single`. A grace buffer of `dual_exit_persistence_sec` ($0.75\text{s}$) tolerates brief dropouts. If the absence persists beyond $0.75\text{s}$, it cleanly transitions to `single`.
- **Minimum Segment Hold**: Any layout segment must maintain a minimum duration of `min_segment_dur_sec` ($1.00\text{s}$) before a non-cut layout change is permitted.
- **Camera Cut Reset Invariant**: At every camera scene cut (`is_shot_cut = True`), all DualFrame state, track associations, and slot assignments are **completely reset**. The new shot independently evaluates layout from scratch.

### 2.4 Slot Assignment & Stability (Left-to-Right → Top/Bottom)
When a transition to `dual_stack` is triggered:
1. **Spatial Ordering at Entry**:
   - The subject whose horizontal center $cx$ is further to the left in the source frame is assigned to the **TOP PANEL**.
   - The subject whose horizontal center $cx$ is further to the right is assigned to the **BOTTOM PANEL**.
2. **Persistent Track ID Locking**:
   - `locked_top_track_id = left_track.id`
   - `locked_bottom_track_id = right_track.id`
3. **Zero Slot-Flicker Rule**:
   - Once assigned, `locked_top_track_id` remains in the top panel and `locked_bottom_track_id` remains in the bottom panel for the entire duration of the `dual_stack` segment.
   - Panel slots are **never swapped** because of speaker turns, prominence variations, or temporary physical crossings.
   - Slots are re-evaluated only when DualFrame exits or when a camera cut occurs.

---

## 3. Panel Geometry & Independent Crop Trajectory

### 3.1 Geometry Invariants
- Final Canvas: $1080 \times 1920$ (9:16).
- Top Panel: $1080 \times 960$ (offset $y=0$ to $960$).
- Bottom Panel: $1080 \times 960$ (offset $y=960$ to $1920$).
- Fixed Panel Aspect Ratio:
  $$\frac{\text{crop\_w}}{\text{crop\_h}} = \frac{1080}{960} = \frac{9}{8} = 1.125$$

### 3.2 Dynamic Subject Framing & Scaling
1. **Zero Non-Uniform Stretching**:
   - All panel crops strictly enforce `crop_w = round(crop_h * 1.125)`.
   - When scaled to $1080 \times 960$ using `scale=1080:960:flags=lanczos`, aspect geometry is preserved with zero distortion.
2. **Independent Subject Scale**:
   - Crop scale is not hardcoded. The existing framing intelligence dynamically derives the ideal crop scale for each subject based on head size, bounding box, and eye-line requirements.
   - Top and bottom subjects may have independent crop scales.
3. **Independent Trajectory Generation**:
   - Subject A (Top): $x_{\text{top}}(t)$, $y_{\text{top}}(t)$, $w_{\text{top}}(t)$, $h_{\text{top}}(t)$.
   - Subject B (Bottom): $x_{\text{bottom}}(t)$, $y_{\text{bottom}}(t)$, $w_{\text{bottom}}(t)$, $h_{\text{bottom}}(t)$.
   - Both trajectories are clamped to legal source coordinates:
     $$0 \le x \le \text{source\_w} - \text{crop\_w}, \quad 0 \le y \le \text{source\_h} - \text{crop\_h}$$
   - Smoothed using the existing camera trajectory optimization and deadband logic.

---

## 4. Data Contract & Rust-Python Boundary

### 4.1 JSON Contract
`speaker_tracker.py` writes a structured JSON payload to `stdout`:
```json
{
  "mode": "single",
  "x": "min(max(0, ...), 760)",
  "y": "0",
  "w": "608",
  "h": "1080",
  "segments": [
    {
      "mode": "single",
      "start": 0.0,
      "end": 14.5,
      "crop": {
        "x": "...",
        "y": "...",
        "w": "608",
        "h": "1080"
      }
    },
    {
      "mode": "dual_stack",
      "start": 14.5,
      "end": 32.0,
      "top_track_id": 1,
      "bottom_track_id": 2,
      "top_crop": {
        "x": "...",
        "y": "...",
        "w": "900",
        "h": "800"
      },
      "bottom_crop": {
        "x": "...",
        "y": "...",
        "w": "900",
        "h": "800"
      }
    }
  ]
}
```

### 4.2 Rust Data Types (`media.rs`)
```rust
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct CropRectExpr {
    pub x: String,
    pub y: String,
    pub w: String,
    pub h: String,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum LayoutSegment {
    Single {
        start: f64,
        end: f64,
        crop: CropRectExpr,
    },
    DualStack {
        start: f64,
        end: f64,
        top_track_id: Option<i64>,
        bottom_track_id: Option<i64>,
        top_crop: CropRectExpr,
        bottom_crop: CropRectExpr,
    },
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SmartFramingPlan {
    pub mode: String, // "single" or "dual_stack"
    pub x: String,
    pub y: String,
    pub w: String,
    pub h: String,
    #[serde(default)]
    pub segments: Vec<LayoutSegment>,
}
```

---

## 5. Unified FFmpeg Filtergraph & Audio Synchronization

### 5.1 Filtergraph Construction
When `segments` contains at least one `dual_stack` segment:
For $N$ contiguous segments spanning $[0.0, T]$:

1. **Segment Filter Formulation**:
   - `Single` Segment $i$:
     ```text
     [0:v]trim=start=S_i:end=E_i,setpts=PTS-STARTPTS,crop=w=W_i:h=H_i:x='X_i':y='Y_i',scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int[v_seg_i];
     ```
   - `DualStack` Segment $i$:
     ```text
     [0:v]trim=start=S_i:end=E_i,setpts=PTS-STARTPTS,split=2[v_top_src_i][v_bot_src_i];
     [v_top_src_i]crop=w=W_top:h=H_top:x='X_top':y='Y_top',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int[v_top_i];
     [v_bot_src_i]crop=w=W_bot:h=H_bot:x='X_bot':y='Y_bot',scale=1080:960:flags=lanczos+accurate_rnd+full_chroma_int[v_bot_i];
     [v_top_i][v_bot_i]vstack=inputs=2[v_seg_i];
     ```
2. **Video Stream Concatenation**:
   ```text
   [v_seg_0][v_seg_1]...[v_seg_{N-1}]concat=n=N:v=1:a=0[v_composed];
   ```
3. **Subtitles Overlay**:
   ```text
   [v_composed]subtitles='escaped_path'[v_final]
   ```
4. **Audio Verification**:
   - Source is cut via `-ss start_sec -t dur_sec -i source_path`.
   - Video stream is mapped to `[v_final]`, audio stream mapped to `0:a`.
   - Audio is encoded once via `-c:a aac -b:a 192k`.
   - The implementation will explicitly verify that total video PTS matches total audio duration, with zero A/V drift across segment transitions.

---

## 6. Configuration & Telemetry

### 6.1 Configuration (`speaker_tracker.py`)
```python
@dataclass(frozen=True)
class DualFrameConfig:
    enabled: bool = True
    dual_entry_persistence_sec: float = 0.60
    dual_exit_persistence_sec: float = 0.75
    min_segment_dur_sec: float = 1.00
    min_track_age_sec: float = 0.50
    max_lost_sec: float = 0.60
    panel_aspect_ratio: float = 9.0 / 8.0  # 1.125
```

- **Environment Override**:
  - `AUTOSHORTS_DUALFRAME_ENABLED="false"` cleanly disables DualFrame, returning 100% to single-mode framing. Default is `"true"`.

### 6.2 Telemetry Protocol
Structured log lines emitted to `stderr` by Python, forwarded by Rust `media.rs` to terminal `stdout`:
```text
[DualFrame] mode=single
[DualFrame] mode=dual_stack
[DualFrame] enter t=14.50 top_track=1 bottom_track=2
[DualFrame] exit t=32.00 reason=subject_lost
[DualFrame] shot_reset t=45.00
[DualFrame] layout_plan total_segments=3 dual_segments=1
[DualFrame Render] Executing layout plan with 3 segments (1 dual-stack, 2 single)
```

---

## 7. Verification Plan

1. **Automated Unit & Integration Tests**:
   - DualFrame entry on 2 persistent subjects.
   - Single fallback on 1 subject.
   - 3+ subjects fallback to single mode.
   - Occlusion tolerance before exit.
   - Slot stability (zero top/bottom swapping during speaker turn or motion).
   - Zero aspect ratio distortion (exact 9:8 crop and 1080x960 scaling).
   - Clean reset across shot cuts.
   - `AUTOSHORTS_DUALFRAME_ENABLED=false` complete bypass verification.
2. **Real-Video Render & Synchronization Verification**:
   - Render a real test clip with a `SINGLE → DUAL_STACK → SINGLE` transition.
   - Measure audio vs. video duration using `ffprobe` to prove zero A/V drift ($|\Delta t| < 0.05\text{s}$).
   - Verify subtitle alignment on the composed 9:16 canvas.
   - A/B comparison against AutoShorts 5.0 output.
