# AutoShorts 11.0 — AutoClip Presentation Mode Architecture Contract
**Document Version:** 1.0.0  
**Date:** 2026-10-12  
**Status:** Approved Architectural Contract (Read-Only Planning Phase)  
**Target Delivery File:** `docs/reports/AUTOCLIP_PRESENTATION_MODE_ARCHITECTURE_CONTRACT_2026-10-12.md`  

---

## Executive Summary & Feasibility Verdict

### Feasibility Verdict: `INTEGRABLE WITH MATERIAL ADAPTERS`

AutoClip's presentation engine can be faithfully and robustly integrated into AutoShorts 11.0 as **Presentation Mode 2** without compromising or rewriting AutoShorts' existing **Presentation Mode 1**. 

The two presentation systems possess fundamentally different design philosophies, heuristics, and visual expressions:
- **Mode 1 (AutoShorts Native):** Centered on BoT-SORT continuous tracking, YOLOv8/11 face detection, dynamic scale deadbands, Camera Lock stabilization, vertical 1080×1920 or 1080×1080 square Adaptive Framing with shared two-person retention, and AutoShorts kinetic ASS caption presets.
- **Mode 2 (AutoClip Studio):** Centered on OpenCV YuNet face detection, FFmpeg scene-change hard-cut detection, optical-flow mouth motion frame-pair analysis, piecewise-constant hold framing with speaker-switching during long shots, full-frame blurred-backdrop fallback (`mode: "fit"`), CJK/Latin width-aware 2-line maximum subtitle formatting, versioned PIL TitleArt (Comic, Neon, Arena, Editorial, Pixel, Frosted), and template packaging (`interview_zh`, `podcast_en`, and optional Playwright HTML overlay templates `editorial`/`street`).

This architectural contract defines the **strict isolation boundaries, data adapter specifications, UI workflow, and execution contracts** necessary to integrate Mode 2 cleanly. Under this contract, **zero presentation logic is shared between the two modes**.

---

## 1. Verified Repository, Source Archives & Git Information

### 1.1 Source Locations
- **AutoShorts 11.0 Workspace:** `D:\College\Autoshorts 11.0`
  - Active Branch: `feature/autoclip-scoring-topn`
  - Current HEAD Commit: `33791fe784eb8132ca4d88e0db433ee5bb5746ee` (clean working tree following Desktop UI Redesign & Adaptive Framing Audit).
- **AutoClip Source Archive:** `D:\College\autoclip.zip`
  - Size: 130,393,164 bytes (Verified present on disk).
  - Last Modified: `2026-10-12 13:30`.
- **Extracted AutoClip Codebase:** `D:\College\autoclip\autoclip`
  - Backend Root: `D:\College\autoclip\autoclip\backend`
  - Services Root: `D:\College\autoclip\autoclip\backend\services`
  - Studio Services: `D:\College\autoclip\autoclip\backend\services\studio`
  - Asset Repository: `D:\College\autoclip\autoclip\backend\assets` (models, fonts, templates, outro clips).

### 1.2 Safety & Integrity Confirmation
- No application source code, dependency manifests, configuration files, or database tables were modified during this inspection.
- Zero billable API calls (Deepgram, DeepSeek, NVIDIA NIM, OpenRouter, OpenAI) were made.
- All conclusions are derived from direct source code analysis, deterministic test cases, and file-system assets.

---

## 2. Non-Negotiable Architectural Decision: Two Independent Presentation Modes

AutoShorts 11.0 must support two distinct, parallel presentation engines selected by an explicit user preference.

```
                           ┌──────────────────────────────────────────────┐
                           │      AutoShorts Candidate Discovery &        │
                           │           Selection (Shared Engine)          │
                           └──────────────────────┬───────────────────────┘
                                                  │
                                   [Selected Presentation Mode]
                                                  │
                   ┌──────────────────────────────┴──────────────────────────────┐
                   ▼                                                             ▼
     ┌───────────────────────────┐                                 ┌───────────────────────────┐
     │  Mode 1: AutoShorts Native │                                 │  Mode 2: AutoClip Studio  │
     ├───────────────────────────┤                                 ├───────────────────────────┤
     │ • YOLOv8/11 + BoT-SORT    │                                 │ • OpenCV YuNet Face Net   │
     │ • ReID Feature Tracking   │                                 │ • FFmpeg Scene Cut Slices │
     │ • Camera Lock Operator    │                                 │ • Mouth Motion Pair Diff  │
     │ • Adaptive Framing 1:1    │                                 │ • Hold-Segment Framing    │
     │   (Shared Two-Shot Mid)   │                                 │ • Blurred Fit Fallback    │
     │ • Original 9:16 Framing   │                                 │ • PlatformStrategy Specs  │
     │ • DualFrame / DualStack   │                                 │ • CJK-Clause 2-Line Sub   │
     │ • AutoShorts Caption ASS  │                                 │ • PIL Multi-Layer TitleArt│
     │   Presets (MrBeast, etc.) │                                 │ • Classic / HTML Packaging│
     └───────────────────────────┘                                 └───────────────────────────┘
```

### 2.1 Mode 1 — AutoShorts Presentation
Mode 1 preserves all pre-existing AutoShorts presentation behaviors untouched:
1. Face detection via `ultralytics` YOLO models.
2. Temporal tracking and re-identification via BoT-SORT and TorchReID.
3. Framing strategies:
   - `adaptive`: 1080×1080 square inner canvas centered at `y=420` in a 1080×1920 portrait frame with shared two-person retention centered on the pair midpoint. DualFrame is suppressed.
   - `original`: 1080×1920 portrait crop with active-speaker following and solo speaker isolation. DualFrame/DualStack enabled if configured.
4. Camera motion stabilization via Camera Lock Operator and Pacing GMC pan decoupling.
5. Caption generation via AutoShorts' internal caption template intelligence and preset library (`preset_minimal_capsule`, `preset_viral_bold`, `preset_mrbeast_pop`, `preset_t7`, etc.).

### 2.2 Mode 2 — AutoClip Presentation
Mode 2 faithfully executes AutoClip's native presentation pipeline without injecting AutoShorts heuristics:
1. **Face Detection & Tracking:** OpenCV YuNet ONNX detector (`cv2.FaceDetectorYN`) bundled in `assets/models/face_detection_yunet_2023mar.onnx`.
2. **Shot Segmentation:** Hard cuts identified via FFmpeg scene filter (`select='gt(scene,0.3)'`). Transitions closer than 0.4s collapsed (`split_shots`).
3. **Active Speaker Identification:** Frame-pair sampling with `PAIR_GAP = 0.25s` evaluating mouth-region pixel delta (`abs(grey_a - grey_b).mean() > 4.0`).
4. **Speaker Following & Handoffs:** Hold segments (`min_hold = 2` samples, `min_jump = 0.12`). In long shots ($> 6.0\text{s}$), the crop window switches to follow a new speaker once they persist.
5. **Blurred Fit Fallback:** Shots without detectable faces (slides, text cards, screen recordings) automatically use `mode: "fit"`, rendering the complete frame over an 8× downscaled blurred backdrop (`blurred_backdrop`).
6. **Subtitles & Text Wrapping:** CJK and Latin width-aware clause breaking (`caption_layout.py`), guaranteeing $\le 2$ lines per screen with ASS `\N` breaks. Styles: `clean`, `bold`, `box` (translucent dark box card), and `accent` (bold yellow).
7. **Title Artwork:** PIL ImageDraw-rendered multi-layer title art (`title_art.py`: `comic`, `neon`, `arena`, `editorial`, `pixel`, `frosted`) with rotation, drop shadows, and motion overlays.
8. **Packaging:** Template packaging via ASS (`interview_zh`, `podcast_en`) or Playwright HTML overlay (`editorial`, `street`).

### 2.3 Strict Isolation Rules
- **No Hybrid Heuristics:** AutoShorts Adaptive framing (two-person retention) must **never** be applied to Mode 2. AutoClip's native speaker-following and blurred-fit logic must run as implemented.
- **No DualFrame in Mode 2:** Mode 2 does not use AutoShorts' DualFrame or DualStack split-screen engine.
- **No Shared Caption Logic:** AutoShorts caption presets must not govern Mode 2; AutoClip's `caption_layout.py` and `SUBTITLE_STYLES` must not govern Mode 1.
- **No Silent Fallbacks:** If Mode 2 encounters a missing dependency (e.g., missing YuNet ONNX model or uninstalled runtime), it must report an explicit error or use AutoClip's internal fallback (e.g. downgrading HTML templates to Classic ASS). It must **never** silently fall back to Mode 1 rendering.

---

## 3. Separation Between Selection and Presentation

To prevent architectural coupling, AutoShorts strictly decouples **Candidate Selection** from **Presentation Rendering**:

```
[Audio/Video Source] ──► [Transcription] ──► [Candidate Discovery] ──► [Scoring Authority]
                                                                                │
                                                                   [Selected Candidates]
                                                                                │
                                                    ┌───────────────────────────┴───────────────────────────┐
                                                    ▼                                                       ▼
                                          [Mode 1: AutoShorts]                                    [Mode 2: AutoClip]
                                          • Framing Mode: adaptive / original                     • Platform: douyin / tiktok / etc.
                                          • Caption Preset: viral_bold / etc.                     • Subtitle: clean / bold / box / accent
                                          • DualFrame: true / false                               • Title Art: comic / neon / etc.
                                          • Render: pacing.rs FFmpeg chain                        • Packaging: classic / editorial
                                                                                                  • Render: AutoClip FFmpeg chain
```

### Invariants:
1. **Scoring Backend Independence:** Selection of Mode 2 does **not** change the scoring backend from Legacy to AutoClip holistic scoring, nor does selecting AutoClip scoring change the presentation mode. They are independent settings.
2. **Timestamp Invariance:** Changing presentation mode does not alter candidate start/end timestamps or selection ranges.
3. **No Redundant Processing:** Changing presentation mode or restyling a clip does not trigger transcription, candidate re-discovery, or re-scoring when cached data is available.
4. **Default Backward Compatibility:** Existing AutoShorts projects that do not specify a presentation mode default to Mode 1.

---

## 4. User Workflow and UI State Transitions

The choice of presentation mode takes place at two defined touchpoints in the AutoShorts UI:

### 4.1 Touchpoint A: Initial Setup Modal (Screen C — On Import)
When the user imports a video file, the **Configure & Start Processing** setup modal (`showStyleModal` in `main.tsx`) provides three distinct configuration sections:
1. **Section 1: Presentation Engine**
   - Option A: `AutoShorts Native (Mode 1)` — "Kinetic adaptive framing and dynamic subtitle presets"
   - Option B: `AutoClip Studio (Mode 2)` — "Platform packaging, YuNet speaker-following, and title art"
2. **Section 2: Engine-Specific Styling**
   - *When Mode 1 is active:*
     - Framing: Adaptive Framing (1:1 Smart) vs. Original 9:16
     - Caption Presets: Clean Minimal, Viral Bold, Word Highlight, MrBeast Pop, T7 Dynamic
   - *When Mode 2 is active:*
     - Destination Platform: TikTok (9:16 Crop), YouTube Shorts (9:16 Crop), Douyin (9:16 Blur), Bilibili (16:9 Fit), Xiaohongshu (9:16 Blur Box)
     - Subtitle Style: Clean, Bold, Box (Dark Card), Accent (Yellow)
     - Title Artwork: Comic Punch, Neon Glow, Arena Slanted, Editorial Bold, Frosted Glass, None
     - Packaging Template: Classic ASS (`interview_zh`, `podcast_en`) vs HTML Overlay (`editorial`, `street`)
3. **Section 3: Scoring Authority**
   - Legacy Heuristic Scoring vs. AutoClip Holistic Scoring (NVIDIA NIM)

### 4.2 Touchpoint B: Style Studio Drawer (Screen D & E — Restyling)
The slide-in Style Studio drawer exposes the same isolated control groups for rendered clips, permitting re-rendering with new presentation settings without re-running transcription or scoring.

---

## 5. AutoClip Presentation Dependency Inventory

Detailed source inspection of `D:\College\autoclip\autoclip` yields the complete inventory of AutoClip's presentation components:

### 5.1 Framing Engine (`backend/services/studio/framing.py`)
- **Detector Model:** OpenCV `FaceDetectorYN` loading `backend/assets/models/face_detection_yunet_2023mar.onnx` (232 KB, Apache-2.0 / MIT).
- **Python Dependencies:** `opencv-python-headless>=4.10`, `numpy`.
- **Cuts & Sampling:**
  - `detect_cuts`: Runs FFmpeg with `select='gt(scene,0.3)'` on a 320px downscaled stream (`showinfo`).
  - `split_shots`: Enforces `MIN_SHOT = 0.4s` and `MAX_SHOTS = 120`.
  - `sample_offsets`: Samples clear of transitions (`CUT_MARGIN = 0.3s`, `SAMPLE_INTERVAL = 2.5s`).
  - `_grab_pair`: Grabs 2 frames spaced by `PAIR_GAP = 0.25s` scaled to `FRAME_WIDTH = 480`.
- **Speaker Heuristic (`_speaker_center`):**
  - YuNet detects faces with confidence threshold 0.6. Faces smaller than `MIN_FACE = 0.035` of frame width are discarded.
  - Mouth corners (landmarks index 10-13) define mouth bounding box.
  - Pixel difference between frame 1 and frame 2 in mouth box:
    $$\Delta_{\text{mouth}} = \text{mean}\left(\left| I_1[y, x] - I_2[y, x] \right|\right)$$
  - Face with highest motion $> 4.0$ is designated the speaker. If all $\le 4.0$, the largest face by area is chosen.
- **Hold Segments (`segment_track`):**
  - `min_hold = 2` consecutive samples required before moving the window.
  - `min_jump = 0.12` normalized distance threshold.
- **Rendering Graph (`layout_filter`):**
  - `crop`: Cover scale + `crop={w}:{h}:x='(iw-ow)*{expr}':y=(ih-oh)/2`.
  - `fit`: 3-way split applying `blurred_backdrop(w, h)` in background with fitted source overlaid.

### 5.2 Output Formats & Strategy Profiles (`backend/services/platform_strategy.py`)
AutoClip establishes explicit platform presets:

| Strategy ID | Label | Output Dims | Layout | Subtitle Look | Title Style | Default Template |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `douyin` | 抖音 9:16 | 1080×1920 | `blur` | `accent` | `comic` | `interview_zh` |
| `tiktok` | TikTok 9:16 | 1080×1920 | `crop` | `bold` | `comic` | `podcast_en` |
| `instagram_reels` | Instagram Reels | 1080×1920 | `crop` | `bold` | `card` | `podcast_en` |
| `youtube_shorts` | YouTube Shorts | 1080×1920 | `crop` | `bold` | `card` | `podcast_en` |
| `bilibili` | B站横屏 | 1920×1080 | `fit` | `clean` | `editorial` | `landscape` |
| `xiaohongshu` | 小红书 9:16 | 1080×1920 | `blur` | `box` | `card` | `interview_zh` |
| `original` | 原画重编码 | Source | `none` | `clean` | `plain` | `none` |

### 5.3 Caption & Subtitle System (`caption_layout.py`, `publish_export.py`, `burned_subtitles.py`)
- **CJK/Latin Typography (`caption_layout.py`):**
  - Character width: CJK = 1.0, Latin = 0.55.
  - Clause splitting: Splits at punctuation (`，。！？!?；;、,：:`).
  - Joint ranking: Breaks before particles (`和与及在把被就而但...`) or after (`的了着过地得们...`).
  - Max lines: Strictly $\le 2$ lines per screen (`split_screens`, `timed_screens`).
- **Subtitle Styles (`publish_export.py`):**
  - `clean`: White text, 2px dark outline, 48px margin.
  - `bold`: Large bold white, 3px dark outline, 1px drop shadow, 56px margin.
  - `box`: White text enclosed in translucent dark rectangular card (`BackColour=&H88000000`, `BorderStyle=3`, `Outline=6`).
  - `accent`: Bold yellow (`PrimaryColour=&H0000E5FF`), 3px dark outline.
  - Scaling: Scaled by `factor = 0.47` for portrait 9:16 (`portrait_subtitle_style`) to account for libass 288px virtual canvas.
- **Burned Subtitle Detection (`burned_subtitles.py`):**
  - Samples 12 frames across bottom 35% of frame. Detects outlined text glyphs.
  - Computes `caption_band(masks)`: returns `(top, bottom)` fraction.
  - When present, `render.py:_mask_captions` inserts an FFmpeg `gblur=sigma=40` band over the old subtitles prior to laying out the new composition.

### 5.4 Title Artwork & Templates (`title_art.py`, `packaging_html.py`)
- **PIL Title Art (`title_art.py`):**
  - Renders PNG artwork directly via `Pillow` (PIL) using bundled fonts (`NotoSansSC.ttf`, `BarlowCondensed-Black.ttf`, `Anton-Regular.ttf`, `PressStart2P-Regular.ttf`).
  - Implements multi-line polygon bounding plates with skew angle, drop shadows, and motion overlays (`title_art.overlay_motion`).
- **Template Packaging:**
  - Classic ASS: Generates ASS events for titles, kicker, nameplates, and highlight pills (`packaging_render.py`).
  - HTML Overlay (`editorial`, `street`): Uses Playwright to render HTML/CSS templates in headless browser, encoding an `ffv1` intermediate video with alpha channel (`packaging_html.py`).
  - Automatic Downgrade: If Playwright/Chromium is missing, AutoClip automatically falls back to `classic` ASS without aborting the export.

### 5.5 Hardware Video Encoder Probing (`video_encoder.py`)
- Probes hardware encoders using a 0.3s test encode:
  - macOS: `h264_videotoolbox`
  - Windows: `h264_nvenc`, `h264_qsv`, `h264_amf`
  - Fallback: `libx264` (`-preset veryfast -crf 20`)

### 5.6 Asset Inventory & Licensing
- `backend/assets/models/face_detection_yunet_2023mar.onnx` (232 KB) — Apache-2.0 / MIT License.
- `backend/assets/fonts/NotoSansSC.ttf` (17.7 MB) — SIL Open Font License 1.1.
- `backend/assets/fonts/Anton-Regular.ttf` (170 KB) — SIL OFL 1.1.
- `backend/assets/fonts/BarlowCondensed-Black.ttf` (110 KB) — SIL OFL 1.1.
- `backend/assets/fonts/PressStart2P-Regular.ttf` (118 KB) — SIL OFL 1.1.
- AutoClip source code: MIT License.

---

## 6. Proposed Integration Architecture

AutoShorts will integrate AutoClip Presentation as a clean sidecar module, reusing AutoClip's Python presentation services via a dedicated adapter.

### 6.1 Component Ownership Table

| Component | Architecture Role | Owner / Source Code | Shared Logic? |
|---|---|---|:---:|
| **Candidate Discovery & Top-N** | Project Selection | AutoShorts (`lib.rs`, `scoring.rs`) | **NO** (AutoShorts Only) |
| **Project Storage (SQLite)** | Data Store | AutoShorts (`db.rs`) | **NO** (AutoShorts Only) |
| **Presentation Mode Dispatcher** | Gateway / Switch | AutoShorts (`media.rs`, `render.rs`) | **Data Only** (Dispatches by mode) |
| **AutoShorts Native Renderer (Mode 1)** | Presentation Engine | AutoShorts (`speaker_tracker.py`, `pacing.rs`) | **NO** (Mode 1 Only) |
| **AutoClip Candidate Adapter** | Data Translation | New Adapter (`autoclip_adapter.py`) | **Data Only** (Maps models) |
| **YuNet Framing & Scene Cuts** | Presentation Engine | AutoClip (`framing.py`) | **NO** (Mode 2 Only) |
| **Platform Strategy Profiles** | Presentation Engine | AutoClip (`platform_strategy.py`) | **NO** (Mode 2 Only) |
| **Caption Layout & Wrapping** | Presentation Engine | AutoClip (`caption_layout.py`) | **NO** (Mode 2 Only) |
| **PIL Title Artwork** | Presentation Engine | AutoClip (`title_art.py`, fonts) | **NO** (Mode 2 Only) |
| **AutoClip FFmpeg Renderer (Mode 2)** | Presentation Engine | AutoClip (`render.py`, `publish_export.py`) | **NO** (Mode 2 Only) |

### 6.2 Data Adapter Specification (`autoclip_adapter.py`)
To feed AutoShorts candidate data into AutoClip's renderer without altering AutoClip's logic:

1. **Input from AutoShorts:**
   - `project_id`: string.
   - `video_path`: absolute path to source video.
   - `candidate`: object containing `id`, `start` (sec), `end` (sec), `hook_headline`, `transcript_words`.
   - `config`: Mode 2 settings (`platform_strategy`, `subtitle_style`, `title_style`, `template`).
2. **Adapter Transformation:**
   - Constructs AutoClip `Scene(id=f"scene_{candidate.id}", start=candidate.start, end=candidate.end)`.
   - Constructs AutoClip `Draft(id=candidate.id, title=candidate.hook_headline, scenes=[scene], ...)` matching `models.Draft`.
   - Converts AutoShorts `TranscriptWord` list into AutoClip SRT rows for the slice duration.
3. **Execution Call:**
   - Invokes AutoClip's `auto_frame(video_path, draft, source_w, source_h)` to compute shot-aligned crop tracks.
   - Invokes AutoClip's `render_draft(project_id, video_path, draft, job_id, progress_callback)` to execute FFmpeg rendering.
4. **Output Return:**
   - Returns rendered MP4 path, dimensions, and render telemetry back to AutoShorts database and UI.

---

## 7. Integration Conflicts, Blockers & Compatibility Matrix

| Issue / Dependency | Classification | Source Citation | Resolution & Architectural Guardrail |
|---|:---:|---|---|
| **Python Environment & Dependencies** | `RUNTIME DEPENDENCY REQUIRED` | `framing.py:34`, `pyproject.toml` | AutoShorts' Python venv must include `opencv-python-headless>=4.10`, `pillow>=10.0`, `pydantic>=2.0`. |
| **OpenCV YuNet ONNX Model** | `COMPATIBLE` | `framing.py:35`, `backend/assets/models/` | Model file (232 KB) must be placed in AutoShorts sidecar asset path. |
| **OpenCV / NumPy Version Clashes** | `COMPATIBLE` | `framing.py:63-75` | AutoClip includes built-in `ensure_on_path()` module isolation guard to prevent PyAV/NumPy corruption. |
| **Hard Scene-Cut Detection** | `COMPATIBLE` | `framing.py:276` | FFmpeg `gt(scene,0.3)` filter is natively supported by AutoShorts' bundled FFmpeg binary. |
| **FFmpeg Filter Availability** | `COMPATIBLE` | `publish_export.py:166` | Uses standard `gblur`, `scale`, `crop`, `overlay`, and `subtitles` (libass). Verified present in FFmpeg 6.x+. |
| **CJK & Special Fonts** | `COMPATIBLE` | `backend/assets/fonts/` | Bundled fonts (`NotoSansSC.ttf`, `Anton-Regular.ttf`, `BarlowCondensed`) must be packaged into the sidecar assets. |
| **Playwright HTML Overlay Engine** | `ADAPTER REQUIRED` | `packaging_runtime.py:35` | Playwright is large (140 MB). Handled gracefully: Mode 2 defaults to Classic ASS; HTML templates are optional and downgrade automatically if runtime is missing. |
| **Database & Project Store** | `ADAPTER REQUIRED` | `store.py:40`, `autoshorts/src-tauri/src/db.rs` | AutoShorts uses SQLite; AutoClip uses `studio.json`. Adapter will generate in-memory or transient project context without replacing AutoShorts' SQLite DB. |
| **Hardware Video Encoders** | `COMPATIBLE` | `video_encoder.py:40` | AutoClip's probing logic tests NVENC/QSV/AMF/VideoToolbox and falls back to `libx264` automatically. |
| **LLM Translation / Packaging** | `ADAPTER REQUIRED` | `render.py:102`, `packaging.py` | AutoClip's translation prompt in `render.py` must be disabled or bypassed when no external API key is set; deterministic fallback `recover_empty_packaging` already exists. |
| **Licensing & Copyright** | `COMPATIBLE` | `LICENSE`, `*-OFL.txt` | AutoClip MIT code and OFL fonts can be distributed with AutoShorts provided copyright notices are retained in legal credits. |

---

## 8. Acceptance Contract for Future Implementation

A subsequent implementation task must satisfy the following criteria:

### 8.1 Offline Test Criteria (Zero Renders, Zero Billable APIs)
1. **Model Validation:** AutoClip `Draft`, `Scene`, `CropPoint`, and `Packaging` models validate and serialize correctly under Pydantic.
2. **Crop Geometry Math:** `framing.window_fraction` and `crop_x_for_center` match exact AutoClip unit test assertions ([`test_studio_framing.py#L21-L33`](file:///D:/College/autoclip/autoclip/backend/tests/test_studio_framing.py#L21-L33)).
3. **Hold-Segment Segmentation:** `framing.segment_track` correctly filters single glances and holds positions across turns ([`test_studio_framing.py#L35-L42`](file:///D:/College/autoclip/autoclip/backend/tests/test_studio_framing.py#L35-L42)).
4. **Nested Crop Expression:** `framing.crop_expression` generates nested FFmpeg `if(lt(t,...))` expressions correctly.
5. **CJK Caption Layout:** `caption_layout.timed_screens` breaks lines at punctuation/particles, never exceeds 2 lines per screen, and computes correct proportional durations.
6. **Detector Loading:** OpenCV FaceDetectorYN successfully initializes with `face_detection_yunet_2023mar.onnx` offline.
7. **Scoring Independence:** Tests verify that selecting Mode 2 does not mutate `scoring_backend` in the project state.

### 8.2 Local-Render Acceptance Criteria (Using Existing Offline Test Media)
1. **Mode 1 Invariance:** Renders of existing test clips under Mode 1 produce byte-for-byte or visually identical output with Adaptive 1080×1080 framing and AutoShorts captions.
2. **Mode 2 Blurred Fit on Non-Face Media:** Renders of slides/cards in Mode 2 produce 9:16 portrait video with fitted source centered over 8× blurred backdrop.
3. **Mode 2 Speaker Following on Two-Shots:** In long two-person scenes ($>6\text{s}$), Mode 2 follows the active speaker with hold intervals, contrasting with Mode 1's stationary pair-midpoint anchor.
4. **Mode 2 Box Captions:** Renders with `subtitle_style="box"` produce white text over translucent dark rectangular boxes.
5. **Mode 2 Title Art:** Renders with `title_style="comic"` produce multi-layer rotated title artwork with drop shadows.
6. **Hardware Encoder Fallback:** Simulating hardware encoder failure cleanly executes `libx264` fallback without crashing the render job.

---

## 9. Numbered Implementation Roadmap (Future Authorized Task)

When implementation is authorized, execution must proceed in the following strict order:

1. **Phase 1: Environment & Asset Staging**
   - Copy `face_detection_yunet_2023mar.onnx` and required fonts (`NotoSansSC.ttf`, `Anton-Regular.ttf`, `BarlowCondensed-*.ttf`) into AutoShorts sidecar asset directory.
   - Verify `opencv-python-headless`, `pillow`, `pydantic` in Python environment.
2. **Phase 2: AutoClip Presentation Sidecar Integration**
   - Place AutoClip presentation modules (`framing.py`, `platform_strategy.py`, `caption_layout.py`, `title_art.py`, `render.py`, `publish_export.py`, `video_encoder.py`) into an isolated sidecar package `autoshorts/src-tauri/scripts/autoclip_studio/`.
3. **Phase 3: Data Adapter Implementation**
   - Implement `autoclip_adapter.py` translating AutoShorts candidate data and SQLite transcripts into AutoClip `Draft` and `Scene` objects.
4. **Phase 4: Rust Backend IPC & Presentation Dispatcher**
   - Add `presentation_mode` and `presentation_config` fields to AutoShorts database schema and Tauri IPC commands.
   - Route render requests in `media.rs` to Mode 1 (`speaker_tracker.py`) or Mode 2 (`autoclip_adapter.py`).
5. **Phase 5: Frontend UI Wiring**
   - Update Initial Setup Modal (`Screen C`) and Style Studio Drawer (`Screen D/E`) to support Mode 1 vs Mode 2 selection with isolated option panels.
6. **Phase 6: Offline & Local Render Verification**
   - Execute test suites against both presentation modes using existing local test media.

---

## 10. Explicit Unverified Items

1. **HTML Playwright Overlay Templates on Windows ARM:** AutoClip's HTML overlay rendering is verified for x86_64 and Apple Silicon; Windows ARM performance remains unverified and will automatically downgrade to Classic ASS.
2. **Bilingual Caption Alignment on Non-English/Non-Chinese Languages:** AutoClip's `_needs_translation` and CJK particle rules are optimized for Chinese and English. Japanese and other languages use Latin width approximations (0.55).
