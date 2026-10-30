# Comprehensive Technical Audit: AutoClip Presentation Mode 2 Styling Architecture & Integration State
**Document Version:** 1.0  
**Audit Date:** 2026-10-13  
**Auditor:** AutoShorts Deep Research Subagent  
**Scope:** AutoClip Presentation Mode 2 (Style Studio, Data Translation Adapter, Isolated Sidecar Studio Pipeline)  
**Primary Repositories Audited:**  
- Integrated Workspace: `D:\College\Autoshorts 11.0`  
- Upstream AutoClip Reference: `D:\College\autoclip\autoclip`  

---

## 1. Executive Summary & Verification State

### 1.1 Git State & Working Copy Integrity
- **Branch:** `feature/autoclip-scoring-topn` [SOURCE VERIFIED]
- **HEAD Commit:** `33791fee3a3ffca540173150425520a5f9e8195a` [SOURCE VERIFIED]
- **Working Tree Status:** Clean with respect to audit execution (no code changes, branch switches, or unstaged modifications introduced during this audit). Existing working-tree modifications in `media.rs`, `lib.rs`, `main.tsx`, and `autoclip_adapter.py` reflect current implementation progress. [SOURCE VERIFIED]

### 1.2 Core Audit Discoveries
1. **The Fundamental Architectural Duality:** Upstream AutoClip has two distinct rendering paradigms:
   - **Mode A: Automated Packaging Templates** (`interview_zh`, `podcast_en`, and HTML templates `editorial`, `street`). In this mode, the template takes 100% control over visual hierarchy, typography, colors, canvas layout, lower-third tags, and subtitles.
   - **Mode B: Standalone / Unpackaged Rendering** (`template: "none"`). In this mode, user-selected `subtitle_style` (`clean`, `bold`, `box`, `accent`), PIL-based `title_style` (`comic`, `neon`, `arena`, `editorial`, `pixel`, `frosted`), and `title_motion` are rendered via FFmpeg overlay filters.
2. **The Critical UI Shadowing Defect:** The current AutoShorts Style Studio UI (`main.tsx`) exposes both Mode A controls and Mode B controls simultaneously in a single panel, but defaults `template` to `"interview_zh"` and provides **no option** to select `"none"` / Standalone.
   - Because `template` is always either `"interview_zh"` or `"podcast_en"`, `autoclip_adapter.py` always constructs a `Packaging` model (`draft.packaging is not None`).
   - In `autoclip_studio/render.py` (lines 132-265), whenever `draft.packaging is not None`, the renderer sets `hook = ""` and takes the `elif packaged:` branch.
   - **Consequently, the selected Subtitle Look (`clean`, `bold`, `box`, `accent`), Title Art Style (`comic`, `neon`, `arena`, etc.), and Title Motion toggle are completely shadowed and never rendered into the output video.**
3. **The 8 Gallery Examples Breakdown:** The gallery labels "Interview Style", "Full Screen Podcast", and "Original Aspect Ratio" are **not** independent platform strategies; they represent specific combinations of `PlatformStrategy` + `Packaging Template` + `Framing Layout`:
   - Only 2 of the 8 examples (`MrBeast` and `Andrew Garfield × Amy Poehler`) can be 100% reproduced through the current UI.
   - 4 examples (`Stallone`, `Gates`, `AI Experts Debate`, `Tony Robbins`) are partially reproduced because the adapter does not set `layout: 'window'` for `interview_zh` or `layout: 'crop'` for portrait podcast.
   - 2 examples (`Tong Liya` and `Sam Altman`) are distorted because the UI forces vertical packaging templates onto 16:9 landscape Bilibili output.

---

## 2. Definitive Architectural Taxonomy: Platform Strategy vs Layout vs Subtitle vs Title Art vs Title Motion

Understanding Presentation Mode 2 requires separating 5 distinct technical layers that were conflated during initial integration:

| Layer | Defined In | Primary Types / IDs | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Platform Strategy** | `platform_strategy.py` | `douyin`, `tiktok`, `youtube_shorts`, `instagram_reels`, `bilibili`, `youtube_long`, `xiaohongshu`, `original` | Destination container spec: Canvas dimensions (w, h), duration caps, aspect ratio (portrait/landscape/original), default framing layout, audience language (`zh`/`en`), default packaging template. |
| **2. Layout / Framing Mode** | `packaging_render.py`, `framing.py` | `window`, `crop`, `blur`, `fit`, `none` | Geometric placement of video on canvas: `window` (4:3 speaker window at y=560), `crop` (9:16 vertical crop tracking faces), `blur` (9:16 blurred backdrop), `fit` (black padding), `none` (passthrough). |
| **3. Packaging Template** | `models.py`, `packaging_render.py` | `interview_zh`, `podcast_en`, `landscape`, `none` (HTML: `editorial`, `street`) | Comprehensive show packaging: Pinned top titles, timed subtitle cards or word-pop karaoke, lower-third speaker nameplates, editorial quote tags, progress line. |
| **4. Subtitle Look** | `publish_export.py` (`SUBTITLE_STYLES`) | `clean`, `bold`, `box`, `accent` | Standalone SRT subtitle typography via `libass force_style`. Only active when `draft.packaging is None`. |
| **5. Title Artwork & Motion** | `title_art_v6.py`, `comic_art.py`, `titles.py` | Styles: `comic`, `neon`, `arena`, `editorial`, `pixel`, `frosted`, `plain`, `impact`, `card`<br>Motion: `overlay_motion` (sliding entrance) | Raster PNG title card (inputs 1:v / 2:v in FFmpeg) displayed for the first 4 seconds. Only active when `draft.packaging is None`. |

[SOURCE VERIFIED: `backend/services/platform_strategy.py`, `backend/services/studio/models.py`, `backend/services/studio/render.py`, `backend/services/publish_export.py`]

---

## 3. The Eight Gallery Examples Audit & Matrix

Every example in the original AutoClip gallery (`docs/images/demos/manifest.json` and `docs/images/README.md`) was mapped to its actual runtime code and verified against AutoShorts 11.0 capabilities.

### 3.1 Detailed Case Analysis

#### Case 1: Sylvester Stallone Interview — TikTok / Chinese
- **Gallery Manifest Entry:** `stallone-nyt/01` | Guest: Sylvester Stallone | Title: "60岁写《洛奇》 是向衰老宣战"
- **Platform Strategy:** `douyin` (or `tiktok`) [1080x1920, 9:16]
- **Runtime Template:** `interview_zh` (Audience language: `zh`)
- **Framing Layout:** `window` (4:3 cropped window at 1080x810, y=560)
- **Subtitle Mechanism:** `packaging_render.py` -> `Caption` style at y=1348
- **Title Mechanism:** `packaging_render.py` -> `Title` / `TitleAccent` at y=150, 300
- **AutoShorts 11.0 Reality:** Adapter sets `layout = "blur"` (from `strat.layout`). Renders subtitles and title with `interview_zh` layout coordinates, but against a blurred 9:16 backdrop instead of the 4:3 interview window.
- **Classification:** `PARTIALLY INTEGRATED` [SOURCE VERIFIED]

#### Case 2: MrBeast Full Screen Podcast — TikTok / English
- **Gallery Manifest Entry:** `mrbeast-colin-samir/01` | Guest: MrBeast | Title: "YouTube's first 5 seconds matter more than thumbnails"
- **Platform Strategy:** `tiktok` [1080x1920, 9:16, layout: `crop`, template: `podcast_en`]
- **Runtime Template:** `podcast_en` (Audience language: `en`)
- **Framing Layout:** `crop` (9:16 tight vertical face crop tracking speaker)
- **Subtitle Mechanism:** `packaging_render.py` -> `Words` style (pop karaoke word-by-word at y=1400)
- **Title Mechanism:** `packaging_render.py` -> `Hook` style at y=250 for first 2.8s
- **AutoShorts 11.0 Reality:** Fully matches! Platform `tiktok` gives `layout="crop"`, template `podcast_en` attaches `Packaging(template="podcast_en")`. Both are user-selectable in Style Studio.
- **Classification:** `FULLY INTEGRATED AND USER-SELECTABLE` [SOURCE VERIFIED]

#### Case 3: Bill Gates Interview — TikTok / Chinese
- **Gallery Manifest Entry:** `gates-ezra/01` | Guest: Bill Gates | Title: "AI测试时竟会装傻？"
- **Platform Strategy:** `douyin` [1080x1920, 9:16]
- **Runtime Template:** `interview_zh`
- **Framing Layout:** `window` (4:3 cropped window at 1080x810, y=560)
- **AutoShorts 11.0 Reality:** Same as Case 1 (partially integrated due to blurred backdrop fallback instead of 4:3 window).
- **Classification:** `PARTIALLY INTEGRATED` [SOURCE VERIFIED]

#### Case 4: AI Experts Debate — Interview Style / Shorts / English
- **Gallery Manifest Entry:** `ai-labs-debate/01` | Guest: AI Experts Debate | Title: "We're Driving Toward a Cliff in the Fog"
- **Platform Strategy:** `youtube_shorts` [1080x1920, 9:16, audience: `en`]
- **Runtime Template:** `interview_zh` (used for interview framing with English text)
- **Framing Layout:** `window` (4:3 cropped speaker window)
- **AutoShorts 11.0 Reality:** User can select `youtube_shorts` + `interview_zh`. However, `youtube_shorts` defaults to `layout="crop"`. The adapter does not force `layout="window"`, resulting in full 9:16 face crop with misaligned lower interview title/caption cards.
- **Classification:** `PARTIALLY INTEGRATED` [SOURCE VERIFIED]

#### Case 5: Tony Robbins Full Screen Podcast — TikTok / Chinese
- **Gallery Manifest Entry:** `robbins-36months/01` | Guest: Tony Robbins | Title: "风险极小，回报极大？"
- **Platform Strategy:** `douyin` [1080x1920, 9:16] with `portrait_style: "podcast"`
- **Runtime Template:** `podcast_en` with Chinese language
- **Framing Layout:** `crop` (Full screen 9:16 vertical crop)
- **Subtitle Mechanism:** `packaging_render.py` lines 238-247 (Chinese text in podcast template renders 2-line bottom captions at y=1420 instead of single-word pop)
- **AutoShorts 11.0 Reality:** User can select `douyin` + `podcast_en`. However, `douyin` has `strat.layout = "blur"`. The video renders as blurred backdrop instead of full-screen crop.
- **Classification:** `PARTIALLY INTEGRATED` [SOURCE VERIFIED]

#### Case 6: Andrew Garfield × Amy Poehler — Full-Screen Podcast / Shorts / English
- **Gallery Manifest Entry:** `garfield-poehler/01` | Guest: Andrew Garfield × Amy Poehler | Title: "I Love Competition—but Hate Fake Casualness"
- **Platform Strategy:** `youtube_shorts` [1080x1920, 9:16, layout: `crop`, template: `podcast_en`]
- **Runtime Template:** `podcast_en` (Audience language: `en`)
- **Framing Layout:** `crop` (9:16 vertical crop)
- **Subtitle Mechanism:** Word karaoke (`Words` pop at y=1400)
- **AutoShorts 11.0 Reality:** Fully matches! Platform `youtube_shorts` gives `layout="crop"`, template `podcast_en` gives word karaoke. Both selectable in UI.
- **Classification:** `FULLY INTEGRATED AND USER-SELECTABLE` [SOURCE VERIFIED]

#### Case 7: Tong Liya × Lu Yu — Original Aspect Ratio / Bilibili / Chinese
- **Gallery Manifest Entry:** `luyu-tongliya/01` | Guest: 佟丽娅 × 鲁豫 | Title: "舞蹈是我骨子里的东西"
- **Platform Strategy:** `bilibili` [1920x1080, 16:9, layout: `fit`] (or `original`)
- **Runtime Template:** `landscape` (no vertical packaging)
- **AutoShorts 11.0 Reality:** Style Studio offers `bilibili` (16:9 Fit), but forces user to pick either `interview_zh` or `podcast_en`. `packaging_render.py` is hardcoded to `PlayResX: 1080, PlayResY: 1920`. Applying vertical ASS to a 1920x1080 video causes extreme vertical stretching and off-screen text distortion. `template: "none"` / `landscape` is not selectable.
- **Classification:** `PARTIALLY INTEGRATED` [SOURCE VERIFIED]

#### Case 8: Sam Altman — Original Aspect Ratio / Bilibili / Chinese
- **Gallery Manifest Entry:** `altman-uses-ai/01` | Guest: Sam Altman | Title: "AI是文艺复兴，还是工业革命？"
- **Platform Strategy:** `bilibili` [1920x1080, 16:9, layout: `fit`]
- **Runtime Template:** `landscape`
- **AutoShorts 11.0 Reality:** Same as Case 7.
- **Classification:** `PARTIALLY INTEGRATED` [SOURCE VERIFIED]

---

### 3.2 Eight-Example Integration Summary Matrix

| # | Example Name | Platform Strategy | Layout | Template ID | AutoShorts Sidecar Status | UI Selectable? | Audit Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Sylvester Stallone Interview | `douyin` | `window` | `interview_zh` | Engine supports, Adapter defaults to blur | Partially (layout mismatch) | `PARTIALLY INTEGRATED` |
| **2** | MrBeast Full Screen Podcast | `tiktok` | `crop` | `podcast_en` | Fully supported | Yes (TikTok + Podcast EN) | `FULLY INTEGRATED AND USER-SELECTABLE` |
| **3** | Bill Gates Interview | `douyin` | `window` | `interview_zh` | Engine supports, Adapter defaults to blur | Partially (layout mismatch) | `PARTIALLY INTEGRATED` |
| **4** | AI Experts Debate | `youtube_shorts` | `window` | `interview_zh` | Engine supports, Adapter defaults to crop | Partially (layout mismatch) | `PARTIALLY INTEGRATED` |
| **5** | Tony Robbins Podcast | `douyin` | `crop` | `podcast_en` | Engine supports, Adapter defaults to blur | Partially (layout mismatch) | `PARTIALLY INTEGRATED` |
| **6** | Andrew Garfield × Amy Poehler | `youtube_shorts` | `crop` | `podcast_en` | Fully supported | Yes (Shorts + Podcast EN) | `FULLY INTEGRATED AND USER-SELECTABLE` |
| **7** | Tong Liya × Lu Yu | `bilibili` | `fit` | `landscape` | Engine supports, UI attaches vertical pack | Partially (distorted ASS) | `PARTIALLY INTEGRATED` |
| **8** | Sam Altman | `bilibili` | `fit` | `landscape` | Engine supports, UI attaches vertical pack | Partially (distorted ASS) | `PARTIALLY INTEGRATED` |

---

## 4. Deep Technical Analysis: Interview (ZH) & Podcast (EN) Templates

### 4.1 Interview (ZH) Template (`interview_zh`)
- **Code Locations:** `backend/services/studio/packaging_render.py` (lines 196-236, 281-292, 320-343), `jobs.py` (lines 769-813).
- **Spatial Geometry:**
  - Designed for a vertical 1080x1920 canvas.
  - Places a 4:3 speaker window (`1080x810`) at `y = 560`.
  - Pinned Title at top: `y = 150` (line 1), `y = 300` (line 2). Font: `Noto Sans SC Thin`, size up to 112pt. Line 2 uses `TitleAccent` color if `title_accent_line == 1`.
  - Dynamic Progress Bar: 6px accent bar placed at `y = 1370` (`WIN_Y + WIN_H`), animating across the width: `x = -w + w*(t+offset)/total`.
  - Subtitle Placement: Pinned directly below the window at `y = 1348` (`WIN_Y + WIN_H - 22`) or `y = 1400` (`+30` if burned captions detected). Maximum 2 lines per screen.
  - Lower-Third Editorial Tags: Animated pills (`Tag` / `TagPill`) at `y = 1630` (`WIN_Y + WIN_H + 260`), displaying concise commentary (e.g., "七分钟干完三个月").
  - Speaker Nameplate: Lower-left animated card at `y = 1740` (`WIN_Y + WIN_H + 370`), with vertical accent bar, speaker name, and role.
- **Palette System:** Colors follow content mood (`azure`, `amber`, `coral`, `mint`, `lemon`, `rose`, `lilac`), with background tinted near-black.

### 4.2 Podcast (EN) Template (`podcast_en`)
- **Code Locations:** `backend/services/studio/packaging_render.py` (lines 202-216, 237-271, 344-350).
- **Spatial Geometry:**
  - Full-screen vertical presentation (typically 9:16 `crop`).
  - Hook Card: Displays opening headline at `y = 250` for the first 2.8 seconds, with slide-in animation `\move(540,300,540,250,0,260)ad(140,220)`.
  - Dynamic Word-by-Word Karaoke:
    - Parses cues into individual timed words (`_words`).
    - Chunks words into 2-4 word screens (`_chunks`).
    - Active word is highlighted in accent color, while inactive words remain white.
    - First word of each chunk has a pop scale effect `scx86scy86	(0,110,scx100scy100)`.
    - Placed in lower third at `y = 1400`.
  - Cinematic Style Variant: If `style == "cinematic"`, applies 2-layer glow effect (`CineGlow` + `Cine` at `y = 1320`) and video color grading (`eq=saturation=0.72:brightness=-0.03,vignette=PI/5`).
  - Chinese Audience Fallback: If `audience_language == "zh"`, bypasses word pop and renders 2-line bottom captions at `y = 1420`.

---

## 5. Beginner-Friendly Reference: Subtitle Looks (Standalone Mode)

When `draft.packaging is None`, subtitles are rendered via `publish_export._build_filter` using FFmpeg `subtitles` filter with `force_style`.

| Subtitle Look | ASS PrimaryColour | Outline | Shadow | MarginV (Landscape / Portrait) | Visual Aesthetic & Best Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean** | `&H00FFFFFF` (White) | 2px black | 0px | 48 / 22.6 | **Minimalist clean white.** Standard subtitle look; unobtrusive, crisp, excellent for documentary, instructional, and formal content. |
| **Bold** | `&H00FFFFFF` (White) | 3px black | 1px | 56 / 26.3 | **Punchy high-impact white.** Extra-large bold glyphs with heavy stroke. Designed specifically for readability on handheld mobile displays. |
| **Box** | `&H00FFFFFF` (White) | 6px padding (`&H88000000`) | 0px | 52 / 24.4 | **Translucent dark card.** Renders an opaque capsule behind text (`BorderStyle=3`). 100% readable even over complex, high-contrast video footage. |
| **Accent** | `&H0000E5FF` (Yellow `#FFE500`) | 3px black | 0px | 56 / 26.3 | **Electric yellow short-form.** High-vibrancy saturated yellow text with solid black outline. The iconic viral social-media caption look. |

> **Portrait Scaling Formula:** libass operates on a fixed 288px virtual vertical canvas. For 9:16 portrait videos (1080x1920), font size and borders are scaled by `0.47x` via `portrait_subtitle_style()` to prevent oversized captions.

---

## 6. Beginner-Friendly Reference: Title Art Styles & Renderers

Title Art generates a high-resolution raster graphic (1080x1920) for the first 4 seconds of the clip.

| Title Art Style | Engine File | Font Used | Accent Color | Visual Aesthetic & Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Comic** | `comic_art.py` | `Anton-Regular.ttf` / `NotoSansSC.ttf` | `#ffe52d` (Yellow) | **3D Comic Punch.** Sheared italic glyphs, heavy black ink outline, 3D extruded shadow depth, tilted 5 degrees. |
| **Neon** | `title_art_v6.py` | `Anton-Regular.ttf` / `NotoSansSC.ttf` | `#ccff00` (Neon Lime) | **Cyberpunk Highlighter.** Glowing marker slab, angled highlighter rectangle behind the payoff line, tilted 4 degrees. |
| **Arena** | `title_art_v6.py` | `Anton-Regular.ttf` / `NotoSansSC.ttf` | `#dfff00` / `#13a9ff` | **Esports Broadcast Badge.** Aggressive sharp polygon plates, dark navy base with cyan accent, tilted 9 degrees. |
| **Editorial** | `title_art_v6.py` | `Anton-Regular.ttf` / `NotoSansSC.ttf` | `#ff4826` (Warm Coral) | **Magazine Headline.** Left-aligned bold headline typography with clean horizontal underline block beneath the payoff phrase. 0 degree tilt. |
| **Pixel** | `title_art_v6.py` | `PressStart2P-Regular.ttf` | `#ed327c` (Arcade Pink) | **Retro 8-Bit Arcade.** Chunky pixelated retro bitmap glyphs with nearest-neighbor scaling and hot pink underline. |
| **Frosted** | `title_art_v6.py` + `title_materials.py` | `Anton-Regular.ttf` / `NotoSansSC.ttf` | `#00e6dc` (Cyan) | **iOS Frosted Glass.** Dual-layer composite: extracts alpha mask and blurs source video behind text (`gblur=sigma=12`, `maskedmerge`). |
| **Plain** | `titles.py` | Arial Bold / msyhbd | White | Pure FFmpeg `drawtext` without decorative graphics. |
| **Impact** | `titles.py` | Arial Bold / msyhbd | `0xffec47` (Yellow) | Centered uppercase text with dark border and drop shadow. |
| **Card** | `titles.py` | Arial Bold / msyhbd | White on dark blue | FFmpeg `drawbox` banner (`0x142231@0.9`) with cyan left accent bar (`0x56ecbb`). |

---

## 7. Title Motion Mechanics (`Enable Title Motion`)

- **Code Location:** `autoclip_studio/title_art.py` (`overlay_motion()`), `autoclip_studio/render.py` (line 259).
- **Mechanism:** When `title_motion == True`, FFmpeg dynamic overlay coordinates are injected into `[styled]` overlay filter:
  - **Arena Style:** Slides horizontally from left: `x = "-36*max(0,1-t/0.18)"`, `y = "0"`. Starts 36px to the left, snaps to center over 180ms.
  - **Comic, Neon, Editorial Styles:** Slides upward from bottom: `x = "0"`, `y = "{distance}*max(0,1-t/0.18)"` where `distance = round(h * 0.016)` (~31px for 1920h).
  - **Pixel & Frosted Styles:** Remain static (`x = '0'`, `y = '0'`) to preserve crisp pixel alignment and avoid glass blur artifacts.
  - **When Disabled (`title_motion == False`):** Coordinates evaluate to `'0', '0'`, perfectly static overlay.
  - **Visibility Window:** Fixed at 4.0 seconds via `enable='lt(t,4)'`.

---

## 8. End-to-End Pipeline Trace & The Feature Shadowing Root Cause

```mermaid
flowchart TD
    UI[Frontend Style Studio: main.tsx] -->|JSON presentationConfig| IPC[Tauri IPC: update_project_style]
    IPC --> DB[(SQLite: projects.presentation_config)]
    DB --> RENDER[Backend: render_flat_clip_for_candidate_impl]
    RENDER --> MEDIA[media.rs: render_autoclip_clip]
    MEDIA --> ADAPTER[Python: autoclip_adapter.py]
    ADAPTER --> DRAFT[Build Draft & Packaging]
    DRAFT --> STUDIO[autoclip_studio/render.py]
    
    subgraph Execution Path in render.py
        STUDIO --> CHECK{draft.packaging is not None?}
        CHECK -->|YES: Current Behavior| PACKAGED[elif packaged: packaging_render.scene_ass]
        CHECK -->|NO: template == none| STANDALONE[else: _build_filter + title_art.png]
    end
    
    PACKAGED --> OUT1[Video with Packaging ASS only
(Title Art, Motion & Subtitle Look IGNORED)]
    STANDALONE --> OUT2[Video with Subtitle Look + Title Art + Motion]
```

### The Exact Code Chain
1. **Frontend (`main.tsx` lines 1274-1284):** Sends `platform_strategy`, `subtitle_style`, `title_style`, `title_motion`, and `template: "interview_zh"`.
2. **Adapter (`autoclip_adapter.py` lines 310-324):**
   ```python
   if template_name in ("interview_zh", "podcast_en"):
       packaging_obj = Packaging(
           template=template_name,
           audience_language=strat.audience_language,
           title_lines=[hook_text[:40]] if hook_text else [],
           cues=packaging_cues,
           fallback=True,
       )
   ```
   Because `template_name` is `"interview_zh"`, `packaging_obj` is instantiated and attached to `draft.packaging`.
3. **Studio Renderer (`render.py` lines 132, 138, 213-225):**
   ```python
   packaged = draft.packaging is not None  # True!
   hook = "" if packaged else draft.hook  # hook is wiped to empty string!
   ...
   elif packaged:
       built = packaging_render.scene_video_graph(draft, scene, i, ass_path, w, h)
   else:
       built = _build_filter(req, spec, ..., draft.subtitle_style)  # SKIPPED!
   ...
   elif hook and i == 0 and draft.title_style in title_art.STYLES:
       # SKIPPED because hook == ""! Title Art & Motion are NEVER rendered!
   ```

---

## 9. Comprehensive Integration Gaps Audit

| Gap ID | Severity | Gap Type | Description | Technical Impact | Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GAP-01** | **CRITICAL** | Architecture / UI Shadowing | Packaging template shadows Standalone Styling controls. | Selecting Subtitle Look (Clean/Bold/Box/Accent), Title Art Style (Comic/Neon/Arena/etc.), or Title Motion has ZERO effect on output. | Add `template: "none"` ("Standalone Style") option in UI; when selected, adapter sets `draft.packaging = None`. |
| **GAP-02** | **HIGH** | Data Adapter Omission | Adapter does not map `interview_zh` template to `layout: "window"`. | Interview template renders on a blurred backdrop instead of the iconic 4:3 window. | In `autoclip_adapter.py`, if `template_name == "interview_zh"`, force `layout_val = "window"`. |
| **GAP-03** | **HIGH** | Geometry Incompatibility | Landscape platforms (`bilibili`) receive vertical packaging ASS. | Selecting Bilibili (16:9) produces stretched, off-screen ASS subtitles due to hardcoded 1080x1920 PlayRes. | For `bilibili` or `landscape`, default template to `"none"` or `"landscape"`. |
| **GAP-04** | **MEDIUM** | UI Incompleteness | 3 of 8 platform strategies missing from UI. | Users cannot select `original`, `instagram_reels`, or `youtube_long`. | Add missing platforms to platform selector in `main.tsx`. |
| **GAP-05** | **LOW** | Adapter Constraint Clamping | `autoclip_adapter.py` clamps `strat.layout == "none"` to `"fit"`. | Prevents true 1:1 original passthrough export. | Expand allowed layouts in adapter: `("fit", "crop", "blur", "window", "none")`. |

---

## 10. Strategic Recommendations & Minimal Remediation Plan

### 10.1 Smallest Possible Non-Breaking Fix (To Expose Already-Integrated Features)
The quickest, highest-value change requires **zero changes to the rendering engine** and only minor adjustments in the UI and adapter:

1. **In `main.tsx` (Style Studio & Import Modal):**
   - In the Packaging Template selector, add a 3rd option:  
     `{ id: "none", label: "Standalone (Custom Title & Subtitle)" }`
   - When `template === "none"`:
     - The Subtitle Look cards (Clean, Bold, Box, Accent) and Title Art Style chips (Comic, Neon, Arena, etc.) and Title Motion checkbox are fully active and labeled "Active Custom Style".
   - When `template === "interview_zh"` or `"podcast_en"`:
     - Label the section as "Template-Managed (Overrides Standalone Styles)" or auto-select "none" when the user clicks a specific Title Art style.
2. **In `autoclip_adapter.py`:**
   - In `build_draft_and_scene`:
     ```python
     if template_name == "interview_zh":
         layout_val = "window"
     elif template_name == "none":
         packaging_obj = None
     ```
   - Allow `"none"` in layout check:
     ```python
     if layout_val not in ("fit", "crop", "blur", "window", "none"):
         layout_val = "fit"
     ```

### 10.2 Features Requiring Real Future Implementation (Backlog)
- **HTML Packaging Engine:** `editorial` and `street` templates in upstream AutoClip rely on headless browser rendering via Playwright / Chromium to capture animated HTML/CSS DOM elements. This is currently disabled for offline desktop stability and would require a bundled headless browser sidecar.
- **Multi-Speaker DualFrame in Mode 2:** Upstream AutoClip only supports single 4:3 speaker window or 9:16 single-face crop. AutoShorts Mode 1 includes DualFrame stacked layout. Porting DualFrame into AutoClip's `packaging_render.py` would allow 2-person interviews in true vertical split.

---

## 11. Answers to Critical User Questions

1. **Are 'Interview' and 'Full Screen Podcast' separate platform strategies or just descriptive gallery labels?**  
   **Answer:** They are **gallery descriptions** representing a combination of a Packaging Template (`interview_zh` vs `podcast_en`) and a Layout mode (`window` vs `crop`). They are **not** platform strategies.
2. **Are 'Bilibili 16:9', 'TikTok 9:16', 'Interview', and 'Full Screen Podcast' different TYPES of choices or equivalent?**  
   **Answer:** They are different **TYPES** of choices. 'Bilibili 16:9' and 'TikTok 9:16' are **Platform Strategies** (defining resolution, aspect ratio, and container limits). 'Interview' and 'Full Screen Podcast' are **Packaging Templates / Layouts** (defining visual framing, subtitles, and typography).
3. **Which of the 8 examples can be reproduced through the current UI?**  
   **Answer:** Exactly 2 of the 8: **MrBeast** (TikTok / English) and **Andrew Garfield × Amy Poehler** (Shorts / English). The remaining 6 are partially reproduced or misaligned due to layout and packaging gaps.
4. **Does the current Style Studio UI actually fully wire to the renderer?**  
   **Answer:** The IPC wiring from UI -> DB -> `media.rs` -> `autoclip_adapter.py` is 100% complete. However, because the UI always sends a non-null template (`interview_zh` / `podcast_en`), the renderer enters the packaging branch and silently discards Subtitle Look, Title Art Style, and Title Motion.
5. **Which features are implemented but hidden, vs genuinely missing?**  
   - **Implemented but Hidden / Shadowed:** All 4 Subtitle Looks (`clean`, `bold`, `box`, `accent`), all 6 PIL Title Art Styles (`comic`, `neon`, `arena`, `editorial`, `pixel`, `frosted`), Title Motion entrance animation, and `window` layout rendering.
   - **Genuinely Missing:** HTML templates (`editorial`, `street` browser capture) and Multi-Speaker DualFrame in Mode 2.
