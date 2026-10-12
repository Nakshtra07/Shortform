# AutoShorts Desktop UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the AutoShorts UI from the dark dashboard into a polished, high-fidelity light desktop application matching the reference industrial-design aesthetic and Google Stitch design system, preserving all existing backend contracts and functionality.

**Architecture:** Update `styles.css` with the new design system tokens (cool grey backgrounds, warm off-white cards, charcoal typography, mint accents, pill controls) and refactor `main.tsx` to cleanly implement Screen A (All Projects), Screen B (Project Workspace), Screen C (Framing & Captions Setup with Scoring Backend selector), and Screen D/E (Integrated Style Studio side panel for single and bulk clip restyling).

**Tech Stack:** React 19, TypeScript, Vite, Tauri v2, Lucide Icons, CSS Custom Properties.

## Global Constraints
- Zero backend Rust logic alterations or schema breaking changes.
- Preserve all existing Tauri invoke command names, arguments, and return types (`create_project_from_path`, `generate_candidates`, `render_flat_clip_for_candidate`, `set_selected_clip_count`, `set_selected_rank_range`, etc.).
- Preserve the two-page application structure: (1) All Projects and (2) Project Workspace.
- Scoring backend selector appears ONLY in Screen C (Initial Framing & Captions setup) — never as a persistent dropdown in the workspace.
- Style Studio is an integrated slide-in drawer inside the Project Workspace (not a new top-level page).
- Zero live video previews or automatic rendering workflows; all restyling choices require explicit "Apply & Re-render".
- Color system: Background `#C7C9CD`, Surface `#E0E1E3`, Cards `#F3F3F1`, Primary text `#17191B`, Muted text `#777D85`, Borders `#C8CACE`, Info accent `#A8BBC9`, Mint accent `#9CDDC9`.
- No neon green, glowing borders, heavy drop shadows, or glassmorphism.
- TypeScript compilation (`tsc && vite build`) and Rust checks must pass with 0 errors.

---

### Task 1: Design System & Color Tokens in `styles.css`

**Files:**
- Modify: `autoshorts/src/styles.css:1-150`

**Interfaces:**
- Consumes: CSS root variables
- Produces: CSS custom properties (`--bg-base`, `--bg-surface`, `--bg-card`, `--text-primary`, `--text-secondary`, `--border-color`, `--accent-mint`, `--accent-info`) used by all components in `main.tsx`.

- [ ] **Step 1: Define Google Stitch design tokens in `styles.css`**

Replace the dark theme variables in `styles.css` with the light industrial-design tokens:
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  /* Core Color Tokens from Google Stitch & Reference Aesthetic */
  --bg-base: #C7C9CD;           /* Cool grey application background shell */
  --bg-surface: #E0E1E3;        /* Pale grey content surfaces and panels */
  --bg-card: #F3F3F1;           /* Warm off-white cards and elevated surfaces */
  --bg-card-hover: #ECECE9;     /* Subtle hover on cards */
  --bg-card-selected: #F0FAF7;  /* Tinted mint background for selected state */

  --text-primary: #17191B;      /* Near-black charcoal for headings and body */
  --text-secondary: #777D85;    /* Muted slate grey for metadata and secondary text */
  --text-muted: #8E949D;        /* Inactive and placeholder text */
  --text-inverse: #FFFFFF;      /* White text on dark charcoal elements */

  --border-color: #C8CACE;      /* Subtle grey divider and border */
  --border-color-subtle: #D8DADC;
  --border-color-focus: #9CDDC9; /* Mint focus ring */

  --accent-primary: #17191B;    /* Charcoal primary actions */
  --accent-mint: #9CDDC9;       /* Mint/sage accent for selection, ready status */
  --accent-mint-dark: #195D4E;  /* High-contrast text on mint */
  --accent-info: #A8BBC9;       /* Muted blue-grey metadata accent */

  --status-ready-bg: #9CDDC9;
  --status-ready-text: #17191B;
  --status-processing-bg: #C9B87B;
  --status-processing-text: #3D3517;
  --status-failed-bg: #C97B7B;
  --status-failed-text: #FFFFFF;

  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --radius-full: 9999px;

  --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.05);
  --shadow-card: 0 2px 6px rgba(0, 0, 0, 0.06);
  --shadow-drawer: -4px 0 24px rgba(0, 0, 0, 0.12);

  color: var(--text-primary);
  background: var(--bg-base);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
```

- [ ] **Step 2: Update typography and basic layout rules**

Apply Manrope font to headings (`h1, h2, h3, h4, .font-heading`), Inter for body text, and ensure clean scrollbar and base container styling.

- [ ] **Step 3: Verify build with new styling tokens**

Run: `npm run build` in `autoshorts/`
Expected: PASS (0 errors)

---

### Task 2: Screen A — Refined All Projects Dashboard

**Files:**
- Modify: `autoshorts/src/main.tsx` (Dashboard section ~lines 1620-1685 and Sidebar ~lines 960-1015)
- Modify: `autoshorts/src/styles.css` (Sidebar and `.home-dashboard`, `.projects-grid`, `.project-card` rules)

**Interfaces:**
- Consumes: `projects: Project[]`, `selectProject`, `importMedia`, `renameProject`, `deleteProject`
- Produces: Polished All Projects page matching Screen A from Google Stitch.

- [ ] **Step 1: Refactor Sidebar in `main.tsx`**

- Refine the sidebar: AutoShorts wordmark with mint logo icon at the top.
- Navigation pill for "All Projects" with active mint indicator.
- Collapsible/compact "Recent" project list with active indicator dot and timestamps.
- Remove redundant, oversized duplicate import buttons from the sidebar.
- Bottom user footer with avatar and Settings toggle.

- [ ] **Step 2: Refactor All Projects Header & Filter Bar in `main.tsx`**

- Left header: "All Projects" heading in Manrope Bold 28px, subtitle "Manage and open your video projects."
- Right header: Pair of action buttons: Primary filled charcoal "Import Recording" with upload icon, secondary outlined "Import from YouTube".
- Filter bar:
  - Search input with search icon (`projectSearchTerm` state).
  - Pill segmented filters: `All`, `Ready`, `Processing`, `Failed` with counts.
  - Sorting dropdown: `Recently opened`, `Date created`, `Duration`.

- [ ] **Step 3: Refactor Project Card Grid**

- 3-column responsive grid of `#F3F3F1` cards with 8px radius and `#C8CACE` 1px border.
- 16:9 thumbnail preview area with duration overlay badge and film icon.
- Status chip:
  - Ready: mint `#9CDDC9` background with dark text.
  - Processing: muted amber `#C9B87B`.
  - Failed: muted red `#C97B7B`.
- Action buttons: Clean filled charcoal `Open` button, Kebab dropdown for `Rename` and `Delete`.
- Hover state: 2px mint `#9CDDC9` outline with subtle elevation.

- [ ] **Step 4: Verify build and test behavior**

Run: `npm run build` in `autoshorts/`
Expected: PASS (0 errors)

---

### Task 3: Screen B — Project Workspace Layout & Polish

**Files:**
- Modify: `autoshorts/src/main.tsx` (Workspace section ~lines 1015-1620)
- Modify: `autoshorts/src/styles.css` (Workspace panels, candidate cards, transcript styling)

**Interfaces:**
- Consumes: `detail: ProjectDetail`, `transcript: NormalizedTranscript`, `candidates: Candidate[]`, `clips: Clip[]`
- Produces: Polished Project Workspace matching Screen B from Google Stitch.

- [ ] **Step 1: Refactor Project Workspace Header & Workflow Navigation**

- Top bar: Back chevron button to return to All Projects (`setDetail(null)`), project title in Manrope semibold 18px charcoal, status chip in mint.
- Centered workflow bar: Horizontal segmented pill container (`Transcript`, `Moments`, `Cut`, `Captions`, `Export`) with active tab highlighted in solid charcoal.
- Right actions: API Settings icon button, Refresh icon button.

- [ ] **Step 2: Polish Transcript Panel (Left Column)**

- Panel header: "Transcript" in Manrope semibold 15px, compact search input, outlined "Transcribe" button.
- Transcript entries: Monospace timestamps in muted blue-grey `#A8BBC9`, speaker labels, quiet 1px bottom separators.
- Active segment: highlighted with 3px mint `#9CDDC9` left border and soft mint background tint.

- [ ] **Step 3: Polish Clip Candidates Panel (Right Column)**

- Panel header: "Clip Candidates", compact count chips (`12 Selected`, `22 Pool`), segmented toggle (`Selected | Pool`), primary charcoal `Cut Selected` button.
- Secondary controls: Top-N stepper, Range controls (`From` - `To` - `Last`), and outlined `Find Viral Moments` button.
- Ensure **NO persistent scoring backend dropdown** appears on this page (retained solely in Screen C).

- [ ] **Step 4: Polish Candidate Cards**

- 2px mint `#9CDDC9` border and light tint for selected candidates.
- Checked mint checkbox, `#Rank` badge, time range in muted blue-grey, score provenance badges (`AutoClip Score`, `Composite Score`).
- Hook headline and recommendation rationale in clear typography.
- Action row: `Cut` / `Re-cut` button, "Restyle" button for clips that have output paths.

- [ ] **Step 5: Verify build**

Run: `npm run build` in `autoshorts/`
Expected: PASS (0 errors)

---

### Task 4: Screen C — Initial Framing & Captions Setup Modal

**Files:**
- Modify: `autoshorts/src/main.tsx` (Setup modal ~lines 1700-1860)
- Modify: `autoshorts/src/styles.css` (Modal overlay, cards, typography previews)

**Interfaces:**
- Consumes: `selectedFramingMode`, `selectedStyle`, `scoringMode`, `confirmImport`
- Produces: High-fidelity Screen C modal overlaying dimmed workspace.

- [ ] **Step 1: Refactor Modal Layout & Framing Section**

- Centered 800px card on `#F3F3F1` with 12px radius, `#C8CACE` border, dimmed workspace backdrop.
- Modal header: "Configure & Start Processing" in Manrope Bold 22px charcoal, explanatory subtitle, and project name pill.
- Section 1: Framing:
  - Adaptive Framing card (selected: 2px mint border, light mint tint, crop icon).
  - Original 9:16 card (unselected: 1px border, phone icon).

- [ ] **Step 2: Refactor Caption Style Presets with Static Typography Samples**

- Section 2: Caption Style:
  - 3 primary selectable cards with static typography preview boxes:
    - **Clean** (Selected): minimal white subtitle centered in dark preview box.
    - **Bold**: high-contrast punchy typography sample.
    - **Word Highlight**: dynamic highlighted word sample.
  - Dropdown or expandable row for additional specialized templates (`Hormozi Viral`, `Narrative Pop`, `T7 Reference Style`, etc.).

- [ ] **Step 3: Refactor Scoring Backend Section (Exclusive Selection Surface)**

- Section 3: Scoring Backend:
  - Card A: **Legacy Scoring (Recommended / Default)** with shield checkmark icon, charcoal `DEFAULT` badge, offline fast composite scoring description.
  - Card B: **AutoClip Holistic Scoring (NVIDIA)** with chip icon, NVIDIA NIM description, and amber badge `External API · API Key Required`.
  - Fail-fast missing key warning message rendered cleanly beneath the cards.

- [ ] **Step 4: Refactor Modal Footer**

- Left: Outlined `Cancel` button.
- Right: Helper note "This will begin transcription and analysis." + Prominent charcoal `Start Processing →` button.

- [ ] **Step 5: Verify build**

Run: `npm run build` in `autoshorts/`
Expected: PASS (0 errors)

---

### Task 5: Screen D & E — Style Studio Drawer (Single Clip & Bulk)

**Files:**
- Modify: `autoshorts/src/main.tsx` (Add Style Studio state and drawer rendering inside workspace)
- Modify: `autoshorts/src/styles.css` (Add `.style-studio-drawer`, backdrop, and controls)

**Interfaces:**
- Consumes: `detail.clips`, `selectedCandidates`, `cutCandidate`, `cutSelected`
- Produces: Integrated 400px right-side slide-in panel for single and bulk clip restyling.

- [ ] **Step 1: Add Style Studio State in `main.tsx`**

```typescript
const [styleStudioOpen, setStyleStudioOpen] = useState(false);
const [styleStudioTargetCandidateId, setStyleStudioTargetCandidateId] = useState<string | null>(null); // null = bulk
const [styleStudioFraming, setStyleStudioFraming] = useState<string>("adaptive");
const [styleStudioCaption, setStyleStudioCaption] = useState<string>("clean");
const [styleStudioScope, setStyleStudioScope] = useState<"target" | "selected" | "all">("target");
const [previousRenderPaths, setPreviousRenderPaths] = useState<Map<string, string>>(new Map());
```

- [ ] **Step 2: Implement Entry Points & Trigger Handlers**

- Add "Restyle" button on individual cut clips (`openSingleClipStyleStudio(candidateId)`).
- Add "Restyle Selected" button in Candidate Panel actions when selected clips have renders (`openBulkStyleStudio()`).

- [ ] **Step 3: Build Screen D (Single Clip) & Screen E (Bulk) Drawer UI**

- 400px wide right-hand slide-in drawer (`#F3F3F1` bg, `#C8CACE` left border) overlaying the dimmed workspace.
- Header: "Style Studio", subtitle, dismiss 'x' button, and context badge:
  - Single clip: `Clip #N — 00:12:34…00:17:42`
  - Bulk mode: `N clips selected` banner with layers icon and list of clip numbers.
- Framing options: Adaptive Framing vs Original 9:16 / DualFrame.
- Caption Style options: Clean, Bold, Word Highlight.
- Application Scope: Radio pills for `This clip`, `Selected clips (N)`, `All clips in project (N)`.
- Summary Card: key-value summary (`Applying to`, `Framing`, `Captions`, `Estimated renders`).
- Recommendation Card: AI recommendation rationale for single-speaker/interview content.
- Footer:
  - "Restore Previous Version" (reverts to previous render path if stored).
  - Prominent charcoal "Apply & Re-render" button.
  - Notice that re-styling affects burned-in appearance only and does not restart scoring or transcription.

- [ ] **Step 4: Implement Apply & Re-render Workflow**

When "Apply & Re-render" is clicked:
1. Save the current output paths into `previousRenderPaths` map for recovery.
2. Trigger `cutCandidate` for single clip or batch loop for target clips.
3. Close the drawer upon starting and refresh project details.

- [ ] **Step 5: Verify build**

Run: `npm run build` in `autoshorts/`
Expected: PASS (0 errors)

---

### Task 6: Full Verification & System Health Check

**Files:**
- Test: `autoshorts/` build and Rust integration checks

**Interfaces:**
- Consumes: Whole application
- Produces: Verified production build

- [ ] **Step 1: Run frontend build**

Run: `npm run build` in `autoshorts/`
Expected: `tsc && vite build` completed with 0 errors.

- [ ] **Step 2: Run Rust cargo check**

Run: `cargo check -j 2 --lib` in `autoshorts/src-tauri/`
Expected: 0 errors.

- [ ] **Step 3: Verify all 5 screens against design constraints**

1. Screen A: All Projects has clean card grid, filters, sorting, sidebar.
2. Screen B: Project Workspace has two columns, no scoring selector, horizontal workflow tabs.
3. Screen C: Initial Framing & Captions setup is the ONLY visible scoring backend selector.
4. Screen D: Style Studio single-clip drawer operates inside the workspace without video preview.
5. Screen E: Style Studio bulk drawer operates for multiple clips with summary and apply.
6. Design tokens: light cool-grey theme with mint accents and charcoal typography throughout.

---
