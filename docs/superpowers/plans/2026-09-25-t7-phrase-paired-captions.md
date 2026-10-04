# T7 Phrase-Paired Rolling Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the T7 caption engine in AutoShorts 10.0 to replace the flawed single-chunk FIFO conveyor belt with the forensic reference model: **Phrase-Paired Display Units** (Line 1 White lead-in $\to$ Line 2 Yellow arrival $\to$ stable coexistence $\to$ simultaneous clean clear $\to$ next phrase starts fresh), verified against the reference video `vidssave.com Best Answer by Kunal shah.💯 1080P.mp4` and rendered on `clip01_flat.mp4`.

**Architecture:** 
1. **Prosodic Chunking:** Group word tokens into 1–3 word rhythm chunks using prosodic boundaries (pauses $>0.15\text{s}$, punctuation, clause commas, conjunctions, duration ceiling $1.5\text{s}$).
2. **Phrase-Pair Grouping:** Group consecutive chunks into independent phrase units: 2-chunk units (Line 1 + Line 2) or 1-chunk units (solo emphatic words / phrase-ending isolated chunks).
3. **Reference-Calibrated Display Timing:** Apply measured $+40\text{ms}$ visual anticipation during running speech, clip-initial lead-in when speech starts near $t=0$, and $+0.25\text{s}$ post-phrase hold.
4. **Stable Geometry & Pure Solid Styling:** Line 1 is pinned at $y_1$ ($1020\text{px}$) and Line 2 at $y_2$ ($1140\text{px}$). Zero line jumping, zero conveyor belt motion, pure solid `#FFFFFF` and `#F9EF07` text with Matt Bold.

**Tech Stack:** Rust (`captions.rs`, `caption_qa.rs`, `libass`), Python / OpenCV / NumPy / SciPy (pixel & audio forensic validation).

## Global Constraints

- **T1–T6 Untouched:** Only `preset_t7` generation logic and template parameters are modified.
- **Caption Intelligence 2.0 Untouched:** No modifications to global semantic keyword ranking or emphasis logic; T7 strictly bypasses CI emphasis.
- **Font Family:** `Matt_Trial-Bold` (Matt Bold, weight 700), cap-height ratio 0.70, size 110.
- **Solid Text Only:** Outline = 0, Shadow = 0, BorderStyle = 1, BackgroundBox = None. Zero stroke, shadow, box, or glow.
- **Colors:** Primary Line 1 = `#FFFFFF` (White), Secondary Line 2 = `#F9EF07` (Yellow, BGR: `[7, 239, 249]`).
- **Capitalization:** Natural sentence capitalization preserved; never force all-caps or all-lowercase.
- **Independence:** No hardcoded words, timestamps, or phrase counts. Deterministic and generalizable to unseen clips.

---

## Task 1: Implement Phrase-Pair Segmentation Engine in `captions.rs`

**Files:**
- Modify: `autoshorts/src-tauri/src/captions.rs`
- Test: `autoshorts/src-tauri/src/captions.rs` (unit tests)

**Interfaces:**
- Produces: 
  ```rust
  #[derive(Debug, Clone, PartialEq)]
  pub enum T7PhraseUnit<'a> {
      Solo {
          chunk: Vec<(usize, &'a TranscriptWord)>,
          start_sec: f64,
          end_sec: f64,
      },
      Pair {
          first: Vec<(usize, &'a TranscriptWord)>,
          second: Vec<(usize, &'a TranscriptWord)>,
          first_start: f64,
          second_start: f64,
          end_sec: f64,
      },
  }

  pub fn group_t7_phrase_units<'a>(
      words: &'a [(usize, &TranscriptWord)],
      clip_start: f64,
      clip_end: f64,
  ) -> Vec<T7PhraseUnit<'a>>;
  ```

- [ ] **Step 1: Write failing unit tests for `group_t7_phrase_units`**
Add unit tests in `captions.rs::tests`:
1. `test_t7_phrase_units_no_line_recycling`: Asserts that consecutive pair units do not share or recycle chunks across phrase boundaries.
2. `test_t7_phrase_units_solo_on_long_pause`: Asserts that when a chunk is followed by a pause $>0.35\text{s}$ or sentence terminator, it yields a `T7PhraseUnit::Solo`.
3. `test_t7_phrase_units_pair_formation`: Asserts that connected speech forms a 2-chunk `Pair` with distinct `first_start`, `second_start`, and unified `end_sec`.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p autoshorts --lib captions::tests::test_t7_phrase_units`
Expected: Compilation failure (`group_t7_phrase_units` and `T7PhraseUnit` not defined).

- [ ] **Step 3: Implement `group_t7_phrase_units`**
In `autoshorts/src-tauri/src/captions.rs`:
1. Segment words into 1–3 word chunks using `segment_speech_chunks` with prosodic boundaries (pause $>0.12\text{s}$, clause comma, sentence terminator, max duration $1.5\text{s}$).
2. Group chunks into `T7PhraseUnit`s:
   - If chunk ends with a sentence terminator (`.`, `!`, `?`) or has pause to next $>0.35\text{s}$ $\implies$ `Solo`.
   - Otherwise, pair current chunk with next chunk $\implies$ `Pair`.
   - Cap total pair duration at $2.6\text{s}$ to prevent excessively long pairs.
   - Advance iterator past both chunks for pairs, so no chunk is ever reused.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p autoshorts --lib captions::tests::test_t7_phrase_units`
Expected: PASS (3/3 tests pass).

---

## Task 2: Implement T7 ASS Dialogue Generation in `captions.rs`

**Files:**
- Modify: `autoshorts/src-tauri/src/captions.rs:2103-2234`
- Test: `autoshorts/src-tauri/src/captions.rs` (unit tests)

**Interfaces:**
- Consumes: `group_t7_phrase_units`, `T7PhraseUnit`
- Produces: ASS Dialogue lines with fixed `\pos(540, y1)` and `\pos(540, y2)` without FIFO conveyor belt motion.

- [ ] **Step 1: Write failing unit test for T7 ASS dialogue generation**
Add unit test `test_t7_ass_phrase_paired_structure`:
- Generates ASS for a multi-phrase sample.
- Asserts that for every phrase pair:
  - Line 1 has `{\pos(540, 1020)}` in White (`#FFFFFF`).
  - Line 2 has `{\pos(540, 1140)}` in Yellow (`#F9EF07`).
  - Both lines have the exact same end timestamp (`second_end + 0.25s`).
  - No Dialogue line moves text from position $y_2$ to $y_1$ in subsequent events.

- [ ] **Step 2: Run test to verify it fails**
Run: `cargo test -p autoshorts --lib captions::tests::test_t7_ass_phrase_paired_structure`
Expected: FAIL (current implementation emits conveyor-belt ASS).

- [ ] **Step 3: Rewrite T7 generator in `captions.rs`**
Replace the loop in `captions.rs` lines 2103–2233:
1. Geometry constants:
   - Line pitch: $120\text{px}$.
   - Center anchor: $y_1 = 1020$, $y_2 = 1140$ (in dual stack mode, respect seam clamp).
2. Timing constants:
   - Anticipation lead: $+0.040\text{s}$ ($40\text{ms}$).
   - Clip-initial lead-in: If first chunk speech starts at $<0.4\text{s}$, start at $0.00\text{s}$.
   - Hold after unit end: $+0.25\text{s}$.
3. For each `T7PhraseUnit::Solo`:
   - Emit single Dialogue event at `\pos(540, y1)` in White (`#FFFFFF`) from `start - lead` to `end + hold`.
4. For each `T7PhraseUnit::Pair`:
   - Stage 1 (solo Line 1): From `first_start - lead` to `second_start - lead`:
     - Dialogue at `\pos(540, y1)`: `{first_text}` in White.
   - Stage 2 (paired coexistence): From `second_start - lead` to `pair_end + hold`:
     - Dialogue 1 at `\pos(540, y1)`: `{first_text}` in White.
     - Dialogue 0 at `\pos(540, y2)`: `{\c#F9EF07}{second_text}{\c#FFFFFF}` in Yellow.
   - Both lines finish at the exact same millisecond and clear together. Zero carry-over.

- [ ] **Step 4: Run test to verify it passes**
Run: `cargo test -p autoshorts --lib captions::tests::test_t7_ass_phrase_paired_structure`
Expected: PASS.

---

## Task 3: Regression Suite & Architecture Validation

**Files:**
- Modify: `autoshorts/src-tauri/src/bin/caption_qa.rs` (update validation assertions to enforce phrase-pair rules)
- Test: Full cargo test suite

- [ ] **Step 1: Update `caption_qa.rs` T7 checks**
In `autoshorts/src-tauri/src/bin/caption_qa.rs`:
- Update check `EVT-NO-CONVEYOR`: Verify that for T7, no text from Line 2 of event $N$ appears in Line 1 of event $N+1$.
- Verify that T1..T6 remain 100% untouched.

- [ ] **Step 2: Run full Rust test suite**
Run: `cargo test -p autoshorts --lib captions`
Expected: PASS (all tests pass).

---

## Task 4: Real Render & Forensic Frame Validation (`clip01_flat.mp4`)

**Files:**
- Create: `scratch/validate_phrase_paired_render.py`
- Target render: `clip01_flat.mp4`

- [ ] **Step 1: Render fresh `clip01_flat.mp4`**
Run:
```powershell
cargo run --bin caption_qa -- render "D:/College/Autoshorts 10.0/My thoughts on my 2023_24 season and the truth about my future. Talk with my friend Rio (PART 1) - UR · Cristiano (1080p, h264).mp4" autoshorts/src-tauri/tmp/clip01_render/words.json autoshorts/src-tauri/tmp/clip01_render/candidate.json 1142.915 1181.44 preset_t7 autoshorts/src-tauri/tmp/clip01_render
Copy-Item autoshorts/src-tauri/tmp/clip01_render/rendered.mp4 clip01_flat.mp4 -Force
```

- [ ] **Step 2: Execute automated pixel & timing validation on rendered MP4**
Write and run `scratch/validate_phrase_paired_render.py`:
- Checks:
  1. `NO_CONVEYOR_BELT`: Line 2 text does not shift upward to Line 1 in subsequent frames.
  2. `NO_VERTICAL_SNAP`: Line 1 stays pinned at $y_1$ throughout its visibility.
  3. `SYNCHRONIZED_CLEAR`: Whenever Line 2 clears, Line 1 clears at the identical frame.
  4. `COLOR_INTEGRITY`: Line 1 is pure white `#FFFFFF`, Line 2 is pure yellow `#F9EF07`.
  5. `HOLD_DURATION`: Inter-phrase hold is between $0.20\text{s}$ and $0.35\text{s}$.
  6. `ANTICIPATION_LEAD`: Normal running speech lead is $\le 60\text{ms}$.
  7. `SOLID_GLYPHS`: Zero outline, shadow, or box.

- [ ] **Step 3: Human / Visual verification**
Inspect key timestamps in `clip01_flat.mp4` to confirm smooth, natural, readable phrase-paired presentation matching the Kunal Shah reference video.

---

## Verification Plan

### Automated Commands
```powershell
# 1. Rust unit tests
cargo test -p autoshorts --lib captions

# 2. Render fresh clip
cargo run --bin caption_qa -- render "D:/College/Autoshorts 10.0/My thoughts on my 2023_24 season and the truth about my future. Talk with my friend Rio (PART 1) - UR · Cristiano (1080p, h264).mp4" autoshorts/src-tauri/tmp/clip01_render/words.json autoshorts/src-tauri/tmp/clip01_render/candidate.json 1142.915 1181.44 preset_t7 autoshorts/src-tauri/tmp/clip01_render
Copy-Item autoshorts/src-tauri/tmp/clip01_render/rendered.mp4 clip01_flat.mp4 -Force

# 3. Python forensic verification against acceptance criteria
& "d:\College\Autoshorts 10.0\.venv\Scripts\python.exe" scratch/validate_phrase_paired_render.py

# 4. Frontend build check
npm run build
```

### Manual Verification
- Play `clip01_flat.mp4` side-by-side with `vidssave.com Best Answer by Kunal shah.💯 1080P.mp4`.
- Confirm absence of any conveyor-belt scrolling or upward jumping.
- Confirm snappy, stable, phrase-by-phrase delivery.
