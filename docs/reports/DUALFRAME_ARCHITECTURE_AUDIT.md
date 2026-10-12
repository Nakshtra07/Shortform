# AUTOSHORTS 11.0 — DUALFRAME / DUALSTACK ARCHITECTURE FORENSIC AUDIT REPORT

**Date:** 2026-10-11  
**Auditor:** DeepMind Antigravity Pair-Programming Audit Engine  
**Workspace:** `D:\College\Autoshorts 11.0`  
**Git Branch:** `feature/autoclip-scoring-topn`  
**Git Commit:** `ef6067cb11fa85633d0628fc75bd2f596254f6a3`  
**Working Tree State:** Clean (untracked directory `docs/reports/dualframe_audit_2026-10-11/` preserved untouched)

---

## 1. Executive Verdict

AutoShorts 11.0 implements a **dual-regime framing architecture** governed strictly by the project-level framing configuration (`projects.framing_mode` in SQLite / `AUTOSHORTS_FRAMING_MODE` environment variable):

1. **When `framing_mode == "adaptive"` (Default for 10.0/11.0 projects):**
   - **DualFrame / DualStack is unconditionally DISABLED at runtime** (`dual_enabled = False`, `sys.stderr` logs: `[Framing] adaptive=on dualframe_disabled=true`).
   - When two people appear in the same shot (close or far), the system **never creates top/bottom panels**.
   - Instead, the Visual Subject Resolver (`resolve_visual_subject`) classifies the shot as `two_shot_both_fit`, sets the role to `two_shot`, centers the crop on the pair midpoint `(l_tr["cx"] + r_tr["cx"]) / 2.0`, and outputs a **single unified 1:1 square video centered in the 9:16 portrait canvas** (or letterboxed/pillarboxed as configured).
   - Active-speaker following is **suppressed** so the camera does not pan back and forth between speakers.

2. **When `framing_mode == "original"` (Legacy 9:16 portrait framing):**
   - **DualFrame / DualStack is ENABLED** by default (subject to the `AUTOSHORTS_DUALFRAME_ENABLED` kill switch).
   - In this mode, two distinct speakers trigger a temporal state machine (`DualFrameLayoutResolver`). If both speakers persist for $\ge 0.60\text{s}$, the layout transitions from `SINGLE` to `DUAL_STACK` (1080×960 top panel, 1080×960 bottom panel, stacked vertically via FFmpeg `vstack`).
   - Slot assignment is strictly deterministic: the **left speaker ($cx$ smaller) is placed in the TOP panel**, and the **right speaker ($cx$ larger) is placed in the BOTTOM panel**. Slots are locked for the duration of the shot and **never swap** during speaker turns.
   - **Close-Speaker Isolation Zoom:** Inside each 9:8 panel, AutoShorts inspects whether the projected baseline crop contains the other speaker. If contamination occurs, the solver applies **deterministic pinch-zoom scaling** up to `MAX_ISOLATION_SCALE = 1.80` (step `0.02`, clearance $38\text{px}$) to exclude the other speaker from that panel.
   - If two speakers are so close that full isolation is geometrically impossible even at `1.80` zoom, the solver applies a **bounded fallback**: it keeps the scale that maximizes isolated moments, maintains hard containment on the panel subject, and logs `fallback=True`, never crashing and never disabling DualFrame.

---

## 2. Source-Level Decision Flow

```
                                      Source Video Frame + Audio
                                                  │
                                                  ▼
                         Tracking & Identity (YOLO11n + BoT-SORT / YuNet)
                         Diarization & Active Speaker Fusion (OSNet Re-ID)
                                                  │
                                                  ▼
                                 Check Framing Mode Configuration
                                (`AUTOSHORTS_FRAMING_MODE` env var)
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
           ADAPTIVE FRAMING MODE                                   ORIGINAL 9:16 MODE
   (`adaptive_framing == True`)                           (`adaptive_framing == False`)
                       │                                                     │
           DualFrame forced OFF:                                  DualFrame eligible:
    `dual_enabled = False`                                 `dual_enabled = _is_dualframe_env_enabled()`
                       │                                                     │
                       ▼                                                     ▼
           Visual Subject Resolver                                Temporal State Machine
          (`resolve_visual_subject`)                            (`DualFrameLayoutResolver`)
                       │                                                     │
         N >= 2 persons in frame?                               2 persistent eligible tracks (>=0.60s)?
                       │                                                     │
         ┌─────────────┴─────────────┐                         ┌─────────────┴─────────────┐
         ▼                           ▼                         ▼                           ▼
        YES                         NO                        YES                         NO
  Shot: two_shot_both_fit     Single-subject           Transition to DUAL_STACK      Stay in SINGLE
  Role: two_shot             (solo_close_up,           Top: Left speaker (cx min)    Follow active speaker
  Target: (cx_l + cx_r) / 2   speaker, etc.)           Bottom: Right speaker (cx max)or two-shot if span < 0.65
  Canvas: Square 1:1 inside                                    │
  9:16, both retained                                          ▼
                                                     Independent Panel Trajectories
                                                     `_solve_single_panel_trajectory`
                                                               │
                                                     Contamination Check:
                                                     Does panel crop include other speaker?
                                                               │
                                                     ┌─────────┴─────────┐
                                                     ▼                   ▼
                                                    YES                  NO
                                            Pinch-Zoom Isolation    Normal composition
                                            Scale up to 1.80        (optional body-tightening)
                                                     │                   │
                                                     └─────────┬─────────┘
                                                               ▼
                                                     Layout Plan Emission
                                                     (`build_layout_plan`)
                                                               │
                                                     FFmpeg Filtergraph:
                                                     `split=2` -> `crop` -> `scale=1080:960` -> `vstack`
```

---

## 3. Detailed Answers to Specific Questions

### Q1: What triggers DualFrame/DualStack?
**Source File:** `autoshorts/src-tauri/scripts/speaker_tracker.py`  
**Functions:** `run()` (lines 6430–6438), `DualFrameLayoutResolver.update()` (lines 3505–3568)  
DualFrame is triggered when:
1. `adaptive_framing` is **False** (i.e. `AUTOSHORTS_FRAMING_MODE != "adaptive"`).
2. Environment kill switch `AUTOSHORTS_DUALFRAME_ENABLED` is not false/0/off.
3. Exactly **two eligible subjects** (`n_eligible == 2`) are tracked simultaneously.
4. The two subjects remain continuously present for at least `dual_entry_persistence_sec = 0.60s`.
5. The subjects satisfy spatial clearance: $|cx_r - cx_l| \ge 0.12 \times \text{source\_w}$ ($230.4\text{px}$ at 1080p).
6. Subject area ratio is balanced: $\min(\text{area}_l, \text{area}_r) \ge 0.15 \times \max(\text{area}_l, \text{area}_r)$.

### Q2: Does the decision explicitly depend on the physical or apparent distance between speakers?
**In Original Mode:**
- Entry requires $|cx_r - cx_l| \ge 0.12 \times \text{source\_w}$ ($230.4\text{px}$ at 1920p). If two face centers are closer than $12\%$ of the screen width, they are deemed a duplicate track or co-located artifact, so DualFrame will not engage.
- Once entered, layout selection does not require speakers to be far apart; instead, **close proximity triggers isolation pinch-zoom within the panels**.

**In Adaptive Mode:**
- DualFrame is **never triggered**, regardless of distance.
- Instead, distance determines whether `_pair_composition_fits()` classifies the pair as `two_shot_both_fit` based on `head_span <= renderable_w * 1.05` ($1134\text{px}$ at 1080p).

### Q3: Is separation measured using face-box coordinates, overlap, crop feasibility, speaker identities, active-speaker confidence or something else?
- **Entry separation:** Measured by horizontal center distance $|cx_r - cx_l|$ between tracks.
- **Contamination separation:** Measured by **subject safe-box coordinates** (`left`, `right`, `head_left`, `head_right`, `top`, `bottom`) where:
  - `head_left = fx - 0.12 * fw`
  - `head_right = fx + 1.12 * fw`
  - `shoulder_left = fx + fw/2 - (2.2 * fw)/2`
  - `shoulder_right = fx + fw/2 + (2.2 * fw)/2`
- **Active-speaker confidence:** Diarization + mouth motion (`fusion_by_track`) determines the active speaker for single-crop tracking, but **two-shot compositions and DualFrame slot locks never follow the active speaker**.

### Q4: What determines which person appears in the upper panel and which appears in the lower panel?
**Source File:** `speaker_tracker.py`, lines 3509–3550, 3603–3614  
At the exact moment of entry into `dual_stack`:
1. Eligible tracks are sorted horizontally by $cx$: `sorted_subjects = sorted(eligible_tracks.values(), key=lambda tr: tr.get("cx", 0.0))`.
2. `locked_top_track_id = left_subj["track_id"]` (the person on the **left** of the shot).
3. `locked_bottom_track_id = right_subj["track_id"]` (the person on the **right** of the shot).
4. **Slot Locking:** These track IDs are permanently bound to the top and bottom panels until a shot reset (camera cut) or exit to single. Even if the speakers move, cross arms, or swap conversational dominance, the slot assignment **never swaps**.

### Q5: What exactly is “isolation zoom” in this codebase?
**Source File:** `speaker_tracker.py`, lines 4665–4777 (`_solve_panel_isolation_scale`) and 5060–5178 (`_solve_isolated_panel_trajectory`)  
- **Definition:** Isolation zoom is a deterministic pinch-zoom mechanism that increases the crop scale (`scale > 1.0`, meaning a tighter 9:8 crop) specifically to push the non-intended speaker outside the panel boundaries.
- **Invocation condition:** Invoked whenever the projected baseline panel crop for Speaker A intersects the safe box of Speaker B (or fails to maintain $38\text{px}$ clearance).
- **Scale Search:** Searches in increments of `ISOLATION_SCALE_STEP = 0.02` from `base_scale` up to `MAX_ISOLATION_SCALE = 1.80`. It picks the minimal scale that completely excludes the other speaker across all paired frames.
- **Bounded Fallback:** If `1.80` zoom is insufficient to isolate (e.g. speakers physically touching), it selects the scale that isolates the highest percentage of frames, sets `fallback=True`, and preserves the panel composition without disabling DualFrame.

### Q6: How does the application determine that one speaker contaminates the other speaker's panel?
**Source File:** `speaker_tracker.py`, lines 4540–4563 (`_crop_excludes_other`)  
Contamination is evaluated geometrically:
- Let the candidate panel crop have horizontal span $[x, x + w]$.
- Let the other speaker's safe box span $[\text{other\_left}, \text{other\_right}]$.
- Clearance requirement: `clearance_px = 0.02 * source_w` ($38.4\text{px}$ in 1080p).
- The other speaker is **excluded** if and only if:
  $$\text{other\_right} \le x - \text{clearance\_px} \quad \text{OR} \quad \text{other\_left} \ge x + w + \text{clearance\_px}$$
- If neither condition holds, the other speaker intrudes into the crop window or violates the safety clearance corridor, flagging **contamination**.

### Q7: What happens when both people are close together but clearly visible in the same shot?
- **In Adaptive Mode:** They are framed together in **one shared single square crop** (`shot_type = "two_shot_both_fit"`, `mode = "single"`). No splitting into panels occurs.
- **In Original Mode:** If $|cx_r - cx_l| \ge 0.12 \times \text{source\_w}$, they are split into **DualStack top and bottom panels**, and **isolation zoom engages** to pinch-zoom each panel and exclude the companion.

### Q8: What happens when a single crop can comfortably retain both speakers?
- **In Adaptive Mode:** It always produces a single shared crop centered at the midpoint of the two speakers.
- **In Original Mode:** A single crop is chosen **only if** `face_span <= crop_w_baseline * 0.65` ($395.2\text{px}$ at 1080p) AND neither speaker is classified as actively speaking/reacting, OR if DualFrame is globally disabled. Otherwise, if two persistent subjects exist, DualFrame engages.

### Q9: What happens when the individual crops overlap or each crop includes part of the other person?
- The isolation solver attempts to zoom in on each subject up to `1.80x` zoom to eliminate the overlap.
- If overlap remains unavoidable, the system executes the **bounded fallback**: it uses the tightest valid zoom up to `1.80`, centers on each subject's head, and permits partial companion intrusion in the outer margin rather than crashing or dropping the split layout.

### Q10: What happens when one speaker is off-screen, partially occluded, moving across the frame or not reliably detected?
**Source File:** `speaker_tracker.py`, lines 3397–3402, 3583–3601  
- If a speaker is occluded (`is_occluded = True`) or undetected, the `DualFrameLayoutResolver` activates an exit hysteresis timer (`dual_exit_persistence_sec = 0.75s`).
- If the speaker returns within $0.75\text{s}$, `DUAL_STACK` mode continues unbroken.
- If absence exceeds $0.75\text{s}$, the system exits to `SINGLE` mode (`reason = subject_lost`) and reframes onto the remaining visible speaker.

### Q11: What causes the system to return to a shared frame or another fallback layout?
1. **Camera cut (`is_shot_cut = True`):** Triggers `reset()`, resetting mode to `single`.
2. **Subject loss:** One or both subjects missing for $> 0.75\text{s}$.
3. **Crowd / 3+ subjects:** If $\ge 3$ eligible subjects appear, DualFrame exits immediately (`reason = too_many_subjects`).
4. **Short duration:** If a dual segment is $< 1.00\text{s}$ in duration, `build_layout_plan` demotes it to `single`.
5. **Adaptive Framing flag:** Setting `framing_mode: "adaptive"` disables DualFrame entirely.

---

## 4. Close-versus-Far Behavior Matrix

| Scenario | Source-Code Rule | Detected Layout | Crop / Panel Behavior | Contamination Handling | Isolation Zoom | Evidence | Verdict |
|---|---|---|---|---|---|---|---|
| **Two speakers far apart (Adaptive)** | `adaptive_framing == True` | `single` (1 segment) | Centered shared two-shot or active speaker follow | None (shared composition) | None (`adaptive=on`) | `rio_closepair_adaptive_plan.json`, `validation_report_dualframe.md` | **PROVEN** |
| **Two speakers far apart (Original)** | `n_eligible == 2`, $|cx_2 - cx_1| \ge 0.12W$ | `dual_stack` | Left speaker $\to$ Top panel (9:8); Right speaker $\to$ Bottom panel (9:8) | Clean pass-through (`_crop_excludes_other == True`) | None (base scale 1.0–1.12 sufficient) | `test_dualframe_suite.py` Test 1, `closepair_render.rs` | **PROVEN** |
| **Two speakers close together, both visible (Adaptive)** | `_pair_composition_fits` $\to$ `head_span <= 1.05W` | `single` (1 segment) | Shared centered two-shot, crop $w=608$ (or square in 1080p), no camera panning | Both retained in unified frame | Disabled | `FINAL_REPORT_CLOSEPAIR.md`, real render `closepair_adaptive.mp4` | **PROVEN** |
| **Two speakers close together, both visible (Original)** | $n=2$, span $> 0.65 \times \text{crop\_w}$, separation $\ge 0.12W$ | `dual_stack` | Left $\to$ Top panel; Right $\to$ Bottom panel | Contamination detected: other safe box inside crop | **ACTIVATED:** pinch-zoom scales up to 1.80x | `test_dualframe_suite.py` Tests 4, 11, 12 | **PROVEN** |
| **Opposite speaker enters individual crop** | Safe box violates $38\text{px}$ clearance | `dual_stack` | Panel crop zooms in and shifts away from companion | Shifts crop within `_isolation_safe_region` | Scales dynamically by $+0.02$ until clearance restored | `test_dualframe_suite.py` Test 4, `_solve_panel_isolation_scale` | **PROVEN** |
| **Both speakers fit in one shared crop (Adaptive)** | `head_span <= renderable_w` | `single` | Midpoint centering: `(cx_l + cx_r) / 2`, both heads inside canvas | No split created | Disabled | `FINAL_REPORT_CLOSEPAIR.md`, `closepair_measure_adaptive_452.json` | **PROVEN** |
| **Both speakers fit in one shared crop (Original)** | `face_span <= 0.65 * crop_w` AND not active speaker | `single` | Single crop centered on pair midpoint | Single 9:16 crop | Disabled | `speaker_tracker.py` line 3185, `test_dualframe_suite.py` | **PROVEN** |
| **One speaker missing or occluded** | Absence $> 0.75\text{s}$ | `single` | Grace period holds dual for $0.75\text{s}$, then drops to single on survivor | Normal single tracking resumes | Reverted to single crop | `test_dualframe_suite.py` Tests 6, 7 | **PROVEN** |
| **Speaker change or camera cut** | Camera cut $\to$ `reset()` | Reset to `single` | On speaker change: slots DO NOT swap; on camera cut: state completely reset | Slot lock preserves stability | Resets on cut | `test_dualframe_suite.py` Test 5, 8 | **PROVEN** |

---

## 5. Existing Tests & Verification Executed

During this forensic audit, the following test suites were executed against the active working tree without modifications:

1. **Python DualFrame Suite (`test_dualframe_suite.py`):**
   - **Command:** `& "D:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_dualframe_suite.py`
   - **Result:** **55 passed, 0 failed** in $8.665\text{s}$.
   - **Coverage:** Verified 1-subject single, 2-subject persistent dual_stack entry, slot locking across speaker turns, occlusion tolerance ($0.75\text{s}$), close-speaker isolation zoom, bounded fallback at `1.80`, and body tightening.

2. **Python Active Speaker Suite (`test_active_speaker_suite.py`):**
   - **Command:** `& "D:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_active_speaker_suite.py`
   - **Result:** **61 passed, 0 failed** in $1.276\text{s}$.
   - **Coverage:** Verified audio-visual fusion, Re-ID embedding association, diarization handoff, and dominance ratios.

3. **Rust DualFrame & DualStack Integration Tests:**
   - **Command:** `cargo test -p autoshorts --lib dualframe -j 2`
   - **Result:** **4 passed, 0 failed** in $36.32\text{s}$ (`verify_real_clip01_dualframe_fix`, `test_sp2_dualframe_survives_micro_edits`, `bhaukal_18_dualframe_seam_clamp`, `test_dualframe_safe_corridor_dynamic_validation_and_collapse_fallback`).
   - **Command:** `cargo test -p autoshorts --lib dual_stack -j 2`
   - **Result:** **2 passed, 0 failed** in $2.06\text{s}$ (`test_dynamic_editorial_dual_stack_clamping`, `test_dual_stack_caption_seam_placement`).

4. **Visual & Render Evidence Located:**
   - Existing render: `tmp/closepair_renders/closepair_adaptive.mp4` ($2.08\text{MB}$) & `closepair_original.mp4` ($2.37\text{MB}$) generated from Rio Ferdinand & Cristiano Ronaldo footage (`452.6s` to `462.2s`).
   - Audit run logs: `docs/reports/dualframe_audit_2026-10-11/rio_closepair_adaptive_stderr.log` confirming that on the reference close-pair footage, `Adaptive` mode outputs `layout_plan total_segments=1 dual_segments=0` with `single` composition, exactly matching the documented architectural contract.

---

## 6. Discovered Inconsistencies & Architectural Observations

1. **Editorial Dualism (Adaptive vs. Original):**
   - In `Adaptive Framing` mode, DualFrame split-screen is **completely disabled**. Users seeking a vertical split-screen for two speakers must configure `framing_mode: "original"`.
   - In `Original 9:16` mode, the threshold for keeping two people together in one frame is narrow (`TWO_SHOT_MAX_SPAN = 0.65`, which is $395.2\text{px}$ in a $1080\text{p}$ wide video). Consequently, almost all standard sit-down interview shots in Original mode trigger **DualStack split-screen** with individual panel isolation zooms rather than a shared two-shot.

2. **No Defects Found in Code Implementation:**
   - The implementation strictly adheres to its documented contracts in `FINAL_REPORT_CLOSEPAIR.md` and `validation_report_dualframe.md`.
   - The isolation zoom algorithm (`_solve_panel_isolation_scale`) is fully implemented, verified by 15 dedicated unit tests in `test_dualframe_suite.py`, and functions as designed.
   - Bounded fallback logic prevents crashes or infinite loops when speakers are too close to isolate.
