# VALIDATION REPORT — RETENTION-AWARE SMART PACING (AutoShorts 8.0)

> **Date:** 2026-09-15
> **Feature:** Retention-Aware Smart Pacing (§29, `AUTOSHORTS_8_0_BRAIN.md`)
> **Verdict:** **PASS — feature verified on real footage, not merely on unit tests.**
> **Baseline discipline:** AutoShorts 7.0 workspace verified FROZEN (no file modified since 2026-09-14). All 8.0 work confined to `d:\College\Autoshorts 8.0`.

---

## 1. Existing Architecture Inspected (before any changes)

- Full read of the 7.0 render path: `lib.rs` (render orchestration, in-flight tracker), `media.rs` (FFmpeg trim/crop/scale/subtitle-burn clipping pipeline), `llm.rs` (candidate discovery, hook scoring), `speaker_tracker.py` (DualFrame, isolation §25/§26, tightening §27, sampling §28), `captions.rs` (ASS/SRT generation), `models.rs` (NormalizedTranscript schema).
- Brain file §4 (end-to-end architecture), §7 (candidate discovery), §23 (protected systems) studied to place the feature **AFTER candidate selection and BEFORE final rendering** without touching any protected system.
- App DB schema surveyed (`projects`, `transcripts` tables) to understand real transcript shapes (word-level timings, speakers, segments).
- Conclusion: the only safe insertion point is the render path, between candidate selection (already done) and FFmpeg invocation — as a **sidecar decision engine + Rust consumer**, with a byte-identical legacy fallback.

## 2. Requirement Analysis (spec → engineering contract)

- "Can I prove this time interval is safe to remove?" → dual-evidence gate: transcript gap (no word coverage) AND acoustic verification (`ffmpeg silencedetect noise=-35dB:d=0.10`, tolerance 0.04s).
- "When uncertain: KEEP THE ORIGINAL CONTENT" → every unproven interval is left untouched; engine returns `skipped` with a reason rather than a partial plan.
- Edit priority ladder verbatim: (1) leading dead air, (2) trailing dead air, (3) long internal pauses (shorten, preserve natural pause), (4) hesitation/false-start (high confidence only).
- "Do NOT create a solution that simply changes the nominal clip duration while leaving captions, framing, speaker state, or audio timestamps referring to the old source timeline" → Rust consumer must remap words, rebuild framing segments, regenerate ASS/SRT, and construct a real trim/concat filtergraph on the NEW timeline.
- "Do NOT let smart pacing become another candidate-selection system" → the engine only removes silence inside an already-selected candidate; it never re-ranks, re-scores, or re-selects candidates.
- Conservative defaults, no fixed percentage reduction, no jump cuts inside words, no aggressive cleanup, kill-switch required.

## 3. Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Python sidecar decision engine (`smart_pacing.py`, stdlib only) | Mirrors the established CV-sidecar pattern; stdlib-only means zero new dependencies; isolates decision logic from render-critical Rust code. |
| Rust consumer (`pacing.rs`) owns all timeline surgery | Word remapping, framing rebuild, filtergraph construction, and caption regeneration must be atomic with the render — Rust is the single source of truth for the output timeline. |
| FFmpeg `silencedetect` as the acoustic witness | Uses the same FFmpeg binary the render path already depends on; `-35dB:d=0.10` is strict enough to reject breath-noise-heavy "gaps". |
| Clip-relative silences in the engine API, absolute internally | Matches how the Rust side measures (clip-relative `-ss`/`-t` probe) while keeping word coordinates absolute — one explicit shift, tested. |
| `skipped`/`invalid`/engine-failure → legacy render | Guarantees the feature can never make rendering WORSE than the 7.0 baseline; failure mode is a no-op, not a broken clip. |
| `pacing_inspect` binary | Read-only end-to-end inspection (plan + remapped words + paced framing + filtergraph + ASS/SRT + summary) for forensic verification without touching production state. |

## 4. Implementation Summary (what was built)

- **`autoshorts/src-tauri/scripts/smart_pacing.py`** (~740 lines): `plan_pacing(words, clip_start, clip_end, silences)` — normalization, edit proposal (priority ladder), clearance/budget/tiling guards, JSON contract emission. Thresholds: LEAD 0.65/0.30, TRAIL 0.85/0.45, SENT 1.25/0.50, MID 1.75/0.60, HES 0.90/0.45, PAUSE_MIN_REMOVAL 0.40, MAX_REMOVED_FRACTION 0.40, MIN_OUTPUT_SEC 3.0, MIN_RETAINED_PIECE 0.60, SNAP_CLEARANCE, EDGE_WINDOW 0.10/EDGE_TOL 0.02, SILENCE_TOL 0.04.
- **`autoshorts/src-tauri/src/pacing.rs`**: plan parsing/validation (rejects non-ok/noop/invalid → legacy), word remapping to output timeline, framing-segment rebuild (`pacedFraming`), trim/concat filtergraph construction, ASS/SRT regeneration, kill-switch (`AUTOSHORTS_SMART_PACING=0|false|off`), script discovery (bundled + 4 dev paths), 16 unit tests.
- **`autoshorts/src-tauri/src/bin/pacing_inspect.rs`**: CLI inspection binary emitting a single clean JSON document (stdout purity enforced — telemetry on stderr).
- **`lib.rs` wiring (additive)**: render path consults the pacing plan; on `ok` renders the paced timeline, else legacy.
- **`test_smart_pacing_suite.py`** (workspace root, 55 tests) and **`scratch/smart_pacing_real_validation.py`** (7-style real-footage validation).

## 5. Engine Bugs Found & Fixed During Verification (4 total)

1. **Coordinate-space bug:** silences were consumed as absolute before normalization; now explicitly shifted `[(s + clip_start, e + clip_start)...]` after `normalize_silences`. (Discovered via suite case failures — plans were querying the wrong timeline region.)
2. **Empty-edits invalid plan:** zero verified edits emitted a malformed plan; now returns `{"status": "skipped", "reason": "no removable silence found - original kept"}`.
3. **Trial-tuple crash (ValueError):** the clearance filter built `trial = kept + [tuple]`, mixing dicts into `word_clearance_ok`'s `for cs, ce in cuts` iteration — crashed with 2+ edits. Now a pure tuple list. (Discovered via CLI fixture with 3 edits.)
4. **Edge-sliver cancellation:** a retained piece shorter than `MIN_RETAINED_PIECE` touching a clip edge, containing no words, and verified silent used to CANCEL the entire adjacent edit (case F: a 0.15s leading sliver cancelled a valid false-start removal). Now absorbed into the adjacent edit with a note. (Discovered via suite case F.)

Each fix is regression-covered in the 55-test suite; the full suite and cargo tests were re-run green after every fix.

## 6. Rust-Side Fixes During Verification (2 total)

1. **CWD-dependent script discovery:** `find_smart_pacing_script()` dev paths missed the workspace-root working directory; added `autoshorts/src-tauri/scripts` as a 4th candidate — `pacing_inspect` now works from any CWD.
2. **stdout JSON pollution:** a `println!` summary line broke JSON consumers; moved to `eprintln!` — stdout is now pure JSON.

Both fixes verified by rebuild + `cargo test --lib` **178 passed, 0 failed, 1 ignored**.

## 7. Test Suite Results (test_smart_pacing_suite.py — 55/55, true exit 0)

- Cases A–N cover: edit-priority ladder (leading/trailing dead air, sentence/mid internal pauses, hesitation/false-start), budget cap ≤40%, min output 3.0s, duration arithmetic, no partial-word cuts (snap clearance), retained-piece tiling exactness, silence-verification gating, uncertainty → keep, kill-switch, invalid-plan rejection, and edge classification.
- CLI/MUSIC section runs the REAL engine as a subprocess against ffmpeg-generated audio fixtures (tone+silence WAVs): the engine finds all 3 edit types (leading [0, 0.7], internal [5.2, 6.75], trailing [11.4, 12.0]; 2.85s/12s = 23.75% removed) and **blocks all cuts on a music fixture**.
- Exit code verified via PowerShell `$LASTEXITCODE` after stderr suppression: **0**.

## 8. Inherited Regression Suites (all green — no protected system touched)

| Suite | Result |
|-------|--------|
| `cargo test --lib` (179 total incl. 16 pacing tests) | **178 passed, 0 failed, 1 ignored** |
| DualFrame (`test_dualframe_suite.py`) | **55/55** |
| Active Speaker (`test_active_speaker_suite.py`) | **58/58** |
| Caption templates (`test_caption_templates_suite.py`) | **14/14** |
| Ultralytics adapter (`test_ultralytics_adapter_suite.py`) | **5/5** |
| Multimodal hook (`test_multimodal_hook_suite.py`) | **6/6** |
| Sampling memory (`test_sampling_memory_suite.py`) | **9/9** |
| DualFrame real render (`test_dualframe_real_render.py`) | **5/5** (A/V sync Δt=0.0000s) |
| Hook closure (`test_hook_closure_suite.py`) | **47/47** |
| Frontend `npm run build` | **exit 0** |

## 9. pacing_inspect End-to-End Verification (real footage, full paced path)

On the Beat clip (38.92–48.0s, real leading silence 0.795s@38.92–39.72 verified at −35dB):
- Plan: `ok`, 1 edit (`leading_dead_air` 0.56s), output 8.52s.
- `remappedWords`: 16 words shifted −0.56s onto the output timeline.
- `pacedFraming`: framing segments rebuilt to 8.52s.
- `pieces` + `filtergraph`: single retained piece, `trim=start=0.56:end=9.08` — a REAL timeline surgery, not a nominal duration change.
- ASS: 16 dialogue lines on the new timeline; SRT: 4 cues — captions refer to the OUTPUT timeline, satisfying the spec's "no stale timeline" requirement.
- Clean single-JSON stdout from workspace root and from `src-tauri`.

## 10. Real-Footage Validation — 7 Speaking Styles (26/26 PASS)

| # | Style | Source | Result | Evidence |
|---|-------|--------|--------|----------|
| 1 | Slow meditative + leading dead air | Beat Emotional Fatigue 38.92–48.0 | **CUT: leading_dead_air 0.56s** (9.08→8.52s) | Real render probe = 8.52s exact; budget 6.2%; tiling exact; no partial-word cuts |
| 2 | Fast dense speech | Speed edit 0–12s | **SKIP (correct)** | Max real silence 0.17s (silencedetect scan) — genuinely dense; nothing provable to remove |
| 3 | Music video | Rick Astley 30–42s | **SKIP, zero cuts (correct)** | Music is never silence-verified → no cut can be proven |
| 4 | DB transcript end-to-end | bd3dc4b4 @ 4024–4040 (39 real DB words) | **SKIP (correct, conservative)** | 1.19s transcript gap verifies only 0.83s at −35dB < 0.90s hesitation threshold — the engine demands MORE proof than the transcript suggests |
| 5 | Two-speaker + leading dead air | Messi 0–12s (real 2.71s lead-in silence) | **CUT: leading_dead_air 2.48s** (12.00→9.52s) | Real render probe = 9.52s exact; budget 20.7%; tiling exact |
| 6 | Hindi speech | Hindi video 0–12s | **SKIP (correct)** | No verified silence ≥ threshold in range |
| 7 | Final-seconds outro | Beat 80–93.36 | **SKIP (correct)** | Conservative on outro content |

**The output must SOUND better — evidence:** both real cuts remove proven dead air (leading silences verified at −35dB), rendered outputs probe at exactly the planned durations (8.52s, 9.52s), and every skipped case is a case where silence could NOT be proven — the system never cuts speech, music, or uncertain audio.

## 11. Conservative-Skip Evidence (the spec's core requirement, proven)

- **Music video:** zero cuts — music never passes the silence gate.
- **DB 1.19s transcript gap:** only 0.83s verifies acoustically → below the 0.90s threshold → NO cut. The engine is deliberately stricter than transcript gaps.
- **Speed (fast dense):** max silence 0.17s → nothing removable — the engine does not invent cuts to hit a quota (there is no quota).
- **Outro/final seconds:** skipped — no aggressive cleanup.
- Every skip returns a machine-readable reason (`no removable silence found - original kept`).

## 12. Timeline Integrity Proof (no stale references)

For every paced plan (suite + real footage): (a) output duration = clip duration − removed total (arithmetic check); (b) retained pieces tile the output timeline exactly with no gaps/overlaps; (c) no edit boundary falls inside a word (snap clearance); (d) `remappedWords` are the ONLY words passed to caption generation; (e) framing segments are rebuilt from the retained pieces (`pacedFraming`); (f) the filtergraph trims/concats the SOURCE per the plan — audio and video share the same filtergraph, so A/V sync is structural. Legacy path (skip/invalid/failure) is byte-identical to 7.0 rendering.

## 13. Kill-Switch & Failure-Mode Verification

- `AUTOSHORTS_SMART_PACING=0|false|off` → feature fully disabled, legacy render (suite-tested).
- Engine non-zero exit / malformed JSON / non-ok status → legacy render, no partial pacing (cargo-tested).
- Missing transcript words → no pacing (no-op).
- The feature can never make output worse than the frozen 7.0 baseline.

## 14. AutoShorts 7.0 Freeze Verification

- Workspace `d:\College\Autoshorts 7.0` scanned: **no file modified since 2026-09-14** (all modification dates predate the 8.0 work).
- All 8.0 changes live in `d:\College\Autoshorts 8.0` only.

## 15. Scope Discipline Audit (protected systems, §23)

| Protected system | Touched? |
|---|---|
| Ultralytics CV backend (`speaker_tracker.py`) | **NO** — byte-identical |
| DualFrame core | **NO** |
| Active Speaker tracking | **NO** |
| Hook detection & scoring (`llm.rs`) | **NO** |
| Candidate discovery (`llm.rs`) | **NO** — pacing runs strictly after selection |
| Clipping pipeline (`media.rs`) | **NO** — paced render reuses the same trim/concat mechanics via the filtergraph; no pipeline semantics changed |

New files: `smart_pacing.py`, `pacing.rs`, `pacing_inspect.rs`, `test_smart_pacing_suite.py`, scratch validation scripts. Additive wiring only in `lib.rs`.

## 16. Remaining Limitations & Recommendations

1. **Gap-rich DB transcripts without local video** (c7ecb0fa 5.80s gap, 23d79265 14.1s lead-in, ac710778 209 gaps, 06e54296 60 gaps): cannot be validated end-to-end because the source videos are not on disk. Recommendation: re-download or point the DB at local sources, then re-run `scratch/smart_pacing_real_validation.py` style 4 for each.
2. **Word-timeline dependency:** pacing engages only when a NormalizedTranscript with word-level timings exists; clips without word timings render legacy (safe no-op).
3. **Hesitation/false-start removal (priority 4)** is high-confidence-only and rarely fires on real footage — by design. Aggressive cleanup remains out of scope per spec; a future tuning pass could revisit thresholds only with real-footage A/B evidence.
4. **Threshold calibration:** current thresholds (LEAD 0.65, TRAIL 0.85, SENT 1.25, MID 1.75, HES 0.90) proved conservative-correct on all 7 styles; if retention data later shows over-conservatism, tune with the same prove-it-safe methodology — never by percentage quotas.
5. **Debug artifacts at workspace root** (`pacing_stderr.txt`, `pacing_stdout.json`) can be deleted; scratch/ and tmp/ validation artifacts are non-production.

---

## FINAL VERDICT

**PASS.** Retention-Aware Smart Pacing is implemented, tested (55/55 suite, 178 cargo, all inherited suites green), and **proven on real footage across 7 speaking styles (26/26)** — with real cuts rendering at exactly the planned durations and every unprovable case conservatively skipped. The feature is additive, kill-switchable, fails safe to the byte-identical 7.0 legacy render, and touched zero protected systems. AutoShorts 7.0 remains frozen.
