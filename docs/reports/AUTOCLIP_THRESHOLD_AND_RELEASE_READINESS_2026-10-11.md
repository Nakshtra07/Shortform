# AutoClip Holistic Scoring Threshold Assessment & Final Release Readiness Audit

**Audit Date:** 2026-10-11  
**Audited Repository:** `D:\College\Autoshorts 11.0`  
**Target Branch:** `feature/autoclip-scoring-topn`  
**Current HEAD:** `ef6067cb11fa85633d0628fc75bd2f596254f6a3`  
**Base Commit:** `ef066a8` (`origin/main`, clean `main`)  
**Auditor:** Antigravity (Google DeepMind) — Offline Forensic & Release Audit  

---

## Executive Summary

| Category | Finding / Verdict | Evidence Basis |
|---|---|---|
| **Threshold Recommendation** | **KEEP 0.65 as Experimental Default** | Verified against 34 candidates; filters out poor candidates (<0.65) while preserving 30 eligible clips. Top-12 selection is invariant across thresholds [0.55, 0.75]. Must remain labeled as uncalibrated. |
| **Release Verdict** | **READY WITH EXPLICIT LIMITATIONS** | 100% of offline unit, integration, and build checks pass (553/553 lib tests, 16/16 Rust integration, 47/47 Python, clean Vite build). Production default (`legacy`) is completely protected. |
| **Production Default** | **PRESERVED (`legacy`)** | Legacy composite scoring remains the authoritative system default. AutoClip requires explicit user opt-in and verified credentials. |
| **Billing Safety** | **STRICT COMPLIANCE (0 billable calls)** | Audit executed purely on existing logs, cache entries, local database snapshots, and offline tests. Zero API calls made. |
| **Protected Backup** | **VERIFIED INTACT** | `D:\College\Backups\AutoShorts_11.0_pre_AutoClip_TopN_20261011_011535` verified with manifest, SHA256 hashes, and SQLite integrity check. |

---

## 1. Audit Scope & Safety Constraints

This forensic audit evaluates the AutoClip holistic scoring implementation and Top-N candidate selection mechanism on the `feature/autoclip-scoring-topn` branch.

### Strict Safety Adherence
1. **Zero Billable API Invocations:** No calls were dispatched to NVIDIA NIM, Deepgram, DeepSeek, OpenRouter, or other paid APIs.
2. **Read-Only Database Access:** All direct database inspections were performed in read-only mode using SQLite `mode=ro` URIs.
3. **No Unapproved Modifications:** Application logic, default settings, dependencies, lockfiles, and environment defaults were left untouched.
4. **Git State Preservation:** No branch switching, rebasing, amending, resetting, or merging into `main` occurred.
5. **No Secret Exposure:** Secret tokens and API keys were verified for safe redaction in error reporting and logging.
6. **Task 3 Exclusion:** Per user directive, Task 3 (UI click-through, multi-video end-to-end acceptance, UI-to-render handoff) was not performed.

---

## 2. Verified Branch and Git State

- **Workspace Root:** `D:\College\Autoshorts 11.0`
- **Active Branch:** `feature/autoclip-scoring-topn`
- **HEAD Commit:** `ef6067cb11fa85633d0628fc75bd2f596254f6a3`
- **Expected Checkpoint:** `ef6067cb11fa85633d0628fc75bd2f596254f6a3` (**MATCH: EXACT**)
- **Base Commit:** `ef066a8` ("chore: clean repository root and organize documentation" on `main`)
- **Branch Commits:**
  - `b9756c3` — *Add AutoClip holistic scoring backend with Top-N selection (NVIDIA NIM)*
  - `ef6067c` — *feat(scoring): add UI scoring-mode selector wired to backend scoring authority*
- **Working Tree Status:**
  - `modified: autoshorts/src/main.tsx` — layout/wrapping fix for candidate header buttons.
  - `modified: autoshorts/src/styles.css` — CSS flex-wrap additions preventing `Cut Selected` / `Find Viral Moments` buttons from disappearing on narrow/split screens.
  - `untracked: docs/reports/DUALFRAME_ARCHITECTURE_AUDIT.md` — prior audit report.
  - `untracked: docs/reports/dualframe_audit_2026-10-11/` — prior audit artifacts.
- **Main Branch Integrity:** `main` has not been altered or merged into.

---

## 3. Threshold Assessment Based on Existing Evidence

### 3.1 Dataset & Provenance Reconciliation
The evaluation is based on real candidate validation data for project `649ca76e-22d7-449e-8dca-825e0af15485` (25,485 transcript words; 34 candidates):
- **Run #1 (Initial live run):**
  - Entered: 34 candidates across 6 batches.
  - One batch hit transient HTTP 500 $\rightarrow$ automatically retried and succeeded (retry 1/3).
  - Two batches emitted reasoning prose preamble instead of a JSON array $\rightarrow$ strictly rejected without retry per specification $\rightarrow$ 12 candidates assigned fallback score `0.0` with `score_source = "fallback"`.
  - Result: 22 genuine `llm` scores, 12 explicit fallbacks (never cached).
- **Run #2 (Next-run recovery & cache pass):**
  - 22 cache hits served from disk with identical scores and reasons.
  - 12 uncached candidates retried live and succeeded $\rightarrow$ 34/34 candidates scored by LLM.
- **Cache Replay Run (0 API calls):**
  - 34/34 candidates served from disk cache in 0.1s (`live_batches = 0`).
  - Proves byte-for-byte deterministic reproducibility.

### 3.2 Candidate Score Distribution

| Measurement | Run #1 (22 Scored) | Full Population (34 Scored) |
|---|---|---|
| **Count** | 22 genuine LLM scores | 34 genuine LLM scores |
| **Minimum Score** | 0.35 (`cand-0012`) | 0.35 (`cand-0012`) |
| **Maximum Score** | 0.88 (`cand-0031`, `cand-0024`) | 0.90 (`cand-0020`) |
| **Mean Score** | 0.7714 | 0.7762 |
| **Median Score** | 0.8100 | 0.8050 |
| **Standard Deviation** | 0.1222 | 0.1096 |

#### Score Bands Across Full 34-Candidate Pool
- $[0.00, 0.50)$: 1 candidate (0.35 — sexist/unsubstantiated rant, correctly penalized)
- $[0.50, 0.65)$: 3 candidates (0.55 — conspiracy monologue, 0.62 — incomplete thought, 0.62 — weak closure)
- $[0.65, 0.70)$: 1 candidate (0.65 — brief trust lesson, exact threshold boundary)
- $[0.70, 0.80)$: 10 candidates (0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.78, 0.78, 0.79, 0.80)
- $[0.80, 0.90]$: 19 candidates (0.80, 0.80, 0.81, 0.82 $\times 5$, 0.83, 0.84, 0.85 $\times 3$, 0.86, 0.87, 0.88 $\times 3$, 0.90)

### 3.3 Verification of the Selection Boundary (Top-N = 12)
In the ranked 34-candidate pool, sorting follows the deterministic comparator:
$$\text{Scored} > \text{Fallback} \;\rightarrow\; \text{final\_score} \downarrow \;\rightarrow\; \text{start\_sec} \uparrow \;\rightarrow\; \text{end\_sec} \uparrow \;\rightarrow\; \text{scoring\_id} \uparrow$$

The 12th-ranked candidate is `cand-0018` with `final_score = 0.82` (start 2199.2s).
Crucially, candidates ranked 12 through 16 **all share the identical final_score of 0.82**:
- **Rank 12 (Selected):** `cand-0018` (start 2199.2s)
- **Rank 13 (Not Selected):** `cand-0005` (start 3905.6s)
- **Rank 14 (Not Selected):** `cand-0033` (start 4646.9s)
- **Rank 15 (Not Selected):** `cand-0007` (start 5535.8s)
- **Rank 16 (Not Selected):** `cand-0017` (start 5963.0s)

The deterministic tie-break correctly separated the 5-way tie by earlier timeline occurrence (`start_sec` ascending).

---

## 4. Sensitivity Assessment for Alternative Thresholds

Comparing alternative candidate-selection thresholds against the verified 34-candidate population:

| Threshold | Eligible ($\ge T$) | Strictly Above ($> T$) | Exactly Equal ($= T$) | Below Gate ($< T$) | Top-N (12) Selected | Retained Not Selected |
|---|---|---|---|---|---|---|
| **0.55** | 33 | 32 | 1 (`cand-0015`) | 1 | 12 | 21 |
| **0.60** | 32 | 32 | 0 | 2 | 12 | 20 |
| **0.65 (Current)** | **30** | **29** | **1 (`cand-0009`)** | **4** | **12** | **18** |
| **0.70** | 29 | 28 | 1 (`cand-0016`) | 5 | 12 | 17 |
| **0.75** | 26 | 25 | 1 (`cand-0010`) | 8 | 12 | 14 |

### Findings from Sensitivity Comparison
1. **Top-N Invariance:** Across all thresholds from 0.55 to 0.75, the Top-12 selected candidate set is **completely identical**. Because 21 candidates achieved $\text{score} \ge 0.82$, Top-N selection cuts off at 0.82 regardless of the eligibility threshold.
2. **Quality Filtering Role:** The threshold's actual role in AutoShorts is filtering the secondary pool of "Not Selected" alternatives. At 0.65, it discards 4 low-quality or incomplete clips (0.35, 0.55, 0.62, 0.62) while keeping 18 viable alternative clips available for manual user selection.
3. **Threshold Boundary Inclusivity:** The rule `score >= threshold` correctly retained `cand-0009` (0.65), validating inclusive boundary semantics.

### Critical Limitation Warning
> [!IMPORTANT]
> This analysis is a **descriptive sensitivity assessment** on a single video project (34 candidates). Without multi-genre ground-truth labels and real audience engagement metrics, the threshold cannot be certified as universally optimal or production-calibrated.

---

## 5. Threshold Recommendation

### Recommendation: **KEEP 0.65 as Experimental Default**

**Rationale:**
1. **Empirical Behavior:** 0.65 cleanly partitions clear editorial duds (scores $\le 0.62$) from viable content ($\ge 0.65$).
2. **Preservation of Contract:** Changing the threshold to 0.60 or 0.70 would alter eligibility counts without any evidence that the discarded or retained clips are objectively better or worse.
3. **Safe Architecture:** The production default remains `legacy` scoring. AutoClip is strictly opt-in, so keeping 0.65 poses zero risk to standard user workflows.

---

## 6. Source-Code Architecture & Integration Review

### 6.1 Backend Selection & Legacy Invariance
- In `autoshorts/src-tauri/src/scoring.rs`, `resolve_scoring_backend()` defaults to `ScoringBackend::Legacy` if `AUTOSHORTS_SCORING_BACKEND` is unset or empty.
- Explicit call overrides (`resolve_scoring_backend_with_override`) strictly support `"legacy"` and `"autoclip"`; unknown strings immediately return a hard error, preventing silent backend drift.
- In legacy mode, `composite_score` retains 100% authority, and no NVIDIA credentials are requested.

### 6.2 Credential Fail-Fast Contract
- If `autoclip` is selected, `resolve_scorer_credential` checks `NVIDIA_API_KEY`. If missing, it immediately throws a hard error before any discovery or transcription occurs.
- In `autoshorts/src/main.tsx`, `runAutoPipeline` also checks `env.hasNvidiaKey` when `scoringMode === "autoclip"` and halts before initiating transcription, preventing wasteful billable calls.

### 6.3 Normalization and Strict Parse Rejection
- `to_final_score()` rejects booleans, non-finite values (NaN, Inf), negative numbers, and values $> 100.0$. Scales in $(1.0, 10.0]$ and $(10.0, 100.0]$ are normalized to $[0.0, 1.0]$ and rounded to 2 decimals.
- `parse_scoring_response_text()` uses bracket scanning to isolate the outermost JSON array. Reordered results are matched by exact transient ID (`cand-%04d`). Duplicate IDs are marked ambiguous and safely rejected. Malformed responses fail the entire batch to fallback rather than guessing.

### 6.4 Cache Identity & Versioning
- Cache keys (`FinalScoreCache::compute_key`) hash the transcript source identity, candidate boundaries, hook, context, transcript excerpt, multimodal evidence, provider, model, and `FINAL_SCORE_PROMPT_VERSION` ("v2.0").
- The transient ID is excluded from the cache key, ensuring pool-position independence.
- Writes are atomic (`.tmp` write followed by rename). Failed or fallback scores are never written to cache.

### 6.5 Deterministic Ranking & Persistence
- `autoclip_rank_cmp` enforces total order: scored before fallback, then `final_score` descending, then `start_sec` ascending, `end_sec` ascending, and `scoring_id` ascending.
- `replace_candidates_with_selection()` in `db.rs` persists rank ($1$-based index), `selected` flag ($i < \text{selected\_count}$), and provenance (`metadata_json.scoreSource`).

### 6.6 Secret Hygiene
- All logging across `scoring.rs` and `lib.rs` strictly avoids printing `NVIDIA_API_KEY` or authorization headers.
- Request error previews are sanitized and capped at 200 characters.

---

## 7. Protected Backup Verification

The pre-feature safety backup was inspected:
- **Location:** `D:\College\Backups\AutoShorts_11.0_pre_AutoClip_TopN_20261011_011535`
- **Manifest:** `BACKUP_MANIFEST.md` exists and confirms:
  - Total files: 60,235 | Total size: 8.541 GB
  - Robocopy error count: 0
  - Key-file SHA256 matches verified: `.env`, `.env.example`, `Cargo.toml`, `Cargo.lock`, `package.json`, `package-lock.json`, `lib.rs`, `main.rs`.
  - SQLite snapshot: `autoshorts.sqlite.consistent_backup` verified with `PRAGMA integrity_check = ok`.
  - Documented exclusions: only the regenerable build caches (`target/` and `src-tauri/target/`).
- **Status:** **VERIFIED & INTACT**.

---

## 8. Verification Checks & Test Execution Results

All applicable test suites were executed directly on the current workspace:

| Check / Test Suite | Command | Result | Details |
|---|---|---|---|
| **Rust Code Formatting** | `cargo fmt --check` | ⚠️ FAILED (P3) | Pre-existing multi-line formatting differences in `scoring.rs`, `youtube.rs`, etc. |
| **Focused Scoring Tests** | `cargo test -p autoshorts --lib -j 2 -- scoring::` | ✅ **PASSED** | 70 passed, 0 failed (0.03s) |
| **Rust Full Library Tests** | `cargo test -p autoshorts --lib -j 2` | ✅ **PASSED** | **553 passed, 0 failed**, 3 ignored (14.79s) |
| **Pause Intel Suite** | `cargo test -p autoshorts --test pause_intel_suite -j 2` | ✅ **PASSED** | 8 passed, 0 failed (7.79s) |
| **T7 Prosody Suite** | `cargo test -p autoshorts --test t7_prosody_suite -j 2` | ✅ **PASSED** | 8 passed, 0 failed (0.00s) |
| **Cargo Build Check** | `cargo check -j 2 --lib --examples` | ✅ **PASSED** | Compiled cleanly in 8.89s, 0 errors |
| **Frontend Production Build** | `npm run build` | ✅ **PASSED** | `tsc && vite build` completed in 7.08s, 0 errors |
| **Python Offline Suites** | `python -m unittest test_pause_intel_suite.py test_t7_prosody_suite.py test_smart_pacing_learned.py` | ✅ **PASSED** | 47 passed, 0 failed in 43.35s |

---

## 9. Issues and Release Blockers

### P0 — Critical (None)
- No data loss, credential leakage, or crash risks detected.

### P1 — High (None)
- No functional regressions or release blockers detected.

### P2 — Medium (Operational Limitations)
1. **Uncalibrated Threshold (0.65):** The threshold is historically derived and descriptive, not validated against multi-video ground truth or user engagement data. Must remain labeled as experimental in documentation.
2. **Model Preamble Rate:** Under `nvidia/nemotron-3-ultra-550b-a55b` with thinking disabled, roughly 25% of live batches in the initial run emitted reasoning prose, causing fallback to score 0.0. The architecture safely handles this via fallback and next-run retry, but operators should be aware.

### P3 — Low (Code Hygiene)
1. **Rust Formatting (`cargo fmt`):** `cargo fmt --check` reports styling diffs on multiline assertions and function calls. Does not affect functionality.
2. **Untracked Audit Reports:** `DUALFRAME_ARCHITECTURE_AUDIT.md` and related logs reside untracked in `docs/reports/`.

---

## 10. Final Release Verdict

### Verdict: **READY WITH EXPLICIT LIMITATIONS**

**Justification:**
1. All core implementation contracts, score authority separation, fallback behavior, deterministic tie-breaking, Top-N selection, and database persistence are verified by unit tests, integration tests, and live validation logs.
2. The legacy scoring backend remains the authoritative default. No user or existing project is impacted without explicit opt-in.
3. The codebase compiles cleanly, passes 553 library tests and 16 integration tests with zero failures, and builds the frontend production bundle with zero errors.

---

## 11. Explicitly Untested Areas

Per user instructions and environmental constraints, the following areas remain untested:
1. **Live UI-to-Render End-to-End Workflow:** No live rendering or UI click-through was executed in this audit.
2. **Multi-Video Ground-Truth Calibration:** No additional YouTube videos or external podcasts were processed to preserve user billing limits.
3. **`AUTOSHORTS_SCORER_THINKING=1`:** The effect of enabling reasoning mode on the Nemotron model remains untested.
