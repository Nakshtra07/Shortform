# AutoClip Holistic Scoring + Top-N Selection — Acceptance Report

**Date:** 2026-10-11
**Branch:** `feature/autoclip-scoring-topn` (single descriptive commit; `main` untouched)
**Baseline commit:** `ef066a8` (clean `main`)
**Pre-change backup:** `D:\College\Backups\AutoShorts_11.0_pre_AutoClip_TopN_20261011_011535\` (verified: 60,235 files / 8.541 GB, 0 robocopy failures, key-file SHA256 match, sqlite integrity `ok` + `.consistent_backup`; manifest included. The backup intentionally excludes the two regenerable `target/` build-cache directories — documented in the manifest, not silent.)

---

## 1. What was built

Reconstruction of the deleted AutoShorts 11.1 feature set inside AutoShorts 11.0:

- **`scoring.rs` (new)** — NVIDIA NIM holistic scorer: OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1/chat/completions`), model `nvidia/nemotron-3-ultra-550b-a55b`, temperature 0.15, batch 6, concurrency 2, connect timeout 30 s / request timeout 180 s. Holistic 4-dimension prompt producing a single `final_score` in [0,1] + `recommend_reason` (no sub-score blending). Transient ids `cand-%04d` by pool position; exact-id alignment only (duplicate→ambiguous, unknown→ignored, missing→unscored, never positional); strict normalization ladder (reject bool/NaN/inf/negative/>100; >1 → ÷10 if ≤10 else ÷100; round to 2 decimals); fallback = `score 0.0` + `score_source="fallback"` (genuine 0.0 keeps `"llm"`); versioned score cache (`FINAL_SCORE_PROMPT_VERSION v2.0`) with atomic writes; retry/classification (transient HTTP errors retried; malformed responses strictly rejected, never retried); deterministic comparator; eligibility gate; Top-N selection. ~50 unit tests.
- **Backend switch** — `AUTOSHORTS_SCORING_BACKEND` = `legacy` (production default, unchanged behavior) or `autoclip`; unknown value = config error; explicit `autoclip` with missing `NVIDIA_API_KEY` = hard error before discovery; no silent fallback between backends.
- **Threshold** — default 0.65, inclusive `>=`, env `AUTOCLIP_RECOMMEND_THRESHOLD`.
- **Top-N** — `AUTOSHORTS_TOP_N` default 12; `<= 0` = all eligible; `selected_count = min(N, eligible_count)`; never fabricate or select ineligible; Not Selected alternatives retained.
- **`llm.rs`** — ranking engine parameterized by comparator; legacy wrapper preserves composite_score authority, gates, dedup, temporal balancing, `min(12).max(3)` cutoff; autoclip wrapper uses scored-before-fallback → final_score desc → start asc → end asc → scoring_id asc (`total_cmp`). Historical hardcoded `"v1.0"` cache-version defect fixed (now uses `config.scoring_prompt_version`).
- **`db.rs`** — new `replace_candidates_with_selection` (persists rank = index+1, selected = index < selected_count, provenance via `metadata_json.scoreSource`); legacy `replace_candidates` delegates unchanged.
- **`models.rs`** — `score_source` / `scoring_id` serde-default fields (old rows unaffected).
- **`main.tsx`** — backend-authoritative provenance badges (`llm` / `fallback`), Top-N input via `set_selected_clip_count`, stale-async race fix (`selectionSeqRef`).
- **Docs** — `docs/development/AUTOCLIP_SCORING_BACKEND.md` (operations/contract), `docs/development/AUTOCLIP_TOPN_PRE_IMPLEMENTATION_PLAN.md` (recorded pre-change plan).
- **Live validation harness** — `autoshorts/src-tauri/examples/autoclip_live_validation.rs` (non-destructive: operates on a disposable copy of the production DB).

---

## 2. Verification status

| Area | Status |
|---|---|
| Unit tests (lib) | ✅ 544 passed / 2 failed / 3 ignored — the 2 failures are **pre-existing** yt-dlp environment issues, identical before and after the change |
| Integration suites | ✅ `pause_intel_suite` 8/8, `t7_prosody_suite` 8/8 |
| Frontend build | ✅ `tsc && vite build` clean |
| NVIDIA model availability | ✅ `nvidia/nemotron-3-ultra-550b-a55b` confirmed present in the account's model list (free listing call, 2026-10-11) |
| Live scoring (billable, NVIDIA NIM) | ✅ Executed — see §4 |
| Cache replay attribution | ✅ Executed — see §4 |
| DB persistence roundtrip | ✅ Verified live (per-row asserts) |
| Full UI click-through e2e with live discovery LLM (DeepSeek/other provider) | ❌ **Not performed** — discovery-provider billable calls were outside the authorization for this task |

### 2.1 Baseline vs. after

| Suite | Baseline (main @ ef066a8) | After |
|---|---|---|
| lib tests | 494 passed / 2 failed / 3 ignored | 544 passed / 2 failed / 3 ignored (**+50 new tests**) |
| pause_intel_suite | 8/8 | 8/8 |
| t7_prosody_suite | 8/8 | 8/8 |
| npm run build | ok | ok |

The 2 persistent failures are pre-existing environment issues (yt-dlp not on PATH):
`media::tests::test_find_ytdlp_cmd_discovers_project_venv_install`,
`youtube::tests::test_real_youtube_metadata_probe_on_previously_failing_url`.
They were **not touched, fixed, or hidden** — they are reported here as baseline.

**Transient anomaly, honestly reported:** one full-suite run showed a third lib-test failure (543/3). It occurred while `cargo test` ran in parallel with `npm run build` on this 7.7 GB machine, did not reproduce in two subsequent full runs, and is consistent with resource contention rather than a code defect. All suites were green on re-run with `--no-fail-fast` (lib 544/2/3, both suites 8/8).

`cargo test` must use `-j 2` on this machine (default parallelism exhausts the pagefile, os error 1455).

---

## 3. Defects found by the verification process (and fixed)

1. **Comparator direction bug — caught by the new unit tests before any live call.** The scored-before-fallback ordering needs `b_scored.cmp(&a_scored)`; the first implementation had the comparison inverted. Found by the deterministic ranking tests, fixed, tests green.
2. **Cache-key spec violation — caught by live validation.** The first implementation included the transient `cand-%04d` id in the cache key, making cache hits **pool-position-dependent**. The spec's key list (media identity, timestamps, hook/rationale, transcript excerpt, evidence, provider, model, prompt version) deliberately excludes the transient id. Fixed: the id is no longer part of the key; a new test (`cache_key_is_position_independent`) locks this in. Cache entries written by the pre-fix keying are unreachable and inert (they are simply never read again; they are never returned as results).
3. **Run-summary accounting bug — caught by live validation.** Cache hits did not count toward `valid_scores`, so a fully-served-from-cache run printed a spurious "no usable scoring result" warning while every candidate carried a genuine `llm` score. Fixed: cache hits count as valid scores.
4. **Parse-failure diagnostics — added after live observation.** Malformed model responses now include a 200-char sanitized content preview in the error, so operators can distinguish reasoning-prose, truncation, and dict-style outputs. Strict rejection (no retry, batch falls back) is unchanged and is the specified behavior.

---

## 4. Live validation evidence (2026-10-11)

Authorization: free models-list check + real AutoClip scoring runs against NVIDIA NIM chat-completions (billable). Discovery-provider billable calls were **not** authorized (see §2 scope limit).

**Method.** The harness (`examples/autoclip_live_validation.rs`) copies the production DB to a disposable temp file, picks a real project (id `649ca76e-22d7-449e-8dca-825e0af15485`, 25,485 transcript words, 34 existing candidates), resets the pool to unscored exactly as a fresh discovery pool enters the scorer, runs `scoring::run_autoclip_scoring` **live**, then mirrors the exact `generate_candidates` autoclip branch: eligibility gate → dedup/rank (`deduplicate_and_rank_all_candidates_autoclip`) → Top-N (`effective_selected_count`) → `replace_candidates_with_selection` into the copy → reload and assert the persistence contract. The production DB was never modified; production effect limited to the `final_scores` cache namespace (by design).

Full transcripts (committed with this report, force-added past the global `*.log` ignore):

- `docs/reports/autoclip_live_validation_2026-10-11.log` — the **live** run
- `docs/reports/autoclip_cache_replay_2026-10-11.log` — the pure cache-replay run (0 API calls)

### 4.1 Live run #1 (real NVIDIA NIM calls)

- 34 candidates → 6 live batches (batch size 6, concurrency 2; prompts 10.5k–15k chars), **171.8 s**.
- One batch hit a **transient HTTP 500** → auto-retried per policy and **succeeded** (retry 1/3).
- Two batches returned **reasoning prose instead of a JSON array** (nemotron-3-ultra with thinking off intermittently emits preamble; content preview captured in the log). Per spec these are malformed: **strictly rejected, not retried in-run**, all 6 candidates per batch became explicit unscored fallbacks (`score 0.0`, `score_source="fallback"`).
- Result: **22 genuine `llm` scores** (observed range 0.35–0.91) with `recommend_reason`s, **12 explicit fallbacks**.
- **Eligibility gate (>= 0.65):** 3 of 34 removed; unscored fallbacks retained (they are alternatives, not fabrications).
- **Ranking:** dedup removed 0; deterministic tie-breaking observed live — e.g. at final_score 0.88, start 2454.2 s ordered before 4131.3 s; at 0.82, starts 3905.6 < 4646.9 < 5963.0 (final_score desc → start asc → scoring_id asc).
- **Top-N:** eligible = 19, `AUTOSHORTS_TOP_N` = 12 → `selected_count = min(12, 19) = 12`.
- **Persistence:** 31 rows written, rank = index+1 asserted per row, exactly 12 selected, 19 Not Selected alternatives retained, `metadata_json.scoreSource` survived the roundtrip (asserted per row).

### 4.2 Live run #2 (cache attribution, from the original pool state)

- **22 cache hits: identical `final_score` AND identical `recommend_reason` asserted per candidate** — cached entries are genuine model scores, replayed deterministically.
- **12 live misses** — the run-#1 fallback batches were **never cached as successes** (failed batches are not cached, per spec) and were legitimately retried. Both retry batches **succeeded** this time → final attribution 34 `llm` / 0 `fallback`. 43.4 s.
- This run also demonstrates **next-run recovery**: candidates that fell back in one run are re-scored live in the next.

### 4.3 Cache-replay run (0 API calls, committed log)

After all 34 candidates were cached, a fresh harness invocation replayed **34/34 from cache in 0.1 s** (`live_batches=0`) and re-verified the entire contract chain end-to-end (gate now removes 4 below 0.65 under the complete score set; 30 ranked; 12 selected; 18 retained; persistence asserts pass).

### 4.4 Observed model behavior worth knowing (operational note)

Roughly 2 of 8 live batches returned reasoning-preamble instead of the JSON array (thinking off). The contract handles this by design: reject → explicit fallback → never cached → next run retries. There is no in-run retry for malformed responses (matching the reference implementation's behavior for scoring parse failures and the master prompt's "never retry unclear/malformed responses"). `AUTOSHORTS_SCORER_THINKING=1` exists as an override for experimentation; its effect on this preamble rate is **untested** and not claimed.

### 4.5 Log artifact caveat (cosmetic)

The live-run log was captured through PowerShell and non-ASCII characters inside the 90-char truncated reason previews (em-dashes, non-breaking hyphens) were mojibake-damaged to `?`. **All ASCII data — ids, scores, sources, counts, batch lines, assertions — is intact.** The cache-replay log was captured byte-accurately and is pristine UTF-8.

---

## 5. Calibration status — INCOMPLETE (by design of this change)

`AUTOCLIP_RECOMMEND_THRESHOLD` default **0.65 is uncalibrated**. It is the historical 11.1 default value, restored verbatim; it has **not** been validated against real publish outcomes in 11.0. It is reported as such wherever it appears (docs, UI). The `legacy` backend remains the production default; no user is switched to autoclip behavior without an explicit opt-in.

---

## 6. Spec-source discrepancy note (documented, not a defect)

The reference repository `D:\College\autoclip\autoclip` (git `beabee8f`, MIT, read-only) does **not** contain the historical 11.1 specifics — no `nemotron` model reference, no `cand-%04d`, no `v2.0` prompt version, no 0.65 threshold, no `final_scores` cache. The user's master prompt is therefore authoritative where values differ; the reference repo contributed **behavioral semantics only** (exact-id alignment rules, normalization ladder, no-fabrication contract, strict parse rejection). Known value differences (master prompt wins in all cases):

| Parameter | Master prompt (implemented) | Reference repo |
|---|---|---|
| Eligibility threshold | **0.65** | 0.7 |
| Fallback score | **0.0** | 0.5 |
| Transcript excerpt cap | **1200 chars** | 600 |
| Batch size / concurrency | **6 / 2** | — |

---

## 7. Files changed

| File | Change |
|---|---|
| `autoshorts/src-tauri/src/scoring.rs` | new — scorer, contracts, ~50 tests |
| `autoshorts/src-tauri/src/lib.rs` | backend resolution + autoclip branch in `generate_candidates`; missing-key hard error |
| `autoshorts/src-tauri/src/llm.rs` | comparator-parameterized engine + wrappers; cache-version defect fix; new ranking tests |
| `autoshorts/src-tauri/src/db.rs` | `replace_candidates_with_selection` + tests |
| `autoshorts/src-tauri/src/models.rs` | `score_source` / `scoring_id` fields |
| `autoshorts/src/main.tsx` | provenance badges, Top-N input, stale-async race fix |
| `autoshorts/.env.example` | documented placeholders for all new env vars (no values) |
| `autoshorts/src-tauri/examples/autoclip_live_validation.rs` | new — live validation harness |
| `docs/development/AUTOCLIP_SCORING_BACKEND.md` | operations/contract doc |
| `docs/development/AUTOCLIP_TOPN_PRE_IMPLEMENTATION_PLAN.md` | pre-change plan (recorded) |
| `docs/reports/AUTOCLIP_TOPN_ACCEPTANCE_REPORT.md` | this report |
| `docs/reports/autoclip_live_validation_2026-10-11.log` | live-run evidence (force-added past `*.log` ignore) |
| `docs/reports/autoclip_cache_replay_2026-10-11.log` | cache-replay evidence (force-added past `*.log` ignore) |

Secret scan performed over every changed/new/log file before commit: **0 occurrences of the actual `NVIDIA_API_KEY` value** (literal-value search) and 0 secret-like patterns beyond pre-existing dummy test fixtures (`nvapi-test-key`, `arg-key-456`) in untouched baseline code.
