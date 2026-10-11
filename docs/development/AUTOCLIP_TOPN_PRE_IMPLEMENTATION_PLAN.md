# Pre-Implementation Plan — AutoClip-Style Scoring & Top-N Selection (AutoShorts 11.0)

Date: 2026-10-11 · Branch: `feature/autoclip-scoring-topn` (from clean `main` @ `ef066a8`) · Backup: verified at `D:\College\Backups\AutoShorts_11.0_pre_AutoClip_TopN_20261011_011535\`

## 1. Existing candidate pipeline (actual, inspected)

`generate_candidates` (`autoshorts\src-tauri\src\lib.rs:719-1315`): transcript load → discovery (`llm::discover_candidates_full_timeline`) → per-draft boundary snapping (`llm.rs:880-989`) → hook quality gate 1b (`lib.rs:991-1013`) → **composite-score gate 1c** (`lib.rs:1015-1030`, `composite_score >= 0.65`) → multimodal analysis (`media::analyze_multimodal_hook_signals`, `lib.rs:1032-1038`) → dedup + adaptive temporal balancing (`llm::deduplicate_and_rank_all_candidates`, `llm.rs:4712-4795`, comparator hardcoded to `composite_score`) → Phase 4 advisory passes (redundancy / PANNs / VLM) → `db.replace_candidates` (`db.rs:460-544`, rank=index+1, selected=index<min(len,12).max(3)).

## 2. Existing score authority

`llm::composite_score` (`llm.rs:4939-5053`) is the ranking/gate authority; `CandidateDraft.score` is the raw discovery LLM score persisted to DB. No `score_source`/`scoring_id` fields exist yet.

## 3. Existing Top-N behavior

No true Top-N exists. DB/backend already support: `set_selected_rank_range` (used by UI From/To inputs), `set_selected_clip_count` (registered command, **no UI caller**). Persistence cutoff is `min(len,12).max(3)`.

## 4. Existing schema/persistence

`candidates` table (`db.rs:167-187`) already stores `score`, `rank`, `selected`, `metadata_json`. **No schema migration required** — scoring provenance rides in `metadata_json` via two new serde-default `CandidateDraft` fields (`score_source`, `scoring_id`), keeping old rows deserializable.

## 5. Source-level AutoClip reference findings (`D:\College\autoclip\autoclip` @ `beabee8f`, MIT, read-only)

Current AutoClip has NO NVIDIA NIM specifics (no `integrate.api.nvidia.com`, nemotron, `cand-%04d`, prompt-version constants, 0.65 threshold, score cache). It contributes **semantics**: holistic 4-dimension scoring prompt principles (info value / emotional resonance / shareability / structural completeness), one `final_score` [0,1] + `recommend_reason`, appended output contract with no-fabrication instruction, score normalization ladder (`bool`→reject; non-finite/<0/>100→reject; >1→÷10 if ≤10 else ÷100; round 2), exact-id alignment (duplicate id→ambiguous→don't assign; unknown→ignore; never positional), fallback provenance (`score_source`), `>=` threshold comparison, failures never cached. **The master-prompt historical values are authoritative where they differ** (threshold 0.65, fallback score 0.0, cand-%04d ids, 1200-char excerpt, 3×160-char evidence, batch 6, concurrency 2, temp 0.15, max_tokens 4096, NVIDIA NIM endpoint + `nvidia/nemotron-3-ultra-550b-a55b`, `FINAL_SCORE_PROMPT_VERSION` v2.0, Top-N 12, unscored never eligible/selected).

## 6. Missing integration points

1. No scoring backend switch (`AUTOSHORTS_SCORING_BACKEND`).
2. No NVIDIA holistic scorer module, cache, or identity/alignment layer.
3. Ranking engine comparator not parameterizable; REZE window cache key hardcodes `"v1.0"` instead of `WindowDiscoveryConfig.scoring_prompt_version` (`llm.rs:1399` — the historical 11.1 defect, present here).
4. No explicit Top-N selection at persistence (cutoff is implicit `min(12).max(3)`).
5. UI: no Top-N control; `updateSelectionRange` has a stale-async response race (unsequenced invokes, closure-captured `detail`).

## 7. Files that will change

| File | Change |
|---|---|
| `src-tauri/src/scoring.rs` (NEW) | AutoClip scorer: config, cand-%04d identity, prompt, bounded NVIDIA NIM requests, parse/normalize, exact-id alignment, fallback semantics, versioned `final_scores` cache, diagnostics, tests |
| `src-tauri/src/models.rs` | Add `score_source`, `scoring_id` (serde default) to `CandidateDraft` |
| `src-tauri/src/llm.rs` | Fix cache-version defect (use `config.scoring_prompt_version`); comparator-parameterized dedup/rank engine + legacy & AutoClip wrappers |
| `src-tauri/src/db.rs` | `replace_candidates_with_selection(project_id, drafts, selected_count)`; legacy `replace_candidates` semantics unchanged |
| `src-tauri/src/lib.rs` | `mod scoring;` backend resolution + AutoClip branch in `generate_candidates` (scoring after multimodal, before gate; eligibility gate retains unscored; AutoClip ranking; Top-N; explicit selection persistence; diagnostics) |
| `src/main.tsx` | Provenance badges, Top-N control (existing `set_selected_clip_count`), serialized selection updates (race fix), counts |
| `.env.example`, `docs/development/AUTOCLIP_SCORING_BACKEND.md` | Configuration + documentation |

## 8. Baseline (pre-change) test results

- `cargo test` (lib): **494 passed, 2 failed, 3 ignored**. Both failures are pre-existing environment issues (missing `yt-dlp` executable): `media::tests::test_find_ytdlp_cmd_discovers_project_venv_install`, `youtube::tests::test_real_youtube_metadata_probe_on_previously_failing_url`.
- `cargo test --test pause_intel_suite`: 8/8 ok. `--test t7_prosody_suite`: 8/8 ok.
- `npm run build` (tsc + vite): ok.
- Environment notes: 7.7GB RAM — full `cargo test` requires `-j 2` (default parallelism exhausts the pagefile); the stale 32GB `target/` cache referenced a dead `Autoshorts 8.0` path and was fully cleaned/rebuilt.

## 9. Minimum implementation plan

1. `scoring.rs` module with unit tests (config, normalization ladder, alignment, prompt, cache keying, comparator/threshold/Top-N resolution).
2. `models.rs` fields → `llm.rs` engine refactor + defect fix → `db.rs` persistence → `lib.rs` pipeline wiring.
3. Frontend minimal wiring.
4. Docs + report; rerun full suites; live validation gated on user authorization.
