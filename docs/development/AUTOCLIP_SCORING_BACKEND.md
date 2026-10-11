# AutoClip Scoring Backend — Architecture, Contract and Operations

This document describes the reconstructed AutoClip-style holistic candidate
scoring and Top-N selection integration in AutoShorts 11.0 (branch
`feature/autoclip-scoring-topn`). It covers architecture, the score
contract, the backend switch, configuration, normalization and alignment,
failure semantics, cache design, threshold/Top-N behavior, testing, known
calibration limitations, and how to reproduce a live acceptance run.

## 1. Architecture and data flow

```
generate_candidates (src-tauri/src/lib.rs)
  ├─ resolve AUTOSHORTS_SCORING_BACKEND          ← default: legacy
  │    └─ autoclip + missing NVIDIA_API_KEY → hard config error (before discovery)
  ├─ candidate discovery (llm::discover_candidates_full_timeline)     [unchanged]
  ├─ boundary snapping + hook/payoff processing (llm.rs:880-989)        [unchanged]
  ├─ hook quality gate 1b (unresolved-pronoun rejection)               [unchanged]
  ├─ legacy composite gate 1c                       [legacy mode ONLY]
  ├─ multimodal analysis (media::analyze_multimodal_hook_signals)       [unchanged; evidence prepared BEFORE scoring]
  ├─ AUTOCLIP MODE:
  │    ├─ scoring::run_autoclip_scoring  (NVIDIA NIM holistic final_score)
  │    ├─ eligibility gate (final_score >= threshold; unscored retained)
  │    └─ llm::deduplicate_and_rank_all_candidates_autoclip (shared dedup +
  │         temporal balancing engine, AutoClip comparator, pure final order)
  ├─ Phase 4 advisory passes (redundancy / PANNs / VLM)                 [unchanged, advisory only]
  └─ persistence
       ├─ legacy:   db.replace_candidates                 (original min(12).max(3) cutoff)
       └─ autoclip: db.replace_candidates_with_selection  (rank=index+1,
                     selected=index < min(effective_top_n, eligible_count))
```

Score authority per mode (logged explicitly on every run):

- **legacy (default):** `llm::composite_score` remains authoritative for
  gating, ranking and selection. No NVIDIA credential is required.
- **autoclip:** the accepted NVIDIA `final_score` is authoritative. The
  legacy composite score, REZE/window-discovery scores, evaluator results
  and multimodal numeric scores never override it. Multimodal evidence is
  prompt context only, never blended into the score.

## 2. The score contract

The scorer evaluates each candidate holistically on four dimensions —
information value, emotional resonance, shareability, structural
completeness — and returns exactly:

- one authoritative `final_score` in `[0.0, 1.0]` (no weighted sub-scores,
  no blending with discovery/composite/multimodal numbers), and
- one concise `recommend_reason` (persisted as the candidate rationale).

Input per candidate (built in `scoring::build_scoring_inputs`):

| Input | Source |
|---|---|
| `id` | transient identity `cand-%04d` from pool position |
| `hook` | `CandidateDraft.hook` (AutoClip `outline` analog) |
| `context` | `CandidateDraft.rationale` (AutoClip `content` analog) |
| `start_seconds` / `end_seconds` | floating-point seconds |
| `transcript` | word-aligned excerpt, capped at 1200 chars |
| `evidence` | up to 3 naturally available multimodal evidence strings, 160 chars each; the v9.2 fallback marker is filtered out; empty when no genuine evidence exists |

Provenance is persisted inside `metadata_json` via two serde-default
`CandidateDraft` fields (no DB schema migration; old rows deserialize as
`None`):

- `score_source`: `"llm"` (genuine model score) / `"fallback"` (explicit
  unscored) / absent (legacy).
- `scoring_id`: the transient identity, persisted for reproducible
  tie-breaks.

## 3. Backend switch

`AUTOSHORTS_SCORING_BACKEND`:

- `legacy` — default. Original scoring, threshold, ranking, dedup and
  persistence semantics, byte-for-byte preserved.
- `autoclip` — explicit NVIDIA holistic scoring mode. Requires
  `NVIDIA_API_KEY`; a missing key is a hard configuration error returned
  before any discovery work.
- any other value — hard configuration error. No silent substitution,
  no silent fallback between backends, no cross-backend cache reuse.

## 4. NVIDIA NIM configuration

| Setting | Default | Override |
|---|---|---|
| Endpoint | `https://integrate.api.nvidia.com/v1/chat/completions` | `NVIDIA_SCORER_BASE_URL` (base URL; `/chat/completions` appended when missing) |
| Model | `nvidia/nemotron-3-ultra-550b-a55b` | `AUTOSHORTS_SCORER_MODEL` |
| Credential | — | `NVIDIA_API_KEY` (required in autoclip mode) |
| Reasoning | off | `AUTOSHORTS_SCORER_THINKING` |
| Temperature | 0.15 | fixed |
| Max output tokens | 4096 | `AUTOSHORTS_SCORER_MAX_TOKENS` |
| Batch size | 6 | fixed |
| Concurrency | 2 | fixed (semaphore-bounded) |
| Connect timeout | 30 s | fixed |
| Request timeout | 180 s | fixed |

Retries: transient failures (HTTP 429 and eligible 5xx, timeouts) are
retried up to 3 times with exponential backoff. Invalid-configuration 4xx
errors fail fast and are never retried indefinitely.

> Note: the default model identifier is the historic AutoShorts 11.1
> target. If NVIDIA NIM no longer serves it, keep the implementation and
> report the live-validation blocker — do not silently substitute another
> model (that would change scoring behavior and potentially invalidate the
> threshold).

## 5. Normalization and ID alignment

Normalization ladder (`scoring::to_final_score`, strict — invalid values are
rejected, never clamped):

- JSON booleans rejected (not scores).
- Numbers and numeric strings parsed.
- Non-finite (NaN / ±inf), negative, > 100 → rejected.
- Values in (1.0, 10.0] → interpreted as 0-10, divided by 10.
- Values in (10.0, 100.0] → interpreted as 0-100, divided by 100.
- Accepted scores rounded to 2 decimals.

Alignment (`scoring::align_final_scores`, identity-based — NEVER positional):

- exact unique `id` → aligned to its candidate;
- duplicate `id` → ambiguous, neither result assigned;
- unknown `id` → ignored (counted/logged);
- missing `id` → candidate stays unscored;
- reordered/partial responses still align correctly;
- a result is never assigned to a candidate because of array position.

## 6. Failure and fallback semantics

- Omitted, invalid or unrecoverable scores become the documented unscored
  representation: `score = 0.0`, `score_source = "fallback"`.
- No fabricated confidence scores (e.g. a default 0.5) are ever invented.
- Fallback candidates are visibly marked in the UI ("Unscored (fallback)"),
  are never eligible for AutoClip Top-N selection, never enter the
  successful score cache, and never silently trigger legacy scoring.
- A genuine model score of `0.0` retains `score_source = "llm"` — zero is a
  valid score, not "unscored".
- A failed scoring run never destroys candidate records: the pool is
  persisted with explicit fallback markers, and the failure is reported in
  the diagnostics (`[AutoClip Scoring] RUN FAILED: no usable scoring result`).

## 7. Cache design

- Dedicated, versioned namespace:
  `<cache_dir>/autoshorts/final_scores/final_score_<sha256>.json`
  (separate from REZE/window-discovery caches; `dirs::cache_dir()`).
- Key covers every materially different input: source/media identity,
  candidate identity, start/end timestamps, hook, context, exact transcript
  excerpt, included multimodal evidence, provider, model, and the single
  authoritative prompt version `FINAL_SCORE_PROMPT_VERSION = "v2.0"`.
- Cache hits preserve `score_source = "llm"` (they ARE genuine model
  scores); misses are visible in the per-run diagnostics.
- Only successful model scores are cached; failures/unscored results never
  are. Writes are atomic (`.tmp` + rename). Per-candidate keys make
  cross-candidate contamination impossible.
- Historical defect fixed in scope: the REZE window-score cache key at
  `llm.rs` hardcoded `"v1.0"` instead of reading
  `WindowDiscoveryConfig.scoring_prompt_version`; it now reads the
  configured version so future bumps invalidate coherently.

## 8. Threshold, ranking and Top-N

- Eligibility threshold: `AUTOCLIP_RECOMMEND_THRESHOLD`, default `0.65`
  (historic 11.1 value, **not** statistically calibrated — see §10).
  Rule: eligible ⟺ `score_source == "llm"` AND `final_score >= threshold`
  (inclusive; a score of exactly 0.65 passes at threshold 0.65).
- Ranking comparator (`scoring::autoclip_rank_cmp`, total order):
  1. valid model-scored before unscored/fallback;
  2. `final_score` descending (`total_cmp`, never subtraction-based);
  3. start ascending; 4. end ascending;
  5. transient scoring identity ascending (final tie-break).
- The final ranked order is a pure comparator sort over the full pool —
  bucket interleaving from temporal balancing never reorders it. Dedup
  rules, temporal diversity and boundary enforcement are unchanged.
- Eligibility gate: scored-below-threshold candidates are removed;
  unscored/fallback candidates are RETAINED as visible Not Selected
  alternatives. Counts (removed / unscored / eligible-but-not-selected)
  are reported separately.
- Top-N: `AUTOSHORTS_TOP_N`, default 12; `<= 0` disables the cutoff
  (select all eligible). `selected_count = min(N, eligible_count)`.
  Exactly the first `selected_count` entries of the ranked pool are
  selected; the rest persist as Not Selected alternatives with scores,
  ranks and provenance intact. Nothing is fabricated to reach N.

## 9. UI behavior

The UI only displays backend state; it never recomputes scores, ranks,
eligibility or Top-N.

- Rank, score, Selected/Pool status and counts: existing controls.
- New: `AutoClip Score` badge (llm provenance) / `Unscored (fallback)`
  badge, read from `metadata_json`.
- New: `Top N` input wired to the existing backend command
  `set_selected_clip_count` (marks ranks 1..N).
- Race fix: selection updates (range and Top-N) are serialized with a
  monotonic request token; a stale async response is discarded so an older
  invoke resolution can never overwrite a newer selection.

## 10. Known calibration limitations

- The `0.65` threshold is the historic default. It was **never calibrated**
  on a fixed representative validation set (English/Hindi/Hinglish). Do
  not treat it as statistically grounded.
- The production default backend remains `legacy` on purpose. Switching the
  default requires: a fixed validation set, repeated controlled runs
  (distinguishing live inference from cached replay), score distributions,
  unscored/failure rates, eligibility/selection outcomes, legacy-vs-AutoClip
  comparisons and a usefulness review of the actual selected clips.
- Status vocabulary: `implemented` / `unit-tested` / `runtime-tested` /
  `calibrated` — this integration is implemented and unit-tested; runtime
  verification depends on NVIDIA NIM availability; calibration is INCOMPLETE.

## 11. Testing

```powershell
cd "D:\College\Autoshorts 11.0\autoshorts\src-tauri"
cargo test -j 2 --lib                 # unit + integration logic (NOTE: -j 2 required on this 7.7GB RAM machine)
cargo test -j 2 --test pause_intel_suite --test t7_prosody_suite
cd "D:\College\Autoshorts 11.0\autoshorts"
npm run build                         # tsc + vite frontend build
```

Focused suites: `cargo test -j 2 scoring::`, `cargo test -j 2 test_replace_with_selection`,
`cargo test -j 2 test_autoclip_ranking`, `cargo test -j 2 test_legacy_ranking`.

## 12. Reproducing a live acceptance run

1. Set in `.env` (or environment): `NVIDIA_API_KEY=...`,
   `AUTOSHORTS_SCORING_BACKEND=autoclip` (optionally
   `AUTOSHORTS_TOP_N=5`, `AUTOCLIP_RECOMMEND_THRESHOLD=0.65`).
2. Use a DISPOSABLE test project (never a production project).
3. Transcribe, then run moment detection. The log must show:
   `[Scoring Backend] autoclip`, `[AutoClip Scoring] Backend=autoclip
   provider=nvidia_nim model=... prompt_version=v2.0`, per-batch alignment
   lines with cache-hit/miss attribution, the eligibility-gate counts and
   the Top-N selection summary.
4. Verify in the UI: `AutoClip Score` badges on scored candidates,
   `Unscored (fallback)` on failures, selected counts = min(N, eligible).
5. Re-run with `AUTOSHORTS_SCORING_BACKEND=legacy` and confirm the
   original behavior is unchanged and no AutoClip cache/metadata leaks in.

A live run requires an authorized NVIDIA NIM account with access to the
configured model. Never make unexpected billable calls: confirm endpoint
availability and authorization before the run. If the model is
unavailable, report live scoring as BLOCKED rather than substituting a
model.
