# AutoShorts 11.0: NVIDIA NIM DiffusionGemma REZE Window Scoring Design Spec

- **Author**: Antigravity (AutoShorts Integration Engineer)
- **Date**: 2026-10-03
- **Status**: APPROVED BY USER (Pending implementation plan)
- **Target Component**: Candidate Discovery / REZE Window Scoring Pipeline (`src/lib.rs`, `src/llm.rs`)

---

## 1. Architectural Overview & Invariants

### 1.1 Invariant: Zero Regression to Default Timestamp Generation
The baseline production discovery pipeline uses **TimestampGeneration with DeepSeek Chat** (`deepseek-chat` via `DEEPSEEK_API_KEY`).
This default behavior MUST remain 100% untouched:
- `WindowDiscoveryConfig::default().discovery_mode` remains `DiscoveryMode::TimestampGeneration`.
- No modifications to timestamp generation prompts, model parameters, or DeepSeek API client execution.
- No modifications to Render QA, VLM Candidate Scoring, PANNs Reactions, or Redundancy Detection.

### 1.2 Invariant: Strict 3-Way Opt-In Gating
REZE Window Scoring with `diffusiongemma-26b-a4b-it` is **strictly opt-in**. It only executes if **ALL THREE** of the following conditions are simultaneously met:
1. `AUTOSHORTS_DISCOVERY_MODE` resolves to `DiscoveryMode::WindowScoring` (via environment variable `AUTOSHORTS_DISCOVERY_MODE=window_scoring` / `reze` / `scoring`, or CLI/IPC `discovery_mode`).
2. `AUTOSHORTS_REZE_PROVIDER` is set to `"nvidia_diffusiongemma"` (case-insensitive, trimmed).
3. `NVIDIA_API_KEY` is present (either in the environment or passed in the command payload).

If **any** of the three conditions is absent:
- `discovery_mode` is coerced immediately to `DiscoveryMode::TimestampGeneration`.
- A debug-level log is emitted: `[REZE Gate] 3-way gate not met -> falling back to TimestampGeneration`.
- Discovery proceeds immediately through the standard DeepSeek Chat timestamp generator without invoking any REZE or NVIDIA endpoints.

---

## 2. Environment Configuration

| Variable | Required | Default Value | Description |
|---|---|---|---|
| `AUTOSHORTS_DISCOVERY_MODE` | Optional | `timestamp_generation` | Discovery mode switch (`window_scoring` / `reze` to opt in). |
| `AUTOSHORTS_REZE_PROVIDER` | Optional | `""` (empty string) | Set to `"nvidia_diffusiongemma"` to opt into NVIDIA NIM REZE scoring. |
| `NVIDIA_API_KEY` | Conditional | *None* | Bearer authentication token for the NVIDIA NIM API. |
| `NVIDIA_REZE_BASE_URL` | Optional | `https://integrate.api.nvidia.com/v1` | Base URL for NIM endpoint (allows local proxies or mock testing). |
| `NVIDIA_REZE_MODEL` | Optional | `google/diffusiongemma-26b-a4b-it` | Model identifier string sent in the NIM request payload. |

---

## 3. Implementation Details

### 3.1 Gateway Resolution in `src/lib.rs`
In `generate_candidates`:
1. Parse raw `discovery_mode` from IPC or environment (`resolve_discovery_mode_from_env()`).
2. Read `AUTOSHORTS_REZE_PROVIDER`.
3. Check for `NVIDIA_API_KEY`.
4. Apply the strict 3-way gate:
   ```rust
   let is_reze_raw = matches!(discovery_mode, models::DiscoveryMode::WindowScoring);
   let reze_provider_env = std::env::var("AUTOSHORTS_REZE_PROVIDER").unwrap_or_default();
   let is_nvidia_reze = reze_provider_env.trim().to_ascii_lowercase() == "nvidia_diffusiongemma";
   let nvidia_key = std::env::var("NVIDIA_API_KEY")
       .ok()
       .or_else(|| api_key.clone().filter(|k| !k.trim().is_empty()));

   let (discovery_mode, active_provider, key, effective_model) = if is_reze_raw && is_nvidia_reze && nvidia_key.is_some() {
       let base_model = std::env::var("NVIDIA_REZE_MODEL")
           .ok()
           .filter(|m| !m.trim().is_empty())
           .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string());
       (
           models::DiscoveryMode::WindowScoring,
           "nvidia_diffusiongemma".to_string(),
           nvidia_key.unwrap(),
           Some(base_model),
       )
   } else {
       if is_reze_raw {
           // Emitted at debug level only via log::debug! to keep production terminal clean
           log::debug!("[REZE Gate] 3-way gate not met -> falling back to TimestampGeneration");
       }
       // Baseline timestamp generation path (DeepSeek Chat untouched)
       let prov = provider
           .filter(|p| !p.trim().is_empty() && p != "openrouter")
           .or_else(|| std::env::var("LLM_PROVIDER").ok().filter(|p| p != "openrouter"))
           .unwrap_or_else(|| "deepseek".to_string())
           .to_lowercase();
       let k = match prov.as_str() {
           "claude" => api_key.or_else(|| std::env::var("ANTHROPIC_API_KEY").ok()).ok_or_else(...),
           "gemini" => api_key.or_else(|| std::env::var("GEMINI_API_KEY").ok()).ok_or_else(...),
           "openai" => api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok()).ok_or_else(...),
           "groq" => api_key.or_else(|| std::env::var("GROQ_API_KEY").ok()).ok_or_else(...),
           "local" | "ollama" => String::new(),
           _ => api_key.filter(|k| !k.trim().is_empty()).or_else(|| std::env::var("DEEPSEEK_API_KEY").ok()).ok_or_else(...),
       }?;
       let m = if prov == "deepseek" && (model_name.is_none() || model_name.as_deref().unwrap_or("").trim().is_empty() || model_name.as_deref().unwrap_or("").starts_with("google/")) {
           Some("deepseek-chat".to_string())
       } else {
           model_name
       };
       (
           models::DiscoveryMode::TimestampGeneration,
           prov,
           k,
           m,
       )
   };
   ```

### 3.2 NVIDIA NIM Client Execution in `src/llm.rs`
In `score_window_with_provider_raw`:
Add match arm `"nvidia_diffusiongemma" | "nvidia"`:
- **Endpoint URL**: `format!("{}/chat/completions", base_url.trim_end_matches('/'))` where `base_url` defaults to `https://integrate.api.nvidia.com/v1`.
- **Model Name**: Defaults to `"google/diffusiongemma-26b-a4b-it"` (overridable via `NVIDIA_REZE_MODEL` or passed `model_name`).
- **Telemetry**: Log `[NVIDIA NIM] Scoring window {window_idx} with model: {model} at {url} (prompt length: {len} chars)` without logging the API key.
- **Headers**:
  - `Authorization: Bearer <api_key>`
  - `Content-Type: application/json`
- **Request Timeout**: Bounded timeout (60s) via `std::time::Duration::from_secs(60)`.
- **Payload**:
  ```json
  {
    "model": model,
    "messages": [
      { "role": "user", "content": prompt }
    ],
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 0.95,
    "chat_template_kwargs": {
      "enable_thinking": true
    }
  }
  ```
- **Response Parsing**:
  - Validates `response.status().is_success()`.
  - Deserializes `ChatCompletionResponse` and extracts `choices[0].message.content`.
  - Passes content text to `parse_window_score_json(...)`.
  - Reuses existing bounded JSON extraction and thinking token sanitizer to safely parse score fields while ignoring any emitted timestamps.

### 3.3 Model Name & Cache Key Isolation
In `resolve_effective_model_name`:
```rust
"nvidia_diffusiongemma" | "nvidia" => model_name
    .filter(|m| !m.trim().is_empty())
    .map(|m| m.trim().to_string())
    .or_else(|| std::env::var("NVIDIA_REZE_MODEL").ok().filter(|m| !m.trim().is_empty()))
    .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string()),
```
In `WindowScoreCache::compute_cache_key`:
Key hash components:
`{source_identity}_{idx}_{w_start}_{w_end}_{text_hash}_{provider}_{effective_model}_{prompt_version}`
When `provider = "nvidia_diffusiongemma"` and `effective_model = "google/diffusiongemma-26b-a4b-it"`, on-disk files (`reze_scores/<key>.json`) and in-memory cache entries are guaranteed never to collide with OpenRouter (`google/gemini-2.5-flash`) or DeepSeek entries.

### 3.4 Safe Fallback on Failure
In `discover_candidates_full_timeline`:
When `DiscoveryMode::WindowScoring` is executed with `nvidia_diffusiongemma`:
If `discover_candidates_reze_scoring` fails (due to network failure, API error, 401, timeout, or 0 candidates produced):
- Emits existing verbatim log:
  `[REZE Scoring] Warning: Scoring discovery failed or produced 0 candidates. Falling back safely to timestamp generation with DeepSeek.`
- Automatically executes `discover_candidates_timestamp_generation` using `deepseek` / `DEEPSEEK_API_KEY`.
- No unhandled errors or panics.

---

## 4. Verification & Testing Matrix

### 4.1 Unit Tests in `src/llm.rs`
1. `test_resolve_effective_model_name_nvidia`:
   - Verifies `"nvidia_diffusiongemma"` resolves to `"google/diffusiongemma-26b-a4b-it"`.
   - Verifies `NVIDIA_REZE_MODEL` override works as expected.
2. `test_nvidia_reze_cache_key_isolation`:
   - Generates cache key for `nvidia_diffusiongemma` + `google/diffusiongemma-26b-a4b-it`.
   - Generates cache key for `openrouter` + `google/gemini-2.5-flash` with the same window and transcript.
   - Asserts keys are strictly non-equal and disk path incorporates distinct hash.
3. `test_nvidia_diffusiongemma_payload_construction`:
   - Validates that the request payload matches the exact schema: `model`, `max_tokens: 4096`, `temperature: 1.0`, `top_p: 0.95`, and `chat_template_kwargs.enable_thinking: true`.
4. `test_reze_scoring_fallback_to_deepseek_on_nvidia_failure`:
   - Confirms that a failed `WindowScoring` run with `nvidia_diffusiongemma` falls back safely to `discover_candidates_timestamp_generation`.

### 4.2 Integration & Gating Tests in `src/lib.rs`
5. `test_reze_3_way_gate_behavior`:
   - Case A: Unset env vars $\to$ `DiscoveryMode::TimestampGeneration`.
   - Case B: `AUTOSHORTS_DISCOVERY_MODE=window_scoring` but `AUTOSHORTS_REZE_PROVIDER` unset $\to$ `DiscoveryMode::TimestampGeneration`.
   - Case C: `AUTOSHORTS_DISCOVERY_MODE=window_scoring` + `AUTOSHORTS_REZE_PROVIDER=nvidia_diffusiongemma` but `NVIDIA_API_KEY` missing $\to$ `DiscoveryMode::TimestampGeneration`.
   - Case D: All 3 conditions satisfied $\to$ `DiscoveryMode::WindowScoring` with `active_provider = "nvidia_diffusiongemma"`.

### 4.3 Full System Regression
- `cargo test -p autoshorts --lib -j 2` (All 471+ tests pass).
- `cargo test -p autoshorts --test t7_prosody_suite -j 2` (All 8 tests pass).
- `cargo test -p autoshorts --test pause_intel_suite -j 2` (All 8 tests pass).
- Python ML test suites (`test_t7_prosody_suite.py`, `test_pause_intel_suite.py`, `test_smart_pacing_learned.py`).
- Frontend build `npm run build`.
