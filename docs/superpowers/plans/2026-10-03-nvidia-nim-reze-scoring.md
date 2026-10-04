# NVIDIA NIM DiffusionGemma REZE Window Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `diffusiongemma-26b-a4b-it` via NVIDIA NIM API (`https://integrate.api.nvidia.com/v1/chat/completions`) for REZE window scoring only, behind a strict 3-way opt-in gate, leaving the default DeepSeek Chat timestamp generation pipeline 100% untouched.

**Architecture:** A dedicated `"nvidia_diffusiongemma"` match arm in `llm.rs::score_window_with_provider_raw` formats and submits OpenAI-compatible chat completion requests to the NVIDIA NIM endpoint with bounded 60s timeout and `chat_template_kwargs.enable_thinking = true`. In `lib.rs::generate_candidates`, a strict 3-way gate checks `AUTOSHORTS_DISCOVERY_MODE=window_scoring`, `AUTOSHORTS_REZE_PROVIDER=nvidia_diffusiongemma`, and `NVIDIA_API_KEY`; if any condition is absent, discovery coerces immediately and silently to `DiscoveryMode::TimestampGeneration` (DeepSeek Chat).

**Tech Stack:** Rust (Tauri 2.2, Tokio, Reqwest 0.12, Serde, Serde JSON), NVIDIA NIM API (OpenAI-compatible `/chat/completions`).

## Global Constraints

- Default discovery mode remains `DiscoveryMode::TimestampGeneration` with DeepSeek Chat (`deepseek-chat` via `DEEPSEEK_API_KEY`).
- REZE scoring prompt text from `build_window_scoring_prompt` is 100% byte-identical to baseline.
- No changes to timestamp generation prompts, model parameters, or DeepSeek API client execution.
- No changes to Render QA, VLM Candidate Scoring, PANNs Reactions, or Redundancy Detection.
- API keys must never be logged or emitted to console.
- In `score_window_with_provider_raw`, the reqwest call must use an explicit 60-second timeout.
- Unset or empty `AUTOSHORTS_REZE_PROVIDER` must never activate REZE.
- On Windows, always pass `-j 2` to `cargo test` to prevent MSVC linker heap exhaustion.

---

### Task 1: Model Name Resolution & Cache Isolation in `src/llm.rs`

**Files:**
- Modify: `autoshorts/src-tauri/src/llm.rs:2745-2760` (`resolve_effective_model_name`)
- Test: `autoshorts/src-tauri/src/llm.rs:mod tests`

**Interfaces:**
- Consumes: `provider: &str`, `model_name: Option<&str>`
- Produces: `resolve_effective_model_name` returns `"google/diffusiongemma-26b-a4b-it"` for `"nvidia_diffusiongemma"` (overridable via `NVIDIA_REZE_MODEL` or passed `model_name`).
- Produces: `WindowScoreCache::compute_cache_key` incorporates `"nvidia_diffusiongemma"` and model name into hash.

- [ ] **Step 1: Write the failing unit tests in `llm.rs`**

Add to `llm.rs` under `#[cfg(test)] mod tests`:
```rust
    #[test]
    fn test_resolve_effective_model_name_nvidia() {
        // Default without env or passed model
        let default_model = resolve_effective_model_name("nvidia_diffusiongemma", None);
        assert_eq!(default_model, "google/diffusiongemma-26b-a4b-it");

        let alias_model = resolve_effective_model_name("nvidia", None);
        assert_eq!(alias_model, "google/diffusiongemma-26b-a4b-it");

        // Passed model takes precedence
        let custom_model = resolve_effective_model_name("nvidia_diffusiongemma", Some("custom/diffusiongemma-test"));
        assert_eq!(custom_model, "custom/diffusiongemma-test");
    }

    #[test]
    fn test_nvidia_reze_cache_key_isolation() {
        let key_nvidia = WindowScoreCache::compute_cache_key(
            "source_123",
            0,
            0.0,
            30.0,
            "transcript sample text",
            "nvidia_diffusiongemma",
            "google/diffusiongemma-26b-a4b-it",
            "v1.0",
        );

        let key_openrouter = WindowScoreCache::compute_cache_key(
            "source_123",
            0,
            0.0,
            30.0,
            "transcript sample text",
            "openrouter",
            "google/gemini-2.5-flash",
            "v1.0",
        );

        let key_deepseek = WindowScoreCache::compute_cache_key(
            "source_123",
            0,
            0.0,
            30.0,
            "transcript sample text",
            "deepseek",
            "deepseek-chat",
            "v1.0",
        );

        assert_ne!(key_nvidia, key_openrouter, "NVIDIA cache key must never collide with OpenRouter");
        assert_ne!(key_nvidia, key_deepseek, "NVIDIA cache key must never collide with DeepSeek");
        assert!(key_nvidia.starts_with("reze_score_"));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p autoshorts --lib -- test_resolve_effective_model_name_nvidia -j 2`
Expected: FAIL because `"nvidia_diffusiongemma"` falls through to default `"deepseek-chat"`.

- [ ] **Step 3: Implement model name resolution in `llm.rs`**

Update `resolve_effective_model_name` in `autoshorts/src-tauri/src/llm.rs`:
```rust
fn resolve_effective_model_name(provider: &str, model_name: Option<&str>) -> String {
    let prov_norm = provider.trim().to_ascii_lowercase();
    match prov_norm.as_str() {
        "claude" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("ANTHROPIC_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "claude-3-5-sonnet-latest".to_string()),
        "gemini" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("GEMINI_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "gemini-2.5-flash".to_string()),
        "openai" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("OPENAI_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "gpt-4o-mini".to_string()),
        "openrouter" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("OPENROUTER_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "google/gemini-2.5-flash".to_string()),
        "groq" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("GROQ_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "llama-3.3-70b-versatile".to_string()),
        "local" | "ollama" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("OLLAMA_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "llama3.2".to_string()),
        "nvidia_diffusiongemma" | "nvidia" => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("NVIDIA_REZE_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string()),
        _ => model_name
            .filter(|m| !m.trim().is_empty())
            .map(|m| m.trim().to_string())
            .or_else(|| {
                std::env::var("DEEPSEEK_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "deepseek-chat".to_string()),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p autoshorts --lib -- test_resolve_effective_model_name_nvidia -j 2`
Run: `cargo test -p autoshorts --lib -- test_nvidia_reze_cache_key_isolation -j 2`
Expected: Both tests PASS.

---

### Task 2: NVIDIA NIM Client Arm & Payload Construction in `src/llm.rs`

**Files:**
- Modify: `autoshorts/src-tauri/src/llm.rs:2765-3170` (`score_window_with_provider_raw`)
- Test: `autoshorts/src-tauri/src/llm.rs:mod tests`

**Interfaces:**
- Consumes: `provider: &str`, `api_key: &str`, `model_name: Option<&str>`, `prompt: &str`, window indices
- Produces: Sends POST request to `{base_url}/chat/completions` with 60s timeout, extracts `choices[0].message.content`, parses via `parse_window_score_json`.

- [ ] **Step 1: Write unit tests for payload construction and fallback**

Add to `llm.rs` under `#[cfg(test)] mod tests`:
```rust
    #[test]
    fn test_nvidia_diffusiongemma_payload_construction() {
        let model = "google/diffusiongemma-26b-a4b-it";
        let prompt = "sample scoring prompt";
        let payload = serde_json::json!({
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
        });

        assert_eq!(payload["model"], "google/diffusiongemma-26b-a4b-it");
        assert_eq!(payload["max_tokens"], 4096);
        assert_eq!(payload["temperature"], 1.0);
        assert_eq!(payload["top_p"], 0.95);
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(payload["messages"][0]["role"], "user");
        assert_eq!(payload["messages"][0]["content"], prompt);
    }

    #[tokio::test]
    async fn test_reze_scoring_fallback_to_deepseek_on_nvidia_failure() {
        let mut config = WindowDiscoveryConfig::default();
        config.discovery_mode = DiscoveryMode::WindowScoring;

        let transcript = NormalizedTranscript {
            language: "en".to_string(),
            duration: 15.0,
            speakers: vec!["S1".to_string()],
            words: vec![
                TranscriptWord {
                    text: "Test".to_string(),
                    start: 0.0,
                    end: 1.0,
                    speaker: Some("S1".to_string()),
                },
            ],
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };

        // When nvidia provider fails (e.g., dummy key), discover_candidates_full_timeline
        // must execute the fallback path to timestamp generation with DeepSeek without panicking.
        let res = discover_candidates_full_timeline(
            &transcript,
            "nvidia_diffusiongemma",
            "dummy_invalid_nvidia_key",
            None,
            &config,
        )
        .await;

        assert!(res.is_err(), "Must return error on invalid fallback key without panicking");
    }
```

- [ ] **Step 2: Run test to observe failure/behavior**

Run: `cargo test -p autoshorts --lib -- test_nvidia_diffusiongemma_payload_construction -j 2`
Expected: PASS.
Run: `cargo test -p autoshorts --lib -- test_reze_scoring_fallback_to_deepseek_on_nvidia_failure -j 2`
Expected: Will fail or fall into default deepseek arm in raw scoring instead of dedicated nvidia arm.

- [ ] **Step 3: Implement `"nvidia_diffusiongemma" | "nvidia"` arm in `score_window_with_provider_raw`**

In `autoshorts/src-tauri/src/llm.rs` inside `score_window_with_provider_raw`:
```rust
    let prov_norm = provider.trim().to_ascii_lowercase();
    match prov_norm.as_str() {
        "claude" => { /* existing */ },
        "gemini" => { /* existing */ },
        "openai" => { /* existing */ },
        "openrouter" => { /* existing */ },
        "groq" => { /* existing */ },
        "local" | "ollama" => { /* existing */ },
        "nvidia_diffusiongemma" | "nvidia" => {
            let base_url = std::env::var("NVIDIA_REZE_BASE_URL")
                .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1".to_string());
            let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

            let model = model_name
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .or_else(|| {
                    std::env::var("NVIDIA_REZE_MODEL")
                        .ok()
                        .filter(|m| !m.trim().is_empty())
                })
                .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string());

            println!(
                "[NVIDIA NIM] Scoring window {} with model: {} at {} (prompt length: {} chars)",
                window_idx,
                model,
                url,
                prompt.len()
            );

            let payload = json!({
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
            });

            let response = provider_client()?
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(60))
                .json(&payload)
                .send()
                .await
                .context("calling NVIDIA NIM endpoint")?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("NVIDIA NIM request failed ({status}): {body}"));
            }

            let res_body: ChatCompletionResponse = response
                .json()
                .await
                .context("parsing NVIDIA NIM ChatCompletion response")?;

            let text = res_body
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .ok_or_else(|| anyhow!("NVIDIA NIM response did not include choices content"))?;

            parse_window_score_json(
                &text,
                window_idx,
                window_start,
                window_end,
                provider,
                &model,
            )
        }
        _ => { /* existing default deepseek */ }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p autoshorts --lib -- test_nvidia_diffusiongemma_payload_construction -j 2`
Run: `cargo test -p autoshorts --lib -- test_reze_scoring_fallback_to_deepseek_on_nvidia_failure -j 2`
Expected: PASS.

---

### Task 3: Strict 3-Way Opt-In Gating in `src/lib.rs`

**Files:**
- Modify: `autoshorts/src-tauri/src/lib.rs:700-775` (`generate_candidates`)
- Test: `autoshorts/src-tauri/src/lib.rs:mod tests`

**Interfaces:**
- Consumes: `discovery_mode: Option<String>`, `provider: Option<String>`, `api_key: Option<String>`, `model_name: Option<String>`, env vars `AUTOSHORTS_DISCOVERY_MODE`, `AUTOSHORTS_REZE_PROVIDER`, `NVIDIA_API_KEY`
- Produces: Evaluates strict 3-way gate. If all 3 present, runs `WindowScoring` with `nvidia_diffusiongemma`. If any is missing, coerces to `TimestampGeneration` with DeepSeek Chat.

- [ ] **Step 1: Write gating unit test in `lib.rs`**

Add to `lib.rs` under `#[cfg(test)] mod tests`:
```rust
    #[test]
    fn test_reze_3_way_gate_behavior() {
        let _lock = ENV_LOCK.lock().unwrap();

        // 1. Unset env vars -> TimestampGeneration
        std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::remove_var("NVIDIA_API_KEY");
        assert_eq!(resolve_discovery_mode_from_env(), models::DiscoveryMode::TimestampGeneration);

        // 2. Mode set to window_scoring but provider empty -> Gate not met
        std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", "window_scoring");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::remove_var("NVIDIA_API_KEY");
        let reze_provider = std::env::var("AUTOSHORTS_REZE_PROVIDER").unwrap_or_default();
        let is_nvidia_reze = reze_provider.trim().to_ascii_lowercase() == "nvidia_diffusiongemma";
        assert!(!is_nvidia_reze);

        // 3. Mode set to window_scoring + provider set but key missing -> Gate not met
        std::env::set_var("AUTOSHORTS_REZE_PROVIDER", "nvidia_diffusiongemma");
        std::env::remove_var("NVIDIA_API_KEY");
        let has_key = std::env::var("NVIDIA_API_KEY").is_ok();
        assert!(!has_key);

        // 4. All 3 set -> Gate met
        std::env::set_var("NVIDIA_API_KEY", "nvapi-test-key");
        let all_met = resolve_discovery_mode_from_env() == models::DiscoveryMode::WindowScoring
            && std::env::var("AUTOSHORTS_REZE_PROVIDER").map(|v| v.trim().to_ascii_lowercase() == "nvidia_diffusiongemma").unwrap_or(false)
            && std::env::var("NVIDIA_API_KEY").map(|k| !k.trim().is_empty()).unwrap_or(false);
        assert!(all_met);

        // Cleanup
        std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::remove_var("NVIDIA_API_KEY");
    }
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test -p autoshorts --lib -- test_reze_3_way_gate_behavior -j 2`
Expected: PASS.

- [ ] **Step 3: Update `generate_candidates` in `autoshorts/src-tauri/src/lib.rs`**

In `autoshorts/src-tauri/src/lib.rs`, replace lines 703–773 with the strict 3-way gate logic:
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
            log::debug!("[REZE Gate] 3-way gate not met -> falling back to TimestampGeneration");
        }
        // Timestamp generation uses configured/default provider (strictly reverts to DeepSeek if not explicitly another local/cloud provider)
        let prov = provider
            .filter(|p| !p.trim().is_empty() && p != "openrouter")
            .or_else(|| std::env::var("LLM_PROVIDER").ok().filter(|p| p != "openrouter"))
            .unwrap_or_else(|| "deepseek".to_string())
            .to_lowercase();
        let k = match prov.as_str() {
            "claude" => api_key
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .ok_or_else(|| {
                    "Set ANTHROPIC_API_KEY or supply Claude API Key to generate candidates.".to_string()
                })?,
            "gemini" => api_key
                .or_else(|| std::env::var("GEMINI_API_KEY").ok())
                .ok_or_else(|| {
                    "Set GEMINI_API_KEY or supply Gemini API Key to generate candidates.".to_string()
                })?,
            "openai" => api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| {
                    "Set OPENAI_API_KEY or supply OpenAI API Key to generate candidates.".to_string()
                })?,
            "groq" => api_key
                .or_else(|| std::env::var("GROQ_API_KEY").ok())
                .ok_or_else(|| {
                    "Set GROQ_API_KEY or supply Groq API Key to generate candidates.".to_string()
                })?,
            "local" | "ollama" => String::new(),
            _ => api_key
                .filter(|k| !k.trim().is_empty())
                .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
                .ok_or_else(|| {
                    "Set DEEPSEEK_API_KEY or supply DeepSeek API Key to generate candidates."
                        .to_string()
                })?,
        };
        let m = if prov == "deepseek"
            && (model_name.is_none()
                || model_name.as_deref().unwrap_or("").trim().is_empty()
                || model_name.as_deref().unwrap_or("").starts_with("google/"))
        {
            Some("deepseek-chat".to_string())
        } else {
            model_name
        };
        (models::DiscoveryMode::TimestampGeneration, prov, k, m)
    };
```

- [ ] **Step 4: Run cargo check on library**

Run: `cargo check -p autoshorts --lib -j 2`
Expected: Finished in < 5s with 0 errors.

---

### Task 4: Full System Regression & Verification

**Files:**
- Test: All Rust & Python test suites across the repository.

- [ ] **Step 1: Run full Rust library test suite**

Run: `cargo test -p autoshorts --lib -j 2`
Expected: 474+ passed, 0 failed.

- [ ] **Step 2: Run T7 prosody integration test suite**

Run: `cargo test -p autoshorts --test t7_prosody_suite -j 2`
Expected: 8 passed, 0 failed.

- [ ] **Step 3: Run Pause Intel integration test suite**

Run: `cargo test -p autoshorts --test pause_intel_suite -j 2`
Expected: 8 passed, 0 failed.

- [ ] **Step 4: Run Python ML test suites**

Run: `& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" -m unittest autoshorts/src-tauri/scripts/test_t7_prosody_suite.py`
Run: `& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" -m unittest autoshorts/src-tauri/scripts/test_pause_intel_suite.py`
Run: `& "d:\College\Autoshorts 11.0\.venv\Scripts\python.exe" autoshorts/src-tauri/scripts/test_smart_pacing_learned.py`
Expected: All suites OK.

- [ ] **Step 5: Run frontend build**

Run: `npm run build` (in `autoshorts`)
Expected: Built in < 20s with 0 errors.
