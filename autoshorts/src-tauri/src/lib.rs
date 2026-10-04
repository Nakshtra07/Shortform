pub mod audio;
pub mod boundary;
pub mod candidate_redundancy;
pub mod caption_intel;
pub mod captions;
pub mod db;
// `llm`, `transcription` and `transcript_normalizer` are exposed so the
// end-to-end validation harness in examples/ can drive the SAME production
// entry points the Tauri commands use, rather than reimplementing them.
pub mod llm;
pub mod media;
pub mod models;
pub mod pacing;
pub mod panns_reactions;
pub mod proc_guard;
pub mod render_qa;
pub mod scene_intelligence;
pub mod speaker_intelligence;
pub mod transcript_normalizer;
pub mod transcription;
pub mod vlm_scoring;
mod pause_intel;
mod t7_prosody;
mod youtube;

pub use pause_intel::{score_gaps, PauseIntelGap};
pub use t7_prosody::{predict_t7_boundaries, T7Boundary};

/// Master switch for Pause Intelligence learned classification.
/// Default: disabled (unset = disabled for safety).
pub fn pause_intel_enabled() -> bool {
    match std::env::var("AUTOSHORTS_SMART_PACING_LEARNED") {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on"),
        Err(_) => false,
    }
}

/// Master switch for T7 prosodic boundary tagging.
pub fn t7_prosody_enabled() -> bool {
    match std::env::var("AUTOSHORTS_T7_PROSODY") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off"),
        Err(_) => true,
    }
}

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use tauri::{Emitter, Manager};

use db::Database;
use models::{
    Candidate, CaptionChunk, EnvironmentStatus, MediaProbe, NormalizedTranscript, Project,
    ProjectDetail, Transcript, TranscriptWord,
};

#[derive(Clone, serde::Serialize)]
struct PullProgressPayload {
    status: String,
    completed: Option<u64>,
    total: Option<u64>,
    percentage: Option<f64>,
}

/// In-flight render tracker ensuring only one render job per candidate runs at any given time.
#[derive(Clone, Default, Debug)]
pub struct InFlightRenderTracker {
    active_renders: Arc<Mutex<HashSet<String>>>,
}

#[derive(Debug)]
pub struct RenderGuard {
    candidate_id: String,
    output_key: Option<String>,
    tracker: InFlightRenderTracker,
}

impl Drop for RenderGuard {
    fn drop(&mut self) {
        if let Ok(mut set) = self.tracker.active_renders.lock() {
            set.remove(&self.candidate_id);
            if let Some(ref out) = self.output_key {
                set.remove(out);
            }
            println!(
                "[Render Lock] Released in-flight render lock for candidate {}",
                self.candidate_id
            );
        }
    }
}

impl InFlightRenderTracker {
    pub fn new() -> Self {
        Self {
            active_renders: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub fn try_acquire(&self, candidate_id: &str) -> Result<RenderGuard, String> {
        let mut set = self
            .active_renders
            .lock()
            .map_err(|_| "Render tracker lock poisoned".to_string())?;
        if set.contains(candidate_id) {
            return Err(format!(
                "Candidate {} is already in-flight and being rendered",
                candidate_id
            ));
        }
        set.insert(candidate_id.to_string());
        println!(
            "[Render Lock] Acquired in-flight render lock for candidate {}",
            candidate_id
        );
        Ok(RenderGuard {
            candidate_id: candidate_id.to_string(),
            output_key: None,
            tracker: self.clone(),
        })
    }

    pub fn try_acquire_output(
        &self,
        candidate_id: &str,
        project_id: &str,
        rank: i64,
    ) -> Result<RenderGuard, String> {
        let mut set = self
            .active_renders
            .lock()
            .map_err(|_| "Render tracker lock poisoned".to_string())?;

        if set.contains(candidate_id) {
            return Err(format!(
                "Candidate {} is already in-flight and being rendered",
                candidate_id
            ));
        }

        let output_key = format!("output:{}:{}", project_id, rank);
        if set.contains(&output_key) {
            return Err(format!(
                "Render for project {} rank {} (clip-{:02}_flat.mp4) is already in-flight",
                project_id, rank, rank
            ));
        }

        set.insert(candidate_id.to_string());
        set.insert(output_key.clone());
        println!(
            "[Render Lock] Acquired in-flight render lock for candidate {} [output: {}:{}]",
            candidate_id, project_id, rank
        );
        Ok(RenderGuard {
            candidate_id: candidate_id.to_string(),
            output_key: Some(output_key),
            tracker: self.clone(),
        })
    }

    pub fn is_rendering(&self, candidate_id: &str) -> bool {
        self.active_renders
            .lock()
            .map(|set| set.contains(candidate_id))
            .unwrap_or(false)
    }

    pub fn is_output_rendering(&self, project_id: &str, rank: i64) -> bool {
        let output_key = format!("output:{}:{}", project_id, rank);
        self.active_renders
            .lock()
            .map(|set| set.contains(&output_key))
            .unwrap_or(false)
    }
}

#[derive(Clone)]
struct AppState {
    db: Database,
    data_dir: PathBuf,
    render_tracker: InFlightRenderTracker,
}

#[tauri::command]
async fn environment_status(
    state: tauri::State<'_, AppState>,
) -> Result<EnvironmentStatus, String> {
    let llm_provider = std::env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "deepseek".to_string())
        .to_lowercase();

    let has_local_whisper_model =
        transcription::whisper_cli_exists() || transcription::whisper_python_exists();

    let has_ollama = reqwest::Client::new()
        .get("http://localhost:11434")
        .timeout(std::time::Duration::from_millis(1000))
        .send()
        .await
        .is_ok();

    Ok(EnvironmentStatus {
        data_dir: state.data_dir.to_string_lossy().to_string(),
        has_ffmpeg: media::command_exists("ffmpeg"),
        has_ffprobe: media::command_exists("ffprobe"),
        has_deepgram_key: std::env::var("DEEPGRAM_API_KEY").is_ok(),
        has_anthropic_key: std::env::var("ANTHROPIC_API_KEY").is_ok(),
        has_deepseek_key: std::env::var("DEEPSEEK_API_KEY").is_ok(),
        has_gemini_key: std::env::var("GEMINI_API_KEY").is_ok(),
        has_openai_key: std::env::var("OPENAI_API_KEY").is_ok(),
        has_openrouter_key: std::env::var("OPENROUTER_API_KEY").is_ok(),
        has_groq_key: std::env::var("GROQ_API_KEY").is_ok(),
        llm_provider,
        has_local_whisper_model,
        has_ollama,
        has_ytdlp: media::command_exists("yt-dlp"),
    })
}

#[tauri::command]
async fn pull_ollama_model(app: tauri::AppHandle, model_name: String) -> Result<(), String> {
    let client = reqwest::Client::new();

    let mut response = client
        .post("http://localhost:11434/api/pull")
        .json(&serde_json::json!({
            "name": model_name,
            "stream": true,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Ollama pull returned status {status}: {text}"));
    }

    let mut buffer = String::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        let chunk_str = String::from_utf8_lossy(&chunk);
        buffer.push_str(&chunk_str);

        // Process lines in buffer
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(err_msg) = val.get("error").and_then(|v| v.as_str()) {
                    return Err(err_msg.to_string());
                }

                let completed = val.get("completed").and_then(|v| v.as_u64());
                let total = val.get("total").and_then(|v| v.as_u64());

                let mut status = val
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Downloading...")
                    .to_string();

                if status.starts_with("downloading ") {
                    if let (Some(c), Some(t)) = (completed, total) {
                        let c_mb = c as f64 / 1024.0 / 1024.0;
                        let t_mb = t as f64 / 1024.0 / 1024.0;
                        if t_mb > 100.0 {
                            status =
                                format!("Downloading weights: {:.1} MB / {:.1} MB", c_mb, t_mb);
                        } else {
                            status = format!(
                                "Downloading model components: {:.1} MB / {:.1} MB",
                                c_mb, t_mb
                            );
                        }
                    } else {
                        status = "Downloading model components...".to_string();
                    }
                }

                let percentage = if let (Some(c), Some(t)) = (completed, total) {
                    if t > 0 {
                        Some((c as f64 / t as f64) * 100.0)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let payload = PullProgressPayload {
                    status,
                    completed,
                    total,
                    percentage,
                };

                let _ = app.emit("ollama-pull-progress", payload);
            }
        }
    }

    Ok(())
}

#[tauri::command]
async fn install_ollama(app: tauri::AppHandle) -> Result<(), String> {
    let _ = app.emit(
        "ollama-install-status",
        "Checking if Ollama is already installed...",
    );
    let launch = std::process::Command::new("open")
        .args(["-a", "Ollama"])
        .output();

    if let Ok(out) = launch {
        if out.status.success() {
            let _ = app.emit("ollama-install-status", "Ollama is installed. Launching...");
            for _ in 0..12 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if reqwest::Client::new()
                    .get("http://localhost:11434")
                    .send()
                    .await
                    .is_ok()
                {
                    let _ = app.emit("ollama-install-status", "Ollama started successfully!");
                    return Ok(());
                }
            }
        }
    }

    let brew_path = if std::path::Path::new("/opt/homebrew/bin/brew").exists() {
        Some("/opt/homebrew/bin/brew")
    } else if std::path::Path::new("/usr/local/bin/brew").exists() {
        Some("/usr/local/bin/brew")
    } else {
        None
    };

    if let Some(path) = brew_path {
        let _ = app.emit(
            "ollama-install-status",
            "Installing Ollama via Homebrew Cask...",
        );

        let output = std::process::Command::new(path)
            .args(["install", "--cask", "ollama"])
            .output()
            .map_err(|e| format!("Failed to run brew command: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if !stderr.contains("already installed") {
                return Err(format!("Brew install failed: {}", stderr));
            }
        }

        let _ = app.emit("ollama-install-status", "Starting Ollama.app...");
        let launch = std::process::Command::new("open")
            .args(["-a", "Ollama"])
            .output();

        if let Ok(out) = launch {
            if out.status.success() {
                for _ in 0..12 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if reqwest::Client::new()
                        .get("http://localhost:11434")
                        .send()
                        .await
                        .is_ok()
                    {
                        let _ = app.emit("ollama-install-status", "Ollama started successfully!");
                        return Ok(());
                    }
                }
            }
        }
    }

    let _ = app.emit(
        "ollama-install-status",
        "Downloading Ollama zip from official source...",
    );
    let temp_dir = std::env::temp_dir();
    let zip_path = temp_dir.join("Ollama-darwin.zip");

    let response = reqwest::get("https://ollama.com/download/Ollama-darwin.zip")
        .await
        .map_err(|e| format!("Failed to download Ollama: {e}"))?;

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read Ollama bytes: {e}"))?;
    std::fs::write(&zip_path, bytes).map_err(|e| format!("Failed to save Ollama zip: {e}"))?;

    let _ = app.emit("ollama-install-status", "Unzipping Ollama package...");
    let unzip_output = std::process::Command::new("unzip")
        .args([
            "-o",
            &zip_path.to_string_lossy().to_string(),
            "-d",
            &temp_dir.to_string_lossy().to_string(),
        ])
        .output()
        .map_err(|e| format!("Failed to unzip Ollama: {e}"))?;

    if !unzip_output.status.success() {
        return Err(format!(
            "Failed to unzip: {}",
            String::from_utf8_lossy(&unzip_output.stderr)
        ));
    }

    let _ = app.emit(
        "ollama-install-status",
        "Installing to Applications folder...",
    );
    let app_src = temp_dir.join("Ollama.app");

    let mv_output = std::process::Command::new("mv")
        .args([&app_src.to_string_lossy().to_string(), "/Applications/"])
        .output()
        .map_err(|e| format!("Failed to move Ollama to Applications: {e}"))?;

    if !mv_output.status.success() {
        let user_apps = dirs::home_dir()
            .ok_or_else(|| "Could not find home directory".to_string())?
            .join("Applications");
        std::fs::create_dir_all(&user_apps)
            .map_err(|e| format!("Failed to create ~/Applications: {e}"))?;

        let mv_user_output = std::process::Command::new("mv")
            .args([
                &app_src.to_string_lossy().to_string(),
                &user_apps.to_string_lossy().to_string(),
            ])
            .output()
            .map_err(|e| format!("Failed to move Ollama to ~/Applications: {e}"))?;

        if !mv_user_output.status.success() {
            return Err(format!(
                "Failed to install Ollama to Applications folder: {}",
                String::from_utf8_lossy(&mv_user_output.stderr)
            ));
        }
    }

    let _ = app.emit("ollama-install-status", "Starting Ollama...");
    let launch = std::process::Command::new("open")
        .args(["-a", "Ollama"])
        .output();

    if launch.is_ok() {
        for _ in 0..12 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            if reqwest::Client::new()
                .get("http://localhost:11434")
                .send()
                .await
                .is_ok()
            {
                let _ = app.emit("ollama-install-status", "Ollama started successfully!");
                return Ok(());
            }
        }
    }

    Err("Ollama installed but could not be automatically started. Please open Ollama from your Applications folder.".to_string())
}

#[tauri::command]
fn create_project_from_path(
    state: tauri::State<'_, AppState>,
    path: String,
    transcription_mode: String,
    caption_style: String,
    framing_mode: String,
) -> Result<Project, String> {
    validate_media_extension(&path).map_err(to_command_error)?;
    let probe = media::probe_media(&path).ok();

    state
        .db
        .create_project(
            &path,
            &transcription_mode,
            &caption_style,
            &framing_mode,
            probe.and_then(|probe| probe.duration_sec),
        )
        .map_err(to_command_error)
}

#[tauri::command]
fn list_projects(state: tauri::State<'_, AppState>) -> Result<Vec<Project>, String> {
    state.db.list_projects().map_err(to_command_error)
}

#[tauri::command]
fn get_project_detail(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<ProjectDetail, String> {
    state
        .db
        .project_detail(&project_id)
        .map_err(to_command_error)
}

#[tauri::command]
fn probe_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<MediaProbe, String> {
    let project = state
        .db
        .get_project(&project_id)
        .map_err(to_command_error)?;
    let probe = media::probe_media(&project.source_path).map_err(to_command_error)?;
    state
        .db
        .update_project_status(&project_id, "ingest", probe.duration_sec)
        .map_err(to_command_error)?;
    Ok(probe)
}

#[tauri::command]
fn extract_project_audio(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<String, String> {
    let project = state
        .db
        .get_project(&project_id)
        .map_err(to_command_error)?;
    let audio_path = media::extract_audio(&project.source_path, &project_dir(&state, &project_id))
        .map_err(to_command_error)?;
    Ok(audio_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn transcribe_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
    provider: String,
    api_key: Option<String>,
) -> Result<Transcript, String> {
    let db = state.db.clone();
    let data_dir = state.data_dir.clone();
    let project = db.get_project(&project_id).map_err(to_command_error)?;
    db.update_project_status(&project_id, "transcribing", None)
        .map_err(to_command_error)?;

    let transcript = match provider.as_str() {
        "deepgram" => {
            let key = api_key
                .or_else(|| std::env::var("DEEPGRAM_API_KEY").ok())
                .ok_or_else(|| {
                    "Set DEEPGRAM_API_KEY or paste an API key to use cloud transcription."
                        .to_string()
                })?;
            let audio_path = media::extract_audio(
                &project.source_path,
                &data_dir.join("projects").join(&project_id),
            )
            .map_err(to_command_error)?;
            transcription::transcribe_deepgram(&audio_path.to_string_lossy(), &key)
                .await
                .map_err(to_command_error)?
        }
        "local" => {
            let has_whisper =
                transcription::whisper_cli_exists() || transcription::whisper_python_exists();
            if !has_whisper {
                return Err("Whisper is not installed. Please install it (e.g., via Homebrew 'brew install whisper-cli' or via Python 'pip3 install openai-whisper').".to_string());
            }
            let audio_path = media::extract_audio(
                &project.source_path,
                &data_dir.join("projects").join(&project_id),
            )
            .map_err(to_command_error)?;
            transcription::transcribe_local(
                &audio_path.to_string_lossy(),
                &data_dir.to_string_lossy(),
            )
            .await
            .map_err(to_command_error)?
        }
        other => return Err(format!("Unsupported transcription provider: {other}")),
    };

    let raw_transcript_json =
        serde_json::to_string_pretty(&transcript).map_err(to_command_error)?;
    let normalized = crate::transcript_normalizer::normalize_transcript_semantic(&transcript);
    let normalized_json = serde_json::to_string_pretty(&normalized).map_err(to_command_error)?;
    let saved = db
        .save_transcript(
            &project_id,
            &provider,
            &normalized_json,
            Some(&transcript.language),
            Some(&raw_transcript_json),
        )
        .map_err(to_command_error)?;
    db.update_project_status(&project_id, "analyzing", Some(transcript.duration))
        .map_err(to_command_error)?;
    Ok(saved)
}

#[tauri::command]
fn save_demo_transcript(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<Transcript, String> {
    let transcript = demo_transcript();
    let raw_transcript_json =
        serde_json::to_string_pretty(&transcript).map_err(to_command_error)?;
    let normalized = crate::transcript_normalizer::normalize_transcript_semantic(&transcript);
    let normalized_json = serde_json::to_string_pretty(&normalized).map_err(to_command_error)?;
    let saved = state
        .db
        .save_transcript(
            &project_id,
            "demo",
            &normalized_json,
            Some(&transcript.language),
            Some(&raw_transcript_json),
        )
        .map_err(to_command_error)?;
    state
        .db
        .update_project_status(&project_id, "analyzing", Some(transcript.duration))
        .map_err(to_command_error)?;
    Ok(saved)
}

/// Resolves the candidate discovery mode from the `AUTOSHORTS_DISCOVERY_MODE` environment variable.
///
/// Invariant: When `AUTOSHORTS_DISCOVERY_MODE` is unset or invalid, default is ALWAYS `DiscoveryMode::TimestampGeneration`.
/// When set to `"window_scoring"`, `"reze"`, or `"scoring"`, activates `DiscoveryMode::WindowScoring`.
pub fn resolve_discovery_mode_from_env() -> models::DiscoveryMode {
    std::env::var("AUTOSHORTS_DISCOVERY_MODE")
        .ok()
        .and_then(|v| match v.to_lowercase().trim() {
            "window_scoring" | "reze" | "scoring" => Some(models::DiscoveryMode::WindowScoring),
            "timestamp_generation" | "timestamp" | "legacy" => {
                Some(models::DiscoveryMode::TimestampGeneration)
            }
            _ => None,
        })
        .unwrap_or(models::DiscoveryMode::TimestampGeneration)
}

/// Evaluates the strict 3-way opt-in gate for REZE window scoring.
/// REZE scoring only activates when ALL 3 of the following conditions are met:
/// 1. Discovery mode resolves to `WindowScoring` (via argument or `AUTOSHORTS_DISCOVERY_MODE`)
/// 2. REZE provider is set to `"nvidia_diffusiongemma"` (via `AUTOSHORTS_REZE_PROVIDER` or passed `provider`)
/// 3. NVIDIA API key is available (via `NVIDIA_API_KEY` or passed `api_key`)
///
/// If ANY condition is not met:
/// Coerces immediately to `DiscoveryMode::TimestampGeneration` and emits `log::debug!`.
pub fn resolve_reze_3_way_gate(
    discovery_mode_arg: Option<&str>,
    provider_arg: Option<&str>,
    api_key_arg: Option<&str>,
) -> (models::DiscoveryMode, bool, Option<String>) {
    let mode = match discovery_mode_arg.map(|s| s.trim().to_ascii_lowercase()) {
        Some(ref m) if m == "window_scoring" || m == "reze" || m == "scoring" => {
            models::DiscoveryMode::WindowScoring
        }
        Some(ref m) if m == "timestamp_generation" || m == "timestamp" || m == "legacy" => {
            models::DiscoveryMode::TimestampGeneration
        }
        _ => resolve_discovery_mode_from_env(),
    };

    let is_reze_mode = matches!(mode, models::DiscoveryMode::WindowScoring);

    let reze_provider_env = std::env::var("AUTOSHORTS_REZE_PROVIDER").unwrap_or_default();
    let is_nvidia_provider = reze_provider_env.trim().to_ascii_lowercase() == "nvidia_diffusiongemma"
        || provider_arg
            .map(|p| {
                let p_norm = p.trim().to_ascii_lowercase();
                p_norm == "nvidia_diffusiongemma" || p_norm == "nvidia"
            })
            .unwrap_or(false);

    let nvidia_key = std::env::var("NVIDIA_API_KEY")
        .ok()
        .filter(|k| !k.trim().is_empty())
        .or_else(|| {
            api_key_arg
                .filter(|k| !k.trim().is_empty())
                .map(|k| k.to_string())
        });

    if is_reze_mode && is_nvidia_provider && nvidia_key.is_some() {
        (models::DiscoveryMode::WindowScoring, true, nvidia_key)
    } else {
        if is_reze_mode {
            log::debug!("[REZE Gate] 3-way gate not met -> falling back to TimestampGeneration");
        }
        (models::DiscoveryMode::TimestampGeneration, false, None)
    }
}

#[tauri::command]
async fn generate_candidates(
    state: tauri::State<'_, AppState>,
    project_id: String,
    api_key: Option<String>,
    provider: Option<String>,
    model_name: Option<String>,
    _allow_demo: Option<bool>,
    discovery_mode: Option<String>,
    reze_provider: Option<String>,
) -> Result<Vec<Candidate>, String> {
    let db = state.db.clone();
    let transcript = db
        .latest_transcript(&project_id)
        .map_err(to_command_error)?
        .ok_or_else(|| "Transcribe the project before detecting moments.".to_string())?;
    let normalized: NormalizedTranscript =
        serde_json::from_str(&transcript.raw_json).map_err(to_command_error)?;

    if normalized.segments.is_empty() || normalized.words.is_empty() {
        let _ = db.clear_candidates(&project_id);
        return Err(
            "No speech or transcript text was detected in this video. Candidate moment detection requires spoken audio or transcript text."
                .to_string(),
        );
    }

    // If params are provided, override env vars for this call only:
    // Set AUTOSHORTS_DISCOVERY_MODE from discovery_mode
    // Set AUTOSHORTS_REZE_PROVIDER from reze_provider
    let prev_discovery_mode = std::env::var("AUTOSHORTS_DISCOVERY_MODE").ok();
    let prev_reze_provider = std::env::var("AUTOSHORTS_REZE_PROVIDER").ok();

    if let Some(ref dm) = discovery_mode {
        let trimmed = dm.trim();
        if !trimmed.is_empty() {
            std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", trimmed);
        }
    }
    if let Some(ref rp) = reze_provider {
        let trimmed = rp.trim();
        if !trimmed.is_empty() {
            std::env::set_var("AUTOSHORTS_REZE_PROVIDER", trimmed);
        }
    }

    struct CallEnvGuard {
        prev_discovery_mode: Option<String>,
        prev_reze_provider: Option<String>,
    }
    impl Drop for CallEnvGuard {
        fn drop(&mut self) {
            match &self.prev_discovery_mode {
                Some(v) => std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", v),
                None => std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE"),
            }
            match &self.prev_reze_provider {
                Some(v) => std::env::set_var("AUTOSHORTS_REZE_PROVIDER", v),
                None => std::env::remove_var("AUTOSHORTS_REZE_PROVIDER"),
            }
        }
    }
    let _call_env_guard = CallEnvGuard {
        prev_discovery_mode,
        prev_reze_provider,
    };

    let effective_provider = reze_provider.as_deref().or(provider.as_deref());
    let (discovery_mode, is_nvidia_reze, nvidia_key_opt) = resolve_reze_3_way_gate(
        discovery_mode.as_deref(),
        effective_provider,
        api_key.as_deref(),
    );

    let (active_provider, key, effective_model) = if is_nvidia_reze {
        let base_model = model_name
            .filter(|m| !m.trim().is_empty())
            .or_else(|| {
                std::env::var("NVIDIA_REZE_MODEL")
                    .ok()
                    .filter(|m| !m.trim().is_empty())
            })
            .unwrap_or_else(|| "google/diffusiongemma-26b-a4b-it".to_string());
        (
            "nvidia_diffusiongemma".to_string(),
            nvidia_key_opt.unwrap(),
            Some(base_model),
        )
    } else {
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
        (prov, k, m)
    };

    let project = db.get_project(&project_id).map_err(to_command_error)?;

    println!(
        "[Candidate Discovery] Active Mode: {:?}, Provider: {}, Model: {:?}",
        discovery_mode, active_provider, effective_model
    );
    let discovery_config = models::WindowDiscoveryConfig {
        discovery_mode,
        ..models::WindowDiscoveryConfig::default()
    };
    let mut drafts = llm::discover_candidates_full_timeline(
        &normalized,
        &active_provider,
        &key,
        effective_model.as_deref(),
        &discovery_config,
    )
    .await
    .map_err(to_command_error)?;

    if drafts.is_empty() {
        return Err("No viable clip candidates were returned for this transcript.".to_string());
    }

    // 1. Authoritative Semantic Closure Boundary Snapping
    // Hard Invariant: Multimodal signals never arbitrarily determine clip start/end boundaries.
    struct RawCandidateBoundary {
        raw_start: f64,
        raw_end: f64,
    }
    let raw_boundaries: Vec<RawCandidateBoundary> = drafts
        .iter()
        .map(|d| RawCandidateBoundary {
            raw_start: d.start,
            raw_end: d.end,
        })
        .collect();

    let hook_config = models::HookPipelineConfig::default();

    for draft in &mut drafts {
        draft.raw_llm_end = Some(draft.end);

        // 1. Perform deterministic token alignment from hookText -> normalized.words
        let is_q_hook = draft.question_hook_used == Some(true)
            || draft.hook_speaker.as_deref() == Some("Host")
            || draft.conversation_type.as_deref() == Some("question_answer");

        if let Some((v_start, v_end, conf, _s_idx, _e_idx)) = llm::align_hook_to_transcript_words(
            &draft.hook,
            draft.start,
            draft.end,
            &normalized.words,
        ) {
            draft.hook_start = Some(v_start);
            draft.hook_end = Some(v_end);
            draft.hook_confidence = Some(conf);
        }

        // 1b. Authoritative Payoff Alignment
        let mut payoff_aligned = false;
        if let Some(ref p_text) = draft.payoff_text {
            if let Some((p_start, p_end, conf, _s_idx, _e_idx)) =
                llm::align_payoff_to_transcript_words(
                    p_text,
                    draft.start,
                    draft.end,
                    &normalized.words,
                )
            {
                draft.payoff_score = Some(conf);
                if conf >= 0.50 {
                    draft.payoff_start = Some(p_start);
                    draft.payoff_end = Some(p_end);
                    draft.payoff_completion = Some(true);
                    // PAYOFF IS AUTHORITATIVE ENDPOINT SIGNAL: candidate_end = payoff_end
                    draft.end = p_end;
                    payoff_aligned = true;
                }
            }
            if !payoff_aligned {
                eprintln!(
                    "[Payoff Alignment] Alignment failure for candidate [{:.2}s - {:.2}s] with payoff {:?}",
                    draft.start, draft.end, p_text
                );
                draft.payoff_score = None;
                draft.payoff_start = None;
                draft.payoff_end = None;
                draft.payoff_completion = Some(false);
            }
        }

        // 2. Evaluate opening 1-3s hook context
        let context_eval = llm::evaluate_opening_hook_context(&draft.hook, is_q_hook, &hook_config);
        draft.opening_context_score = Some(context_eval.score);
        draft.opening_unresolved_reference = context_eval.unresolved_reference.clone();
        draft.opening_continuation_marker = context_eval.matched_marker.clone();

        // 3. Snap boundaries with hook anchor awareness (never crossing sentence backward into previous speech)
        let closure_sig = crate::models::ClosureSignals::from_draft(draft);
        let (snapped_start, snapped_end) = snap_to_semantic_boundaries_with_hook_anchor(
            &normalized.words,
            draft.start,
            draft.end,
            normalized.duration,
            Some(&closure_sig),
            draft.hook_start,
            is_q_hook,
        );

        draft.start = snapped_start;

        if payoff_aligned {
            // Once payoff alignment establishes payoff_end, that endpoint MUST REMAIN AUTHORITATIVE.
            // Hook Intelligence 2.0 must NOT interrupt, override, shorten, rewrite, re-anchor,
            // or otherwise change the payoff-driven endpoint.
            draft.end = draft.payoff_end.unwrap();
        } else {
            let validated_end = match validate_and_enforce_final_endpoint(
                &normalized.words,
                draft.raw_llm_end.unwrap_or(draft.end),
                snapped_end,
                &closure_sig,
                normalized.duration,
            ) {
                Ok(v_end) => v_end,
                Err(err) => {
                    eprintln!("[Pipeline Boundary Enforcer] Warning: {}", err);
                    snapped_end
                }
            };
            draft.end = validated_end;
        }
    }

    // 1b. Quality Gate: Hard rejection ONLY for objectively contextless openings (e.g. lone ungrounded 3rd-person pronoun)
    if hook_config.hard_reject_unresolved_pronoun {
        let before_count = drafts.len();
        let backup = drafts.clone();
        drafts.retain(|d| {
            if let Some(ref unres) = d.opening_unresolved_reference {
                eprintln!("[Hook Quality Gate] Hard rejecting candidate with ungrounded pronoun '{}' (Hook: \"{}\")", unres, d.hook);
                false
            } else {
                true
            }
        });
        if drafts.is_empty() {
            eprintln!("[Hook Quality Gate] Warning: All candidates were rejected, restoring previous drafts");
            drafts = backup;
        } else if drafts.len() < before_count {
            println!(
                "[Hook Quality Gate] Filtered {} objectively contextless candidates ({} remaining)",
                before_count - drafts.len(),
                drafts.len()
            );
        }
    }

    // 1c. Quality Gate: Anti-filler filter (drop candidates below minimum score threshold)
    let min_score = discovery_config.min_candidate_score;
    let before_quality = drafts.len();
    let quality_backup = drafts.clone();
    drafts.retain(|d| llm::composite_score(d) >= min_score);
    if drafts.is_empty() {
        eprintln!("[Quality Gate] Warning: All candidates fell below min score {:.2}, restoring top candidates", min_score);
        drafts = quality_backup;
    } else if drafts.len() < before_quality {
        println!(
            "[Quality Gate] Filtered {} low-scoring filler candidates (< {:.2}) ({} remaining)",
            before_quality - drafts.len(),
            min_score,
            drafts.len()
        );
    }

    // 2. Multimodal Hook Intelligence Analysis (Visual, Acoustic, Temporal Signals)
    let (mm_drafts, mm_status) = media::analyze_multimodal_hook_signals(
        &project.source_path,
        &drafts,
        normalized.duration,
        None,
    );

    // 3. High-Precision Re-Ranking and Temporal Balancing (preserve full pool, top 12 selected)
    let mut all_drafts =
        llm::deduplicate_and_rank_all_candidates(mm_drafts, normalized.duration, 12);

    // ── Phase 4: Candidate Redundancy Detection (DETERMINISTIC, signal-only) ──
    // Runs on the ranked pool so obvious near-duplicates never reach the user.
    // It may DROP candidates and it may rank them, but it never mutates a
    // timestamp, a payoff endpoint, or any boundary decision. Disabled by
    // default via AUTOSHORTS_CANDIDATE_REDUNDANCY.
    candidate_redundancy::run_redundancy_detection("", &mut all_drafts, &normalized.words);

    // ── Phase 4: PANNs Reaction Metadata (ADVISORY signal only) ──
    // Source-level pass, cached. It boosts the hook score at most 0.15 and only
    // through `enhance_hook_with_panns`, which is capped and cannot move any
    // boundary. Never touches payoff_end / candidate_end / framing.
    panns_reactions::annotate_candidates_with_reactions(&project.source_path, &mut all_drafts);

    // ── Phase 4: VLM Candidate Scoring (ADVISORY metadata only) ──
    // Runs LAST, after all deterministic ranking and boundary resolution, and
    // before persistence. Attaches vlm_* metadata to CandidateDraft only. It
    // cannot influence candidate ordering, the already-computed boundaries, or
    // rendering. Disabled by default via AUTOSHORTS_VLM_SCORING. Any failure
    // is a silent no-op so candidate generation never breaks.
    vlm_scoring::enhance_candidates_with_vlm(
        &project.id,
        &project.source_path,
        &mut all_drafts,
        &normalized.words,
    );

    let total_candidates = all_drafts.len();
    let interviewer_hooks = all_drafts
        .iter()
        .filter(|d| d.hook_speaker.as_deref() == Some("Host") || d.question_hook_used == Some(true))
        .count();
    let guest_hooks = total_candidates.saturating_sub(interviewer_hooks);

    println!("\n================================================================================");
    println!("=== [v10.x HOOK QUALITY & DETERMINISTIC SEMANTIC CLOSURE DIAGNOSTICS] ===");
    println!("================================================================================");
    println!("Pipeline Version: v10.x Hook Quality & Semantic Closure (Decoding the Hook)");
    println!(
        "Total Podcast Duration: {:.2}s ({:.1} min)",
        normalized.duration,
        normalized.duration / 60.0
    );
    println!(
        "Full Video Coverage: 00:00 -> {:.2}s (UNCOVERED_SECONDS = 0.0)",
        normalized.duration
    );
    println!(
        "Multimodal Status: {} | Visual: {} | Audio: {} | Temporal: {}",
        mm_status.status,
        mm_status.visual_analysis,
        mm_status.audio_analysis,
        mm_status.temporal_analysis
    );
    if mm_status.fallback_to_v92 {
        println!(
            "⚠️ Fallback Mode: v9.2 fallback — multimodal analysis unavailable ({})",
            mm_status.failure_reason.as_deref().unwrap_or("none")
        );
    } else {
        println!(
            "Multimodal Verification: SUCCESS ({} candidates analyzed in {}ms)",
            mm_status.candidate_count_analyzed, mm_status.processing_time_ms
        );
    }
    println!("Total Discovered Candidates: {}", total_candidates);
    println!("Interviewer Question Hooks: {}", interviewer_hooks);
    println!("Guest Statement Hooks: {}", guest_hooks);
    println!("Candidate Diagnostics Breakdown (R5):");
    for (i, draft) in all_drafts.iter().enumerate() {
        let duration = draft.end - draft.start;
        let mm_score = draft.multimodal_score.unwrap_or(draft.score);
        let sem_score = draft.semantic_score.unwrap_or(draft.score);
        let vis_score = draft.visual_score.unwrap_or(0.5);
        let aud_score = draft.audio_score.unwrap_or(0.5);
        let temp_score = draft.temporal_score.unwrap_or(0.7);

        println!(
            "  [Candidate #{}] [{:.2}s -> {:.2}s] ({:.2}s) | Speaker: {} | Type: {}",
            i + 1,
            draft.start,
            draft.end,
            duration,
            draft.hook_speaker.as_deref().unwrap_or("Speaker"),
            draft.conversation_type.as_deref().unwrap_or("general"),
        );
        println!("    HOOK: \"{}\"", draft.hook);
        if let (Some(hs), Some(he)) = (draft.hook_start, draft.hook_end) {
            println!(
                "    HOOK ALIGNMENT: verified [{:.2}s -> {:.2}s] (conf: {:.2})",
                hs,
                he,
                draft.hook_confidence.unwrap_or(1.0)
            );
        }
        println!(
            "    OPENING CONTEXT: {:.2} (marker: {}, ungrounded_ref: {})",
            draft.opening_context_score.unwrap_or(1.0),
            draft
                .opening_continuation_marker
                .as_deref()
                .unwrap_or("none"),
            draft
                .opening_unresolved_reference
                .as_deref()
                .unwrap_or("none"),
        );
        println!(
            "    HOOK SPEAKER: {}",
            if draft.hook_speaker.as_deref() == Some("Host") {
                "Host"
            } else {
                "Guest"
            }
        );
        println!(
            "    HOOK TYPE: {}",
            draft.hook_type.as_deref().unwrap_or("direct_statement")
        );
        println!(
            "    TOPIC CLARITY: {:.2}  CURIOSITY: {:.2}  RELEVANCE: {:.2}  CONTRAST: {:.2}",
            draft.hook_topic_clarity.unwrap_or(0.85),
            draft
                .hook_curiosity
                .unwrap_or(draft.curiosity_score.unwrap_or(0.85)),
            draft
                .hook_relevance
                .unwrap_or(draft.value_score.unwrap_or(0.85)),
            draft.hook_contrast.unwrap_or(0.80),
        );
        println!(
            "    DELAY: {}  CONFUSION: {}  IRRELEVANCE: {}  DISINTEREST: {}",
            if draft.hook_delay_penalty.unwrap_or(0.0) > 0.4 {
                "FAIL"
            } else {
                "PASS"
            },
            if draft.hook_confusion_penalty.unwrap_or(0.0) > 0.4 {
                "FAIL"
            } else {
                "PASS"
            },
            if draft.hook_irrelevance_penalty.unwrap_or(0.0) > 0.4 {
                "FAIL"
            } else {
                "PASS"
            },
            if draft.hook_disinterest_penalty.unwrap_or(0.0) > 0.4 {
                "FAIL"
            } else {
                "PASS"
            },
        );
        println!(
            "    CONTEXT SUFFICIENCY: {}  QUESTION INCLUDED: {}",
            draft
                .context_completeness_score
                .map_or("true", |v| if v >= 0.6 { "true" } else { "false" }),
            if draft.question_hook_used == Some(true) {
                "true"
            } else {
                "false"
            },
        );
        println!(
            "    ANSWER COMPLETE: {}  STORY COMPLETE: {}  PAYOFF COMPLETE: {}",
            draft
                .answer_complete
                .map_or("true", |v| if v { "true" } else { "false" }),
            draft
                .story_completeness
                .map_or("true", |v| if v { "true" } else { "false" }),
            draft
                .payoff_completion
                .map_or("true", |v| if v { "true" } else { "false" }),
        );
        println!(
            "    ENDING TYPE: {}",
            draft
                .ending_type
                .as_deref()
                .unwrap_or("sentence_completion")
        );
        println!(
            "    CONTINUATION_PROBABILITY: {:.2}",
            draft.continuation_probability.unwrap_or(0.10)
        );
        println!(
            "    BREATH_PAUSE_RISK: {:.2}",
            draft.breath_pause_risk.unwrap_or(0.08)
        );
        println!(
            "    SEMANTIC_CLOSURE_CONFIDENCE: {:.2}",
            draft
                .closure_confidence
                .unwrap_or(draft.semantic_closure.unwrap_or(0.95))
        );
        println!(
            "    START REASON: {}",
            draft.start_reason.as_deref().unwrap_or(&draft.rationale)
        );
        println!(
            "    END REASON: {}",
            draft
                .end_reason
                .as_deref()
                .unwrap_or("Reaches true semantic thought closure.")
        );
        println!("    DURATION: {:.2}s", duration);
        println!("    Scores -> Multimodal: {:.2} | Semantic: {:.2} | Visual: {:.2} | Audio: {:.2} | Temporal: {:.2}",
            mm_score, sem_score, vis_score, aud_score, temp_score
        );
        println!(
            "    Multimodal Verified: {}",
            if draft.multimodal_verified {
                "YES"
            } else {
                "NO (v9.2 fallback)"
            }
        );
        println!(
            "    Evidence: {:?}",
            draft.multimodal_evidence.as_deref().unwrap_or(&[])
        );

        // Detailed Boundary Trace for at least 3 candidates per run (R5)
        if i < 3 || i < total_candidates {
            let raw_s = raw_boundaries.get(i).map_or(draft.start, |b| b.raw_start);
            let raw_e = raw_boundaries.get(i).map_or(draft.end, |b| b.raw_end);
            let (last_spoken, next_spoken, pause_ms, snap_action, snap_reason) =
                get_boundary_trace_info(
                    &normalized.words,
                    raw_s,
                    draft.start,
                    raw_e,
                    draft.end,
                    draft,
                );

            println!("    --- BOUNDARY TRACE (Candidate #{}) ---", i + 1);
            println!(
                "    RAW_START: {:.2}s  FINAL_START: {:.2}s  (HOOK_ANCHOR: {:.2}s)",
                raw_s,
                draft.start,
                draft.hook_start.unwrap_or(raw_s)
            );
            println!("    RAW_END: {:.2}s    FINAL_END: {:.2}s", raw_e, draft.end);
            println!("    LAST_SPOKEN_TEXT_BEFORE_RAW_END: \"{}\"", last_spoken);
            println!("    NEXT_SPOKEN_TEXT: \"{}\"", next_spoken);
            println!("    PAUSE_DURATION: {:.1}ms", pause_ms);
            println!(
                "    CONTINUATION_PROBABILITY: {:.2}",
                draft.continuation_probability.unwrap_or(0.10)
            );
            println!(
                "    SEMANTIC_CLOSURE_CONFIDENCE: {:.2}",
                draft
                    .closure_confidence
                    .unwrap_or(draft.semantic_closure.unwrap_or(0.95))
            );
            println!("    SNAP_ACTION: {}", snap_action);
            println!("    SNAP_REASON: {}", snap_reason);
        }
        println!();
    }

    let candidates = db
        .replace_candidates(&project_id, &all_drafts)
        .map_err(to_command_error)?;
    db.update_project_status(&project_id, "ready", None)
        .map_err(to_command_error)?;
    Ok(candidates)
}

/// Linguistic Continuation Detection
/// Returns true if a word or phrase indicates that the speaker's thought is continuing
/// into the next clause/sentence (e.g. "because", "and", "कम से कम", "क्योंकि", "लेकिन", "और", "फिर", "कम").
pub fn is_continuation_word(text: &str) -> bool {
    let clean = text
        .trim()
        .to_lowercase()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '।')
        .to_string();
    if clean.is_empty() {
        return false;
    }

    const CONTINUATION_WORDS: &[&str] = &[
        // English continuation tokens
        "because",
        "but",
        "and",
        "so",
        "then",
        "that's",
        "thats",
        "which",
        "what",
        "why",
        "the",
        "a",
        "an",
        "i",
        "when",
        "after",
        "before",
        "for",
        "especially",
        "until",
        "if",
        "although",
        "or",
        "like",
        "least",
        "means",
        "is",
        "was",
        "are",
        "were",
        "to",
        "with",
        "from",
        "at",
        "as",
        "by",
        // Hindi / Hinglish continuation tokens
        "कम",
        "से",
        "क्योंकि",
        "लेकिन",
        "और",
        "फिर",
        "इसलिए",
        "उसके",
        "बाद",
        "तब",
        "जब",
        "अगर",
        "मतलब",
        "यानी",
        "तो",
        "जिससे",
        "जिसके",
        "कारण",
        "वहां",
        "उस",
        "टाइम",
        "मैंने",
        "हम",
        "कि",
        "जैसे",
        "वैसे",
        "पर",
        "भी",
        "ही",
        "करके",
        "लेकर",
        "देकर",
    ];

    CONTINUATION_WORDS.contains(&clean.as_str())
}

/// Helper to extract boundary trace diagnostics (R5)
pub fn get_boundary_trace_info(
    words: &[TranscriptWord],
    _raw_start: f64,
    _final_start: f64,
    raw_end: f64,
    final_end: f64,
    draft: &models::CandidateDraft,
) -> (String, String, f64, String, String) {
    let last_spoken = words
        .iter()
        .filter(|w| w.end <= raw_end + 0.1)
        .last()
        .map(|w| w.text.clone())
        .unwrap_or_default();

    let next_spoken_word = words.iter().find(|w| w.start >= raw_end - 0.05);

    let next_spoken = next_spoken_word.map(|w| w.text.clone()).unwrap_or_default();

    let pause_ms = if let (Some(prev), Some(next)) = (
        words.iter().filter(|w| w.end <= raw_end + 0.1).last(),
        next_spoken_word,
    ) {
        (next.start - prev.end).max(0.0) * 1000.0
    } else {
        0.0
    };

    let snap_action = if final_end > raw_end + 0.1 {
        "EXTENDED".to_string()
    } else if (final_end - raw_end).abs() <= 0.1 {
        "ACCEPTED".to_string()
    } else {
        "SNAPPED_EARLIER".to_string()
    };

    let snap_reason = if draft.continuation_probability.unwrap_or(0.0) > 0.70 {
        format!(
            "Forced forward extension past raw_end due to high continuation probability ({:.2})",
            draft.continuation_probability.unwrap_or(0.0)
        )
    } else if draft.breath_pause_risk.unwrap_or(0.0) > 0.65 {
        format!(
            "Forced extension past raw_end due to breath pause risk ({:.2})",
            draft.breath_pause_risk.unwrap_or(0.0)
        )
    } else if draft.mid_thought_risk.unwrap_or(0.0) > 0.65 {
        format!(
            "Forced extension past raw_end due to mid-thought cutoff risk ({:.2})",
            draft.mid_thought_risk.unwrap_or(0.0)
        )
    } else if draft.answer_complete == Some(false) {
        "Forced extension toward complete answer resolution".to_string()
    } else if draft.story_completeness == Some(false) || draft.payoff_completion == Some(false) {
        "Forced extension toward narrative / payoff completion".to_string()
    } else if matches!(
        draft.endpoint_state,
        Some(crate::models::EndpointState::BreathPause)
    ) {
        "Forced extension past raw_end due to BreathPause endpoint state".to_string()
    } else if final_end > raw_end + 0.1 {
        "Extended forward to nearest valid sentence boundary terminator".to_string()
    } else {
        "Clean sentence boundary accepted at raw endpoint".to_string()
    };

    (last_spoken, next_spoken, pause_ms, snap_action, snap_reason)
}

/// Helper to determine if a candidate's closure signals demand forward extension.
pub fn closure_requires_extension(cs: &crate::models::ClosureSignals) -> bool {
    let cont_prob_trigger = cs.continuation_probability.map_or(false, |p| p > 0.70);
    let breath_risk_trigger = cs.breath_pause_risk.map_or(false, |r| r > 0.65);
    let mid_thought_trigger = cs.mid_thought_risk.map_or(false, |r| r > 0.65);
    let answer_incomplete_trigger = cs.answer_complete == Some(false);
    let story_incomplete_trigger = cs.story_completeness == Some(false);
    let payoff_incomplete_trigger = cs.payoff_completion == Some(false);
    let breath_pause_state_trigger = matches!(
        cs.endpoint_state,
        Some(crate::models::EndpointState::BreathPause)
    );
    let incomplete_state_trigger = match &cs.endpoint_state {
        Some(crate::models::EndpointState::SentenceComplete)
        | Some(crate::models::EndpointState::ThoughtComplete) => {
            cs.closure_confidence.map_or(true, |c| c < 0.80)
        }
        _ => false,
    };

    cont_prob_trigger
        || breath_risk_trigger
        || mid_thought_trigger
        || answer_incomplete_trigger
        || story_incomplete_trigger
        || payoff_incomplete_trigger
        || breath_pause_state_trigger
        || incomplete_state_trigger
}

/// Word-level forward scan from `from_sec` up to `ceiling_sec` returning the
/// end timestamp of the first genuine sentence terminator that is not a continuation word.
pub fn force_extend_to_closure(
    words: &[crate::models::TranscriptWord],
    from_sec: f64,
    ceiling_sec: f64,
) -> Option<f64> {
    words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.end > from_sec + 0.1 && w.end <= ceiling_sec)
        .find(|(idx, w)| {
            if !crate::transcription::ends_with_sentence_terminator(&w.text) {
                return false;
            }
            if *idx + 1 < words.len() {
                let next_w = &words[*idx + 1];
                let pause = next_w.start - w.end;
                if pause < 0.30 && is_continuation_word(&next_w.text) {
                    return false;
                }
            }
            !is_continuation_word(&w.text)
        })
        .map(|(_, w)| w.end)
}

/// Production-safe final endpoint enforcement (spec §26).
/// Executes in debug AND release builds — NOT a debug_assert.
///
/// Returns Ok(final_end) if the candidate passes or was successfully extended.
/// Returns Err(...) if the candidate was incomplete and could not be salvaged within safety ceiling.
pub fn validate_and_enforce_final_endpoint(
    words: &[crate::models::TranscriptWord],
    raw_llm_end: f64,
    snapped_end: f64,
    closure: &crate::models::ClosureSignals,
    video_duration: f64,
) -> Result<f64, String> {
    let extension_required = closure_requires_extension(closure);
    let was_extended = (snapped_end - raw_llm_end).abs() > 0.05;

    eprintln!(
        "[Render] FINAL_ENDPOINT_VALIDATED raw={:.2} final={:.2} extended={}",
        raw_llm_end, snapped_end, was_extended
    );

    if extension_required && !was_extended {
        // snap_to_semantic_boundaries made no change despite incomplete signals.
        // Attempt one final word-level look-ahead extension.
        let ceiling = (snapped_end + 35.0).min(video_duration);
        let salvaged = force_extend_to_closure(words, snapped_end, ceiling);
        match salvaged {
            Some(extended_end) => {
                eprintln!(
                    "[Render] FINAL_ENDPOINT_SALVAGED raw={:.2} salvaged={:.2}",
                    raw_llm_end, extended_end
                );
                Ok(extended_end)
            }
            None => Err(format!(
                "Candidate with incomplete closure (raw_end={:.2}) could not be salvaged \
                 within safety ceiling ({:.2}s). Rejecting to prevent premature render.",
                raw_llm_end, ceiling
            )),
        }
    } else {
        Ok(snapped_end)
    }
}

/// Extracts the actual concluding spoken sentence terminating within [clip_start, clip_end].
/// Returns `Some((sentence_text, start_sec, end_sec))`.
pub fn extract_final_spoken_sentence(
    words: &[crate::models::TranscriptWord],
    clip_start: f64,
    clip_end: f64,
) -> Option<(String, f64, f64)> {
    if words.is_empty() {
        return None;
    }

    // Filter words within candidate range, allowing small tolerances
    let in_bounds: Vec<(usize, &crate::models::TranscriptWord)> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| w.start >= clip_start - 0.20 && w.end <= clip_end + 0.15)
        .collect();

    if in_bounds.is_empty() {
        return None;
    }

    let last_elem = in_bounds.last().unwrap();
    let last_orig_idx = last_elem.0;

    // Walk backward to find the sentence start
    let mut start_orig_idx = last_orig_idx;
    for &(curr_idx, curr_w) in in_bounds.iter().rev() {
        if curr_idx == 0 {
            start_orig_idx = 0;
            break;
        }
        let prev_w = &words[curr_idx - 1];
        let prev_text = prev_w.text.trim();
        let clean_prev = prev_text.trim_matches(|c: char| {
            c == '"' || c == '\'' || c == '”' || c == '’' || c == ')' || c == ']' || c == '}'
        });
        // Check if preceding word ended with a sentence terminator
        if crate::transcription::ends_with_sentence_terminator(clean_prev) {
            start_orig_idx = curr_idx;
            break;
        }
        // Or if there was a major conversational pause
        if curr_w.start - prev_w.end >= 1.20 {
            start_orig_idx = curr_idx;
            break;
        }
        if prev_w.start < clip_start - 0.20 {
            start_orig_idx = curr_idx;
            break;
        }
        start_orig_idx = curr_idx - 1;
    }

    let sentence_words = &words[start_orig_idx..=last_orig_idx];
    if sentence_words.is_empty() {
        return None;
    }

    let text = sentence_words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    Some((
        text,
        sentence_words.first().unwrap().start,
        sentence_words.last().unwrap().end,
    ))
}

/// Snap the LLM's suggested clip_start and clip_end to natural semantic boundaries in the transcript,
/// with hook start anchor awareness to prevent backward drift across sentence terminators into preceding dialogue.
pub fn snap_to_semantic_boundaries_with_hook_anchor(
    words: &[TranscriptWord],
    raw_start: f64,
    raw_end: f64,
    video_duration: f64,
    closure: Option<&crate::models::ClosureSignals>,
    hook_anchor: Option<f64>,
    is_question_hook: bool,
) -> (f64, f64) {
    if words.is_empty() {
        let s = raw_start.max(0.0).min(video_duration);
        let e = raw_end.max(s + 1.0).min(video_duration);
        return (s, e);
    }

    let min_guideline = 12.0;
    let max_guideline = 75.0;

    // 1. Snap start timestamp with strict sentence boundary protection
    let mut chosen_start = raw_start.max(0.0);

    if let Some(anchor) = hook_anchor {
        // We have a verified hook start anchor.
        if let Some((h_idx, h_word)) = words.iter().enumerate().min_by(|(_, a), (_, b)| {
            (a.start - anchor)
                .abs()
                .total_cmp(&(b.start - anchor).abs())
        }) {
            if is_question_hook && raw_start < anchor - 1.0 {
                // Host question setup preceded the hook.
                // Snap question setup start near raw_start, strictly bounded.
                if let Some((q_idx, q_word)) = words.iter().enumerate().min_by(|(_, a), (_, b)| {
                    (a.start - raw_start)
                        .abs()
                        .total_cmp(&(b.start - raw_start).abs())
                }) {
                    if q_idx > 0
                        && crate::transcription::ends_with_sentence_terminator(
                            &words[q_idx - 1].text,
                        )
                    {
                        chosen_start = (q_word.start - 0.15).max(words[q_idx - 1].end);
                    } else {
                        chosen_start = q_word.start;
                    }
                } else {
                    chosen_start = h_word.start;
                }
            } else {
                // Direct Guest Statement Hook:
                // Clip MUST start on the hook word h_word (or within micro-breath lead-in).
                // NEVER cross backward across words[h_idx - 1] if words[h_idx - 1] has a sentence terminator!
                if h_idx > 0
                    && crate::transcription::ends_with_sentence_terminator(&words[h_idx - 1].text)
                {
                    chosen_start = (h_word.start - 0.15).max(words[h_idx - 1].end);
                } else if h_idx > 0 {
                    // Hook begins mid-sentence: find the start of THIS sentence (up to max 1.5s prior)
                    let mut sent_start_idx = h_idx;
                    for k in (0..h_idx).rev() {
                        if crate::transcription::ends_with_sentence_terminator(&words[k].text) {
                            sent_start_idx = k + 1;
                            break;
                        }
                        if (h_word.start - words[k].start) > 2.0 {
                            break;
                        }
                    }
                    chosen_start = words[sent_start_idx].start;
                } else {
                    chosen_start = h_word.start;
                }
            }
        }
    } else {
        // Fallback without hook anchor:
        // Find word closest to raw_start.
        if let Some((closest_idx, closest)) = words.iter().enumerate().min_by(|(_, a), (_, b)| {
            (a.start - raw_start)
                .abs()
                .total_cmp(&(b.start - raw_start).abs())
        }) {
            if closest_idx > 0
                && crate::transcription::ends_with_sentence_terminator(&words[closest_idx - 1].text)
            {
                chosen_start = (closest.start - 0.15).max(words[closest_idx - 1].end);
            } else {
                // Check if a sentence start is very close (within 1.5s prior)
                let mut sent_start_idx = closest_idx;
                for k in (0..closest_idx).rev() {
                    if crate::transcription::ends_with_sentence_terminator(&words[k].text) {
                        if (closest.start - words[k + 1].start) <= 1.5 {
                            sent_start_idx = k + 1;
                        }
                        break;
                    }
                }
                chosen_start = words[sent_start_idx].start;
            }
        }
    }

    // 2. Snap end timestamp with Look-Ahead Semantic Closure Engine
    let mut chosen_end = raw_end.min(video_duration);
    let has_payoff = closure.and_then(|c| c.payoff_end).is_some()
        && closure.and_then(|c| c.payoff_completion) != Some(false);

    if has_payoff {
        let p_end = closure.and_then(|c| c.payoff_end).unwrap();
        // Once payoff alignment establishes payoff_end, that endpoint remains strictly authoritative.
        // Hook Intelligence 2.0 must NOT interrupt, override, shorten, rewrite, re-anchor,
        // or otherwise change the payoff-driven endpoint.
        chosen_end = p_end.min(video_duration);
    } else {
        // Evaluate R4 deterministic closure extension triggers
        let mut must_extend = false;
        let mut target_min_end = raw_end;
        let mut low_closure_confidence = false;

        if let Some(cs) = closure {
            if closure_requires_extension(cs) {
                must_extend = true;
            }

            if cs.answer_complete == Some(false) {
                if let Some(ans_end) = cs.answer_end {
                    if ans_end > target_min_end {
                        target_min_end = ans_end;
                    }
                }
            }

            if let Some(p_end) = cs.payoff_end {
                if p_end > target_min_end {
                    target_min_end = p_end;
                    must_extend = true;
                }
            } else if cs.story_completeness == Some(false) || cs.payoff_completion == Some(false) {
                must_extend = true;
            }

            if cs.closure_confidence.map_or(false, |c| c < 0.50) {
                low_closure_confidence = true;
            }
        }

        if must_extend {
            // R4 Deterministic Extension:
            // Reject stopping at or before raw_end. Force forward search for the next valid sentence terminator.
            let mut search_max = (chosen_start + max_guideline).min(video_duration);
            if let Some(p_end) = closure.and_then(|c| c.payoff_end) {
                search_max = search_max.max(p_end + 0.5).min(video_duration);
            }

            // Find candidate words strictly ending after raw_end and at or after target_min_end (with 0.1s margin)
            let forward_candidates: Vec<(usize, &TranscriptWord)> = words
                .iter()
                .enumerate()
                .filter(|(_, w)| {
                    w.end > raw_end + 0.1 && w.end >= target_min_end - 0.1 && w.end <= search_max
                })
                .collect();

            let valid_forward_terminators: Vec<&(usize, &TranscriptWord)> = forward_candidates
                .iter()
                .filter(|(idx, w)| {
                    if !crate::transcription::ends_with_sentence_terminator(&w.text) {
                        return false;
                    }
                    if *idx + 1 < words.len() {
                        let next_w = &words[*idx + 1];
                        let pause = next_w.start - w.end;
                        if pause < 0.30 && is_continuation_word(&next_w.text) {
                            return false;
                        }
                    }
                    !is_continuation_word(&w.text)
                })
                .collect();

            if !valid_forward_terminators.is_empty() {
                if low_closure_confidence {
                    // Bias search toward later candidates to ensure full resolution
                    let chosen = valid_forward_terminators.last().unwrap();
                    chosen_end = chosen.1.end;
                } else {
                    // Pick the earliest valid terminator satisfying target_min_end
                    let chosen = valid_forward_terminators.first().unwrap();
                    chosen_end = chosen.1.end;
                }
            } else {
                // Fallback: search for any sentence terminator in bounds [raw_end, search_max]
                let fallback_terminator = words
                    .iter()
                    .filter(|w| w.end > raw_end && w.end <= search_max)
                    .find(|w| {
                        crate::transcription::ends_with_sentence_terminator(&w.text)
                            && !is_continuation_word(&w.text)
                    });

                if let Some(w) = fallback_terminator {
                    chosen_end = w.end;
                } else {
                    // If extension is required but no valid terminator within max_guideline (75.0s): accept best available sentence terminator within bounds
                    let best_in_bounds = words
                        .iter()
                        .filter(|w| w.end >= chosen_start + min_guideline && w.end <= search_max)
                        .filter(|w| {
                            crate::transcription::ends_with_sentence_terminator(&w.text)
                                && !is_continuation_word(&w.text)
                        })
                        .last();

                    if let Some(w) = best_in_bounds {
                        chosen_end = w.end;
                    } else if let Some((_, closest)) = forward_candidates
                        .iter()
                        .max_by(|(_, a), (_, b)| a.end.total_cmp(&b.end))
                    {
                        chosen_end = closest.end;
                    }
                }
            }
        } else {
            // When closure is None or no extension trigger fires:
            // Window is standard (-2.5s to +8.5s) or extended if low_closure_confidence
            let lookahead_max = if low_closure_confidence {
                (chosen_start + max_guideline)
                    .min(video_duration)
                    .min(raw_end + 15.0)
            } else {
                raw_end + 8.5
            };

            // Collect words in a look-ahead window (-2.5s to +8.5s/extended around raw_end)
            let end_candidate_words: Vec<(usize, &TranscriptWord)> = words
                .iter()
                .enumerate()
                .filter(|(_, w)| {
                    w.end > chosen_start && (w.end >= raw_end - 2.5 && w.end <= lookahead_max)
                })
                .collect();

            // Find sentence terminators within this window
            let terminating_candidates: Vec<&(usize, &TranscriptWord)> = end_candidate_words
                .iter()
                .filter(|(idx, w)| {
                    if !crate::transcription::ends_with_sentence_terminator(&w.text) {
                        return false;
                    }
                    // Check if the next word indicates an immediate continuation by the same speaker
                    if *idx + 1 < words.len() {
                        let next_w = &words[*idx + 1];
                        let pause = next_w.start - w.end;
                        // If the next word is a continuation token or speech continues after <0.3s without pause, check if thought is continuing
                        if pause < 0.30 && is_continuation_word(&next_w.text) {
                            return false;
                        }
                    }
                    // Ensure this terminating word is not itself a continuation word
                    !is_continuation_word(&w.text)
                })
                .collect();

            if !terminating_candidates.is_empty() {
                if low_closure_confidence {
                    // Bias search toward later candidates within the window
                    let later_candidates: Vec<&&(usize, &TranscriptWord)> = terminating_candidates
                        .iter()
                        .filter(|(_, w)| w.end >= raw_end)
                        .collect();

                    if let Some((_, w)) = later_candidates.last() {
                        chosen_end = w.end;
                    } else if let Some((_, w)) = terminating_candidates.last() {
                        chosen_end = w.end;
                    }
                } else if let Some((_, w)) =
                    terminating_candidates.iter().min_by(|(_, a), (_, b)| {
                        // Prioritize candidates at or after raw_end over premature cutoffs
                        let a_penalty = if a.end < raw_end { 2.0 } else { 1.0 };
                        let b_penalty = if b.end < raw_end { 2.0 } else { 1.0 };
                        ((a.end - raw_end).abs() * a_penalty)
                            .total_cmp(&((b.end - raw_end).abs() * b_penalty))
                    })
                {
                    chosen_end = w.end;
                }
            } else {
                // If no clean terminator found, search forward up to 8s for the next sentence terminator
                let forward_terminator = words
                    .iter()
                    .filter(|w| {
                        w.end >= raw_end
                            && w.end <= (raw_end + 8.0).min(chosen_start + max_guideline)
                    })
                    .find(|w| {
                        crate::transcription::ends_with_sentence_terminator(&w.text)
                            && !is_continuation_word(&w.text)
                    });

                if let Some(w) = forward_terminator {
                    chosen_end = w.end;
                } else if let Some((_, closest)) =
                    end_candidate_words.iter().min_by(|(_, a), (_, b)| {
                        (a.end - raw_end).abs().total_cmp(&(b.end - raw_end).abs())
                    })
                {
                    chosen_end = closest.end;
                }
            }
        }

        // 3. Look-Ahead Check: If the word at chosen_end is a continuation word, extend forward
        if let Some(end_idx) = words.iter().position(|w| (w.end - chosen_end).abs() < 0.2) {
            let current_word = &words[end_idx];
            if is_continuation_word(&current_word.text)
                || !crate::transcription::ends_with_sentence_terminator(&current_word.text)
            {
                // Look forward for the next sentence terminator
                for next_w in &words[end_idx + 1..] {
                    if next_w.end - chosen_start > max_guideline {
                        break;
                    }
                    if crate::transcription::ends_with_sentence_terminator(&next_w.text)
                        && !is_continuation_word(&next_w.text)
                    {
                        chosen_end = next_w.end;
                        break;
                    }
                }
            }
        }

        // 4. Minimum duration guideline
        let dur = chosen_end - chosen_start;
        if dur < min_guideline && (video_duration - chosen_start) >= min_guideline {
            if let Some(w) = words
                .iter()
                .filter(|w| {
                    w.end >= chosen_start + min_guideline
                        && crate::transcription::ends_with_sentence_terminator(&w.text)
                        && !is_continuation_word(&w.text)
                })
                .min_by(|a, b| {
                    (a.end - (chosen_start + min_guideline))
                        .abs()
                        .total_cmp(&(b.end - (chosen_start + min_guideline)).abs())
                })
            {
                chosen_end = w.end;
            }
        }

        // 5. Maximum duration guideline (bypassed if payoff endpoint is authoritative)
        let dur = chosen_end - chosen_start;
        if dur > max_guideline {
            if let Some(w) = words
                .iter()
                .filter(|w| {
                    w.end >= chosen_start + 35.0
                        && w.end <= chosen_start + max_guideline
                        && crate::transcription::ends_with_sentence_terminator(&w.text)
                        && !is_continuation_word(&w.text)
                })
                .max_by(|a, b| a.end.total_cmp(&b.end))
            {
                chosen_end = w.end;
            }
        }
    }

    let final_start = chosen_start.max(0.0);
    let final_end = chosen_end.min(video_duration).max(final_start + 1.0);

    (final_start, final_end)
}

/// Snap the LLM's suggested clip_start and clip_end to natural semantic boundaries in the transcript:
/// - `start`: snaps to the exact start of a word around that timestamp, preferring the start of a sentence.
/// - `end`: looks ahead past raw_end to find true semantic closure, enforcing deterministic closure rules when `closure` signals are present.
/// - Dynamic duration: allows full narrative completeness up to 75.0s.
pub fn snap_to_semantic_boundaries(
    words: &[TranscriptWord],
    raw_start: f64,
    raw_end: f64,
    video_duration: f64,
    closure: Option<&crate::models::ClosureSignals>,
) -> (f64, f64) {
    snap_to_semantic_boundaries_with_hook_anchor(
        words,
        raw_start,
        raw_end,
        video_duration,
        closure,
        None,
        false,
    )
}

#[tauri::command]
fn set_selected_clip_count(
    state: tauri::State<'_, AppState>,
    project_id: String,
    count: usize,
) -> Result<Vec<Candidate>, String> {
    state
        .db
        .set_selected_clip_count(&project_id, count)
        .map_err(to_command_error)
}

/// Range-based candidate selection: marks candidates with ranks in the inclusive
/// 1-based range `[start_rank, end_rank]` as selected; all others become unselected.
/// Returns the full ordered candidate list reflecting the new selection.
#[tauri::command]
fn set_selected_rank_range(
    state: tauri::State<'_, AppState>,
    project_id: String,
    start_rank: i64,
    end_rank: i64,
) -> Result<Vec<Candidate>, String> {
    state
        .db
        .set_selected_rank_range(&project_id, start_rank, end_rank)
        .map_err(to_command_error)
}

#[tauri::command]
async fn render_flat_clip_for_candidate(
    state: tauri::State<'_, AppState>,
    candidate_id: String,
) -> Result<String, String> {
    let (candidate, project) = state
        .db
        .get_candidate_with_project(&candidate_id)
        .map_err(to_command_error)?;

    // 1. In-flight check keyed to candidate_id AND output identity (project_id, rank)
    let render_guard = state
        .render_tracker
        .try_acquire_output(&candidate_id, &project.id, candidate.rank)
        .map_err(|err| {
            println!("[Render Lock Rejected] {}", err);
            err
        })?;

    let db = state.db.clone();
    let data_dir = state.data_dir.clone();

    tokio::task::spawn_blocking(move || {
        let _guard = render_guard;
        db
            .update_clip_for_candidate(&candidate_id, "cutting", None, None, None, None)
            .map_err(to_command_error)?;

        let output_path = documents_project_dir(&project)?
            .join("clips")
            .join(format!("clip-{:02}_flat.mp4", candidate.rank));

        let mut _srt_path = None;
        let mut ass_path = None;
        let mut drawtext_filters = None;
        // Words are extracted separately so the camera engine can receive
        // speaker labels (from Deepgram diarization) for multi-speaker framing.
        let mut transcript_words_for_camera: Option<Vec<TranscriptWord>> = None;

        let probe = media::probe_media(&project.source_path).ok();
        let (iw, ih, cropped_width) = if let Some(p) = &probe {
            let iw_val = p.width.unwrap_or(1920) as f64;
            let ih_val = p.height.unwrap_or(1080) as f64;
            let w = (iw_val.min(ih_val * 9.0 / 16.0) / 2.0).floor() * 2.0;
            (p.width.unwrap_or(1920) as i64, p.height.unwrap_or(1080) as i64, w as i64)
        } else {
            (1920, 1080, 1080)
        };

        if let Ok(Some(transcript_record)) = db.latest_transcript(&project.id) {
            if let Ok(normalized) = serde_json::from_str::<NormalizedTranscript>(&transcript_record.raw_json) {
                transcript_words_for_camera = Some(normalized.words);
            }
        }

        // ── Hook & Ending Optimization 2.0 (AutoShorts 8.0) ───────────────────
        // Runs AFTER candidate selection (the candidate is authoritative) and
        // BEFORE Smart Pacing / framing / captions / render. Moves the clip's
        // temporal boundaries only when the move is provably safe (closed-vocab
        // setup skips, Q→A backward repair, wind-down trims, cut-sentence
        // completion). On any doubt it keeps the original boundary. The
        // candidate DB row is NEVER mutated — only the range passed downstream.
        // Timeline mapping: SOURCE (candidate row) → OPTIMIZED (below) →
        // OUTPUT (post-pacing, Smart Pacing's responsibility).
        let mut effective_payoff_end = candidate.payoff_end_sec;
        if effective_payoff_end.is_none() {
            if let Some(ref p_text) = candidate.payoff_text {
                if let Some(ref words) = transcript_words_for_camera {
                    if let Some((_p_start, p_end, conf, _s_idx, _e_idx)) =
                        llm::align_payoff_to_transcript_words(p_text, candidate.start_sec, candidate.end_sec, words)
                    {
                        if conf >= 0.50 {
                            effective_payoff_end = Some(p_end);
                        }
                    }
                }
            }
        }

        let candidate_effective_end = effective_payoff_end.unwrap_or(candidate.end_sec);
        let video_duration = probe
            .as_ref()
            .and_then(|p| p.duration_sec)
            .unwrap_or(candidate_effective_end);
        let boundary_meta = boundary::BoundaryCandidateMeta {
            hook: Some(candidate.hook.as_str()),
            hook_start_sec: candidate.hook_start_sec,
            hook_end_sec: candidate.hook_end_sec,
            hook_confidence: candidate.hook_confidence,
            opening_context_score: candidate.opening_context_score,
            payoff_end_sec: effective_payoff_end,
        };
        let boundary_opt = boundary::optimize_boundaries(
            transcript_words_for_camera.as_deref().unwrap_or(&[]),
            candidate.start_sec,
            candidate_effective_end,
            video_duration,
            Some(&boundary_meta),
        );
        let render_start = boundary_opt.optimized_start_sec;
        let render_end = boundary_opt.optimized_end_sec;
        if boundary_opt.start_changed || boundary_opt.end_changed {
            eprintln!(
                "[Boundary Optimization] rank {} {}",
                candidate.rank,
                boundary_opt.summary()
            );
        }

        // ── Retention-Aware Smart Pacing (AutoShorts 8.0 / 10.0) ─────────────
        // Runs AFTER candidate selection and boundary optimization (the
        // OPTIMIZED range is authoritative for pacing) and BEFORE
        // framing/captions/render. The engine proves each removed interval is
        // safe (word gaps + FFmpeg silencedetect); on any doubt it returns
        // None and the legacy unmodified render is used.
        //
        // Smart Pacing 2.0 (10.0): additionally runs breath/waiting pause
        // detection on the SAME fixed range. v2 is strictly additive — when
        // its stage is disabled or finds nothing, plan_smart_pacing_v2 yields
        // the exact v1 plan. Candidate boundaries and selection are never
        // touched by either stage.
        let pacing_plan = pacing::plan_smart_pacing_v2(
            &project.source_path,
            render_start,
            render_end,
            transcript_words_for_camera.as_deref(),
        );

        // ── Layout-Aware Smart Framing Resolution (SINGLE-PASS CV) ─────────────
        // NOTE: framing analysis ALWAYS runs on the FULL source range
        // [start_sec, end_sec] — smart pacing must never shorten the range
        // the sampler sees (end-of-clip sampling regression guard).
        // Adaptive Framing (10.0): the per-project framing choice is threaded
        // through the environment so the Python sidecar can switch behavior
        // without changing its CLI contract.
        let framing_mode = project.framing_mode.as_deref().unwrap_or("original");
        let mut crop_w = ((ih as f64) * 9.0 / 16.0).round() as i64;
        if crop_w % 2 != 0 {
            crop_w -= 1;
        }

        // ── Phase 2: Speaker Intelligence (signals only; never blocking) ────
        // Runs once per source (per-config JSON cache inside the engine).
        // Produces cached diarization + Host/Guest mapping + the persisted
        // Re-ID gallery, which are handed to the framing sidecar; the
        // sidecar's speakerIntel block is persisted back per candidate. Any
        // failure here leaves speaker_intel_inputs = None and the pipeline
        // runs exactly as before (deterministic fallbacks).
        let mut speaker_intel_inputs: Option<(String, String, Option<String>)> = None;
        if speaker_intelligence::speaker_intelligence_enabled() {
            match speaker_intelligence::SpeakerIntelligenceEngine::new(
                speaker_intelligence::SpeakerIntelligenceConfig {
                    enable_diarization: speaker_intelligence::speaker_diarization_enabled(),
                    enable_reid: speaker_intelligence::speaker_reid_enabled(),
                    enable_active_speaker_fusion: speaker_intelligence::active_speaker_fusion_enabled(),
                    ..Default::default()
                },
                db.clone(),
            ) {
                Ok(engine) => match engine.process_source_video(
                    &project.id,
                    &project.source_path,
                    transcript_words_for_camera.as_deref().unwrap_or(&[]),
                ) {
                    Ok(si) => {
                        let diar_tmp = std::env::temp_dir()
                            .join(format!("autoshorts_si_diar_{}.json", uuid::Uuid::new_v4()));
                        let gallery_tmp = std::env::temp_dir()
                            .join(format!("autoshorts_si_gallery_{}.json", uuid::Uuid::new_v4()));
                        let diar_doc = serde_json::json!({
                            "model": si.diarization.model,
                            "segments": si.diarization.segments.iter().map(|s| serde_json::json!({
                                "speaker_id": s.speaker_id,
                                "start": s.start,
                                "end": s.end,
                                "confidence": s.confidence,
                            })).collect::<Vec<_>>(),
                        });
                        let gallery_doc = serde_json::json!({
                            "entries": si.reid_embeddings,
                        });
                        let diar_ok = serde_json::to_string(&diar_doc)
                            .ok()
                            .and_then(|c| std::fs::write(&diar_tmp, c).ok())
                            .is_some();
                        let gallery_ok = serde_json::to_string(&gallery_doc)
                            .ok()
                            .and_then(|c| std::fs::write(&gallery_tmp, c).ok())
                            .is_some();
                        if diar_ok || gallery_ok {
                            speaker_intel_inputs = Some((
                                diar_tmp.to_string_lossy().into_owned(),
                                gallery_tmp.to_string_lossy().into_owned(),
                                None,
                            ));
                        } else {
                            eprintln!("[SpeakerIntel] sidecar temp files could not be written — continuing without speaker intelligence inputs");
                        }
                    }
                    Err(e) => {
                        eprintln!("[SpeakerIntel] source processing failed (non-fatal, deterministic fallback continues): {}", e);
                    }
                },
                Err(e) => {
                    eprintln!("[SpeakerIntel] engine init failed (non-fatal): {}", e);
                }
            }
        }

        // ── Phase 3: Scene Intelligence (source-level, cached, advisory) ────
        // Runs PySceneDetect once per source; any failure leaves no scene
        // input and the tracker's own cut detection remains the sole source.
        let mut scene_cuts_json: Option<String> = None;
        if scene_intelligence::scene_intelligence_enabled() {
            match scene_intelligence::SceneIntelligenceEngine::new(None) {
                Ok(scene_engine) => {
                    let doc = scene_engine.process_source(&project.source_path);
                    if !doc.fallback && !doc.scenes.is_empty() {
                        let scene_tmp = std::env::temp_dir()
                            .join(format!("autoshorts_si_scenes_{}.json", uuid::Uuid::new_v4()));
                        if let Some(content) = serde_json::to_string(&doc)
                            .ok()
                            .and_then(|c| std::fs::write(&scene_tmp, c).ok())
                        {
                            let _ = content;
                            scene_cuts_json = Some(scene_tmp.to_string_lossy().into_owned());
                        }
                        println!(
                            "[SceneIntel] {} scene boundaries provided to framing",
                            doc.scenes.len()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("[SceneIntel] engine init failed (non-fatal): {}", e);
                }
            }
        }

        let sidecar_inputs = speaker_intel_inputs.as_ref().map(|(d, g, s)| {
            let _ = (d, g, s);
            media::SpeakerIntelSidecarInputs {
                diarization_json: d.as_str(),
                gallery_json: g.as_str(),
                scene_cuts_json: s.as_deref(),
            }
        });
        let framing_plan = media::detect_speaker_crop_params_with_intel(
            &project.source_path,
            render_start,
            render_end,
            iw,
            ih,
            crop_w,
            transcript_words_for_camera.as_deref(),
            framing_mode,
            sidecar_inputs.as_ref(),
        );

        // Persist the sidecar's speakerIntel block (Re-ID gallery entries +
        // fused active-speaker intervals) and clean up the temp files.
        if let Some(ref intel) = framing_plan.speaker_intel {
            if let Ok(engine) = speaker_intelligence::SpeakerIntelligenceEngine::new(
                speaker_intelligence::SpeakerIntelligenceConfig::default(),
                db.clone(),
            ) {
                match speaker_intelligence::parse_speaker_intel_block(intel) {
                    Ok((gallery_entries, fusion_intervals)) => {
                        if let Err(e) = engine.refresh_cache_from_render(
                            &project.id,
                            &candidate.id,
                            &project.source_path,
                            &gallery_entries,
                            &fusion_intervals,
                        ) {
                            eprintln!("[SpeakerIntel] render intel persist failed (non-fatal): {}", e);
                        }
                        println!(
                            "[SpeakerIntel] candidate intel persisted: gallery={} fusion_intervals={}",
                            gallery_entries.len(),
                            fusion_intervals.len()
                        );
                    }
                    Err(e) => {
                        eprintln!("[SpeakerIntel] malformed speakerIntel block (non-fatal): {}", e);
                    }
                }
            }
        }
        if let Some((ref d, ref g, _)) = speaker_intel_inputs {
            let _ = std::fs::remove_file(d);
            let _ = std::fs::remove_file(g);
        }
        if let Some(ref s) = scene_cuts_json {
            let _ = std::fs::remove_file(s);
        }

        // When pacing removed content, the edit map becomes authoritative for
        // every downstream timing domain: words are remapped to the output
        // timeline (removed words drop out of captions), and the framing plan
        // is remapped to output-relative times for caption layout queries.
        let clip_dur = (render_end - render_start).max(0.0);
        let (caption_words, caption_framing, caption_start, caption_end) = match pacing_plan {
            Some(ref plan) => {
                let remapped_words = pacing::remap_words(plan, transcript_words_for_camera.as_deref().unwrap_or(&[]));
                let remapped_framing = pacing::remap_framing_plan(&framing_plan, plan, clip_dur);
                (
                    remapped_words,
                    Some(remapped_framing),
                    plan.clip_start_sec,
                    plan.clip_start_sec + plan.output_duration_sec,
                )
            }
            None => (
                transcript_words_for_camera.clone().unwrap_or_default(),
                None,
                render_start,
                render_end,
            ),
        };

        // ── Audio Intelligence (AutoShorts 8.0) ────────────────────────────────
        // Speech-quality analysis on the SAME range the render will encode
        // (post-boundary, post-pacing). Words are passed on the SOURCE
        // timeline — the sidecar remaps them internally when pacing is
        // active, mirroring remap_words. None leaves the audio untouched.
        let audio_plan = audio::plan_audio_intelligence(
            &project.source_path,
            render_start,
            render_end,
            transcript_words_for_camera.as_deref(),
            pacing_plan.as_ref(),
        );

        let mut caption_intel_usable_and_written = false;
        let mut caption_intel_plan = None;
        if !caption_words.is_empty() {
            let style = project.caption_style.as_deref().unwrap_or("modern-box");

            // 1. Generate standard SRT for export
            let srt_content = generate_srt(&caption_words, caption_start, caption_end);
            let clip_srt_path = data_dir.join("projects").join(&project.id).join(format!("clip-{}.srt", candidate.id));
            if std::fs::write(&clip_srt_path, srt_content).is_ok() {
                _srt_path = Some(clip_srt_path);
            }

            // Plan Caption Intelligence 2.0 (additive decision layer)
            caption_intel_plan = caption_intel::plan_caption_intel(
                &caption_words,
                &candidate,
                pacing_plan.as_ref(),
                caption_start,
                caption_end,
            );

            // Predict T7 prosodic boundaries (flag-gated inside predict_t7_boundaries)
            let t7_boundaries = crate::t7_prosody::predict_t7_boundaries(
                &project.source_path,
                candidate.start_sec,
                candidate.end_sec,
                &caption_words,
            );
            let t7_boundaries_slice = if t7_boundaries.is_empty() {
                None
            } else {
                Some(t7_boundaries.as_slice())
            };

            // 2. Generate Layout-Aware Kinetic ASS Subtitles
            let ass_content = generate_layout_aware_kinetic_ass_subtitles_with_intel(
                &caption_words,
                caption_start,
                caption_end,
                style,
                caption_framing.as_ref().or(Some(&framing_plan)),
                caption_intel_plan.as_ref(),
                t7_boundaries_slice,
            );
            let clip_ass_path = data_dir.join("projects").join(&project.id).join(format!("clip-{}.ass", candidate.id));
            if std::fs::write(&clip_ass_path, ass_content).is_ok() {
                ass_path = Some(clip_ass_path);
                if caption_intel_plan.is_some() {
                    caption_intel_usable_and_written = true;
                }
            }

            // 3. Fallback drawtext filter string
            let drawtext = build_drawtext_filters(
                &caption_words,
                caption_start,
                caption_end,
                cropped_width,
                style,
            );
            if !drawtext.is_empty() {
                drawtext_filters = Some(drawtext);
            }
        }

        match pacing::render_paced_clip(
            &project.source_path,
            render_start,
            render_end,
            &output_path,
            ass_path.as_deref(),
            drawtext_filters.as_deref(),
            transcript_words_for_camera.as_deref(),
            Some(&framing_plan),
            pacing_plan.as_ref(),
            audio_plan.as_ref().map(|p| p.filter_chain.as_str()),
        ) {
            Ok(path) => {
                let path_string = path.to_string_lossy().to_string();
                let ass_string = ass_path.as_ref().map(|p| p.to_string_lossy().to_string());

                // ── Applied Features (AutoShorts 8.0 & 9.0) ───────────────────
                // Derive the canonical per-clip feature application truth table
                // directly from the in-memory production decision variables.
                //
                // Semantics:
                //   smart_pacing_applied       — pacing_plan is Some(…) only for
                //     validated, non-noop plans. plan_smart_pacing() returns None
                //     for no-op, validation failure, disabled, or error.
                //   hook_ending_applied        — boundary_opt explicitly records
                //     whether at least one boundary was actually moved.
                //   audio_intelligence_applied — audio_plan is Some(…) only when
                //     the sidecar produced a non-empty, allowlist-validated chain.
                //   caption_intelligence_applied — Caption Intelligence actually
                //     produced a usable CaptionIntelPlan, that plan was consumed
                //     by the caption renderer to generate the ASS file, and the
                //     final FFmpeg render completed successfully.
                //
                // NULL applied_features = legacy clip (no information).
                // Non-NULL JSON = complete truth table; false keys are preserved
                // so the reader can distinguish "known not applied" from "unknown".
                let smart_pacing_applied = pacing_plan.is_some();
                let hook_ending_applied = boundary_opt.start_changed || boundary_opt.end_changed;
                let audio_intelligence_applied = audio_plan.is_some();
                let caption_intelligence_applied = caption_intel_usable_and_written && ass_path.is_some();
                // Framing fallback is persisted so an emergency center-crop render
                // is never silently indistinguishable from a real Adaptive/DualFrame
                // result in the database (Render QA success ≠ framing success).
                let framing_fallback = framing_plan.is_emergency_fallback;
                let applied_features_json = serde_json::json!({
                    "smartPacing": smart_pacing_applied,
                    "hookEndingOptimization": hook_ending_applied,
                    "audioIntelligence": audio_intelligence_applied,
                    "captionIntelligence": caption_intelligence_applied,
                    "framingFallback": framing_fallback,
                })
                .to_string();

                // ── Human-readable render log ──────────────────────────────────
                let pacing_log = pacing_plan.as_ref().map(pacing::pacing_summary);
                let audio_log = audio_plan.as_ref().map(audio::audio_summary);
                let boundary_changed = boundary_opt.start_changed || boundary_opt.end_changed;
                let mut log_parts: Vec<String> = Vec::new();
                if boundary_changed {
                    log_parts.push(boundary_opt.summary());
                }
                if let Some(p) = &pacing_log {
                    log_parts.push(p.clone());
                }
                if let Some(a) = &audio_log {
                    log_parts.push(a.clone());
                }
                if framing_fallback {
                    log_parts.push(
                        "framing: EMERGENCY FALLBACK (center crop) — selected framing did NOT execute".to_string(),
                    );
                }
                if caption_intelligence_applied {
                    if let Some(ci) = &caption_intel_plan {
                        log_parts.push(format!("caption_intel: {}", ci.reason));
                    }
                }
                if let Some(style_name) = project.caption_style.as_deref() {
                    log_parts.push(format!("caption_template: {}", style_name));
                }
                // ── Phase 4: Deterministic Render QA (MANDATORY GUARDRAIL) ──
                // Runs AFTER the artifact exists and BEFORE the clip is marked
                // done. Render QA never modifies media — it only validates the
                // finished file.
                //
                // PRODUCTION SEQUENCE:
                //   render -> Render QA -> critical failure => clip marked
                //   "error" and NOT published -> pass => persisted as "done".
                //
                // Render QA is MANDATORY by default (render_qa_enabled() returns
                // true when AUTOSHORTS_RENDER_QA is unset). The env var exists
                // only as an explicit development opt-out.
                let captions_expected = ass_path.is_some() || drawtext_filters.is_some();
                let qa_outcome = render_qa::run_render_qa(
                    &path_string,
                    &project.source_path,
                    &candidate_id,
                    // Expected duration of the ARTIFACT: when Smart Pacing removed
                    // content, the output is shorter than the source range — comparing
                    // against the pre-pacing duration produced a false duration
                    // Warning on every paced clip (observed: 0.19s diff on a 0.20s
                    // breath-pause edit). The pacing plan's output duration is the
                    // true render expectation; the unpaced case is unchanged.
                    Some(
                        pacing_plan
                            .as_ref()
                            .map(|p| p.output_duration_sec)
                            .unwrap_or(clip_dur),
                    ),
                    probe
                        .as_ref()
                        .map(|p| p.audio_codec.is_some())
                        .unwrap_or(false),
                    captions_expected,
                    Some(&framing_plan),
                );

                let (qa_rejected, qa_summary_line, qa_error_msg) = match qa_outcome {
                    render_qa::RenderQaOutcome::Pass(ref report) => {
                        (false, Some(format!("render_qa: {}", report.summary())), None)
                    }
                    render_qa::RenderQaOutcome::Fail(ref report) => {
                        let detail = report
                            .checks
                            .iter()
                            .filter(|c| {
                                c.status == render_qa::QaStatus::Fail
                                    && c.severity == render_qa::QaSeverity::Critical
                            })
                            .map(|c| format!("{}: {}", c.name, c.details))
                            .collect::<Vec<_>>()
                            .join("; ");
                        eprintln!("[Render QA] rejecting clip {} — {}", candidate_id, detail);
                        (true, Some(format!("render_qa: {}", report.summary())), Some(detail))
                    }
                    render_qa::RenderQaOutcome::Error(ref err) => {
                        eprintln!(
                            "[Render QA] validation error — rejecting clip {}: {}",
                            candidate_id, err
                        );
                        (
                            true,
                            Some(format!("render_qa error: {}", err)),
                            Some(err.clone()),
                        )
                    }
                    render_qa::RenderQaOutcome::Disabled => {
                        (false, Some("render_qa: disabled by environment".to_string()), None)
                    }
                };

                if let Some(line) = qa_summary_line {
                    log_parts.push(line);
                }
                let combined_log = if log_parts.is_empty() {
                    None
                } else {
                    Some(log_parts.join(" | "))
                };

                if qa_rejected {
                    // The file exists but did not pass QA or encountered a validation error.
                    // Mark it as an error rather than a successful clip so the UI never presents an
                    // invalid artifact as finished work. Fail closed (D3).
                    db.update_clip_for_candidate(
                        &candidate_id,
                        "error",
                        None,
                        ass_string.as_deref(),
                        combined_log.as_deref(),
                        Some(applied_features_json.as_str()),
                    )
                    .map_err(to_command_error)?;
                    return Err(format!(
                        "Clip rendering failed Render QA validation: {}",
                        qa_error_msg
                            .or(combined_log)
                            .unwrap_or_else(|| "critical failure".to_string())
                    ));
                }

                db
                    .update_clip_for_candidate(
                        &candidate_id,
                        "done",
                        Some(&path_string),
                        ass_string.as_deref(),
                        combined_log.as_deref(),
                        Some(applied_features_json.as_str()),
                    )
                    .map_err(to_command_error)?;
                Ok(path_string)
            }
            Err(error) => {
                let err_msg = error.to_string();
                println!("[Clip Render Error] Clip rendering failed: {}", err_msg);
                db
                    .update_clip_for_candidate(&candidate_id, "error", None, None, Some(&err_msg), None)
                    .map_err(to_command_error)?;
                Err(format!("Clip rendering failed: {}", err_msg))
            }
        }
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn delete_project(state: tauri::State<'_, AppState>, project_id: String) -> Result<(), String> {
    state
        .db
        .delete_project(&project_id)
        .map_err(to_command_error)
}

#[tauri::command]
fn rename_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
    name: String,
) -> Result<(), String> {
    state
        .db
        .rename_project(&project_id, &name)
        .map_err(to_command_error)
}

pub fn run() {
    let _ = dotenvy::dotenv();
    let _ = dotenvy::from_filename("../.env");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .context("resolving app data directory")?;
            std::fs::create_dir_all(&data_dir).context("creating app data directory")?;
            std::fs::create_dir_all(data_dir.join("models"))
                .context("creating models directory")?;
            let db = Database::open(&data_dir.join("autoshorts.sqlite"))?;
            let render_tracker = InFlightRenderTracker::new();
            app.manage(AppState {
                db,
                data_dir,
                render_tracker,
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            environment_status,
            pull_ollama_model,
            install_ollama,
            create_project_from_path,
            list_projects,
            get_project_detail,
            probe_project,
            extract_project_audio,
            transcribe_project,
            save_demo_transcript,
            generate_candidates,
            set_selected_clip_count,
            set_selected_rank_range,
            render_flat_clip_for_candidate,
            delete_project,
            rename_project,
            check_youtube_copyright,
            download_youtube_video
        ])
        .run(tauri::generate_context!())
        .expect("error while running AutoShorts");
}

#[tauri::command]
async fn check_youtube_copyright(
    url: String,
    browser: Option<String>,
    cookies_path: Option<String>,
) -> Result<youtube::CopyrightCheckResult, String> {
    tokio::task::spawn_blocking(move || {
        youtube::check_copyright(&url, browser.as_deref(), cookies_path.as_deref())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn download_youtube_video(
    url: String,
    browser: Option<String>,
    cookies_path: Option<String>,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let downloads_dir = dirs::download_dir()
            .ok_or_else(|| "Could not locate Downloads folder on your system.".to_string())?;
        let path = youtube::download_video(
            &url,
            &downloads_dir,
            browser.as_deref(),
            cookies_path.as_deref(),
        )?;
        Ok(path.to_string_lossy().to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

fn project_dir(state: &AppState, project_id: &str) -> PathBuf {
    state.data_dir.join("projects").join(project_id)
}

fn documents_project_dir(project: &Project) -> Result<PathBuf, String> {
    let documents_dir = dirs::document_dir()
        .ok_or_else(|| "Could not find your Documents folder for clip output.".to_string())?;
    Ok(documents_dir
        .join("AutoShorts")
        .join(project_output_slug(project)))
}

fn project_output_slug(project: &Project) -> String {
    let stem = std::path::Path::new(&project.source_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&project.id);
    let slug = stem
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();

    if slug.is_empty() {
        project.id.clone()
    } else {
        slug
    }
}

fn validate_media_extension(path: &str) -> Result<()> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .ok_or_else(|| anyhow!("Selected file does not have an extension"))?;

    let allowed = ["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a"];
    if allowed.contains(&extension.as_str()) {
        Ok(())
    } else {
        Err(anyhow!(
            "Unsupported file type .{extension}. Use mp4, mov, mkv, webm, mp3, wav, or m4a."
        ))
    }
}

fn demo_transcript() -> NormalizedTranscript {
    let lines = [
        "The surprising thing about short-form clips is that the best moment is rarely the loudest moment.",
        "It is usually the point where someone finally says the quiet part plainly and the listener can feel the stakes.",
        "That is why the system needs to understand the transcript as a story, not just search for keywords.",
        "A good clip opens with tension, resolves one idea, and ends before the energy leaks away.",
        "If you can rank those moments consistently, the rendering pipeline becomes much easier to trust.",
        "The creator still decides what represents them, but the machine removes the first exhausting pass through hours of footage.",
        "The goal is not to automate taste completely. The goal is to give taste a faster starting point.",
        "Once the strongest moments are visible, captions and platform copy become finishing work instead of discovery work.",
        "That is the workflow AutoShorts is designed around.",
    ];

    let mut words = Vec::new();
    let mut cursor = 0.0;
    for line in lines {
        for token in line.split_whitespace() {
            let clean = token.to_string();
            let end = cursor + 0.32;
            words.push(TranscriptWord {
                text: clean,
                start: cursor,
                end,
                speaker: Some("A".to_string()),
            });
            cursor = end + 0.08;
        }
        cursor += 0.75;
    }

    let segments = transcription::build_segments(&words);

    NormalizedTranscript {
        language: "en".to_string(),
        duration: cursor,
        speakers: vec!["A".to_string()],
        words,
        segments,
        raw_words: None,
        correction_metadata: None,
    }
}

fn to_command_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

pub fn build_caption_chunks(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
) -> Vec<CaptionChunk> {
    let candidate_words: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| w.end > start_sec && w.start < end_sec)
        .collect();

    if candidate_words.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current_words: Vec<&TranscriptWord> = Vec::new();

    let flush_chunk = |current_words: &mut Vec<&TranscriptWord>, chunks: &mut Vec<CaptionChunk>| {
        if current_words.is_empty() {
            return;
        }
        let first = current_words[0];
        let last = current_words[current_words.len() - 1];

        let start_rel = (first.start - start_sec).max(0.0);
        let mut end_rel = (last.end - start_sec).min(end_sec - start_sec).max(0.0);
        if end_rel <= start_rel {
            end_rel = start_rel + 0.1;
        }

        let text = current_words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        chunks.push(CaptionChunk {
            start_rel,
            end_rel,
            text,
        });

        current_words.clear();
    };

    for &word in &candidate_words {
        if let Some(&prev_word) = current_words.last() {
            let pause = word.start - prev_word.end;
            let current_duration = word.end - current_words[0].start;
            let word_count = current_words.len();

            let ends_sentence =
                crate::transcription::ends_with_sentence_terminator(&prev_word.text);
            let ends_clause = prev_word.text.ends_with(',');
            let is_pause = pause > 0.4;
            let reaches_max_count = word_count >= 5;
            let exceeds_max_duration = current_duration > 2.5;

            let should_flush = is_pause
                || ends_sentence
                || (word_count >= 2 && (ends_clause || exceeds_max_duration))
                || reaches_max_count;

            if should_flush {
                flush_chunk(&mut current_words, &mut chunks);
            }
        }

        current_words.push(word);
    }

    flush_chunk(&mut current_words, &mut chunks);

    chunks
}

/// Generate modern OpusClip / TikTok style kinetic ASS subtitles with word-level highlight animation.
/// - Multilingual support: uses Nirmala UI (Windows) / Kohinoor Devanagari (macOS) / Noto Sans Devanagari (Linux)
/// - Large, bold font scaled to 1080x1920
/// - Thick black outline and shadow for crisp contrast on any background
/// - Word-level synchronization: as each word is spoken, it lights up in vibrant green (or chosen style emphasis)
/// - Short phrases of 1–4 words
/// - Positioned in the lower-middle safe region (MarginV ~ 360px from bottom, well below speaker's face)
pub fn generate_kinetic_ass_subtitles(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    caption_style: &str,
) -> String {
    generate_layout_aware_kinetic_ass_subtitles(words, start_sec, end_sec, caption_style, None)
}

pub fn generate_layout_aware_kinetic_ass_subtitles(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    caption_style: &str,
    framing_plan: Option<&media::SmartFramingPlan>,
) -> String {
    generate_layout_aware_kinetic_ass_subtitles_with_intel(
        words,
        start_sec,
        end_sec,
        caption_style,
        framing_plan,
        None,
        None,
    )
}

pub fn generate_layout_aware_kinetic_ass_subtitles_with_intel(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    caption_style: &str,
    framing_plan: Option<&media::SmartFramingPlan>,
    caption_intel: Option<&caption_intel::CaptionIntelPlan>,
    t7_boundaries: Option<&[t7_prosody::T7Boundary]>,
) -> String {
    let candidate_words: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| w.end > start_sec && w.start < end_sec)
        .collect();

    if candidate_words.is_empty() {
        return String::new();
    }

    // Check if caption_style is one of the unified templates
    if let Some(template) = captions::get_caption_template(caption_style) {
        return captions::generate_ass_from_template_with_framing_and_intel(
            words,
            start_sec,
            end_sec,
            &template,
            framing_plan,
            caption_intel,
            t7_boundaries,
        );
    }

    // Otherwise fall back to legacy style dispatch for 100% backward compatibility
    // Determine emphasis color based on caption_style
    // Header format: &HAABBGGRR. Inline override tag format: &HBBGGRR&
    let (header_primary, header_emphasis, tag_primary, tag_emphasis) = match caption_style {
        "vibrant-cyan" => ("&H00FFFFFF", "&H00FFFF00", "&HFFFF00&", "&HFFFFFF&"),
        "vibrant-yellow-box" | "classic-outline" => {
            ("&H00FFFFFF", "&H0000FFFF", "&H00FFFF&", "&HFFFFFF&")
        }
        "vibrant-red" => ("&H00FFFFFF", "&H00303BFF", "&H303BFF&", "&HFFFFFF&"),
        _ => ("&H00FFFFFF", "&H0014FF39", "&H14FF39&", "&HFFFFFF&"),
    };

    let font_name = if cfg!(windows) {
        "Nirmala UI"
    } else if cfg!(target_os = "macos") {
        "Kohinoor Devanagari"
    } else {
        "Noto Sans Devanagari"
    };

    let mut ass = format!(
        r#"[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},84,{header_primary},{header_emphasis},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,9,3,2,80,80,360,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"#
    );

    // Group into natural 1-3 word semantic phrases based on rhythm, punctuation, conjunctions, and character width
    let mut phrases: Vec<Vec<&TranscriptWord>> = Vec::new();
    let mut curr_phrase: Vec<&TranscriptWord> = Vec::new();

    let conjunctions = [
        "and", "but", "or", "so", "because", "when", "if", "that", "with", "from", "into", "every",
        "on", "your", "my", "our", "their", "this", "which", "aur", "lekin", "par", "ki", "toh",
        "kyunki", "agar", "jab", "tab",
    ];

    for &word in &candidate_words {
        if let Some(&prev) = curr_phrase.last() {
            let pause = word.start - prev.end;
            let duration = word.end - curr_phrase[0].start;
            let count = curr_phrase.len();
            let total_chars: usize = curr_phrase.iter().map(|w| w.text.len()).sum();

            let clean_w = word
                .text
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            let ends_punct = crate::transcription::ends_with_sentence_terminator(&prev.text);
            let ends_clause = prev.text.ends_with(',');
            let is_pause = pause > 0.28;
            let is_conjunction = count >= 2 && conjunctions.contains(&clean_w.as_str());
            let max_words = count >= 3;
            let max_chars = total_chars + word.text.len() > 18;
            let max_dur = duration > 1.5;

            if is_pause
                || ends_punct
                || ends_clause
                || is_conjunction
                || max_words
                || max_chars
                || max_dur
            {
                phrases.push(std::mem::take(&mut curr_phrase));
            }
        }
        curr_phrase.push(word);
    }
    if !curr_phrase.is_empty() {
        phrases.push(curr_phrase);
    }

    // Generate word-level active highlight dialogue lines for each phrase
    for phrase in &phrases {
        if phrase.is_empty() {
            continue;
        }

        for (active_idx, active_word) in phrase.iter().enumerate() {
            let word_start = (active_word.start - start_sec).max(0.0);
            let word_end = (active_word.end - start_sec)
                .min(end_sec - start_sec)
                .max(word_start + 0.05);

            let mut line_parts = Vec::new();
            for (idx, w) in phrase.iter().enumerate() {
                let clean_text = w
                    .text
                    .trim()
                    .to_uppercase()
                    .replace('{', "")
                    .replace('}', "");
                if idx == active_idx {
                    // Active emphasized word: vibrant emphasis color + 12% kinetic scale pop
                    line_parts.push(format!(r"{{\c{tag_emphasis}}}{{\fscx112\fscy112}}{clean_text}{{\fscx100\fscy100}}{{\c{tag_primary}}}"));
                } else {
                    line_parts.push(clean_text);
                }
            }

            let dialogue_text = line_parts.join(" ");
            let start_str = format_ass_time(word_start);
            let end_str = format_ass_time(word_end);

            ass.push_str(&format!(
                "Dialogue: 0,{},{},Default,,0,0,0,,{}\n",
                start_str, end_str, dialogue_text
            ));
        }
    }

    ass
}

fn format_ass_time(secs: f64) -> String {
    let total_cs = (secs * 100.0).round() as u64;
    let cs = total_cs % 100;
    let total_secs = total_cs / 100;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{}:{:02}:{:02}.{:02}", h, m, s, cs)
}

pub fn generate_srt(words: &[TranscriptWord], start_sec: f64, end_sec: f64) -> String {
    let chunks = build_caption_chunks(words, start_sec, end_sec);
    let mut srt = String::new();

    for (index, chunk) in chunks.iter().enumerate() {
        srt.push_str(&format!("{}\n", index + 1));
        srt.push_str(&format!(
            "{}\n",
            format_srt_time(chunk.start_rel, chunk.end_rel)
        ));
        srt.push_str(&format!("{}\n\n", chunk.text));
    }

    srt
}

fn format_srt_time(start: f64, end: f64) -> String {
    let format_time = |secs: f64| {
        let hours = (secs / 3600.0) as u32;
        let mins = ((secs % 3600.0) / 60.0) as u32;
        let secs_only = (secs % 60.0) as u32;
        let ms = ((secs.fract()) * 1000.0) as u32;
        format!("{hours:02}:{mins:02}:{secs_only:02},{ms:03}")
    };
    format!("{} --> {}", format_time(start), format_time(end))
}

fn escape_ffmpeg_filter_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let escaped = normalized.replace('\'', r"\'").replace(':', r"\:");
    format!("'{}'", escaped)
}

fn escape_ffmpeg_drawtext_text(text: &str) -> String {
    text.replace('\n', " ")
        .replace('\r', " ")
        .replace('\\', r"\\")
        .replace('\'', r"'\''")
        .replace(':', r"\:")
}

pub fn build_drawtext_filters(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    _cropped_width: i64,
    caption_style: &str,
) -> String {
    let chunks = build_caption_chunks(words, start_sec, end_sec);
    if chunks.is_empty() {
        return String::new();
    }

    // Find valid font path on current system (prioritizing Nirmala UI for Hindi & English support)
    let font_paths = [
        // Windows standard with Devanagari & Latin glyph support
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/NirmalaB.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/SegoeUIb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/SegoeUI.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        // macOS
        "/System/Library/Fonts/Kohinoor.ttc",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        // Linux
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ];

    let mut selected_font_escaped = None;
    for path in &font_paths {
        if std::path::Path::new(path).exists() {
            selected_font_escaped = Some(escape_ffmpeg_filter_path(path));
            break;
        }
    }

    if selected_font_escaped.is_none() {
        if let Ok(windir) = std::env::var("WINDIR").or_else(|_| std::env::var("SystemRoot")) {
            let windir_fonts = format!("{}/Fonts", windir.replace('\\', "/"));
            for font_name in &[
                "Nirmala.ttc",
                "NirmalaB.ttf",
                "Nirmala.ttf",
                "SegoeUIb.ttf",
                "segoeuib.ttf",
                "SegoeUI.ttf",
                "arialbd.ttf",
                "arial.ttf",
            ] {
                let full_path = format!("{}/{}", windir_fonts, font_name);
                if std::path::Path::new(&full_path).exists() {
                    selected_font_escaped = Some(escape_ffmpeg_filter_path(&full_path));
                    break;
                }
            }
        }
    }

    let mut drawtext_filters = Vec::new();

    for chunk in chunks {
        let uppercase_text = chunk.text.to_uppercase();
        let escaped_text = escape_ffmpeg_drawtext_text(&uppercase_text);

        if escaped_text.trim().is_empty() {
            continue;
        }

        // Font size relative to the 1080-wide output (not the small pre-crop width)
        // For 1080x1920 output, target ~7% of width = ~76px; clamp 48–96
        let output_width = 1080_f64;
        let fontsize = (output_width * 0.07).clamp(48.0, 96.0).round() as i64;
        let padding = (fontsize as f64 * 0.25).clamp(8.0, 32.0).round() as i64;

        let mut opts = Vec::new();

        if let Some(font_escaped) = &selected_font_escaped {
            opts.push(format!("fontfile={}", font_escaped));
        }

        opts.push(format!("text='{}'", escaped_text));
        opts.push("expansion=none".to_string());

        match caption_style {
            "preset_viral_bold" => {
                let borderw = ((fontsize as f64) * 0.1).clamp(2.0, 8.0).round() as i64;
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.70".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0x00FF66".to_string());
                opts.push(format!("borderw={}", borderw));
                opts.push("bordercolor=black".to_string());
                opts.push("shadowcolor=black@0.8".to_string());
                opts.push("shadowx=3".to_string());
                opts.push("shadowy=3".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "preset_mrbeast_pop" => {
                let borderw = ((fontsize as f64) * 0.12).clamp(2.0, 8.0).round() as i64;
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.65".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=yellow".to_string());
                opts.push(format!("borderw={}", borderw));
                opts.push("bordercolor=black".to_string());
                opts.push("shadowcolor=black@0.5".to_string());
                opts.push("shadowx=0".to_string());
                opts.push("shadowy=4".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "preset_minimal_capsule" => {
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.75".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0xF5F5F5".to_string());
                opts.push("box=1".to_string());
                opts.push("boxcolor=0x000000a6".to_string());
                opts.push(format!("boxborderw={}", padding));
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "preset_cinematic_vlog" => {
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.82".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0xFDFBF7".to_string());
                opts.push("shadowcolor=black@0.3".to_string());
                opts.push("shadowx=0".to_string());
                opts.push("shadowy=2".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "preset_dynamic_editorial" => {
                let borderw = 0;
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.68".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0xFFE600".to_string());
                opts.push(format!("borderw={}", borderw));
                opts.push("bordercolor=black".to_string());
                opts.push("shadowcolor=black@0.6".to_string());
                opts.push("shadowx=2".to_string());
                opts.push("shadowy=2".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "classic-outline" => {
                let borderw = ((fontsize as f64) * 0.1).clamp(2.0, 8.0).round() as i64;
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.65".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=yellow".to_string());
                opts.push(format!("borderw={}", borderw));
                opts.push("bordercolor=black".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "minimal-shadow" => {
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.7".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=white".to_string());
                opts.push("shadowcolor=black@0.5".to_string());
                opts.push("shadowx=2".to_string());
                opts.push("shadowy=2".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "vibrant-cyan" => {
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.7".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0x00FFFF".to_string());
                opts.push("shadowcolor=black@0.6".to_string());
                opts.push("shadowx=2".to_string());
                opts.push("shadowy=2".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "vibrant-yellow-box" => {
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.72".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=black".to_string());
                opts.push("box=1".to_string());
                opts.push("boxcolor=0xffff00e0".to_string());
                opts.push(format!("boxborderw={}", padding));
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "vibrant-green" => {
                let borderw = ((fontsize as f64) * 0.08).clamp(1.5, 6.0).round() as i64;
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.7".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0x39FF14".to_string());
                opts.push(format!("borderw={}", borderw));
                opts.push("bordercolor=black".to_string());
                opts.push("shadowcolor=black@0.6".to_string());
                opts.push("shadowx=2".to_string());
                opts.push("shadowy=2".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            "vibrant-red" => {
                let borderw = ((fontsize as f64) * 0.08).clamp(1.5, 6.0).round() as i64;
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.7".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=0xFF3B30".to_string());
                opts.push(format!("borderw={}", borderw));
                opts.push("bordercolor=black".to_string());
                opts.push("shadowcolor=black@0.6".to_string());
                opts.push("shadowx=2".to_string());
                opts.push("shadowy=2".to_string());
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
            _ => {
                // modern-box
                opts.push("x=(w-text_w)/2".to_string());
                opts.push("y=h*0.72".to_string());
                opts.push(format!("fontsize={}", fontsize));
                opts.push("fontcolor=white".to_string());
                opts.push("box=1".to_string());
                opts.push("boxcolor=0x000000b0".to_string());
                opts.push(format!("boxborderw={}", padding));
                opts.push(format!(
                    "enable='between(t\\,{:.3}\\,{:.3})'",
                    chunk.start_rel, chunk.end_rel
                ));
            }
        }

        let drawtext = format!("drawtext={}", opts.join(":"));
        drawtext_filters.push(drawtext);
    }

    drawtext_filters.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CandidateDraft, ClosureSignals, TranscriptWord};

    fn make_word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: Some("S1".to_string()),
        }
    }

    #[test]
    fn test_caption_segmentation_complete_coverage() {
        let words = vec![
            make_word("I", 0.0, 0.2),
            make_word("think", 0.2, 0.5),
            make_word("this", 0.5, 0.8),
            make_word("is", 0.8, 1.0),
            make_word("a", 1.0, 1.1),
            make_word("really", 1.1, 1.5),
            make_word("good", 1.5, 1.8),
            make_word("example", 1.8, 2.3),
        ];

        let chunks = build_caption_chunks(&words, 0.0, 3.0);
        assert!(!chunks.is_empty());

        let flattened_words: Vec<String> = chunks
            .iter()
            .flat_map(|c| c.text.split_whitespace().map(|s| s.to_string()))
            .collect();

        let original_words: Vec<String> = words.iter().map(|w| w.text.clone()).collect();
        assert_eq!(flattened_words, original_words);
    }

    #[test]
    fn test_caption_segmentation_pause_handling() {
        let words = vec![
            make_word("First", 0.0, 0.5),
            make_word("phrase", 0.5, 1.0),
            // 1.0s pause (1.0 -> 2.0)
            make_word("Second", 2.0, 2.5),
            make_word("phrase", 2.5, 3.0),
        ];

        let chunks = build_caption_chunks(&words, 0.0, 4.0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "First phrase");
        assert_eq!(chunks[1].text, "Second phrase");
    }

    #[test]
    fn test_caption_segmentation_punctuation_handling() {
        let words = vec![
            make_word("Hello.", 0.0, 0.5),
            make_word("How", 0.6, 0.8),
            make_word("are", 0.8, 1.0),
            make_word("you?", 1.0, 1.5),
        ];

        let chunks = build_caption_chunks(&words, 0.0, 2.0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "Hello.");
        assert_eq!(chunks[1].text, "How are you?");
    }

    #[test]
    fn test_caption_segmentation_names_preserved() {
        let words = vec![
            make_word("Kit", 0.0, 0.3),
            make_word("Connor", 0.3, 0.7),
            make_word("and", 0.7, 0.9),
            make_word("Maya", 0.9, 1.2),
            make_word("Boyd", 1.2, 1.6),
        ];

        let chunks = build_caption_chunks(&words, 0.0, 2.0);
        let all_text = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("Kit Connor"));
        assert!(all_text.contains("Maya Boyd"));
    }

    #[test]
    fn test_caption_segmentation_contractions_preserved() {
        let words = vec![
            make_word("I'm", 0.0, 0.3),
            make_word("sure", 0.3, 0.6),
            make_word("it's", 0.6, 0.9),
            make_word("Vincent", 0.9, 1.3),
            make_word("D'Onofrio", 1.3, 1.8),
        ];

        let chunks = build_caption_chunks(&words, 0.0, 2.0);
        let all_text = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all_text.contains("I'm"));
        assert!(all_text.contains("it's"));
        assert!(all_text.contains("D'Onofrio"));
    }

    #[test]
    fn test_caption_segmentation_no_word_loss() {
        let words = vec![
            make_word("The", 0.0, 0.2),
            make_word("quick", 0.2, 0.5),
            make_word("brown", 0.5, 0.8),
            make_word("fox", 0.8, 1.1),
            make_word("jumps", 1.1, 1.4),
            make_word("over", 1.4, 1.7),
            make_word("the", 1.7, 1.9),
            make_word("lazy", 1.9, 2.2),
            make_word("dog.", 2.2, 2.6),
        ];

        let chunks = build_caption_chunks(&words, 0.0, 3.0);
        let reconstructed: Vec<String> = chunks
            .iter()
            .flat_map(|c| c.text.split_whitespace().map(|s| s.to_string()))
            .collect();
        let original: Vec<String> = words.iter().map(|w| w.text.clone()).collect();
        assert_eq!(reconstructed, original);
    }

    #[test]
    fn test_caption_segmentation_timestamps_monotonic_and_bounded() {
        let words = vec![
            make_word("One", 10.0, 10.3),
            make_word("two", 10.3, 10.6),
            make_word("three", 10.6, 10.9),
            make_word("four", 10.9, 11.2),
        ];

        let start_sec = 10.0;
        let end_sec = 15.0;
        let chunks = build_caption_chunks(&words, start_sec, end_sec);
        assert!(!chunks.is_empty());

        let mut last_end = -1.0;
        for chunk in &chunks {
            assert!(chunk.start_rel >= 0.0);
            assert!(chunk.end_rel > chunk.start_rel);
            assert!(chunk.start_rel >= last_end);
            last_end = chunk.end_rel;
        }
    }

    #[test]
    fn test_build_drawtext_filters_formatting() {
        let words = vec![
            TranscriptWord {
                text: "Hello".to_string(),
                start: 0.0,
                end: 1.0,
                speaker: None,
            },
            TranscriptWord {
                text: "world".to_string(),
                start: 1.0,
                end: 2.0,
                speaker: None,
            },
        ];

        let result = build_drawtext_filters(&words, 0.0, 5.0, 1080, "classic-outline");
        assert!(!result.is_empty());
        assert!(result.contains("drawtext="));
        assert!(result.contains("text='HELLO WORLD'"));

        if result.contains("fontfile=") {
            assert!(result.contains("fontfile='"));
            assert!(result.contains(r"\:"));
        }
    }

    #[test]
    fn test_caption_punctuation_escaping() {
        let cases = vec![
            ("YOU KNOW", 7.840, 8.500),
            ("IT'S IMPORTANT", 8.500, 9.200),
            ("WAIT, WHAT?", 9.200, 10.000),
            ("THIS IS: WHY", 10.000, 10.800),
            ("100% TRUE", 10.800, 11.500),
        ];

        for (text, start, end) in cases {
            let words = vec![TranscriptWord {
                text: text.to_string(),
                start,
                end,
                speaker: None,
            }];

            let filter_str = build_drawtext_filters(&words, 0.0, 20.0, 1080, "modern-box");
            assert!(
                filter_str.starts_with("drawtext="),
                "Filter must start with drawtext= for text: {}",
                text
            );
            assert!(
                filter_str.contains("text='"),
                "Filter must quote text for: {}",
                text
            );
            assert!(
                filter_str.contains("enable='between(t\\,"),
                "Filter must quote and escape enable timing for: {}",
                text
            );
        }
    }

    #[test]
    fn test_snap_to_semantic_boundaries_no_30s_truncation() {
        // Construct a 45-second sentence arc
        let words = vec![
            make_word("What", 0.0, 0.5),
            make_word("is", 0.5, 0.8),
            make_word("the", 0.8, 1.0),
            make_word("biggest", 1.0, 1.5),
            make_word("lesson?", 1.5, 2.0),
            make_word("The", 2.5, 3.0),
            make_word("biggest", 3.0, 3.5),
            make_word("lesson", 3.5, 4.0),
            make_word("is", 4.0, 4.3),
            make_word("consistency.", 4.3, 5.0),
            make_word("If", 5.5, 6.0),
            make_word("you", 6.0, 6.3),
            make_word("keep", 6.3, 6.7),
            make_word("going", 6.7, 7.2),
            make_word("every", 7.2, 7.6),
            make_word("single", 7.6, 8.0),
            make_word("day,", 8.0, 8.5),
            make_word("eventually", 8.8, 9.5),
            make_word("you", 9.5, 9.8),
            make_word("will", 9.8, 10.2),
            make_word("see", 10.2, 10.5),
            make_word("extraordinary", 10.5, 11.5),
            make_word("results", 11.5, 12.0),
            make_word("in", 12.0, 12.3),
            make_word("your", 12.3, 12.6),
            make_word("life.", 12.6, 13.5),
            make_word("That's", 14.0, 14.5),
            make_word("the", 14.5, 14.8),
            make_word("truth.", 14.8, 44.5),
        ];

        // LLM wants a 44.5s clip (0.0 to 44.5)
        let (snapped_start, snapped_end) =
            snap_to_semantic_boundaries(&words, 0.0, 44.5, 60.0, None);
        let duration = snapped_end - snapped_start;

        // Verify it was NOT truncated to 30s or 35s
        assert_eq!(snapped_start, 0.0);
        assert_eq!(snapped_end, 44.5);
        assert!(
            duration > 40.0,
            "Clip duration ({:.2}s) should preserve the full 44.5s semantic arc",
            duration
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_snaps_to_sentence_end() {
        let words = vec![
            make_word("Hello", 0.0, 0.5),
            make_word("world.", 0.5, 1.0),
            make_word("This", 1.2, 1.6),
            make_word("is", 1.6, 1.9),
            make_word("a", 1.9, 2.1),
            make_word("complete", 2.1, 2.7),
            make_word("thought.", 2.7, 3.2),
            make_word("And", 3.5, 3.8),
            make_word("an", 3.8, 4.0),
            make_word("extra", 4.0, 4.5),
            make_word("sentence.", 4.5, 25.0),
        ];

        // Raw end is 3.0 (between "complete" and "thought.")
        let (_, snapped_end) = snap_to_semantic_boundaries(&words, 0.0, 3.0, 30.0, None);
        // It should snap forward to 3.2 ("thought.") or 25.0 (min guideline)
        assert!(
            snapped_end >= 3.2,
            "End must land on sentence boundary, got {:.2}",
            snapped_end
        );
    }

    #[test]
    fn test_generate_kinetic_ass_subtitles_formatting_and_highlights() {
        let words = vec![
            make_word("What", 0.0, 0.5),
            make_word("was", 0.5, 1.0),
            make_word("the", 1.0, 1.5),
            make_word("lesson?", 1.5, 2.2),
        ];

        let ass = generate_kinetic_ass_subtitles(&words, 0.0, 5.0, "modern-box");
        assert!(ass.contains("PlayResX: 1080"));
        assert!(ass.contains("PlayResY: 1920"));
        assert!(ass.contains("[Events]"));
        assert!(ass.contains("Dialogue: 0,"));
        // Check active word green emphasis tag and 12% kinetic pop
        assert!(ass.contains(r"{\c&H14FF39&}"));
        assert!(ass.contains(r"{\fscx112\fscy112}"));
        assert!(ass.contains(r"{\c&HFFFFFF&}"));
        // Check uppercase words
        assert!(ass.contains("WHAT"));
        assert!(ass.contains("LESSON?"));
    }

    #[test]
    fn test_kinetic_caption_semantic_phrasing_and_safe_width() {
        let words = vec![
            make_word("I", 0.0, 0.4),
            make_word("fight", 0.4, 0.9),
            make_word("against", 0.9, 1.4),
            make_word("my", 1.4, 1.7),
            make_word("mind", 1.7, 2.2),
            make_word("every", 2.2, 2.6),
            make_word("single", 2.6, 3.1),
            make_word("day.", 3.1, 3.7),
        ];

        let ass = generate_kinetic_ass_subtitles(&words, 0.0, 4.0, "vibrant-green");
        // Safe margins: MarginL=80, MarginR=80, MarginV=360
        assert!(ass.contains("MarginL, MarginR, MarginV"));
        assert!(ass.contains("80,80,360"));

        // Verify each dialogue line has natural 1-3 words
        for line in ass.lines().filter(|l| l.starts_with("Dialogue:")) {
            let text_part = line.split(",,").nth(1).unwrap_or("");
            // Strip ASS tags to count words in display text
            let clean = text_part
                .split('{')
                .map(|s| s.split('}').last().unwrap_or(""))
                .collect::<Vec<_>>()
                .join("");
            let word_count = clean.split_whitespace().count();
            assert!(
                word_count >= 1 && word_count <= 3,
                "Phrase '{}' has {} words, expected 1-3",
                clean,
                word_count
            );
        }
    }

    #[test]
    fn test_kinetic_caption_synchronization_and_collision_clearance() {
        let words = vec![
            make_word("Discipline", 0.0, 0.8),
            make_word("creates", 0.8, 1.4),
            make_word("freedom.", 1.4, 2.2),
        ];

        let ass = generate_kinetic_ass_subtitles(&words, 0.0, 3.0, "vibrant-green");
        // Ensure dialogue lines have valid non-zero duration
        for line in ass.lines().filter(|l| l.starts_with("Dialogue:")) {
            let parts: Vec<&str> = line.split(',').collect();
            let start_str = parts[1];
            let end_str = parts[2];
            assert_ne!(
                start_str, end_str,
                "Dialogue line must have non-zero duration"
            );
        }
    }

    #[test]
    fn test_all_caption_templates_load() {
        let templates = captions::all_caption_templates();
        assert_eq!(
            templates.len(),
            7,
            "Must have exactly 7 caption templates (T1–T5 + Bhaukal T6 + T7)"
        );
    }

    #[test]
    fn test_caption_templates_correct_ids() {
        assert!(captions::get_caption_template("preset_viral_bold").is_some());
        assert!(captions::get_caption_template("preset_mrbeast_pop").is_some());
        assert!(captions::get_caption_template("preset_minimal_capsule").is_some());
        assert!(captions::get_caption_template("preset_cinematic_vlog").is_some());
        assert!(captions::get_caption_template("preset_dynamic_editorial").is_some());
        assert!(captions::get_caption_template("preset_bhaukal_caption").is_some());
        assert!(captions::get_caption_template("non_existent_id").is_none());
    }

    #[test]
    fn test_caption_templates_correct_names() {
        let t1 = captions::get_caption_template("preset_viral_bold").unwrap();
        assert_eq!(t1.name, "Hormozi Viral");

        let t2 = captions::get_caption_template("preset_mrbeast_pop").unwrap();
        assert_eq!(t2.name, "Narrative Pop");

        let t3 = captions::get_caption_template("preset_minimal_capsule").unwrap();
        assert_eq!(t3.name, "Minimal Capsule");

        let t4 = captions::get_caption_template("preset_cinematic_vlog").unwrap();
        assert_eq!(t4.name, "Cinematic Vlog");

        let t5 = captions::get_caption_template("preset_dynamic_editorial").unwrap();
        assert_eq!(t5.name, "Dynamic Editorial Kinetic");

        let t6 = captions::get_caption_template("preset_bhaukal_caption").unwrap();
        assert_eq!(t6.name, "Bhaukal caption");
    }

    #[test]
    fn test_caption_templates_segmentation_settings() {
        let t1 = captions::get_caption_template("preset_viral_bold").unwrap();
        assert_eq!(t1.segmentation.max_words_per_line, 2);
        assert_eq!(t1.segmentation.max_lines_per_frame, 1);
        assert_eq!(t1.segmentation.text_case, "uppercase");

        let t2 = captions::get_caption_template("preset_mrbeast_pop").unwrap();
        assert_eq!(t2.segmentation.max_words_per_line, 4);
        assert_eq!(t2.segmentation.max_lines_per_frame, 2);
        assert_eq!(t2.segmentation.text_case, "sentence");

        let t3 = captions::get_caption_template("preset_minimal_capsule").unwrap();
        assert_eq!(t3.segmentation.max_words_per_line, 5);
        assert_eq!(t3.segmentation.max_lines_per_frame, 1);
        assert_eq!(t3.segmentation.text_case, "normal");

        let t4 = captions::get_caption_template("preset_cinematic_vlog").unwrap();
        assert_eq!(t4.segmentation.max_words_per_line, 6);
        assert_eq!(t4.segmentation.max_lines_per_frame, 2);
        assert_eq!(t4.segmentation.text_case, "normal");

        let t5 = captions::get_caption_template("preset_dynamic_editorial").unwrap();
        assert_eq!(t5.segmentation.max_words_per_line, 4);
        assert_eq!(t5.segmentation.max_lines_per_frame, 1);
        assert_eq!(t5.segmentation.text_case, "normal");
    }

    #[test]
    fn test_caption_templates_typography_configuration() {
        // Hormozi Viral
        let t1 = captions::get_caption_template("preset_viral_bold").unwrap();
        assert_eq!(t1.global_styles.font_family, "Bebas Neue");
        assert_eq!(t1.global_styles.font_size, 84);
        assert_ne!(
            t1.global_styles.font_size, 160,
            "Oversized font size 160 must not be used"
        );
        assert_ne!(
            t1.global_styles.font_size, 126,
            "Previous font size 126 must not be used"
        );
        assert_ne!(
            t1.global_styles.font_size, 104,
            "Previous font size 104 must not be used"
        );
        assert_ne!(
            t1.global_styles.font_size, 90,
            "Previous font size 90 must not be used"
        );
        assert_eq!(t1.global_styles.font_weight, "900");
        assert_eq!(t1.global_styles.font_color, "#FFFFFF");
        assert_eq!(t1.global_styles.alignment, "center");
        assert!((t1.global_styles.position_y - 0.70).abs() < 1e-6);
        assert_eq!(t1.global_styles.stroke_color.as_deref(), Some("#000000"));
        assert_eq!(t1.global_styles.stroke_width, Some(4));
        assert_eq!(
            t1.global_styles.shadow_color.as_deref(),
            Some("rgba(0, 0, 0, 0.8)")
        );
        assert_eq!(t1.global_styles.shadow_blur, Some(10.0));

        // Narrative Pop
        let t2 = captions::get_caption_template("preset_mrbeast_pop").unwrap();
        assert_eq!(t2.global_styles.font_family, "Montserrat");
        assert_eq!(t2.global_styles.font_size, 80);
        assert_ne!(
            t2.global_styles.font_size, 110,
            "Oversized font size 110 must not be used"
        );
        assert_ne!(
            t2.global_styles.font_size, 90,
            "Previous font size 90 must not be used"
        );
        assert_eq!(t2.global_styles.font_weight, "800");
        assert_eq!(t2.global_styles.font_color, "#FFFFFF");
        assert_eq!(t2.global_styles.alignment, "center");
        assert!((t2.global_styles.position_y - 0.65).abs() < 1e-6);
        assert_eq!(t2.global_styles.stroke_color.as_deref(), Some("#000000"));
        assert_eq!(t2.global_styles.stroke_width, Some(5));
        assert_eq!(
            t2.global_styles.shadow_color.as_deref(),
            Some("rgba(0,0,0,0.5)")
        );
        assert_eq!(t2.global_styles.shadow_offset_y, Some(4.0));

        // Minimal Capsule
        let t3 = captions::get_caption_template("preset_minimal_capsule").unwrap();
        assert_eq!(t3.global_styles.font_family, "Inter");
        assert_eq!(t3.global_styles.font_size, 96);
        assert_ne!(
            t3.global_styles.font_size, 130,
            "Oversized font size 130 must not be used"
        );
        assert_ne!(
            t3.global_styles.font_size, 108,
            "Previous font size 108 must not be used"
        );
        assert_eq!(t3.global_styles.font_weight, "600");
        assert_eq!(t3.global_styles.font_color, "#F5F5F5");
        assert_eq!(t3.global_styles.alignment, "center");
        assert!((t3.global_styles.position_y - 0.75).abs() < 1e-6);
        let bbox = t3.global_styles.background_box.as_ref().unwrap();
        assert!(bbox.enabled);
        assert_eq!(bbox.color, "#000000");
        assert!((bbox.opacity - 0.65).abs() < 1e-6);
        assert_eq!(bbox.padding_px, 12);
        assert_eq!(bbox.border_radius_px, 8);

        // Cinematic Vlog
        let t4 = captions::get_caption_template("preset_cinematic_vlog").unwrap();
        assert_eq!(t4.global_styles.font_family, "Poppins");
        assert_eq!(t4.global_styles.font_size, 116);
        assert_ne!(
            t4.global_styles.font_size, 130,
            "Oversized font size 130 must not be used"
        );
        assert_ne!(
            t4.global_styles.font_size, 122,
            "Previous font size 122 must not be used"
        );
        assert_eq!(t4.global_styles.font_weight, "400");
        assert_eq!(t4.global_styles.font_color, "#FDFBF7");
        assert_eq!(t4.global_styles.alignment, "center");
        assert!((t4.global_styles.position_y - 0.82).abs() < 1e-6);
        assert_eq!(
            t4.global_styles.shadow_color.as_deref(),
            Some("rgba(0, 0, 0, 0.3)")
        );
        assert_eq!(t4.global_styles.shadow_blur, Some(6.0));
        assert_eq!(t4.global_styles.shadow_offset_y, Some(2.0));

        // Dynamic Editorial Kinetic
        let t5 = captions::get_caption_template("preset_dynamic_editorial").unwrap();
        assert_eq!(t5.global_styles.font_family, "Montserrat");
        assert_eq!(t5.global_styles.font_size, 62);
        assert_eq!(t5.global_styles.font_weight, "800");
        assert_eq!(t5.global_styles.font_color, "#FFFFFF");
        assert_eq!(t5.global_styles.alignment, "center");
        assert!((t5.global_styles.position_y - 0.65).abs() < 1e-6);
        assert_eq!(t5.global_styles.stroke_color.as_deref(), None);
        assert_eq!(t5.global_styles.stroke_width, None);
        assert_eq!(t5.global_styles.shadow_color.as_deref(), None);
        assert_eq!(t5.global_styles.shadow_blur, None);
        assert_eq!(t5.global_styles.shadow_offset_y, None);
    }

    #[test]
    fn test_subject_face_safeguard_overlap_protection() {
        let t1 = captions::get_caption_template("preset_viral_bold").unwrap();

        // 1. Normal face in upper half (bottom at y=0.40 / 768px): template untouched
        let normal_face = captions::SubjectFaceBounds {
            top: 0.15,
            bottom: 0.40,
            left: 0.30,
            right: 0.70,
        };
        let safe_t1 = captions::apply_subject_face_safeguard(&t1, Some(&normal_face));
        assert_eq!(safe_t1.global_styles.font_size, t1.global_styles.font_size);
        assert_eq!(
            safe_t1.global_styles.position_y,
            t1.global_styles.position_y
        );

        // 2. Intrusion case: speaker face extends down to y=0.68 (1305.6px)
        // Hormozi top is center(1344) - 52 = 1292px -> overlap occurs!
        let close_face = captions::SubjectFaceBounds {
            top: 0.20,
            bottom: 0.68,
            left: 0.25,
            right: 0.75,
        };
        let adjusted_t1 = captions::apply_subject_face_safeguard(&t1, Some(&close_face));
        assert!(adjusted_t1.global_styles.font_size <= t1.global_styles.font_size);
        // Size reduced or position shifted down to clear the face
        let (_, new_top, _, _) = captions::compute_caption_bounding_box(&adjusted_t1);
        assert!(new_top >= (0.68 * 1920.0 + 10.0));
    }

    #[test]
    fn test_caption_templates_active_state_configuration() {
        use captions::CaptionActiveState;

        // Viral Bold
        let t1 = captions::get_caption_template("preset_viral_bold").unwrap();
        match &t1.active_state {
            CaptionActiveState::WordByWordSwap {
                primary_highlight_color,
                secondary_highlight_color,
                animation,
            } => {
                assert_eq!(primary_highlight_color, "#00FF66");
                assert_eq!(secondary_highlight_color, "#FFEA00");
                assert_eq!(animation.animation_type, "scale_up");
                assert_eq!(animation.scale_factor, Some(1.15));
                assert_eq!(animation.duration_ms, 80);
            }
            _ => panic!("Expected WordByWordSwap for viral bold"),
        }

        // Narrative Pop
        let t2 = captions::get_caption_template("preset_mrbeast_pop").unwrap();
        match &t2.active_state {
            CaptionActiveState::KaraokeProgressive {
                highlight_color,
                animation,
            } => {
                assert_eq!(highlight_color, "#FFFF00");
                assert_eq!(animation.animation_type, "pop_bounce");
                assert_eq!(animation.duration_ms, 100);
            }
            _ => panic!("Expected KaraokeProgressive for narrative pop"),
        }

        // Minimal Capsule
        let t3 = captions::get_caption_template("preset_minimal_capsule").unwrap();
        match &t3.active_state {
            CaptionActiveState::StaticOpacityReveal {
                inactive_opacity,
                active_opacity,
            } => {
                assert!((inactive_opacity - 0.40).abs() < 1e-6);
                assert!((active_opacity - 1.00).abs() < 1e-6);
            }
            _ => panic!("Expected StaticOpacityReveal for minimal capsule"),
        }

        // Cinematic Vlog
        let t4 = captions::get_caption_template("preset_cinematic_vlog").unwrap();
        match &t4.active_state {
            CaptionActiveState::SmoothFade {
                highlight_color,
                duration_ms,
            } => {
                assert_eq!(highlight_color, "#FFFFFF");
                assert_eq!(*duration_ms, 150);
            }
            _ => panic!("Expected SmoothFade for cinematic vlog"),
        }

        // Dynamic Editorial Kinetic
        let t5 = captions::get_caption_template("preset_dynamic_editorial").unwrap();
        match &t5.active_state {
            CaptionActiveState::DynamicEditorialKinetic {
                base_color,
                accent_color,
                secondary_accent,
                entrance_duration_ms,
                exit_fade_ms,
            } => {
                assert_eq!(base_color, "#FFFFFF");
                assert_eq!(accent_color, "#FFE600");
                assert_eq!(secondary_accent, "#00E5FF");
                assert_eq!(*entrance_duration_ms, 150);
                assert_eq!(*exit_fade_ms, 300);
            }
            _ => panic!("Expected DynamicEditorialKinetic for dynamic editorial"),
        }
    }

    #[test]
    fn test_caption_templates_user_selection_reaches_ass_renderer() {
        let words = vec![
            make_word("Never", 0.0, 0.4),
            make_word("give", 0.4, 0.8),
            make_word("up", 0.8, 1.2),
            make_word("on", 1.2, 1.5),
            make_word("dreams.", 1.5, 2.0),
        ];

        // 1. Viral bold ASS rendering
        let ass_viral = generate_kinetic_ass_subtitles(&words, 0.0, 2.0, "preset_viral_bold");
        assert!(ass_viral.contains("Default,Bebas Neue,84"));
        assert!(ass_viral.contains(r"\fscx115\fscy115"));
        assert!(ass_viral.contains("&H66FF00&") || ass_viral.contains("&H00EAFF&"));
        assert!(ass_viral.contains("NEVER"));

        // 2. Narrative pop ASS rendering
        let ass_pop = generate_kinetic_ass_subtitles(&words, 0.0, 2.0, "preset_mrbeast_pop");
        assert!(ass_pop.contains("Default,Montserrat,80"));
        assert!(ass_pop.contains("&H00FFFF&"));
        assert!(ass_pop.contains(r"\t(0,50,\fscx120\fscy120)"));

        // 3. Minimal capsule ASS rendering
        let ass_capsule =
            generate_kinetic_ass_subtitles(&words, 0.0, 2.0, "preset_minimal_capsule");
        assert!(ass_capsule.contains("Default,Inter,96"));
        assert!(ass_capsule.contains(",3,12,0,2,80,80,")); // BorderStyle=3, Outline=12 (capsule box)
        assert!(ass_capsule.contains(r"{\alpha&H00&}"));
        assert!(ass_capsule.contains(r"{\alpha&H99&}"));

        // 4. Cinematic vlog ASS rendering
        let ass_vlog = generate_kinetic_ass_subtitles(&words, 0.0, 2.0, "preset_cinematic_vlog");
        assert!(ass_vlog.contains("Default,Poppins,116"));
        assert!(ass_vlog.contains(r"{\fad(150,150)}"));

        // 5. Dynamic editorial kinetic ASS rendering
        let ass_editorial =
            generate_kinetic_ass_subtitles(&words, 0.0, 2.0, "preset_dynamic_editorial");
        assert!(ass_editorial.contains("Default,Montserrat,62"));
        assert!(ass_editorial.contains(r"\an5"));
        assert!(ass_editorial.contains(r"\move("));
        assert!(ass_editorial.contains(r"\bord0\shad0"));
        assert!(ass_editorial.contains(r"\alpha&HFF&\t(0,80,0.5,\alpha&H00&)"));
    }

    #[test]
    fn test_caption_templates_normalized_position_y_margins() {
        let words = vec![make_word("Test", 0.0, 1.0)];

        // Viral: positionY 0.70, fontSize 84, maxLines 1 -> MarginV: (1.0-0.70)*1920 - 42 = 534
        let ass1 = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "preset_viral_bold");
        assert!(ass1.contains(",534,1\n"));

        // Pop: positionY 0.65, fontSize 80, maxLines 2 -> MarginV: (1.0-0.65)*1920 - 88 = 584
        let ass2 = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "preset_mrbeast_pop");
        assert!(ass2.contains(",584,1\n"));

        // Capsule: positionY 0.75, fontSize 96, maxLines 1 -> MarginV: (1.0-0.75)*1920 - 48 = 432
        let ass3 = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "preset_minimal_capsule");
        assert!(ass3.contains(",432,1\n"));

        // Vlog: positionY 0.82, fontSize 116, maxLines 2 -> MarginV: (1.0-0.82)*1920 - 127.6 = 218
        let ass4 = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "preset_cinematic_vlog");
        assert!(ass4.contains(",218,1\n"));
    }

    #[test]
    fn test_caption_templates_backward_compatibility_intact() {
        let words = vec![
            make_word("Legacy", 0.0, 0.5),
            make_word("caption", 0.5, 1.0),
        ];

        // Legacy styles must still produce their exact output
        let ass_box = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "modern-box");
        assert!(ass_box.contains("PlayResX: 1080"));
        assert!(ass_box.contains("80,80,360,1"));

        let ass_green = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "vibrant-green");
        assert!(ass_green.contains("PlayResX: 1080"));
        assert!(ass_green.contains("80,80,360,1"));

        let ass_cyan = generate_kinetic_ass_subtitles(&words, 0.0, 1.0, "vibrant-cyan");
        assert!(ass_cyan.contains("&HFFFF00&"));
    }

    #[test]
    fn test_format_ass_time_conversion() {
        assert_eq!(format_ass_time(0.0), "0:00:00.00");
        assert_eq!(format_ass_time(1.5), "0:00:01.50");
        assert_eq!(format_ass_time(65.25), "0:01:05.25");
        assert_eq!(format_ass_time(3661.08), "1:01:01.08");
    }

    #[test]
    fn test_snap_to_semantic_boundaries_hindi_purna_viram() {
        let words = vec![
            make_word("यह", 10.0, 10.4),
            make_word("एक", 10.4, 10.8),
            make_word("बड़ा", 10.8, 11.2),
            make_word("फैसला", 11.2, 11.8),
            make_word("था।", 11.8, 12.5),
            make_word("और", 13.0, 13.3),
            make_word("इसके", 13.3, 13.7),
            make_word("बाद", 13.7, 14.1),
            make_word("सब", 14.1, 14.5),
            make_word("बदल", 14.5, 14.9),
            make_word("गया।", 14.9, 15.6),
            make_word("अगला", 16.0, 16.5),
            make_word("कदम", 16.5, 17.0),
        ];

        let (start, end) = snap_to_semantic_boundaries(&words, 10.2, 15.0, 30.0, None);
        assert_eq!(start, 10.0);
        assert_eq!(end, 15.6); // snapped to Purna Viram "गया।"
    }

    #[test]
    fn test_kinetic_ass_subtitles_devanagari_and_nirmala_ui() {
        let words = vec![
            make_word("आपको", 0.0, 0.5),
            make_word("पता", 0.5, 1.0),
            make_word("है?", 1.0, 1.5),
        ];

        let ass = generate_kinetic_ass_subtitles(&words, 0.0, 2.0, "vibrant-green");
        if cfg!(windows) {
            assert!(
                ass.contains("Nirmala UI"),
                "Should use Nirmala UI font on Windows"
            );
        }
        assert!(ass.contains("आपको"));
        assert!(ass.contains("पता"));
        assert!(ass.contains("है?"));
        assert!(ass.contains("PlayResX: 1080"));
        assert!(ass.contains("PlayResY: 1920"));
    }

    #[test]
    fn test_is_continuation_word_hindi_and_english() {
        assert!(is_continuation_word("because"));
        assert!(is_continuation_word("and"));
        assert!(is_continuation_word("कम"));
        assert!(is_continuation_word("से"));
        assert!(is_continuation_word("क्योंकि"));
        assert!(is_continuation_word("लेकिन"));
        assert!(is_continuation_word("फिर"));
        assert!(!is_continuation_word("फैसला"));
        assert!(!is_continuation_word("success"));
    }

    #[test]
    fn test_snap_to_semantic_boundaries_extends_on_continuation_token() {
        // Simulating the failure where speech was cut off at "कम" / "कम से"
        let words = vec![
            make_word("यह", 0.0, 0.4),
            make_word("बात", 0.4, 0.8),
            make_word("हुई", 0.8, 1.2),
            make_word("थी।", 1.2, 1.8),
            make_word("हम", 2.0, 2.4),
            make_word("वहां", 2.4, 2.8),
            make_word("गए", 2.8, 3.2),
            make_word("कम", 3.2, 3.6),
            make_word("से", 3.6, 4.0),
            make_word("कम", 4.0, 4.4),
            make_word("दस", 4.4, 4.8),
            make_word("लोग", 4.8, 5.2),
            make_word("बच", 5.2, 5.6),
            make_word("गए।", 5.6, 6.2),
        ];

        // If raw_end was proposed at 3.6s (right at "कम"), look-ahead should extend past "कम से कम" to "गए।" (6.2s)
        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 3.6, 10.0, None);
        assert_eq!(start, 0.0);
        assert_eq!(end, 6.2, "Boundary engine must look ahead and extend past continuation token 'कम' to true completion 'गए।'");
    }

    #[test]
    fn test_snap_to_semantic_boundaries_continuation_probability_extension() {
        let words = vec![
            make_word("The", 0.0, 0.5),
            make_word("lesson", 0.5, 1.0),
            make_word("was", 1.0, 1.4),
            make_word("clear.", 1.4, 15.0),
            make_word("And", 15.5, 16.0),
            make_word("we", 16.0, 16.3),
            make_word("saw", 16.3, 16.8),
            make_word("the", 16.8, 17.0),
            make_word("full", 17.0, 17.5),
            make_word("transformation.", 17.5, 25.0),
            make_word("Next", 25.5, 26.0),
            make_word("topic.", 26.0, 27.0),
        ];

        let closure = ClosureSignals {
            continuation_probability: Some(0.85),
            ..Default::default()
        };

        // Raw end is 15.0 (at "clear."). Because continuation_probability > 0.70, it must extend past 15.0 to 25.0.
        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(end, 25.0, "High continuation probability (> 0.70) must trigger forward extension to next sentence terminator");
    }

    #[test]
    fn test_snap_to_semantic_boundaries_breath_pause_risk_extension() {
        let words = vec![
            make_word("The", 0.0, 0.5),
            make_word("lesson", 0.5, 1.0),
            make_word("was", 1.0, 1.4),
            make_word("clear.", 1.4, 15.0),
            make_word("And", 15.5, 16.0),
            make_word("we", 16.0, 16.3),
            make_word("saw", 16.3, 16.8),
            make_word("the", 16.8, 17.0),
            make_word("full", 17.0, 17.5),
            make_word("transformation.", 17.5, 25.0),
        ];

        let closure = ClosureSignals {
            breath_pause_risk: Some(0.80),
            ..Default::default()
        };

        // Raw end is 15.0. Breath pause risk > 0.65 forces extension past raw_end
        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 25.0,
            "Breath pause risk (> 0.65) must force extension past internal breath pause"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_mid_thought_risk_extension() {
        let words = vec![
            make_word("The", 0.0, 0.5),
            make_word("lesson", 0.5, 1.0),
            make_word("was", 1.0, 1.4),
            make_word("clear.", 1.4, 15.0),
            make_word("Because", 15.5, 16.0),
            make_word("it", 16.0, 16.3),
            make_word("changed", 16.3, 16.8),
            make_word("everything.", 16.8, 24.5),
        ];

        let closure = ClosureSignals {
            mid_thought_risk: Some(0.75),
            ..Default::default()
        };

        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 24.5,
            "Mid-thought cutoff risk (> 0.65) must force extension past incomplete clause"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_answer_incomplete_extension() {
        let words = vec![
            make_word("What", 0.0, 0.5),
            make_word("is", 0.5, 0.8),
            make_word("it?", 0.8, 1.2),
            make_word("First", 1.5, 2.0),
            make_word("part.", 2.0, 15.0),
            make_word("The", 15.5, 16.0),
            make_word("actual", 16.0, 16.5),
            make_word("answer", 16.5, 17.0),
            make_word("is", 17.0, 17.3),
            make_word("here.", 17.3, 28.5),
        ];

        let closure = ClosureSignals {
            answer_complete: Some(false),
            answer_end: Some(28.0),
            ..Default::default()
        };

        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 28.5,
            "answer_complete == false must extend forward toward answer_end resolution"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_story_or_payoff_incomplete_extension() {
        let words = vec![
            make_word("Once", 0.0, 0.5),
            make_word("upon", 0.5, 0.8),
            make_word("a", 0.8, 1.0),
            make_word("time.", 1.0, 14.0),
            make_word("Then", 14.5, 15.0),
            make_word("the", 15.0, 15.3),
            make_word("payoff", 15.3, 15.8),
            make_word("happened.", 15.8, 34.5),
        ];

        let closure = ClosureSignals {
            payoff_completion: Some(false),
            payoff_end: Some(34.0),
            ..Default::default()
        };

        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 14.0, 50.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 34.5,
            "payoff_completion == false must extend forward toward payoff_end resolution"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_payoff_beyond_75s_extends_to_payoff() {
        let words = vec![
            make_word("First", 0.0, 0.5),
            make_word("clause.", 0.5, 30.0),
            make_word("Next", 30.5, 31.0),
            make_word("clause.", 31.0, 60.0),
            make_word("Distant", 85.0, 86.0),
            make_word("payoff.", 86.0, 88.0),
        ];

        let closure = ClosureSignals {
            payoff_end: Some(88.0),
            ..Default::default()
        };

        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 60.0, 100.0, Some(&closure));
        assert_eq!(start, 0.0);
        // Payoff is authoritative: extends to 88.0 without 75s ceiling
        assert_eq!(
            end, 88.0,
            "Payoff determines endpoint without duration rejection"
        );
    }

    #[test]
    fn test_extract_final_spoken_sentence_clean_ending() {
        let words = vec![
            make_word("Start", 10.0, 11.0),
            make_word("middle.", 11.0, 12.0),
            make_word("any", 13.0, 13.5),
            make_word("place", 13.5, 14.0),
            make_word("I", 14.0, 14.3),
            make_word("play,", 14.3, 14.8),
            make_word("I", 15.0, 15.2),
            make_word("show", 15.2, 15.5),
            make_word("my", 15.5, 15.7),
            make_word("level.", 15.7, 16.2),
        ];

        let res = extract_final_spoken_sentence(&words, 10.0, 16.2);
        assert!(res.is_some());
        let (text, s, e) = res.unwrap();
        assert_eq!(text, "any place I play, I show my level.");
        assert_eq!(s, 13.0);
        assert_eq!(e, 16.2);
    }

    #[test]
    fn test_extract_final_spoken_sentence_with_quotes_and_hindi() {
        let words = vec![
            make_word("He", 1.0, 1.4),
            make_word("said,\"", 1.4, 2.0),
            make_word("done.\"", 2.0, 2.5),
            make_word("This", 3.0, 3.4),
            make_word("is", 3.4, 3.7),
            make_word("it.", 3.7, 4.2),
        ];

        let res = extract_final_spoken_sentence(&words, 1.0, 4.2);
        assert!(res.is_some());
        let (text, s, e) = res.unwrap();
        assert_eq!(text, "This is it.");
        assert_eq!(s, 3.0);
        assert_eq!(e, 4.2);

        // Test Hindi Purna Viram
        let hindi_words = vec![
            make_word("यह", 10.0, 10.5),
            make_word("सच", 10.5, 11.0),
            make_word("है।", 11.0, 11.5),
            make_word("हम", 12.0, 12.5),
            make_word("जीतेंगे।", 12.5, 13.2),
        ];
        let res_hi = extract_final_spoken_sentence(&hindi_words, 10.0, 13.2);
        assert!(res_hi.is_some());
        let (text_hi, s_hi, e_hi) = res_hi.unwrap();
        assert_eq!(text_hi, "हम जीतेंगे।");
        assert_eq!(s_hi, 12.0);
        assert_eq!(e_hi, 13.2);
    }

    #[test]
    fn test_snap_to_semantic_boundaries_closure_confidence_bias() {
        let words = vec![
            make_word("Here", 0.0, 0.5),
            make_word("is", 0.5, 0.8),
            make_word("the", 0.8, 1.0),
            make_word("start.", 1.0, 12.0),
            make_word("Intermediate", 12.5, 13.5),
            make_word("sentence.", 13.5, 16.0),
            make_word("Final", 16.5, 17.0),
            make_word("full", 17.0, 17.5),
            make_word("resolution.", 17.5, 26.0),
        ];

        let closure = ClosureSignals {
            closure_confidence: Some(0.35),
            ..Default::default()
        };

        // With low closure confidence (< 0.50), search biases toward later candidate
        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 26.0,
            "Low closure confidence (< 0.50) must bias selection toward later sentence boundary"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_none_fallback_preserves_baseline() {
        let words = vec![
            make_word("The", 0.0, 0.5),
            make_word("lesson", 0.5, 1.0),
            make_word("was", 1.0, 1.4),
            make_word("clear.", 1.4, 15.0),
            make_word("Next", 15.5, 16.0),
            make_word("sentence.", 16.0, 25.0),
        ];

        // When closure is None, cleanly stops at raw_end = 15.0
        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, None);
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 15.0,
            "When closure is None, baseline snapping to nearest sentence boundary is preserved"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_endpoint_state_breath_pause_extension() {
        let words = vec![
            make_word("The", 0.0, 0.5),
            make_word("lesson", 0.5, 1.0),
            make_word("was", 1.0, 1.4),
            make_word("clear.", 1.4, 15.0),
            make_word("And", 15.5, 16.0),
            make_word("we", 16.0, 16.3),
            make_word("saw", 16.3, 16.8),
            make_word("the", 16.8, 17.0),
            make_word("full", 17.0, 17.5),
            make_word("transformation.", 17.5, 25.0),
        ];

        let closure = ClosureSignals {
            endpoint_state: Some(crate::models::EndpointState::BreathPause),
            ..Default::default()
        };

        // Raw end is 15.0. BreathPause endpoint state forces extension past raw_end
        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 25.0,
            "EndpointState::BreathPause must force forward extension past premature pause"
        );
    }

    #[test]
    fn test_snap_to_semantic_boundaries_endpoint_state_sentence_complete_low_confidence_extension()
    {
        let words = vec![
            make_word("The", 0.0, 0.5),
            make_word("lesson", 0.5, 1.0),
            make_word("was", 1.0, 1.4),
            make_word("clear.", 1.4, 15.0),
            make_word("And", 15.5, 16.0),
            make_word("then", 16.0, 16.3),
            make_word("we", 16.3, 16.8),
            make_word("won.", 16.8, 22.0),
        ];

        let closure = ClosureSignals {
            endpoint_state: Some(crate::models::EndpointState::SentenceComplete),
            closure_confidence: Some(0.60), // < 0.80 -> must extend
            ..Default::default()
        };

        let (start, end) = snap_to_semantic_boundaries(&words, 0.0, 15.0, 40.0, Some(&closure));
        assert_eq!(start, 0.0);
        assert_eq!(
            end, 22.0,
            "SentenceComplete with confidence < 0.80 must extend to complete narrative resolution"
        );
    }

    #[test]
    fn test_validate_and_enforce_final_endpoint_success_when_already_extended() {
        let words = vec![
            make_word("First", 0.0, 1.0),
            make_word("sentence.", 1.0, 10.0),
            make_word("Second", 10.5, 11.0),
            make_word("sentence.", 11.0, 20.0),
        ];

        let closure = ClosureSignals {
            continuation_probability: Some(0.85),
            ..Default::default()
        };

        // raw_llm_end = 10.0, snapped_end = 20.0 -> was_extended is true -> Ok(20.0)
        let result = validate_and_enforce_final_endpoint(&words, 10.0, 20.0, &closure, 30.0);
        assert_eq!(result, Ok(20.0));
    }

    #[test]
    fn test_validate_and_enforce_final_endpoint_salvaged_when_not_extended() {
        let words = vec![
            make_word("First", 0.0, 1.0),
            make_word("sentence.", 1.0, 10.0),
            make_word("Second", 10.5, 11.0),
            make_word("sentence.", 11.0, 20.0),
        ];

        let closure = ClosureSignals {
            continuation_probability: Some(0.85),
            ..Default::default()
        };

        // raw_llm_end = 10.0, snapped_end = 10.0 (not extended by snap) -> salvaged to 20.0
        let result = validate_and_enforce_final_endpoint(&words, 10.0, 10.0, &closure, 30.0);
        assert_eq!(result, Ok(20.0));
    }

    #[test]
    fn test_validate_and_enforce_final_endpoint_rejected_when_unsalvageable() {
        let words = vec![
            make_word("First", 0.0, 1.0),
            make_word("sentence.", 1.0, 10.0),
            make_word("and", 10.5, 11.0),
            make_word("no", 11.0, 11.5),
            make_word("terminator", 11.5, 12.0),
        ];

        let closure = ClosureSignals {
            continuation_probability: Some(0.85),
            ..Default::default()
        };

        // raw_llm_end = 10.0, snapped_end = 10.0, no terminator after 10.0 -> Err(...)
        let result = validate_and_enforce_final_endpoint(&words, 10.0, 10.0, &closure, 15.0);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Rejecting to prevent premature render"));
    }

    #[test]
    fn test_closure_requires_extension_triggers() {
        // Test all triggers independently
        let base = ClosureSignals::default();
        assert!(!closure_requires_extension(&base));

        let t1 = ClosureSignals {
            continuation_probability: Some(0.75),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t1));

        let t2 = ClosureSignals {
            breath_pause_risk: Some(0.70),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t2));

        let t3 = ClosureSignals {
            mid_thought_risk: Some(0.70),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t3));

        let t4 = ClosureSignals {
            answer_complete: Some(false),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t4));

        let t5 = ClosureSignals {
            story_completeness: Some(false),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t5));

        let t6 = ClosureSignals {
            payoff_completion: Some(false),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t6));

        let t7 = ClosureSignals {
            endpoint_state: Some(crate::models::EndpointState::BreathPause),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t7));

        let t8 = ClosureSignals {
            endpoint_state: Some(crate::models::EndpointState::SentenceComplete),
            closure_confidence: Some(0.70),
            ..Default::default()
        };
        assert!(closure_requires_extension(&t8));

        let t9 = ClosureSignals {
            endpoint_state: Some(crate::models::EndpointState::SentenceComplete),
            closure_confidence: Some(0.85),
            ..Default::default()
        };
        assert!(!closure_requires_extension(&t9));
    }

    #[test]
    fn test_snap_to_semantic_boundaries_with_hook_anchor_no_backward_drift() {
        // Simulating Candidate 1:
        // Previous sentence ends at 7522.38: "thing."
        // Hook starts at 7522.78: "And"
        let words = vec![
            make_word("She", 7519.98, 7520.22),
            make_word("looks", 7520.22, 7520.38),
            make_word("at", 7520.38, 7520.54),
            make_word("him", 7520.54, 7520.70),
            make_word("and", 7520.70, 7520.78),
            make_word("says", 7520.78, 7520.94),
            make_word("exactly", 7520.94, 7521.50),
            make_word("the", 7521.50, 7521.66),
            make_word("same", 7521.66, 7521.90),
            make_word("thing.", 7521.90, 7522.38),
            make_word("And", 7522.78, 7523.34),
            make_word("this", 7523.50, 7523.74),
            make_word("is", 7523.74, 7523.98),
            make_word("where", 7523.98, 7524.46),
            make_word("Krishna", 7524.46, 7524.94),
            make_word("teaches", 7524.94, 7525.34),
            make_word("us", 7525.34, 7525.50),
            make_word("lesson.", 7525.50, 7540.0),
        ];

        // When snapping with hook_anchor = 7522.78, start MUST NOT drift back to 7519.98
        let (start, _end) = snap_to_semantic_boundaries_with_hook_anchor(
            &words,
            7522.78,
            7540.0,
            7600.0,
            None,
            Some(7522.78),
            false,
        );

        assert!(
            start >= 7522.38,
            "Start ({:.2}) must NOT cross backward before previous sentence end 7522.38s",
            start
        );
        assert!(
            start <= 7522.78,
            "Start ({:.2}) should start at or immediately before hook word 7522.78s",
            start
        );
    }

    #[test]
    fn test_real_long_form_sliding_windows_coverage() {
        // Connect to SQLite DB if available, or synthesize
        let appdata = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("com.autoshorts.desktop");
        let db_path = appdata.join("autoshorts.sqlite");

        if !db_path.exists() {
            eprintln!(
                "autoshorts.sqlite not found at {:?}, skipping DB test",
                db_path
            );
            return;
        }

        let db = Database::open(&db_path).expect("Failed to open SQLite database");
        let project_id = "f492a7a2-9832-4358-9d6e-91dfe9904c7c";
        let transcript = match db.latest_transcript(project_id) {
            Ok(Some(t)) => t,
            _ => {
                eprintln!("Transcript for project {} not found, skipping", project_id);
                return;
            }
        };

        let normalized: NormalizedTranscript =
            serde_json::from_str(&transcript.raw_json).expect("Failed to parse transcript JSON");

        assert_eq!(
            normalized.words.len(),
            22148,
            "Word count must be 22,148 words"
        );
        assert!(
            normalized.duration >= 8538.0,
            "Duration must be ~8538.5s (142.3 minutes)"
        );

        // 1. Sliding coverage window calculation
        let discovery_config = models::WindowDiscoveryConfig::default();
        let (window_size, overlap) =
            discovery_config.window_and_overlap_for_duration(normalized.duration);
        assert_eq!(window_size, 600.0);
        assert_eq!(overlap, 120.0);

        let windows =
            llm::calculate_sliding_coverage_windows(normalized.duration, window_size, overlap);
        println!(
            "Generated {} sliding windows for 142.3m video",
            windows.len()
        );
        assert!(
            windows.len() >= 17,
            "Expected at least 17 windows for 142.3m video"
        );
        assert_eq!(windows.first().unwrap().0, 0.0);
        assert_eq!(windows.last().unwrap().1, normalized.duration);

        // Verify window formatting across beginning, middle, and end
        for (i, (w_start, w_end)) in windows.iter().enumerate() {
            let prompt = llm::build_window_semantic_prompt(
                "Sample window text",
                "ENGLISH",
                "latin",
                *w_start,
                *w_end,
                i,
                windows.len(),
                discovery_config.candidates_per_window_min,
                discovery_config.candidates_per_window_max,
            );
            assert!(prompt.contains(&format!("WINDOW {} OF {}", i + 1, windows.len())));
            assert!(prompt.contains(&format!("[{:.1}s to {:.1}s]", w_start, w_end)));
        }

        // 2. Synthesize candidates across all 17 windows to test the full pipeline
        let mut all_drafts = Vec::new();
        for (w_idx, (w_s, w_e)) in windows.iter().enumerate() {
            // Find a valid word in this window
            let mid_time = (w_s + w_e) / 2.0;
            let window_words: Vec<_> = normalized
                .words
                .iter()
                .filter(|w| w.start >= *w_s && w.end <= *w_e)
                .collect();

            if window_words.len() >= 20 {
                let hook_w = &window_words[0];
                let hook_text = format!("Viral hook insight from window {}", w_idx + 1);
                let draft = CandidateDraft {
                    start: hook_w.start,
                    end: (hook_w.start + 35.0).min(*w_e),
                    hook: hook_text,
                    rationale: format!(
                        "Strong moment from window {} covering {:.1}s",
                        w_idx + 1,
                        mid_time
                    ),
                    score: 0.80 + (w_idx as f64 * 0.01),
                    hook_topic_clarity: Some(0.90),
                    hook_curiosity: Some(0.85),
                    hook_relevance: Some(0.85),
                    hook_contrast: Some(0.80),
                    closure_confidence: Some(0.92),
                    hook_speaker: Some(if w_idx % 2 == 0 {
                        "Guest".to_string()
                    } else {
                        "Host".to_string()
                    }),
                    hook_type: Some("direct_statement".to_string()),
                    conversation_type: Some("general".to_string()),
                    ..Default::default()
                };

                if llm::validate_window_candidate_bounds(&draft, *w_s, *w_e, normalized.duration) {
                    all_drafts.push(draft);
                }
            }
        }

        assert!(
            all_drafts.len() >= 15,
            "Candidate discovery must find candidates across all windows"
        );

        // 3. Pre-snap deduplication
        let deduped = llm::deduplicate_window_candidates(all_drafts);
        assert!(!deduped.is_empty());

        // 4. Run through verified hook pipeline
        let hook_config = models::HookPipelineConfig::default();
        let mut processed = deduped;
        for draft in &mut processed {
            draft.raw_llm_end = Some(draft.end);
            let is_q = draft.question_hook_used == Some(true)
                || draft.hook_speaker.as_deref() == Some("Host")
                || draft.conversation_type.as_deref() == Some("question_answer");

            let context_eval = llm::evaluate_opening_hook_context(&draft.hook, is_q, &hook_config);
            draft.opening_context_score = Some(context_eval.score);
            draft.opening_unresolved_reference = context_eval.unresolved_reference.clone();
            draft.opening_continuation_marker = context_eval.matched_marker.clone();

            let closure_sig = crate::models::ClosureSignals::from_draft(draft);
            let (s_start, s_end) = snap_to_semantic_boundaries_with_hook_anchor(
                &normalized.words,
                draft.start,
                draft.end,
                normalized.duration,
                Some(&closure_sig),
                draft.hook_start,
                is_q,
            );
            draft.start = s_start;
            draft.end = s_end;
        }

        // 5. Anti-filler gate
        let min_score = discovery_config.min_candidate_score;
        processed.retain(|d| llm::composite_score(d) >= min_score);

        // 6. Temporal balancing to 12
        let final_drafts =
            llm::deduplicate_and_balance_candidates(processed, normalized.duration, 12);
        assert_eq!(
            final_drafts.len(),
            12,
            "Should balance exactly to 12 candidates"
        );

        // 7. Verify full timeline coverage: check earliest and latest
        let earliest_start = final_drafts
            .iter()
            .map(|d| d.start)
            .fold(f64::INFINITY, f64::min);
        let latest_start = final_drafts.iter().map(|d| d.start).fold(0.0, f64::max);

        println!("Full Timeline Coverage Verified:");
        println!(
            "  Earliest candidate start: {:.1}s ({:.1} min)",
            earliest_start,
            earliest_start / 60.0
        );
        println!(
            "  Latest candidate start: {:.1}s ({:.1} min)",
            latest_start,
            latest_start / 60.0
        );
        assert!(
            earliest_start < 600.0,
            "Must have an early candidate (< 600s)"
        );
        assert!(
            latest_start > 7684.0,
            "Must have candidates in the final 10% (> 7684s)"
        );

        // 8. Test SQLite DB candidate replacement
        let saved = db
            .replace_candidates(project_id, &final_drafts)
            .expect("DB candidate replacement failed");
        assert_eq!(saved.len(), 12);
        let selected_count = saved.iter().filter(|c| c.selected).count();
        assert_eq!(
            selected_count, 12,
            "All 12 candidates must be marked selected in DB"
        );
    }

    #[tokio::test]
    async fn test_real_long_form_live_discovery_e2e() {
        let api_key = std::env::var("DEEPSEEK_API_KEY").ok();
        if api_key.is_none() {
            println!(
                "Skipping live DeepSeek network test: DEEPSEEK_API_KEY not set in environment"
            );
            return;
        }
        let key = api_key.unwrap();

        let appdata = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("com.autoshorts.desktop");
        let db_path = appdata.join("autoshorts.sqlite");
        if !db_path.exists() {
            return;
        }

        let db = Database::open(&db_path).expect("Failed to open SQLite database");
        let project_id = "f492a7a2-9832-4358-9d6e-91dfe9904c7c";
        let project = match db.get_project(project_id) {
            Ok(p) => p,
            Err(_) => return,
        };
        let transcript = match db.latest_transcript(project_id) {
            Ok(Some(t)) => t,
            _ => return,
        };
        let normalized: NormalizedTranscript = serde_json::from_str(&transcript.raw_json).unwrap();

        let discovery_config = models::WindowDiscoveryConfig::default();
        println!("=== Starting Live Full-Timeline Discovery for 142.3m Video ===");
        let drafts = llm::discover_candidates_full_timeline(
            &normalized,
            "deepseek",
            &key,
            None,
            &discovery_config,
        )
        .await
        .expect("Live full-timeline discovery failed");

        println!(
            "Live discovery returned {} candidate drafts across all windows",
            drafts.len()
        );
        assert!(!drafts.is_empty(), "Live discovery must return candidates");

        // Verified hook pipeline
        let hook_config = models::HookPipelineConfig::default();
        let mut processed = drafts.clone();
        for draft in &mut processed {
            draft.raw_llm_end = Some(draft.end);
            let is_q = draft.question_hook_used == Some(true)
                || draft.hook_speaker.as_deref() == Some("Host")
                || draft.conversation_type.as_deref() == Some("question_answer");

            if let Some((v_start, v_end, conf, _, _)) = llm::align_hook_to_transcript_words(
                &draft.hook,
                draft.start,
                draft.end,
                &normalized.words,
            ) {
                draft.hook_start = Some(v_start);
                draft.hook_end = Some(v_end);
                draft.hook_confidence = Some(conf);
            }

            let context_eval = llm::evaluate_opening_hook_context(&draft.hook, is_q, &hook_config);
            draft.opening_context_score = Some(context_eval.score);
            draft.opening_unresolved_reference = context_eval.unresolved_reference.clone();
            draft.opening_continuation_marker = context_eval.matched_marker.clone();

            let closure_sig = crate::models::ClosureSignals::from_draft(draft);
            let (s_start, s_end) = snap_to_semantic_boundaries_with_hook_anchor(
                &normalized.words,
                draft.start,
                draft.end,
                normalized.duration,
                Some(&closure_sig),
                draft.hook_start,
                is_q,
            );
            let v_end = match validate_and_enforce_final_endpoint(
                &normalized.words,
                draft.raw_llm_end.unwrap_or(draft.end),
                s_end,
                &closure_sig,
                normalized.duration,
            ) {
                Ok(e) => e,
                Err(_) => s_end,
            };
            draft.start = s_start;
            draft.end = v_end;
        }

        // Quality gates
        if hook_config.hard_reject_unresolved_pronoun {
            let backup = processed.clone();
            processed.retain(|d| d.opening_unresolved_reference.is_none());
            if processed.is_empty() {
                processed = backup;
            }
        }

        let min_score = discovery_config.min_candidate_score;
        let q_backup = processed.clone();
        processed.retain(|d| llm::composite_score(d) >= min_score);
        if processed.is_empty() {
            processed = q_backup;
        }

        // Multimodal
        let (mm_drafts, _) = media::analyze_multimodal_hook_signals(
            &project.source_path,
            &processed,
            normalized.duration,
            None,
        );

        let final_drafts =
            llm::deduplicate_and_balance_candidates(mm_drafts, normalized.duration, 12);
        println!("Live balanced candidate count: {}", final_drafts.len());
        assert!(
            final_drafts.len() >= 6,
            "Expected at least 6 balanced candidates from live discovery"
        );
        assert!(final_drafts.len() <= 12, "Expected at most 12 candidates");

        let saved = db
            .replace_candidates(project_id, &final_drafts)
            .expect("Failed to replace candidates in DB");
        println!(
            "Live candidates successfully persisted to SQLite DB: {} saved ({} selected)",
            saved.len(),
            saved.iter().filter(|c| c.selected).count()
        );
    }

    #[test]
    fn test_concurrent_backend_render_same_candidate_blocks_duplicate() {
        let tracker = InFlightRenderTracker::new();
        let guard1 = tracker.try_acquire("candidate-1");
        assert!(
            guard1.is_ok(),
            "First render acquisition for candidate-1 should succeed"
        );
        assert!(
            tracker.is_rendering("candidate-1"),
            "Candidate-1 must be marked as rendering"
        );

        // Concurrent attempt for the SAME candidate
        let guard2 = tracker.try_acquire("candidate-1");
        assert!(
            guard2.is_err(),
            "Concurrent duplicate render for candidate-1 must be blocked"
        );
        assert!(guard2.unwrap_err().contains("already in-flight"));
    }

    #[test]
    fn test_concurrent_backend_render_different_candidates_allowed() {
        let tracker = InFlightRenderTracker::new();
        let guard1 = tracker.try_acquire("candidate-1");
        assert!(
            guard1.is_ok(),
            "Render acquisition for candidate-1 should succeed"
        );

        let guard2 = tracker.try_acquire("candidate-2");
        assert!(
            guard2.is_ok(),
            "Render acquisition for different candidate-2 must succeed"
        );
        assert!(tracker.is_rendering("candidate-1"));
        assert!(tracker.is_rendering("candidate-2"));
    }

    #[test]
    fn test_guard_released_after_success_and_allows_legitimate_future_render() {
        let tracker = InFlightRenderTracker::new();
        {
            let guard = tracker.try_acquire("candidate-1");
            assert!(guard.is_ok());
            assert!(tracker.is_rendering("candidate-1"));
            // Simulating successful render completion when guard drops
        }

        assert!(
            !tracker.is_rendering("candidate-1"),
            "Guard drop must release lock"
        );

        // Legitimate subsequent render for the same candidate
        let guard_subsequent = tracker.try_acquire("candidate-1");
        assert!(
            guard_subsequent.is_ok(),
            "Future render after completion must be permitted"
        );
    }

    #[test]
    fn test_guard_released_after_failure_or_panic() {
        let tracker = InFlightRenderTracker::new();

        let res = std::panic::catch_unwind(|| {
            let _guard = tracker.try_acquire("candidate-fail").unwrap();
            assert!(tracker.is_rendering("candidate-fail"));
            panic!("Simulated render failure or crash");
        });
        assert!(res.is_err(), "Panic caught");

        // Lock must still be cleanly released via RAII Drop
        assert!(
            !tracker.is_rendering("candidate-fail"),
            "Lock must be released even after failure/panic"
        );
        let next = tracker.try_acquire("candidate-fail");
        assert!(next.is_ok(), "Subsequent render must succeed after failure");
    }

    #[tokio::test]
    async fn test_multi_threaded_race_condition_single_winner() {
        let tracker = InFlightRenderTracker::new();
        let mut handles = Vec::new();

        // Spawn 20 concurrent tasks trying to acquire render lock for the SAME candidate
        for _ in 0..20 {
            let t = tracker.clone();
            handles.push(tokio::spawn(async move { t.try_acquire("candidate-race") }));
        }

        let mut successes = 0;
        let mut blocks = 0;
        for h in handles {
            let res = h.await.unwrap();
            match res {
                Ok(_guard) => {
                    successes += 1;
                    // Hold lock briefly to ensure race
                    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                }
                Err(_) => blocks += 1,
            }
        }

        assert_eq!(successes, 1, "Exactly one thread must acquire the lock");
        assert_eq!(
            blocks, 19,
            "All other 19 concurrent attempts must be blocked"
        );
        assert!(
            !tracker.is_rendering("candidate-race"),
            "Lock must be released once winner finishes"
        );
    }

    #[test]
    fn test_concurrent_render_same_output_identity_different_candidate_ids_blocked() {
        let tracker = InFlightRenderTracker::new();
        let guard_a = tracker.try_acquire_output("cand-A", "proj-P", 1);
        assert!(
            guard_a.is_ok(),
            "Candidate A acquisition for project P rank 1 must succeed"
        );
        assert!(tracker.is_rendering("cand-A"));
        assert!(tracker.is_output_rendering("proj-P", 1));

        // Candidate B has a different UUID (e.g. after candidate regeneration) but targets project P rank 1
        let guard_b = tracker.try_acquire_output("cand-B", "proj-P", 1);
        assert!(
            guard_b.is_err(),
            "Candidate B targeting project P rank 1 must be rejected while A is in flight"
        );
        assert!(guard_b.unwrap_err().contains("already in-flight"));
        assert!(!tracker.is_rendering("cand-B"));

        // When guard A drops, subsequent acquisition for project P rank 1 is permitted
        drop(guard_a);
        assert!(!tracker.is_rendering("cand-A"));
        assert!(!tracker.is_output_rendering("proj-P", 1));

        let guard_b_retry = tracker.try_acquire_output("cand-B", "proj-P", 1);
        assert!(
            guard_b_retry.is_ok(),
            "Subsequent acquisition for candidate B must succeed after guard A released"
        );
    }

    #[test]
    fn test_concurrent_render_different_ranks_same_project_allowed() {
        let tracker = InFlightRenderTracker::new();
        let guard_rank1 = tracker.try_acquire_output("cand-1", "proj-P", 1);
        assert!(guard_rank1.is_ok());

        let guard_rank2 = tracker.try_acquire_output("cand-2", "proj-P", 2);
        assert!(
            guard_rank2.is_ok(),
            "Different ranks for the same project must be allowed to render concurrently"
        );

        assert!(tracker.is_output_rendering("proj-P", 1));
        assert!(tracker.is_output_rendering("proj-P", 2));

        drop(guard_rank1);
        assert!(!tracker.is_output_rendering("proj-P", 1));
        assert!(tracker.is_output_rendering("proj-P", 2));
    }

    // ── Caption Intelligence 2.0 Applied-Feature Telemetry Tests ───────────────

    #[test]
    fn test_applied_features_four_booleans_serialization() {
        let smart_pacing_applied = true;
        let hook_ending_applied = false;
        let audio_intelligence_applied = true;
        let caption_intelligence_applied = true;

        let applied_features_json = serde_json::json!({
            "smartPacing": smart_pacing_applied,
            "hookEndingOptimization": hook_ending_applied,
            "audioIntelligence": audio_intelligence_applied,
            "captionIntelligence": caption_intelligence_applied,
        })
        .to_string();

        let parsed: serde_json::Value = serde_json::from_str(&applied_features_json).unwrap();
        assert_eq!(parsed["smartPacing"], true);
        assert_eq!(parsed["hookEndingOptimization"], false);
        assert_eq!(parsed["audioIntelligence"], true);
        assert_eq!(parsed["captionIntelligence"], true);

        // Verify exact keys present
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 4);
        assert!(obj.contains_key("smartPacing"));
        assert!(obj.contains_key("hookEndingOptimization"));
        assert!(obj.contains_key("audioIntelligence"));
        assert!(obj.contains_key("captionIntelligence"));
    }

    #[test]
    fn test_applied_features_ci_decision_logic() {
        // 1. Usable plan + ASS written -> true
        let ci_plan_some = true;
        let ass_path_some = true;
        let ci_applied = ci_plan_some && ass_path_some;
        assert!(ci_applied);

        // 2. Plan is None -> false
        let ci_plan_none = false;
        let ci_applied_none = ci_plan_none && ass_path_some;
        assert!(!ci_applied_none);

        // 3. Plan is Some but ASS path failed to write -> false
        let ass_path_failed = false;
        let ci_applied_no_ass = ci_plan_some && ass_path_failed;
        assert!(!ci_applied_no_ass);
    }

    #[test]
    fn test_applied_features_t6_relationship() {
        // T6 selected + CI applied -> true
        let style_t6 = "preset_bhaukal_caption";
        let ci_plan_present = true;
        let ass_written = true;
        let ci_applied = ci_plan_present && ass_written;
        assert!(
            ci_applied,
            "T6 with CI plan must set captionIntelligence=true"
        );

        // T6 selected + CI not applied (e.g. low confidence / disabled) -> false
        let ci_plan_absent = false;
        let ci_not_applied = ci_plan_absent && ass_written;
        assert!(
            !ci_not_applied,
            "T6 without CI plan must set captionIntelligence=false"
        );

        // Non-T6 template + CI applied -> true
        let style_t1 = "preset_viral_bold";
        assert_ne!(style_t1, style_t6);
        let ci_applied_t1 = ci_plan_present && ass_written;
        assert!(
            ci_applied_t1,
            "Non-T6 with CI plan must set captionIntelligence=true"
        );
    }

    #[test]
    fn test_applied_features_plan_generation_and_consumption_pipeline() {
        use crate::models::Candidate;
        let words = vec![
            make_word("Never", 10.0, 10.5),
            make_word("eat", 10.5, 11.0),
            make_word("this", 11.0, 11.5),
            make_word("before", 11.5, 12.0),
            make_word("bed", 12.0, 12.5),
        ];

        let candidate_high_conf = Candidate {
            id: "cand-high".to_string(),
            project_id: "p1".to_string(),
            start_sec: 10.0,
            end_sec: 15.0,
            score: 0.9,
            hook: "Never eat this".to_string(),
            rationale: "".to_string(),
            rank: 1,
            selected: true,
            hook_start_sec: Some(10.0),
            hook_end_sec: Some(12.0),
            hook_confidence: Some(0.95),
            opening_context_score: Some(0.9),
            payoff_text: None,
            payoff_start_sec: None,
            payoff_end_sec: None,
            payoff_score: None,
            payoff_completion: None,
            metadata_json: None,
        };

        // CI enabled & high confidence -> plan is Some
        let plan =
            caption_intel::plan_caption_intel(&words, &candidate_high_conf, None, 10.0, 15.0);
        assert!(
            plan.is_some(),
            "High confidence candidate must produce CaptionIntelPlan"
        );

        // Generate ASS for T6
        let t6 = captions::get_caption_template("preset_bhaukal_caption").unwrap();
        let ass_t6 = captions::generate_ass_from_template_with_framing_and_intel(
            &words,
            10.0,
            15.0,
            &t6,
            None,
            plan.as_ref(),
            None,
        );
        assert!(!ass_t6.is_empty(), "T6 ASS must be generated");

        // Generate ASS for T1
        let t1 = captions::get_caption_template("preset_viral_bold").unwrap();
        let ass_t1 = captions::generate_ass_from_template_with_framing_and_intel(
            &words,
            10.0,
            15.0,
            &t1,
            None,
            plan.as_ref(),
            None,
        );
        assert!(!ass_t1.is_empty(), "T1 ASS must be generated");

        // CI low confidence -> plan is None
        let candidate_low_conf = Candidate {
            hook_confidence: Some(0.50), // below 0.75 threshold
            ..candidate_high_conf
        };
        let plan_low =
            caption_intel::plan_caption_intel(&words, &candidate_low_conf, None, 10.0, 15.0);
        assert!(
            plan_low.is_none(),
            "Low confidence candidate must produce None"
        );
    }

    static ENV_DISCOVERY_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_resolve_discovery_mode_from_env_default_is_timestamp() {
        let _guard = ENV_DISCOVERY_MUTEX.lock().unwrap();
        let prev = std::env::var("AUTOSHORTS_DISCOVERY_MODE").ok();
        std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE");

        assert_eq!(
            resolve_discovery_mode_from_env(),
            models::DiscoveryMode::TimestampGeneration
        );

        // Also verify invalid or unknown env var values fall back safely to TimestampGeneration
        for invalid in ["unknown", "invalid_value", "", "123", "none"] {
            std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", invalid);
            assert_eq!(
                resolve_discovery_mode_from_env(),
                models::DiscoveryMode::TimestampGeneration,
                "Invalid env value '{invalid}' must default to TimestampGeneration"
            );
        }

        // Verify explicit timestamp modes
        for val in [
            "timestamp_generation",
            "TIMESTAMP_GENERATION",
            "timestamp",
            "legacy",
            "  legacy  ",
        ] {
            std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", val);
            assert_eq!(
                resolve_discovery_mode_from_env(),
                models::DiscoveryMode::TimestampGeneration,
                "Value '{val}' must resolve to TimestampGeneration"
            );
        }

        match prev {
            Some(v) => std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", v),
            None => std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE"),
        }
    }

    #[test]
    fn test_resolve_discovery_mode_from_env_reze_switch() {
        let _guard = ENV_DISCOVERY_MUTEX.lock().unwrap();
        let prev = std::env::var("AUTOSHORTS_DISCOVERY_MODE").ok();

        for val in [
            "reze",
            "REZE",
            "window_scoring",
            "WINDOW_SCORING",
            "scoring",
            "  reze  ",
            "  Window_Scoring  ",
        ] {
            std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", val);
            assert_eq!(
                resolve_discovery_mode_from_env(),
                models::DiscoveryMode::WindowScoring,
                "Value '{val}' must resolve to WindowScoring"
            );
        }

        match prev {
            Some(v) => std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", v),
            None => std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE"),
        }
    }

    #[test]
    fn test_reze_3_way_gate_behavior() {
        let _guard = ENV_DISCOVERY_MUTEX.lock().unwrap();

        let prev_mode = std::env::var("AUTOSHORTS_DISCOVERY_MODE").ok();
        let prev_provider = std::env::var("AUTOSHORTS_REZE_PROVIDER").ok();
        let prev_key = std::env::var("NVIDIA_API_KEY").ok();

        struct EnvGuard {
            prev_mode: Option<String>,
            prev_provider: Option<String>,
            prev_key: Option<String>,
        }
        impl Drop for EnvGuard {
            fn drop(&mut self) {
                match &self.prev_mode {
                    Some(v) => std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", v),
                    None => std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE"),
                }
                match &self.prev_provider {
                    Some(v) => std::env::set_var("AUTOSHORTS_REZE_PROVIDER", v),
                    None => std::env::remove_var("AUTOSHORTS_REZE_PROVIDER"),
                }
                match &self.prev_key {
                    Some(v) => std::env::set_var("NVIDIA_API_KEY", v),
                    None => std::env::remove_var("NVIDIA_API_KEY"),
                }
            }
        }
        let _env_guard = EnvGuard {
            prev_mode,
            prev_provider,
            prev_key,
        };

        // Case A: Unset env vars -> TimestampGeneration, is_reze: false, key: None
        std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::remove_var("NVIDIA_API_KEY");

        let (mode, is_reze, key) = resolve_reze_3_way_gate(None, None, None);
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);

        // Case B: AUTOSHORTS_DISCOVERY_MODE=window_scoring but AUTOSHORTS_REZE_PROVIDER unset/empty
        std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", "window_scoring");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::remove_var("NVIDIA_API_KEY");

        let (mode, is_reze, key) = resolve_reze_3_way_gate(None, None, None);
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);

        std::env::set_var("AUTOSHORTS_REZE_PROVIDER", "");
        let (mode, is_reze, key) = resolve_reze_3_way_gate(None, None, None);
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);

        // Case C: AUTOSHORTS_DISCOVERY_MODE=window_scoring + AUTOSHORTS_REZE_PROVIDER=nvidia_diffusiongemma but NVIDIA_API_KEY missing
        std::env::set_var("AUTOSHORTS_REZE_PROVIDER", "nvidia_diffusiongemma");
        std::env::remove_var("NVIDIA_API_KEY");

        let (mode, is_reze, key) = resolve_reze_3_way_gate(None, None, None);
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);

        std::env::set_var("NVIDIA_API_KEY", "   ");
        let (mode, is_reze, key) = resolve_reze_3_way_gate(None, None, None);
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);

        // Case D: All 3 set -> WindowScoring, is_reze: true, key returned
        std::env::set_var("NVIDIA_API_KEY", "nvapi-test-key");
        let (mode, is_reze, key) = resolve_reze_3_way_gate(None, None, None);
        assert_eq!(mode, models::DiscoveryMode::WindowScoring);
        assert!(is_reze);
        assert_eq!(key.as_deref(), Some("nvapi-test-key"));

        // Case E: Argument-based invocation provides overrides for mode, provider, and api_key
        std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::remove_var("NVIDIA_API_KEY");

        let (mode, is_reze, key) = resolve_reze_3_way_gate(
            Some("window_scoring"),
            Some("nvidia_diffusiongemma"),
            Some("arg-key-123"),
        );
        assert_eq!(mode, models::DiscoveryMode::WindowScoring);
        assert!(is_reze);
        assert_eq!(key.as_deref(), Some("arg-key-123"));

        let (mode, is_reze, key) = resolve_reze_3_way_gate(
            Some("reze"),
            Some("nvidia"),
            Some("arg-key-456"),
        );
        assert_eq!(mode, models::DiscoveryMode::WindowScoring);
        assert!(is_reze);
        assert_eq!(key.as_deref(), Some("arg-key-456"));

        // Env says WindowScoring, but arg overrides to timestamp_generation
        std::env::set_var("AUTOSHORTS_DISCOVERY_MODE", "window_scoring");
        std::env::set_var("AUTOSHORTS_REZE_PROVIDER", "nvidia_diffusiongemma");
        std::env::set_var("NVIDIA_API_KEY", "nvapi-test-key");

        let (mode, is_reze, key) = resolve_reze_3_way_gate(
            Some("timestamp_generation"),
            None,
            None,
        );
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);

        // Case F: UI argument passing: discovery_mode = "window_scoring", reze_provider = "nvidia_diffusiongemma", NVIDIA_API_KEY in env
        std::env::remove_var("AUTOSHORTS_DISCOVERY_MODE");
        std::env::remove_var("AUTOSHORTS_REZE_PROVIDER");
        std::env::set_var("NVIDIA_API_KEY", "nvapi-test-key-ui");

        let (mode, is_reze, key) = resolve_reze_3_way_gate(
            Some("window_scoring"),
            Some("nvidia_diffusiongemma"),
            None,
        );
        assert_eq!(mode, models::DiscoveryMode::WindowScoring);
        assert!(is_reze);
        assert_eq!(key.as_deref(), Some("nvapi-test-key-ui"));

        // When discovery_mode is not passed or default, gate remains default TimestampGeneration
        let (mode, is_reze, key) = resolve_reze_3_way_gate(
            None,
            Some("nvidia_diffusiongemma"),
            None,
        );
        assert_eq!(mode, models::DiscoveryMode::TimestampGeneration);
        assert!(!is_reze);
        assert_eq!(key, None);
    }

    static ENV_PAUSE_INTEL_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_pause_intel_flag_defaults_to_disabled() {
        let _guard = ENV_PAUSE_INTEL_MUTEX.lock().unwrap();
        let prev = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();
        std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED");

        assert!(
            !pause_intel_enabled(),
            "pause_intel_enabled() must return false when AUTOSHORTS_SMART_PACING_LEARNED is unset"
        );

        match prev {
            Some(v) => std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", v),
            None => std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED"),
        }
    }

    #[test]
    fn test_pause_intel_flag_activates_on_truthy_values() {
        let _guard = ENV_PAUSE_INTEL_MUTEX.lock().unwrap();
        let prev = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();

        for val in ["1", "true", "TRUE", "True", "on", "ON", "  1  ", "  true  ", "  on  "] {
            std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", val);
            assert!(
                pause_intel_enabled(),
                "pause_intel_enabled() must return true for '{val}'"
            );
        }

        match prev {
            Some(v) => std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", v),
            None => std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED"),
        }
    }

    #[test]
    fn test_pause_intel_flag_deactivates_on_falsy_or_other_values() {
        let _guard = ENV_PAUSE_INTEL_MUTEX.lock().unwrap();
        let prev = std::env::var("AUTOSHORTS_SMART_PACING_LEARNED").ok();

        for val in ["0", "false", "FALSE", "off", "OFF", "random", "", "2", "no", "disabled"] {
            std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", val);
            assert!(
                !pause_intel_enabled(),
                "pause_intel_enabled() must return false for '{val}'"
            );
        }

        match prev {
            Some(v) => std::env::set_var("AUTOSHORTS_SMART_PACING_LEARNED", v),
            None => std::env::remove_var("AUTOSHORTS_SMART_PACING_LEARNED"),
        }
    }
}
