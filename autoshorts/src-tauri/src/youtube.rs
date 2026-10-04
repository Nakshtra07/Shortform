use std::{
    path::{Path, PathBuf},
    process::Command,
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopyrightCheckResult {
    pub is_safe: bool,
    pub license: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum YoutubeAuthMode {
    Auto,
    Browser(String),
    CookieFile(PathBuf),
    None,
}

/// Normalizes YouTube video URLs, YouTube Shorts, mobile URLs, and shortlinks.
/// - https://www.youtube.com/watch?v=ywd-Ve8a8Tc
/// - https://youtu.be/ywd-Ve8a8Tc?si=xyz -> https://www.youtube.com/watch?v=ywd-Ve8a8Tc
/// - https://www.youtube.com/shorts/ywd-Ve8a8Tc -> https://www.youtube.com/watch?v=ywd-Ve8a8Tc
/// - https://m.youtube.com/watch?v=ywd-Ve8a8Tc -> https://www.youtube.com/watch?v=ywd-Ve8a8Tc
pub fn normalize_youtube_url(raw_url: &str) -> String {
    let trimmed = raw_url.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut video_id: Option<String> = None;

    if trimmed.contains("youtu.be/") {
        if let Some(after) = trimmed.split("youtu.be/").nth(1) {
            let id = after.split(&['?', '&', '/', '#'][..]).next().unwrap_or("");
            if !id.is_empty() {
                video_id = Some(id.to_string());
            }
        }
    } else if trimmed.contains("/shorts/") {
        if let Some(after) = trimmed.split("/shorts/").nth(1) {
            let id = after.split(&['?', '&', '/', '#'][..]).next().unwrap_or("");
            if !id.is_empty() {
                video_id = Some(id.to_string());
            }
        }
    } else if trimmed.contains("/embed/") {
        if let Some(after) = trimmed.split("/embed/").nth(1) {
            let id = after.split(&['?', '&', '/', '#'][..]).next().unwrap_or("");
            if !id.is_empty() {
                video_id = Some(id.to_string());
            }
        }
    } else if trimmed.contains("v=") {
        if let Some(after) = trimmed.split("v=").nth(1) {
            let id = after.split(&['&', '#', '/'][..]).next().unwrap_or("");
            if !id.is_empty() {
                video_id = Some(id.to_string());
            }
        }
    }

    if let Some(id) = video_id {
        format!("https://www.youtube.com/watch?v={}", id)
    } else {
        trimmed.to_string()
    }
}

/// Resolves user or environment authentication preferences.
pub fn resolve_auth_mode(
    browser_pref: Option<&str>,
    cookie_path_pref: Option<&str>,
) -> YoutubeAuthMode {
    // 1. Explicit cookie file path
    if let Some(path_str) = cookie_path_pref {
        let trimmed = path_str.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            if path.exists() {
                return YoutubeAuthMode::CookieFile(path);
            }
        }
    }
    if let Ok(env_cookies) = std::env::var("YTDLP_COOKIES_PATH") {
        let trimmed = env_cookies.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            if path.exists() {
                return YoutubeAuthMode::CookieFile(path);
            }
        }
    }

    // 2. Explicit browser preference
    let browser_raw = browser_pref
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("AUTOSHORTS_YOUTUBE_BROWSER").ok())
        .or_else(|| std::env::var("YTDLP_COOKIES_BROWSER").ok())
        .unwrap_or_else(|| "auto".to_string());

    match browser_raw.to_lowercase().as_str() {
        "none" | "unauthenticated" => YoutubeAuthMode::None,
        "auto" => YoutubeAuthMode::Auto,
        other => YoutubeAuthMode::Browser(other.to_string()),
    }
}

/// Determines whether stderr contains YouTube bot verification / sign-in challenges.
pub fn is_bot_verification_error(stderr: &str) -> bool {
    let lower = stderr.to_lowercase();
    lower.contains("sign in to confirm you’re not a bot")
        || lower.contains("sign in to confirm you're not a bot")
        || lower.contains("confirm you’re not a bot")
        || lower.contains("confirm you're not a bot")
        || lower.contains("sign in to confirm your age")
        || lower.contains("use --cookies-from-browser or --cookies")
        || lower.contains("bot detection")
        || lower.contains("http error 429")
}

/// Determines whether stderr indicates browser cookie database locking or DPAPI decryption failure.
pub fn is_cookie_lock_or_access_error(stderr: &str) -> (bool, Option<&'static str>) {
    let lower = stderr.to_lowercase();
    if lower.contains("could not copy chrome cookie database")
        || lower.contains("could not copy edge cookie database")
    {
        (true, Some("Browser database is currently locked by a running browser instance. Please completely close the browser and try again."))
    } else if lower.contains("failed to decrypt with dpapi") {
        (true, Some("Windows DPAPI cookie decryption failed for this browser. Please use Firefox, or export cookies to a cookies.txt file."))
    } else if lower.contains("unsupported browser specified") {
        (true, Some("The specified browser is not supported by yt-dlp. Supported browsers: chrome, firefox, edge, brave, opera, safari, vivaldi, chromium."))
    } else if lower.contains("could not extract cookies") {
        (true, Some("Could not extract YouTube cookies from this browser profile. Please ensure you are logged into YouTube in this browser."))
    } else if (lower.contains("could not find") || lower.contains("cannot find"))
        && lower.contains("cookies database")
    {
        (true, Some("Could not find browser cookie database. Please ensure the browser is installed and has a valid user profile."))
    } else {
        (false, None)
    }
}

/// Generates a structured, actionable error message for the user.
pub fn format_actionable_error(stderr: &str, attempted_auths: &[String]) -> String {
    let (is_lock, lock_hint) = is_cookie_lock_or_access_error(stderr);
    let is_bot = is_bot_verification_error(stderr);

    let mut msg = String::new();

    if is_bot {
        msg.push_str("YouTube Authentication / Bot Verification Required:\n");
        msg.push_str("YouTube is requiring sign-in verification for this video.\n\n");
        msg.push_str("Actionable Steps to Resolve:\n");
        msg.push_str(
            "1. Open your browser (e.g. Firefox, Chrome, Edge, or Brave) and log into YouTube.\n",
        );
        msg.push_str("2. In AutoShorts API Settings, select your browser under 'YouTube Authentication Browser'.\n");
        msg.push_str("3. If using Chrome or Edge on Windows, close the browser before downloading so yt-dlp can access cookies.\n");
        msg.push_str("4. Alternatively, use a Firefox profile or provide a 'cookies.txt' file in Settings.\n");
    } else if is_lock {
        msg.push_str("Browser Cookie Extraction Error:\n");
        if let Some(hint) = lock_hint {
            msg.push_str(&format!("{}\n\n", hint));
        }
        msg.push_str("Steps to Resolve:\n");
        msg.push_str(
            "1. Close all open windows of the selected browser to release the database lock.\n",
        );
        msg.push_str(
            "2. Or switch the YouTube Authentication Browser to 'Firefox' or 'Auto' in Settings.\n",
        );
    } else {
        msg.push_str("YouTube Download / Metadata Request Failed:\n");
        msg.push_str(stderr.trim());
    }

    if !attempted_auths.is_empty() {
        msg.push_str(&format!(
            "\n\n(Authentication attempts made: {})",
            attempted_auths.join(" -> ")
        ));
    }

    msg
}

/// Build yt-dlp arguments for authentication
fn build_auth_args(auth: &YoutubeAuthMode) -> Vec<String> {
    match auth {
        YoutubeAuthMode::Browser(b) => vec!["--cookies-from-browser".to_string(), b.clone()],
        YoutubeAuthMode::CookieFile(path) => {
            vec!["--cookies".to_string(), path.to_string_lossy().to_string()]
        }
        YoutubeAuthMode::Auto | YoutubeAuthMode::None => Vec::new(),
    }
}

/// Build yt-dlp arguments for network configuration.
/// By default, yt-dlp uses system-native IPv4/IPv6 resolution.
/// `--force-ipv4` is only added if explicitly requested via AUTOSHORTS_YOUTUBE_FORCE_IPV4=1,
/// because forcing IPv4 frequently triggers YouTube HTTP 429 rate-limiting and bot captchas on ISP IPv4 ranges.
pub fn build_network_args() -> Vec<String> {
    if let Ok(val) = std::env::var("AUTOSHORTS_YOUTUBE_FORCE_IPV4") {
        if matches!(val.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on") {
            return vec!["--force-ipv4".to_string()];
        }
    }
    Vec::new()
}

/// Build yt-dlp arguments for extractor settings.
/// Note: Do NOT hardcode "youtube:player_client=default,tv,web_embedded,mweb,android".
/// Forcing mweb (requires GVS PO Token) and android (SABR experiment) blocks DASH formats
/// and forces yt-dlp down to format 18 (360p progressive).
/// Modern yt-dlp automatically negotiates optimal clients (including visionos and web).
/// Custom extractor args can be supplied via AUTOSHORTS_YOUTUBE_EXTRACTOR_ARGS.
pub fn build_extractor_args() -> Vec<String> {
    if let Ok(val) = std::env::var("AUTOSHORTS_YOUTUBE_EXTRACTOR_ARGS") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return vec!["--extractor-args".to_string(), trimmed.to_string()];
        }
    }
    Vec::new()
}

/// Redact arguments for safe logging without leaking tokens or private file paths
fn sanitize_args_for_logging(args: &[String]) -> String {
    let mut sanitized = Vec::new();
    let mut skip_next = false;

    for (i, arg) in args.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--cookies" {
            sanitized.push("--cookies [REDACTED_COOKIE_PATH]".to_string());
            skip_next = true;
        } else if arg == "--cookies-from-browser" {
            if let Some(browser) = args.get(i + 1) {
                sanitized.push(format!("--cookies-from-browser {}", browser));
                skip_next = true;
            } else {
                sanitized.push(arg.clone());
            }
        } else {
            sanitized.push(arg.clone());
        }
    }

    sanitized.join(" ")
}

/// Determines the sequence of authentication strategies to try for a request.
/// - If `Auto`: tries Unauthenticated first. If bot verification is detected, tries available browser sessions (`firefox`, `chrome`, `edge`, `brave`) in a single clean pass.
/// - If explicit `Browser`: tries that browser directly.
/// - If explicit `CookieFile`: tries that cookie file directly.
/// - If `None`: tries unauthenticated only.
fn get_auth_strategy_chain(mode: &YoutubeAuthMode) -> Vec<YoutubeAuthMode> {
    match mode {
        YoutubeAuthMode::Auto => {
            vec![
                YoutubeAuthMode::None,
                YoutubeAuthMode::Browser("firefox".to_string()),
                YoutubeAuthMode::Browser("chrome".to_string()),
                YoutubeAuthMode::Browser("edge".to_string()),
                YoutubeAuthMode::Browser("brave".to_string()),
            ]
        }
        YoutubeAuthMode::Browser(b) => vec![YoutubeAuthMode::Browser(b.clone())],
        YoutubeAuthMode::CookieFile(p) => vec![YoutubeAuthMode::CookieFile(p.clone())],
        YoutubeAuthMode::None => vec![YoutubeAuthMode::None],
    }
}

/// Runs yt-dlp metadata probe to check copyright / license with authentication support.
pub fn check_copyright(
    url: &str,
    browser_pref: Option<&str>,
    cookie_path_pref: Option<&str>,
) -> Result<CopyrightCheckResult, String> {
    let ytdlp_path = crate::media::find_ytdlp_cmd().ok_or_else(|| {
        "yt-dlp executable was not found. Please ensure yt-dlp is installed and available on your PATH.".to_string()
    })?;

    let normalized_url = normalize_youtube_url(url);
    let auth_mode = resolve_auth_mode(browser_pref, cookie_path_pref);
    let strategy_chain = get_auth_strategy_chain(&auth_mode);

    let mut last_stderr = String::new();
    let mut attempted_auths: Vec<String> = Vec::new();

    for auth in &strategy_chain {
        let auth_label = match auth {
            YoutubeAuthMode::None => "Unauthenticated".to_string(),
            YoutubeAuthMode::Browser(b) => format!("Browser ({b})"),
            YoutubeAuthMode::CookieFile(_) => "CookieFile".to_string(),
            YoutubeAuthMode::Auto => "Auto".to_string(),
        };
        attempted_auths.push(auth_label.clone());

        let mut args: Vec<String> = vec![
            "--ignore-config".to_string(),
        ];
        args.extend(build_network_args());
        args.extend(build_extractor_args());
        args.extend(vec![
            "--dump-json".to_string(),
            "--no-playlist".to_string(),
        ]);
        args.extend(build_auth_args(auth));
        args.push(normalized_url.clone());

        println!(
            "[YouTube Copyright Check] Executing: {} {}",
            ytdlp_path.display(),
            sanitize_args_for_logging(&args)
        );

        let output = match Command::new(&ytdlp_path).args(&args).output() {
            Ok(out) => out,
            Err(e) => {
                return Err(format!(
                    "Failed to execute yt-dlp at {}: {}",
                    ytdlp_path.display(),
                    e
                ));
            }
        };

        if output.status.success() {
            let json_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                let license = parsed
                    .get("license")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let is_safe = if let Some(lic) = &license {
                    let lower = lic.to_lowercase();
                    lower.contains("creative commons") || lower.contains("reuse allowed")
                } else {
                    false
                };

                println!("[YouTube Copyright Check] Succeeded with auth: {auth_label}. License: {:?}, is_safe: {}", license, is_safe);
                return Ok(CopyrightCheckResult { is_safe, license });
            }
        }

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        last_stderr = stderr.clone();

        println!(
            "[YouTube Copyright Check] Auth '{}' failed (status {}). Stderr: {}",
            auth_label,
            output.status,
            stderr.trim()
        );

        // If this wasn't an auth / bot / lock issue (e.g. invalid URL or private video), don't keep cycling through browsers
        if !is_bot_verification_error(&stderr) && !is_cookie_lock_or_access_error(&stderr).0 {
            break;
        }
    }

    Err(format_actionable_error(&last_stderr, &attempted_auths))
}

/// Extracts YouTube video ID from normalized or raw URLs.
pub fn extract_video_id(raw_url: &str) -> Option<String> {
    let normalized = normalize_youtube_url(raw_url);
    if let Some(pos) = normalized.find("v=") {
        let id_part = &normalized[pos + 2..];
        let id = id_part.split(&['&', '#', '/'][..]).next().unwrap_or("");
        if !id.is_empty() {
            return Some(id.to_string());
        }
    }
    None
}

/// Parses video frame rate from ffprobe fraction strings (e.g. "30/1", "24000/1001").
pub fn parse_frame_rate(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() == 2 {
        let num: f64 = parts[0].trim().parse().ok()?;
        let den: f64 = parts[1].trim().parse().ok()?;
        if den > 0.0 {
            return Some(num / den);
        }
    } else if let Ok(val) = s.trim().parse::<f64>() {
        return Some(val);
    }
    None
}

/// Comprehensive verified quality metadata for a downloaded or cached YouTube video file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DownloadedQualityMetadata {
    pub source_url: String,
    pub video_id: Option<String>,
    pub file_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: Option<f64>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub duration_sec: Option<f64>,
    pub bitrate: Option<u64>,
    pub container: Option<String>,
    pub file_size_bytes: u64,
}

/// Probes media stream properties via ffprobe to rigorously verify actual downloaded quality.
pub fn probe_downloaded_quality(
    path: &Path,
    source_url: &str,
) -> Result<DownloadedQualityMetadata, String> {
    let ffprobe_bin = crate::media::find_ffprobe_cmd()
        .ok_or_else(|| "ffprobe is not installed or not available on PATH".to_string())?;

    let path_str = path.to_string_lossy().to_string();
    let mut cmd = Command::new(&ffprobe_bin);
    cmd.args([
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        &path_str,
    ]);

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute ffprobe at {}: {}", ffprobe_bin.display(), e))?;

    if !output.status.success() {
        return Err(format!(
            "ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse ffprobe JSON output: {}", e))?;

    let streams = json
        .get("streams")
        .and_then(|s| s.as_array())
        .ok_or_else(|| "ffprobe JSON missing 'streams' array".to_string())?;

    let video_stream = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("video"));

    let audio_stream = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|v| v.as_str()) == Some("audio"));

    let format_obj = json.get("format");

    let width = video_stream
        .and_then(|s| s.get("width"))
        .and_then(|w| w.as_u64())
        .unwrap_or(0) as u32;

    let height = video_stream
        .and_then(|s| s.get("height"))
        .and_then(|h| h.as_u64())
        .unwrap_or(0) as u32;

    let fps = video_stream.and_then(|s| {
        s.get("r_frame_rate")
            .and_then(|r| r.as_str())
            .and_then(parse_frame_rate)
            .or_else(|| {
                s.get("avg_frame_rate")
                    .and_then(|r| r.as_str())
                    .and_then(parse_frame_rate)
            })
    });

    let video_codec = video_stream
        .and_then(|s| s.get("codec_name"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());

    let audio_codec = audio_stream
        .and_then(|s| s.get("codec_name"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());

    let duration_sec = format_obj
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| {
            video_stream
                .and_then(|s| s.get("duration"))
                .and_then(|d| d.as_str())
                .and_then(|s| s.parse::<f64>().ok())
        });

    let bitrate = format_obj
        .and_then(|f| f.get("bit_rate"))
        .and_then(|b| b.as_str())
        .and_then(|s| s.parse::<u64>().ok())
        .or_else(|| {
            video_stream
                .and_then(|s| s.get("bit_rate"))
                .and_then(|b| b.as_str())
                .and_then(|s| s.parse::<u64>().ok())
        });

    let container = format_obj
        .and_then(|f| f.get("format_name"))
        .and_then(|fmt| fmt.as_str())
        .map(|fmt| {
            if fmt.contains("mp4") || fmt.contains("mov") {
                "mp4".to_string()
            } else if fmt.contains("matroska") {
                "mkv".to_string()
            } else if fmt.contains("webm") {
                "webm".to_string()
            } else {
                fmt.to_string()
            }
        })
        .or_else(|| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|s| s.to_string())
        });

    let file_size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let video_id = extract_video_id(source_url);

    Ok(DownloadedQualityMetadata {
        source_url: source_url.to_string(),
        video_id,
        file_path: path.to_path_buf(),
        width,
        height,
        fps,
        video_codec,
        audio_codec,
        duration_sec,
        bitrate,
        container,
        file_size_bytes,
    })
}

pub const QUALITY_CACHE_VERSION: &str = "autoshorts_best_quality_v2";

/// Metadata sidecar recording verified quality information for downloaded or cached files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredQualityMetadata {
    pub version: String,
    pub source_url: String,
    pub video_id: Option<String>,
    pub file_path: String,
    pub width: u32,
    pub height: u32,
    pub fps: Option<f64>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub duration_sec: Option<f64>,
    pub bitrate: Option<u64>,
    pub container: Option<String>,
    pub file_size_bytes: u64,
    pub tier_used: String,
    pub max_available_height: Option<u32>,
    pub downloaded_at: String,
}

pub fn quality_metadata_path(target_dir: &Path, vid: &str) -> PathBuf {
    target_dir.join(format!("AutoShorts_{}.quality.json", vid))
}

pub fn save_quality_metadata(
    meta: &DownloadedQualityMetadata,
    tier: FormatFallbackTier,
    max_avail_h: Option<u32>,
    target_dir: &Path,
) -> Result<PathBuf, String> {
    let vid = meta.video_id.as_deref().unwrap_or("unknown");
    let json_path = quality_metadata_path(target_dir, vid);
    let stored = StoredQualityMetadata {
        version: QUALITY_CACHE_VERSION.to_string(),
        source_url: meta.source_url.clone(),
        video_id: meta.video_id.clone(),
        file_path: meta.file_path.to_string_lossy().to_string(),
        width: meta.width,
        height: meta.height,
        fps: meta.fps,
        video_codec: meta.video_codec.clone(),
        audio_codec: meta.audio_codec.clone(),
        duration_sec: meta.duration_sec,
        bitrate: meta.bitrate,
        container: meta.container.clone(),
        file_size_bytes: meta.file_size_bytes,
        tier_used: tier.description().to_string(),
        max_available_height: max_avail_h,
        downloaded_at: chrono::Utc::now().to_rfc3339(),
    };
    let json_str = serde_json::to_string_pretty(&stored)
        .map_err(|e| format!("Failed to serialize quality metadata: {e}"))?;
    std::fs::write(&json_path, json_str)
        .map_err(|e| format!("Failed to write quality metadata to {}: {e}", json_path.display()))?;
    Ok(json_path)
}

pub fn read_quality_metadata(target_dir: &Path, vid: &str) -> Option<StoredQualityMetadata> {
    let json_path = quality_metadata_path(target_dir, vid);
    if json_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&json_path) {
            if let Ok(stored) = serde_json::from_str::<StoredQualityMetadata>(&content) {
                return Some(stored);
            }
        }
    }
    None
}

/// Discovered resolution capability from YouTube format metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvailableQuality {
    pub max_height: u32,
    pub max_width: u32,
    pub max_fps: Option<f64>,
    pub best_vcodec: Option<String>,
    pub best_acodec: Option<String>,
    pub format_id: Option<String>,
}

impl AvailableQuality {
    pub fn is_higher_than(&self, width: u32, height: u32) -> bool {
        if self.max_height == 0 && self.max_width == 0 {
            return false;
        }
        if width == 0 || height == 0 {
            return true;
        }
        let available_area = (self.max_width as u64) * (self.max_height as u64);
        let current_area = (width as u64) * (height as u64);
        if available_area > current_area {
            return true;
        }
        self.max_height > height || (self.max_width > width && self.max_height >= height)
    }
}

/// Fallback format selection tiers for robust highest-quality YouTube ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormatFallbackTier {
    /// Tier 1: Best separate video + best audio stream, losslessly remuxed into MP4.
    BestSeparateMp4,
    /// Tier 2: Best separate video + best audio stream, losslessly remuxed into MKV (resilient container).
    BestSeparateMkv,
    /// Tier 3: Best combined stream (single progressive stream, no merge needed).
    BestCombined,
}

impl FormatFallbackTier {
    pub fn all() -> &'static [FormatFallbackTier] {
        &[
            FormatFallbackTier::BestSeparateMp4,
            FormatFallbackTier::BestSeparateMkv,
            FormatFallbackTier::BestCombined,
        ]
    }

    pub fn format_selector(&self) -> &'static str {
        match self {
            FormatFallbackTier::BestSeparateMp4 | FormatFallbackTier::BestSeparateMkv => {
                "bestvideo+bestaudio/bestvideo*+bestaudio/bestvideo+bestaudio*/best"
            }
            FormatFallbackTier::BestCombined => "best/bestvideo+bestaudio",
        }
    }

    pub fn format_sort(&self) -> &'static str {
        "res,fps,codec,size,br"
    }

    pub fn merge_output_format(&self) -> Option<&'static str> {
        match self {
            FormatFallbackTier::BestSeparateMp4 => Some("mp4"),
            FormatFallbackTier::BestSeparateMkv => Some("mkv"),
            FormatFallbackTier::BestCombined => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            FormatFallbackTier::BestSeparateMp4 => {
                "Tier 1: Best available video + audio (lossless remux to MP4)"
            }
            FormatFallbackTier::BestSeparateMkv => {
                "Tier 2: Best available video + audio (lossless remux to MKV)"
            }
            FormatFallbackTier::BestCombined => {
                "Tier 3: Best available combined progressive stream"
            }
        }
    }
}

/// Determines whether a filename matches intermediate yt-dlp temporary or stream part patterns.
pub fn is_intermediate_or_part_file(name: &str) -> bool {
    let lower = name.to_lowercase();
    if lower.ends_with(".part")
        || lower.ends_with(".ytdl")
        || lower.contains(".temp.")
        || lower.ends_with(".temp")
    {
        return true;
    }
    // Check for intermediate stream part files like .f251.webm, .f313.mp4, .f137.mp4
    if let Some(pos) = lower.find(".f") {
        let after = &lower[pos + 2..];
        if let Some(dot_pos) = after.find('.') {
            let format_id_candidate = &after[..dot_pos];
            if !format_id_candidate.is_empty()
                && format_id_candidate.chars().all(|c| c.is_ascii_alphanumeric())
            {
                let ext = &after[dot_pos + 1..];
                if ["mp4", "webm", "m4a", "mkv"].contains(&ext) {
                    return true;
                }
            }
        }
    }
    false
}

/// Inspects parsed yt-dlp format metadata to determine the best available video quality.
/// Filters out storyboard/image placeholders and audio-only streams.
pub fn parse_available_quality_from_formats(formats: &[serde_json::Value]) -> Option<AvailableQuality> {
    let mut best_height = 0u32;
    let mut best_width = 0u32;
    let mut best_fps: Option<f64> = None;
    let mut best_vcodec: Option<String> = None;
    let mut best_acodec: Option<String> = None;
    let mut best_format_id: Option<String> = None;
    let mut best_area = 0u64;

    for f in formats {
        let ext = f.get("ext").and_then(|e| e.as_str()).unwrap_or("");
        let vcodec = f.get("vcodec").and_then(|v| v.as_str()).unwrap_or("");
        let note = f.get("format_note").and_then(|n| n.as_str()).unwrap_or("");

        if ext == "mhtml"
            || vcodec == "images"
            || vcodec == "none"
            || note.to_lowercase().contains("storyboard")
        {
            continue;
        }

        let height = f.get("height").and_then(|h| h.as_u64()).unwrap_or(0) as u32;
        let width = f.get("width").and_then(|w| w.as_u64()).unwrap_or(0) as u32;
        let area = (height as u64) * (width as u64);

        if area > best_area || (area == best_area && height > best_height) {
            best_area = area;
            best_height = height;
            best_width = width;
            best_fps = f.get("fps").and_then(|fps| fps.as_f64());
            best_vcodec = Some(vcodec.to_string());
            best_acodec = f.get("acodec").and_then(|a| a.as_str()).map(|s| s.to_string());
            best_format_id = f.get("format_id").and_then(|id| id.as_str()).map(|s| s.to_string());
        }
    }

    if best_height > 0 || best_width > 0 {
        Some(AvailableQuality {
            max_height: best_height,
            max_width: best_width,
            max_fps: best_fps,
            best_vcodec,
            best_acodec,
            format_id: best_format_id,
        })
    } else {
        None
    }
}

/// Inspects parsed yt-dlp format metadata to determine if higher video quality (> current_height) is available.
/// Returns Some(max_height) if higher quality exists, or None if current_height is already the highest available.
#[allow(dead_code)]
pub fn parse_higher_quality_from_formats(
    formats: &[serde_json::Value],
    current_height: u32,
) -> Option<u32> {
    let avail = parse_available_quality_from_formats(formats)?;
    if avail.max_height > current_height {
        Some(avail.max_height)
    } else {
        None
    }
}

/// Parses yt-dlp metadata JSON string to detect if higher video quality (> current_height) is available.
/// Returns Some(max_height) if higher quality exists, or None if current_height is already highest available or parsing fails.
#[allow(dead_code)]
pub fn parse_higher_quality_from_json(json_str: &str, current_height: u32) -> Option<u32> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let formats = parsed.get("formats")?.as_array()?;
    parse_higher_quality_from_formats(formats, current_height)
}

/// Queries yt-dlp metadata to determine available video quality using the configured authentication strategy chain.
pub fn query_available_quality(
    ytdlp_path: &Path,
    normalized_url: &str,
    auth_mode: &YoutubeAuthMode,
) -> Result<AvailableQuality, String> {
    let strategy_chain = get_auth_strategy_chain(auth_mode);
    let mut last_err = String::new();

    for auth in &strategy_chain {
        let mut args: Vec<String> = vec![
            "--ignore-config".to_string(),
        ];
        args.extend(build_network_args());
        args.extend(build_extractor_args());
        args.extend(vec![
            "--dump-json".to_string(),
            "--no-playlist".to_string(),
        ]);
        args.extend(build_auth_args(auth));
        args.push(normalized_url.to_string());

        let output = match Command::new(ytdlp_path).args(&args).output() {
            Ok(out) => out,
            Err(e) => return Err(format!("Failed to execute yt-dlp: {}", e)),
        };

        if output.status.success() {
            let json_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json_str) {
                if let Some(formats) = parsed.get("formats").and_then(|f| f.as_array()) {
                    if let Some(avail) = parse_available_quality_from_formats(formats) {
                        return Ok(avail);
                    }
                }
            }
        }
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        last_err = stderr.clone();

        if !is_bot_verification_error(&stderr) && !is_cookie_lock_or_access_error(&stderr).0 {
            break;
        }
    }

    Err(last_err)
}

/// Queries yt-dlp metadata to detect if higher video quality (> current_height) is available on YouTube.
/// Returns Some(max_height) if higher quality exists, or None if current_height is already highest available or query fails.
#[allow(dead_code)]
pub fn detect_higher_quality_available(
    ytdlp_path: &Path,
    normalized_url: &str,
    auth: &YoutubeAuthMode,
    current_height: u32,
) -> Option<u32> {
    let avail = query_available_quality(ytdlp_path, normalized_url, auth).ok()?;
    if avail.max_height > current_height {
        Some(avail.max_height)
    } else {
        None
    }
}

/// Runs yt-dlp to download the YouTube video with authenticated browser cookies and robust post-download verification.
/// Reliably selects the highest available resolution (including 1080p, 1440p, 4K/2160p+) with best audio.
pub fn download_video(
    url: &str,
    target_dir: &Path,
    browser_pref: Option<&str>,
    cookie_path_pref: Option<&str>,
) -> Result<PathBuf, String> {
    let ytdlp_path = crate::media::find_ytdlp_cmd().ok_or_else(|| {
        "yt-dlp executable was not found. Please ensure yt-dlp is installed and available on your PATH.".to_string()
    })?;

    std::fs::create_dir_all(target_dir)
        .map_err(|e| format!("Failed to create download target dir: {e}"))?;

    let normalized_url = normalize_youtube_url(url);
    let video_id = extract_video_id(&normalized_url);
    let auth_mode = resolve_auth_mode(browser_pref, cookie_path_pref);
    let strategy_chain = get_auth_strategy_chain(&auth_mode);

    // 0. Cache Inspection & Invalidation:
    // Check if an existing file for this video is present in target_dir.
    // Verify quality metadata or probe media to ensure cached file satisfies highest available quality.
    if let Some(ref vid) = video_id {
        let possible_cached = [
            target_dir.join(format!("AutoShorts_{}.mp4", vid)),
            target_dir.join(format!("AutoShorts_{}.mkv", vid)),
            target_dir.join(format!("AutoShorts_{}.webm", vid)),
        ];

        for candidate in &possible_cached {
            if candidate.exists() && candidate.metadata().map(|m| m.len() > 0).unwrap_or(false) {
                if let Ok(cached_meta) = probe_downloaded_quality(candidate, &normalized_url) {
                    if cached_meta.height > 0 && cached_meta.duration_sec.unwrap_or(0.0) > 0.0 {
                        let sidecar = read_quality_metadata(target_dir, vid);
                        let is_v2_verified = sidecar.as_ref()
                            .map(|s| s.version == QUALITY_CACHE_VERSION && s.width == cached_meta.width && s.height == cached_meta.height)
                            .unwrap_or(false);

                        // Check if higher quality is available on YouTube
                        let higher = query_available_quality(&ytdlp_path, &normalized_url, &auth_mode).ok();

                        if let Some(ref avail) = higher {
                            if avail.is_higher_than(cached_meta.width, cached_meta.height) {
                                println!(
                                    "[YouTube Cache] Existing file {} ({}x{}) is lower quality than available ({}x{}). Invalidating cache to download highest quality.",
                                    candidate.display(),
                                    cached_meta.width,
                                    cached_meta.height,
                                    avail.max_width,
                                    avail.max_height
                                );
                                let _ = std::fs::remove_file(candidate);
                                let _ = std::fs::remove_file(quality_metadata_path(target_dir, vid));
                                continue;
                            }
                        }

                        // Not invalidated: valid cache hit!
                        println!(
                            "[YouTube Cache] Valid cache hit: Existing file {} already satisfies highest available resolution ({}x{}). Reusing.",
                            candidate.display(),
                            cached_meta.width,
                            cached_meta.height
                        );
                        if !is_v2_verified {
                            let _ = save_quality_metadata(
                                &cached_meta,
                                FormatFallbackTier::BestSeparateMp4,
                                higher.as_ref().map(|a| a.max_height),
                                target_dir,
                            );
                        }
                        return Ok(candidate.clone());
                    }
                }
            }
        }
    }

    let output_template = target_dir.join("AutoShorts_%(id)s.%(ext)s");
    let output_template_str = output_template.to_string_lossy().to_string();

    let mut last_stderr = String::new();
    let mut attempted_auths: Vec<String> = Vec::new();

    // Query available quality once upfront if possible, to know target resolution
    let mut available_target = query_available_quality(&ytdlp_path, &normalized_url, &auth_mode).ok();

    for auth in &strategy_chain {
        let auth_label = match auth {
            YoutubeAuthMode::None => "Unauthenticated".to_string(),
            YoutubeAuthMode::Browser(b) => format!("Browser ({b})"),
            YoutubeAuthMode::CookieFile(_) => "CookieFile".to_string(),
            YoutubeAuthMode::Auto => "Auto".to_string(),
        };
        attempted_auths.push(auth_label.clone());

        let mut auth_failed_with_challenge = false;
        let mut fallback_candidate: Option<(PathBuf, DownloadedQualityMetadata, FormatFallbackTier)> = None;

        for tier in FormatFallbackTier::all() {
            println!(
                "[YouTube Download] Attempting auth: '{}', Tier: {}",
                auth_label,
                tier.description()
            );

            let mut args: Vec<String> = vec![
                "--ignore-config".to_string(),
            ];
            args.extend(build_network_args());
            args.extend(build_extractor_args());
            args.extend(vec![
                "--format".to_string(),
                tier.format_selector().to_string(),
                "--format-sort".to_string(),
                tier.format_sort().to_string(),
            ]);

            if let Some(merge_fmt) = tier.merge_output_format() {
                args.push("--merge-output-format".to_string());
                args.push(merge_fmt.to_string());
            }

            args.extend(vec![
                "-o".to_string(),
                output_template_str.clone(),
                "--print".to_string(),
                "after_move:filepath".to_string(),
                "--print".to_string(),
                "after_move:FORMAT_META:%(vcodec)s|%(resolution)s|%(fps)s|%(acodec)s|%(format_id)s|%(ext)s".to_string(),
                "--no-simulate".to_string(),
                "--no-playlist".to_string(),
            ]);

            if let Some(ffmpeg_path) = crate::media::find_ffmpeg_cmd() {
                if ffmpeg_path.is_absolute() && ffmpeg_path.exists() {
                    args.push("--ffmpeg-location".to_string());
                    args.push(ffmpeg_path.to_string_lossy().to_string());
                }
            }

            args.extend(build_auth_args(auth));
            args.push(normalized_url.clone());

            println!(
                "[YouTube Download] Executing: {} {}",
                ytdlp_path.display(),
                sanitize_args_for_logging(&args)
            );

            let output = match Command::new(&ytdlp_path).args(&args).output() {
                Ok(out) => out,
                Err(e) => {
                    return Err(format!(
                        "Failed to execute yt-dlp binary at {}: {}",
                        ytdlp_path.display(),
                        e
                    ));
                }
            };

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            last_stderr = stderr.to_string();

            if output.status.success() {
                let mut candidates: Vec<PathBuf> = Vec::new();
                let mut format_meta_line: Option<String> = None;

                for line in stdout.lines().rev() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("FORMAT_META:") {
                        if format_meta_line.is_none() {
                            format_meta_line =
                                Some(trimmed.trim_start_matches("FORMAT_META:").to_string());
                        }
                    } else if !trimmed.is_empty() {
                        let p = PathBuf::from(trimmed);
                        if p.exists() && !candidates.contains(&p) {
                            candidates.push(p);
                        }
                    }
                }

                // Canonical expected targets strictly for this video ID
                if let Some(ref vid) = video_id {
                    for ext in &["mp4", "mkv", "webm"] {
                        let p = target_dir.join(format!("AutoShorts_{}.{}", vid, ext));
                        if p.exists() && !candidates.contains(&p) {
                            candidates.push(p);
                        }
                    }
                }

                // Filter and probe candidate:
                // Must exist, non-empty, strictly match video ID (if known), not be intermediate part file,
                // and ffprobe probe must confirm valid video track (height > 0)
                let mut tier_selection: Option<(PathBuf, DownloadedQualityMetadata)> = None;

                for cand in candidates {
                    if !cand.exists() {
                        continue;
                    }
                    if let Some(ref vid) = video_id {
                        let fname = cand.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        let expected = format!("AutoShorts_{}", vid);
                        if !fname.starts_with(&expected) {
                            continue;
                        }
                    }
                    if let Some(fname) = cand.file_name().and_then(|n| n.to_str()) {
                        if is_intermediate_or_part_file(fname) {
                            continue;
                        }
                    }
                    let file_size = std::fs::metadata(&cand).map(|m| m.len()).unwrap_or(0);
                    if file_size == 0 {
                        continue;
                    }

                    if let Ok(meta) = probe_downloaded_quality(&cand, &normalized_url) {
                        if meta.height > 0 && meta.video_codec.is_some() {
                            tier_selection = Some((cand, meta));
                            break;
                        }
                    }
                }

                if let Some((downloaded_file, meta)) = tier_selection {
                    // If available_target was not resolved upfront, try querying quality with the current working auth
                    if available_target.is_none() {
                        available_target = query_available_quality(&ytdlp_path, &normalized_url, auth).ok();
                    }

                    // Check if downloaded stream achieves the available highest quality
                    let higher_exists = if let Some(ref avail) = available_target {
                        avail.is_higher_than(meta.width, meta.height)
                    } else {
                        false
                    };

                    if (higher_exists || (meta.height <= 480 && available_target.as_ref().map(|a| a.max_height >= 720).unwrap_or(false)))
                        && *tier == FormatFallbackTier::BestSeparateMp4
                    {
                        // Tier 1 (MP4) produced a lower resolution than available (e.g. 4K/1080p VP9/AV1 couldn't remux into MP4).
                        // Keep Tier 1 as fallback_candidate and attempt Tier 2 (MKV).
                        println!(
                            "[YouTube Download] Tier 1 MP4 downloaded {}x{} (meta: {:?}), but higher resolution is available ({:?}). Trying Tier 2 MKV for full resolution...",
                            meta.width, meta.height, format_meta_line, available_target.as_ref().map(|a| a.max_height)
                        );
                        fallback_candidate = Some((downloaded_file, meta, *tier));
                        continue;
                    }

                    // Clean up earlier lower fallback candidate if present and this tier is better or equal
                    if let Some((old_file, old_meta, _)) = fallback_candidate.take() {
                        if meta.height >= old_meta.height && downloaded_file != old_file {
                            let _ = std::fs::remove_file(&old_file);
                        }
                    }

                    println!(
                        "[YouTube Download] SUCCESS with auth '{}' (Tier: {})!\n  Path: {}\n  Resolution: {}x{}\n  FPS: {:?}\n  Bitrate: {:?}\n  Size: {:.2} MB\n  Duration: {:.1}s\n  Container: {:?}\n  Video Codec: {:?}\n  Audio Codec: {:?}\n  Raw Format Meta: {:?}",
                        auth_label,
                        tier.description(),
                        downloaded_file.display(),
                        meta.width,
                        meta.height,
                        meta.fps,
                        meta.bitrate,
                        meta.file_size_bytes as f64 / (1024.0 * 1024.0),
                        meta.duration_sec.unwrap_or(0.0),
                        meta.container,
                        meta.video_codec,
                        meta.audio_codec,
                        format_meta_line
                    );

                    let _ = save_quality_metadata(
                        &meta,
                        *tier,
                        available_target.as_ref().map(|a| a.max_height),
                        target_dir,
                    );
                    return Ok(downloaded_file);
                }
            } else {
                println!(
                    "[YouTube Download] Auth '{}' Tier '{}' failed (status {}). Stderr: {}",
                    auth_label,
                    tier.description(),
                    output.status,
                    stderr.trim()
                );

                if is_bot_verification_error(&stderr) || is_cookie_lock_or_access_error(&stderr).0 {
                    auth_failed_with_challenge = true;
                    break;
                }
            }
        }

        // If we had a fallback candidate from an earlier tier (e.g. Tier 1 succeeded at 1080p, but Tier 2 failed),
        // preserve and return that fallback candidate rather than failing!
        if let Some((fallback_file, fallback_meta, fallback_tier)) = fallback_candidate {
            println!(
                "[YouTube Download] Preserving best achievable fallback {}x{} (Tier: {})",
                fallback_meta.width, fallback_meta.height, fallback_tier.description()
            );
            let _ = save_quality_metadata(
                &fallback_meta,
                fallback_tier,
                available_target.as_ref().map(|a| a.max_height),
                target_dir,
            );
            return Ok(fallback_file);
        }

        if !auth_failed_with_challenge
            && !is_bot_verification_error(&last_stderr)
            && !is_cookie_lock_or_access_error(&last_stderr).0
        {
            break;
        }
    }

    Err(format_actionable_error(&last_stderr, &attempted_auths))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Heavy live-network E2E (downloads a full video, ~264 MB). Run explicitly: cargo test --lib test_real_youtube_download_e2e_venv_ytdlp -- --ignored --nocapture
    fn test_real_youtube_download_e2e_venv_ytdlp() {
        // E2E: exercises the full download path (find_ytdlp_cmd discovery ->
        // normalize -> auth chain -> yt-dlp execution -> file verification).
        // Uses a short Creative Commons video to keep the download small.
        let url = "https://www.youtube.com/watch?v=ywd-Ve8a8Tc";
        let target_dir = std::env::temp_dir().join("autoshorts_ytdlp_e2e_test");
        let _ = std::fs::remove_dir_all(&target_dir);
        std::fs::create_dir_all(&target_dir).expect("must create temp download dir");

        let res = download_video(url, &target_dir, Some("auto"), None);
        assert!(res.is_ok(), "download_video failed: {:?}", res.err());

        let downloaded = res.unwrap();
        assert!(
            downloaded.exists(),
            "Downloaded file must exist: {}",
            downloaded.display()
        );
        let size = std::fs::metadata(&downloaded).map(|m| m.len()).unwrap_or(0);
        assert!(size > 0, "Downloaded file must be non-empty");
        println!(
            "[E2E YouTube Download] OK: {} ({} bytes)",
            downloaded.display(),
            size
        );

        let _ = std::fs::remove_dir_all(&target_dir);
    }

    #[test]
    fn test_normalize_youtube_url() {
        assert_eq!(
            normalize_youtube_url("https://www.youtube.com/watch?v=ywd-Ve8a8Tc"),
            "https://www.youtube.com/watch?v=ywd-Ve8a8Tc"
        );
        assert_eq!(
            normalize_youtube_url("https://youtu.be/ywd-Ve8a8Tc?si=abc123xyz"),
            "https://www.youtube.com/watch?v=ywd-Ve8a8Tc"
        );
        assert_eq!(
            normalize_youtube_url("https://www.youtube.com/shorts/ywd-Ve8a8Tc"),
            "https://www.youtube.com/watch?v=ywd-Ve8a8Tc"
        );
        assert_eq!(
            normalize_youtube_url("https://m.youtube.com/watch?v=ywd-Ve8a8Tc&feature=shared"),
            "https://www.youtube.com/watch?v=ywd-Ve8a8Tc"
        );
    }

    #[test]
    fn test_is_bot_verification_error() {
        assert!(is_bot_verification_error("ERROR: [youtube] ywd-Ve8a8Tc: Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies"));
        assert!(is_bot_verification_error(
            "ERROR: [youtube] 12345: Sign in to confirm you're not a bot."
        ));
        assert!(is_bot_verification_error(
            "ERROR: [youtube] 12345: Sign in to confirm your age"
        ));
        assert!(!is_bot_verification_error(
            "ERROR: [youtube] 12345: Video unavailable"
        ));
    }

    #[test]
    fn test_is_cookie_lock_or_access_error() {
        let (locked, hint) =
            is_cookie_lock_or_access_error("ERROR: Could not copy Chrome cookie database.");
        assert!(locked);
        assert!(hint.is_some());

        let (dpapi, hint2) = is_cookie_lock_or_access_error("ERROR: Failed to decrypt with DPAPI. See https://github.com/yt-dlp/yt-dlp/issues/10927");
        assert!(dpapi);
        assert!(hint2.is_some());

        let (missing1, hint3) =
            is_cookie_lock_or_access_error("ERROR: could not find Chrome cookies database");
        assert!(missing1);
        assert!(hint3.is_some());

        let (missing2, hint4) = is_cookie_lock_or_access_error(
            "ERROR: cannot find firefox cookies database in profile",
        );
        assert!(missing2);
        assert!(hint4.is_some());
    }

    #[test]
    fn test_resolve_auth_mode() {
        assert_eq!(
            resolve_auth_mode(Some("chrome"), None),
            YoutubeAuthMode::Browser("chrome".to_string())
        );
        assert_eq!(resolve_auth_mode(Some("none"), None), YoutubeAuthMode::None);
        assert_eq!(resolve_auth_mode(Some("auto"), None), YoutubeAuthMode::Auto);
    }

    #[test]
    fn test_format_actionable_error() {
        let err = format_actionable_error(
            "ERROR: [youtube] ywd-Ve8a8Tc: Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies",
            &["Unauthenticated".to_string(), "Browser (firefox)".to_string()],
        );
        assert!(err.contains("YouTube Authentication / Bot Verification Required"));
        assert!(err.contains("Actionable Steps to Resolve"));
        assert!(err.contains("Authentication attempts made"));
    }

    #[test]
    fn test_real_youtube_metadata_probe_on_previously_failing_url() {
        // Test copyright check on the previously failing URL
        let url = "https://www.youtube.com/watch?v=ywd-Ve8a8Tc";
        let res = check_copyright(url, Some("auto"), None);
        assert!(
            res.is_ok(),
            "check_copyright on {} failed: {:?}",
            url,
            res.err()
        );
    }

    #[test]
    fn test_query_available_quality_discovers_1080p_on_reported_url() {
        let ytdlp_path = match crate::media::find_ytdlp_cmd() {
            Some(p) => p,
            None => return,
        };
        let url = "https://www.youtube.com/watch?v=1nvcZaD8vQk";
        let avail = query_available_quality(&ytdlp_path, url, &YoutubeAuthMode::Auto)
            .expect("query_available_quality must succeed on 1nvcZaD8vQk");
        println!(
            "[Available Quality for 1nvcZaD8vQk] {}x{} (format_id: {:?})",
            avail.max_width, avail.max_height, avail.format_id
        );
        assert!(
            avail.max_height >= 1080,
            "Expected 1080p or higher available for 1nvcZaD8vQk, got {}",
            avail.max_height
        );
    }

    #[test]
    fn test_format_selection_string_integrity() {
        let tier1 = FormatFallbackTier::BestSeparateMp4;
        assert!(tier1.format_selector().contains("bestvideo+bestaudio"));
        assert!(tier1.format_selector().contains("best"));
    }

    #[test]
    fn test_format_sort_string_integrity() {
        let tier1 = FormatFallbackTier::BestSeparateMp4;
        assert!(tier1.format_sort().contains("res"));
        assert!(tier1.format_sort().contains("fps"));
        assert!(tier1.format_sort().contains("codec"));
        assert!(tier1.format_sort().contains("size"));
        assert!(tier1.format_sort().contains("br"));
    }

    #[test]
    fn test_extractor_and_network_args_defaults() {
        let ext_args = build_extractor_args();
        assert!(
            ext_args.is_empty() || !ext_args.iter().any(|a| a.contains("mweb") || a.contains("android")),
            "Default extractor args must not force mweb/android which drop DASH streams"
        );

        let net_args = build_network_args();
        assert!(
            !net_args.contains(&"--force-ipv4".to_string()) || std::env::var("AUTOSHORTS_YOUTUBE_FORCE_IPV4").is_ok(),
            "Default network args should not unconditionally force IPv4"
        );
    }

    #[test]
    fn test_network_and_extractor_env_overrides() {
        std::env::set_var("AUTOSHORTS_YOUTUBE_FORCE_IPV4", "1");
        assert_eq!(build_network_args(), vec!["--force-ipv4".to_string()]);
        std::env::remove_var("AUTOSHORTS_YOUTUBE_FORCE_IPV4");
        assert!(build_network_args().is_empty());

        std::env::set_var("AUTOSHORTS_YOUTUBE_EXTRACTOR_ARGS", "youtube:player_client=ios");
        assert_eq!(
            build_extractor_args(),
            vec!["--extractor-args".to_string(), "youtube:player_client=ios".to_string()]
        );
        std::env::remove_var("AUTOSHORTS_YOUTUBE_EXTRACTOR_ARGS");
        assert!(build_extractor_args().is_empty());
    }

    #[test]
    fn test_detect_higher_quality_on_regression_url() {
        let regression_metadata = r#"{
            "id": "kbKldiDOgEE",
            "title": "Regression Invariant Test",
            "formats": [
                {"format_id": "18", "ext": "mp4", "height": 360, "width": 640},
                {"format_id": "134", "ext": "mp4", "height": 360, "width": 640},
                {"format_id": "135", "ext": "mp4", "height": 480, "width": 854},
                {"format_id": "136", "ext": "mp4", "height": 720, "width": 1280},
                {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920}
            ]
        }"#;

        let higher = parse_higher_quality_from_json(regression_metadata, 360);
        assert!(
            higher.is_some(),
            "detect_higher_quality_available should find higher than 360p on kbKldiDOgEE"
        );
        assert!(
            higher.unwrap() >= 720,
            "Higher quality should be at least 720p or 1080p, got {:?}",
            higher
        );
        assert_eq!(
            higher.unwrap(),
            1080,
            "Expected max available height to be 1080"
        );

        assert_eq!(
            parse_higher_quality_from_json(regression_metadata, 1080),
            None
        );

        let low_res_metadata = r#"{
            "id": "kbKldiDOgEE_low_only",
            "formats": [
                {"format_id": "18", "ext": "mp4", "height": 360, "width": 640}
            ]
        }"#;
        assert_eq!(parse_higher_quality_from_json(low_res_metadata, 360), None);
    }

    #[test]
    fn test_extract_video_id() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=ywd-Ve8a8Tc"),
            Some("ywd-Ve8a8Tc".to_string())
        );
        assert_eq!(
            extract_video_id("https://youtu.be/QhcY-XE0byw?si=123"),
            Some("QhcY-XE0byw".to_string())
        );
        assert_eq!(
            extract_video_id("https://www.youtube.com/shorts/w6uX9jamcwQ"),
            Some("w6uX9jamcwQ".to_string())
        );
        assert_eq!(
            extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        assert_eq!(extract_video_id("https://example.com/video.mp4"), None);
    }

    #[test]
    fn test_parse_frame_rate() {
        assert_eq!(parse_frame_rate("30/1"), Some(30.0));
        assert_eq!(parse_frame_rate("25/1"), Some(25.0));
        assert_eq!(parse_frame_rate("60/1"), Some(60.0));
        let fps24 = parse_frame_rate("24000/1001").unwrap();
        assert!((fps24 - 23.976).abs() < 0.01);
        assert_eq!(parse_frame_rate("29.97"), Some(29.97));
        assert_eq!(parse_frame_rate("0/0"), None);
        assert_eq!(parse_frame_rate("invalid"), None);
    }

    #[test]
    fn test_fallback_tiers_properties() {
        let tiers = FormatFallbackTier::all();
        assert_eq!(tiers.len(), 3);

        // Tier 1
        assert_eq!(tiers[0], FormatFallbackTier::BestSeparateMp4);
        assert!(tiers[0].format_selector().contains("bestvideo+bestaudio"));
        assert!(tiers[0].format_sort().contains("res,fps"));
        assert_eq!(tiers[0].merge_output_format(), Some("mp4"));

        // Tier 2
        assert_eq!(tiers[1], FormatFallbackTier::BestSeparateMkv);
        assert!(tiers[1].format_selector().contains("bestvideo+bestaudio"));
        assert_eq!(tiers[1].merge_output_format(), Some("mkv"));

        // Tier 3
        assert_eq!(tiers[2], FormatFallbackTier::BestCombined);
        assert!(tiers[2].format_selector().contains("best"));
        assert_eq!(tiers[2].merge_output_format(), None);
    }

    #[test]
    fn test_detect_higher_quality_1080p_maximum_fixture() {
        let max_1080p_metadata = r#"{
            "id": "vid_1080p_max",
            "title": "1080p Max Video",
            "formats": [
                {"format_id": "18", "ext": "mp4", "height": 360, "width": 640, "vcodec": "avc1"},
                {"format_id": "136", "ext": "mp4", "height": 720, "width": 1280, "vcodec": "avc1"},
                {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1"}
            ]
        }"#;

        assert_eq!(parse_higher_quality_from_json(max_1080p_metadata, 720), Some(1080));
        assert_eq!(parse_higher_quality_from_json(max_1080p_metadata, 1080), None);
    }

    #[test]
    fn test_detect_higher_quality_1440p_max_fixture() {
        let max_1440p_metadata = r#"{
            "id": "QhcY-XE0byw",
            "title": "1440p Max Video",
            "formats": [
                {"format_id": "136", "ext": "mp4", "height": 720, "width": 1280, "vcodec": "avc1"},
                {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1"},
                {"format_id": "271", "ext": "webm", "height": 1440, "width": 2560, "vcodec": "vp9"}
            ]
        }"#;

        assert_eq!(parse_higher_quality_from_json(max_1440p_metadata, 1080), Some(1440));
        assert_eq!(parse_higher_quality_from_json(max_1440p_metadata, 1440), None);
    }

    #[test]
    fn test_detect_higher_quality_4k_and_1440p_fixtures() {
        let multi_res_metadata = r#"{
            "id": "multi_res_4k",
            "title": "4K and 1440p Test Video",
            "formats": [
                {"format_id": "18", "ext": "mp4", "height": 360, "width": 640, "vcodec": "avc1.42001E"},
                {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1.640028"},
                {"format_id": "271", "ext": "webm", "height": 1440, "width": 2560, "vcodec": "vp9"},
                {"format_id": "313", "ext": "webm", "height": 2160, "width": 3840, "vcodec": "vp9"},
                {"format_id": "sb0", "ext": "mhtml", "height": 180, "width": 320, "vcodec": "images"},
                {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2"}
            ]
        }"#;

        // When current is 720p -> detects up to 2160p
        let higher_from_720 = parse_higher_quality_from_json(multi_res_metadata, 720);
        assert_eq!(higher_from_720, Some(2160));

        // When current is 1080p -> detects up to 2160p
        let higher_from_1080 = parse_higher_quality_from_json(multi_res_metadata, 1080);
        assert_eq!(higher_from_1080, Some(2160));

        // When current is 1440p -> detects up to 2160p
        let higher_from_1440 = parse_higher_quality_from_json(multi_res_metadata, 1440);
        assert_eq!(higher_from_1440, Some(2160));

        // When current is 2160p -> already at max quality, returns None
        let higher_from_2160 = parse_higher_quality_from_json(multi_res_metadata, 2160);
        assert_eq!(higher_from_2160, None);
    }

    #[test]
    fn test_separate_video_and_audio_streams_selection_fixture() {
        let fixture = r#"{
            "id": "sep_test",
            "formats": [
                {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1.640028", "acodec": "none"},
                {"format_id": "248", "ext": "webm", "height": 1080, "width": 1920, "vcodec": "vp9", "acodec": "none"},
                {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2", "abr": 128},
                {"format_id": "251", "ext": "webm", "vcodec": "none", "acodec": "opus", "abr": 160}
            ]
        }"#;

        let parsed: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let formats = parsed.get("formats").unwrap().as_array().unwrap();
        let avail = parse_available_quality_from_formats(formats).unwrap();

        assert_eq!(avail.max_height, 1080);
        assert_eq!(avail.max_width, 1920);
        assert!(avail.best_vcodec.is_some());
    }

    #[test]
    fn test_multiple_codecs_resolution_and_codec_preference_fixture() {
        let multi_codec_json = r#"{
            "id": "multi_codec",
            "formats": [
                {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "vcodec": "avc1.640028"},
                {"format_id": "271", "ext": "webm", "height": 1440, "width": 2560, "vcodec": "vp9"},
                {"format_id": "401", "ext": "mp4", "height": 2160, "width": 3840, "vcodec": "av01.0.12M.08"}
            ]
        }"#;

        let parsed: serde_json::Value = serde_json::from_str(multi_codec_json).unwrap();
        let formats = parsed.get("formats").unwrap().as_array().unwrap();
        let avail = parse_available_quality_from_formats(formats).unwrap();

        assert_eq!(avail.max_height, 2160);
        assert_eq!(avail.max_width, 3840);
        assert_eq!(avail.best_vcodec.as_deref(), Some("av01.0.12M.08"));
    }

    #[test]
    fn test_no_suitable_separate_streams_tier3_combined_fallback() {
        let combined_only_json = r#"{
            "id": "combined_only",
            "formats": [
                {"format_id": "18", "ext": "mp4", "height": 360, "width": 640, "vcodec": "avc1", "acodec": "mp4a.40.2"},
                {"format_id": "22", "ext": "mp4", "height": 720, "width": 1280, "vcodec": "avc1", "acodec": "mp4a.40.2"}
            ]
        }"#;

        let parsed: serde_json::Value = serde_json::from_str(combined_only_json).unwrap();
        let formats = parsed.get("formats").unwrap().as_array().unwrap();
        let avail = parse_available_quality_from_formats(formats).unwrap();

        assert_eq!(avail.max_height, 720);
        assert_eq!(avail.max_width, 1280);

        // Verify Tier 3 combined format selector accepts progressive stream
        let tier3 = FormatFallbackTier::BestCombined;
        assert_eq!(tier3.merge_output_format(), None);
        assert!(tier3.format_selector().contains("best"));
    }

    #[test]
    fn test_fallback_format_required_tier_progression() {
        let tiers = FormatFallbackTier::all();
        assert_eq!(tiers[0], FormatFallbackTier::BestSeparateMp4);
        assert_eq!(tiers[1], FormatFallbackTier::BestSeparateMkv);
        assert_eq!(tiers[2], FormatFallbackTier::BestCombined);

        // Tier 1 uses MP4 container
        assert_eq!(tiers[0].merge_output_format(), Some("mp4"));
        // Tier 2 uses MKV container for resilient AV1/VP9/Opus remuxing
        assert_eq!(tiers[1].merge_output_format(), Some("mkv"));
        // Tier 3 does not remux (progressive direct download)
        assert_eq!(tiers[2].merge_output_format(), None);
    }

    #[test]
    fn test_browser_cookies_unavailable_graceful_handling() {
        let stderr_lock = "ERROR: Could not copy Chrome cookie database. Database is locked.";
        let (is_lock, hint) = is_cookie_lock_or_access_error(stderr_lock);
        assert!(is_lock);
        assert!(hint.is_some());

        let attempted = vec!["Browser (chrome)".to_string(), "Unauthenticated".to_string()];
        let actionable = format_actionable_error(stderr_lock, &attempted);
        assert!(actionable.contains("Browser Cookie Extraction Error"));
        assert!(actionable.contains("Authentication attempts made"));
    }

    #[test]
    fn test_download_failure_explicit_error_diagnostics() {
        let fatal_stderr = "ERROR: [youtube] invalid_id: Video unavailable";
        let attempted = vec!["Unauthenticated".to_string()];
        let err_msg = format_actionable_error(fatal_stderr, &attempted);
        assert!(err_msg.contains("YouTube Download / Metadata Request Failed"));
        assert!(err_msg.contains("Video unavailable"));
        assert!(err_msg.contains("Unauthenticated"));
    }

    #[test]
    fn test_ffmpeg_merge_failure_resilience() {
        // Test that Tier 2 provides an MKV fallback when MP4 remux fails
        let tier2 = FormatFallbackTier::BestSeparateMkv;
        assert_eq!(tier2.merge_output_format(), Some("mkv"));
        assert!(tier2.format_selector().contains("bestvideo+bestaudio"));
    }

    #[test]
    fn test_ffprobe_validation_failure_rejection() {
        let non_existent = PathBuf::from("target/non_existent_video_file_xyz.mp4");
        let probe = probe_downloaded_quality(&non_existent, "https://youtube.com/watch?v=123");
        assert!(probe.is_err(), "Probing non-existent file must return Err");
    }

    #[test]
    fn test_quality_cache_version_semantics_and_sidecar_metadata() {
        let temp_dir = std::env::temp_dir().join("autoshorts_test_cache_version");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let dummy_meta = DownloadedQualityMetadata {
            source_url: "https://www.youtube.com/watch?v=test123".to_string(),
            video_id: Some("test123".to_string()),
            file_path: temp_dir.join("AutoShorts_test123.mp4"),
            width: 3840,
            height: 2160,
            fps: Some(60.0),
            video_codec: Some("av01".to_string()),
            audio_codec: Some("opus".to_string()),
            duration_sec: Some(42.0),
            bitrate: Some(15000000),
            container: Some("mp4".to_string()),
            file_size_bytes: 50000000,
        };

        let saved_path = save_quality_metadata(
            &dummy_meta,
            FormatFallbackTier::BestSeparateMp4,
            Some(2160),
            &temp_dir,
        ).expect("Must save quality metadata");
        assert!(saved_path.exists());

        let read_meta = read_quality_metadata(&temp_dir, "test123").expect("Must read saved metadata");
        assert_eq!(read_meta.version, QUALITY_CACHE_VERSION);
        assert_eq!(read_meta.width, 3840);
        assert_eq!(read_meta.height, 2160);
        assert_eq!(read_meta.video_codec.as_deref(), Some("av01"));
        assert_eq!(read_meta.audio_codec.as_deref(), Some("opus"));
        assert_eq!(read_meta.max_available_height, Some(2160));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_aspect_ratio_ultrawide_and_vertical_shorts_higher_quality_detection() {
        let avail_ultrawide = AvailableQuality {
            max_height: 1080,
            max_width: 2560,
            max_fps: Some(60.0),
            best_vcodec: Some("vp9".to_string()),
            best_acodec: Some("opus".to_string()),
            format_id: Some("271".to_string()),
        };

        // 2560x1080 (ultrawide) is higher than standard 1080p (1920x1080)
        assert!(avail_ultrawide.is_higher_than(1920, 1080));
        // Equal resolution is not higher
        assert!(!avail_ultrawide.is_higher_than(2560, 1080));
        // 4K is higher than ultrawide 1080p
        assert!(!avail_ultrawide.is_higher_than(3840, 2160));

        // Vertical YouTube Short: 1080x1920
        let avail_short = AvailableQuality {
            max_height: 1920,
            max_width: 1080,
            max_fps: Some(30.0),
            best_vcodec: Some("avc1".to_string()),
            best_acodec: Some("mp4a".to_string()),
            format_id: Some("137".to_string()),
        };
        // 1080x1920 is higher than 720x1280
        assert!(avail_short.is_higher_than(720, 1280));
        assert!(!avail_short.is_higher_than(1080, 1920));
    }

    #[test]
    fn test_is_intermediate_or_part_file() {
        assert!(is_intermediate_or_part_file("AutoShorts_abc.f251.webm"));
        assert!(is_intermediate_or_part_file("AutoShorts_abc.f313.mp4"));
        assert!(is_intermediate_or_part_file("AutoShorts_abc.part"));
        assert!(is_intermediate_or_part_file("AutoShorts_abc.ytdl"));
        assert!(is_intermediate_or_part_file("AutoShorts_abc.temp.mp4"));
        assert!(!is_intermediate_or_part_file("AutoShorts_abc.mp4"));
        assert!(!is_intermediate_or_part_file("AutoShorts_abc.mkv"));
        assert!(!is_intermediate_or_part_file("AutoShorts_abc.webm"));
    }

    #[test]
    fn test_probe_downloaded_quality_real_or_fixture() {
        let test_file = PathBuf::from("target/test_1440p_QhcY-XE0byw.mp4");
        if test_file.exists() {
            let meta = probe_downloaded_quality(&test_file, "https://www.youtube.com/watch?v=QhcY-XE0byw");
            assert!(meta.is_ok(), "probe_downloaded_quality failed: {:?}", meta.err());
            let m = meta.unwrap();
            assert_eq!(m.width, 2560);
            assert_eq!(m.height, 1440);
            assert_eq!(m.video_codec, Some("vp9".to_string()));
            assert_eq!(m.audio_codec, Some("opus".to_string()));
            assert_eq!(m.container, Some("mp4".to_string()));
            assert!(m.file_size_bytes > 0);
        }
    }

    #[test]
    #[ignore] // Live 1440p download test: cargo test --lib test_real_youtube_download_1440p_e2e -- --ignored --nocapture
    fn test_real_youtube_download_1440p_e2e() {
        let url = "https://www.youtube.com/watch?v=QhcY-XE0byw";
        let target_dir = std::env::temp_dir().join("autoshorts_1440p_test");
        let _ = std::fs::remove_dir_all(&target_dir);
        std::fs::create_dir_all(&target_dir).expect("must create temp dir");

        let res = download_video(url, &target_dir, Some("auto"), None);
        assert!(res.is_ok(), "1440p download failed: {:?}", res.err());
        let file = res.unwrap();
        assert!(file.exists());

        let meta = probe_downloaded_quality(&file, url).expect("probe must succeed");
        assert_eq!(meta.height, 1440, "Expected 1440p resolution, got {}x{}", meta.width, meta.height);
        assert_eq!(meta.width, 2560);
        println!("[1440p E2E VERIFIED] {}x{} | {:?} | {:?} | {} bytes", meta.width, meta.height, meta.video_codec, meta.audio_codec, meta.file_size_bytes);

        let _ = std::fs::remove_dir_all(&target_dir);
    }

    #[test]
    #[ignore] // Live 4K download test: cargo test --lib test_real_youtube_download_4k_e2e -- --ignored --nocapture
    fn test_real_youtube_download_4k_e2e() {
        let url = "https://www.youtube.com/watch?v=w6uX9jamcwQ";
        let target_dir = std::env::temp_dir().join("autoshorts_4k_test");
        let _ = std::fs::remove_dir_all(&target_dir);
        std::fs::create_dir_all(&target_dir).expect("must create temp dir");

        let res = download_video(url, &target_dir, Some("auto"), None);
        assert!(res.is_ok(), "4K download failed: {:?}", res.err());
        let file = res.unwrap();
        assert!(file.exists());

        let meta = probe_downloaded_quality(&file, url).expect("probe must succeed");
        assert!(meta.width >= 3840 || meta.height >= 1920, "Expected 4K resolution (>=3840 wide or >=1920 high), got {}x{}", meta.width, meta.height);
        println!("[4K E2E VERIFIED] {}x{} | {:?} | {:?} | {} bytes", meta.width, meta.height, meta.video_codec, meta.audio_codec, meta.file_size_bytes);

        let _ = std::fs::remove_dir_all(&target_dir);
    }
}
