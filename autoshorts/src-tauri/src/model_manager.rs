use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const YOLO_UPSTREAM_URL: &str =
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt";
pub const YOLO_MIN_BYTES: u64 = 1_000_000;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelDownloadProgress {
    pub model: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub percent: Option<f64>,
}

/// Returns the models directory (%LOCALAPPDATA%/autoshorts/models or fallback).
pub fn get_models_dir() -> PathBuf {
    if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
        let trimmed = local_appdata.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("autoshorts").join("models");
        }
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("autoshorts")
        .join("models")
}

/// Checks whether the file exists and is at least YOLO_MIN_BYTES.
pub fn is_yolo_file_valid(path: &Path) -> bool {
    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.len() >= YOLO_MIN_BYTES,
        Err(_) => false,
    }
}

/// Checks whether the YOLO11 model is already installed and valid in the models directory.
pub fn is_yolo_installed() -> bool {
    let target = get_models_dir().join("yolo11n.pt");
    is_yolo_file_valid(&target)
}

/// Ensures YOLO11 weights are downloaded into the models directory.
/// If already present and valid, returns target path immediately.
/// Otherwise streams download from upstream with progress events and atomic rename.
pub async fn ensure_yolo_weights<R: tauri::Runtime>(
    app: Option<&tauri::AppHandle<R>>,
) -> Result<PathBuf> {
    let models_dir = get_models_dir();
    std::fs::create_dir_all(&models_dir)
        .with_context(|| format!("Failed to create models directory at {}", models_dir.display()))?;

    let target_path = models_dir.join("yolo11n.pt");
    if is_yolo_file_valid(&target_path) {
        return Ok(target_path);
    }

    let part_path = models_dir.join("yolo11n.pt.part");

    let download_result: Result<(u64, Option<u64>)> = async {
        let client = reqwest::Client::builder()
            .build()
            .context("Failed to build HTTP client for YOLO download")?;

        let mut response = client
            .get(YOLO_UPSTREAM_URL)
            .send()
            .await
            .context("Failed to send HTTP request to download YOLO weights")?;

        if !response.status().is_success() {
            anyhow::bail!(
                "Failed to download YOLO weights from {}: HTTP status {}",
                YOLO_UPSTREAM_URL,
                response.status()
            );
        }

        let total_bytes = response.content_length();
        let mut downloaded_bytes: u64 = 0;

        let file = std::fs::File::create(&part_path).with_context(|| {
            format!("Failed to create partial download file at {}", part_path.display())
        })?;
        let mut writer = std::io::BufWriter::new(file);

        use std::io::Write;
        use tauri::Emitter;

        while let Some(chunk) = response
            .chunk()
            .await
            .context("Error reading streaming chunk during YOLO weights download")?
        {
            writer
                .write_all(&chunk)
                .context("Error writing chunk to temporary YOLO weights file")?;
            downloaded_bytes += chunk.len() as u64;

            if let Some(app_handle) = app {
                let percent = total_bytes.map(|total| {
                    if total > 0 {
                        (downloaded_bytes as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    }
                });

                let progress = ModelDownloadProgress {
                    model: "yolo11n.pt".to_string(),
                    downloaded: downloaded_bytes,
                    total: total_bytes,
                    percent,
                };
                let _ = app_handle.emit("model_download_progress", &progress);
            }
        }

        writer
            .flush()
            .context("Failed to flush temporary YOLO weights file")?;
        drop(writer);

        Ok((downloaded_bytes, total_bytes))
    }
    .await;

    let (downloaded_bytes, total_bytes) = match download_result {
        Ok(vals) => vals,
        Err(e) => {
            let _ = std::fs::remove_file(&part_path);
            return Err(e);
        }
    };

    if downloaded_bytes < YOLO_MIN_BYTES {
        let _ = std::fs::remove_file(&part_path);
        anyhow::bail!(
            "Downloaded YOLO weights file is too small ({} bytes, expected >= {} bytes)",
            downloaded_bytes,
            YOLO_MIN_BYTES
        );
    }

    if target_path.exists() {
        let _ = std::fs::remove_file(&target_path);
    }

    std::fs::rename(&part_path, &target_path).with_context(|| {
        format!(
            "Failed to rename {} to {}",
            part_path.display(),
            target_path.display()
        )
    })?;

    if let Some(app_handle) = app {
        use tauri::Emitter;
        let progress = ModelDownloadProgress {
            model: "yolo11n.pt".to_string(),
            downloaded: downloaded_bytes,
            total: total_bytes.or(Some(downloaded_bytes)),
            percent: Some(100.0),
        };
        let _ = app_handle.emit("model_download_progress", &progress);
    }

    if !is_yolo_file_valid(&target_path) {
        anyhow::bail!(
            "Downloaded YOLO weights at {} failed validation",
            target_path.display()
        );
    }

    Ok(target_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_get_models_dir_structure() {
        let dir = get_models_dir();
        assert!(dir.ends_with(Path::new("autoshorts").join("models")));
    }

    #[test]
    fn test_get_models_dir_with_localappdata() {
        let _guard = ENV_MUTEX.lock().unwrap();
        let prev = std::env::var("LOCALAPPDATA").ok();
        let custom_dir = std::env::temp_dir().join(format!("test_appdata_{}", uuid::Uuid::new_v4()));

        std::env::set_var("LOCALAPPDATA", &custom_dir);
        let resolved = get_models_dir();
        assert_eq!(resolved, custom_dir.join("autoshorts").join("models"));

        match prev {
            Some(v) => std::env::set_var("LOCALAPPDATA", v),
            None => std::env::remove_var("LOCALAPPDATA"),
        }
    }

    #[test]
    fn test_is_yolo_file_valid() {
        let temp_dir = std::env::temp_dir();
        let non_existent = temp_dir.join(format!("non_existent_{}.pt", uuid::Uuid::new_v4()));
        assert!(!is_yolo_file_valid(&non_existent));

        // Directory test
        let test_dir = temp_dir.join(format!("test_dir_{}", uuid::Uuid::new_v4()));
        let _ = std::fs::create_dir_all(&test_dir);
        assert!(!is_yolo_file_valid(&test_dir));
        let _ = std::fs::remove_dir(&test_dir);

        // Small file test (< 1MB)
        let small_file = temp_dir.join(format!("test_small_{}.pt", uuid::Uuid::new_v4()));
        std::fs::write(&small_file, b"too small").expect("write small file");
        assert!(!is_yolo_file_valid(&small_file));
        let _ = std::fs::remove_file(&small_file);

        // Valid sized file test (>= YOLO_MIN_BYTES)
        let valid_file = temp_dir.join(format!("test_valid_{}.pt", uuid::Uuid::new_v4()));
        let file = std::fs::File::create(&valid_file).expect("create valid file");
        file.set_len(YOLO_MIN_BYTES).expect("set valid file len");
        drop(file);
        assert!(is_yolo_file_valid(&valid_file));
        let _ = std::fs::remove_file(&valid_file);
    }
}
