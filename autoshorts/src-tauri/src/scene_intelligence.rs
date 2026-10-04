//! AutoShorts 11.0 Phase 3 — Scene Intelligence (source-level shot boundaries)
//!
//! Runs PySceneDetect (AdaptiveDetector) ONCE per source video via a Python
//! sidecar, caches the result per (source hash, detector version, config
//! version), and exposes the boundary list to the framing sidecar. Scene
//! boundaries are ADVISORY metadata: the framing engine's own cut detection,
//! safety logic, and invariants remain authoritative. Any failure falls back
//! to single-scene behavior (rendering never depends on this subsystem).

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufReader, Read};
use std::path::PathBuf;

/// Scene detector identity written to the cache and DB.
pub const SCENE_DETECTOR: &str = "PySceneDetect-AdaptiveDetector";
pub const SCENE_DETECTOR_VERSION: &str = "0.7.1";
pub const SCENE_CONFIG_VERSION: &str = "2-frame_skip";

/// Wall-clock budget for the scene sidecar, derived from source duration.
///
/// The sidecar's cost is linear in runtime: measured 20.1 s per 60 s of 1080p
/// source (~0.33x realtime). The budget is therefore a multiple of the source
/// duration rather than a fixed constant, so a 25-minute source is allowed
/// enough time to finish while a pathological child is still killed.
const SCENE_TIMEOUT_FLOOR_SEC: f64 = 120.0;
const SCENE_TIMEOUT_REALTIME_FACTOR: f64 = 2.5;
const SCENE_TIMEOUT_CEILING_SEC: f64 = 3600.0;

/// Default frame-sampling stride for the sidecar, mirroring the Python
/// `FRAME_SKIP` default. Declared here so the value is asserted in tests rather
/// than duplicated as a magic number.
pub const FRAME_SKIP_DEFAULT: u32 = 2;

fn scene_timeout_for(duration_sec: Option<f64>) -> std::time::Duration {
    let secs = match duration_sec {
        Some(d) if d.is_finite() && d > 0.0 => SCENE_TIMEOUT_REALTIME_FACTOR * d,
        // Unknown duration: assume a long source rather than guess short and
        // truncate a legitimate detection run.
        _ => SCENE_TIMEOUT_CEILING_SEC,
    };
    let clamped = secs.clamp(SCENE_TIMEOUT_FLOOR_SEC, SCENE_TIMEOUT_CEILING_SEC);
    std::time::Duration::from_secs_f64(clamped)
}

/// Probe media duration in seconds (ffprobe), or None when unavailable.
pub fn probe_media_duration_seconds(path: &str) -> Option<f64> {
    if !crate::media::command_exists("ffprobe") {
        return None;
    }
    let mut cmd = std::process::Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]);
    // ffprobe on a local file is fast; a short bound still prevents a hung
    // probe from stalling the pipeline before the real work starts.
    let out = crate::proc_guard::run_bounded(
        &mut cmd,
        std::time::Duration::from_secs(30),
        "SceneIntel/duration-probe",
    )
    .ok()?;
    if !out.success {
        return None;
    }
    out.stdout.trim().lines().next()?.trim().parse::<f64>().ok()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneBoundary {
    pub scene_id: i64,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneIntelligenceResult {
    pub source_hash: String,
    pub detector: String,
    pub detector_version: String,
    pub config_version: String,
    pub scenes: Vec<SceneBoundary>,
    /// True when the sidecar failed and the single-scene fallback was used.
    pub fallback: bool,
    pub created_at: String,
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Master switch for scene intelligence.
pub fn scene_intelligence_enabled() -> bool {
    env_flag("AUTOSHORTS_SCENE_INTELLIGENCE")
}

/// Structurally validate a scene boundary list before it is trusted
/// downstream.
///
/// Scene data is consumed by the framing engine to reset crop trajectories at
/// cuts, so a malformed list is actively harmful, not merely useless:
/// non-monotonic boundaries would produce a crop trajectory that jumps
/// backwards in time. This is the Rust-side counterpart of "output must be
/// consumed downstream", not a cosmetic check.
pub fn validate_scene_boundaries(
    scenes: &[SceneBoundary],
    duration_hint: Option<f64>,
) -> Result<()> {
    if scenes.is_empty() {
        return Err(anyhow!("no scenes"));
    }
    let mut prev_end = f64::NEG_INFINITY;
    for (i, s) in scenes.iter().enumerate() {
        if !s.start.is_finite() || !s.end.is_finite() {
            return Err(anyhow!(
                "scene {} has non-finite bounds ({}, {})",
                i,
                s.start,
                s.end
            ));
        }
        if s.start < 0.0 {
            return Err(anyhow!("scene {} starts before zero ({})", i, s.start));
        }
        if s.end <= s.start {
            return Err(anyhow!(
                "scene {} has non-positive duration ({} -> {})",
                i,
                s.start,
                s.end
            ));
        }
        if s.start < prev_end {
            return Err(anyhow!(
                "scene {} starts at {} before previous scene ended at {} (non-monotonic)",
                i,
                s.start,
                prev_end
            ));
        }
        prev_end = s.end;
    }
    if let Some(d) = duration_hint.filter(|d| d.is_finite() && *d > 0.0) {
        // A cut a hair past the end is rounding; far past it means the sidecar
        // analysed a different file than the one probed.
        let last = prev_end;
        if last > d * 1.05 + 5.0 {
            return Err(anyhow!(
                "last scene ends at {:.1}s, beyond probed source duration {:.1}s",
                last,
                d
            ));
        }
    }
    Ok(())
}

pub struct SceneIntelligenceEngine {
    cache_dir: PathBuf,
}

impl SceneIntelligenceEngine {
    pub fn new(cache_dir: Option<String>) -> Result<Self> {
        let dir = cache_dir.map(PathBuf::from).unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("autoshorts")
                .join("scene_intel")
        });
        std::fs::create_dir_all(&dir).context("creating scene intelligence cache dir")?;
        Ok(Self { cache_dir: dir })
    }

    pub fn compute_source_hash(&self, source_path: &str) -> Result<String> {
        let file = std::fs::File::open(source_path).context("opening source for hashing")?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = reader
                .read(&mut buffer)
                .context("reading source for hash")?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn cache_path(&self, source_hash: &str) -> PathBuf {
        self.cache_dir.join(format!("scenes_{}.json", source_hash))
    }

    /// Load a cached result. Stale (detector/config version mismatch) or
    /// corrupted caches are treated as a miss and removed — never served.
    fn load_cache(&self, source_hash: &str) -> Option<SceneIntelligenceResult> {
        let path = self.cache_path(source_hash);
        let content = std::fs::read_to_string(&path).ok()?;
        match serde_json::from_str::<SceneIntelligenceResult>(&content) {
            Ok(doc)
                if doc.source_hash == source_hash
                    && doc.detector_version == SCENE_DETECTOR_VERSION
                    && doc.config_version == SCENE_CONFIG_VERSION
                    && !doc.scenes.is_empty() =>
            {
                Some(doc)
            }
            _ => {
                eprintln!(
                    "[SceneIntel] stale/corrupt cache removed: {}",
                    path.display()
                );
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    fn save_cache(&self, doc: &SceneIntelligenceResult) -> Result<()> {
        let path = self.cache_path(&doc.source_hash);
        std::fs::write(&path, serde_json::to_string_pretty(doc)?)
            .context("writing scene intelligence cache")?;
        Ok(())
    }

    fn find_sidecar_script(&self) -> Result<PathBuf> {
        let script_name = "scene_intelligence.py";
        let candidates = [
            PathBuf::from("src-tauri/scripts").join(script_name),
            PathBuf::from("scripts").join(script_name),
            PathBuf::from("../src-tauri/scripts").join(script_name),
        ];
        for c in &candidates {
            if c.exists() {
                return Ok(c.clone());
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let c = parent.join("scripts").join(script_name);
                if c.exists() {
                    return Ok(c);
                }
            }
        }
        Err(anyhow!("scene_intelligence.py not found"))
    }

    /// Run scene detection (cached). On ANY failure returns the single-scene
    /// fallback so the pipeline continues — never an error path for callers.
    pub fn process_source(&self, source_path: &str) -> SceneIntelligenceResult {
        let started = std::time::Instant::now();
        let fallback = |reason: &str| {
            eprintln!("[SceneIntel] {} — single-scene fallback", reason);
            SceneIntelligenceResult {
                source_hash: String::new(),
                detector: SCENE_DETECTOR.to_string(),
                detector_version: SCENE_DETECTOR_VERSION.to_string(),
                config_version: SCENE_CONFIG_VERSION.to_string(),
                scenes: Vec::new(),
                fallback: true,
                created_at: chrono::Utc::now().to_rfc3339(),
            }
        };

        let source_hash = match self.compute_source_hash(source_path) {
            Ok(h) => h,
            Err(e) => return fallback(&format!("hash failed: {}", e)),
        };
        if let Some(cached) = self.load_cache(&source_hash) {
            println!(
                "[SceneIntel] COMPLETE (cache hit) {} scenes in {:.0}ms",
                cached.scenes.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );
            return cached;
        }

        let script = match self.find_sidecar_script() {
            Ok(p) => p,
            Err(e) => return fallback(&format!("{}: {}", e, "script missing")),
        };
        let python = crate::media::find_python_cmd();
        // The sidecar prints the scene JSON document to stdout (no --out).
        //
        // Scene detection decodes the whole source and is therefore
        // proportional to runtime. On a 25-minute source the sidecar took 8.5
        // minutes, so the deadline scales with the probed duration instead of
        // being a fixed value that either truncates long sources or leaves short
        // ones waiting. It is advisory work, so a timeout degrades to the
        // single-scene fallback rather than failing the render.
        let duration_hint = probe_media_duration_seconds(source_path);
        let budget = scene_timeout_for(duration_hint);
        println!(
            "[SceneIntel] START duration={} budget={}s",
            duration_hint
                .map(|d| format!("{:.0}s", d))
                .unwrap_or_else(|| "unknown".to_string()),
            budget.as_secs_f64()
        );

        let mut cmd = std::process::Command::new(&python);
        cmd.arg(&script)
            .arg(source_path)
            .arg("--cache-dir")
            .arg(&self.cache_dir);

        let output = match crate::proc_guard::run_bounded(&mut cmd, budget, "SceneIntel") {
            Ok(o) => o,
            Err(e) => return fallback(&format!("scene sidecar spawn failed: {}", e)),
        };

        if output.timed_out {
            return fallback(&format!(
                "scene sidecar TIMEOUT after {:.0}s (duration {:?})",
                output.elapsed.as_secs_f64(),
                duration_hint
            ));
        }
        if !output.success {
            return fallback(&format!(
                "scene sidecar failed (rc={:?}): {}",
                output.code,
                crate::proc_guard::sanitize_stderr(&output.stderr, 400)
            ));
        }
        println!(
            "[SceneIntel] sidecar returned in {:.1}s (rc={:?})",
            output.elapsed.as_secs_f64(),
            output.code
        );
        // The sidecar's stdout is now a single JSON document: the Python side
        // sends ALL human diagnostics to stderr. Parse it structurally and
        // report the concrete failure instead of discarding the error.
        let stdout = output.stdout.as_str().trim();
        let parsed: std::result::Result<SceneIntelligenceResult, _> =
            serde_json::from_str(stdout);
        let doc: SceneIntelligenceResult = match parsed {
            Ok(d) if !d.scenes.is_empty() => d,
            Ok(d) => {
                return fallback(&format!(
                    "scene sidecar returned a structurally valid document but \
                     {} scene(s) with fallback={}",
                    d.scenes.len(),
                    d.fallback
                ))
            }
            Err(e) => {
                // serde reports a missing field as "missing field `fallback`";
                // that is a CONTRACT BUG in the sidecar, not an empty result,
                // and must be visible rather than collapsed into "no JSON".
                return fallback(&format!(
                    "scene sidecar JSON contract violation: {} (stdout was {} bytes, \
                     stderr: {})",
                    e,
                    stdout.len(),
                    crate::proc_guard::sanitize_stderr(&output.stderr, 300)
                ));
            }
        };

        // Structural validation: boundaries must be usable downstream. A scene
        // list that is not monotonic or runs negative is not real intelligence,
        // so it is rejected here rather than poisoning framing.
        let doc = SceneIntelligenceResult { source_hash, ..doc };
        if let Err(e) = validate_scene_boundaries(&doc.scenes, duration_hint) {
            return fallback(&format!("scene sidecar produced invalid boundaries: {}", e));
        }
        if let Err(e) = self.save_cache(&doc) {
            eprintln!("[SceneIntel] cache save failed (non-fatal): {}", e);
        }
        println!(
            "[SceneIntel] COMPLETE {} scenes in {:.1}s",
            doc.scenes.len(),
            started.elapsed().as_secs_f64()
        );
        doc
    }

    /// Cut times (absolute source seconds) for the framing sidecar.
    pub fn cut_times(result: &SceneIntelligenceResult) -> Vec<f64> {
        result
            .scenes
            .iter()
            .map(|s| s.start)
            .filter(|t| *t > 0.0)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_timeout_scales_with_duration_and_is_bounded() {
        // Short source: floored so a normal short video is never truncated.
        let short = scene_timeout_for(Some(20.0));
        assert!(short.as_secs_f64() >= 120.0, "{:?}", short);
        // The real 25-minute source gets a budget above its measured 183s run,
        // so the fix is not merely "wait longer" -- it makes the stage finish
        // inside a bounded window.
        let long = scene_timeout_for(Some(1482.4));
        assert!(
            long.as_secs_f64() > 183.0,
            "budget {:?} must exceed the measured 183s runtime",
            long
        );
        // Unknown duration assumes long rather than guessing short and
        // truncating a legitimate detection run.
        assert_eq!(scene_timeout_for(None).as_secs_f64(), 3600.0);
        // Bounded on both ends: never unbounded, never absurd.
        assert!(long.as_secs_f64() <= 3600.0);
        assert!(scene_timeout_for(Some(1e9)).as_secs_f64() <= 3600.0);
        assert!(scene_timeout_for(Some(f64::NAN)).as_secs_f64() <= 3600.0);
    }

    #[test]
    fn frame_skip_is_configurable_and_reaches_detector() {
        // The speed fix must be a real, adjustable knob, not a hardcoded value
        // that cannot be turned off when bit-exact detection is required.
        let engine = SceneIntelligenceEngine::new(None).expect("engine");
        let script = engine.find_sidecar_script().expect("sidecar script");
        let src = std::fs::read_to_string(&script).unwrap();
        assert!(
            src.contains("AUTOSHORTS_SCENE_FRAME_SKIP"),
            "frame skip must be overridable via env"
        );
        assert!(
            src.contains("frame_skip=FRAME_SKIP"),
            "the configured value must actually reach detect_scenes()"
        );
        assert_eq!(FRAME_SKIP_DEFAULT, 2, "measured-safe default");
    }

    #[test]
    fn config_version_bumped_and_matches_sidecar() {
        // Changing detector behavior must invalidate per-source scene caches,
        // otherwise a stale full-frame result is silently reused. Rust and
        // Python must also agree, or the cache key diverges unnoticed.
        assert_ne!(
            SCENE_CONFIG_VERSION, "1",
            "config version must change with behavior"
        );
        let engine = SceneIntelligenceEngine::new(None).expect("engine");
        let script = engine.find_sidecar_script().expect("sidecar script");
        let py = std::fs::read_to_string(&script).unwrap();
        assert!(
            py.contains(&format!(
                "SCENE_CONFIG_VERSION = \"{}\"",
                SCENE_CONFIG_VERSION
            )),
            "python SCENE_CONFIG_VERSION must match rust ({})",
            SCENE_CONFIG_VERSION
        );
    }

    #[test]
    fn test_scene_flag_semantics() {
        std::env::remove_var("AUTOSHORTS_SCENE_INTELLIGENCE");
        assert!(scene_intelligence_enabled());
        for off in ["0", "false", "off", "OFF", "False"] {
            std::env::set_var("AUTOSHORTS_SCENE_INTELLIGENCE", off);
            assert!(!scene_intelligence_enabled(), "{} must disable", off);
        }
        std::env::set_var("AUTOSHORTS_SCENE_INTELLIGENCE", "1");
        assert!(scene_intelligence_enabled());
        std::env::remove_var("AUTOSHORTS_SCENE_INTELLIGENCE");
    }

    #[test]
    fn test_scene_json_contract_roundtrip() {
        let doc = SceneIntelligenceResult {
            source_hash: "abc".into(),
            detector: SCENE_DETECTOR.into(),
            detector_version: SCENE_DETECTOR_VERSION.into(),
            config_version: SCENE_CONFIG_VERSION.into(),
            scenes: vec![
                SceneBoundary {
                    scene_id: 0,
                    start: 0.0,
                    end: 5.96,
                },
                SceneBoundary {
                    scene_id: 1,
                    start: 5.96,
                    end: 7.96,
                },
            ],
            fallback: false,
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let text = serde_json::to_string(&doc).unwrap();
        assert!(text.contains("\"sceneId\""));
        let back: SceneIntelligenceResult = serde_json::from_str(&text).unwrap();
        assert_eq!(back.scenes.len(), 2);
        assert_eq!(back.scenes[1].start, 5.96);
    }

    #[test]
    fn test_cut_times_excludes_zero() {
        let doc = SceneIntelligenceResult {
            source_hash: "abc".into(),
            detector: SCENE_DETECTOR.into(),
            detector_version: SCENE_DETECTOR_VERSION.into(),
            config_version: SCENE_CONFIG_VERSION.into(),
            scenes: vec![
                SceneBoundary {
                    scene_id: 0,
                    start: 0.0,
                    end: 5.0,
                },
                SceneBoundary {
                    scene_id: 1,
                    start: 5.0,
                    end: 9.0,
                },
            ],
            fallback: false,
            created_at: "x".into(),
        };
        assert_eq!(SceneIntelligenceEngine::cut_times(&doc), vec![5.0]);
    }

    #[test]
    fn test_process_source_missing_file_falls_back() {
        let cache_dir = std::env::temp_dir().join(format!("scene_test_{}", std::process::id()));
        let engine =
            SceneIntelligenceEngine::new(Some(cache_dir.to_string_lossy().to_string())).unwrap();
        let doc = engine.process_source("Z:/definitely/missing/video.mp4");
        assert!(
            doc.fallback,
            "missing source must produce the single-scene fallback"
        );
        let _ = std::fs::remove_dir_all(&cache_dir);
    }

    // ── REGRESSION: the JSON contract that silently produced 0 scenes ──────
    //
    // The sidecar omitted the `fallback` field, so serde failed on EVERY
    // successful detection run and the error was discarded with `.ok()`,
    // reporting "no usable JSON" for a run that actually found 170 scenes.
    // These tests pin the contract from both sides.

    /// Exact document the sidecar emitted before the fix: every field the
    /// Rust struct needs EXCEPT `fallback`.
    #[test]
    fn regression_sidecar_document_without_fallback_fails_loudly() {
        let text = r#"{
            "sourceHash":"abc",
            "detector":"PySceneDetect-AdaptiveDetector",
            "detectorVersion":"0.7.1",
            "configVersion":"2-frame_skip",
            "config":{"adaptiveThreshold":3.0,"minSceneLenFrames":12,"frameSkip":2},
            "scenes":[{"sceneId":0,"start":0.0,"end":2.76}],
            "elapsedSec":174.2,
            "createdAt":"2026-09-30T00:00:00Z"
        }"#;
        let err = serde_json::from_str::<SceneIntelligenceResult>(text)
            .expect_err("a document lacking `fallback` must NOT deserialize");
        let msg = err.to_string();
        assert!(
            msg.contains("fallback"),
            "error must name the missing field, got: {}",
            msg
        );
    }

    #[test]
    fn sidecar_document_with_fallback_deserializes() {
        let text = r#"{
            "sourceHash":"abc",
            "detector":"PySceneDetect-AdaptiveDetector",
            "detectorVersion":"0.7.1",
            "configVersion":"2-frame_skip",
            "fallback":false,
            "config":{"adaptiveThreshold":3.0,"minSceneLenFrames":12,"frameSkip":2},
            "scenes":[
                {"sceneId":0,"start":0.0,"end":2.76},
                {"sceneId":1,"start":2.76,"end":3.36},
                {"sceneId":2,"start":3.36,"end":3.84}
            ],
            "elapsedSec":174.2,
            "createdAt":"2026-09-30T00:00:00Z"
        }"#;
        let doc: SceneIntelligenceResult =
            serde_json::from_str(text).expect("fixed sidecar document must deserialize");
        assert!(!doc.fallback);
        assert_eq!(doc.scenes.len(), 3);
        assert_eq!(doc.scenes[1].start, 2.76);
    }

    #[test]
    fn sidecar_emits_fallback_field_on_every_path() {
        // The Python side must always write `fallback`, on both the real and the
        // degraded path. A cache-hit document must too, or the cache writes a
        // file Rust cannot read.
        let engine = SceneIntelligenceEngine::new(None).expect("engine");
        let script = engine.find_sidecar_script().expect("sidecar script");
        let py = std::fs::read_to_string(&script).unwrap();
        assert!(
            py.contains("\"fallback\": used_fallback"),
            "sidecar must emit the `fallback` key the Rust struct requires"
        );
        // stdout must stay a pure JSON channel.
        assert!(
            !py.contains("print(f\"[SceneIntel] {len(scenes)}"),
            "progress diagnostics must not be written to stdout"
        );
    }

    #[test]
    fn stdout_json_parses_without_scanning_for_brace() {
        // The old parser searched for the first line starting with '{' and
        // joined the remainder. A stray leading diagnostic broke it. The fixed
        // parser consumes stdout as a whole document.
        let doc = r#"{"sourceHash":"h","detector":"d","detectorVersion":"0.7.1",
            "configVersion":"2-frame_skip","fallback":false,"createdAt":"t",
            "scenes":[{"sceneId":0,"start":0.0,"end":1.0}]}"#;
        let parsed: SceneIntelligenceResult =
            serde_json::from_str(doc.trim()).expect("whitespace-tolerant whole-doc parse");
        assert_eq!(parsed.scenes.len(), 1);
        // Leading progress noise is now a hard contract violation, not something
        // to paper over by scanning.
        let noisy = format!("[SceneIntel] working...\n{}", doc);
        assert!(
            serde_json::from_str::<SceneIntelligenceResult>(noisy.trim()).is_err(),
            "stdout must be pure JSON; diagnostics belong on stderr"
        );
    }

    #[test]
    fn boundary_validation_accepts_real_monotonic_scenes() {
        let scenes: Vec<SceneBoundary> = (0..170)
            .map(|i| SceneBoundary {
                scene_id: i,
                start: i as f64 * 8.7,
                end: (i + 1) as f64 * 8.7,
            })
            .collect();
        assert!(
            validate_scene_boundaries(&scenes, Some(1482.4)).is_ok(),
            "a real monotonic list must validate"
        );
    }

    #[test]
    fn boundary_validation_rejects_corrupt_scene_lists() {
        let mono = vec![
            SceneBoundary {
                scene_id: 0,
                start: 0.0,
                end: 2.76,
            },
            SceneBoundary {
                scene_id: 1,
                start: 2.76,
                end: 3.36,
            },
        ];
        assert!(validate_scene_boundaries(&mono, Some(1482.4)).is_ok());

        // Empty.
        assert!(validate_scene_boundaries(&[], None).is_err());
        // Non-monotonic: would drive the crop trajectory backwards in time.
        let nonmono = vec![
            SceneBoundary {
                scene_id: 0,
                start: 0.0,
                end: 10.0,
            },
            SceneBoundary {
                scene_id: 1,
                start: 5.0,
                end: 12.0,
            },
        ];
        assert!(validate_scene_boundaries(&nonmono, None).is_err());
        // Negative.
        let neg = vec![SceneBoundary {
            scene_id: 0,
            start: -1.0,
            end: 5.0,
        }];
        assert!(validate_scene_boundaries(&neg, None).is_err());
        // Zero/negative duration.
        let inverted = vec![SceneBoundary {
            scene_id: 0,
            start: 5.0,
            end: 5.0,
        }];
        assert!(validate_scene_boundaries(&inverted, None).is_err());
        // Non-finite.
        let nan = vec![SceneBoundary {
            scene_id: 0,
            start: 0.0,
            end: f64::NAN,
        }];
        assert!(validate_scene_boundaries(&nan, None).is_err());
        // Beyond the probed source: wrong file analysed.
        let beyond = vec![SceneBoundary {
            scene_id: 0,
            start: 0.0,
            end: 9999.0,
        }];
        assert!(validate_scene_boundaries(&beyond, Some(1482.4)).is_err());
        // Small rounding overshoot is tolerated.
        let rounding = vec![SceneBoundary {
            scene_id: 0,
            start: 0.0,
            end: 1483.0,
        }];
        assert!(validate_scene_boundaries(&rounding, Some(1482.4)).is_ok());
    }

    #[test]
    fn sidecar_and_rust_cache_files_do_not_collide() {
        // Rust wrote `scenes_<hash>.json` and deleted it as "stale/corrupt"
        // because the Python document lacked `fallback`, so the Python-level
        // cache could never produce a hit. Each layer now owns its own file.
        let engine = SceneIntelligenceEngine::new(None).expect("engine");
        let script = engine.find_sidecar_script().expect("sidecar script");
        let py = std::fs::read_to_string(&script).unwrap();
        assert!(
            py.contains("pyscenes_"),
            "sidecar must use its own cache filename"
        );
        assert!(
            !py.contains("f\"scenes_{source_hash}.json\""),
            "sidecar must not write the filename Rust reserves"
        );
        assert_eq!(
            engine.cache_path("HASH").file_name().unwrap().to_string_lossy(),
            "scenes_HASH.json",
            "rust keeps the plain name"
        );
    }

    #[test]
    fn cut_times_are_valid_for_framing_consumer() {
        // Scene data is only useful if the framing consumer gets real cuts in
        // global source time.
        let scenes = vec![
            SceneBoundary {
                scene_id: 0,
                start: 0.0,
                end: 2.76,
            },
            SceneBoundary {
                scene_id: 1,
                start: 2.76,
                end: 3.36,
            },
            SceneBoundary {
                scene_id: 2,
                start: 397.95,
                end: 441.9,
            },
        ];
        let cuts = SceneIntelligenceEngine::cut_times(&SceneIntelligenceResult {
            source_hash: "h".into(),
            detector: SCENE_DETECTOR.into(),
            detector_version: SCENE_DETECTOR_VERSION.into(),
            config_version: SCENE_CONFIG_VERSION.into(),
            scenes,
            fallback: false,
            created_at: "t".into(),
        });
        assert_eq!(cuts, vec![2.76, 397.95]);
        assert!(cuts.iter().all(|c| *c > 0.0));
        assert!(cuts.windows(2).all(|w| w[0] < w[1]), "cuts must be monotonic");
    }
}
