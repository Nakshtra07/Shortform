use std::collections::BTreeSet;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::models::{NormalizedTranscript, TranscriptSegment, TranscriptWord};

/// Unified sentence terminators supporting English, Hindi (Purna Viram U+0964, Deergh Viram U+0965), and common punctuation
pub const SENTENCE_TERMINATORS: [char; 7] = ['.', '!', '?', ';', ':', '।', '॥'];

pub fn ends_with_sentence_terminator(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.ends_with(SENTENCE_TERMINATORS)
}

/// Multi-signal Language & Script Classifier
/// Combines ASR metadata + Devanagari Unicode character count + Latin token count
/// to categorize transcript into: ENGLISH, HINDI, HINGLISH, or MULTILINGUAL
/// and script into: devanagari, latin, mixed, or roman_hindi
pub fn classify_language_and_script(asr_lang: &str, words: &[TranscriptWord]) -> (String, String) {
    if words.is_empty() {
        return (asr_lang.to_uppercase(), "latin".to_string());
    }

    let mut devanagari_chars = 0usize;
    let mut latin_chars = 0usize;
    let mut total_chars = 0usize;

    for w in words {
        for ch in w.text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() || ch == '।' || ch == '॥' {
                continue;
            }
            total_chars += 1;
            if ('\u{0900}'..='\u{097F}').contains(&ch) {
                devanagari_chars += 1;
            } else if ch.is_ascii_alphabetic() {
                latin_chars += 1;
            }
        }
    }

    if total_chars == 0 {
        return ("ENGLISH".to_string(), "latin".to_string());
    }

    let devanagari_ratio = devanagari_chars as f64 / total_chars as f64;
    let latin_ratio = latin_chars as f64 / total_chars as f64;

    let (lang, script) = if devanagari_ratio > 0.65 {
        ("HINDI".to_string(), "devanagari".to_string())
    } else if devanagari_ratio > 0.10 && latin_ratio > 0.15 {
        ("HINGLISH".to_string(), "mixed".to_string())
    } else if devanagari_ratio > 0.05 {
        ("MULTILINGUAL".to_string(), "mixed".to_string())
    } else if asr_lang.to_lowercase().starts_with("hi") && devanagari_ratio < 0.02 {
        ("HINGLISH".to_string(), "roman_hindi".to_string())
    } else {
        ("ENGLISH".to_string(), "latin".to_string())
    };

    (lang, script)
}

/// Transcription Quality Gate
/// Detects corrupted/hallucinated ASR output (e.g. repetitive word loops like "right right right",
/// or severe word loss where a 60s speech audio yields only 10 words).
pub fn validate_transcript_quality(transcript: &NormalizedTranscript) -> Result<(), String> {
    if transcript.duration > 15.0 && transcript.words.is_empty() {
        return Err("Transcription failed: 0 words detected in non-empty audio.".to_string());
    }

    let duration_sec = transcript.duration.max(1.0);
    let word_count = transcript.words.len();
    let words_per_sec = word_count as f64 / duration_sec;

    // Check for repetitive word loops (e.g., ASR hallucinating same token 6+ times in a row)
    let mut repeat_streak = 1;
    let mut max_repeat_streak = 1;
    for i in 1..transcript.words.len() {
        let prev = transcript.words[i - 1]
            .text
            .to_lowercase()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        let curr = transcript.words[i]
            .text
            .to_lowercase()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        if !curr.is_empty() && curr == prev {
            repeat_streak += 1;
            if repeat_streak > max_repeat_streak {
                max_repeat_streak = repeat_streak;
            }
        } else {
            repeat_streak = 1;
        }
    }

    if max_repeat_streak >= 5 && word_count < 50 {
        return Err(format!("Transcription failed quality gate: detected repetitive hallucination loop (repeated {max_repeat_streak} times)."));
    }

    // If audio is over 45s and word density is less than 0.20 words/sec, ASR likely dropped speech
    if duration_sec > 45.0 && words_per_sec < 0.20 {
        return Err(format!("Transcription failed quality gate: abnormally low word density ({words_per_sec:.2} words/sec over {duration_sec:.1}s audio)."));
    }

    Ok(())
}

pub async fn transcribe_deepgram(audio_path: &str, api_key: &str) -> Result<NormalizedTranscript> {
    let audio_file = std::path::Path::new(audio_path);
    let bytes = tokio::fs::read(audio_path)
        .await
        .with_context(|| format!("reading audio file {audio_path}"))?;

    let ext = audio_file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_else(|| "mp3".to_string());

    let content_type = match ext.as_str() {
        "mp3" => "audio/mp3",
        "m4a" | "mp4" => "audio/m4a",
        "ogg" | "opus" => "audio/ogg",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        _ => "audio/mp3",
    };

    println!(
        "[Deepgram] Uploading audio payload: {} ({} bytes, Content-Type: {})",
        audio_path,
        bytes.len(),
        content_type
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .connect_timeout(std::time::Duration::from_secs(30))
        .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
        .pool_idle_timeout(Some(std::time::Duration::from_secs(90)))
        .build()
        .context("building HTTP client for Deepgram")?;

    // Attempt 1: Primary Nova-3 with multilingual code-switching support
    let primary_url = "https://api.deepgram.com/v1/listen?model=nova-3&language=multi&smart_format=true&diarize=true&punctuate=true&filler_words=true";
    println!("[Deepgram] Calling primary endpoint: Nova-3 (multilingual)...");

    let primary_response = client
        .post(primary_url)
        .header("Authorization", format!("Token {api_key}"))
        .header("Content-Type", content_type)
        .header("Content-Length", bytes.len().to_string())
        .body(bytes.clone())
        .send()
        .await;

    let mut final_value = None;

    if let Ok(resp) = primary_response {
        if resp.status().is_success() {
            if let Ok(val) = resp.json::<Value>().await {
                if let Ok(candidate_transcript) = normalize_deepgram(val.clone()) {
                    if let Ok(()) = validate_transcript_quality(&candidate_transcript) {
                        println!(
                            "[Deepgram] Primary Nova-3 transcription succeeded with {} words.",
                            candidate_transcript.words.len()
                        );
                        return Ok(candidate_transcript);
                    } else {
                        println!("[Deepgram Quality Gate] Nova-3 transcript failed quality check. Trying fallback endpoint...");
                        final_value = Some(val);
                    }
                }
            }
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            println!("[Deepgram Warning] Primary Nova-3 request failed ({status}): {body}. Attempting fallback endpoint...");
        }
    }

    // Attempt 2: Fallback to Nova-2 with Hindi language parameter
    let fallback_url = "https://api.deepgram.com/v1/listen?model=nova-2&language=hi&smart_format=true&diarize=true&punctuate=true&filler_words=true";
    println!("[Deepgram] Calling fallback endpoint: Nova-2 (language=hi)...");

    let fallback_response = client
        .post(fallback_url)
        .header("Authorization", format!("Token {api_key}"))
        .header("Content-Type", content_type)
        .header("Content-Length", bytes.len().to_string())
        .body(bytes)
        .send()
        .await
        .context("calling Deepgram fallback")?;

    if !fallback_response.status().is_success() {
        let status = fallback_response.status();
        let body = fallback_response.text().await.unwrap_or_default();
        if let Some(val) = final_value {
            println!("[Deepgram Warning] Fallback failed ({status}): {body}. Returning initial transcript.");
            return normalize_deepgram(val);
        }
        return Err(anyhow!("Deepgram transcription failed ({status}): {body}"));
    }

    let val: Value = fallback_response
        .json()
        .await
        .context("parsing Deepgram fallback response")?;
    normalize_deepgram(val)
}

fn normalize_deepgram(value: Value) -> Result<NormalizedTranscript> {
    let alternative = value
        .pointer("/results/channels/0/alternatives/0")
        .ok_or_else(|| anyhow!("Deepgram response did not include an alternative transcript"))?;

    let detected_lang = value
        .pointer("/results/channels/0/detected_language")
        .and_then(Value::as_str)
        .or_else(|| value.pointer("/metadata/language").and_then(Value::as_str))
        .unwrap_or("en")
        .to_string();

    let duration = value
        .pointer("/metadata/duration")
        .and_then(Value::as_f64)
        .unwrap_or_default();

    let raw_words = alternative
        .get("words")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Deepgram response did not include word timestamps"))?;

    let mut speakers = BTreeSet::new();
    let mut words = Vec::with_capacity(raw_words.len());

    for word in raw_words {
        let text = word
            .get("punctuated_word")
            .or_else(|| word.get("word"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }

        let speaker = word
            .get("speaker")
            .and_then(Value::as_i64)
            .map(|speaker| format!("S{}", speaker + 1));
        if let Some(speaker) = &speaker {
            speakers.insert(speaker.clone());
        }

        words.push(TranscriptWord {
            text,
            start: word
                .get("start")
                .and_then(Value::as_f64)
                .unwrap_or_default(),
            end: word.get("end").and_then(Value::as_f64).unwrap_or_default(),
            speaker,
        });
    }

    let (classified_lang, _script) = classify_language_and_script(&detected_lang, &words);
    let language = if classified_lang == "HINDI" {
        "hi".to_string()
    } else if classified_lang == "HINGLISH" || classified_lang == "MULTILINGUAL" {
        "hi-en".to_string()
    } else {
        detected_lang
    };

    let segments = build_segments(&words);

    Ok(NormalizedTranscript {
        language,
        duration,
        speakers: speakers.into_iter().collect(),
        words,
        segments,
        raw_words: None,
        correction_metadata: None,
    })
}

pub fn build_segments(words: &[TranscriptWord]) -> Vec<TranscriptSegment> {
    let mut segments = Vec::new();
    let mut current: Option<TranscriptSegment> = None;

    for word in words {
        let should_break = current.as_ref().map_or(false, |segment| {
            let pause = word.start - segment.end;
            let speaker_changed = segment.speaker != word.speaker;
            let sentence_end = ends_with_sentence_terminator(&segment.text);
            pause > 0.85 || speaker_changed || sentence_end
        });

        if should_break {
            if let Some(segment) = current.take() {
                segments.push(segment);
            }
        }

        match &mut current {
            Some(segment) => {
                segment.end = word.end;
                segment.text.push(' ');
                segment.text.push_str(&word.text);
            }
            None => {
                current = Some(TranscriptSegment {
                    start: word.start,
                    end: word.end,
                    speaker: word.speaker.clone(),
                    text: word.text.clone(),
                });
            }
        }
    }

    if let Some(segment) = current {
        segments.push(segment);
    }

    segments
}

pub fn find_python_command() -> Option<String> {
    let candidates: &[&str] = if cfg!(windows) {
        &["python", "py", "python3"]
    } else {
        &["python3", "python"]
    };

    for &cmd in candidates {
        if let Ok(out) = std::process::Command::new(cmd)
            .args(["-c", "import whisper"])
            .output()
        {
            if out.status.success() {
                return Some(cmd.to_string());
            }
        }
    }
    for &cmd in candidates {
        if std::process::Command::new(cmd)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(cmd.to_string());
        }
    }
    None
}

pub fn whisper_cli_exists() -> bool {
    let candidates: &[&str] = if cfg!(windows) {
        &["whisper", "whisper.exe"]
    } else {
        &["whisper"]
    };

    for &cmd in candidates {
        if let Ok(out) = std::process::Command::new(cmd).arg("--help").output() {
            if out.status.success() {
                return true;
            }
        }
    }
    false
}

pub fn whisper_python_exists() -> bool {
    let candidates: &[&str] = if cfg!(windows) {
        &["python", "py", "python3"]
    } else {
        &["python3", "python"]
    };

    for &cmd in candidates {
        if let Ok(out) = std::process::Command::new(cmd)
            .args(["-c", "import whisper"])
            .output()
        {
            if out.status.success() {
                return true;
            }
        }
    }
    false
}

fn normalize_whisper_raw_json(raw: serde_json::Value) -> Result<NormalizedTranscript> {
    let language = raw
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("en")
        .to_string();

    let segments_arr = raw
        .get("segments")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("Missing 'segments' in Whisper JSON"))?;

    let duration = segments_arr
        .last()
        .and_then(|s| s.get("end").and_then(|e| e.as_f64()))
        .unwrap_or(0.0);

    let mut segments = Vec::new();
    let mut words = Vec::new();

    for seg in segments_arr {
        let start = seg.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let end = seg.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let text = seg
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        segments.push(TranscriptSegment {
            start,
            end,
            speaker: Some("S1".to_string()),
            text,
        });

        if let Some(words_arr) = seg.get("words").and_then(|v| v.as_array()) {
            for w in words_arr {
                let word_text = w
                    .get("word")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let word_start = w.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let word_end = w.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);

                words.push(TranscriptWord {
                    text: word_text,
                    start: word_start,
                    end: word_end,
                    speaker: Some("S1".to_string()),
                });
            }
        }
    }

    Ok(NormalizedTranscript {
        language,
        duration,
        speakers: vec!["S1".to_string()],
        words,
        segments,
        raw_words: None,
        correction_metadata: None,
    })
}

pub async fn transcribe_local(audio_path: &str, model_path: &str) -> Result<NormalizedTranscript> {
    let audio_path = audio_path.to_string();
    let model_path = model_path.to_string();

    if whisper_cli_exists() {
        let audio_path_buf = std::path::Path::new(&audio_path);
        let audio_dir = audio_path_buf
            .parent()
            .ok_or_else(|| anyhow!("Invalid audio path parent"))?;
        let audio_stem = audio_path_buf
            .file_stem()
            .ok_or_else(|| anyhow!("Invalid audio file stem"))?
            .to_string_lossy();

        let output_json_path = audio_dir.join(format!("{}.json", audio_stem));
        let output_json_path_str = output_json_path.to_string_lossy().to_string();
        let audio_dir_str = audio_dir.to_string_lossy().to_string();
        let audio_dir_clone = audio_dir.to_path_buf();
        let audio_stem_clone = audio_stem.to_string();

        tokio::task::spawn_blocking(move || {
            let output = std::process::Command::new("whisper")
                .arg(&audio_path)
                .args(["--model", "base"])
                .args(["--output_format", "json"])
                .args(["--output_dir", &audio_dir_str])
                .args(["--word_timestamps", "True"])
                .output()
                .context("executing whisper CLI")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                return Err(anyhow!(
                    "Whisper CLI failed:\nStderr: {}\nStdout: {}",
                    stderr,
                    stdout
                ));
            }

            let json_bytes = std::fs::read(&output_json_path_str)
                .context("reading output transcript JSON from CLI")?;
            let raw_json: serde_json::Value =
                serde_json::from_slice(&json_bytes).context("parsing output transcript JSON")?;

            // Clean up the output JSON file
            let _ = std::fs::remove_file(&output_json_path_str);

            // Clean up any extra formats whisper CLI might have written (it sometimes generates them by default)
            for ext in &["txt", "srt", "vtt", "tsv"] {
                let extra_file = audio_dir_clone.join(format!("{}.{}", audio_stem_clone, ext));
                if extra_file.exists() {
                    let _ = std::fs::remove_file(extra_file);
                }
            }

            normalize_whisper_raw_json(raw_json)
        })
        .await
        .context("spawn_blocking failed")?
    } else {
        // Resolve the directory where the model lives. We'll put transcribe.py there.
        let model_dir = std::path::Path::new(&model_path)
            .parent()
            .ok_or_else(|| anyhow!("Invalid model path"))?;

        let script_path = model_dir.join("transcribe.py");
        if !script_path.exists() {
            let script_content = r#"import sys
import json
import whisper

def main():
    if len(sys.argv) < 3:
        print("Usage: transcribe.py <audio_path> <output_json_path> [model_name]")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    output_json_path = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "base"
    
    # Load model. Automatically uses MPS on Apple Silicon if PyTorch supports it.
    model = whisper.load_model(model_name)
    
    # Transcribe with word-level timestamps without forcing English translation
    result = model.transcribe(audio_path, word_timestamps=True, task="transcribe")
    
    normalized = {
        "language": result.get("language", "en"),
        "duration": result.get("segments", [])[-1]["end"] if result.get("segments") else 0.0,
        "speakers": ["S1"],
        "words": [],
        "segments": []
    }
    
    for segment in result.get("segments", []):
        normalized["segments"].append({
            "start": segment["start"],
            "end": segment["end"],
            "speaker": "S1",
            "text": segment["text"].strip()
        })
        
        for word in segment.get("words", []):
            cleaned_text = word["word"].strip()
            normalized["words"].append({
                "text": cleaned_text,
                "start": word["start"],
                "end": word["end"],
                "speaker": "S1"
            })
            
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
"#;
            std::fs::write(&script_path, script_content).context("writing transcribe.py script")?;
        }

        let output_json_path =
            model_dir.join(format!("temp_transcript_{}.json", uuid::Uuid::new_v4()));
        let script_path_str = script_path.to_string_lossy().to_string();
        let output_json_path_str = output_json_path.to_string_lossy().to_string();

        tokio::task::spawn_blocking(move || {
            let python_bin = find_python_command().unwrap_or_else(|| "python".to_string());
            let output = std::process::Command::new(&python_bin)
                .arg(&script_path_str)
                .arg(&audio_path)
                .arg(&output_json_path_str)
                .arg("base") // default model size
                .output()
                .context("executing python transcribe.py")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                return Err(anyhow!(
                    "Python transcription script failed:\nStderr: {}\nStdout: {}",
                    stderr,
                    stdout
                ));
            }

            let json_bytes =
                std::fs::read(&output_json_path_str).context("reading output transcript JSON")?;
            let transcript: NormalizedTranscript =
                serde_json::from_slice(&json_bytes).context("parsing output transcript JSON")?;

            // Cleanup temp file
            let _ = std::fs::remove_file(&output_json_path_str);

            Ok(transcript)
        })
        .await
        .context("spawn_blocking failed")?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_deepgram_word_timestamps_and_speakers() {
        let sample_json: serde_json::Value = serde_json::json!({
            "metadata": {
                "language": "en",
                "duration": 12.5
            },
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "Hello world. How are you today?",
                                "words": [
                                    {
                                        "word": "hello",
                                        "punctuated_word": "Hello",
                                        "start": 0.0,
                                        "end": 0.5,
                                        "speaker": 0
                                    },
                                    {
                                        "word": "world",
                                        "punctuated_word": "world.",
                                        "start": 0.5,
                                        "end": 1.0,
                                        "speaker": 0
                                    },
                                    {
                                        "word": "how",
                                        "punctuated_word": "How",
                                        "start": 1.5,
                                        "end": 1.8,
                                        "speaker": 1
                                    },
                                    {
                                        "word": "are",
                                        "punctuated_word": "are",
                                        "start": 1.8,
                                        "end": 2.0,
                                        "speaker": 1
                                    },
                                    {
                                        "word": "you",
                                        "punctuated_word": "you",
                                        "start": 2.0,
                                        "end": 2.2,
                                        "speaker": 1
                                    },
                                    {
                                        "word": "today",
                                        "punctuated_word": "today?",
                                        "start": 2.2,
                                        "end": 2.8,
                                        "speaker": 1
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        });

        let normalized =
            normalize_deepgram(sample_json).expect("Failed to normalize Deepgram JSON");
        assert_eq!(normalized.language, "en");
        assert_eq!(normalized.duration, 12.5);
        assert_eq!(normalized.words.len(), 6);
        assert_eq!(normalized.speakers.len(), 2);
        assert!(normalized.speakers.contains(&"S1".to_string()));
        assert!(normalized.speakers.contains(&"S2".to_string()));
        assert_eq!(normalized.words[0].text, "Hello");
        assert_eq!(normalized.words[1].text, "world.");
        assert_eq!(normalized.words[1].speaker, Some("S1".to_string()));
        assert_eq!(normalized.words[5].text, "today?");
        assert_eq!(normalized.words[5].speaker, Some("S2".to_string()));
        assert!(!normalized.segments.is_empty());
    }

    #[test]
    fn test_classify_language_and_script_multilingual() {
        // 1. Pure English
        let en_words = vec![
            TranscriptWord {
                text: "The".to_string(),
                start: 0.0,
                end: 0.3,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "future".to_string(),
                start: 0.3,
                end: 0.8,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "of".to_string(),
                start: 0.8,
                end: 1.0,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "technology.".to_string(),
                start: 1.0,
                end: 1.8,
                speaker: Some("S1".to_string()),
            },
        ];
        let (cat, script) = classify_language_and_script("en", &en_words);
        assert_eq!(cat, "ENGLISH");
        assert_eq!(script, "latin");

        // 2. Pure Hindi (Devanagari)
        let hi_words = vec![
            TranscriptWord {
                text: "यह".to_string(),
                start: 0.0,
                end: 0.3,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "एक".to_string(),
                start: 0.3,
                end: 0.6,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "बहुत".to_string(),
                start: 0.6,
                end: 0.9,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "बड़ा".to_string(),
                start: 0.9,
                end: 1.2,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "फैसला".to_string(),
                start: 1.2,
                end: 1.6,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "था।".to_string(),
                start: 1.6,
                end: 2.0,
                speaker: Some("S1".to_string()),
            },
        ];
        let (cat, script) = classify_language_and_script("hi", &hi_words);
        assert_eq!(cat, "HINDI");
        assert_eq!(script, "devanagari");

        // 3. Hinglish (Mixed Code-Switching)
        let hinglish_words = vec![
            TranscriptWord {
                text: "यह".to_string(),
                start: 0.0,
                end: 0.3,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "startup".to_string(),
                start: 0.3,
                end: 0.8,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "का".to_string(),
                start: 0.8,
                end: 1.0,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "turning".to_string(),
                start: 1.0,
                end: 1.4,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "point".to_string(),
                start: 1.4,
                end: 1.8,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "था।".to_string(),
                start: 1.8,
                end: 2.2,
                speaker: Some("S1".to_string()),
            },
        ];
        let (cat, script) = classify_language_and_script("hi", &hinglish_words);
        assert_eq!(cat, "HINGLISH");
        assert_eq!(script, "mixed");

        // 4. Roman Hindi (Latin Script Hindi)
        let roman_hi_words = vec![
            TranscriptWord {
                text: "Mujhe".to_string(),
                start: 0.0,
                end: 0.3,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "lagta".to_string(),
                start: 0.3,
                end: 0.6,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "hai".to_string(),
                start: 0.6,
                end: 0.9,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "ki".to_string(),
                start: 0.9,
                end: 1.1,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "yeh".to_string(),
                start: 1.1,
                end: 1.4,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "sahi".to_string(),
                start: 1.4,
                end: 1.7,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "decision".to_string(),
                start: 1.7,
                end: 2.2,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "tha.".to_string(),
                start: 2.2,
                end: 2.6,
                speaker: Some("S1".to_string()),
            },
        ];
        let (cat, script) = classify_language_and_script("hi", &roman_hi_words);
        assert_eq!(cat, "HINGLISH");
        assert_eq!(script, "roman_hindi");
    }

    #[test]
    fn test_validate_transcript_quality_and_repetition_loop() {
        // Valid transcript
        let valid_words = vec![
            TranscriptWord {
                text: "Aapko".to_string(),
                start: 0.0,
                end: 0.4,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "kya".to_string(),
                start: 0.4,
                end: 0.8,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "lagta".to_string(),
                start: 0.8,
                end: 1.2,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "hai?".to_string(),
                start: 1.2,
                end: 1.6,
                speaker: Some("S1".to_string()),
            },
        ];
        let t_valid = NormalizedTranscript {
            language: "hi".to_string(),
            duration: 5.0,
            speakers: vec!["S1".to_string()],
            words: valid_words,
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };
        assert!(validate_transcript_quality(&t_valid).is_ok());

        // Repetition loop defect
        let loop_words = vec![
            TranscriptWord {
                text: "right".to_string(),
                start: 0.0,
                end: 0.5,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "right".to_string(),
                start: 0.5,
                end: 1.0,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "right".to_string(),
                start: 1.0,
                end: 1.5,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "right".to_string(),
                start: 1.5,
                end: 2.0,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "right".to_string(),
                start: 2.0,
                end: 2.5,
                speaker: Some("S1".to_string()),
            },
            TranscriptWord {
                text: "right".to_string(),
                start: 2.5,
                end: 3.0,
                speaker: Some("S1".to_string()),
            },
        ];
        let t_loop = NormalizedTranscript {
            language: "en".to_string(),
            duration: 10.0,
            speakers: vec!["S1".to_string()],
            words: loop_words,
            segments: vec![],
            raw_words: None,
            correction_metadata: None,
        };
        let err = validate_transcript_quality(&t_loop).unwrap_err();
        assert!(err.contains("repetitive hallucination loop"));
    }

    #[test]
    fn test_ends_with_sentence_terminator_hindi() {
        assert!(ends_with_sentence_terminator("था।"));
        assert!(ends_with_sentence_terminator("फैसला॥"));
        assert!(ends_with_sentence_terminator("Hello."));
        assert!(ends_with_sentence_terminator("Why?"));
        assert!(ends_with_sentence_terminator("Awesome!"));
        assert!(!ends_with_sentence_terminator("comma,"));
        assert!(!ends_with_sentence_terminator("word"));
    }
}
