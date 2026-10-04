#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 2 — Speaker Diarization Sidecar

Provides speaker diarization with multiple backends:
- Primary: Deepgram Nova-3 (cloud, requires API key)
- Fallback: pyannote.audio 3.1 (local, requires HF token)
- Fallback: WhisperX (local, requires whisperx)

Outputs canonical SpeakerDiarizationResult JSON to stdout.
"""

import sys
import os
import json
import hashlib
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# ─── Configuration ─────────────────────────────────────────────────────────────

DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LANGUAGE = "multi"
DEEPGRAM_DIARIZE = True
DEEPGRAM_SMART_FORMAT = True
DEEPGRAM_PUNCTUATE = True
DEEPGRAM_FILLER_WORDS = True

# ─── Deepgram HTTP / long-form configuration ───────────────────────────────────
#
# The request strategy is bounded, not a single unbounded upload. A source
# video can be arbitrarily long, so the audio payload is split into chunks that
# each stay under DEEPGRAM_MAX_UPLOAD_BYTES and are transcribed independently,
# then merged back into one ordered, globally-timestamped segment list.
#
# Timeouts are explicit and configurable. NOTE ON WRITE TIMEOUTS: `requests`
# has no write-timeout knob, and urllib3 reuses the *connect* timeout as the
# socket timeout for the request-body write (urllib3/connectionpool.py sets
# `conn.timeout = timeout_obj.connect_timeout` before `conn.request()` sends
# the body, and only switches to the read timeout afterwards). The body write
# is therefore governed by the connect value passed here. It is derived from
# the payload size and a conservative sustained-throughput floor rather than
# hardcoded, so a large payload automatically gets a proportionally larger
# write budget and a small one is not given a needlessly long hang.
#
# Never log DEEPGRAM_API_KEY. The key is only ever sent in the Authorization
# header and is redacted from every diagnostic below.

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"[Diarization] Invalid {name}={raw!r}; using default {default}", file=sys.stderr)
        return default
    return value if value > 0 else default

def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, float(default)))

# Read timeout (server processing + response) in seconds.
DEEPGRAM_READ_TIMEOUT = _env_float("AUTOSHORTS_DEEPGRAM_READ_TIMEOUT", 600.0)
# Maximum bytes per single upload. 0 disables chunking (single request).
DEEPGRAM_MAX_UPLOAD_BYTES = _env_int("AUTOSHORTS_DEEPGRAM_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
# Conservative sustained upload floor used to size the write budget (bytes/sec).
DEEPGRAM_MIN_UPLOAD_BYTES_PER_SEC = _env_int("AUTOSHORTS_DEEPGRAM_MIN_UPLOAD_BYTES_PER_SEC", 64 * 1024)
# Safety multiplier applied to the estimated write time.
DEEPGRAM_WRITE_SAFETY_FACTOR = _env_float("AUTOSHORTS_DEEPGRAM_WRITE_SAFETY_FACTOR", 2.0)
# Never go below this write budget regardless of payload size.
DEEPGRAM_MIN_WRITE_TIMEOUT = _env_float("AUTOSHORTS_DEEPGRAM_MIN_WRITE_TIMEOUT", 30.0)
# Smallest chunk duration used when splitting long media.
DEEPGRAM_MIN_CHUNK_SEC = _env_float("AUTOSHORTS_DEEPGRAM_MIN_CHUNK_SEC", 60.0)
# Seconds of overlap requested on each side of a chunk cut so a word is never
# clipped; overlapping output is trimmed back to the nominal chunk boundary.
DEEPGRAM_CHUNK_OVERLAP_SEC = _env_float("AUTOSHORTS_DEEPGRAM_CHUNK_OVERLAP_SEC", 0.25)
# Window (seconds) either side of a chunk boundary used to re-align speaker
# labels, which Deepgram re-numbers from 0 on every independent request.
DEEPGRAM_SPEAKER_ALIGN_WINDOW = _env_float("AUTOSHORTS_DEEPGRAM_SPEAKER_ALIGN_WINDOW", 60.0)
# Upper bound on uploads made per source; protects against a pathological
# duration estimate producing an unbounded number of requests.
DEEPGRAM_MAX_CHUNKS = _env_int("AUTOSHORTS_DEEPGRAM_MAX_CHUNKS", 64)

PYANNOTE_SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
PYANNOTE_EMBEDDING_MODEL = "pyannote/embedding"
PYANNOTE_SD_MODEL = "pyannote/speaker-diarization-3.1"

WHISPERX_MODEL = "large-v3"
WHISPERX_BATCH_SIZE = 16
WHISPERX_COMPUTE_TYPE = "float32"  # or "int8" for CPU

# ─── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class DiarizedSpeaker:
    diarization_id: str
    total_speech_sec: float
    segment_count: int
    avg_confidence: float

@dataclass
class DiarizedSegment:
    speaker_id: str
    start: float
    end: float
    confidence: float

@dataclass
class SpeakerDiarizationResult:
    source_hash: str
    model: str
    version: str
    speakers: List[DiarizedSpeaker]
    segments: List[DiarizedSegment]
    confidence: float
    created_at: str

# ─── Utility Functions ─────────────────────────────────────────────────────────

def compute_source_hash(filepath: str) -> str:
    """Compute SHA256 hash of source file for cache key."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        f = float(val)
        return default if (f != f or f == float('inf') or f == float('-inf')) else f
    except:
        return default

def _human_bytes(n: Optional[int]) -> str:
    """Approximate size for logs. Never includes credentials."""
    if n is None:
        return "unknown"
    if n >= 1024 * 1024:
        return f"{n / (1024.0 * 1024.0):.1f}MB"
    if n >= 1024:
        return f"{n / 1024.0:.1f}KB"
    return f"{n}B"

def merge_adjacent_segments(segments: List[Dict], max_gap: float = 0.5) -> List[Dict]:
    """Merge adjacent segments from same speaker within max_gap seconds."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x["start"])
    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] <= max_gap:
            last["end"] = seg["end"]
            # Average confidence
            last["confidence"] = (last["confidence"] + seg["confidence"]) / 2.0
        else:
            merged.append(seg)
    return merged

# ─── Audio probing helpers ─────────────────────────────────────────────────────

# Audio-only containers. Video containers (.mp4/.mov/.webm/.mkv) are
# deliberately excluded: they are exactly what used to be uploaded to Deepgram
# wholesale, and the sidecar must refuse them rather than pay for that.
AUDIO_SUFFIXES = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac", ".wma")

VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".flv", ".m4v", ".wmv", ".mpg", ".mpeg")

def _content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {
        "mp3": "audio/mpeg",
        "m4a": "audio/m4a",
        "aac": "audio/aac",
        "mp4": "audio/mp4",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "wma": "audio/x-ms-wma",
    }.get(ext, "audio/mpeg")

def probe_audio_duration(path: str) -> Optional[float]:
    """Media duration in seconds, or None when it cannot be determined.

    Uses ffprobe when available; falls back to parsing the file directly for
    WAV/AIFF headers. Returns None rather than guessing so callers can decide
    explicitly instead of inheriting a fabricated length.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode == 0 and out.stdout.strip():
            value = float(out.stdout.strip().splitlines()[0])
            if value > 0:
                return value
    except Exception:
        pass
    return _wav_duration(path)

def _wav_duration(path: str) -> Optional[float]:
    """Duration of a PCM WAV file from its header, without dependencies."""
    try:
        import struct
        with open(path, "rb") as fh:
            riff = fh.read(12)
            if len(riff) < 12 or riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
                return None
            byte_rate = 0
            while True:
                header = fh.read(8)
                if len(header) < 8:
                    return None
                chunk_id, chunk_size = struct.unpack("<4sI", header)
                if chunk_id == b"fmt ":
                    body = fh.read(chunk_size)
                    if len(body) < 16:
                        return None
                    _fmt, channels, sample_rate, byte_rate = struct.unpack("<HHII", body[:12])
                    if byte_rate > 0:
                        data_size = max(0, os.path.getsize(path) - fh.tell() - 8)
                        return data_size / float(byte_rate)
                    if channels and sample_rate:
                        return None
                elif chunk_id == b"data":
                    if byte_rate > 0:
                        return chunk_size / float(byte_rate)
                    return None
                else:
                    fh.seek(chunk_size + (chunk_size % 2), os.SEEK_CUR)
    except Exception:
        return None
    return None

# ─── Deepgram Backend ──────────────────────────────────────────────────────────

class DeepgramError(RuntimeError):
    """Deepgram request failed. Carries an explicit failure stage for logs."""

    def __init__(self, message: str, stage: str = "request", status: Optional[int] = None):
        super().__init__(message)
        self.stage = stage
        self.status = status

def compute_write_timeout(payload_bytes: int) -> float:
    """Write budget for a body of `payload_bytes`.

    urllib3 applies the *connect* timeout to the body write, so this value is
    what actually bounds the upload. It scales with the payload: at the
    configured sustained-throughput floor, times a safety factor, and never
    drops below DEEPGRAM_MIN_WRITE_TIMEOUT.
    """
    estimated = (payload_bytes / float(DEEPGRAM_MIN_UPLOAD_BYTES_PER_SEC)) * DEEPGRAM_WRITE_SAFETY_FACTOR
    return max(DEEPGRAM_MIN_WRITE_TIMEOUT, estimated)

def plan_chunks(duration_sec: float, bytes_per_sec: float) -> List[Tuple[float, float]]:
    """Nominal (start, end) chunks covering [0, duration).

    Sized so each encoded chunk stays under DEEPGRAM_MAX_UPLOAD_BYTES. Returns a
    single whole-file chunk when the payload already fits, which keeps short
    media on exactly the previous one-request path.
    """
    if duration_sec <= 0 or bytes_per_sec <= 0 or DEEPGRAM_MAX_UPLOAD_BYTES <= 0:
        return [(0.0, max(0.0, duration_sec))]

    total_bytes = duration_sec * bytes_per_sec
    if total_bytes <= DEEPGRAM_MAX_UPLOAD_BYTES:
        return [(0.0, duration_sec)]

    chunk_dur = DEEPGRAM_MAX_UPLOAD_BYTES / bytes_per_sec
    chunk_dur = max(chunk_dur, DEEPGRAM_MIN_CHUNK_SEC)
    count = int(duration_sec / chunk_dur) + (1 if duration_sec % chunk_dur else 0)
    if count > DEEPGRAM_MAX_CHUNKS:
        chunk_dur = duration_sec / float(DEEPGRAM_MAX_CHUNKS)
        count = DEEPGRAM_MAX_CHUNKS

    chunks: List[Tuple[float, float]] = []
    cursor = 0.0
    for _ in range(count):
        end = min(duration_sec, cursor + chunk_dur)
        if end <= cursor:
            break
        chunks.append((cursor, end))
        cursor = end
    if cursor < duration_sec:
        chunks.append((cursor, duration_sec))
    return chunks

def _ffmpeg_slice(src: str, start: float, end: float, out_path: str) -> None:
    """Encode one time range of the source to a small mono MP3 via ffmpeg."""
    import subprocess
    # Re-encode (not stream copy) so each chunk is independently decodable and
    # cuts land on a packet boundary rather than a keyframe.
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{max(0.0, end - start):.3f}",
        "-i", src,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "64k",
        out_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if out.returncode != 0 or not os.path.exists(out_path):
        raise DeepgramError(
            f"ffmpeg chunk extraction failed (rc={out.returncode}): {out.stderr.strip()[-300:]}",
            stage="chunk-extract",
        )

def _post_chunk(session, url: str, params: Dict[str, str], headers: Dict[str, str],
                chunk_path: str, write_timeout: float) -> Dict[str, Any]:
    """Stream one chunk to Deepgram without loading it fully into memory.

    The file handle is passed straight to `requests`, which streams it in
    blocks and sets Content-Length from the file size, so a large chunk is
    never materialised as a second full copy in RAM.
    """
    timeout = (write_timeout, DEEPGRAM_READ_TIMEOUT)
    with open(chunk_path, "rb") as fh:
        try:
            response = session.post(url, params=params, headers=headers, data=fh, timeout=timeout)
        except Exception as exc:
            raise DeepgramError(
                f"upload/response failed: {type(exc).__name__}: {exc}",
                stage="write" if "write" in str(exc).lower() else "request",
            ) from exc
    if not response.ok:
        raise DeepgramError(
            f"Deepgram API error: {response.status_code} - {response.text[:300]}",
            stage="api", status=response.status_code,
        )
    try:
        return response.json()
    except Exception as exc:
        raise DeepgramError(f"Deepgram response was not valid JSON: {exc}", stage="parse") from exc

def _words_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        alternatives = payload["results"]["channels"][0]["alternatives"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepgramError(f"unexpected Deepgram response shape: {exc}", stage="parse") from exc
    words = alternatives[0].get("words", [])
    if not words:
        raise DeepgramError("No words returned from Deepgram", stage="parse")
    return words

def _words_to_segments(words: List[Dict[str, Any]], offset: float,
                       allowed_speakers: Optional[set] = None) -> List[Dict]:
    """Group consecutive words into contiguous speaker segments.

    `offset` re-bases chunk-local timestamps into global source-video time.
    `allowed_speakers` restricts output to the caller's canonical label set so
    a per-chunk re-numbering cannot introduce a new speaker identity.
    """
    segments: List[Dict] = []
    current_speaker = None
    current_start = None
    current_end = None
    speaker_confidences: Dict[str, float] = {}

    for w in words:
        speaker = w.get("speaker")
        if speaker is None:
            continue
        speaker_id = f"S{int(speaker) + 1}"
        if allowed_speakers is not None and speaker_id not in allowed_speakers:
            continue
        start = safe_float(w.get("start")) + offset
        end = safe_float(w.get("end")) + offset
        confidence = safe_float(w.get("confidence"), 0.5)

        if speaker_id != current_speaker:
            if current_speaker is not None:
                segments.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "confidence": speaker_confidences.get(current_speaker, 0.5),
                })
            current_speaker = speaker_id
            current_start = start
            current_end = end
            speaker_confidences[speaker_id] = confidence
        else:
            current_end = end
            speaker_confidences[speaker_id] = (
                speaker_confidences[speaker_id] + confidence
            ) / 2.0

    if current_speaker is not None:
        segments.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "confidence": speaker_confidences.get(current_speaker, 0.5),
        })
    return segments

def _align_speakers(prev_segments: List[Dict], next_segments: List[Dict],
                    boundary: float) -> List[Dict]:
    """Map chunk-local speaker labels onto the established identity set.

    Deepgram numbers speakers from 0 on every independent request, so chunk N
    may call the guest "S1" where chunk N-1 called the host "S1". Each incoming
    label is matched against the established speakers by maximum speech overlap
    inside a window around the boundary; unmatched labels are left unmapped
    rather than being guessed into an existing identity.
    """
    known = sorted({s["speaker"] for s in prev_segments})
    if not known or not next_segments:
        return next_segments

    window_start = boundary - DEEPGRAM_SPEAKER_ALIGN_WINDOW
    def overlap(seg: Dict) -> float:
        lo = max(seg["start"], window_start)
        hi = min(seg["end"], boundary + DEEPGRAM_SPEAKER_ALIGN_WINDOW)
        return max(0.0, hi - lo)

    # Score every (incoming label, established identity) pair.
    incoming_labels = sorted({s["speaker"] for s in next_segments})
    score: Dict[Tuple[str, str], float] = {}
    for label in incoming_labels:
        for identity in known:
            same = sum(overlap(s) for s in next_segments if s["speaker"] == label)
            other = sum(overlap(s) for s in prev_segments if s["speaker"] == identity)
            score[(label, identity)] = same * other

    mapping: Dict[str, str] = {}
    used = set()
    # Greedy best-first assignment keeps this deterministic and one-to-one.
    for pair, value in sorted(score.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        label, identity = pair
        if value <= 0.0 or label in mapping or identity in used:
            continue
        mapping[label] = identity
        used.add(identity)

    for seg in next_segments:
        if seg["speaker"] in mapping:
            seg["speaker"] = mapping[seg["speaker"]]
    return next_segments

def run_deepgram_diarization(audio_path: str, api_key: str) -> Tuple[str, str, List[Dict], float]:
    """Run Deepgram Nova-3 diarization over an AUDIO file.

    Accepts audio only. A video container is rejected up front rather than
    being uploaded wholesale: sending the source video wastes an order of
    magnitude more bytes than the audio it contains and is what made long
    sources fail during the upload write.
    """
    import requests

    ext = os.path.splitext(audio_path)[1].lower()
    if ext in VIDEO_SUFFIXES:
        raise DeepgramError(
            f"Deepgram requires extracted audio, got video container {ext!r} for "
            f"{os.path.basename(audio_path)}. Run media::extract_audio() first.",
            stage="input-validation",
        )
    if ext not in AUDIO_SUFFIXES:
        raise DeepgramError(
            f"Deepgram requires a supported audio container, got {ext or 'no extension'!r} for "
            f"{os.path.basename(audio_path)}.",
            stage="input-validation",
        )

    if not api_key:
        raise DeepgramError("Deepgram API key is empty", stage="config")

    params = {
        "model": DEEPGRAM_MODEL,
        "language": DEEPGRAM_LANGUAGE,
        "diarize": "true" if DEEPGRAM_DIARIZE else "false",
        "smart_format": "true" if DEEPGRAM_SMART_FORMAT else "false",
        "punctuate": "true" if DEEPGRAM_PUNCTUATE else "false",
        "filler_words": "true" if DEEPGRAM_FILLER_WORDS else "false",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": _content_type_for(audio_path),
    }

    try:
        payload_bytes = os.path.getsize(audio_path)
    except OSError as exc:
        raise DeepgramError(f"cannot stat audio file: {exc}", stage="input-validation") from exc

    duration = probe_audio_duration(audio_path)
    if duration is None:
        # Never let an unmeasurable duration silently become a single unbounded
        # upload -- that is the exact failure being fixed. Without a duration we
        # cannot time-slice, so fail loudly at input validation instead.
        raise DeepgramError(
            f"could not determine duration of {os.path.basename(audio_path)} "
            f"({_human_bytes(payload_bytes)}); refusing to upload an unbounded payload. "
            "Install ffprobe or supply a decodable audio file.",
            stage="input-validation",
        )

    bytes_per_sec = payload_bytes / duration
    chunks = plan_chunks(duration, bytes_per_sec)
    single_request = len(chunks) <= 1

    # ── Diagnostics: everything needed to diagnose the next failure. ──
    print(
        "[Diarization] deepgram: format={} duration={} payload={} requests={} "
        "strategy={} write_timeout={}s read_timeout={}s".format(
            ext.lstrip(".") or "unknown",
            f"{duration:.1f}s",
            _human_bytes(payload_bytes),
            len(chunks),
            "single-request" if single_request
            else f"chunked({len(chunks)}x <= {_human_bytes(DEEPGRAM_MAX_UPLOAD_BYTES)})",
            "per-chunk" if not single_request else f"{compute_write_timeout(payload_bytes):.0f}",
            DEEPGRAM_READ_TIMEOUT,
        ),
        file=sys.stderr,
    )

    import tempfile
    session = requests.Session()
    all_segments: List[Dict] = []
    chunk_errors: List[str] = []
    import math as _math

    try:
        for index, (start, end) in enumerate(chunks):
            nominal_start, nominal_end = start, end
            is_only = single_request

            if is_only:
                chunk_path = audio_path
                offset = 0.0
                write_timeout = compute_write_timeout(payload_bytes)
            else:
                # Pad the extraction window so a word is never clipped, then
                # trim the returned words back to the nominal range.
                pad = DEEPGRAM_CHUNK_OVERLAP_SEC
                tmp_dir = tempfile.mkdtemp(prefix="as_diar_chunk_")
                chunk_path = os.path.join(tmp_dir, f"chunk_{index:04d}.mp3")
                _ffmpeg_slice(audio_path, max(0.0, start - pad), min(duration or end, end + pad), chunk_path)
                try:
                    chunk_bytes = os.path.getsize(chunk_path)
                except OSError:
                    chunk_bytes = 0
                offset = max(0.0, start - pad) if start > 0 else 0.0
                write_timeout = compute_write_timeout(chunk_bytes)
                print(
                    f"[Diarization] deepgram chunk {index + 1}/{len(chunks)}: "
                    f"global={start:.1f}-{end:.1f}s payload={_human_bytes(chunk_bytes)} "
                    f"write_timeout={write_timeout:.0f}s",
                    file=sys.stderr,
                )

            try:
                data = _post_chunk(session, DEEPGRAM_URL, params, headers, chunk_path, write_timeout)
                words = _words_from_payload(data)
                raw_segments = _words_to_segments(words, offset)

                # Trim the padded overlap back to the nominal chunk so no word
                # is counted twice across a boundary.
                if not is_only and (nominal_start > 0 or nominal_end < (duration or nominal_end)):
                    clamped = []
                    for s in raw_segments:
                        if s["end"] <= nominal_start or s["start"] >= nominal_end:
                            continue
                        s = dict(s)
                        s["start"] = max(s["start"], nominal_start)
                        s["end"] = min(s["end"], nominal_end)
                        if s["end"] > s["start"]:
                            clamped.append(s)
                    raw_segments = clamped

                if not raw_segments:
                    print(f"[Diarization] deepgram chunk {index + 1}/{len(chunks)} returned no speech",
                          file=sys.stderr)
                    continue

                if all_segments and not is_only:
                    raw_segments = _align_speakers(all_segments, raw_segments, nominal_start)
                # Merge against the accumulated list so ordering and
                # adjacency hold across the whole source, not per chunk.
                all_segments = merge_adjacent_segments(all_segments + raw_segments)
            except DeepgramError as exc:
                if is_only:
                    # Single-request path: preserve the original all-or-nothing
                    # contract so the caller still sees a hard failure.
                    raise
                # Chunked path: one bad chunk must not discard the chunks that
                # already succeeded. Recorded and reported at the end.
                message = f"chunk {index + 1}/{len(chunks)} [{nominal_start:.1f}-{nominal_end:.1f}s] " \
                          f"stage={exc.stage}: {exc}"
                chunk_errors.append(message)
                print(f"[Diarization] deepgram {message}", file=sys.stderr)
            finally:
                if not is_only:
                    try:
                        os.remove(chunk_path)
                        os.rmdir(os.path.dirname(chunk_path))
                    except OSError:
                        pass
    finally:
        session.close()

    if not all_segments:
        detail = chunk_errors[0] if chunk_errors else "no usable speaker segments"
        raise DeepgramError(
            f"Deepgram produced no segments (failed_chunks={len(chunk_errors)}): {detail}",
            stage="all-chunks-failed",
        )
    if chunk_errors:
        # Partial success is still success, but must never be silent.
        print(
            f"[Diarization] deepgram WARNING: {len(chunk_errors)}/{len(chunks)} chunks failed; "
            f"using {len(all_segments)} segments from the remainder. First error: {chunk_errors[0]}",
            file=sys.stderr,
        )

    segments = sorted(all_segments, key=lambda s: s["start"])
    if not _math.isfinite(segments[0]["start"]):
        raise DeepgramError("Deepgram produced a non-finite timestamp", stage="parse")

    speakers = []
    for spk_id in sorted(set(s["speaker"] for s in segments)):
        spk_segments = [s for s in segments if s["speaker"] == spk_id]
        total_speech = sum(s["end"] - s["start"] for s in spk_segments)
        avg_conf = sum(s["confidence"] for s in spk_segments) / len(spk_segments)
        speakers.append({
            "diarization_id": spk_id,
            "total_speech_sec": total_speech,
            "segment_count": len(spk_segments),
            "avg_confidence": avg_conf
        })

    overall_conf = sum(s["confidence"] for s in segments) / len(segments) if segments else 0.0

    print(
        f"[Diarization] deepgram succeeded: {len(segments)} segments, "
        f"{len(speakers)} speakers over {duration or 0:.1f}s",
        file=sys.stderr,
    )
    return "deepgram", DEEPGRAM_MODEL, segments, overall_conf


# ─── Pyannote Backend ──────────────────────────────────────────────────────────

def run_pyannote_diarization(audio_path: str, hf_token: str) -> Tuple[str, str, List[Dict], float]:
    """Run pyannote.audio 3.1 speaker diarization."""
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        raise RuntimeError("pyannote.audio not installed. Run: pip install pyannote.audio")
    
    # Load pipeline
    pipeline = Pipeline.from_pretrained(
        PYANNOTE_SD_MODEL,
        use_auth_token=hf_token
    )
    
    # Use GPU if available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    
    # Run diarization
    diarization = pipeline(audio_path)
    
    # Parse segments
    segments = []
    speakers = set()
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
            "confidence": 0.9  # pyannote doesn't provide per-segment confidence
        })
    
    # Merge adjacent
    segments = merge_adjacent_segments(segments)
    
    # Compute speaker stats
    speaker_list = []
    for spk_id in sorted(speakers):
        spk_segments = [s for s in segments if s["speaker"] == spk_id]
        total_speech = sum(s["end"] - s["start"] for s in spk_segments)
        avg_conf = sum(s["confidence"] for s in spk_segments) / len(spk_segments)
        speaker_list.append({
            "diarization_id": spk_id,
            "total_speech_sec": total_speech,
            "segment_count": len(spk_segments),
            "avg_confidence": avg_conf
        })
    
    overall_conf = sum(s["confidence"] for s in segments) / len(segments) if segments else 0.0
    
    return "pyannote", PYANNOTE_SD_MODEL, segments, overall_conf

# ─── WhisperX Backend ──────────────────────────────────────────────────────────

def run_whisperx_diarization(audio_path: str, hf_token: str) -> Tuple[str, str, List[Dict], float]:
    """Run WhisperX diarization (requires whisperx)."""
    try:
        import whisperx
        import torch
    except ImportError:
        raise RuntimeError("whisperx not installed. Run: pip install whisperx")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = whisperx.load_model(WHISPERX_MODEL, device, compute_type=WHISPERX_COMPUTE_TYPE)
    
    # Transcribe
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=WHISPERX_BATCH_SIZE)
    
    # Align
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Diarize
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    
    # Assign speakers
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Parse segments
    segments = []
    speakers = set()
    
    for seg in result["segments"]:
        speaker = seg.get("speaker", "S1")
        speakers.add(speaker)
        segments.append({
            "speaker": speaker,
            "start": seg["start"],
            "end": seg["end"],
            "confidence": seg.get("avg_logprob", 0.5)
        })
    
    # Merge adjacent
    segments = merge_adjacent_segments(segments)
    
    # Compute speaker stats
    speaker_list = []
    for spk_id in sorted(speakers):
        spk_segments = [s for s in segments if s["speaker"] == spk_id]
        total_speech = sum(s["end"] - s["start"] for s in spk_segments)
        avg_conf = sum(s["confidence"] for s in spk_segments) / len(spk_segments)
        speaker_list.append({
            "diarization_id": spk_id,
            "total_speech_sec": total_speech,
            "segment_count": len(spk_segments),
            "avg_confidence": avg_conf
        })
    
    overall_conf = sum(s["confidence"] for s in segments) / len(segments) if segments else 0.0
    
    return "whisperx", WHISPERX_MODEL, segments, overall_conf

# ─── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AutoShorts Speaker Diarization")
    parser.add_argument("source", help="Path to source audio/video file")
    parser.add_argument("--output", help="Output JSON file path (default: stdout)")
    parser.add_argument("--deepgram-key", help="Deepgram API key (env: DEEPGRAM_API_KEY)")
    parser.add_argument("--hf-token", help="HuggingFace token for pyannote/whisperx (env: HF_TOKEN)")
    parser.add_argument("--model", choices=["auto", "deepgram", "pyannote", "whisperx"], default="auto",
                        help="Diarization model to use")
    parser.add_argument("--cache-dir", help="Cache directory for results")
    parser.add_argument("--source-hash", help="SHA256 of the SOURCE VIDEO, for cache/DB identity")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cached")
    
    args = parser.parse_args()
    
    # Resolve credentials
    deepgram_key = args.deepgram_key or os.environ.get("DEEPGRAM_API_KEY")
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    source_path = args.source
    if not os.path.exists(source_path):
        print(json.dumps({"error": f"Source file not found: {source_path}"}), file=sys.stderr)
        sys.exit(1)
    
    # Identity for cache + DB. The Rust engine keys diarization on the SOURCE
    # VIDEO hash, so prefer the hash it supplies. Without it, hash whatever we
    # were given -- which is already the correct identity when the sidecar is
    # handed the source video directly.
    source_hash = args.source_hash or compute_source_hash(source_path)
    
    # Check cache
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "autoshorts" / "diarization"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{source_hash}.json"
    
    if not args.force and cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        # Verify model matches
        if args.model != "auto" and cached.get("model") != args.model:
            pass  # Will re-run
        else:
            print(json.dumps(cached, indent=2))
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(cached, f, indent=2)
            return
    
    # Run diarization with fallback chain
    segments = []
    speakers = []
    model_used = "none"
    model_version = "unknown"
    overall_confidence = 0.0
    last_error = None
    
    # Model selection logic
    models_to_try = []
    if args.model == "auto" or args.model == "deepgram":
        if deepgram_key:
            models_to_try.append(("deepgram", lambda: run_deepgram_diarization(args.source, deepgram_key)))
    if args.model == "auto" or args.model == "pyannote":
        if hf_token:
            models_to_try.append(("pyannote", lambda: run_pyannote_diarization(args.source, hf_token)))
    if args.model == "auto" or args.model == "whisperx":
        if hf_token:
            models_to_try.append(("whisperx", lambda: run_whisperx_diarization(args.source, hf_token)))
    
    # Fallback: if no models configured but auto, try Deepgram without key (will fail fast)
    if not models_to_try and args.model == "auto":
        models_to_try.append(("deepgram", lambda: run_deepgram_diarization(args.source, "")))
    
    for model_name, model_fn in models_to_try:
        try:
            print(f"[Diarization] Trying {model_name}...", file=sys.stderr)
            model_used, model_version, segments, overall_conf = model_fn()
            print(f"[Diarization] {model_name} succeeded: {len(segments)} segments", file=sys.stderr)
            break
        except Exception as e:
            last_error = e
            # Report the failure stage when the backend provides one. The
            # message itself never contains credentials -- the key only ever
            # travels in the Authorization header.
            stage = getattr(e, "stage", None)
            status = getattr(e, "status", None)
            detail = f"stage={stage}" if stage else type(e).__name__
            if status is not None:
                detail += f" http_status={status}"
            print(f"[Diarization] {model_name} failed ({detail}): {e}", file=sys.stderr)
            continue
    else:
        # All models failed
        error_msg = f"All diarization models failed. Last error: {last_error}"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    # Normalize segment dicts so the dataclasses can consume them: backends
    # emit "speaker"; DiarizedSegment expects "speaker_id". Confidence is
    # clamped to [0, 1] (WhisperX avg_logprob is a negative log-probability,
    # not a 0-1 confidence — converted via exp() where applicable upstream).
    def _norm_confidence(v):
        try:
            c = float(v)
        except (TypeError, ValueError):
            return 0.5
        if c < 0.0:
            import math
            c = math.exp(max(c, -10.0))
        return min(1.0, max(0.0, c))

    normalized_segments = []
    for s in segments:
        sid = s.get("speaker", s.get("speaker_id", "S1"))
        normalized_segments.append({
            "speaker": sid,
            "speaker_id": sid,
            "start": float(s["start"]),
            "end": float(s["end"]),
            "confidence": _norm_confidence(s.get("confidence", 0.5)),
        })
    segments = normalized_segments
    overall_confidence = _norm_confidence(overall_conf)

    # Build speaker stats
    speaker_ids = sorted(set(s["speaker"] for s in segments))
    speaker_list = []
    for spk_id in speaker_ids:
        spk_segments = [s for s in segments if s["speaker"] == spk_id]
        total_speech = sum(s["end"] - s["start"] for s in spk_segments)
        avg_conf = sum(s["confidence"] for s in spk_segments) / len(spk_segments)
        speaker_list.append(DiarizedSpeaker(
            diarization_id=spk_id,
            total_speech_sec=total_speech,
            segment_count=len(spk_segments),
            avg_confidence=avg_conf
        ))
    
    # Build final result
    result = SpeakerDiarizationResult(
        source_hash=source_hash,
        model=model_used,
        version=model_version,
        speakers=speaker_list,
        segments=[
            DiarizedSegment(
                speaker_id=s.get("speaker_id", s.get("speaker", "S1")),
                start=float(s["start"]),
                end=float(s["end"]),
                confidence=float(s.get("confidence", 0.5)),
            )
            for s in segments
        ],
        confidence=overall_confidence,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    # Serialize
    result_dict = asdict(result)
    # Convert dataclasses to dicts for JSON
    result_dict["speakers"] = [asdict(s) for s in result.speakers]
    result_dict["segments"] = [asdict(s) for s in result.segments]

    # Rust contract (Phase 2): the Rust SpeakerIntelligenceEngine deserializes
    # this JSON into serde models with rename_all = "camelCase", which require
    # "speakerId", "diarizationId", "sourceHash", "createdAt", ... — emit the
    # camelCase aliases alongside the original snake_case keys so both the
    # Rust engine and any legacy consumers keep working.
    for s in result_dict.get("segments", []):
        s["speakerId"] = s.get("speaker_id", s.get("speaker"))
    for spk in result_dict.get("speakers", []):
        spk["diarizationId"] = spk.get("diarization_id")
        spk["totalSpeechSec"] = spk.get("total_speech_sec")
        spk["segmentCount"] = spk.get("segment_count")
        spk["avgConfidence"] = spk.get("avg_confidence")
    result_dict["sourceHash"] = result_dict.get("source_hash")
    result_dict["createdAt"] = result_dict.get("created_at")

    # Output
    output_json = json.dumps(result_dict, indent=2)
    print(output_json)
    
    # Cache
    if args.cache_dir or not args.force:
        try:
            with open(cache_file, "w") as f:
                f.write(output_json)
        except Exception as e:
            print(f"[Diarization] Cache write failed: {e}", file=sys.stderr)
    
    # Write to output file if specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)

if __name__ == "__main__":
    main()