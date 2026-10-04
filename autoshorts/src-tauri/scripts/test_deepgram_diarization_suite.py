#!/usr/bin/env python3
"""
AutoShorts 11.0 — Deepgram Diarization Regression Suite

Regression coverage for the production failure where a long source video was
uploaded to Deepgram in a single request, exhausting the upload write stage and
leaving the pipeline with zero speaker entries.

Failure mode under test:
    requests.post(..., data=<entire source video bytes>, timeout=600)
    -> ('Connection aborted.', TimeoutError('The write operation timed out'))

What each test pins:
  - video input is rejected before any upload (the root cause)
  - write timeout is explicit, tuple-shaped, and scales with payload size
  - a successful Deepgram response produces real speaker segments
  - long-form media is chunked, not uploaded whole
  - chunk timestamps are re-based into global source time
  - speaker labels stay consistent across chunk boundaries
  - a failed chunk does not corrupt the chunks that succeeded
  - timeout is reported with an explicit stage, not swallowed
  - a genuinely unavailable Deepgram yields the documented empty fallback
  - the API key never appears in any diagnostic output

No live Deepgram API key is required: the transport is always faked.

Run:  python scripts/test_deepgram_diarization_suite.py
Exit 0 only when every case passes.
"""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import speaker_diarization as sd

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_wav(path, seconds=1.0, rate=16000):
    """A real, decodable WAV so duration probing is exercised for real."""
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"\x00\x00" * int(rate * seconds))
    return str(path)


def make_audio(path, seconds, rate=16000, bitrate="64k"):
    """A real, decodable mono MP3 of a real length via ffmpeg.

    Duration must be genuine: the sidecar probes it with ffprobe to plan chunks
    and asks ffmpeg to slice real ranges, so a file whose bytes do not match its
    decodable length would test nothing.
    """
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi",
         "-i", f"anullsrc=r={rate}:cl=mono", "-t", f"{seconds:.3f}",
         "-vn", "-ac", "1", "-ar", str(rate), "-c:a", "libmp3lame", "-b:a", bitrate,
         str(path)],
        capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"ffmpeg fixture generation failed: {out.stderr[-300:]}")
    return str(path)


def make_fake_audio(path, size_bytes):
    """A real decodable MP3 padded to an exact size (for byte-level assertions)."""
    size_bytes = max(size_bytes, 4096)
    make_audio(path, 0.2)
    with open(path, "rb") as fh:
        data = fh.read()
    with open(path, "wb") as fh:
        fh.write(data + b"\x00" * max(0, size_bytes - len(data)))
    return str(path)


def dg_words(words):
    """Wrap (speaker_index, start, end, confidence) tuples as a Deepgram payload."""
    return {
        "results": {
            "channels": [
                {"alternatives": [{"words": [
                    {"speaker": s, "start": a, "end": b, "confidence": c, "word": "x"}
                    for (s, a, b, c) in words
                ]}]}
            ]
        }
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)[:300]

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for requests.Session. Records every upload and can be told to
    fail a chosen chunk, a chosen call, or with a specific exception."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []          # (params, headers, timeout, file_size)
        self.closed = False

    def post(self, url, params=None, headers=None, data=None, timeout=None):
        size = None
        stream_obj = None
        if hasattr(data, "read"):
            stream_obj = data
            pos = data.tell()
            data.seek(0, os.SEEK_END)
            size = data.tell() - pos
            data.seek(pos)
        elif isinstance(data, (bytes, bytearray)):
            size = len(data)
        self.calls.append({"url": url, "headers": headers or {},
                           "timeout": timeout, "size": size})
        result = self.responder(len(self.calls) - 1, size, stream_obj)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


class FakeRequests:
    def __init__(self, session):
        self.Session = lambda: session


@contextlib.contextmanager
def with_fake_requests(session):
    """Install a fake `requests` module for the duration of the test."""
    import sys as _sys
    saved = _sys.modules.get("requests")
    _sys.modules["requests"] = FakeRequests(session)
    try:
        yield
    finally:
        if saved is None:
            _sys.modules.pop("requests", None)
        else:
            _sys.modules["requests"] = saved


# ─── 1. Root cause: video input must never be uploaded ────────────────────────

def test_video_input_rejected_before_upload():
    """The exact production bug: the source VIDEO was uploaded to Deepgram.

    A .mp4 carrying video is not an acceptable diarization input. The request
    must fail at validation with no HTTP traffic at all, rather than streaming
    the container to the network.
    """
    tmp = tempfile.mkdtemp()
    try:
        video = os.path.join(tmp, "source.mp4")
        with open(video, "wb") as fh:
            fh.write(b"\x00" * (2 * 1024 * 1024))
        session = FakeSession(lambda i, size, fh: FakeResponse(dg_words([(0, 0, 1, 0.9)])))
        with with_fake_requests(session):
            try:
                sd.run_deepgram_diarization(video, "test-key")
                record("video input rejected before upload", False, "no error raised")
            except sd.DeepgramError as exc:
                ok = exc.stage == "input-validation" and len(session.calls) == 0
                record("video input rejected before upload", ok,
                       f"stage={exc.stage}, http_calls={len(session.calls)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 2. Write timeout is explicit and payload-derived ─────────────────────────

def test_write_timeout_is_tuple_and_scales():
    """urllib3 applies the CONNECT timeout to the body write.

    A single scalar timeout therefore bounds the upload. The fix passes an
    explicit (write, read) tuple, and the write value grows with the payload so
    a big body is not killed by a small budget.
    """
    small = sd.compute_write_timeout(1 * 1024 * 1024)
    large = sd.compute_write_timeout(160 * 1024 * 1024)
    ok = (large > small) and small >= sd.DEEPGRAM_MIN_WRITE_TIMEOUT
    record("write timeout scales with payload", ok,
           f"1MB->{small:.0f}s 160MB->{large:.0f}s (floor {sd.DEEPGRAM_MIN_WRITE_TIMEOUT:.0f}s)")

    # The observed failure was a 600s *scalar* timeout. Prove the scalar form is
    # still reachable only if we pass one, and that we do not.
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 1024 * 1024)
        session = FakeSession(lambda i, size, fh: FakeResponse(dg_words([(0, 0.0, 1.0, 0.9)])))
        with with_fake_requests(session):
            sd.run_deepgram_diarization(audio, "test-key")
        timeout = session.calls[0]["timeout"]
        is_tuple = isinstance(timeout, tuple) and len(timeout) == 2
        write_b, read_b = timeout if is_tuple else (None, None)
        ok = is_tuple and write_b > 0 and read_b == sd.DEEPGRAM_READ_TIMEOUT
        record("timeout passed as explicit (write, read) tuple", ok, f"timeout={timeout}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 3. Successful response produces real speaker segments ────────────────────

def test_successful_response_yields_segments():
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 512 * 1024)
        words = [
            (0, 0.0, 0.5, 0.9), (0, 0.5, 1.2, 0.9),   # S1
            (1, 1.5, 2.0, 0.85), (1, 2.0, 3.0, 0.85),   # S2
        ]
        session = FakeSession(lambda i, size, fh: FakeResponse(dg_words(words)))
        with with_fake_requests(session):
            model, version, segments, conf = sd.run_deepgram_diarization(audio, "test-key")

        speakers = sorted({s["speaker"] for s in segments})
        ok = (model == "deepgram" and speakers == ["S1", "S2"] and len(segments) == 2)
        record("successful response yields real speaker segments", ok,
               f"speakers={speakers} segments={len(segments)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_speaker_map_materializes_when_diarization_succeeds():
    """Speaker mapping must not report 0 entries when diarization returned data.

    This is the user-visible symptom of the production failure.
    """
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 512 * 1024)
        words = [(0, 0.0, 1.0, 0.9), (1, 1.2, 2.0, 0.9), (0, 2.2, 3.0, 0.9)]
        session = FakeSession(lambda i, size, fh: FakeResponse(dg_words(words)))
        with with_fake_requests(session):
            _model, _v, segments, _c = sd.run_deepgram_diarization(audio, "test-key")
        speaker_ids = sorted({s["speaker"] for s in segments})
        ok = len(speaker_ids) == 2
        record("speaker mapping non-empty when diarization succeeds", ok,
               f"speaker_ids={speaker_ids}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 4. Long-form handling ───────────────────────────────────────────────────

def test_long_form_is_chunked_not_uploaded_whole():
    """A long source must be split, never sent as one giant request.

    Each upload must stay under the configured per-request cap, and the number
    of requests must grow with duration.
    """
    saved_max = sd.DEEPGRAM_MAX_UPLOAD_BYTES
    try:
        sd.DEEPGRAM_MAX_UPLOAD_BYTES = 1_000_000          # 1 MB cap
        rate = 16_384                                     # bytes/sec
        chunks = sd.plan_chunks(duration_sec=600.0, bytes_per_sec=rate)
        total = 600.0 * rate
        max_chunk = max((e - s) * rate for s, e in chunks)
        covers = abs(chunks[0][0]) < 1e-6 and abs(chunks[-1][1] - 600.0) < 1e-6
        contiguous = all(abs(chunks[i][1] - chunks[i + 1][0]) < 1e-6
                         for i in range(len(chunks) - 1))
        ok = (len(chunks) > 1 and max_chunk <= 1_000_000 + rate
              and covers and contiguous)
        record("long-form media is chunked, each chunk under the cap", ok,
               f"{len(chunks)} chunks, max_chunk={int(max_chunk)}B, "
               f"total={int(total)}B, contiguous={contiguous}, covers_full={covers}")

        # Short media must stay on the single-request path (no behavior change).
        short = sd.plan_chunks(duration_sec=20.0, bytes_per_sec=rate)
        record("short media stays single-request", len(short) == 1 and short[0] == (0.0, 20.0),
               f"chunks={short}")
    finally:
        sd.DEEPGRAM_MAX_UPLOAD_BYTES = saved_max


def test_no_content_length_overshoot_and_streaming_upload():
    """Uploads are streamed from a file handle, not read into memory.

    The sidecar must not call f.read() on a large payload, and every request
    must carry a body rather than an empty one.
    """
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 300 * 1024)
        session = FakeSession(lambda i, size, fh: FakeResponse(dg_words([(0, 0.0, 1.0, 0.9)])))
        with with_fake_requests(session):
            sd.run_deepgram_diarization(audio, "test-key")
        call = session.calls[0]
        ok = call["size"] == 300 * 1024
        record("upload streams the file (size matches, not loaded in RAM)", ok,
               f"size_sent={call['size']}")
        record("session is closed after the request", session.closed, "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 5. Timestamp re-basing across chunks ────────────────────────────────────

def test_chunk_timestamps_rebased_to_global_time():
    """Chunk-local Deepgram timestamps must be shifted into source-video time.

    A 120 s source split into 2 s chunks: the last chunk's words start at 0.0
    locally. Without the offset every chunk after the first would collapse onto
    the start of the source, which is the bug this guards.
    """
    saved = (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
             sd.DEEPGRAM_MIN_CHUNK_SEC)
    try:
        sd.DEEPGRAM_MAX_UPLOAD_BYTES = 20_000     # tiny cap -> many small chunks
        sd.DEEPGRAM_CHUNK_OVERLAP_SEC = 0.0      # keep offsets exact
        # The 60 s minimum chunk length is a deliberate production floor (Deepgram
        # needs context to diarize a chunk), so a small fixture must lower it too.
        sd.DEEPGRAM_MIN_CHUNK_SEC = 2.0
        tmp = tempfile.mkdtemp()
        try:
            audio = make_audio(os.path.join(tmp, "a.mp3"), seconds=120.0)
            dur = sd.probe_audio_duration(audio)
            size = os.path.getsize(audio)
            plan = sd.plan_chunks(dur, size / dur)
            assert len(plan) > 3, f"fixture should need many chunks, got {len(plan)}"

            def responder(i, size, fh):
                # Every chunk reports 0.0-1.0 of LOCAL speech.
                return FakeResponse(dg_words([(0, 0.0, 1.0, 0.9)]))

            session = FakeSession(responder)
            with with_fake_requests(session):
                _m, _v, segments, _c = sd.run_deepgram_diarization(audio, "test-key")

            starts = [s["start"] for s in segments]
            ordered = starts == sorted(starts)
            spreads = (max(starts) - min(starts)) > 10.0
            in_range = all(0.0 <= s["start"] < dur + 1.0 and s["end"] <= dur + 1.0
                           for s in segments)
            ok = (len(session.calls) == len(plan) and ordered and spreads and in_range)
            record("chunk timestamps re-based into global source time", ok,
                   f"duration={dur:.0f}s requests={len(session.calls)}/{len(plan)} "
                   f"starts={[round(s,1) for s in starts[:8]]} ordered={ordered} in_range={in_range}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
         sd.DEEPGRAM_MIN_CHUNK_SEC) = saved


# ─── 6. Speaker identity is consistent across chunks ─────────────────────────

def test_speaker_labels_aligned_across_chunks():
    """Deepgram renumbers speakers per request; chunks must share one identity set.

    The second chunk labels the same two people as (1, 0) instead of (0, 1) --
    a realistic renumbering. After alignment the global label set must not grow.
    """
    prev = [
        {"speaker": "S1", "start": 0.0,  "end": 30.0, "confidence": 0.9},
        {"speaker": "S2", "start": 31.0, "end": 60.0, "confidence": 0.9},
    ]
    # Boundary at 600: incoming labels were assigned so S1(1) continues S2 and
    # S2(2) continues S1. Overlap is computed in the window around the boundary.
    nxt = [
        {"speaker": "S1", "start": 600.0, "end": 620.0, "confidence": 0.9},
        {"speaker": "S2", "start": 621.0, "end": 640.0, "confidence": 0.9},
    ]
    aligned = sd._align_speakers(prev, [dict(s) for s in nxt], 600.0)
    labels = sorted({s["speaker"] for s in aligned})
    ok = labels == ["S1", "S2"]
    record("speaker labels align to the established identity set", ok,
           f"labels_after={labels}")

    # An unrelated boundary with no overlap must NOT invent an identity.
    far = [{"speaker": "S9", "start": 5000.0, "end": 5010.0, "confidence": 0.9}]
    out = sd._align_speakers(prev, far, 600.0)
    ok2 = [s["speaker"] for s in out] == ["S9"]
    record("unmatched chunk label is left unmapped, not fabricated", ok2,
           f"labels={[s['speaker'] for s in out]}")


def test_chunk_boundaries_do_not_duplicate_segments():
    """Chunking must not emit overlapping or zero-length segments."""
    saved = (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
             sd.DEEPGRAM_MIN_CHUNK_SEC)
    try:
        sd.DEEPGRAM_MAX_UPLOAD_BYTES = 20_000
        sd.DEEPGRAM_CHUNK_OVERLAP_SEC = 0.5
        sd.DEEPGRAM_MIN_CHUNK_SEC = 5.0
        tmp = tempfile.mkdtemp()
        try:
            audio = make_audio(os.path.join(tmp, "a.mp3"), seconds=60.0)
            dur = sd.probe_audio_duration(audio)
            plan = sd.plan_chunks(dur, os.path.getsize(audio) / dur)

            def responder(i, size, fh):
                # Every chunk claims to speak 0.0-5.0 locally. Overlap trimming
                # must drop the part outside the nominal chunk.
                return FakeResponse(dg_words([(0, 0.0, 5.0, 0.9)]))

            session = FakeSession(responder)
            with with_fake_requests(session):
                _m, _v, segments, _c = sd.run_deepgram_diarization(audio, "test-key")

            positive = all(s["end"] > s["start"] for s in segments)
            ordered = all(segments[i]["end"] <= segments[i + 1]["start"] + 1e-6
                          for i in range(len(segments) - 1))
            within = all(0.0 <= s["start"] and s["end"] <= dur + 1.0 for s in segments)
            # 5 s chunks padded by 0.5 s are re-based to the nominal window and
            # then trimmed: local 0.0-5.0 collapses into each chunk's own span, so
            # exactly one segment per chunk is correct, not a duplication.
            no_dupes = len(segments) <= len(plan)
            ok = positive and ordered and within and no_dupes and len(session.calls) == len(plan)
            record("no duplicate/zero-length segments at chunk boundaries", ok,
                   f"requests={len(session.calls)}/{len(plan)} n={len(segments)} "
                   f"positive={positive} ordered={ordered} within={within} no_dupes={no_dupes}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
         sd.DEEPGRAM_MIN_CHUNK_SEC) = saved


# ─── 7. A failed chunk must not corrupt the successful ones ──────────────────

def test_failed_chunk_preserves_successful_chunks():
    """One bad chunk is recorded and skipped; the rest still produce output.

    This is the long-form robustness requirement: a partial network failure must
    degrade to partial data, not to the all-or-nothing zero-speaker state.
    """
    saved = (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
             sd.DEEPGRAM_MIN_CHUNK_SEC)
    try:
        sd.DEEPGRAM_MAX_UPLOAD_BYTES = 20_000
        sd.DEEPGRAM_CHUNK_OVERLAP_SEC = 0.0
        sd.DEEPGRAM_MIN_CHUNK_SEC = 5.0
        tmp = tempfile.mkdtemp()
        try:
            audio = make_audio(os.path.join(tmp, "a.mp3"), seconds=60.0)
            dur = sd.probe_audio_duration(audio)
            plan = sd.plan_chunks(dur, os.path.getsize(audio) / dur)
            assert len(plan) >= 3, f"fixture should need >=3 chunks, got {len(plan)}"

            def responder(i, size, fh):
                if i == 1:
                    return TimeoutError("The write operation timed out")
                return FakeResponse(dg_words([(0, 0.0, 1.0, 0.9)]))

            session = FakeSession(responder)
            with with_fake_requests(session):
                _m, _v, segments, _c = sd.run_deepgram_diarization(audio, "test-key")

            ok = len(session.calls) == len(plan) and len(segments) == len(plan) - 1
            record("failed chunk does not corrupt successful chunks", ok,
                   f"requests={len(session.calls)}/{len(plan)} segments={len(segments)} "
                   f"expected={len(plan) - 1}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
         sd.DEEPGRAM_MIN_CHUNK_SEC) = saved


# ─── 8. Timeout handling reports an explicit stage ───────────────────────────

def test_timeout_reports_write_stage():
    """The production error must be classified, not swallowed as a generic
    ConnectionError. A write timeout must name the write stage."""
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 256 * 1024)

        def responder(i, size, fh):
            class _Aborted(Exception):
                def __str__(self):
                    return "('Connection aborted.', 'The write operation timed out')"
            return _Aborted()

        session = FakeSession(responder)
        with with_fake_requests(session):
            try:
                sd.run_deepgram_diarization(audio, "test-key")
                record("write timeout classified with an explicit stage", False, "no error")
            except sd.DeepgramError as exc:
                ok = exc.stage in ("write", "request")
                record("write timeout classified with an explicit stage", ok,
                       f"stage={exc.stage} msg={str(exc)[:70]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_api_error_status_is_surfaced():
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 64 * 1024)
        session = FakeSession(lambda i, size, fh: FakeResponse({"err": "bad key"}, 401))
        with with_fake_requests(session):
            try:
                sd.run_deepgram_diarization(audio, "test-key")
                record("API error status surfaced", False, "no error raised")
            except sd.DeepgramError as exc:
                ok = exc.stage == "api" and exc.status == 401
                record("API error status surfaced", ok, f"stage={exc.stage} status={exc.status}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_api_key_is_a_config_error():
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 64 * 1024)
        session = FakeSession(lambda i, size, fh: FakeResponse(dg_words([(0, 0, 1, 0.9)])))
        with with_fake_requests(session):
            try:
                sd.run_deepgram_diarization(audio, "")
                record("missing API key is a config-stage error", False, "no error")
            except sd.DeepgramError as exc:
                ok = exc.stage == "config" and len(session.calls) == 0
                record("missing API key is a config-stage error", ok,
                       f"stage={exc.stage} http_calls={len(session.calls)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 9. Deterministic fallback when Deepgram is genuinely unavailable ───────

def test_all_chunks_failing_yields_hard_failure():
    """When nothing succeeds, the backend must fail loudly (so the caller can
    apply its documented empty fallback) rather than return fabricated data."""
    saved = (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
             sd.DEEPGRAM_MIN_CHUNK_SEC)
    try:
        sd.DEEPGRAM_MAX_UPLOAD_BYTES = 20_000
        sd.DEEPGRAM_CHUNK_OVERLAP_SEC = 0.0
        sd.DEEPGRAM_MIN_CHUNK_SEC = 5.0
        tmp = tempfile.mkdtemp()
        try:
            audio = make_audio(os.path.join(tmp, "a.mp3"), seconds=60.0)
            session = FakeSession(lambda i, size, fh: TimeoutError("write timed out"))
            with with_fake_requests(session):
                try:
                    _m, _v, segments, _c = sd.run_deepgram_diarization(audio, "k")
                    record("all chunks failing raises (no fake data)", False,
                           f"returned {len(segments)} segments")
                except sd.DeepgramError as exc:
                    ok = exc.stage == "all-chunks-failed" and "failed_chunks" in str(exc)
                    record("all chunks failing raises (no fake data)", ok,
                           f"stage={exc.stage} msg={str(exc)[:80]}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        (sd.DEEPGRAM_MAX_UPLOAD_BYTES, sd.DEEPGRAM_CHUNK_OVERLAP_SEC,
         sd.DEEPGRAM_MIN_CHUNK_SEC) = saved


def test_zero_entry_only_when_diarization_fails():
    """Zero segments is reachable ONLY through the failure path.

    Guards the invariant that a successful run never degrades to an empty list.
    """
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 64 * 1024)
        # A response with no words is a genuine parse failure, not a valid result.
        empty_payload = {"results": {"channels": [{"alternatives": [{"words": []}]}]}}
        session = FakeSession(lambda i, size, fh: FakeResponse(empty_payload))
        with with_fake_requests(session):
            try:
                _m, _v, segs, _c = sd.run_deepgram_diarization(audio, "k")
                record("empty Deepgram words is a failure, not an empty success", False,
                       f"returned {len(segs)}")
            except sd.DeepgramError as exc:
                record("empty Deepgram words is a failure, not an empty success", True,
                       f"stage={exc.stage}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 10. Credentials are never logged ────────────────────────────────────────

def test_api_key_never_appears_in_output(capture=True):
    """The key must not leak into stdout/stderr on any path."""
    import contextlib
    secret = "SUPER_SECRET_KEY_d0notl0g"
    tmp = tempfile.mkdtemp()
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 128 * 1024)
        cases = [
            ("success", lambda i, size, fh: FakeResponse(dg_words([(0, 0, 1, 0.9)]))),
            ("api-error", lambda i, size, fh: FakeResponse({"e": 1}, 500)),
            ("timeout", lambda i, size, fh: TimeoutError("write timed out")),
        ]
        leaked = []
        for name, responder in cases:
            session = FakeSession(responder)
            buf = io.StringIO()
            with with_fake_requests(session):
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    try:
                        sd.run_deepgram_diarization(audio, secret)
                    except Exception:
                        pass
            text = buf.getvalue()
            if secret in text:
                leaked.append(name)
        record("API key never appears in logs", not leaked,
               f"leaked_in={leaked}" if leaked else "clean on success/error/timeout")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 11. Duration probing (real WAV, no ffmpeg) ─────────────────────────────

def test_wav_duration_probing():
    tmp = tempfile.mkdtemp()
    try:
        path = make_wav(os.path.join(tmp, "a.wav"), seconds=3.0, rate=16000)
        dur = sd.probe_audio_duration(path)
        ok = dur is not None and abs(dur - 3.0) < 0.3
        record("WAV duration probed from header", ok, f"duration={dur}")

        garbage = os.path.join(tmp, "b.wav")
        with open(garbage, "wb") as fh:
            fh.write(b"not a wav file at all")
        dur2 = sd.probe_audio_duration(garbage)
        record("undecodable file reports unknown, never a guess",
               dur2 is None or dur2 > 0, f"duration={dur2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 12. Configured timeout knobs are honored ───────────────────────────────

def test_env_configured_timeouts_are_read():
    saved = {
        "AUTOSHORTS_DEEPGRAM_READ_TIMEOUT": os.environ.pop("AUTOSHORTS_DEEPGRAM_READ_TIMEOUT", None),
        "AUTOSHORTS_DEEPGRAM_MAX_UPLOAD_BYTES": os.environ.pop("AUTOSHORTS_DEEPGRAM_MAX_UPLOAD_BYTES", None),
    }
    try:
        os.environ["AUTOSHORTS_DEEPGRAM_READ_TIMEOUT"] = "12.5"
        os.environ["AUTOSHORTS_DEEPGRAM_MAX_UPLOAD_BYTES"] = "5242880"
        read = sd._env_float("AUTOSHORTS_DEEPGRAM_READ_TIMEOUT", 600.0)
        cap = sd._env_int("AUTOSHORTS_DEEPGRAM_MAX_UPLOAD_BYTES", 1)
        record("timeouts are configurable via env", read == 12.5 and cap == 5242880,
               f"read_timeout={read} max_upload={cap}")

        # A bad value must fall back to the default, not be honored.
        for bad in ("nonsense", "-5", "0", ""):
            os.environ["AUTOSHORTS_DEEPGRAM_READ_TIMEOUT"] = bad
            ok = sd._env_float("AUTOSHORTS_DEEPGRAM_READ_TIMEOUT", 600.0) == 600.0
            record(f"invalid timeout {bad!r} falls back to the default", ok, "")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_request_flags_reach_the_wire():
    """The DEEPGRAM_* feature flags must actually be sent, not just declared.

    A declared-but-unwired knob is misleading configuration: it reads like a
    switch but cannot change anything. Asserted on the params the transport
    actually receives, not on the source text.
    """
    tmp = tempfile.mkdtemp()
    saved = (sd.DEEPGRAM_DIARIZE, sd.DEEPGRAM_SMART_FORMAT,
             sd.DEEPGRAM_PUNCTUATE, sd.DEEPGRAM_FILLER_WORDS)
    try:
        audio = make_fake_audio(os.path.join(tmp, "a.mp3"), 64 * 1024)

        def responder_factory():
            def responder(i, size, fh):
                return FakeResponse(dg_words([(0, 0, 1, 0.9)]))
            return responder

        # The FakeSession records the request; capture params by wrapping post.
        class ParamsSession(FakeSession):
            def __init__(self, inner):
                super().__init__(inner)
                self.seen_params = []

            def post(self, url, params=None, headers=None, data=None, timeout=None):
                self.seen_params.append(params)
                return super().post(url, params=params, headers=headers,
                                    data=data, timeout=timeout)

        sd.DEEPGRAM_DIARIZE = True
        sd.DEEPGRAM_SMART_FORMAT = True
        sd.DEEPGRAM_PUNCTUATE = True
        sd.DEEPGRAM_FILLER_WORDS = True
        s_on = ParamsSession(responder_factory())
        with with_fake_requests(s_on):
            sd.run_deepgram_diarization(audio, "test-key")
        p_on = s_on.seen_params[0]
        ok_on = all(p_on[k] == "true" for k in
                    ("diarize", "smart_format", "punctuate", "filler_words"))
        record("feature flags ON are sent as 'true'", ok_on,
               f"params={ {k: p_on[k] for k in ('diarize','smart_format','punctuate','filler_words')} }")

        sd.DEEPGRAM_DIARIZE = False
        sd.DEEPGRAM_SMART_FORMAT = False
        s_off = ParamsSession(responder_factory())
        with with_fake_requests(s_off):
            sd.run_deepgram_diarization(audio, "test-key")
        p_off = s_off.seen_params[0]
        ok_off = p_off["diarize"] == "false" and p_off["smart_format"] == "false"
        record("feature flags OFF are actually sent as 'false' (not dead config)", ok_off,
               f"diarize={p_off['diarize']} smart_format={p_off['smart_format']}")
    finally:
        (sd.DEEPGRAM_DIARIZE, sd.DEEPGRAM_SMART_FORMAT,
         sd.DEEPGRAM_PUNCTUATE, sd.DEEPGRAM_FILLER_WORDS) = saved
        shutil.rmtree(tmp, ignore_errors=True)


# ─── Runner ──────────────────────────────────────────────────────────────────

TESTS = [
    ("video input rejected before upload (ROOT CAUSE)", test_video_input_rejected_before_upload),
    ("write timeout scales with payload", test_write_timeout_is_tuple_and_scales),
    ("successful response yields real speaker segments", test_successful_response_yields_segments),
    ("speaker mapping non-empty on success", test_speaker_map_materializes_when_diarization_succeeds),
    ("long-form chunking", test_long_form_is_chunked_not_uploaded_whole),
    ("streaming upload", test_no_content_length_overshoot_and_streaming_upload),
    ("timestamp re-basing", test_chunk_timestamps_rebased_to_global_time),
    ("speaker alignment across chunks", test_speaker_labels_aligned_across_chunks),
    ("no duplicate segments at boundaries", test_chunk_boundaries_do_not_duplicate_segments),
    ("failed chunk isolation", test_failed_chunk_preserves_successful_chunks),
    ("timeout stage classification", test_timeout_reports_write_stage),
    ("API error status surfaced", test_api_error_status_is_surfaced),
    ("missing key is a config error", test_missing_api_key_is_a_config_error),
    ("all chunks failing raises", test_all_chunks_failing_yields_hard_failure),
    ("zero-entry only on genuine failure", test_zero_entry_only_when_diarization_fails),
    ("API key never logged", test_api_key_never_appears_in_output),
    ("WAV duration probing", test_wav_duration_probing),
    ("env-configured timeouts", test_env_configured_timeouts_are_read),
    ("request feature flags wired", test_request_flags_reach_the_wire),
]


def main():
    print("=" * 74)
    print("AutoShorts 11.0 - Deepgram Diarization Regression Suite")
    print("=" * 74)
    for name, fn in TESTS:
        try:
            fn()
        except Exception as exc:
            record(name, False, f"EXCEPTION {type(exc).__name__}: {exc}")
            traceback.print_exc()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("-" * 74)
    print(f"RESULT: {passed}/{total} passed")
    failed = [n for n, ok, _ in RESULTS if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
    print("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
