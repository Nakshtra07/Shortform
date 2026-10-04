#!/usr/bin/env python3
"""
AutoShorts 11.0 Phase 4 — ADVISORY VLM Candidate Scoring via NVIDIA Nemotron

Remote Vision-Language Model integration for richer candidate quality signals.
The VLM is ADVISORY ONLY — it provides additional metadata/scores. It NEVER
produces authoritative candidate boundaries, payoff endpoints, or framing
decisions, and the Rust caller never feeds its output back into ranking,
boundary snapping, or rendering.

CLI contract:
    python vlm_scoring.py SOURCE START_MS END_MS CANDIDATE_JSON WORDS_JSON
        [--candidate-key KEY] [--max-frames N]

    - SOURCE        absolute path to the source video
    - START_MS      candidate start in milliseconds (absolute source time)
    - END_MS        candidate end in milliseconds (absolute source time)
    - CANDIDATE_JSON path to CandidateDraft JSON
    - WORDS_JSON    path to the CANDIDATE'S transcript words JSON
                    [{"text","start","end","speaker"}, ...] (range-filtered by Rust)
    --candidate-key stable key the Rust caller uses to associate this result
    --max-frames    frame budget (from Rust config)

Output: the LAST stdout line is the JSON VLM score (camelCase keys, matching the
Rust `VlmCandidateScore` serde struct). Telemetry goes to stderr. Exit code is
non-zero only when the engine itself could not produce any score.

Input reality (documented honestly): this is KEYFRAME-BASED VISUAL ASSESSMENT.
The model never sees the candidate video, only N uniformly-sampled PNG
keyframes plus transcript context. It is NOT full temporal video understanding.
"""

import sys
import os
import json
import argparse
import subprocess
import time
import base64
from typing import List, Dict, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Never let a frame-extraction ffmpeg call outlive its slice of the budget.
FRAME_EXTRACT_TIMEOUT_SEC = 20
# Bound the decoded keyframe payload: full-resolution PNGs of long sources are
# the single largest cost in this request. 512px on the long edge preserves
# composition/framing judgement (what the VLM is actually scoring) while
# keeping the request small and deterministic.
KEYFRAME_MAX_EDGE = 512


def log(msg):
    sys.stderr.write(f"[VLM Scoring] {msg}\n")
    sys.stderr.flush()


def _force_utf8_streams():
    """Make stdout/stderr UTF-8 regardless of the Windows code page.

    Windows opens piped/console streams with the ANSI code page (cp1252 here)
    and stdout's default error handler is 'strict'. Real model output routinely
    contains non-ASCII (e.g. U+2011 non-breaking hyphen), which would crash the
    sidecar AFTER a successful inference and discard a genuine result. stderr's
    'backslashreplace' default masks this in logs, which is why the bug only
    surfaced on the machine-readable stdout contract.

    Must run before any output is emitted.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Non-reconfigurable stream (pytest capture, exotic wrapper): the
            # emitted JSON stays parseable because `emit` also escapes below.
            pass


_force_utf8_streams()


def emit(score):
    """Print the score JSON as the LAST stdout line.

    Primary form is human-readable (ensure_ascii=False). If the underlying
    stream still cannot encode a character (non-reconfigurable capture with a
    legacy code page), retry with \\uXXXX escapes rather than crashing — a
    successful inference must never be discarded at the print boundary.
    """
    text = json.dumps(score, ensure_ascii=False)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(score, ensure_ascii=True))


def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log(f"cannot read {path}: {exc}")
        return None


def frame_count_for_duration(dur_sec: float, max_frames: int) -> int:
    """Duration-aware keyframe budget.

    A 60-90s candidate sampled at a flat 8 frames yields roughly one keyframe
    every 8-11s, which is too coarse to characterise a candidate. We scale the
    budget with duration but keep it strictly bounded and deterministic so the
    request never becomes unbounded:

        <= 15s   -> min(max_frames, 6)      dense short clips still capped
        <= 60s   -> max_frames                (8) the baseline
        <= 180s  -> max(max_frames, 12)       ~1 frame / 15s
        else     -> min(max(max_frames, 16), 20)   hard ceiling of 20
    """
    if dur_sec <= 0:
        return 0
    if dur_sec <= 15.0:
        return max(1, min(max_frames, 6))
    if dur_sec <= 60.0:
        return max(1, max_frames)
    if dur_sec <= 180.0:
        return max(1, min(max(max_frames, 12), 20))
    return max(1, min(max(max_frames, 16), 20))


def sample_timestamps(start_sec: float, end_sec: float, n: int) -> List[float]:
    """Uniform mid-interval sample points, deterministic for a given (start,end,n)."""
    if n <= 0 or end_sec <= start_sec:
        return []
    dur = end_sec - start_sec
    step = dur / float(n)
    return [round(start_sec + (i + 0.5) * step, 3) for i in range(n)]


def extract_frames_ffmpeg(source_path, start_sec, end_sec, max_frames=8):
    """Extract bounded, downscaled keyframes in a SINGLE ffmpeg invocation.

    The previous implementation spawned one ffmpeg process per frame
    (8 sequential process launches per candidate). A single select-filter pass
    reads the range once and emits every sample, which is both faster and
    deterministic in ordering.

    Returns a list of base64-encoded PNG strings.
    """
    dur = end_sec - start_sec
    if dur <= 0 or max_frames <= 0:
        return []

    n = frame_count_for_duration(dur, max_frames)
    timestamps = sample_timestamps(start_sec, end_sec, n)
    if not timestamps:
        return []

    # One pass: sample the trimmed range with an fps filter and downscale.
    # Sampling by frame index across VFR sources is unreliable; an fps filter
    # over the already-trimmed range is stable and deterministic.
    fps_value = len(timestamps) / dur if dur > 0 else 0.0
    if fps_value <= 0:
        return []

    vf = (
        f"fps={fps_value:.6f},"
        f"scale='if(gt(iw,ih),{KEYFRAME_MAX_EDGE},-2)':'if(gt(iw,ih),-2,{KEYFRAME_MAX_EDGE})'"
    )

    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", source_path,
        "-vf", vf,
        "-frames:v", str(len(timestamps)),
        "-f", "image2pipe",
        "-vcodec", "png",
        "-"
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=FRAME_EXTRACT_TIMEOUT_SEC)
        if proc.returncode != 0 or not proc.stdout:
            log(f"single-pass frame extraction failed rc={proc.returncode}: "
                f"{proc.stderr.decode('utf-8', 'replace')[:300]}")
            return []
    except subprocess.TimeoutExpired:
        log(f"frame extraction timed out after {FRAME_EXTRACT_TIMEOUT_SEC}s")
        return []
    except Exception as exc:  # noqa: BLE001 - sidecar must never crash the pipeline
        log(f"frame extraction error: {exc}")
        return []

    # image2pipe with multiple PNG frames concatenates them; split on the PNG
    # magic bytes rather than assuming a framing we do not control.
    raw = proc.stdout
    marker = b"\x89PNG\r\n\x1a\n"
    frames: List[str] = []
    idx = 0
    while True:
        start = raw.find(marker, idx)
        if start < 0:
            break
        # The next PNG start marks this frame's end.
        nxt = raw.find(marker, start + len(marker))
        blob = raw[start:nxt] if nxt > 0 else raw[start:]
        if blob:
            frames.append(base64.b64encode(blob).decode("ascii"))
        if nxt < 0:
            break
        idx = nxt

    if not frames:
        log("no keyframes decoded from image2pipe output")
    log(f"extracted {len(frames)} keyframes (budget {len(timestamps)}) "
        f"for a {dur:.1f}s candidate via a single ffmpeg pass")
    return frames


def build_transcript_context(words: List[Dict]) -> Dict[str, Any]:
    """Build the transcript excerpt + speaker summary actually sent to the VLM.

    The previous implementation passed an EMPTY list to the API (discarding the
    words JSON that Rust had written), and its fallback filter was a no-op that
    would have taken the first 50 words of the entire source transcript.
    """
    ordered = sorted(words, key=lambda w: (w.get("start", 0.0), w.get("end", 0.0)))
    text = " ".join(w.get("text", "").strip() for w in ordered if w.get("text", "").strip())
    excerpt = text[:800]

    speakers: Dict[str, int] = {}
    for w in ordered:
        spk = w.get("speaker")
        if spk:
            speakers[str(spk)] = speakers.get(str(spk), 0) + 1

    if speakers:
        speaker_desc = ", ".join(f"{k} ({v} words)" for k, v in sorted(speakers.items()))
    else:
        speaker_desc = "not available"

    return {
        "excerpt": excerpt,
        "word_count": len(ordered),
        "speaker_desc": speaker_desc,
        "speaker_count": len(speakers),
    }


# ─── NVIDIA Nemotron 3 Nano Omni (advisory VLM) ──────────────────────────
#
# The VLM is an INFORMATION SOURCE, never a decision-maker. Every score it
# returns is advisory metadata attached to a candidate; nothing here can
# influence boundaries, payoff, pacing, framing, captions, or Render QA.

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

# Bounded HTTP policy. The VLM must never become an unbounded blocking stage,
# and it must never spend minutes retrying a PERMANENT error (bad key, wrong
# model, malformed request) -- those fail fast with a concrete reason.
NVIDIA_CONNECT_TIMEOUT_SEC = 15
NVIDIA_READ_TIMEOUT_SEC = 120
NVIDIA_MAX_ATTEMPTS = 3            # total attempts, not extra retries
NVIDIA_RETRY_BASE_SLEEP_SEC = 2.0
NVIDIA_MAX_RETRY_AFTER_SEC = 30.0

# HTTP statuses worth another attempt. 400/401/403 are permanent: retrying a
# rejected credential or a malformed payload just wastes the pipeline's time.
NVIDIA_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class VlmRequestError(RuntimeError):
    """Raised with a concrete, human-meaningful reason for a failed call."""

    def __init__(self, reason: str, status: Optional[int] = None, retryable: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.retryable = retryable


def requests_module():
    """
    Lazily expose `requests` for both production calls and tests.

    The HTTP client is imported inside the call sites so the module can be
    imported (and unit-tested) without the dependency being mandatory. Tests
    monkeypatch this accessor instead of the global `requests` name, which
    would otherwise leak between cases.
    """
    import requests
    return requests


def _extract_content(response_json: Dict) -> str:
    """Pull assistant text out of a chat-completions response.

    NVIDIA reasoning models may return content under `content` and/or a
    `reasoning_content` field. Anything unexpected becomes a structured error
    rather than an exception that escapes into the pipeline.
    """
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VlmRequestError("response contained no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise VlmRequestError("response choice had no message object")
    content = message.get("content")
    if isinstance(content, list):
        # Some gateways return content as a list of typed parts.
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise VlmRequestError("response message had no textual content")
    return content


def call_nvidia_api(
    frames: List[str],
    candidate: Dict,
    transcript: Dict[str, Any],
    model: str,
    api_key: str,
) -> Dict:
    """
    Call NVIDIA Nemotron 3 Nano Omni with REAL candidate keyframes.

    Returns the parsed JSON score object. Raises VlmRequestError with a concrete
    reason for every failure mode so the caller can report honestly instead of
    degrading to a heuristic score without explanation.
    """
    requests = requests_module()

    hook = candidate.get("hook", "")
    payoff_text = candidate.get("payoffText") or candidate.get("payoff_text", "")
    rationale = candidate.get("rationale", "")

    system_prompt = (
        "You are an expert visual analyst for short-form content. You are given "
        "sampled KEYFRAMES from one candidate clip plus a transcript excerpt. "
        "Judge only what the visual evidence supports, and say so when evidence "
        "is thin. Respond with a single valid JSON object and nothing else: no "
        "markdown fence, no prose before or after."
    )

    user_prompt = (
        f"Analyze the VISUAL content of this candidate clip for short-form use.\n\n"
        f"CANDIDATE METADATA:\n"
        f"- Hook: \"{hook}\"\n"
        f"- Payoff: \"{payoff_text}\"\n"
        f"- Rationale: \"{rationale}\"\n"
        f"- Candidate start (s): {candidate.get('start')}\n"
        f"- Candidate end (s): {candidate.get('end')}\n"
        f"- Speakers: {transcript['speaker_desc']}\n"
        f"- Transcript words in range: {transcript['word_count']}\n"
        f"- Transcript excerpt: \"{transcript['excerpt']}\"\n\n"
        f"KEYFRAMES PROVIDED: {len(frames)} (uniformly sampled across the "
        f"candidate range; keyframe-based visual assessment, NOT full temporal "
        f"video understanding)\n\n"
        f"SCORING CRITERIA (each 0.0-1.0):\n"
        f"1. hook_strength: how compelling is the opening visually?\n"
        f"2. standalone_completeness: does it read without prior context?\n"
        f"3. payoff_strength: is the ending visually supported?\n"
        f"4. visual_engagement: are the visuals dynamic and well-framed?\n"
        f"5. semantic_coherence: does the narrative flow hook -> payoff?\n"
        f"6. highlight_relevance: is this highlight-worthy?\n"
        f"7. production_quality: visual quality, framing, clarity.\n\n"
        f"Also answer briefly in prose: what is visually happening, is the main "
        f"subject clearly visible, and does the visual material support a "
        f"short-form clip? Put that prose in the \"reason\" field.\n\n"
        f"OUTPUT SCHEMA (JSON only):\n"
        f"{{\n"
        f'  "hook_strength": 0.0,\n'
        f'  "standalone_completeness": 0.0,\n'
        f'  "payoff_strength": 0.0,\n'
        f'  "visual_engagement": 0.0,\n'
        f'  "semantic_coherence": 0.0,\n'
        f'  "highlight_relevance": 0.0,\n'
        f'  "production_quality": 0.0,\n'
        f'  "overall_quality": 0.0,\n'
        f'  "observed_evidence": ["specific observation"],\n'
        f'  "reason": "what is happening visually and whether it supports the clip"\n'
        f"}}"
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for frame_b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{frame_b64}"},
        })

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        # Bounded: an advisory stage must not be able to run for minutes.
        "max_tokens": 2048,
        "stream": False,
        "temperature": 0.6,
        "top_p": 0.95,
    }

    # Authorization header is constructed here and never logged.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    last_error: Optional[str] = None
    for attempt in range(1, NVIDIA_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                NVIDIA_INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=(NVIDIA_CONNECT_TIMEOUT_SEC, NVIDIA_READ_TIMEOUT_SEC),
            )
        except requests.exceptions.Timeout:
            last_error = f"timeout after {NVIDIA_READ_TIMEOUT_SEC}s reading NVIDIA response"
            log(f"[VLM] attempt {attempt}/{NVIDIA_MAX_ATTEMPTS}: {last_error}")
        except requests.exceptions.ConnectionError as exc:
            last_error = f"connection failure: {type(exc).__name__}"
            log(f"[VLM] attempt {attempt}/{NVIDIA_MAX_ATTEMPTS}: {last_error}")
        except requests.exceptions.RequestException as exc:
            # Includes TLS/certificate and malformed-URL problems: not retryable.
            raise VlmRequestError(
                f"request failed: {type(exc).__name__}", retryable=False
            ) from exc
        else:
            status = response.status_code
            if status == 200:
                try:
                    raw = response.json()
                except ValueError as exc:
                    last_error = f"malformed JSON body: {type(exc).__name__}"
                    log(f"[VLM] attempt {attempt}/{NVIDIA_MAX_ATTEMPTS}: {last_error}")
                else:
                    text = _extract_content(raw)
                    log(f"[VLM] RAW RESPONSE ({len(text)} chars):\n{text}")
                    return {"_raw_text": text, "_parsed": _parse_model_json(text)}
            if status in NVIDIA_RETRYABLE_STATUS:
                retry_after = response.headers.get("Retry-After")
                last_error = f"HTTP {status} from NVIDIA endpoint"
                log(f"[VLM] attempt {attempt}/{NVIDIA_MAX_ATTEMPTS}: {last_error}")
                if attempt < NVIDIA_MAX_ATTEMPTS:
                    sleep_for = NVIDIA_RETRY_BASE_SLEEP_SEC * (2 ** (attempt - 1))
                    if retry_after:
                        try:
                            sleep_for = max(
                                sleep_for,
                                min(float(retry_after), NVIDIA_MAX_RETRY_AFTER_SEC),
                            )
                        except ValueError:
                            pass
                    time.sleep(sleep_for)
                    continue
            else:
                # Permanent: 400/401/403 and anything else unexpected.
                raise VlmRequestError(
                    f"HTTP {status} from NVIDIA endpoint: "
                    f"{response.text[:200]}",
                    status=status,
                    retryable=False,
                )
        if attempt < NVIDIA_MAX_ATTEMPTS:
            time.sleep(NVIDIA_RETRY_BASE_SLEEP_SEC * (2 ** (attempt - 1)))

    raise VlmRequestError(
        f"exhausted {NVIDIA_MAX_ATTEMPTS} attempts; last error: {last_error}",
        retryable=True,
    )


def _parse_model_json(text: str) -> Dict:
    """
    Parse the model's JSON, tolerating markdown fences and surrounding prose.

    Reasoning models sometimes wrap JSON in ```json fences or prefix it with a
    sentence. Salvage the first balanced object rather than failing the
    candidate outright; a genuinely unparseable response raises.
    """
    candidate_text = text.strip()
    if candidate_text.startswith("```"):
        # Strip an optional ```json fence.
        lines = candidate_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate_text = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate_text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass

    # Salvage the first balanced {...} block.
    start = candidate_text.find("{")
    if start == -1:
        raise VlmRequestError("model response contained no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(candidate_text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(candidate_text[start : i + 1])
                except ValueError as exc:
                    raise VlmRequestError(
                        f"model JSON was malformed: {type(exc).__name__}"
                    ) from exc
                if isinstance(parsed, dict):
                    return parsed
                raise VlmRequestError("model JSON was not an object")
    raise VlmRequestError("model JSON object was truncated (unbalanced braces)")


SCORE_KEYS = (
    "hook_strength",
    "standalone_completeness",
    "payoff_strength",
    "visual_engagement",
    "semantic_coherence",
    "highlight_relevance",
    "production_quality",
    "overall_quality",
)


def normalize_scores(raw: Dict) -> Dict[str, float]:
    """Clamp every requested score into [0,1]; ignore unknown/garbage keys."""
    out: Dict[str, float] = {}
    for key in SCORE_KEYS:
        val = raw.get(key)
        try:
            if val is None:
                continue
            out[key] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            continue
    return out


def heuristic_score(
    candidate: Dict,
    start_sec: float,
    end_sec: float,
    model: str,
    model_version: str,
    prompt_version: str,
    candidate_key: str,
    reason: str,
) -> Dict:
    """Deterministic fallback used when real inference is impossible.

    IMPORTANT HONESTY CONTRACT: this is NOT model output. It is flagged with
    `heuristic_fallback: true` so no downstream consumer can mistake it for a
    genuine VLM judgement.
    """
    hook = candidate.get("hook", "")
    payoff_text = candidate.get("payoffText") or candidate.get("payoff_text", "")
    semantic_score = candidate.get("score", 0.75)

    has_payoff = len(payoff_text.strip()) > 5
    has_strong_hook = len(hook.strip()) > 10

    visual_engagement = min(1.0, semantic_score * 0.9 + 0.1)
    semantic_coherence = min(1.0, semantic_score * (1.1 if has_payoff and has_strong_hook else 0.95))
    production_quality = min(
        1.0,
        0.7
        + (0.1 if candidate.get("hookStart") else 0)
        + (0.1 if candidate.get("hookEnd") else 0)
        + (0.1 if candidate.get("hookConfidence") else 0),
    )
    highlight_relevance = semantic_score
    quality_score = min(1.0, max(0.0,
        visual_engagement * 0.2
        + semantic_coherence * 0.3
        + production_quality * 0.2
        + highlight_relevance * 0.3
    ))

    evidence = [f"heuristic fallback (no VLM inference): {reason}"]
    if has_strong_hook:
        evidence.append(f"Strong hook detected: \"{hook[:80]}\"")
    if has_payoff:
        evidence.append(f"Clear payoff: \"{payoff_text[:80]}\"")
    evidence.append(f"Semantic score: {semantic_score:.2f}")
    evidence.append(f"Duration: {end_sec - start_sec:.1f}s")

    return {
        "candidateId": candidate_key,
        "qualityScore": round(quality_score, 3),
        "visualEngagement": round(visual_engagement, 3),
        "semanticCoherence": round(semantic_coherence, 3),
        "productionQuality": round(production_quality, 3),
        "highlightRelevance": round(highlight_relevance, 3),
        "model": model,
        "modelVersion": model_version,
        "promptVersion": prompt_version,
        "sourceHash": "",
        "scoredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence": evidence,
        "heuristicFallback": True,
        "rawResponse": None,
    }


def score_candidate_with_vlm(
    source_path: str,
    start_sec: float,
    end_sec: float,
    candidate: Dict,
    words: List[Dict],
    model: str,
    model_version: str,
    prompt_version: str,
    api_key: str,
    candidate_key: str,
    max_frames: int,
) -> Dict:
    """Score a candidate with NVIDIA Nemotron 3 Nano Omni (ADVISORY ONLY)."""
    transcript = build_transcript_context(words)
    log(
        f"[VLM] Transcript chars={len(transcript['excerpt'])} "
        f"words={transcript['word_count']} "
        f"speakers={transcript['speaker_count']}"
    )

    frames = extract_frames_ffmpeg(source_path, start_sec, end_sec, max_frames=max_frames)
    log(f"[VLM] Keyframes={len(frames)}")

    if not frames:
        log("[VLM] FALLBACK reason=no keyframes could be extracted")
        return heuristic_score(
            candidate, start_sec, end_sec, model, model_version, prompt_version,
            candidate_key, "no keyframes could be extracted",
        )

    t0 = time.time()
    try:
        result = call_nvidia_api(frames, candidate, transcript, model, api_key)
    except VlmRequestError as exc:
        log(f"[VLM] FALLBACK reason={exc.reason}")
        return heuristic_score(
            candidate, start_sec, end_sec, model, model_version, prompt_version,
            candidate_key, f"inference failed: {exc.reason}",
        )
    elapsed = time.time() - t0
    raw_text = result["_raw_text"]
    raw = result["_parsed"]
    log(f"[VLM] INFERENCE COMPLETE elapsed={elapsed:.1f}s")

    scores = normalize_scores(raw)
    if not scores:
        log("[VLM] FAILED reason=model response contained no usable numeric scores")
        return heuristic_score(
            candidate, start_sec, end_sec, model, model_version, prompt_version,
            candidate_key, "model response contained no usable scores",
        )

    evidence = raw.get("observed_evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(e) for e in evidence][:8]
    reason = str(raw.get("reason", ""))
    if reason:
        evidence.append(f"reason: {reason}")

    return {
        "candidateId": candidate_key,
        "qualityScore": round(scores.get("overall_quality", 0.5), 3),
        "visualEngagement": round(scores.get("visual_engagement", 0.5), 3),
        "semanticCoherence": round(scores.get("semantic_coherence", 0.5), 3),
        "productionQuality": round(scores.get("production_quality", 0.5), 3),
        "highlightRelevance": round(scores.get("highlight_relevance", 0.5), 3),
        "model": model,
        "modelVersion": model_version,
        "promptVersion": prompt_version,
        "sourceHash": "",
        "scoredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence": evidence,
        "heuristicFallback": False,
        # The actual model text is retained (bounded) so the backend window and
        # the DB row both show what the VLM really said.
        "rawResponse": raw_text[:4000],
    }


def resolve_api_key() -> Optional[str]:
    """Resolve the NVIDIA API key from the environment.

    The provider is NVIDIA Nemotron 3 Nano Omni, so the credential is
    NVIDIA_API_KEY. OpenRouter is retained ONLY as a backward-compatible
    fallback so an existing install does not break on upgrade -- but it is no
    longer the active provider and is never logged.
    """
    for name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            if name != "NVIDIA_API_KEY":
                log(f"using legacy {name} credential for the NVIDIA endpoint")
            return value
    return None


def resolve_model() -> str:
    """Active VLM model. NVIDIA Nemotron 3 Nano Omni by default."""
    return (
        os.environ.get("AUTOSHORTS_VLM_MODEL", "").strip()
        or NVIDIA_DEFAULT_MODEL
    )


def main():
    parser = argparse.ArgumentParser(
        description="AutoShorts Phase 4 - advisory VLM scoring via NVIDIA Nemotron"
    )
    parser.add_argument("source", type=str, help="Path to source video")
    parser.add_argument("start_ms", type=float, help="Candidate start in milliseconds")
    parser.add_argument("end_ms", type=float, help="Candidate end in milliseconds")
    parser.add_argument("candidate_json", type=str, help="Path to CandidateDraft JSON")
    parser.add_argument("words_json", type=str, help="Path to candidate transcript words JSON")
    parser.add_argument("--candidate-key", type=str, default=None,
                        help="Stable key for result association")
    parser.add_argument("--max-frames", type=int, default=8, help="Frame budget")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--prompt-version", type=str, default=None)
    args = parser.parse_args()

    source_path = args.source
    start_sec = args.start_ms / 1000.0
    end_sec = args.end_ms / 1000.0
    candidate_key = args.candidate_key or f"cand_{start_sec:.3f}_{end_sec:.3f}"

    model = args.model or resolve_model()
    model_version = args.model_version or os.environ.get("AUTOSHORTS_VLM_MODEL_VERSION", "1.0")
    prompt_version = args.prompt_version or os.environ.get("AUTOSHORTS_VLM_PROMPT_VERSION", "v1.0")

    if not os.path.isfile(source_path):
        log(f"source not found: {source_path}")
        return 1
    if end_sec <= start_sec:
        log("non-positive range")
        return 1

    candidate = load_json_file(args.candidate_json)
    if candidate is None:
        log("candidate JSON invalid")
        return 1

    words = load_json_file(args.words_json) or []

    api_key = resolve_api_key()
    if not api_key:
        log("[VLM] FALLBACK reason=API key not set (NVIDIA_API_KEY)")
        emit(heuristic_score(
            candidate, start_sec, end_sec, model, model_version, prompt_version,
            candidate_key, "NVIDIA_API_KEY not set",
        ))
        return 0

    try:
        emit(score_candidate_with_vlm(
            source_path=source_path,
            start_sec=start_sec,
            end_sec=end_sec,
            candidate=candidate,
            words=words,
            model=model,
            model_version=model_version,
            prompt_version=prompt_version,
            api_key=api_key,
            candidate_key=candidate_key,
            max_frames=args.max_frames,
        ))
        return 0
    except Exception as exc:  # noqa: BLE001 - last-resort guard
        log(f"VLM scoring failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    import time  # noqa: F811 - keep import local to module entry semantics
    sys.exit(main())
