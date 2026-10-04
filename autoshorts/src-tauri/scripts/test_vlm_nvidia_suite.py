#!/usr/bin/env python3
"""
Focused regression tests for the ADVISORY VLM layer (NVIDIA Nemotron 3 Nano Omni).

These tests never call the live API. They pin the behaviour that matters for
production safety:

  * bounded HTTP policy (no infinite retries, permanent errors fail fast)
  * honest status for every failure class (401 vs 429 vs timeout vs bad JSON)
  * a heuristic fallback is NEVER reported as a real inference result
  * no credential can leak into logs, artifacts, or the emitted document
  * the existing structured result contract is preserved
  * the VLM cannot influence clip generation

Run:
    D:/College/Autoshorts 11.0/.venv/Scripts/python.exe -m pytest \\
        test_vlm_nvidia_suite.py -q
"""

import json
import os
import sys

import pytest

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

import vlm_scoring as vlm  # noqa: E402


# ── request construction ────────────────────────────────────────────────────

class TestRequestConstruction:
    def test_endpoint_and_model_are_nvidia(self):
        assert vlm.NVIDIA_INVOKE_URL == (
            "https://integrate.api.nvidia.com/v1/chat/completions"
        )
        assert vlm.NVIDIA_DEFAULT_MODEL == (
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        )

    def test_active_model_default_is_nvidia_not_qwen(self):
        os.environ.pop("AUTOSHORTS_VLM_MODEL", None)
        assert vlm.resolve_model() == vlm.NVIDIA_DEFAULT_MODEL
        assert "qwen" not in vlm.resolve_model().lower()

    def test_env_override_of_model_is_respected(self):
        os.environ["AUTOSHORTS_VLM_MODEL"] = "nvidia/some-other-model"
        try:
            assert vlm.resolve_model() == "nvidia/some-other-model"
        finally:
            os.environ.pop("AUTOSHORTS_VLM_MODEL", None)

    def test_payload_is_multimodal_and_bounded(self, monkeypatch):
        """Real candidate keyframes become image_url parts; generation is bounded."""
        captured = {}

        class FakeResp:
            status_code = 200
            headers = {}
            text = "{}"

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": json.dumps(
                            {"visual_engagement": 0.7, "overall_quality": 0.6}
                        )}}
                    ]
                }

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResp()

        monkeypatch.setattr(vlm.requests_module(), "post", fake_post)

        result = vlm.call_nvidia_api(
            frames=["QUJD", "REVG"],
            candidate={"hook": "h", "payoffText": "p", "rationale": "r"},
            transcript={
                "speaker_desc": "S1",
                "word_count": 3,
                "excerpt": "some words",
            },
            model=vlm.NVIDIA_DEFAULT_MODEL,
            api_key="secret-key-value",
        )

        assert captured["url"] == vlm.NVIDIA_INVOKE_URL
        # Multimodal: text part + one image_url per real keyframe.
        parts = captured["json"]["messages"][1]["content"]
        assert parts[0]["type"] == "text"
        images = [p for p in parts if p["type"] == "image_url"]
        assert len(images) == 2
        assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
        # Bounded generation + explicit non-streaming mode.
        assert captured["json"]["max_tokens"] <= 8192
        assert captured["json"]["stream"] is False
        # Bounded timeouts (connect, read).
        assert captured["timeout"] == (
            vlm.NVIDIA_CONNECT_TIMEOUT_SEC,
            vlm.NVIDIA_READ_TIMEOUT_SEC,
        )
        assert result["_parsed"]["visual_engagement"] == 0.7

    def test_authorization_header_is_sent_but_never_logged(self, monkeypatch, capsys):
        seen = {}

        class FakeResp:
            status_code = 200
            headers = {}
            text = "{}"

            def json(self):
                return {"choices": [{"message": {"content": json.dumps(
                    {"overall_quality": 0.5})}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            seen.update(headers or {})
            return FakeResp()

        monkeypatch.setattr(vlm.requests_module(), "post", fake_post)
        vlm.call_nvidia_api(
            frames=["QUJD"],
            candidate={"hook": "h"},
            transcript={"speaker_desc": "s", "word_count": 1, "excerpt": "x"},
            model=vlm.NVIDIA_DEFAULT_MODEL,
            api_key="sk-must-not-appear",
        )
        assert seen["Authorization"] == "Bearer sk-must-not-appear"

        # The diagnostic stream must never contain the credential.
        captured = capsys.readouterr()
        assert "sk-must-not-appear" not in captured.out
        assert "sk-must-not-appear" not in captured.err


# ── bounded HTTP policy ─────────────────────────────────────────────────────

class TestBoundedHttpPolicy:
    def test_attempts_are_bounded(self):
        assert vlm.NVIDIA_MAX_ATTEMPTS >= 1
        assert vlm.NVIDIA_MAX_ATTEMPTS <= 5, "retries must stay bounded"

    def test_permanent_statuses_are_not_retryable(self):
        for status in (400, 401, 403, 404):
            assert status not in vlm.NVIDIA_RETRYABLE_STATUS, (
                f"HTTP {status} is permanent and must not be retried"
            )

    def test_transient_statuses_are_retryable(self):
        for status in (429, 500, 502, 503, 504):
            assert status in vlm.NVIDIA_RETRYABLE_STATUS

    def test_401_fails_fast_without_retrying(self, monkeypatch):
        calls = {"n": 0}

        class FakeResp:
            status_code = 401
            headers = {}
            text = '{"detail":"Authentication failed"}'

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            return FakeResp()

        monkeypatch.setattr(vlm.requests_module(), "post", fake_post)
        with pytest.raises(vlm.VlmRequestError) as exc:
            vlm.call_nvidia_api(
                frames=["QUJD"],
                candidate={"hook": "h"},
                transcript={"speaker_desc": "s", "word_count": 1, "excerpt": "x"},
                model=vlm.NVIDIA_DEFAULT_MODEL,
                api_key="bad",
            )
        assert "401" in exc.value.reason
        assert calls["n"] == 1, "a permanent auth error must not be retried"

    def test_429_is_retried_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        class RateLimited:
            status_code = 429
            headers = {"Retry-After": "0"}
            text = "slow down"

        class Ok:
            status_code = 200
            headers = {}
            text = "{}"

            def json(self):
                return {"choices": [{"message": {"content": json.dumps(
                    {"overall_quality": 0.42})}}]}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            return RateLimited() if calls["n"] == 1 else Ok()

        monkeypatch.setattr(vlm.requests_module(), "post", fake_post)
        monkeypatch.setattr(vlm.time, "sleep", lambda s: None)
        out = vlm.call_nvidia_api(
            frames=["QUJD"],
            candidate={"hook": "h"},
            transcript={"speaker_desc": "s", "word_count": 1, "excerpt": "x"},
            model=vlm.NVIDIA_DEFAULT_MODEL,
            api_key="k",
        )
        assert calls["n"] == 2, "a transient 429 should be retried once"
        assert out["_parsed"]["overall_quality"] == 0.42

    def test_retry_exhaustion_raises_with_concrete_reason(self, monkeypatch):
        calls = {"n": 0}

        class ServerError:
            status_code = 503
            headers = {}
            text = "unavailable"

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            return ServerError()

        monkeypatch.setattr(vlm.requests_module(), "post", fake_post)
        monkeypatch.setattr(vlm.time, "sleep", lambda s: None)
        with pytest.raises(vlm.VlmRequestError) as exc:
            vlm.call_nvidia_api(
                frames=["QUJD"],
                candidate={"hook": "h"},
                transcript={"speaker_desc": "s", "word_count": 1, "excerpt": "x"},
                model=vlm.NVIDIA_DEFAULT_MODEL,
                api_key="k",
            )
        assert calls["n"] == vlm.NVIDIA_MAX_ATTEMPTS
        assert "503" in exc.value.reason

    def test_timeout_is_reported_as_such(self, monkeypatch):
        class FakeRequests:
            class exceptions:
                class Timeout(Exception):
                    pass

                class ConnectionError(Exception):
                    pass

                class RequestException(Exception):
                    pass

            @staticmethod
            def post(*a, **k):
                raise FakeRequests.exceptions.Timeout("read timed out")

        # The client is reached through the accessor so there is one patch point.
        monkeypatch.setattr(vlm, "requests_module", lambda: FakeRequests)
        monkeypatch.setattr(vlm.time, "sleep", lambda s: None)
        with pytest.raises(vlm.VlmRequestError) as exc:
            vlm.call_nvidia_api(
                frames=["QUJD"],
                candidate={"hook": "h"},
                transcript={"speaker_desc": "s", "word_count": 1, "excerpt": "x"},
                model=vlm.NVIDIA_DEFAULT_MODEL,
                api_key="k",
            )
        assert "timeout" in exc.value.reason.lower()


# ── response parsing ────────────────────────────────────────────────────────

class TestResponseParsing:
    def test_plain_json(self):
        got = vlm._parse_model_json('{"visual_engagement": 0.5}')
        assert got["visual_engagement"] == 0.5

    def test_markdown_fenced_json(self):
        got = vlm._parse_model_json('```json\n{"overall_quality": 0.9}\n```')
        assert got["overall_quality"] == 0.9

    def test_json_with_surrounding_prose(self):
        text = 'Sure! Here you go:\n{"overall_quality": 0.3}\nHope that helps.'
        assert vlm._parse_model_json(text)["overall_quality"] == 0.3

    def test_nested_braces_survive(self):
        text = '{"reason": "a } brace", "overall_quality": 0.6}'
        assert vlm._parse_model_json(text)["overall_quality"] == 0.6

    def test_truncated_json_raises_rather_than_guessing(self):
        with pytest.raises(vlm.VlmRequestError):
            vlm._parse_model_json('{"overall_quality": 0.5')

    def test_no_json_at_all_raises(self):
        with pytest.raises(vlm.VlmRequestError):
            vlm._parse_model_json("I cannot answer that.")

    def test_empty_content_raises(self):
        with pytest.raises(vlm.VlmRequestError):
            vlm._extract_content({"choices": [{"message": {"content": "  "}}]})

    def test_missing_choices_raises(self):
        with pytest.raises(vlm.VlmRequestError):
            vlm._extract_content({"choices": []})

    def test_content_as_list_of_parts(self):
        got = vlm._extract_content(
            {"choices": [{"message": {"content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": " world"},
            ]}}]}
        )
        assert got == "hello world"

    def test_unknown_fields_do_not_break_normalization(self):
        """Extra model fields must be ignored, not crash parsing."""
        parsed = vlm._parse_model_json(
            '{"overall_quality": 0.5, "brand_new_field": 123, "nested": {"a": 1}}'
        )
        scores = vlm.normalize_scores(parsed)
        assert scores["overall_quality"] == 0.5
        assert "brand_new_field" not in scores

    def test_scores_are_clamped_and_garbage_dropped(self):
        scores = vlm.normalize_scores({
            "overall_quality": 5.0,      # clamp high
            "visual_engagement": -2.0,   # clamp low
            "production_quality": "abc", # unparseable
        })
        assert scores["overall_quality"] == 1.0
        assert scores["visual_engagement"] == 0.0
        assert "production_quality" not in scores


# ── fallback honesty ─────────────────────────────────────────────────────────

class TestFallbackHonesty:
    def test_heuristic_score_is_marked_and_not_a_real_result(self):
        out = vlm.heuristic_score(
            candidate={"start": 0.0, "end": 30.0, "hook": "h", "rationale": "r"},
            start_sec=0.0, end_sec=30.0,
            model=vlm.NVIDIA_DEFAULT_MODEL,
            model_version="1.0", prompt_version="v1.0",
            candidate_key="cand_x", reason="NVIDIA_API_KEY not set",
        )
        assert out["heuristicFallback"] is True
        assert out["rawResponse"] is None
        joined = " ".join(out["evidence"])
        assert "heuristic fallback" in joined
        assert "NVIDIA_API_KEY not set" in joined

    def test_missing_key_is_resolved_from_nvidia_first(self):
        for name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
            os.environ.pop(name, None)
        assert vlm.resolve_api_key() is None

        os.environ["OPEN_ROUTER_API_KEY"] = "legacy"
        try:
            assert vlm.resolve_api_key() == "legacy"
        finally:
            os.environ.pop("OPEN_ROUTER_API_KEY", None)

        os.environ["NVIDIA_API_KEY"] = "nv"
        try:
            assert vlm.resolve_api_key() == "nv"
        finally:
            os.environ.pop("NVIDIA_API_KEY", None)


# ── advisory-only contract ───────────────────────────────────────────────────

class TestAdvisoryOnly:
    def test_no_frame_boundaries_are_mutated_by_vlm_code(self):
        """
        The VLM module must not contain the authority-bearing field names. If a
        future change starts rewriting these, it breaks the safety contract.
        """
        # Match real *assignments*, not prose: a docstring mentioning
        # "rendering" is documentation, while `candidate["end"] = ...` would be
        # the VLM overwriting an authority field.
        import re
        src = open(vlm.__file__, encoding="utf-8").read()
        code_only = re.sub(r'""".*?"""', "", src, flags=re.S)
        code_only = re.sub(r"#.*", "", code_only)
        for forbidden in ("payoffEnd", "payoff_end", "candidateEnd",
                          "candidate_end"):
            assert not re.search(
                rf'["\']{forbidden}["\']\s*\]\s*=', code_only
            ), f"VLM must not assign to authority field {forbidden}"

    def test_emitted_document_keeps_existing_schema(self):
        out = vlm.heuristic_score(
            candidate={"start": 1.0, "end": 40.0, "hook": "h", "rationale": "r"},
            start_sec=1.0, end_sec=40.0,
            model=vlm.NVIDIA_DEFAULT_MODEL,
            model_version="1.0", prompt_version="v1.0",
            candidate_key="cand_1", reason="test",
        )
        for field in ("candidateId", "qualityScore", "visualEngagement",
                      "semanticCoherence", "productionQuality",
                      "highlightRelevance", "model", "modelVersion",
                      "promptVersion", "sourceHash", "scoredAt", "evidence",
                      "heuristicFallback", "rawResponse"):
            assert field in out, f"schema field missing: {field}"
        # It must serialize for the Rust side to parse.
        json.dumps(out)