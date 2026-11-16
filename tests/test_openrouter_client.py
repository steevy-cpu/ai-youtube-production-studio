"""OpenRouter client against a mocked HTTP transport (httpx.MockTransport). No real requests."""
import base64
import json

import httpx
import pytest

from studio.config import ConfigError, Settings
from studio.services.openrouter import OpenRouterClient, OpenRouterError, require_capabilities
from studio.services.placeholders import sample_image

KEY = "sk-or-v1-test-key-123456"


def make_client(handler, **changes):
    values = {"openrouter_api_key": KEY, "api_max_retries": 2, "video_poll_interval_s": 0.01,
              "video_timeout_s": 5} | changes
    settings = Settings(**values)
    sleeps = []
    client = OpenRouterClient(settings, http=httpx.Client(transport=httpx.MockTransport(handler)),
                              sleep=sleeps.append)
    return client, sleeps


def test_missing_api_key_stops_before_any_request():
    calls = []
    client = OpenRouterClient(Settings(), http=httpx.Client(transport=httpx.MockTransport(
        lambda r: calls.append(r) or httpx.Response(200, json={}))))
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        client.chat("m", "system", "user")
    assert calls == []


def test_chat_sends_auth_and_parses_usage():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        assert body["response_format"] == {"type": "json_object"} and body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"id": "gen-1", "choices": [{"message": {"content": '{"a": 1}'}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.002}})
    client, _ = make_client(handler)
    text, usage = client.chat("vendor/model", "system", "user")
    assert text == '{"a": 1}' and usage.cost_usd == 0.002 and usage.generation_id == "gen-1"


def test_cost_is_unknown_when_the_provider_reports_none():
    client, _ = make_client(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))
    assert client.chat("m", "s", "u")[1].cost_usd is None


def test_rate_limit_is_retried_with_retry_after_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"Retry-After": "7"},
                                  json={"error": {"code": 429, "message": "Rate limit exceeded"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    client, sleeps = make_client(handler)
    client.chat("m", "s", "u")
    assert len(attempts) == 3 and sleeps == [7.0, 7.0]


def test_retries_are_bounded_and_auth_errors_are_not_retried():
    attempts = []

    def always_503(request):
        attempts.append(1)
        return httpx.Response(503, json={"error": {"code": 503, "message": "no provider"}})
    client, _ = make_client(always_503)
    with pytest.raises(OpenRouterError, match="after 3 attempts"):
        client.chat("m", "s", "u")
    assert len(attempts) == 3                                   # api_max_retries=2 -> 3 tries, then stop

    attempts.clear()

    def unauthorized(request):
        attempts.append(1)
        return httpx.Response(401, json={"error": {"code": 401, "message": f"bad key {KEY}"}})
    client, _ = make_client(unauthorized)
    with pytest.raises(OpenRouterError) as info:
        client.chat("m", "s", "u")
    assert len(attempts) == 1 and info.value.status == 401
    assert KEY not in str(info.value) and "OPENROUTER_API_KEY" in str(info.value)


@pytest.mark.parametrize("payload", [
    {"choices": []}, {"choices": [{"message": {"content": ""}}]}, {"error": {"message": "provider exploded"}}])
def test_malformed_chat_responses_are_rejected(payload):
    client, _ = make_client(lambda r: httpx.Response(200, json=payload))
    with pytest.raises(OpenRouterError):
        client.chat("m", "s", "u")


def test_image_generation_decodes_base64_and_validates(tmp_path):
    png = sample_image("x", 320, 180)
    reference = tmp_path / "ref.png"
    reference.write_bytes(png)

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/v1/images" and body["aspect_ratio"] == "16:9"
        assert body["input_references"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(200, json={"created": 1, "usage": {"cost": 0.04}, "data": [
            {"b64_json": base64.b64encode(png).decode(), "media_type": "image/png"}]})
    client, _ = make_client(handler)
    result = client.generate_image("img/model", "a prompt", references=[reference])
    assert result.data == png and result.usage.cost_usd == 0.04 and result.metadata["references_used"] == 1

    client, _ = make_client(lambda r: httpx.Response(200, json={"data": [{"url": "https://x/y.png"}]}))
    with pytest.raises(OpenRouterError, match="b64_json"):
        client.generate_image("img/model", "a prompt")


def test_video_job_is_submitted_polled_and_downloaded():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["duration"] == 4 and body["resolution"] == "720p"
            return httpx.Response(202, json={"id": "job1", "status": "pending",
                                             "polling_url": "/api/v1/videos/job1"})
        if request.url.path.endswith("/content"):
            assert request.headers["Authorization"] == f"Bearer {KEY}"
            return httpx.Response(200, content=b"MP4DATA")
        polls = sum(1 for m, p in seen if m == "GET" and p == "/api/v1/videos/job1")
        if polls < 2:
            return httpx.Response(200, json={"id": "job1", "status": "in_progress"})
        return httpx.Response(200, json={"id": "job1", "status": "completed", "usage": {"cost": 0.5},
                                         "unsigned_urls": ["/api/v1/videos/job1/content?index=0"]})
    client, sleeps = make_client(handler)
    result = client.generate_video("vid/model", "a prompt", duration_s=4, resolution="720p", aspect_ratio="16:9")
    assert result.data == b"MP4DATA" and result.usage.cost_usd == 0.5 and len(sleeps) == 2


def test_video_failure_and_timeout_are_reported():
    def failed(request):
        if request.method == "POST":
            return httpx.Response(202, json={"id": "job2", "status": "pending"})
        return httpx.Response(200, json={"id": "job2", "status": "failed", "error": "content policy"})
    client, _ = make_client(failed)
    with pytest.raises(OpenRouterError, match="failed.*content policy"):
        client.generate_video("v", "p", duration_s=4, resolution="720p", aspect_ratio="16:9")

    client, _ = make_client(lambda r: httpx.Response(200, json={"id": "job3", "status": "pending"}),
                            video_timeout_s=0.001)
    with pytest.raises(OpenRouterError, match="timed out"):
        client.generate_video("v", "p", duration_s=4, resolution="720p", aspect_ratio="16:9")


def test_text_to_speech_returns_audio_and_looks_up_cost():
    def handler(request):
        if request.url.path.endswith("/audio/speech"):
            body = json.loads(request.content)
            assert body["voice"] == "alloy" and "instructions" not in body and "speed" not in body
            return httpx.Response(200, content=b"ID3audio", headers={
                "content-type": "audio/mpeg", "X-Generation-Id": "gen-tts"})
        assert request.url.params["id"] == "gen-tts"
        return httpx.Response(200, json={"data": {"total_cost": 0.0123}})
    client, _ = make_client(handler)
    result = client.text_to_speech("tts/model", "hello", voice="alloy")
    assert result.data == b"ID3audio" and result.media_type == "audio/mpeg" and result.usage.cost_usd == 0.0123


def test_text_to_speech_cost_stays_unknown_without_generation_data():
    def handler(request):
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=b"ID3audio", headers={"content-type": "audio/mpeg"})
        return httpx.Response(403, json={"error": {"message": "no permission"}})
    client, _ = make_client(handler)
    usage = client.text_to_speech("tts/model", "hello", voice="alloy").usage
    assert usage.cost_usd is None and "cost not returned" in usage.note


def test_capability_validation_reports_unsupported_models():
    catalog = {"data": [
        {"id": "text/llm", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
        {"id": "img/model", "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]}},
        {"id": "tts/model", "architecture": {"output_modalities": ["speech"]}, "supported_parameters": [],
         "supported_voices": ["Zephyr", "Puck"], "endpoints": ["/api/v1/audio/speech"]}]}

    def handler(request):
        if request.url.path.endswith("/images/models"):
            return httpx.Response(200, json={"data": [{"id": "img/model", "supported_parameters": {
                "aspect_ratio": {"type": "enum", "values": ["1:1"]}}}]})
        if request.url.path.endswith("/videos/models"):
            return httpx.Response(200, json={"data": [{"id": "vid/model", "supported_durations": [8],
                                                       "supported_resolutions": ["720p"]}]})
        return httpx.Response(200, json=catalog)
    client, _ = make_client(handler, llm_model="text/llm", director_model="missing/model",
                            vision_model="text/llm", image_model="img/model", tts_model="tts/model",
                            enable_video=True, video_model="vid/model", video_clip_seconds=4)
    caps, problems = client.capabilities()
    text = " | ".join(problems)
    assert "DIRECTOR_MODEL 'missing/model' is not in" in text
    assert "VISION_MODEL 'text/llm' does not accept image input" in text
    assert "THUMBNAIL_MODEL 'anthropic/claude-sonnet-5.5' is not in" in text
    client, _ = make_client(handler, thumbnail_model="text/llm")
    assert "THUMBNAIL_MODEL 'text/llm' does not accept image input" in " | ".join(client.capabilities()[1])
    assert "does not support aspect ratio 16:9" in text
    assert "does not support duration 4" in text
    assert "TTS_VOICE 'Kore' is not a voice of 'tts/model' (available: Zephyr, Puck)" in text
    assert caps["tts_instructions"] is False and caps["video_first_frame"] is False
    with pytest.raises(ConfigError, match="no paid request was made"):
        require_capabilities(client)
