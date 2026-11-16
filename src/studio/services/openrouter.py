"""The single reusable OpenRouter client. Every external AI call goes through here.

Endpoints used (OpenRouter API, documented contracts):
  POST /chat/completions        text + vision reasoning (JSON output)
  POST /images                  image generation (base64 in data[].b64_json)
  POST /videos                  async video job -> GET /videos/{id} -> download
  POST /audio/speech            text to speech (raw audio bytes)
  POST /audio/transcriptions    optional speech to text
  GET  /models, /images/models, /videos/models, /generation   catalog + cost lookup
"""
from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from studio.config import ConfigError, Settings
from studio.utils import redact

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 524, 529}
VIDEO_TERMINAL_FAILURES = {"failed", "cancelled", "expired"}


class OpenRouterError(Exception):
    def __init__(self, message: str, status: int | None = None, error_type: str = ""):
        super().__init__(message)
        self.status, self.error_type = status, error_type


class UnsupportedOperation(OpenRouterError):
    """The configured model cannot perform the requested operation."""


@dataclass
class Usage:
    model: str = ""
    cost_usd: float | None = None          # None means the provider reported no cost
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generation_id: str = ""
    note: str = ""


@dataclass
class MediaResult:
    data: bytes
    media_type: str
    usage: Usage
    metadata: dict[str, Any] = field(default_factory=dict)


def image_data_url(path: Path | str, max_side: int = 1024) -> str:
    """Encode a local image as a JPEG data URL, downscaled to keep vision requests small."""
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenRouterClient:
    is_mock = False

    def __init__(self, settings: Settings, http: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep, log=None):
        self.settings = settings
        self.base_url = settings.openrouter_base_url.rstrip("/")
        if http is None:
            transport = httpx.HTTPTransport(local_address="0.0.0.0") if settings.force_ipv4 else None
            http = httpx.Client(timeout=settings.api_timeout_s, transport=transport)
        self.http = http
        self.sleep, self.log = sleep, log
        self._catalog: dict[str, dict[str, dict]] = {}

    # ---- low level ----------------------------------------------------------------
    def _headers(self, auth: bool = True) -> dict[str, str]:
        headers = {}
        if auth:
            headers["Authorization"] = f"Bearer {self.settings.require_api_key()}"
        if self.settings.app_referer:
            headers["HTTP-Referer"] = self.settings.app_referer
            if self.settings.app_title:
                headers["X-OpenRouter-Title"] = self.settings.app_title
        return headers

    def _url(self, path: str) -> str:
        if path.startswith("/api/"):                     # server-relative (e.g. polling_url)
            parts = urlsplit(self.base_url)
            return f"{parts.scheme}://{parts.netloc}{path}"
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None,
                 auth: bool = True, timeout: float | None = None) -> httpx.Response:
        """HTTP with bounded retries, Retry-After handling and uniform error mapping."""
        url = path if path.startswith("https://") else self._url(path)
        headers = self._headers(auth)                    # raises ConfigError before any network use
        attempts = self.settings.api_max_retries + 1
        last_error = "request failed"
        for attempt in range(1, attempts + 1):
            retry_after = None
            try:
                response = self.http.request(method, url, json=json, params=params, headers=headers,
                                             timeout=timeout or self.settings.api_timeout_s)
            except httpx.HTTPError as exc:
                last_error = f"network error: {type(exc).__name__}: {exc}"
            else:
                if response.status_code < 400:
                    return response
                error = self._error_from(response)
                if response.status_code not in RETRY_STATUS:
                    raise error
                last_error = str(error)
                header = response.headers.get("Retry-After", "")
                retry_after = float(header) if header.replace(".", "", 1).isdigit() else None
            if attempt < attempts:
                delay = min(retry_after if retry_after is not None else 2.0 ** attempt, 60.0)
                if self.log:
                    self.log.info(f"openrouter {method} {path}: {redact(last_error)}; "
                                  f"retry {attempt}/{attempts - 1} in {delay:.0f}s")
                self.sleep(delay)
        raise OpenRouterError(f"{method} {path} failed after {attempts} attempts: {redact(last_error)}")

    @staticmethod
    def _error_from(response: httpx.Response) -> OpenRouterError:
        message, error_type = response.text[:300], ""
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict):
                message = str(error.get("message", message))
                error_type = str((error.get("metadata") or {}).get("error_type", ""))
        except ValueError:
            pass
        hints = {401: "check OPENROUTER_API_KEY", 402: "insufficient OpenRouter credits",
                 403: "forbidden (permissions, guardrail or moderation)", 429: "rate limited"}
        hint = f" ({hints[response.status_code]})" if response.status_code in hints else ""
        return OpenRouterError(f"HTTP {response.status_code}{hint}: {redact(message)}",
                               response.status_code, error_type)

    @staticmethod
    def _json(response: httpx.Response, what: str, error_field_ok: bool = False) -> dict:
        try:
            body = response.json()
        except ValueError:
            raise OpenRouterError(f"{what}: response is not JSON") from None
        if not isinstance(body, dict):
            raise OpenRouterError(f"{what}: unexpected response shape")
        if body.get("error") and not error_field_ok:    # errors can arrive with HTTP 200
            error = body["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise OpenRouterError(f"{what}: {redact(str(message))[:300]}")
        return body

    @staticmethod
    def _usage(model: str, raw: Any, generation_id: str = "") -> Usage:
        raw = raw if isinstance(raw, dict) else {}
        cost = raw.get("cost")
        return Usage(model=model, cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
                     prompt_tokens=raw.get("prompt_tokens"), completion_tokens=raw.get("completion_tokens"),
                     generation_id=generation_id)

    # ---- chat (text + vision) -------------------------------------------------------
    def chat(self, model: str, system: str, user: str, *, images: list[Path] | None = None,
             json_schema: dict | None = None, task: str = "", context: dict | None = None
             ) -> tuple[str, Usage]:
        """One chat completion. Returns the raw assistant text (untrusted) and usage."""
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": image_data_url(p)}} for p in images]
        body: dict[str, Any] = {"model": model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": content}]}
        mode = self.settings.llm_json_mode
        if mode == "json_schema" and json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "result", "strict": False, "schema": json_schema}}
        elif mode != "none":
            body["response_format"] = {"type": "json_object"}
        data = self._json(self._request("POST", "chat/completions", json=body), "chat completion")
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise OpenRouterError("chat completion: response has no choices[0].message.content") from None
        if not isinstance(text, str) or not text.strip():
            raise OpenRouterError("chat completion: empty assistant message")
        return text, self._usage(model, data.get("usage"), str(data.get("id", "")))

    # ---- images ----------------------------------------------------------------------
    def generate_image(self, model: str, prompt: str, *, aspect_ratio: str = "16:9",
                       references: list[Path] | None = None, task: str = "",
                       context: dict | None = None) -> MediaResult:
        body: dict[str, Any] = {"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio}
        if references:
            body["input_references"] = [
                {"type": "image_url", "image_url": {"url": image_data_url(p)}} for p in references]
        data = self._json(self._request("POST", "images", json=body, timeout=self.settings.api_timeout_s * 2),
                          "image generation")
        items = data.get("data")
        if not isinstance(items, list) or not items or not isinstance(items[0], dict) \
                or not items[0].get("b64_json"):
            raise OpenRouterError("image generation: response has no data[0].b64_json")
        try:
            raw = base64.b64decode(items[0]["b64_json"], validate=True)
        except ValueError:
            raise OpenRouterError("image generation: b64_json is not valid base64") from None
        return MediaResult(raw, items[0].get("media_type") or "image/png",
                           self._usage(model, data.get("usage")),
                           {"references_used": len(references or [])})

    # ---- video (asynchronous job) --------------------------------------------------------
    def generate_video(self, model: str, prompt: str, *, duration_s: int, resolution: str,
                       aspect_ratio: str, first_frame: Path | None = None, task: str = "",
                       context: dict | None = None) -> MediaResult:
        """Submit a video job, poll until a terminal status, then download the MP4."""
        body: dict[str, Any] = {"model": model, "prompt": prompt, "duration": duration_s,
                                "resolution": resolution, "aspect_ratio": aspect_ratio,
                                "generate_audio": False}
        if first_frame:
            body["frame_images"] = [{"type": "image_url", "frame_type": "first_frame",
                                     "image_url": {"url": image_data_url(first_frame, 1280)}}]
        job = self._json(self._request("POST", "videos", json=body), "video submission")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise OpenRouterError("video submission: response has no job id")
        poll_path = job.get("polling_url") if str(job.get("polling_url", "")).startswith("/api/") \
            else f"videos/{job_id}"

        deadline = time.monotonic() + self.settings.video_timeout_s
        while True:
            status = str(job.get("status", ""))
            if status == "completed":
                break
            if status in VIDEO_TERMINAL_FAILURES:
                raise OpenRouterError(f"video job {job_id} ended with status '{status}': "
                                      f"{redact(str(job.get('error', 'no error message')))[:300]}")
            if status not in ("pending", "in_progress"):
                raise OpenRouterError(f"video job {job_id}: unexpected status '{status}'")
            if time.monotonic() >= deadline:
                raise OpenRouterError(f"video job {job_id} timed out after "
                                      f"{self.settings.video_timeout_s:.0f}s (last status '{status}')")
            self.sleep(self.settings.video_poll_interval_s)
            job = self._json(self._request("GET", poll_path), "video status", error_field_ok=True)

        unsigned, signed = job.get("unsigned_urls") or [], job.get("urls") or []
        if unsigned and str(unsigned[0]).startswith("/api/"):
            download = self._request("GET", unsigned[0], timeout=self.settings.api_timeout_s * 3)
        elif signed and str(signed[0]).startswith("https://"):
            # Pre-signed storage URL: fetched without our Authorization header.
            download = self._request("GET", signed[0], auth=False, timeout=self.settings.api_timeout_s * 3)
        else:
            raise OpenRouterError(f"video job {job_id} completed without a usable download URL")
        if not download.content:
            raise OpenRouterError(f"video job {job_id}: downloaded file is empty")
        usage = self._usage(model, job.get("usage"), str(job.get("generation_id", "")))
        return MediaResult(download.content, "video/mp4", usage,
                           {"job_id": job_id, "first_frame_used": bool(first_frame)})

    # ---- speech ----------------------------------------------------------------------------
    def text_to_speech(self, model: str, text: str, *, voice: str, fmt: str = "mp3", speed: float = 1.0,
                       instructions: str = "", task: str = "", context: dict | None = None) -> MediaResult:
        body: dict[str, Any] = {"model": model, "input": text, "voice": voice, "response_format": fmt}
        if abs(speed - 1.0) > 1e-6:
            body["speed"] = speed
        if instructions:
            body["instructions"] = instructions
        response = self._request("POST", "audio/speech", json=body)
        content_type = response.headers.get("content-type", "")
        media_type = content_type.split(";")[0].strip()
        rate = re.search(r"rate=(\d+)", content_type)
        if media_type.startswith("application/json"):
            self._json(response, "text to speech")           # raises if it carries an error
            raise OpenRouterError("text to speech: expected audio bytes, received JSON")
        if not response.content:
            raise OpenRouterError("text to speech: empty audio response")
        generation_id = response.headers.get("X-Generation-Id", "")
        usage = Usage(model=model, generation_id=generation_id, cost_usd=self.generation_cost(generation_id))
        if usage.cost_usd is None:
            usage.note = "TTS cost not returned; generation lookup unavailable"
        return MediaResult(response.content, media_type or ("audio/mpeg" if fmt == "mp3" else "audio/pcm"),
                           usage, {"sample_rate": int(rate.group(1))} if rate else {})

    def transcribe(self, model: str, audio_path: Path, *, language: str = "en") -> tuple[str, Usage]:
        fmt = audio_path.suffix.lstrip(".").lower() or "wav"
        body = {"model": model, "language": language, "input_audio": {
            "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"), "format": fmt}}
        data = self._json(self._request("POST", "audio/transcriptions", json=body), "transcription")
        if not isinstance(data.get("text"), str):
            raise OpenRouterError("transcription: response has no text")
        return data["text"], self._usage(model, data.get("usage"))

    def generation_cost(self, generation_id: str) -> float | None:
        """Best-effort cost lookup. Returns None (unknown) rather than guessing."""
        if not generation_id:
            return None
        for wait in (0.0, 1.5):
            if wait:
                self.sleep(wait)
            try:
                response = self.http.get(self._url("generation"), params={"id": generation_id},
                                         headers=self._headers(), timeout=20)
                cost = response.json().get("data", {}).get("total_cost") if response.status_code == 200 else None
            except (httpx.HTTPError, ValueError, AttributeError):
                cost = None
            if isinstance(cost, (int, float)):
                return float(cost)
        return None

    # ---- model catalog and capability validation (free GET requests) ---------------------------
    def catalog(self, which: str) -> dict[str, dict]:
        """`which` is 'models' (all modalities), 'images' or 'videos'. Cached per client."""
        if which not in self._catalog:
            path, params = {"models": ("models", {"output_modalities": "all"}),
                            "images": ("images/models", None),
                            "videos": ("videos/models", None)}[which]
            data = self._json(self._request("GET", path, params=params, auth=False), f"{which} catalog")
            items = data.get("data")
            if not isinstance(items, list):
                raise OpenRouterError(f"{which} catalog: response has no data list")
            self._catalog[which] = {m["id"]: m for m in items if isinstance(m, dict) and "id" in m}
        return self._catalog[which]

    def capabilities(self) -> tuple[dict[str, Any], list[str]]:
        """Check every enabled model against the catalog before any paid request.

        Returns (capabilities, problems). A non-empty problem list must stop the run.
        """
        s, problems, caps = self.settings, [], {"image_references": False, "tts_instructions": False,
                                                 "video_first_frame": False, "notes": []}
        models = self.catalog("models")

        def modalities(model_id: str, direction: str) -> list[str]:
            return (models[model_id].get("architecture") or {}).get(f"{direction}_modalities") or []

        def params(entry: dict) -> list[str]:
            value = entry.get("supported_parameters") or []
            return list(value) if isinstance(value, (list, dict)) else []

        for role, model_id in (("LLM_MODEL", s.llm_model), ("DIRECTOR_MODEL", s.director_model)):
            if model_id not in models:
                problems.append(f"{role} '{model_id}' is not in the OpenRouter model catalog")
            elif "text" not in modalities(model_id, "output"):
                problems.append(f"{role} '{model_id}' does not produce text")
        if s.vision_model not in models:
            problems.append(f"VISION_MODEL '{s.vision_model}' is not in the OpenRouter model catalog")
        elif "image" not in modalities(s.vision_model, "input"):
            problems.append(f"VISION_MODEL '{s.vision_model}' does not accept image input")

        if s.thumbnail_model:
            if s.thumbnail_model not in models:
                problems.append(f"THUMBNAIL_MODEL '{s.thumbnail_model}' is not in the OpenRouter model catalog")
            elif "text" not in modalities(s.thumbnail_model, "output"):
                problems.append(f"THUMBNAIL_MODEL '{s.thumbnail_model}' does not produce text")
            elif "image" not in modalities(s.thumbnail_model, "input"):
                problems.append(f"THUMBNAIL_MODEL '{s.thumbnail_model}' does not accept image input")

        if s.enable_images:
            if s.image_model not in models:
                problems.append(f"IMAGE_MODEL '{s.image_model}' is not in the OpenRouter model catalog")
            elif "image" not in modalities(s.image_model, "output"):
                problems.append(f"IMAGE_MODEL '{s.image_model}' does not generate images")
            else:
                entry = self.catalog("images").get(s.image_model)
                if entry is None:
                    problems.append(f"IMAGE_MODEL '{s.image_model}' is not served by POST /images")
                else:
                    ratios = (entry.get("supported_parameters") or {})
                    ratios = ratios.get("aspect_ratio", {}).get("values") if isinstance(ratios, dict) else None
                    if ratios and s.aspect_ratio not in ratios:
                        problems.append(f"IMAGE_MODEL '{s.image_model}' does not support aspect ratio "
                                        f"{s.aspect_ratio} (supported: {', '.join(map(str, ratios))})")
                    accepts = "input_references" in params(entry) or \
                        "image" in ((entry.get("architecture") or {}).get("input_modalities") or [])
                    if s.image_use_references == "on" and not accepts:
                        problems.append(f"IMAGE_USE_REFERENCES=on but '{s.image_model}' does not accept "
                                        "reference images")
                    caps["image_references"] = accepts and s.image_use_references != "off"
                    limit = entry.get("supported_parameters")
                    limit = limit.get("input_references", {}).get("max") if isinstance(limit, dict) else None
                    caps["image_reference_max"] = int(limit) if isinstance(limit, (int, float)) else 1
                    if not caps["image_references"]:
                        caps["notes"].append("Image model is not given character reference images; "
                                             "character consistency relies on prompt text only.")

        extra_image_models = {"THUMBNAIL_IMAGE_MODEL": s.thumbnail_image_model,
                              "IMAGE_RETRY_MODEL": s.image_retry_model}
        for setting, name in extra_image_models.items() if s.enable_images else ():
            if not name or name == s.image_model:
                continue
            entry = self.catalog("images").get(name)
            if name not in models or "image" not in modalities(name, "output"):
                problems.append(f"{setting} '{name}' is not an image model in the OpenRouter catalog")
            elif entry is None:
                problems.append(f"{setting} '{name}' is not served by POST /images")
            else:
                ratios = entry.get("supported_parameters") or {}
                ratios = ratios.get("aspect_ratio", {}).get("values") if isinstance(ratios, dict) else None
                if ratios and s.aspect_ratio not in ratios:
                    problems.append(f"{setting} '{name}' does not support aspect ratio {s.aspect_ratio}")
                accepts = "input_references" in params(entry) or \
                    "image" in ((entry.get("architecture") or {}).get("input_modalities") or [])
                accepts = accepts and s.image_use_references != "off"
                caps.setdefault("references_by_model", {})[name] = accepts
                if setting == "THUMBNAIL_IMAGE_MODEL":
                    caps["thumbnail_image_references"] = accepts

        if s.enable_tts:
            if s.tts_model not in models:
                problems.append(f"TTS_MODEL '{s.tts_model}' is not in the OpenRouter model catalog")
            elif not {"speech", "audio"} & set(modalities(s.tts_model, "output")):
                problems.append(f"TTS_MODEL '{s.tts_model}' is not a text-to-speech model")
            else:
                entry = models[s.tts_model]
                endpoints = entry.get("endpoints")
                if isinstance(endpoints, list) and endpoints and "/api/v1/audio/speech" not in endpoints:
                    problems.append(f"TTS_MODEL '{s.tts_model}' is not served by POST /audio/speech")
                voices = entry.get("supported_voices") or []
                if voices and s.tts_voice not in voices:
                    problems.append(f"TTS_VOICE '{s.tts_voice}' is not a voice of '{s.tts_model}' "
                                    f"(available: {', '.join(map(str, voices[:12]))}"
                                    f"{', ...' if len(voices) > 12 else ''})")
                supported = "instructions" in params(entry)
                caps["tts_instructions"] = s.tts_instructions == "on" or (
                    s.tts_instructions == "auto" and supported)
                if not caps["tts_instructions"]:
                    caps["notes"].append("TTS delivery instructions are not sent to the speech model "
                                         "(unsupported or disabled).")

        if s.enable_video:
            entry = self.catalog("videos").get(s.video_model)
            if entry is None:
                problems.append(f"VIDEO_MODEL '{s.video_model}' is not in the OpenRouter video catalog")
            else:
                for label, value, key in (
                        ("duration", s.video_clip_seconds, "supported_durations"),
                        ("resolution", s.video_resolution, "supported_resolutions"),
                        ("aspect ratio", s.aspect_ratio, "supported_aspect_ratios")):
                    allowed = entry.get(key)
                    if allowed and value not in allowed:
                        problems.append(f"VIDEO_MODEL '{s.video_model}' does not support {label} {value} "
                                        f"(supported: {', '.join(map(str, allowed))})")
                caps["video_first_frame"] = "first_frame" in (entry.get("supported_frame_images") or [])
                if not caps["video_first_frame"]:
                    caps["notes"].append("Video model does not accept a first-frame image; clips are "
                                         "generated from text only and may not match the still images.")

        if s.stt_model and s.stt_model not in models:
            problems.append(f"STT_MODEL '{s.stt_model}' is not in the OpenRouter model catalog")
        return caps, problems


def require_capabilities(client: Any) -> dict[str, Any]:
    caps, problems = client.capabilities()
    if problems:
        raise ConfigError("Model configuration problems (no paid request was made):\n  - "
                          + "\n  - ".join(problems))
    return caps
