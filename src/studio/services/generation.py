"""Image, video and speech generation services.

Each service checks the budget, calls the shared client, validates and saves the returned
bytes under the project directory and records a MediaAsset (with provenance) in memory.
When a feature is switched off, a labelled local placeholder is produced instead and stored
with is_sample=True, so it can never be mistaken for model output.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from studio.orchestration.context import Ctx
from studio.rendering.ffmpeg import audio_duration, probe
from studio.schemas import AnimationPrompt, MediaAsset, NarrationSegment
from studio.services.openrouter import OpenRouterError
from studio.services.placeholders import pcm_to_wav, sample_audio, sample_image
from studio.utils import safe_path

DIR_FOR_KIND = {"image": "images", "reference": "references", "thumbnail": "thumbnails"}


def _provenance(ctx: Ctx, model: str) -> tuple[str, bool]:
    if ctx.client.is_mock:
        return "dry-run sample generated locally (no AI model used)", True
    return f"AI-generated via OpenRouter model {model}", False


def _next_path(ctx: Ctx, folder: str, name: str, parent_id: str, kind: str, suffix: str) -> Path:
    attempt = len(ctx.store.assets(ctx.project_id, kind, parent_id)) + 1
    return safe_path(ctx.dirs.sub(folder), f"{name}_v{attempt}{suffix}")


class ImageService:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def generate(self, prompt_text: str, *, kind: str, parent_id: str, prompt_id: str | None, name: str,
                 agent: str, label: str = "", references: list[Path] | None = None) -> MediaAsset:
        ctx, s = self.ctx, self.ctx.settings
        path = _next_path(ctx, DIR_FOR_KIND[kind], name, parent_id, kind, ".png")
        metadata: dict = {"prompt": prompt_text}

        if not s.enable_images and not ctx.client.is_mock:
            raw, model = sample_image(label or prompt_text[:120]), ""
            provenance, is_sample = "local placeholder (ENABLE_IMAGES=false; no AI model used)", True
        else:
            use_refs = [p for p in (references or []) if p.is_file()] if ctx.caps.get("image_references") else []
            use_refs = use_refs[:ctx.caps.get("image_reference_max", 1)]
            if references and not use_refs:
                metadata["limitation"] = "reference images not passed to the model (unsupported or disabled)"
            ctx.budget.check("image")
            result = ctx.client.generate_image(s.image_model, prompt_text, aspect_ratio=s.aspect_ratio,
                                               references=use_refs or None, task=ctx.task,
                                               context={"label": label or prompt_text[:120]})
            ctx.budget.record(ctx.task, agent, "image", result.usage)
            raw, model = result.data, result.usage.model
            metadata.update(result.metadata, cost_usd=result.usage.cost_usd)
            provenance, is_sample = _provenance(ctx, model)

        try:                                   # validate the untrusted bytes and normalise to PNG
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                img.convert("RGB").save(path, "PNG")
                width, height = img.size
        except (UnidentifiedImageError, OSError) as exc:
            raise OpenRouterError(f"image generation returned data that is not a valid image: {exc}") from None
        asset = MediaAsset(project_id=ctx.project_id, parent_id=parent_id, kind=kind, path=str(path),
                           mime="image/png", width=width, height=height, model=model, prompt_id=prompt_id,
                           provenance=provenance, is_sample=is_sample, metadata=metadata, status="generated")
        return ctx.store.add_asset(asset)


class VideoService:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def generate(self, prompt: AnimationPrompt, *, shot_id: str, name: str, agent: str,
                 first_frame: Path | None) -> MediaAsset:
        ctx, s = self.ctx, self.ctx.settings
        if not s.enable_video:
            raise OpenRouterError("video generation is disabled (ENABLE_VIDEO=false)")
        frame = first_frame if ctx.caps.get("video_first_frame") else None
        ctx.budget.check("video")
        result = ctx.client.generate_video(
            s.video_model, prompt.prompt, duration_s=s.video_clip_seconds, resolution=s.video_resolution,
            aspect_ratio=s.aspect_ratio, first_frame=frame, task=ctx.task)
        ctx.budget.record(ctx.task, agent, "video", result.usage)
        path = _next_path(ctx, "video", name, shot_id, "video", ".mp4")
        path.write_bytes(result.data)
        info = probe(path)                      # raises if the download is not readable media
        provenance, is_sample = _provenance(ctx, result.usage.model)
        metadata = dict(result.metadata, prompt=prompt.prompt, cost_usd=result.usage.cost_usd)
        if first_frame and not frame:
            metadata["limitation"] = "model does not accept a first-frame image; generated from text only"
        video = info["video"] or {}
        return ctx.store.add_asset(MediaAsset(
            project_id=ctx.project_id, parent_id=shot_id, kind="video", path=str(path), mime="video/mp4",
            width=video.get("width"), height=video.get("height"), duration_s=info["duration_s"],
            model=result.usage.model, prompt_id=prompt.id, provenance=provenance, is_sample=is_sample,
            metadata=metadata, status="generated"))


class SpeechService:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    def generate(self, segment: NarrationSegment, *, agent: str) -> MediaAsset:
        ctx, s = self.ctx, self.ctx.settings
        name, metadata = f"narration_{segment.index:03d}", {"text": segment.text, "voice": segment.voice}
        if not s.enable_tts and not ctx.client.is_mock:
            raw, media_type, model, sample_rate = sample_audio(segment.estimated_s), "audio/wav", "", None
            provenance, is_sample = "local placeholder tone (ENABLE_TTS=false; contains no speech)", True
        else:
            instructions = segment.instructions if ctx.caps.get("tts_instructions") else ""
            ctx.budget.check("tts")
            result = ctx.client.text_to_speech(s.tts_model, segment.text, voice=segment.voice,
                                               fmt=s.tts_format, speed=segment.speed,
                                               instructions=instructions, task=ctx.task)
            ctx.budget.record(ctx.task, agent, "tts", result.usage)
            raw, media_type, model = result.data, result.media_type, result.usage.model
            sample_rate = result.metadata.get("sample_rate")
            metadata.update(cost_usd=result.usage.cost_usd, instructions_sent=bool(instructions))
            provenance, is_sample = _provenance(ctx, model)
            if not is_sample:
                provenance += f", stock voice '{segment.voice}' (no voice cloning)"

        if "wav" in media_type:
            suffix = ".wav"
        elif "pcm" in media_type or media_type in ("audio/L16", "application/octet-stream"):
            rate = sample_rate or s.tts_pcm_sample_rate
            raw, suffix = pcm_to_wav(raw, rate), ".wav"
            metadata["pcm_sample_rate"] = rate
            metadata["pcm_sample_rate_source"] = "response header" if sample_rate else "TTS_PCM_SAMPLE_RATE"
        elif "mpeg" in media_type or "mp3" in media_type:
            suffix = ".mp3"
        else:
            raise OpenRouterError(f"text to speech returned unsupported content type '{media_type}'")
        path = _next_path(ctx, "audio", name, segment.id, "audio", suffix)
        path.write_bytes(raw)
        duration = audio_duration(path)         # raises if the bytes are not valid audio
        return ctx.store.add_asset(MediaAsset(
            project_id=ctx.project_id, parent_id=segment.id, kind="audio", path=str(path),
            mime="audio/wav" if suffix == ".wav" else "audio/mpeg", duration_s=duration, model=model,
            provenance=provenance, is_sample=is_sample, metadata=metadata, status="generated"))
