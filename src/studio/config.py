"""Typed settings loaded from the environment / .env file."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator


class ConfigError(Exception):
    """Invalid or missing configuration."""


class Mode(str, Enum):
    DRY_RUN = "dry_run"   # no OpenRouter calls, labelled sample assets
    PREVIEW = "preview"   # real calls for a short sample
    FULL = "full"         # complete configured production

    @property
    def live(self) -> bool:
        return self is not Mode.DRY_RUN


Switch = Literal["auto", "on", "off"]


class Settings(BaseModel):
    # OpenRouter
    openrouter_api_key: str = Field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_referer: str = ""
    app_title: str = ""

    # Models
    llm_model: str = "google/gemini-3-flash-preview"
    director_model: str = "google/gemini-3-flash-preview"
    vision_model: str = "google/gemini-3-flash-preview"
    image_model: str = "google/gemini-2.5-flash-image"
    video_model: str = "google/veo-3.1"
    tts_model: str = "google/gemini-3.8-flash-lite-tts"
    stt_model: str = ""
    llm_json_mode: Literal["json_object", "json_schema", "none"] = "json_object"

    # Video format
    format_template: str = "Your Life as Every Rank in {topic}"
    target_duration_s: float = Field(default=600, gt=0)
    duration_tolerance: float = Field(default=0.10, gt=0, lt=1)
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: int = Field(default=30, ge=5, le=60)
    aspect_ratio: str = "16:9"
    language: str = "English"
    words_per_minute: int = Field(default=150, ge=60, le=260)
    min_chapters: int = Field(default=5, ge=1)
    max_chapters: int = Field(default=10, ge=1)
    target_shot_seconds: float = Field(default=8, gt=1)
    min_shot_seconds: float = 1.0
    max_shot_seconds: float = 25.0
    captions_enabled: bool = True
    font_path: str = ""

    # Feature switches
    enable_images: bool = True
    enable_video: bool = False
    enable_tts: bool = True
    image_use_references: Switch = "auto"
    max_reference_images: int = Field(default=1, ge=0, le=4)
    max_video_clips: int = Field(default=3, ge=0)
    video_clip_seconds: int = Field(default=4, ge=1)
    video_resolution: str = "720p"

    # Voice
    tts_voice: str = "Kore"
    tts_format: Literal["mp3", "pcm"] = "pcm"
    tts_speed: float = Field(default=1.0, gt=0.25, le=4)
    tts_instructions: Switch = "auto"
    tts_pcm_sample_rate: int = 24000

    # Licensed music
    music_path: str = ""
    music_license: str = ""
    music_volume: float = Field(default=0.12, ge=0, le=1)

    # Judges and retries
    prompt_judge_threshold: float = Field(default=0.7, ge=0, le=1)
    visual_judge_threshold: float = Field(default=0.7, ge=0, le=1)
    thumbnail_threshold: float = Field(default=0.7, ge=0, le=1)
    max_script_attempts: int = Field(default=3, ge=1)
    max_storyboard_attempts: int = Field(default=3, ge=1)
    max_prompt_attempts: int = Field(default=3, ge=1)
    max_visual_attempts: int = Field(default=3, ge=1)
    max_audio_attempts: int = Field(default=2, ge=1)
    max_thumbnail_attempts: int = Field(default=3, ge=1)
    max_llm_repair_attempts: int = Field(default=2, ge=0)

    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    references_dir: Path = Path("data/references")

    # API behaviour
    api_timeout_s: float = Field(default=120, gt=0)
    api_max_retries: int = Field(default=3, ge=0, le=8)
    video_poll_interval_s: float = Field(default=15, gt=0)
    video_timeout_s: float = Field(default=900, gt=0)
    force_ipv4: bool = True      # avoids long stalls on networks with a broken IPv6 route

    # Cost control
    max_budget_usd: float = Field(default=5.0, ge=0)
    max_media_requests: int = Field(default=150, ge=0)

    # Preview / dry-run sizing
    preview_duration_s: float = Field(default=60, gt=0)
    preview_min_duration_s: float = Field(default=50, gt=0)   # accepted final length of a preview
    preview_max_duration_s: float = Field(default=80, gt=0)
    # Set by for_mode() only (never read from the environment): absolute accepted final duration.
    duration_window: tuple[float, float] | None = None
    preview_max_chapters: int = Field(default=2, ge=1)
    dry_run_width: int = 640
    dry_run_height: int = 360
    dry_run_fps: int = 15
    dry_run_demo_rejections: bool = True

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        try:
            a, b = (float(x) for x in self.aspect_ratio.split(":"))
        except ValueError as exc:
            raise ValueError("ASPECT_RATIO must look like 16:9") from exc
        for w, h, label in ((self.width, self.height, "WIDTH/HEIGHT"),
                            (self.dry_run_width, self.dry_run_height, "DRY_RUN_WIDTH/HEIGHT")):
            if abs(w / h - a / b) > 0.02:
                raise ValueError(f"{label} {w}x{h} does not match ASPECT_RATIO {self.aspect_ratio}")
            if w % 2 or h % 2:
                raise ValueError(f"{label} must be even numbers (H.264 requirement)")
        if not self.preview_min_duration_s <= self.preview_duration_s <= self.preview_max_duration_s:
            raise ValueError("PREVIEW_DURATION_S must lie between PREVIEW_MIN_DURATION_S and "
                             "PREVIEW_MAX_DURATION_S")
        if self.min_chapters > self.max_chapters:
            raise ValueError("MIN_CHAPTERS cannot exceed MAX_CHAPTERS")
        if "{topic}" not in self.format_template:
            raise ValueError("FORMAT_TEMPLATE must contain {topic}")
        if bool(self.music_path) != bool(self.music_license):
            raise ValueError("MUSIC_PATH and MUSIC_LICENSE must be set together "
                             "(music is only used with a recorded license)")
        return self

    # ---- derived values -------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"

    @property
    def target_words(self) -> int:
        return round(self.target_duration_s * self.words_per_minute / 60)

    @property
    def duration_bounds(self) -> tuple[float, float]:
        """Accepted final video duration (inclusive): the mode's window, else target +/- tolerance."""
        if self.duration_window:
            return self.duration_window
        return (self.target_duration_s * (1 - self.duration_tolerance),
                self.target_duration_s * (1 + self.duration_tolerance))

    def for_mode(self, mode: Mode) -> "Settings":
        """Settings actually used by a run in the given mode."""
        if mode is Mode.DRY_RUN:
            return self.model_copy(update={
                "width": self.dry_run_width, "height": self.dry_run_height, "fps": self.dry_run_fps})
        if mode is Mode.PREVIEW:
            chapters = min(self.preview_max_chapters, self.max_chapters)
            target = min(self.preview_duration_s, self.target_duration_s)
            # The preview window applies to the configured preview length only; full production
            # (and a preview shortened by a small TARGET_DURATION_S) keeps the percentage tolerance.
            window = (self.preview_min_duration_s, self.preview_max_duration_s) \
                if target == self.preview_duration_s else None
            return self.model_copy(update={
                "target_duration_s": target, "duration_window": window,
                "min_chapters": 1, "max_chapters": chapters,
                "max_video_clips": min(self.max_video_clips, 1)})
        return self

    def require_api_key(self) -> str:
        if not self.openrouter_api_key.strip():
            raise ConfigError("OPENROUTER_API_KEY is not set. Add it to .env to use live modes "
                              "(dry-run mode needs no key).")
        return self.openrouter_api_key.strip()


def load_settings(env_file: str | Path | None = ".env", **overrides) -> Settings:
    """Read settings from the process environment (plus an optional .env file)."""
    if env_file and Path(env_file).is_file():
        load_dotenv(env_file, override=False)
    values = {}
    for name in Settings.model_fields:
        if name == "duration_window":
            continue
        raw = os.environ.get(name.upper())
        if raw is not None and raw.strip() != "":
            values[name] = raw.strip()
    values.update({k: v for k, v in overrides.items() if v is not None})
    try:
        return Settings(**values)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']).upper() or 'settings'}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"Invalid configuration: {problems}") from exc
