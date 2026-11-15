"""Thin, safe wrappers around the ffmpeg / ffprobe executables.

Commands are always passed as argument lists (never through a shell).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any


class FFmpegError(Exception):
    pass


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"{name} is not installed or not on PATH (required for rendering and media checks)")
    return path


def run_ffmpeg(args: list[str], timeout: float = 3600) -> str:
    """Run ffmpeg with the given arguments; returns stderr (where ffmpeg reports analysis output)."""
    command = [_binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", *args]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise FFmpegError(f"ffmpeg timed out after {timeout:.0f}s") from None
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed (exit {result.returncode}): {result.stderr.strip()[-600:]}")
    return result.stderr


def probe(path: Path | str) -> dict[str, Any]:
    """Inspect a media file. Returns duration, format and the first video/audio stream details."""
    path = Path(path)
    if not path.is_file():
        raise FFmpegError(f"file does not exist: {path}")
    result = subprocess.run(
        [_binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)], capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {path.name}: {result.stderr.strip()[-300:]}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise FFmpegError(f"ffprobe returned unreadable output for {path.name}") from None
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    def number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    duration = number(fmt.get("duration")) or number((video or audio or {}).get("duration"))
    return {"format": fmt.get("format_name", ""), "duration_s": duration,
            "size_bytes": int(fmt.get("size", 0) or 0),
            "video": {"codec": video.get("codec_name"), "width": video.get("width"),
                      "height": video.get("height")} if video else None,
            "audio": {"codec": audio.get("codec_name"), "sample_rate": audio.get("sample_rate"),
                      "channels": audio.get("channels")} if audio else None}


def audio_duration(path: Path | str) -> float:
    """Measured audio length in seconds (ffprobe, or the wave module for WAV without ffprobe)."""
    path = Path(path)
    if shutil.which("ffprobe"):
        info = probe(path)
        if info["audio"] is None or not info["duration_s"]:
            raise FFmpegError(f"{path.name} has no readable audio stream")
        return float(info["duration_s"])
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                return wav.getnframes() / float(wav.getframerate())
        except (wave.Error, EOFError) as exc:
            raise FFmpegError(f"{path.name} is not a valid WAV file: {exc}") from None
    raise FFmpegError("ffprobe is required to measure non-WAV audio")


def detect_silence(path: Path | str, noise_db: int = -45, min_s: float = 1.0) -> list[tuple[float, float]]:
    """Silent stretches as (start, duration) pairs, using ffmpeg's silencedetect filter."""
    stderr = run_ffmpeg(["-i", str(path), "-af", f"silencedetect=noise={noise_db}dB:d={min_s}",
                         "-f", "null", "-"], timeout=600)
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", stderr)]
    durations = [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", stderr)]
    return list(zip(starts, durations))
