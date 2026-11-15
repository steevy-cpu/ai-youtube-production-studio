"""Locally generated, clearly labelled placeholder media.

Used by dry-run mode and when a generation feature is switched off. These are never
presented as AI-generated output: every asset made here is stored with is_sample=True.
"""
from __future__ import annotations

import io
import math
import struct
import tempfile
import textwrap
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SAMPLE_LABEL = "SAMPLE PLACEHOLDER - NOT AI-GENERATED"


def load_font(size: int, font_path: str = "") -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def sample_image(text: str, width: int = 1280, height: int = 720, seed: int = 0) -> bytes:
    """A flat-colour card with the placeholder label and a short description."""
    hues = [(52, 73, 94), (22, 96, 110), (96, 62, 110), (110, 78, 38), (40, 100, 70), (110, 48, 60)]
    base = hues[seed % len(hues)]
    img = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(img)
    for i in range(0, width, max(width // 16, 1)):          # simple stripes so pan/zoom is visible
        draw.line([(i, 0), (i - height // 2, height)], fill=tuple(min(c + 14, 255) for c in base), width=3)
    draw.rectangle([0, 0, width, height // 9], fill=(20, 20, 20))
    draw.text((width // 40, height // 40), SAMPLE_LABEL, font=load_font(height // 22), fill=(255, 210, 80))
    body = load_font(height // 16)
    wrapped = textwrap.wrap(text, width=38)[:6]
    draw.multiline_text((width // 12, height // 4), "\n".join(wrapped), font=body, fill=(245, 245, 245),
                        spacing=height // 50)
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    return buffer.getvalue()


def sample_audio(seconds: float, sample_rate: int = 22050) -> bytes:
    """A quiet pulsing tone as WAV bytes (stands in for narration; contains no speech)."""
    frames = max(int(seconds * sample_rate), sample_rate // 10)
    samples = bytearray()
    for n in range(frames):
        t = n / sample_rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)        # 2 Hz pulse, like syllables
        value = 0.06 * envelope * math.sin(2 * math.pi * 196.0 * t)
        samples += struct.pack("<h", int(value * 32767))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(samples))
    return buffer.getvalue()


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit little-endian mono PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def sample_video(text: str, seconds: float, width: int = 640, height: int = 360) -> bytes:
    """A short labelled MP4 built with FFmpeg from a placeholder card."""
    from studio.rendering.ffmpeg import run_ffmpeg
    with tempfile.TemporaryDirectory() as tmp:
        card, out = Path(tmp) / "card.png", Path(tmp) / "clip.mp4"
        card.write_bytes(sample_image(text, width, height, seed=3))
        run_ffmpeg(["-loop", "1", "-framerate", "15", "-i", str(card), "-t", f"{seconds:.2f}",
                    "-vf", "hue=h=t*20", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)])
        return out.read_bytes()

