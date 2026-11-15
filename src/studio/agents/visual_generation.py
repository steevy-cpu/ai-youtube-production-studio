"""Visual Generation Agent: produces images, thumbnails and optional clips from approved prompts.

It selects the image or the video service, passes character references when the model
supports them, and records generation metadata. Thumbnail text is added locally with Pillow.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from studio.agents.base import Agent
from studio.agents.prompt_engineer import compose_image_prompt
from studio.schemas import AnimationPrompt, ImagePrompt, MediaAsset, Shot, StyleBible, ThumbnailBrief
from studio.services.generation import ImageService, VideoService
from studio.services.placeholders import load_font


def thumbnail_size(aspect_ratio: str) -> tuple[int, int]:
    a, b = (float(x) for x in aspect_ratio.split(":"))
    return 1280, int(round(1280 * b / a / 2) * 2)


def finish_thumbnail(source: Path, target: Path, size: tuple[int, int], text: str, font_path: str = "") -> None:
    """Crop/resize to the exact thumbnail size and add the title text in a banner at the bottom.

    The text is drawn by software (never by the image model) on a dark translucent band, so it
    is spelled correctly and stays readable whatever the artwork behind it looks like.
    """
    with Image.open(source) as img:
        thumb = ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)
    text = " ".join(text.split()).upper()
    if text:
        measure = ImageDraw.Draw(thumb)
        font_size = size[1] // 6
        while True:                           # shrink until the text fits 90% of the width
            font = load_font(font_size, font_path)
            stroke = max(font_size // 14, 2)
            box = measure.textbbox((0, 0), text, font=font, stroke_width=stroke)
            if box[2] - box[0] <= size[0] * 0.9 or font_size <= 24:
                break
            font_size -= 6
        text_w, text_h = box[2] - box[0], box[3] - box[1]
        pad = size[1] // 28
        band_top = size[1] - text_h - 2 * pad
        band = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(band).rectangle([0, band_top, size[0], size[1]], fill=(0, 0, 0, 150))
        thumb = Image.alpha_composite(thumb.convert("RGBA"), band).convert("RGB")
        ImageDraw.Draw(thumb).text(((size[0] - text_w) / 2 - box[0], band_top + pad - box[1]), text, font=font,
                                   fill=(255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0))
    thumb.save(target, "PNG")


class VisualGeneration(Agent):
    name = "visual_generation"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.images, self.videos = ImageService(ctx), VideoService(ctx)

    def _references(self, style: StyleBible) -> list[Path]:
        refs = [a for a in self.ctx.store.assets(self.ctx.project_id, "reference")
                if a.id in style.reference_asset_ids]
        return [Path(a.path) for a in refs]

    def shot_image(self, shot: Shot, prompt: ImagePrompt, style: StyleBible, feedback: str = "") -> MediaAsset:
        asset = self.images.generate(
            compose_image_prompt(prompt, style, feedback), kind="image", parent_id=shot.id,
            prompt_id=prompt.id, name=f"shot_{shot.index:03d}", agent=self.name,
            label=f"Shot {shot.index}: {shot.visual_description}", references=self._references(style))
        self.note("generate_image", f"shot {shot.index} -> {Path(asset.path).name}"
                                    f"{' (regeneration with feedback)' if feedback else ''}")
        return asset

    def reference_image(self, prompt: ImagePrompt, style: StyleBible, name: str, feedback: str = "") -> MediaAsset:
        asset = self.images.generate(compose_image_prompt(prompt, style, feedback), kind="reference",
                                     parent_id=style.id, prompt_id=prompt.id, name=name, agent=self.name,
                                     label=f"Character reference: {name}")
        self.note("generate_reference", f"{name} -> {Path(asset.path).name}")
        return asset

    def thumbnail(self, prompt: ImagePrompt, brief: ThumbnailBrief, style: StyleBible,
                  feedback: str = "") -> MediaAsset:
        """Generate a thumbnail candidate, then fit it to 1280x720 and add the text locally."""
        asset = self.images.generate(
            compose_image_prompt(prompt, style, feedback), kind="thumbnail", parent_id=brief.id,
            prompt_id=prompt.id, name="thumbnail_candidate", agent=self.name,
            label=f"Thumbnail: {brief.concept}", references=self._references(style))
        size = thumbnail_size(self.ctx.settings.aspect_ratio)
        raw = Path(asset.path)
        raw_copy = raw.with_name(raw.stem + "_raw.png")
        raw.replace(raw_copy)
        finish_thumbnail(raw_copy, raw, size, brief.overlay_text, self.ctx.settings.font_path)
        asset.width, asset.height = size
        asset.metadata.update(raw_path=str(raw_copy), overlay_text=brief.overlay_text,
                              text_rendering="added locally with Pillow, not by the image model")
        self.ctx.store.update_asset(asset)
        self.note("generate_thumbnail", f"candidate {Path(asset.path).name}"
                                        f"{' (regeneration with feedback)' if feedback else ''}")
        return asset

    def clip(self, shot: Shot, prompt: AnimationPrompt, first_frame: Path | None) -> MediaAsset:
        asset = self.videos.generate(prompt, shot_id=shot.id, name=f"shot_{shot.index:03d}",
                                     agent=self.name, first_frame=first_frame)
        self.note("generate_clip", f"shot {shot.index} -> {Path(asset.path).name}")
        return asset
