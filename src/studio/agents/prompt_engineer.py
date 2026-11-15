"""Prompt Engineer Agent: turns approved scenes and the thumbnail brief into generation prompts."""
from __future__ import annotations

from pydantic import BaseModel, Field

from studio.agents.base import Agent
from studio.schemas import AnimationPrompt, Character, ImagePrompt, Scene, Shot, StyleBible, ThumbnailBrief


class PromptDraft(BaseModel):
    prompt: str = Field(min_length=15)
    negative_prompt: str = ""
    composition: str = ""
    lighting: str = ""
    characters: list[str] = []


class ShotPromptDraft(PromptDraft):
    shot_index: int
    animation_prompt: str = ""


class ScenePrompts(BaseModel):
    prompts: list[ShotPromptDraft] = Field(min_length=1)


NO_TEXT_RULE = ("IMPORTANT: the picture itself must contain no text of any kind: no words, letters, numbers, "
                "captions, titles, labels or logos, even if a correction above mentions text. The title "
                "is added afterwards by software in a banner along the bottom edge, so keep the bottom "
                "sixth of the picture free of faces and important details.")


def style_text(style: StyleBible) -> str:
    characters = "\n".join(f"- {c.name}: {c.appearance}; wearing {c.clothing}" for c in style.characters)
    return (f"Art style: {style.art_style}\nColour palette: {', '.join(style.color_palette)}\n"
            f"Characters (must always look exactly like this):\n{characters}\n"
            f"Environments: {style.environment_guidelines}\nFraming: {style.framing_guidelines}\n"
            f"Never show: {style.negative_prompt}")


def compose_image_prompt(prompt: ImagePrompt, style: StyleBible, feedback: str = "") -> str:
    """The exact text sent to the image model: prompt + Style Bible + exclusions (+ judge feedback)."""
    named = [c for c in style.characters if c.name in prompt.characters] or style.characters[:1]
    parts = [prompt.prompt.strip()]
    if prompt.composition:
        parts.append(f"Composition: {prompt.composition}")
    if prompt.lighting:
        parts.append(f"Lighting: {prompt.lighting}")
    parts.append("Characters: " + "; ".join(f"{c.name} - {c.appearance}, wearing {c.clothing}" for c in named))
    parts.append(f"Style: {style.art_style}. Colour palette: {', '.join(style.color_palette)}.")
    avoid = ", ".join(x for x in (prompt.negative_prompt, style.negative_prompt) if x)
    parts.append(f"Do not include: {avoid}.")
    if feedback:
        parts.append(f"Corrections required compared with the previous attempt: {feedback}")
    if prompt.purpose == "thumbnail":      # placed last so it also overrides anything said above
        parts.append(NO_TEXT_RULE)
    return "\n".join(parts)


def reference_prompt(character: Character, style: StyleBible) -> ImagePrompt:
    """Deterministic character reference sheet prompt derived from the Style Bible."""
    return ImagePrompt(
        purpose="reference", parent_id=style.id, characters=[character.name],
        prompt=(f"Character reference of {character.name}: {character.appearance}, wearing "
                f"{character.clothing}. Full body, neutral friendly pose, facing the viewer."),
        negative_prompt="other characters, text, labels, busy background",
        composition="single character centred on a plain light background", lighting="flat even lighting",
        status="approved")


class PromptEngineer(Agent):
    name = "prompt_engineer"
    system = ("You are the Prompt Engineer. You write precise prompts for an image generation model "
              "(and optional short animation prompts). Describe subject, action, setting, composition, "
              "framing and lighting concretely. Never ask the model to render text.")

    def _image_prompt(self, draft: PromptDraft, parent_id: str, purpose: str) -> ImagePrompt:
        return ImagePrompt(parent_id=parent_id, purpose=purpose, prompt=draft.prompt,
                           negative_prompt=draft.negative_prompt, composition=draft.composition,
                           lighting=draft.lighting, characters=draft.characters,
                           aspect_ratio=self.ctx.settings.aspect_ratio)

    def scene_prompts(self, scene: Scene, shots: list[Shot], style: StyleBible
                      ) -> dict[str, tuple[ImagePrompt, AnimationPrompt | None]]:
        """Prompts for the given shots of one scene, keyed by shot id."""
        listing = "\n".join(f"shot_index {s.index} ({s.framing}, {s.media_type}): {s.visual_description}\n"
                            f"   narration: {s.narration}" for s in shots)
        user = (f"Style Bible:\n{style_text(style)}\n\nScene \"{scene.title}\": {scene.summary}\n"
                f"<data>\n{listing}\n</data>\n\n"
                "Write one image prompt per shot (same shot_index). `prompt` describes the picture; "
                "`composition` the layout and framing; `lighting` the light; `characters` the names of "
                "Style Bible characters visible; `negative_prompt` what must not appear in this shot. "
                "For shots marked animation also write `animation_prompt` describing a few seconds of "
                "simple motion starting from that image; leave it empty for other shots.")
        result = self.ask("prompt.scene", ScenePrompts, self.system, user, context={
            "shots": [{"index": s.index, "visual_description": s.visual_description,
                       "media_type": s.media_type} for s in shots]})
        by_index = {d.shot_index: d for d in result.prompts}
        prompts = {}
        for shot in shots:
            draft = by_index.get(shot.index)
            if draft is None:
                continue                      # the Prompt Guard / orchestrator will report the gap
            animation = None
            if shot.media_type == "animation" and draft.animation_prompt.strip():
                animation = AnimationPrompt(parent_id=shot.id, prompt=draft.animation_prompt.strip(),
                                            duration_s=self.ctx.settings.video_clip_seconds)
            prompts[shot.id] = (self._image_prompt(draft, shot.id, "shot"), animation)
        return prompts

    def revise(self, shot: Shot, current: ImagePrompt | None, style: StyleBible, feedback: str) -> ImagePrompt:
        user = (f"Style Bible:\n{style_text(style)}\n\nShot {shot.index} ({shot.framing}): "
                f"{shot.visual_description}\nNarration: {shot.narration}\n\n"
                f"Current prompt:\n<data>\n{current.prompt if current else '(missing)'}\n</data>\n"
                f"A reviewer rejected it:\n{feedback}\n\nWrite a corrected image prompt for this shot.")
        draft = self.ask("prompt.revise", PromptDraft, self.system, user, context={
            "shot_index": shot.index, "visual_description": shot.visual_description})
        return self._image_prompt(draft, shot.id, "shot")

    def thumbnail_prompt(self, brief: ThumbnailBrief, style: StyleBible, feedback: str = "") -> ImagePrompt:
        user = (f"Style Bible:\n{style_text(style)}\n\nThumbnail brief from the SEO agent:\n"
                f"- concept: {brief.concept}\n- main subject: {brief.main_subject}\n"
                f"- composition: {brief.composition}\n- colours: {', '.join(brief.color_palette)}\n"
                f"- emotion: {brief.emotion}\n- style: {brief.style_requirements}\n"
                f"- avoid: {', '.join(brief.avoid) or 'n/a'}\n"
                f"- overlay text added later by software: \"{brief.overlay_text}\"\n\n"
                "Write the image prompt for this YouTube thumbnail. The image itself must contain NO text "
                "or letters and your prompt must not mention the overlay text or ask for space 'for text': "
                "software adds the title in a banner along the bottom edge, so simply keep faces and key "
                "details out of the bottom sixth. One clear focal subject readable at small size, strong "
                "contrast.")
        if feedback:
            user += f"\n\nYour previous thumbnail prompt was rejected:\n{feedback}"
        draft = self.ask("prompt.thumbnail", PromptDraft, self.system, user,
                         context={"concept": brief.concept})
        prompt = self._image_prompt(draft, brief.id, "thumbnail")
        if "text" not in prompt.negative_prompt.lower():
            prompt.negative_prompt = (prompt.negative_prompt + ", text, letters, watermarks").strip(", ")
        return prompt
