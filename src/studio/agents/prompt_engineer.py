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
                "captions, titles, labels or logos, even if a correction above mentions text. Screens and "
                "holograms show only abstract shapes and glows, never lines of code or glyphs. Draw ONE "
                "scene that fills the whole frame edge to edge: no panels, split screens, borders, "
                "frames or empty margins, and each character appears only once. The title "
                "is added afterwards by software in a banner along the bottom edge, so keep the bottom "
                "sixth of the picture free of faces and important details.")


# One statement of who decides what on a thumbnail, shown word for word to the Prompt Engineer and to
# every reviewer. Without it each agent resolved brief-versus-Style-Bible conflicts differently and
# the review loop could never converge.
THUMBNAIL_RULES = (
    "Rules that settle every conflict on a thumbnail (they apply to authors and reviewers alike):\n"
    "1. The Style Bible decides the art style, the colour palette and how characters look. If the "
    "brief names other colours, the Style Bible palette wins and following it is correct, not an "
    "error. Colours may be written as plain names (\"mustard yellow\") or hex codes; a plain name "
    "that matches a palette colour is the same colour. One or two accent colours from the brief are "
    "acceptable when the palette still dominates.\n"
    "2. The brief decides the idea: concept, main subject, composition and emotion. A thumbnail may "
    "be more dramatic than a normal scene (bigger expression, bolder lighting, a symbolic "
    "background) as long as the art style and the character stay on-model.\n"
    "3. The artwork contains NO text. The overlay text in the brief is drawn afterwards by software "
    "in a banner along the bottom edge. The prompt must not ask for that text or mention it, a "
    "negative prompt that forbids text is correct, and nobody may ask for text to be added.\n"
    "4. Faces and key details stay out of the bottom sixth of the frame.\n"
    "5. A thumbnail is ONE full-frame scene with one large focal subject. If the brief describes a "
    "sequence, several stages, a timeline or a chart, the correct picture is the single most "
    "striking moment, with the rest at most hinted at in the background. Panels, split screens, "
    "borders, margins, or the same character drawn several times are errors, and showing a single "
    "moment instead of every stage is NOT a deviation from the brief.\n"
    "6. Screens, holograms, books and signs show only abstract shapes, bars and icons: nothing that "
    "looks like writing, code, digits or labels.")


# The same idea for normal shots. In the first live runs the Scene Director asked for labels, digits
# and off-palette colours, the Style Bible forbade them, and the Prompt Judge rejected the prompt
# whichever side the Prompt Engineer took. Everyone now reads one rule set.
SHOT_RULES = (
    "Rules that settle every conflict on a shot image (they apply to the Scene Director, the Prompt "
    "Engineer and every reviewer alike):\n"
    "1. The Style Bible decides the art style and how characters look. A character keeps the Style "
    "Bible outfit in every shot; a costume idea (a wizard, a king, a detective) becomes a held prop, "
    "an accessory or a glow around the character, never a replacement of the outfit.\n"
    "2. The Style Bible palette dominates the picture. One accent colour outside it is acceptable "
    "when the idea needs a signal (warning red, alarm, danger) and it stays a small part of the "
    "frame. Using a palette colour instead of a colour the shot names is equally correct. Neither is "
    "an error. Colours are written as plain names, never hex codes.\n"
    "3. The picture contains no writing: no words, letters, digits, percentages, code, labels, "
    "button captions, currency or maths symbols. When the required visual names any of these, the "
    "correct picture replaces it with a wordless stand-in that carries the same idea (icons, plain "
    "shapes, bars, gauges, tally marks, coins, glowing blocks). That replacement is correct, it is "
    "never a missing detail, a negative prompt that forbids text is correct, and nobody may ask for "
    "writing to be added.\n"
    "4. Software appends the full Style Bible description of every character listed in `characters`, "
    "plus the art style and the palette, to the prompt before it reaches the image model. A prompt "
    "that names the character is therefore complete; not repeating hair, face or clothing is not a "
    "problem.\n"
    "5. A shot is ONE still moment. A change over time (an expression that shifts, before and "
    "after, a sequence) is shown as its single clearest moment.\n"
    "6. What must survive is the idea of the required visual: its subject, action, setting and the "
    "key props the rules allow.")


# How a generated picture is reviewed. Live runs showed reviewers treating the prompt as a checklist:
# 57% of shot images were rejected, mostly for "five fingers instead of four", and three good
# thumbnails were rejected for a laptop that was open instead of closed. An image model cannot be
# steered that precisely, so a retry changed nothing and the loop ran out of attempts.
IMAGE_REVIEW_RULES = (
    "How to review a generated picture (these rules decide between approve and reject):\n"
    "A. Judge what a viewer notices at a glance, not whether every phrase of the prompt was "
    "obeyed. The prompt is a means to get the picture, never a checklist.\n"
    "B. SERIOUS problems, the only reasons to reject: the main subject or its action is missing or "
    "wrong; a recurring character is clearly off-model (different hair colour, face, build or main "
    "outfit); the art style is wrong (for example photorealistic or 3D instead of flat 2D); the "
    "picture contains readable writing or a real logo; a defect that jumps out (extra or missing "
    "limbs, a melted or duplicated face, two copies of the same character, garbled artefacts); "
    "panels, borders or split screens.\n"
    "C. NOT serious, report as notes at most and still approve: the number of fingers on a cartoon "
    "hand; exact camera angle, perspective type, crop or centring; a prop that is open instead of "
    "closed, bigger, smaller or elsewhere than asked; an expression or pose that is close but not "
    "exact; an inexact shade or a missing accent colour; small background details; soft gradients; "
    "tiny abstract marks that are not readable writing.\n"
    "D. Score the picture as a whole. A picture with only notes scores 0.8 or higher. Every "
    "suggested change must be something an image model can act on in one sentence.")


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
                f"<data>\n{listing}\n</data>\n\n{SHOT_RULES}\n\n"
                "Write one image prompt per shot (same shot_index). Where a shot description asks for "
                "something the rules forbid, apply the rules (stand-in, prop, palette) instead of "
                "copying it. `prompt` describes the picture; "
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
                f"A reviewer rejected it:\n{feedback}\n\n{SHOT_RULES}\n\n"
                "Write a corrected image prompt for this shot. Fix every point that is compatible with "
                "the rules above and ignore any request that breaks them (for example adding text or "
                "digits). The prompt and its negative prompt must not contradict each other.")
        draft = self.ask("prompt.revise", PromptDraft, self.system, user, context={
            "shot_index": shot.index, "visual_description": shot.visual_description})
        return self._image_prompt(draft, shot.id, "shot")

    def thumbnail_prompt(self, brief: ThumbnailBrief, style: StyleBible, feedback: str = "") -> ImagePrompt:
        user = (f"Style Bible:\n{style_text(style)}\n\nThumbnail brief from the SEO agent:\n"
                f"- concept: {brief.concept}\n- main subject: {brief.main_subject}\n"
                f"- composition: {brief.composition}\n- colours: {', '.join(brief.color_palette)}\n"
                f"- emotion: {brief.emotion}\n- style: {brief.style_requirements}\n"
                f"- avoid: {', '.join(brief.avoid) or 'n/a'}\n\n{THUMBNAIL_RULES}\n\n"
                "Write the image prompt for this YouTube thumbnail. It is viewed at about 160 pixels "
                "wide, so build it around ONE large focal subject (the main character filling roughly "
                "half the frame, with a clear expression) and at most one supporting element that tells "
                "the idea. If the brief asks for a sequence, several stages or a chart, do not draw "
                "several small figures: show the single most striking moment and suggest the rest in "
                "the background. Describe every character exactly as the Style Bible does, including "
                "full clothing. Name colours in plain words taken from the Style Bible palette, never "
                "as hex codes (an image model may paint the code as text). Ask for strong contrast "
                "between subject and background. `characters` holds only the character names.\n"
                "Keep the prompt under about 120 words. An image model follows a short, clear picture "
                "far better than a long list of constraints: state the subject, the pose and expression, "
                "one supporting element, the setting in a few words and the colours. Do not specify "
                "exact angles, projections, crops or the position of every prop.")
        if feedback:
            user += (f"\n\nReviewers rejected the previous attempt (the prompt, or the image generated "
                     f"from it):\n{feedback}\n\nWrite a new prompt, not a patch of the old one. Fix the "
                     "serious points (wrong subject, off-model character, text, clutter, weak focal "
                     "subject) by SIMPLIFYING the picture: fewer elements, a simpler pose, a plainer "
                     "background. Do not add one more constraint per complaint; a longer prompt made the "
                     "previous attempts worse. Ignore any request that breaks the rules above (for "
                     "example adding text, or leaving the Style Bible palette).")
        draft = self.ask("prompt.thumbnail", PromptDraft, self.system, user,
                         context={"concept": brief.concept}, model=self.ctx.settings.thumbnail_llm)
        prompt = self._image_prompt(draft, brief.id, "thumbnail")
        if "text" not in prompt.negative_prompt.lower():
            prompt.negative_prompt = (prompt.negative_prompt + ", text, letters, watermarks").strip(", ")
        return prompt
