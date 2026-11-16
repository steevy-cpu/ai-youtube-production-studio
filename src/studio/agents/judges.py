"""Prompt Judge and Visual Judge: LLM/VLM semantic evaluation with structured verdicts."""
from __future__ import annotations

from pathlib import Path

from typing import Literal

from pydantic import BaseModel

from studio.agents.base import Agent, Verdict
from studio.agents.prompt_engineer import IMAGE_REVIEW_RULES, SHOT_RULES, THUMBNAIL_RULES, style_text
from studio.schemas import EvaluationResult, ImagePrompt, Scene, Shot, StyleBible, ThumbnailBrief

SCORING = ("score is 0.0-1.0 (1.0 = perfect). approved=true only if there is no serious problem. "
           "List each concrete problem in `issues` and a concrete fix for it in `suggested_changes`.")


class ShotVerdict(Verdict):
    shot_index: int
    fault: Literal["prompt", "scene"] = "prompt"     # who has to fix a rejected prompt


class SceneVerdicts(BaseModel):
    verdicts: list[ShotVerdict]


class PromptJudge(Agent):
    name = "prompt_judge"
    system = ("You are the Prompt Judge. You check whether image prompts faithfully represent the "
              "scene they were written for and respect the Style Bible. You are strict but fair.")

    def judge_scene(self, scene: Scene, items: list[tuple[Shot, ImagePrompt]], style: StyleBible
                    ) -> dict[str, EvaluationResult]:
        """Evaluate the prompts of one scene. Returns {shot_id: evaluation}."""
        listing = "\n\n".join(
            f"shot_index {shot.index}\n  required visual: {shot.visual_description}\n  narration: "
            f"{shot.narration}\n  prompt: {prompt.prompt}\n  composition: {prompt.composition}\n"
            f"  characters: {', '.join(prompt.characters)}\n  negative: {prompt.negative_prompt}"
            for shot, prompt in items)
        user = (f"Style Bible:\n{style_text(style)}\n\nScene \"{scene.title}\"\n<data>\n{listing}\n</data>\n\n"
                f"{SHOT_RULES}\n\n"
                "Return one verdict per shot_index, judged under those rules. Serious problems (reject) "
                "are: the prompt misses the idea of the required visual (wrong subject, action or "
                "setting, or a key prop the rules allow is absent); it asks for writing of any kind or a "
                "real brand; the art style or a character contradicts the Style Bible; it describes "
                "several moments; the prompt and its negative prompt contradict each other. NOT serious "
                "(a note at most, and still approve): everything rules 2, 3 and 4 call correct, wording "
                "preferences, a minor prop or lighting detail, and a stand-in you would have chosen "
                "differently. Every suggested change must itself respect the rules.\n"
                "`fault` says who must fix a rejected prompt. \"prompt\": a better prompt for the same "
                "required visual would pass. \"scene\": the required visual itself is the problem (even "
                "with stand-ins it cannot be drawn as one still image under the rules, it is too vague "
                "to draw, or it does not fit the narration), so the Scene Director has to describe a "
                f"different picture. Use \"prompt\" for approved prompts. {SCORING}")
        result = self.ask("judge.prompts", SceneVerdicts, self.system, user,
                          context={"shot_indexes": [shot.index for shot, _ in items]})
        by_index = {v.shot_index: v for v in result.verdicts}
        evaluations = {}
        for shot, prompt in items:
            verdict = by_index.get(shot.index) or Verdict(
                approved=False, score=0.0, issues=["the judge returned no verdict for this prompt"])
            evaluations[shot.id] = self.evaluation(
                verdict, target_id=prompt.id, target_kind="prompt",
                threshold=self.ctx.settings.prompt_judge_threshold,
                metrics={"fault": getattr(verdict, "fault", "prompt")})
        return evaluations

    def judge_thumbnail_prompt(self, prompt: ImagePrompt, brief: ThumbnailBrief, style: StyleBible
                               ) -> EvaluationResult:
        user = (f"Style Bible:\n{style_text(style)}\n\nSEO thumbnail brief:\n- concept: {brief.concept}\n"
                f"- main subject: {brief.main_subject}\n- composition: {brief.composition}\n"
                f"- colours: {', '.join(brief.color_palette)}\n- emotion: {brief.emotion}\n"
                f"- style: {brief.style_requirements}\n- avoid: {', '.join(brief.avoid) or 'n/a'}\n\n"
                f"Image prompt to check:\n<data>\n{prompt.prompt}\ncomposition: {prompt.composition}\n"
                f"lighting: {prompt.lighting}\nnegative: {prompt.negative_prompt}\n</data>\n\n"
                f"{THUMBNAIL_RULES}\n\n"
                "Judge the prompt under those rules. Serious problems are: it asks for text, it misses "
                "the brief's concept or main subject, a character contradicts the Style Bible, the art "
                "style is wrong, it requests something on the avoid list, or it has no single clear "
                "focal subject that would read at small size. Wording preferences, a missing minor "
                "clothing detail, plain colour names instead of hex codes and brief colours replaced "
                "by Style Bible colours are NOT serious: mention them as notes at most and still "
                "approve. Every suggested change must itself respect the rules. " + SCORING)
        verdict = self.ask("judge.thumbnail_prompt", Verdict, self.system, user,
                           model=self.ctx.settings.thumbnail_llm)
        return self.evaluation(verdict, target_id=prompt.id, target_kind="thumbnail_prompt",
                               threshold=self.ctx.settings.prompt_judge_threshold)


class VisualJudge(Agent):
    name = "visual_judge"
    model_setting = "vision_model"
    system = ("You are the Visual Judge. You look at a generated image and compare it with what was "
              "requested. Judge only what is actually visible.")

    def judge_image(self, image: Path, prompt: ImagePrompt, style: StyleBible, *, asset_id: str,
                    purpose: str, threshold: float, extra: str = "") -> EvaluationResult:
        user = (f"Prompt that was given to the image model ({purpose}):\n<data>\n{prompt.prompt}\n"
                f"composition: {prompt.composition}\nmust not appear: {prompt.negative_prompt}\n</data>\n\n"
                f"Style Bible:\n{style_text(style)}\n\n{extra}\n\n{IMAGE_REVIEW_RULES}\n\n"
                "Evaluate the attached image under those review rules. Criteria: (1) the subject and "
                "its action, (2) characters on-model, (3) the Style Bible art style, (4) no defect that "
                "jumps out, no readable writing, logos or real people. approved=false only for a "
                f"problem rule B lists; put everything else in `criteria` notes, not in `issues`. {SCORING}")
        model = self.ctx.settings.thumbnail_vision if purpose == "thumbnail" else None
        verdict = self.ask("judge.image", Verdict, self.system, user, images=[image],
                           context={"purpose": purpose}, model=model)
        return self.evaluation(verdict, target_id=asset_id, target_kind=purpose, threshold=threshold)
