"""Prompt Judge and Visual Judge: LLM/VLM semantic evaluation with structured verdicts."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from studio.agents.base import Agent, Verdict
from studio.agents.prompt_engineer import style_text
from studio.schemas import EvaluationResult, ImagePrompt, Scene, Shot, StyleBible, ThumbnailBrief

SCORING = ("score is 0.0-1.0 (1.0 = perfect). approved=true only if there is no serious problem. "
           "List each concrete problem in `issues` and a concrete fix for it in `suggested_changes`.")


class ShotVerdict(Verdict):
    shot_index: int


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
                "Return one verdict per shot_index. Criteria: (1) every required visual detail is in the "
                "prompt, (2) character and style requirements are respected, (3) nothing contradicts the "
                f"scene or the Style Bible, (4) the prompt does not request text or real brands. {SCORING}")
        result = self.ask("judge.prompts", SceneVerdicts, self.system, user,
                          context={"shot_indexes": [shot.index for shot, _ in items]})
        by_index = {v.shot_index: v for v in result.verdicts}
        evaluations = {}
        for shot, prompt in items:
            verdict = by_index.get(shot.index) or Verdict(
                approved=False, score=0.0, issues=["the judge returned no verdict for this prompt"])
            evaluations[shot.id] = self.evaluation(verdict, target_id=prompt.id, target_kind="prompt",
                                                   threshold=self.ctx.settings.prompt_judge_threshold)
        return evaluations

    def judge_thumbnail_prompt(self, prompt: ImagePrompt, brief: ThumbnailBrief, style: StyleBible
                               ) -> EvaluationResult:
        user = (f"Style Bible:\n{style_text(style)}\n\nSEO thumbnail brief:\n{brief.model_dump_json(indent=1)}"
                f"\n\nImage prompt to check:\n<data>\n{prompt.prompt}\ncomposition: {prompt.composition}\n"
                f"negative: {prompt.negative_prompt}\n</data>\n\n"
                "Does the prompt follow the brief (concept, subject, composition, colours, emotion, style, "
                "things to avoid) and keep the image free of rendered text? " + SCORING)
        verdict = self.ask("judge.thumbnail_prompt", Verdict, self.system, user)
        return self.evaluation(verdict, target_id=prompt.id, target_kind="thumbnail_prompt",
                               threshold=self.ctx.settings.prompt_judge_threshold)


class VisualJudge(Agent):
    name = "visual_judge"
    model_setting = "vision_model"
    system = ("You are the Visual Judge. You look at a generated image and compare it with what was "
              "requested. Judge only what is actually visible.")

    def judge_image(self, image: Path, prompt: ImagePrompt, style: StyleBible, *, asset_id: str,
                    purpose: str, threshold: float, extra: str = "") -> EvaluationResult:
        user = (f"Requested image ({purpose}):\n<data>\n{prompt.prompt}\ncomposition: {prompt.composition}\n"
                f"must not appear: {prompt.negative_prompt}\n</data>\n\nStyle Bible:\n{style_text(style)}\n\n"
                f"{extra}\nEvaluate the attached image. Criteria: (1) content matches the request, "
                "(2) characters match their Style Bible description, (3) composition and 2D cartoon style, "
                "(4) no major visual errors (distorted anatomy, garbled text, artefacts, unwanted logos or "
                f"real people). {SCORING}")
        verdict = self.ask("judge.image", Verdict, self.system, user, images=[image],
                           context={"purpose": purpose})
        return self.evaluation(verdict, target_id=asset_id, target_kind=purpose, threshold=threshold)
