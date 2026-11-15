"""Director Agent: creative goals, production plan, Style Bible and approval decisions.

The Director only decides. Execution, state changes and retries belong to the engine.
"""
from __future__ import annotations

from pydantic import BaseModel

from studio.agents.base import Agent
from studio.schemas import (ComplianceReport, EvaluationResult, ProductionPlan, ProjectBrief, Script,
                            StyleBible, ThumbnailBrief)


class DirectorPlan(BaseModel):
    plan: ProductionPlan
    style_bible: StyleBible


class ChapterNote(BaseModel):
    chapter_index: int
    note: str


class DirectorDecision(BaseModel):
    approved: bool
    reasons: list[str] = []
    chapter_feedback: list[ChapterNote] = []


def _eval_lines(evaluations: list[EvaluationResult]) -> str:
    return "\n".join(f"- {e.evaluator}: {'PASS' if e.passed else 'FAIL'}"
                     f"{'' if e.score is None else f' (score {e.score:.2f})'}"
                     f"{' minor notes: ' if e.passed and e.issues else ' '}{'; '.join(e.issues[:3])}"
                     for e in evaluations) or "- none"


class Director(Agent):
    name = "director"
    model_setting = "director_model"
    system = ("You are the Director of the studio. You set creative direction, keep the production "
              "consistent and approve or reject work. Be decisive and specific.")

    def plan(self, brief: ProjectBrief, feedback: str = "") -> DirectorPlan:
        s = self.ctx.settings
        user = (
            f"Plan a {brief.target_duration_s:.0f}-second {brief.language} video titled \"{brief.title}\" "
            f"about the topic \"{brief.topic}\".\n"
            "Format: the viewer follows one original recurring character through every rank/level of the "
            "topic, from beginner to the top. Entertaining, educational storytelling.\n"
            f"- Use between {s.min_chapters} and {s.max_chapters} chapters, numbered from 1, in rank order. "
            "The first chapter must open with a hook; the last must give a satisfying ending.\n"
            f"- chapter target_seconds must add up to about {brief.target_duration_s:.0f}.\n"
            "- The style bible defines a consistent 2D cartoon look: original characters (appearance and "
            "clothing precise enough to redraw them identically), art style, colour palette, environment "
            "and camera framing guidelines, and a negative prompt listing what must never appear. "
            "Character designs must not include lettering, numbers or logos (image models render them "
            "inconsistently).")
        if feedback:
            user += f"\n\nYour previous plan was rejected:\n{feedback}"
        result = self.ask("director.plan", DirectorPlan, self.system, user, context={
            "topic": brief.topic, "title": brief.title, "target_seconds": brief.target_duration_s,
            "min_chapters": s.min_chapters, "max_chapters": s.max_chapters})
        result.plan.chapters = sorted(result.plan.chapters, key=lambda c: c.index)
        return result

    def review_script(self, script: Script, plan: ProductionPlan, guard: EvaluationResult) -> DirectorDecision:
        chapters = "\n\n".join(f"[Chapter {c.index}: {c.title}]\n{c.narration}" for c in script.chapters)
        user = (f"Production status:\n{self.ctx.store.progress_summary(self.ctx.project_id)}\n\n"
                f"Creative goals: {'; '.join(plan.creative_goals)}\nTone: {plan.tone}\n"
                f"Script Guard: {'passed' if guard.passed else 'failed'} ({guard.metrics.get('words')} words)\n\n"
                f"<data>\n{chapters}\n</data>\n\n"
                "Approve the script only if it has a strong hook, clear progression through the ranks, "
                "humour, educational value and a consistent tone. If you reject it, give chapter_feedback "
                "naming each chapter that must change and exactly what to change.")
        return self.ask("director.review_script", DirectorDecision, self.system, user)

    def final_approval(self, evaluations: list[EvaluationResult], compliance: ComplianceReport) -> DirectorDecision:
        user = (f"Production status:\n{self.ctx.store.progress_summary(self.ctx.project_id)}\n\n"
                f"Final quality checks:\n{_eval_lines(evaluations)}\n\n"
                f"Compliance: {'BLOCKED' if compliance.blocked else 'not blocked'}; items needing human "
                f"review: {len(compliance.human_review_required)}\n\n"
                "Decide whether the finished video is approved for export. Reject only for concrete, "
                "serious problems and state them in reasons.")
        return self.ask("director.final", DirectorDecision, self.system, user)

    def approve_thumbnail(self, brief: ThumbnailBrief, title: str,
                          evaluations: list[EvaluationResult]) -> DirectorDecision:
        user = (f"Video title: {title}\nThumbnail concept: {brief.concept}\nMain subject: "
                f"{brief.main_subject}\nOverlay text: {brief.overlay_text or '(none)'}\n\n"
                f"Reviews of the generated thumbnail:\n{_eval_lines(evaluations)}\n\n"
                "Give final approval of the thumbnail. Every review above already passed; notes listed "
                "after a PASS are minor observations, not failures. Reject only for a serious problem: "
                "the thumbnail misrepresents the video, is off-topic or breaks the channel style. Do not "
                "reject for small composition preferences or anything about the title text (software "
                "draws it in a fixed bottom banner). If you do reject, reasons must describe how the "
                "CONCEPT of the picture has to change; they go to the Prompt Engineer, who rewrites the "
                "thumbnail prompt.")
        return self.ask("director.thumbnail", DirectorDecision, self.system, user)
