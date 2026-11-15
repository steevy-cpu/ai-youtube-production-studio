"""Final QA Agent: deterministic checks on the rendered video plus an LLM continuity review."""
from __future__ import annotations

from pathlib import Path

from studio.agents.base import Agent, Verdict
from studio.guards import final_video_guard
from studio.rendering.ffmpeg import FFmpegError, detect_silence
from studio.schemas import Criterion, EditingTimeline, EvaluationResult, Script, Storyboard
from studio.utils import split_sentences

MAX_FINAL_SILENCE_S = 6.0


class FinalQA(Agent):
    name = "final_qa"
    system = ("You are the Final QA reviewer. You check that the finished video tells one coherent "
              "story from the first rank to the last.")

    def review(self, video: Path, timeline: EditingTimeline, storyboard: Storyboard,
               script: Script) -> list[EvaluationResult]:
        ctx = self.ctx
        guard = final_video_guard(video, timeline, storyboard, ctx.settings)
        guard.project_id = ctx.project_id
        results = [ctx.store.add_evaluation(guard)]
        self.note("final_video_guard", guard.explanation, "ok" if guard.passed else "failed")

        if guard.metrics.get("audio"):                       # audio/video problem scan
            try:
                silences = detect_silence(video, min_s=2.0)
                longest = max((d for _, d in silences), default=0.0)
                ok, note = longest <= MAX_FINAL_SILENCE_S, f"longest silent stretch is {longest:.1f}s"
            except FFmpegError as exc:
                ok, note, longest = False, f"audio scan failed: {exc}", None
            results.append(ctx.store.add_evaluation(EvaluationResult(
                project_id=ctx.project_id, parent_id=timeline.id, target_kind="final_video",
                evaluator="final_av_check", passed=ok, status="passed" if ok else "failed",
                criteria=[Criterion(name="no long silent stretches", passed=ok, note=note)],
                explanation=note, issues=[] if ok else [note], metrics={"longest_silence_s": longest})))

        earlier = ctx.store.evaluations(ctx.project_id, "final_qa_continuity", script.id)
        if earlier:      # same script already reviewed: retrying Final QA costs no model request
            self.note("continuity_review", "reused the stored review of this script")
            return results + earlier[-1:]

        outline = "\n".join(
            f"Chapter {c.index} \"{c.title}\": starts \"{split_sentences(c.narration)[0]}\" ... ends "
            f"\"{split_sentences(c.narration)[-1]}\"" for c in script.chapters)
        scenes = "; ".join(f"{s.index}. {s.title}" for s in storyboard.scenes)
        user = (f"Video: {script.title}\n<data>\n{outline}\n\nScenes in order: {scenes}\n</data>\n\n"
                "Review narrative continuity: does each chapter lead into the next, is the order of "
                "ranks logical, does the opening hook the viewer and does the ending close the story? "
                "score 0.0-1.0; approved=false only for a real continuity break. List issues and fixes.")
        verdict = self.ask("qa.continuity", Verdict, self.system, user)
        results.append(self.evaluation(verdict, target_id=script.id, target_kind="script",
                                       threshold=0.6, evaluator="final_qa_continuity"))
        return results
