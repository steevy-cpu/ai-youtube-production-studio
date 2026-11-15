"""Task handlers for planning, research, script and storyboard (Topic -> approved storyboard).

Every handler has the signature `handler(ctx, feedback) -> summary`. `feedback` holds the
RevisionRequests routed to this task by a downstream reviewer; handlers redo only the
artifacts those requests name and keep everything already approved.
"""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from studio.agents.director import Director
from studio.agents.research import Research
from studio.agents.scene_director import SceneDirector, retime
from studio.agents.story_writer import StoryWriter
from studio.guards import chapter_word_targets, chapters_off_target, script_guard, timeline_guard
from studio.orchestration.context import Ctx
from studio.orchestration.engine import Escalation, Rejection
from studio.schemas import (Chapter, EvaluationResult, ProductionPlan, ProjectBrief, ResearchBrief,
                            RevisionRequest, Script, Storyboard, StyleBible)
from studio.utils import split_sentences

T = TypeVar("T", bound=BaseModel)


def need(ctx: Ctx, kind: str, model: type[T], key: str = "") -> T:
    """Load a required upstream artifact or fail loudly."""
    record = ctx.store.load(ctx.project_id, kind, model, key)
    if record is None:
        raise RuntimeError(f"required artifact '{kind}' is missing")
    return record


def revision(target_id: str | None, kind: str, task: str, agent: str, by: str, issues: list[str],
             fixes: list[str] | None = None) -> RevisionRequest:
    return RevisionRequest(parent_id=target_id, target_kind=kind, target_task=task, responsible_agent=agent,
                           requested_by=by, issues=issues, suggested_changes=fixes or [], status="open")


def from_evaluation(evaluation: EvaluationResult, target_id: str, kind: str, task: str,
                    agent: str) -> RevisionRequest:
    return revision(target_id, kind, task, agent, evaluation.evaluator, evaluation.issues,
                    evaluation.suggested_changes)


def feedback_for(feedback: list[RevisionRequest], target_id: str) -> str:
    return "\n".join(r.as_feedback() for r in feedback if r.parent_id == target_id)


# ---- planning and research --------------------------------------------------------
def director_plan(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief = need(ctx, "brief", ProjectBrief)
    result = Director(ctx).plan(brief, "\n".join(r.as_feedback() for r in feedback))
    plan, style, s = result.plan, result.style_bible, ctx.settings
    count = len(plan.chapters)
    if not s.min_chapters <= count <= s.max_chapters:
        raise Rejection("director_plan", [revision(None, "plan", "director_plan", "director", "orchestrator", [
            f"the plan has {count} chapters; it must have between {s.min_chapters} and {s.max_chapters}"])])
    for number, chapter in enumerate(plan.chapters, start=1):
        chapter.index = number
    plan.status = style.status = "approved"
    style.reference_asset_ids, style.limitations = [], []
    plan.parent_id = style.parent_id = brief.id
    ctx.store.save_artifact(ctx.project_id, "plan", plan)
    ctx.store.save_artifact(ctx.project_id, "style_bible", style)
    ctx.store.remember(ctx.project_id, "creative_goals", plan.creative_goals)
    ctx.store.remember(ctx.project_id, "tone", plan.tone)
    ctx.store.remember(ctx.project_id, "art_style", style.art_style)
    ctx.store.add_decision(ctx.project_id, "director", "production_plan", "established",
                           f"{count} chapters; style: {style.art_style}")
    return f"{count} chapters planned, style bible with {len(style.characters)} character(s)"


def research(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, plan = need(ctx, "brief", ProjectBrief), need(ctx, "plan", ProductionPlan)
    sources_dir = brief.sources_dir or str(ctx.settings.references_dir)
    result = Research(ctx).brief(brief.topic, plan, sources_dir)
    result.parent_id, result.status = plan.id, "complete"
    ctx.store.save_artifact(ctx.project_id, "research", result)
    for note in result.notes:
        if "No local sources" in note:
            ctx.store.add_limitation(ctx.project_id, "Research used no verified sources; facts need human "
                                                     "checking.")
    verified = sum(c.support == "verified_source" for c in result.claims)
    return f"{len(result.claims)} claims ({verified} source-verified), {len(result.sources)} local sources"


# ---- script ---------------------------------------------------------------------------
def story_writing(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, plan = need(ctx, "brief", ProjectBrief), need(ctx, "plan", ProductionPlan)
    brief_research, style = need(ctx, "research", ResearchBrief), need(ctx, "style_bible", StyleBible)
    targets = chapter_word_targets(plan, ctx.settings)
    existing = ctx.store.load_all(ctx.project_id, "chapter", Chapter)
    general = "\n".join(r.as_feedback() for r in feedback if r.target_kind == "script")
    writer, chapters, written, tail = StoryWriter(ctx), [], 0, ""
    for planned in plan.chapters:
        current = existing.get(str(planned.index))
        notes = "\n".join(x for x in (feedback_for(feedback, current.id) if current else "", general) if x)
        if current is None or notes:
            chapter = writer.write_chapter(planned, targets[planned.index], plan, brief_research, style,
                                           brief.topic, tail, current, notes)
            chapter.parent_id = plan.id
            ctx.store.save_artifact(ctx.project_id, "chapter", chapter, key=str(planned.index))
            current, written = chapter, written + 1
        chapters.append(current)
        tail = split_sentences(current.narration)[-1]
    script = Script(parent_id=plan.id, title=brief.title, hook=split_sentences(chapters[0].narration)[0],
                    chapters=chapters)
    ctx.store.save_artifact(ctx.project_id, "script", script)
    return f"script v{script.version}: {script.words} words, {written} chapter(s) written this pass"


def run_script_guard(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    script, plan = need(ctx, "script", Script), need(ctx, "plan", ProductionPlan)
    result = script_guard(script, plan, ctx.settings)
    result.project_id = ctx.project_id
    ctx.store.add_evaluation(result)
    if result.passed:
        return f"passed: {script.words} words (~{result.metrics['estimated_duration_s']}s)"
    by_index = {c.index: c for c in script.chapters}
    requests = [revision(by_index[i].id, "chapter", "story_writing", "story_writer", "script_guard",
                         [f"chapter {i} has {actual} words but should have about {target}"],
                         [f"rewrite chapter {i} to about {target} words"])
                for i, (actual, target) in chapters_off_target(script, plan, ctx.settings).items()]
    if not requests:
        requests = [revision(script.id, "script", "story_writing", "story_writer", "script_guard",
                             result.issues, result.suggested_changes)]
    raise Rejection("story_writing", requests, result.explanation)


def director_script_approval(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    script, plan = need(ctx, "script", Script), need(ctx, "plan", ProductionPlan)
    guard = ctx.store.evaluations(ctx.project_id, "script_guard", script.id)[-1]
    decision = Director(ctx).review_script(script, plan, guard)
    ctx.store.add_decision(ctx.project_id, "director", "script", "approved" if decision.approved else
                           "rejected", "; ".join(decision.reasons))
    if decision.approved:
        script.status = "approved"
        ctx.store.update_artifact(script)
        return f"script v{script.version} approved by the Director"
    by_index = {c.index: c for c in script.chapters}
    requests = [revision(by_index[n.chapter_index].id, "chapter", "story_writing", "story_writer", "director",
                         [n.note]) for n in decision.chapter_feedback if n.chapter_index in by_index]
    if not requests:
        requests = [revision(script.id, "script", "story_writing", "story_writer", "director",
                             decision.reasons or ["the Director rejected the script without details"])]
    raise Rejection("story_writing", requests, "Director rejected the script")


# ---- storyboard ---------------------------------------------------------------------------
def scene_direction(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    script, style = need(ctx, "script", Script), need(ctx, "style_bible", StyleBible)
    previous = ctx.store.load(ctx.project_id, "storyboard", Storyboard)
    chapter_ids = {c.id for c in script.chapters}
    redo = {r.parent_id for r in feedback}
    kept = [scene for scene in (previous.scenes if previous else [])
            if scene.parent_id in chapter_ids and scene.parent_id not in redo]
    done = {scene.parent_id for scene in kept}
    agent, scenes, rebuilt = SceneDirector(ctx), list(kept), 0
    for chapter in script.chapters:
        if chapter.id not in done:
            scenes += agent.chapter_scenes(chapter, style, feedback_for(feedback, chapter.id))
            rebuilt += 1
    storyboard = retime(Storyboard(parent_id=script.id, scenes=scenes), ctx.settings)
    ctx.store.save_artifact(ctx.project_id, "storyboard", storyboard)
    return (f"storyboard v{storyboard.version}: {len(storyboard.scenes)} scenes, {len(storyboard.shots)} "
            f"shots, ~{storyboard.duration_s:.0f}s ({rebuilt} chapter(s) storyboarded this pass)")


def run_timeline_guard(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, script = need(ctx, "storyboard", Storyboard), need(ctx, "script", Script)
    result = timeline_guard(storyboard, script, ctx.settings)
    result.project_id = ctx.project_id
    ctx.store.add_evaluation(result)
    if result.passed:
        storyboard.status = "approved"
        ctx.store.update_artifact(storyboard)
        return f"passed: {result.metrics['shots']} shots, ~{result.metrics['estimated_duration_s']}s"
    by_index = {c.index: c for c in script.chapters}
    requests = [revision(by_index[i].id, "chapter_storyboard", "scene_direction", "scene_director",
                         "timeline_guard", [x for x in result.issues if f"chapter {i}" in x],
                         result.suggested_changes) for i in result.metrics["bad_chapters"] if i in by_index]
    if not requests:
        raise Escalation("Timeline Guard failed for reasons the Scene Director cannot fix: "
                         + result.explanation)
    raise Rejection("scene_direction", requests, result.explanation)
