"""Task handlers for editing, rendering, final review, SEO, the thumbnail loop and export."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from studio.agents.compliance import Compliance
from studio.agents.director import Director
from studio.agents.editor import Editor
from studio.agents.final_qa import FinalQA
from studio.agents.judges import PromptJudge
from studio.agents.prompt_engineer import THUMBNAIL_RULES, PromptEngineer
from studio.agents.scene_director import allowed_removals, numbered_sentences
from studio.agents.seo import SEO
from studio.agents.visual_generation import VisualGeneration, thumbnail_size
from studio.guards import image_guard, prompt_guard
from studio.memory import APPROVED
from studio.orchestration.context import Ctx
from studio.orchestration.engine import Escalation, Rejection
from studio.orchestration.outputs import publish_work_files
from studio.orchestration.stages_content import DURATION_TRIM, from_evaluation, need, revision
from studio.orchestration.stages_media import aspect, judge_image, set_status
from studio.rendering.ffmpeg import probe
from studio.rendering.renderer import render
from studio.schemas import (ComplianceReport, EditingTimeline, ImagePrompt, MediaAsset, NarrationSegment,
                            ProductionPlan, ProductionReport, ProjectBrief, ResearchBrief, RevisionRequest,
                            Script, SEOPackage, Storyboard, StyleBible, ThumbnailBrief)
from studio.utils import now_iso, word_count

FINAL_EVALUATORS = ("final_video_guard", "final_av_check", "final_qa_continuity")


def human_override(ctx: Ctx, subjects: tuple[str, ...]) -> dict | None:
    """The most recent explicitly recorded human decision for any of the given task names."""
    found = [d for d in ctx.store.decisions(ctx.project_id, actor="human") if d["subject"] in subjects]
    return found[-1] if found else None


# ---- editing and rendering ---------------------------------------------------------
def editing(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, s = need(ctx, "storyboard", Storyboard), ctx.settings
    segments = ctx.store.load_all(ctx.project_id, "narration", NarrationSegment)
    visuals, narration = {}, {}
    for shot in storyboard.shots:
        asset = (ctx.store.latest_asset(ctx.project_id, "video", shot.id, APPROVED)
                 or ctx.store.latest_asset(ctx.project_id, "image", shot.id, APPROVED))
        if asset is None or not Path(asset.path).is_file():
            raise RuntimeError(f"shot {shot.index} has no approved visual asset on disk")
        visuals[shot.id] = asset
    for scene in storyboard.scenes:
        asset = ctx.store.latest_asset(ctx.project_id, "audio", segments[scene.id].id, APPROVED)
        if asset is None or not Path(asset.path).is_file():
            raise RuntimeError(f"scene {scene.index} has no approved narration audio on disk")
        narration[scene.id] = asset
    timeline = Editor(ctx).build_timeline(storyboard, visuals, narration)
    ctx.store.save_artifact(ctx.project_id, "timeline", timeline)
    known = {a.path for a in ctx.store.assets(ctx.project_id, "music")}
    if timeline.music_path and timeline.music_path not in known:
        library = bool(timeline.music_title)
        ctx.store.add_asset(MediaAsset(
            project_id=ctx.project_id, kind="music", path=timeline.music_path, status="approved",
            provenance=(f"music library track '{timeline.music_title}', source {timeline.music_source}"
                        if library else "user-supplied file (MUSIC_PATH)"),
            license=timeline.music_license, metadata={"title": timeline.music_title,
                                                      "source": timeline.music_source}))
    for note in timeline.notes:
        ctx.store.add_limitation(ctx.project_id, note)
    return f"timeline: {len(timeline.clips)} clips, {timeline.total_duration_s:.1f}s"


def rendering(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    timeline = need(ctx, "timeline", EditingTimeline)
    out = render(timeline, ctx.dirs.sub("render"), ctx.dirs.sub("final") / "video.mp4",
                 ctx.settings.font_path, ctx.log)
    info = probe(out)
    sample = any(a.is_sample for a in ctx.store.assets(ctx.project_id) if a.kind in ("image", "audio", "video"))
    ctx.store.add_asset(MediaAsset(
        project_id=ctx.project_id, parent_id=timeline.id, kind="final_video", path=str(out), mime="video/mp4",
        width=(info["video"] or {}).get("width"), height=(info["video"] or {}).get("height"),
        duration_s=info["duration_s"], is_sample=sample, status="rendered",
        provenance="assembled locally with FFmpeg from this project's assets"
                   + (" (contains placeholder media)" if sample else "")))
    preview = publish_work_files(ctx.store, ctx.settings, ctx.project_id)   # available before any review
    return f"rendered {out.name}: {info['duration_s']:.1f}s (unapproved copy in {preview.name}/preliminary)"


DURATION_CHECK = "duration within accepted range"
MAX_TRIM_SHARE = 0.4        # beyond this the script itself is wrong; a human decides


def duration_rework(ctx: Ctx, guard, timeline: EditingTimeline, storyboard: Storyboard
                    ) -> RevisionRequest | None:
    """The request Final QA sends to the Scene Director when the ONLY problem is a video that is too
    long, or None when that route does not apply (another check failed, the video is too short, the
    cut would be too deep, or the Scene Director has no attempt left) and a human has to decide."""
    failed = [c.name for c in guard.criteria if not c.passed]
    duration = guard.metrics.get("duration_s") or 0.0
    low, high = ctx.settings.duration_bounds
    if failed != [DURATION_CHECK] or duration <= high:
        return None
    task = ctx.store.get_task(ctx.project_id, "scene_direction")
    if task["attempts"] >= task["max_attempts"]:
        return None
    words = sum(word_count(scene.narration) for scene in storyboard.scenes)
    speech = sum(a.audio_s for a in timeline.audio)
    if not words or not speech:
        return None
    aim = min(max(ctx.settings.target_duration_s, low), high)         # the target, not the edge of the range
    words_to_cut = round((duration - aim) * words / speech)
    if words_to_cut < 1 or words_to_cut > words * MAX_TRIM_SHARE:
        return None
    rows = numbered_sentences(storyboard)          # is a cut of that size possible at all?
    removable = allowed_removals(rows, [r["n"] for r in rows], ctx.settings)
    if sum(r["words"] for r in rows if r["n"] in removable) < words_to_cut * 0.8:
        return None
    return revision(storyboard.id, DURATION_TRIM, "scene_direction", "scene_director", "final_qa",
                    [f"the rendered video lasts {duration:.1f}s; the accepted range is {low:.0f}-{high:.0f}s. "
                     f"The narrator needs {speech / words:.2f}s per word, so about {words_to_cut} of the "
                     f"{words} narration words have to go to reach about {aim:.0f}s"],
                    [str(words_to_cut), str(round(words / speech * 60))])


def final_qa(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    timeline, storyboard = need(ctx, "timeline", EditingTimeline), need(ctx, "storyboard", Storyboard)
    video = ctx.store.latest_asset(ctx.project_id, "final_video", timeline.id)
    results = FinalQA(ctx).review(Path(video.path), timeline, storyboard, need(ctx, "script", Script))
    guard = results[0]
    if not guard.passed:
        request = duration_rework(ctx, guard, timeline, storyboard)
        if request is not None:
            raise Rejection("scene_direction", [request], "Final QA: " + "; ".join(guard.issues))
        raise Escalation("Final Video Guard failed: " + "; ".join(guard.issues)
                         + (" | suggested: " + "; ".join(guard.suggested_changes) if guard.suggested_changes
                            else ""))
    failed = [r.evaluator for r in results if not r.passed]
    return "all final checks passed" if not failed else f"guard passed; advisory issues from {failed}"


def compliance(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    report = Compliance(ctx).review(need(ctx, "script", Script), need(ctx, "research", ResearchBrief),
                                    ctx.store.assets(ctx.project_id))
    ctx.store.save_artifact(ctx.project_id, "compliance", report)
    if report.blocked:
        failed = [f"{i.check}: {i.detail}" for i in report.items if i.mandatory and i.status == "fail"]
        raise Escalation("Mandatory compliance checks failed: " + " | ".join(failed))
    return f"not blocked; {len(report.human_review_required)} item(s) need human review"


def director_final_approval(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    report = need(ctx, "compliance", ComplianceReport)
    if report.blocked:
        override = human_override(ctx, ("compliance",))
        if override is None:    # the Director can never wave a mandatory compliance failure through
            raise Escalation("Compliance is blocked and no human decision is recorded; the Director "
                             "cannot approve.")
        ctx.store.add_decision(ctx.project_id, "director", "final_video", "noted_human_override",
                               f"compliance block overridden by a human: {override['reason']}")
    evaluations = [e for name in FINAL_EVALUATORS for e in ctx.store.evaluations(ctx.project_id, name)[-1:]]
    decision = Director(ctx).final_approval(evaluations, report)
    ctx.store.add_decision(ctx.project_id, "director", "final_video",
                           "approved" if decision.approved else "rejected", "; ".join(decision.reasons))
    if not decision.approved:
        raise Escalation("The Director rejected the finished video: " + "; ".join(decision.reasons))
    ctx.store.update_project(ctx.project_id, final_approval="video_approved")
    return "finished video approved by the Director"


# ---- SEO and thumbnail branch ---------------------------------------------------------
def seo_package(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, style = need(ctx, "brief", ProjectBrief), need(ctx, "style_bible", StyleBible)
    result = SEO(ctx).package(need(ctx, "script", Script), need(ctx, "plan", ProductionPlan),
                              brief.topic, style)
    result.seo.status = result.thumbnail_brief.status = "ready"
    seo = ctx.store.save_artifact(ctx.project_id, "seo", result.seo)
    result.thumbnail_brief.parent_id = seo.id
    ctx.store.save_artifact(ctx.project_id, "thumbnail_brief", result.thumbnail_brief)
    return f"title: {result.seo.selected_title!r}; thumbnail concept: {result.thumbnail_brief.concept[:60]}"


def metadata_compliance(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    report = Compliance(ctx).review_metadata(need(ctx, "compliance", ComplianceReport),
                                             need(ctx, "seo", SEOPackage),
                                             need(ctx, "thumbnail_brief", ThumbnailBrief))
    ctx.store.save_artifact(ctx.project_id, "compliance", report)
    newly_failed = [i for i in report.items if i.check.startswith("metadata_") and i.status == "fail"]
    if newly_failed:
        raise Escalation("Metadata failed compliance screening: "
                         + " | ".join(f"{i.check}: {i.detail}" for i in newly_failed))
    return "metadata screened; no mandatory failure"


def thumbnail_prompt(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, style = need(ctx, "thumbnail_brief", ThumbnailBrief), need(ctx, "style_bible", StyleBible)
    notes = "\n".join(r.as_feedback() for r in feedback)
    prompt = PromptEngineer(ctx).thumbnail_prompt(brief, style, notes)
    ctx.store.save_artifact(ctx.project_id, "thumbnail_prompt", prompt)
    return f"thumbnail prompt v{prompt.version} written"


def thumbnail_prompt_review(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, style = need(ctx, "thumbnail_brief", ThumbnailBrief), need(ctx, "style_bible", StyleBible)
    prompt = need(ctx, "thumbnail_prompt", ImagePrompt)
    result = prompt_guard(prompt, {brief.id})
    result.project_id = ctx.project_id
    ctx.store.add_evaluation(result)
    if result.passed:
        result = PromptJudge(ctx).judge_thumbnail_prompt(prompt, brief, style)
    prompt.status = "approved" if result.passed else "rejected"
    ctx.store.update_artifact(prompt)
    if not result.passed:
        raise Rejection("thumbnail_prompt", [from_evaluation(
            result, prompt.id, "thumbnail_prompt", "thumbnail_prompt", "prompt_engineer")], result.explanation)
    return f"thumbnail prompt approved (score {result.score})"


def thumbnail_generation(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, style = need(ctx, "thumbnail_brief", ThumbnailBrief), need(ctx, "style_bible", StyleBible)
    latest = ctx.store.latest_asset(ctx.project_id, "thumbnail", brief.id)
    if latest and latest.status in APPROVED + ("generated", "reviewed") and not feedback:
        return "thumbnail candidate already waiting for review"
    notes = "; ".join(i for r in feedback for i in r.issues + r.suggested_changes)
    if not notes and latest and latest.status == "rejected":
        # Resumed after the attempts ran out: the task's feedback is gone, but the last candidate's
        # failed reviews are still in memory, so the retry starts from them, not from scratch.
        notes = "; ".join(i for e in ctx.store.evaluations(ctx.project_id, target_id=latest.id)
                          if not e.passed for i in e.issues + e.suggested_changes)
    prompt = need(ctx, "thumbnail_prompt", ImagePrompt)
    if notes:
        # Reviewers rejected the picture. Repeating the same prompt with corrections appended rarely
        # fixes a composition problem, so the Prompt Engineer rewrites the prompt from the review
        # first; the reviewers then judge the new candidate against that rewritten prompt.
        prompt = PromptEngineer(ctx).thumbnail_prompt(brief, style, notes)
        prompt.status = "approved"
        ctx.store.save_artifact(ctx.project_id, "thumbnail_prompt", prompt)
    asset = VisualGeneration(ctx).thumbnail(prompt, brief, style, notes)
    candidates = len(ctx.store.assets(ctx.project_id, "thumbnail", brief.id))
    return f"thumbnail candidate {candidates} generated ({Path(asset.path).name})"


def thumbnail_review(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, style = need(ctx, "thumbnail_brief", ThumbnailBrief), need(ctx, "style_bible", StyleBible)
    prompt, seo = need(ctx, "thumbnail_prompt", ImagePrompt), need(ctx, "seo", SEOPackage)
    asset = ctx.store.latest_asset(ctx.project_id, "thumbnail", brief.id)
    if asset is None:
        raise Rejection("thumbnail_generation", [revision(brief.id, "thumbnail", "thumbnail_generation",
                        "visual_generation", "media_guard", ["no thumbnail candidate exists"])])
    if asset.status in APPROVED + ("reviewed",):
        return "thumbnail candidate already reviewed"
    guard = image_guard(asset.path, aspect_ratio=aspect(ctx), target_id=asset.id,
                        exact_size=thumbnail_size(ctx.settings.aspect_ratio))
    guard.project_id = ctx.project_id
    results = [ctx.store.add_evaluation(guard)]
    if guard.passed and not (asset.is_sample and not ctx.client.is_mock):
        results.append(SEO(ctx).review_thumbnail(Path(asset.path), brief, seo.selected_title, asset.id))
        # The Visual Judge inspects the artwork as the model made it, BEFORE the title is added, so
        # any text it sees was wrongly painted by the model. The SEO agent reviewed the finished image.
        raw = asset.model_copy(update={"path": asset.metadata.get("raw_path") or asset.path})
        results.append(judge_image(
            ctx, raw, prompt, style, "thumbnail", ctx.settings.thumbnail_threshold,
            "This is YouTube thumbnail artwork before software adds the title banner.\n"
            f"What the thumbnail has to communicate (the SEO brief): {brief.concept} Main subject: "
            f"{brief.main_subject} Emotion: {brief.emotion}.\n"
            "Judge whether the picture delivers THAT, with one clear focal subject that still reads at "
            "about 160 pixels wide. The prompt above is only how the picture was asked for: a pose, "
            "prop, perspective or framing that came out differently from the prompt is not a defect "
            "when the idea still reads. Serious for a thumbnail, besides the review rules below: any "
            "title, caption, headline or label text in the artwork (a small mark that is part of a "
            "character's Style Bible design is acceptable), and a face inside the bottom sixth of the "
            "frame, which the banner will cover. Props, furniture, legs or floor in that bottom sixth "
            "are fine.\n" + THUMBNAIL_RULES))
    failed = [r for r in results if not r.passed]
    if failed:
        set_status(ctx, asset, "rejected")
        raise Rejection("thumbnail_generation", [from_evaluation(
            r, brief.id, "thumbnail", "thumbnail_generation", "visual_generation") for r in failed],
            "; ".join(f"{r.evaluator}: {'; '.join(r.issues)}" for r in failed))
    set_status(ctx, asset, "reviewed")
    return f"thumbnail candidate passed {len(results)} review(s)"


def director_thumbnail_approval(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    brief, seo = need(ctx, "thumbnail_brief", ThumbnailBrief), need(ctx, "seo", SEOPackage)
    asset = ctx.store.latest_asset(ctx.project_id, "thumbnail", brief.id)
    if asset.status == "human_approved":
        ctx.store.add_decision(ctx.project_id, "director", "thumbnail", "accepted_human_decision",
                               "thumbnail approved by a recorded human override")
        return "thumbnail accepted by recorded human decision"
    decision = Director(ctx).approve_thumbnail(brief, seo.selected_title,
                                               ctx.store.evaluations(ctx.project_id, target_id=asset.id))
    ctx.store.add_decision(ctx.project_id, "director", "thumbnail",
                           "approved" if decision.approved else "rejected", "; ".join(decision.reasons))
    if not decision.approved:
        # A Director rejection is about the concept, so the PROMPT is revised (and re-judged) first.
        # Sending it only to the image step would leave the judges checking against the old prompt.
        set_status(ctx, asset, "rejected")
        raise Rejection("thumbnail_prompt", [revision(
            need(ctx, "thumbnail_prompt", ImagePrompt).id, "thumbnail_prompt", "thumbnail_prompt",
            "prompt_engineer", "director",
            decision.reasons or ["the Director rejected the thumbnail without details"])],
            "Director rejected the thumbnail concept")
    set_status(ctx, asset, "approved")
    return "thumbnail approved by the Director"


# ---- export ----------------------------------------------------------------------------
def export(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    pid, out, store = ctx.project_id, ctx.dirs.output, ctx.store
    brief, seo = need(ctx, "brief", ProjectBrief), need(ctx, "seo", SEOPackage)
    thumb_brief, report = need(ctx, "thumbnail_brief", ThumbnailBrief), need(ctx, "compliance", ComplianceReport)
    timeline = need(ctx, "timeline", EditingTimeline)
    video = store.latest_asset(pid, "final_video", timeline.id)
    thumbnail = store.latest_asset(pid, "thumbnail", thumb_brief.id, APPROVED)
    if thumbnail is None:
        raise RuntimeError("no approved thumbnail to export")
    if report.blocked and human_override(ctx, ("compliance", "metadata_compliance")) is None:
        raise Escalation("Export refused: compliance is blocked and no human decision is recorded.")

    shutil.copy2(video.path, out / "video.mp4")
    captions = Path(video.path).with_suffix(".srt")
    if captions.is_file():
        shutil.copy2(captions, out / "captions.srt")
    shutil.copy2(thumbnail.path, out / "thumbnail.png")
    sample = video.is_sample or thumbnail.is_sample
    notice = ("SAMPLE OUTPUT: built from locally generated placeholder media. Not AI-generated and not "
              "publishable.") if sample else ""
    metadata = {"title": seo.selected_title, "title_options": seo.titles, "description": seo.description,
                "tags": seo.tags, "keywords": seo.keywords, "hook": seo.hook,
                "retention_suggestions": seo.retention_suggestions, "seo_disclaimer": seo.disclaimer,
                "synthetic_media_disclosure": report.synthetic_media_disclosure,
                "thumbnail": "thumbnail.png", "video": "video.mp4", "language": brief.language,
                "music": ({"title": timeline.music_title or Path(timeline.music_path).name,
                           "license": timeline.music_license, "source": timeline.music_source}
                          if timeline.music_path else None),
                "duration_s": video.duration_s, "resolution": f"{video.width}x{video.height}",
                "mode": brief.mode, "sample_notice": notice, "generated_at": now_iso(),
                "published": False}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "description.txt").write_text(seo.description + "\n\n" + report.synthetic_media_disclosure + "\n",
                                         encoding="utf-8")
    quality = {"sample_notice": notice, "summary": store.evaluation_summary(pid),
               "final_checks": [e.model_dump() for name in FINAL_EVALUATORS
                                for e in store.evaluations(pid, name)[-1:]],
               "thumbnail_reviews": [e.model_dump() for e in store.evaluations(pid, target_id=thumbnail.id)],
               "revision_requests": store.revisions(pid), "limitations": store.recall(pid, "limitations", [])}
    (out / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    compliance_out = report.model_dump() | {"human_decisions": store.decisions(pid, actor="human")}
    (out / "compliance_report.json").write_text(json.dumps(compliance_out, indent=2), encoding="utf-8")

    store.update_project(pid, final_approval="approved")
    final = ("video.mp4", "captions.srt", "thumbnail.png", "metadata.json", "description.txt",
             "quality_report.json", "compliance_report.json")
    files = [name for name in final if (out / name).is_file()] + ["production_report.json"]
    production = ProductionReport(
        project_id=pid, topic=brief.topic, title=seo.selected_title, mode=brief.mode, status="final",
        final_status="approved" + (" (sample output)" if sample else ""),
        outputs={name: str(out / name) for name in sorted(set(files))},
        tasks=[{k: t[k] for k in ("name", "agent", "state", "attempts", "summary")}
               | ({"state": "COMPLETED"} if t["name"] == ctx.task else {}) for t in store.tasks(pid)],
        costs=store.cost_summary(pid), evaluations={"summary": store.evaluation_summary(pid)},
        limitations=store.recall(pid, "limitations", []), human_decisions=store.decisions(pid, actor="human"))
    store.save_artifact(pid, "report", production)
    (out / "production_report.json").write_text(production.model_dump_json(indent=2), encoding="utf-8")
    return f"exported {len(set(files))} files to {out}"
