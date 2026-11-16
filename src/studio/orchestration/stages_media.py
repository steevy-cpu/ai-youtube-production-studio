"""Task handlers for prompts, visual assets and narration (approved storyboard -> approved media).

The visual branch and the narration branch only share the storyboard as a dependency,
so a failure in one does not stop the other.
"""
from __future__ import annotations

from pathlib import Path

from studio.agents.judges import PromptJudge, VisualJudge
from studio.agents.prompt_engineer import SHOT_RULES, PromptEngineer, reference_prompt
from studio.agents.visual_generation import VisualGeneration
from studio.agents.voice_director import VoiceDirector
from studio.evaluation.audio_qa import check_narration
from studio.guards import image_guard, prompt_guard, video_clip_guard
from studio.memory import APPROVED
from studio.orchestration.context import Ctx
from studio.orchestration.engine import Rejection
from studio.orchestration.stages_content import (SHOT_VISUAL, STALE, feedback_for, from_evaluation, need,
                                                 revision)
from studio.schemas import (AnimationPrompt, EvaluationResult, ImagePrompt, MediaAsset, NarrationSegment,
                            ProductionPlan, RevisionRequest, Storyboard, StyleBible)
from studio.services.openrouter import OpenRouterError


def aspect(ctx: Ctx) -> float:
    a, b = (float(x) for x in ctx.settings.aspect_ratio.split(":"))
    return a / b


def judge_image(ctx: Ctx, asset: MediaAsset, prompt: ImagePrompt, style: StyleBible, purpose: str,
                threshold: float, extra: str = "") -> EvaluationResult | None:
    """Run the Visual Judge, except on placeholders produced because image generation is off."""
    if asset.is_sample and not ctx.client.is_mock:
        return None
    return VisualJudge(ctx).judge_image(Path(asset.path), prompt, style, asset_id=asset.id, purpose=purpose,
                                        threshold=threshold, extra=extra)


def set_status(ctx: Ctx, asset: MediaAsset, status: str) -> None:
    asset.status = status
    ctx.store.update_asset(asset)


# ---- character references (Style Bible) ----------------------------------------------
def character_references(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    style, s = need(ctx, "style_bible", StyleBible), ctx.settings
    for note in ctx.caps.get("notes", []):
        ctx.store.add_limitation(ctx.project_id, note)
    if s.max_reference_images == 0:
        ctx.store.add_limitation(ctx.project_id, "No character reference images (MAX_REFERENCE_IMAGES=0); "
                                                 "consistency relies on prompt text only.")
        return "skipped: reference images disabled"
    generator = VisualGeneration(ctx)
    approved = {a.metadata.get("character"): a
                for a in ctx.store.assets(ctx.project_id, "reference", status=APPROVED)}
    for character in style.characters[:s.max_reference_images]:
        if character.name in approved:
            continue
        prompt, notes = reference_prompt(character, style), ""
        ctx.store.save_artifact(ctx.project_id, "reference_prompt", prompt, key=character.name)
        for _ in range(s.max_visual_attempts):
            asset = generator.reference_image(prompt, style, f"ref_{character.name.lower()[:20]}", notes)
            asset.metadata["character"] = character.name
            guard = image_guard(asset.path, aspect_ratio=aspect(ctx), target_id=asset.id)
            guard.project_id = ctx.project_id
            ctx.store.add_evaluation(guard)
            verdict = judge_image(ctx, asset, prompt, style, "reference", s.visual_judge_threshold) \
                if guard.passed else guard
            if guard.passed and (verdict is None or verdict.passed):
                set_status(ctx, asset, "approved")
                approved[character.name] = asset
                break
            set_status(ctx, asset, "rejected")
            notes = "; ".join(verdict.issues + verdict.suggested_changes)
        else:
            ctx.store.add_limitation(ctx.project_id, f"No approved reference image for '{character.name}' "
                                     f"after {s.max_visual_attempts} attempts; consistency relies on prompts.")
    style.reference_asset_ids = [a.id for a in approved.values()]
    if not ctx.caps.get("image_references"):
        style.limitations = ["Reference images are stored but cannot be passed to the image model; "
                             "perfect character consistency is not guaranteed."]
    ctx.store.update_artifact(style)
    return f"{len(approved)} approved character reference(s)"


# ---- prompts ---------------------------------------------------------------------------
def prompt_engineering(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, style = need(ctx, "storyboard", Storyboard), need(ctx, "style_bible", StyleBible)
    existing = ctx.store.load_all(ctx.project_id, "image_prompt", ImagePrompt)
    engineer, created, revised = PromptEngineer(ctx), 0, 0
    for scene in storyboard.scenes:
        # No prompt yet, or the Scene Director gave the shot a new picture: write it from scratch.
        missing = [shot for shot in scene.shots
                   if shot.id not in existing or existing[shot.id].status == STALE]
        if missing:
            for shot_id, (image, animation) in engineer.scene_prompts(scene, missing, style).items():
                ctx.store.save_artifact(ctx.project_id, "image_prompt", image, key=shot_id)
                if animation:
                    ctx.store.save_artifact(ctx.project_id, "animation_prompt", animation, key=shot_id)
                created += 1
    for shot in storyboard.shots:
        current = existing.get(shot.id)
        if current is None or current.status == STALE:
            continue
        notes = feedback_for(feedback, shot.id)
        if not notes and current.status == "rejected":
            # Rejected, but the request did not arrive as feedback (a resume after an escalation, or
            # the review sent other shots to the Scene Director first): use the judge's last verdict.
            failed = [e for e in ctx.store.evaluations(ctx.project_id, target_id=current.id) if not e.passed]
            if failed:
                notes = from_evaluation(failed[-1], shot.id, "prompt", "prompt_engineering",
                                        "prompt_engineer").as_feedback()
        if notes:
            prompt = engineer.revise(shot, current, style, notes)
            ctx.store.save_artifact(ctx.project_id, "image_prompt", prompt, key=shot.id)
            revised += 1
    return f"{created} prompt(s) written, {revised} revised after feedback"


def prompt_review(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, style = need(ctx, "storyboard", Storyboard), need(ctx, "style_bible", StyleBible)
    prompts = ctx.store.load_all(ctx.project_id, "image_prompt", ImagePrompt)
    shot_ids = {shot.id for shot in storyboard.shots}
    judge, requests, approved = PromptJudge(ctx), [], 0
    judged: list[tuple] = []          # (shot, failed Prompt Judge evaluation)

    def reject(shot, prompt, evaluation):
        prompt.status = "rejected"
        ctx.store.update_artifact(prompt)
        requests.append(from_evaluation(evaluation, shot.id, "prompt", "prompt_engineering", "prompt_engineer"))
        if evaluation.evaluator == judge.name:
            judged.append((shot, evaluation))

    for scene in storyboard.scenes:
        to_judge = []
        for shot in scene.shots:
            prompt = prompts.get(shot.id)
            if prompt is None:
                requests.append(revision(shot.id, "prompt", "prompt_engineering", "prompt_engineer",
                                         "prompt_guard", [f"no prompt was produced for shot {shot.index}"]))
            elif prompt.status not in APPROVED:
                guard = prompt_guard(prompt, shot_ids)
                guard.project_id = ctx.project_id
                ctx.store.add_evaluation(guard)
                if guard.passed:
                    to_judge.append((shot, prompt))
                else:
                    reject(shot, prompt, guard)
        if to_judge:
            evaluations = judge.judge_scene(scene, to_judge, style)
            for shot, prompt in to_judge:
                if evaluations[shot.id].passed:
                    prompt.status = "approved"
                    ctx.store.update_artifact(prompt)
                    approved += 1
                else:
                    reject(shot, prompt, evaluations[shot.id])

    for shot_id, animation in ctx.store.load_all(ctx.project_id, "animation_prompt", AnimationPrompt).items():
        if animation.status == "draft":         # animation prompts get the deterministic guard only
            guard = prompt_guard(animation, shot_ids)
            guard.project_id = ctx.project_id
            ctx.store.add_evaluation(guard)
            animation.status = "approved" if guard.passed else "rejected"
            ctx.store.update_artifact(animation)
            if not guard.passed:
                ctx.store.add_limitation(ctx.project_id, f"Animation prompt for shot {shot_id} failed the "
                                                         "Prompt Guard; that shot stays a still image.")
    if requests:
        to_scene = scene_requests(ctx, judged)
        if to_scene:
            raise Rejection("scene_direction", to_scene,
                            f"{len(to_scene)} shot(s) need a different picture ({len(requests)} prompt(s) rejected)")
        raise Rejection("prompt_engineering", requests, f"{len(requests)} prompt(s) rejected")
    return f"{approved} prompt(s) approved this pass; all {len(shot_ids)} shots have approved prompts"


def scene_requests(ctx: Ctx, judged: list[tuple]) -> list[RevisionRequest]:
    """Requests the Prompt Judge sends to the Scene Director instead of the Prompt Engineer.

    A shot goes back to the Scene Director when the judge names the required visual itself as the
    problem, or when the Prompt Engineer has no attempt left and the prompt is still rejected (another
    rewrite of the same picture would not help, and the alternative is to stop for a human). Empty
    when the Scene Director has no attempt left: the normal route and its escalation then apply.
    """
    director = ctx.store.get_task(ctx.project_id, "scene_direction")
    if director["attempts"] >= director["max_attempts"]:
        return []
    engineer = ctx.store.get_task(ctx.project_id, "prompt_engineering")
    exhausted = engineer["attempts"] >= engineer["max_attempts"]
    requests = []
    for shot, evaluation in judged:
        if evaluation.metrics.get("fault") == "scene":
            why = "the Prompt Judge found that the required visual itself cannot be drawn under the rules"
        elif exhausted:
            why = (f"the Prompt Engineer rewrote the prompt {engineer['attempts']} time(s) and the Prompt "
                   "Judge still rejects it")
        else:
            continue
        requests.append(revision(shot.id, SHOT_VISUAL, "scene_direction", "scene_director", PromptJudge.name,
                                 [why] + evaluation.issues, evaluation.suggested_changes))
    return requests



# ---- images ------------------------------------------------------------------------------
def visual_generation(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, style = need(ctx, "storyboard", Storyboard), need(ctx, "style_bible", StyleBible)
    prompts = ctx.store.load_all(ctx.project_id, "image_prompt", ImagePrompt)
    generator, made = VisualGeneration(ctx), 0
    for shot in storyboard.shots:
        latest = ctx.store.latest_asset(ctx.project_id, "image", shot.id)
        if latest and latest.status in APPROVED + ("generated",):
            continue                           # approved, or generated and still waiting for review
        notes = feedback_for(feedback, shot.id)
        if not notes and latest and latest.status == "rejected":
            # Resumed after the attempts ran out: start from the last review, not from nothing.
            notes = "\n".join(f"- problem: {i}" for e in ctx.store.evaluations(ctx.project_id, target_id=latest.id)
                              if not e.passed for i in e.issues)
        # A rejected picture is redrawn by the stronger model: asking the same model again with a
        # correction note fixed only about a third of the rejected images per pass in live runs.
        retry = bool(latest and latest.status == "rejected")
        generator.shot_image(shot, prompts[shot.id], style, notes,
                             model=ctx.settings.image_retry if retry else None)
        made += 1
    return f"{made} image(s) generated; {len(storyboard.shots) - made} kept from earlier passes"


def visual_review(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, style = need(ctx, "storyboard", Storyboard), need(ctx, "style_bible", StyleBible)
    prompts = ctx.store.load_all(ctx.project_id, "image_prompt", ImagePrompt)
    requests, approved = [], 0
    for shot in storyboard.shots:
        asset = ctx.store.latest_asset(ctx.project_id, "image", shot.id)
        if asset is None:
            requests.append(revision(shot.id, "image", "visual_generation", "visual_generation",
                                     "media_guard", [f"no image exists for shot {shot.index}"]))
            continue
        if asset.status in APPROVED:
            continue
        result = image_guard(asset.path, aspect_ratio=aspect(ctx), target_id=asset.id)
        result.project_id = ctx.project_id
        ctx.store.add_evaluation(result)
        if result.passed:
            verdict = judge_image(ctx, asset, prompts[shot.id], style, "image",
                                  ctx.settings.visual_judge_threshold,
                                  f"The image must illustrate: {shot.visual_description}\n\n{SHOT_RULES}\n"
                                  "Judge the image under those rules: a wordless stand-in for text, "
                                  "digits or symbols is correct, and missing writing is never a defect.")
            result = verdict or result
        if result.passed:
            set_status(ctx, asset, "approved")
            approved += 1
        else:
            set_status(ctx, asset, "rejected")
            requests.append(from_evaluation(result, shot.id, "image", "visual_generation", "visual_generation"))
    if requests:
        raise Rejection("visual_generation", requests, f"{len(requests)} image(s) rejected")
    return f"{approved} image(s) approved this pass; all {len(storyboard.shots)} shots have approved images"


def animation_generation(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    s = ctx.settings
    if not s.enable_video:
        return "skipped: video generation is disabled (still images with pan/zoom are used)"
    storyboard = need(ctx, "storyboard", Storyboard)
    prompts = ctx.store.load_all(ctx.project_id, "animation_prompt", AnimationPrompt)
    wanted = [shot for shot in storyboard.shots
              if shot.media_type == "animation" and shot.id in prompts and prompts[shot.id].status in APPROVED]
    generator, made = VisualGeneration(ctx), 0
    for shot in wanted[:s.max_video_clips]:
        if ctx.store.latest_asset(ctx.project_id, "video", shot.id):
            continue                            # one attempt per shot; never regenerated automatically
        still = ctx.store.latest_asset(ctx.project_id, "image", shot.id, APPROVED)
        try:
            asset = generator.clip(shot, prompts[shot.id], Path(still.path) if still else None)
        except OpenRouterError as exc:
            ctx.store.add_error(ctx.project_id, ctx.task, "visual_generation", f"shot {shot.index}: {exc}")
            ctx.store.add_limitation(ctx.project_id, f"Video clip for shot {shot.index} failed; the approved "
                                                     "still image is used for that shot.")
            continue
        guard = video_clip_guard(asset.path, expected_s=s.video_clip_seconds, target_id=asset.id)
        guard.project_id = ctx.project_id
        ctx.store.add_evaluation(guard)
        set_status(ctx, asset, "approved" if guard.passed else "rejected")
        if guard.passed:
            made += 1
        else:
            ctx.store.add_limitation(ctx.project_id, f"Video clip for shot {shot.index} failed the Media "
                                                     "Guard; the approved still image is used instead.")
    if len(wanted) > s.max_video_clips:
        ctx.store.add_limitation(ctx.project_id, f"{len(wanted) - s.max_video_clips} animated shot(s) were "
                                                 "left as still images (MAX_VIDEO_CLIPS).")
    ctx.store.add_limitation(ctx.project_id, "Generated video clips are validated by the Media Guard only; "
                                             "they are not reviewed by the Visual Judge.")
    return f"{made} video clip(s) generated and validated"


# ---- narration -----------------------------------------------------------------------------
def voice_direction(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard, plan = need(ctx, "storyboard", Storyboard), need(ctx, "plan", ProductionPlan)
    existing = ctx.store.load_all(ctx.project_id, "narration", NarrationSegment)
    if all(scene.id in existing and existing[scene.id].text == scene.narration for scene in storyboard.scenes):
        return f"{len(storyboard.scenes)} narration segments already planned"
    planned = [existing[scene.id] for scene in storyboard.scenes if scene.id in existing]
    if planned and len(planned) == len(storyboard.scenes):
        # Same scenes, some with shortened narration (Final QA duration trim): keep the voice plan and
        # every unchanged recording; only the changed scenes get a new segment and are re-recorded.
        changed = 0
        for scene in storyboard.scenes:
            old = existing[scene.id]
            if old.text != scene.narration:
                ctx.store.save_artifact(ctx.project_id, "narration", NarrationSegment(
                    parent_id=scene.id, index=scene.index, text=scene.narration, voice=old.voice,
                    speed=old.speed, instructions=old.instructions, status="approved",
                    estimated_s=round(max(len(scene.narration.split()), 1) / ctx.settings.words_per_minute
                                      * 60 / old.speed, 2)), key=scene.id)
                changed += 1
        return f"{changed} narration segment(s) updated, {len(planned) - changed} kept with their recording"
    voice_plan, segments = VoiceDirector(ctx).plan(storyboard, plan)
    for segment in segments:
        segment.status = "approved"
        ctx.store.save_artifact(ctx.project_id, "narration", segment, key=segment.parent_id)
    ctx.store.remember(ctx.project_id, "voice_plan", voice_plan.model_dump() | {"voice": ctx.settings.tts_voice})
    return f"{len(segments)} narration segments, voice '{ctx.settings.tts_voice}', speed {segments[0].speed}"


def narration_tts(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard = need(ctx, "storyboard", Storyboard)
    segments = ctx.store.load_all(ctx.project_id, "narration", NarrationSegment)
    director, made = VoiceDirector(ctx), 0
    for scene in storyboard.scenes:
        segment = segments[scene.id]
        latest = ctx.store.latest_asset(ctx.project_id, "audio", segment.id)
        if latest and latest.status in APPROVED + ("generated",):
            continue
        director.speak(segment)
        made += 1
    return f"{made} narration file(s) generated"


def audio_qa(ctx: Ctx, feedback: list[RevisionRequest]) -> str:
    storyboard = need(ctx, "storyboard", Storyboard)
    segments = ctx.store.load_all(ctx.project_id, "narration", NarrationSegment)
    requests, total = [], 0.0
    for scene in storyboard.scenes:
        segment = segments[scene.id]
        asset = ctx.store.latest_asset(ctx.project_id, "audio", segment.id)
        if asset is None:
            requests.append(revision(segment.id, "audio", "narration_tts", "voice_director", "audio_qa",
                                     [f"no audio exists for narration segment {segment.index}"]))
            continue
        if asset.status not in APPROVED:
            result = check_narration(ctx, segment, asset)
            set_status(ctx, asset, "approved" if result.passed else "rejected")
            if not result.passed:
                requests.append(from_evaluation(result, segment.id, "audio", "narration_tts", "voice_director"))
                continue
        total += asset.duration_s or 0.0
    if requests:
        raise Rejection("narration_tts", requests, f"{len(requests)} narration segment(s) rejected")
    if not ctx.settings.stt_model:
        ctx.store.add_limitation(ctx.project_id, "Narration wording was not verified by transcription "
                                                 "(no STT_MODEL configured).")
    return f"all {len(storyboard.scenes)} narration segments approved, {total:.1f}s of measured audio"
