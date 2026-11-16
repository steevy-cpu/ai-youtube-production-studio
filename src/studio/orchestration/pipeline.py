"""The production workflow: task graph, project creation, run/resume and human overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from studio.config import ConfigError, Mode, Settings
from studio.memory import APPROVED, MemoryStore
from studio.orchestration import stages_content as content
from studio.orchestration import stages_finish as finish
from studio.orchestration import stages_media as media
from studio.orchestration.context import Budget, Ctx, ProjectDirs
from studio.orchestration.engine import COMPLETED, NEEDS_REVIEW, Engine, TaskSpec
from studio.orchestration.locks import HeldLock, LockBusy
from studio.orchestration.outputs import publish_work_files
from studio.rendering.ffmpeg import ffmpeg_available
from studio.schemas import ImagePrompt, NarrationSegment, ProjectBrief, Storyboard, ThumbnailBrief
from studio.services.mock import MockOpenRouterClient
from studio.services.openrouter import OpenRouterClient, require_capabilities
from studio.utils import get_logger, new_id, slugify


def build_specs(s: Settings) -> list[TaskSpec]:
    """The whole workflow as an ordered, dependency-checked task list."""
    T = TaskSpec
    return [
        # Topic -> approved script
        T("director_plan", "director", content.director_plan, [], 2),
        T("research", "research", content.research, ["director_plan"]),
        T("story_writing", "story_writer", content.story_writing, ["research"], s.max_script_attempts),
        T("script_guard", "script_guard", content.run_script_guard, ["story_writing"]),
        T("director_script_approval", "director", content.director_script_approval, ["script_guard"]),
        # approved script -> approved storyboard
        T("scene_direction", "scene_director", content.scene_direction, ["director_script_approval"],
          s.max_storyboard_attempts),
        T("timeline_guard", "timeline_guard", content.run_timeline_guard, ["scene_direction"]),
        # visual branch
        T("character_references", "visual_generation", media.character_references, ["timeline_guard"]),
        T("prompt_engineering", "prompt_engineer", media.prompt_engineering, ["timeline_guard"],
          s.max_prompt_attempts),
        T("prompt_review", "prompt_judge", media.prompt_review, ["prompt_engineering"]),
        T("visual_generation", "visual_generation", media.visual_generation,
          ["prompt_review", "character_references"], s.max_visual_attempts),
        T("visual_review", "visual_judge", media.visual_review, ["visual_generation"]),
        T("animation_generation", "visual_generation", media.animation_generation, ["visual_review"]),
        # narration branch (independent of the visual branch)
        T("voice_direction", "voice_director", media.voice_direction, ["timeline_guard"]),
        T("narration_tts", "voice_director", media.narration_tts, ["voice_direction"], s.max_audio_attempts),
        T("audio_qa", "audio_qa", media.audio_qa, ["narration_tts"]),
        # assembly and final review
        T("editing", "editor", finish.editing, ["animation_generation", "audio_qa"]),
        T("rendering", "renderer", finish.rendering, ["editing"]),
        T("final_qa", "final_qa", finish.final_qa, ["rendering"]),
        T("compliance", "compliance", finish.compliance, ["final_qa"]),
        T("director_final_approval", "director", finish.director_final_approval, ["compliance"]),
        # SEO and thumbnail branch
        T("seo_package", "seo", finish.seo_package, ["director_final_approval"]),
        T("metadata_compliance", "compliance", finish.metadata_compliance, ["seo_package"]),
        T("thumbnail_prompt", "prompt_engineer", finish.thumbnail_prompt, ["seo_package"],
          s.max_prompt_attempts),
        T("thumbnail_prompt_review", "prompt_judge", finish.thumbnail_prompt_review, ["thumbnail_prompt"]),
        T("thumbnail_generation", "visual_generation", finish.thumbnail_generation,
          ["thumbnail_prompt_review"], s.max_thumbnail_attempts),      # separate thumbnail retry counter
        T("thumbnail_review", "seo", finish.thumbnail_review, ["thumbnail_generation"]),
        T("director_thumbnail_approval", "director", finish.director_thumbnail_approval,
          ["thumbnail_review", "metadata_compliance"]),
        T("export", "orchestrator", finish.export, ["director_thumbnail_approval"]),
    ]


def create_project(settings: Settings, store: MemoryStore, topic: str, mode: Mode, *,
                   title: str | None = None, budget_usd: float | None = None,
                   sources_dir: str | None = None, enable_video: bool = False) -> str:
    """Register a new project and its directories. Makes no API call."""
    topic = " ".join(topic.split())
    if not topic:
        raise ConfigError("the topic must not be empty")
    if sources_dir and not Path(sources_dir).is_dir():
        raise ConfigError(f"sources directory not found: {sources_dir}")
    effective = settings.for_mode(mode)
    if effective.target_words / effective.min_chapters < 25:
        raise ConfigError(f"a {effective.target_duration_s:.0f}s video (~{effective.target_words} words) is "
                          f"too short for MIN_CHAPTERS={effective.min_chapters}; lower MIN_CHAPTERS or raise "
                          "the duration")
    project_id = f"{slugify(topic)}-{new_id()[:6]}"
    brief = ProjectBrief(
        id=project_id, project_id=project_id, topic=topic, mode=mode.value, status="created",
        title=title or settings.format_template.format(topic=topic),
        target_duration_s=effective.target_duration_s, width=effective.width, height=effective.height,
        language=settings.language, sources_dir=sources_dir or "", enable_video=enable_video)
    budget = settings.max_budget_usd if budget_usd is None else budget_usd
    store.create_project(project_id, topic, brief.title, mode.value, budget, brief.model_dump())
    store.save_artifact(project_id, "brief", brief)
    ProjectDirs.create(settings, project_id)
    return project_id


def make_context(settings: Settings, store: MemoryStore, project_id: str, *, client: Any = None,
                 verbose: bool = False) -> Ctx:
    project = store.get_project(project_id)
    if project is None:
        raise ConfigError(f"unknown project '{project_id}'")
    mode = Mode(project["mode"])
    if project["brief"].get("enable_video"):        # the project opted in to AI video clips
        settings = settings.model_copy(update={"enable_video": True})
    effective = settings.for_mode(mode)
    measured_wpm = store.recall(project_id, "measured_wpm")       # set once Final QA measured the narrator
    if measured_wpm:
        effective = effective.model_copy(update={"words_per_minute": int(measured_wpm)})
    dirs = ProjectDirs.create(settings, project_id)
    log = get_logger(dirs.sub("logs") / "run.log", settings.openrouter_api_key, verbose)
    if client is None:
        client = OpenRouterClient(effective, log=log) if mode.live else MockOpenRouterClient(effective)
    budget = Budget(store, project_id, effective.max_media_requests, live=not client.is_mock)
    return Ctx(settings=effective, store=store, client=client, project_id=project_id, mode=mode, dirs=dirs,
               budget=budget, log=log)


def run_project(settings: Settings, store: MemoryStore, project_id: str, *, client: Any = None,
                confirm_live: bool = False, resume: bool = False, retry_escalated: bool = False,
                verbose: bool = False, should_stop: Callable[[], bool] | None = None) -> str:
    """Run (or resume) a project. Live modes need an API key and an explicit confirmation.

    A project can be run by one process at a time: a second run of the same project is refused,
    so two launches can never spend the same project budget concurrently.
    """
    ctx = make_context(settings, store, project_id, client=client, verbose=verbose)
    if not ffmpeg_available():
        raise ConfigError("ffmpeg and ffprobe must be installed and on PATH (needed for audio checks "
                          "and rendering).")
    if not ctx.client.is_mock:
        if not confirm_live:
            raise ConfigError("live modes make paid OpenRouter requests and need explicit confirmation")
        ctx.settings.require_api_key()
    ctx.caps = require_capabilities(ctx.client)      # free catalog requests; aborts on a bad model config
    try:
        lock = HeldLock(settings.db_path, f"project:{project_id}").acquire()
    except LockBusy:
        raise ConfigError(f"project '{project_id}' is already being run by another process") from None
    try:
        engine = Engine(store, project_id, build_specs(ctx.settings), ctx, ctx.log, should_stop)
        if resume:
            reset = engine.prepare_resume(retry_escalated)
            ctx.log.info(f"resume: reset {reset or 'nothing'}")
        try:
            return engine.run()
        finally:    # whatever happened, leave the work done so far (and its status) in output/
            publish_work_files(store, ctx.settings, project_id)
    finally:
        lock.release()


# ---- explicitly recorded human decisions --------------------------------------------------
def _approve_pending(store: MemoryStore, project_id: str, task: str) -> int:
    """Mark the artifacts a review task was blocking as approved by a human."""
    storyboard = store.load(project_id, "storyboard", Storyboard)
    changed = 0
    if task in ("prompt_review", "prompt_engineering") and storyboard:
        for prompt in store.load_all(project_id, "image_prompt", ImagePrompt).values():
            if prompt.status not in APPROVED:
                prompt.status, changed = "human_approved", changed + 1
                store.update_artifact(prompt)
    wanted: list[tuple[str, str]] = []
    if task in ("visual_review", "visual_generation") and storyboard:
        wanted = [("image", shot.id) for shot in storyboard.shots]
    if task in ("audio_qa", "narration_tts"):
        wanted = [("audio", seg.id) for seg in store.load_all(project_id, "narration", NarrationSegment).values()]
    if task in ("thumbnail_review", "thumbnail_generation", "director_thumbnail_approval"):
        brief = store.load(project_id, "thumbnail_brief", ThumbnailBrief)
        wanted = [("thumbnail", brief.id)] if brief else []
    for kind, parent_id in wanted:
        asset = store.latest_asset(project_id, kind, parent_id)
        if asset and asset.status not in APPROVED:
            asset.status, changed = "human_approved", changed + 1
            store.update_asset(asset)
    return changed


def apply_override(store: MemoryStore, project_id: str, task: str, reason: str) -> list[str]:
    """Record a human decision that accepts an escalated task, so `resume` can continue past it."""
    if not reason.strip():
        raise ConfigError("a human override needs a written reason")
    tasks = {t["name"]: t for t in store.tasks(project_id)}
    if task not in tasks:
        raise ConfigError(f"unknown task '{task}'")
    if tasks[task]["state"] != NEEDS_REVIEW:
        raise ConfigError(f"task '{task}' is {tasks[task]['state']}; only NEEDS_REVIEW tasks can be overridden")
    store.add_decision(project_id, "human", task, "override", reason.strip())
    changed = _approve_pending(store, project_id, task)
    # The upstream task that ran out of attempts is released together with its reviewer.
    released = [task] + [d for d in tasks[task]["depends_on"] if tasks[d]["state"] == NEEDS_REVIEW]
    for name in released:
        store.set_task(project_id, name, state=COMPLETED, error="", feedback=[],
                       note=f"human override: {reason.strip()[:200]}",
                       summary=f"accepted by human decision ({changed} artifact(s) approved)")
    return released
