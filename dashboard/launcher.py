"""Dashboard actions: create, queue, recover. No Streamlit code in this module.

This is a thin layer over `studio.orchestration.jobs` (the local queue and worker) and the
existing resume / override functions. It never runs a production itself: it writes a job to
SQLite and makes sure a separate worker process is running. All rules about confirmation,
duplicates and budgets are enforced in the core package, not here.
"""
from __future__ import annotations

import math
from pathlib import Path

from studio.config import ConfigError, Mode, Settings
from studio.memory import MemoryStore
from studio.orchestration import jobs
from studio.orchestration.pipeline import apply_override

# A compliance block can only be lifted deliberately from the command line, never from here.
CLI_ONLY_OVERRIDES = ("compliance", "metadata_compliance")
RECOVERABLE = ("failed", "needs_review", "stopped", "incomplete")


def mode_choices(settings: Settings) -> dict[str, Mode]:
    return {
        f"Demo mode ({settings.for_mode(Mode.PREVIEW).target_duration_s:.0f} seconds, paid)": Mode.PREVIEW,
        f"Full production mode ({settings.target_duration_s:.0f} seconds, paid)": Mode.FULL,
        "Offline test (sample media, no API calls, free)": Mode.DRY_RUN}


def title_for(settings: Settings, topic: str) -> str:
    topic = " ".join(topic.split())
    return settings.format_template.format(topic=topic) if topic else ""


def api_key_configured(settings: Settings) -> bool:
    """Whether a key exists. The key itself is never returned or displayed."""
    return bool(settings.openrouter_api_key.strip())


def scope(settings: Settings, mode: Mode, enable_video: bool) -> list[dict]:
    """What one production of this mode involves. Counts are estimates; retries add to them."""
    s = settings.for_mode(mode)
    shots = math.ceil(s.target_duration_s / s.target_shot_seconds)
    scenes = math.ceil(shots / 2.5)
    chapters = (s.min_chapters + s.max_chapters) / 2
    images = shots + s.max_reference_images + 1
    calls = round(16 + 2 * chapters + 2 * scenes + shots)
    clips = (s.max_video_clips if (enable_video or s.enable_video) else 0)
    live = mode.live
    return [
        {"Item": "Target length", "Value": f"{s.target_duration_s:.0f} s at {s.width}x{s.height}"},
        {"Item": "Script", "Value": f"about {s.target_words} words in {s.min_chapters}-{s.max_chapters} chapters"},
        {"Item": "Images", "Value": f"about {images} requests ({shots} shots, references, thumbnail)"
                                    + (f" - {s.image_model}" if live else " - local placeholders")},
        {"Item": "Narration", "Value": f"about {scenes} speech requests"
                                       + (f" - {s.tts_model}, voice {s.tts_voice}" if live else " - sample tone")},
        {"Item": "AI video clips", "Value": f"up to {clips} - {s.video_model}" if clips else "none"},
        {"Item": "Text and vision calls", "Value": f"roughly {calls}"
                                                   + (f" - {s.llm_model}" if live else " - sample answers")},
        {"Item": "Cost", "Value": "not estimated in advance; the budget below is a hard stop enforced by "
                                  "the orchestrator before every request" if live else "none (no API calls)"}]


def housekeeping(settings: Settings) -> None:
    """Make sure the queue tables exist and repair records left by a worker that died."""
    store = MemoryStore(settings.db_path)
    try:
        jobs.reconcile(store)
    finally:
        store.close()


def start_worker(settings: Settings, env_file: str | Path = ".env") -> bool:
    return jobs.start_worker_process(settings, env_file)


def launch(settings: Settings, topic: str, mode: Mode, budget_usd: float, *, enable_video: bool,
           confirmed: bool, request_id: str, env_file: str | Path = ".env") -> dict:
    """Create one project and queue it. Returns the job (the same job if called twice)."""
    store = MemoryStore(settings.db_path)
    try:
        job = jobs.create_and_submit(store, settings, topic, mode, budget_usd=budget_usd,
                                     enable_video=enable_video, confirmed=confirmed, request_id=request_id)
    finally:
        store.close()
    start_worker(settings, env_file)
    return job


def launch_batch(settings: Settings, topics_text: str, mode: Mode, budget_usd: float, *, enable_video: bool,
                 confirmed: bool, request_id: str, env_file: str | Path = ".env") -> list[dict]:
    """One independent project and job per topic, queued in the order entered."""
    store = MemoryStore(settings.db_path)
    try:
        queued = jobs.submit_batch(store, settings, jobs.parse_topics(topics_text), mode, budget_usd=budget_usd,
                                   enable_video=enable_video, confirmed=confirmed, request_id=request_id)
    finally:
        store.close()
    start_worker(settings, env_file)
    return queued


def recovery_options(project: dict, tasks: list[dict], project_jobs: list[dict]) -> dict:
    """Which recovery actions make sense for this project right now (no side effects)."""
    if any(j["status"] in jobs.ACTIVE for j in project_jobs):
        return {"busy": True, "actions": {}, "override_tasks": []}
    review = [t["name"] for t in tasks if t["state"] == "NEEDS_REVIEW"]
    failed = [t["name"] for t in tasks if t["state"] == "FAILED"]
    actions = {}
    if failed or (project["status"] in ("stopped", "incomplete") and not review):
        actions["resume"] = "Resume from where it stopped (completed tasks and approved assets are kept)"
    if review:
        actions["retry"] = "Retry the task(s) waiting for review with fresh attempts"
    # The reviewer is the later of the escalated tasks; overriding it releases its upstream task too.
    allowed = [t for t in review if t not in CLI_ONLY_OVERRIDES]
    if allowed:
        actions["override"] = "Accept the current result as a recorded human decision, then continue"
    return {"busy": False, "actions": actions, "override_tasks": allowed[-1:],
            "cli_only": [t for t in review if t in CLI_ONLY_OVERRIDES]}


def recover(settings: Settings, project_id: str, action: str, *, confirmed: bool, request_id: str,
            new_budget: float | None = None, task: str | None = None, reason: str = "",
            env_file: str | Path = ".env") -> dict:
    """Queue a recovery job through the existing resume / escalation logic.

    Nothing happens without `confirmed`. Rejected work is never approved automatically: an
    override needs a written reason and is stored as a human decision, and compliance blocks
    cannot be overridden from the dashboard at all.
    """
    if not confirmed:
        raise ConfigError("recovery actions need an explicit confirmation")
    if action not in ("resume", "retry", "override"):
        raise ConfigError(f"unknown recovery action '{action}'")
    store = MemoryStore(settings.db_path)
    try:
        if store.job_by_request(request_id):
            return store.job_by_request(request_id)
        project = store.get_project(project_id)
        if project is None:
            raise ConfigError(f"unknown project '{project_id}'")
        if store.jobs(jobs.ACTIVE, project_id):
            raise jobs.DuplicateJob(f"project '{project_id}' already has a queued or running job")
        if Mode(project["mode"]).live:
            settings.require_api_key()          # fail before anything is recorded
        if action == "override":
            if not task or task in CLI_ONLY_OVERRIDES:
                raise ConfigError("compliance decisions can only be recorded from the command line "
                                  "(python run.py override ...)")
            apply_override(store, project_id, task, reason)
        if new_budget is not None and new_budget != project["budget_usd"]:
            if new_budget <= 0:
                raise ConfigError("the budget must be above 0")
            store.update_project(project_id, budget_usd=new_budget)
        job = jobs.submit_job(store, settings, project_id, kind="resume", confirmed=True,
                              retry_escalated=action == "retry", request_id=request_id)
    finally:
        store.close()
    start_worker(settings, env_file)
    return job


def cancel(settings: Settings, job_id: int) -> bool:
    store = MemoryStore(settings.db_path)
    try:
        return jobs.cancel_job(store, job_id)
    finally:
        store.close()


def stop_after_current_task(settings: Settings, job_id: int) -> bool:
    store = MemoryStore(settings.db_path)
    try:
        return jobs.request_stop(store, job_id)
    finally:
        store.close()
