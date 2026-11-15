"""Local production queue and background worker.

A *job* asks for one project to be run (or resumed). Jobs live in the `jobs` table of the
existing SQLite database. One worker process takes them in order, one at a time, and runs
each through the same `run_project` the CLI uses, so there is exactly one pipeline.

Safety rules enforced here (not in any user interface):
  * a live (paid) project can only be queued with an explicit confirmation and an API key;
  * a project can have at most one queued or running job (unique index in SQLite);
  * a repeated submission with the same request id returns the first job instead of a new one;
  * only one worker runs at a time, and a project is never run by two processes at once;
  * every project keeps its own budget, enforced by the orchestrator while it runs.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import studio
from studio.config import ConfigError, Mode, Settings
from studio.memory import MemoryStore
from studio.orchestration.locks import STALE_AFTER_S, HeldLock, LockBusy
from studio.orchestration.pipeline import create_project, run_project
from studio.services.openrouter import OpenRouterError
from studio.utils import new_id, redact

ACTIVE = ("queued", "running")
WORKER_LOCK = "worker"
# How the pipeline's result maps to a job status shown to the user.
RESULT_STATUS = {"completed": "completed", "needs_review": "blocked", "failed": "failed",
                 "stopped": "stopped", "incomplete": "failed"}


class DuplicateJob(ConfigError):
    """The project already has a queued or running job."""


def parse_topics(text: str) -> list[str]:
    """One topic per line; blank lines and repeated topics (ignoring case) are dropped."""
    topics, seen = [], set()
    for line in text.splitlines():
        topic = " ".join(line.split())
        if topic and topic.lower() not in seen:
            seen.add(topic.lower())
            topics.append(topic)
    return topics


def _check_launch(settings: Settings, mode: Mode, budget_usd: float | None, confirmed: bool) -> None:
    if not mode.live:
        return
    if not confirmed:
        raise ConfigError("a paid production must be explicitly confirmed before it is queued")
    settings.require_api_key()
    if budget_usd is not None and budget_usd <= 0:
        raise ConfigError("a paid production needs a budget above 0")


def submit_job(store: MemoryStore, settings: Settings, project_id: str, *, kind: str = "run",
               confirmed: bool = False, retry_escalated: bool = False, request_id: str | None = None,
               batch_id: str = "") -> dict:
    """Queue an existing project. Idempotent per request id; refuses duplicates per project."""
    if kind not in ("run", "resume"):
        raise ConfigError(f"unknown job kind '{kind}'")
    if request_id and (existing := store.job_by_request(request_id)):
        return existing
    project = store.get_project(project_id)
    if project is None:
        raise ConfigError(f"unknown project '{project_id}'")
    _check_launch(settings, Mode(project["mode"]), project["budget_usd"], confirmed)
    try:
        return store.add_job(project_id, kind, confirmed=confirmed, retry_escalated=retry_escalated,
                             request_id=request_id, batch_id=batch_id)
    except sqlite3.IntegrityError:
        if request_id and (existing := store.job_by_request(request_id)):
            return existing                     # the same request arrived twice at the same moment
        raise DuplicateJob(f"project '{project_id}' already has a queued or running job") from None


def create_and_submit(store: MemoryStore, settings: Settings, topic: str, mode: Mode, *,
                      budget_usd: float | None, enable_video: bool = False, confirmed: bool = False,
                      request_id: str | None = None, batch_id: str = "") -> dict:
    """Create a project from user inputs and queue it. Nothing is created without confirmation."""
    if request_id and (existing := store.job_by_request(request_id)):
        return existing                         # e.g. the form was submitted twice
    _check_launch(settings, mode, settings.max_budget_usd if budget_usd is None else budget_usd, confirmed)
    project_id = create_project(settings, store, topic, mode, budget_usd=budget_usd, enable_video=enable_video)
    return submit_job(store, settings, project_id, confirmed=confirmed, request_id=request_id,
                      batch_id=batch_id)


def submit_batch(store: MemoryStore, settings: Settings, topics: list[str], mode: Mode, *,
                 budget_usd: float | None, enable_video: bool = False, confirmed: bool = False,
                 request_id: str | None = None) -> list[dict]:
    """One independent project (with its own budget) and one queued job per topic, in order."""
    if not topics:
        raise ConfigError("enter at least one topic")
    batch_id = request_id or new_id("batch")
    return [create_and_submit(store, settings, topic, mode, budget_usd=budget_usd, enable_video=enable_video,
                              confirmed=confirmed, request_id=f"{batch_id}:{number}", batch_id=batch_id)
            for number, topic in enumerate(topics, start=1)]


def cancel_job(store: MemoryStore, job_id: int) -> bool:
    """Remove a job from the queue. Only jobs that have not started can be cancelled."""
    job = store.get_job(job_id)
    if job is None or job["status"] != "queued":
        return False
    store.finish_job(job_id, "cancelled", "cancelled before it started")
    return True


def request_stop(store: MemoryStore, job_id: int) -> bool:
    """Ask the worker to stop this job after its current task (nothing completed is lost)."""
    job = store.get_job(job_id)
    if job is None or job["status"] != "running":
        return False
    store.set_job_stop(job_id)
    return True


def worker_alive(store: MemoryStore) -> bool:
    return store.lock_alive(WORKER_LOCK, STALE_AFTER_S)


def reconcile(store: MemoryStore, stale_after: float = STALE_AFTER_S) -> list[int]:
    """Repair records left behind by a worker or run that died without cleaning up.

    A job marked running while no live worker exists becomes `interrupted`; a project marked
    running while no live process holds its lock becomes `incomplete`. Both can be resumed.
    """
    interrupted = []
    if not store.lock_alive(WORKER_LOCK, stale_after):
        for job in store.jobs(("running",)):
            store.finish_job(job["id"], "interrupted", "the worker stopped before this job finished; "
                                                       "completed tasks are kept and it can be resumed")
            interrupted.append(job["id"])
    for project in store.list_projects():
        if project["status"] == "running" and not store.lock_alive(f"project:{project['id']}", stale_after):
            store.update_project(project["id"], status="incomplete")
    return interrupted


def _block_reason(store: MemoryStore, project_id: str) -> str:
    for task in store.tasks(project_id):
        if task["state"] in ("FAILED", "NEEDS_REVIEW") and task["error"]:
            return f"{task['name']}: {task['error']}"
    return ""


def run_job(settings: Settings, store: MemoryStore, job: dict,
            client_factory: Callable[[dict], Any] | None = None) -> str:
    """Run one claimed job through the normal pipeline and record how it ended."""
    try:
        result = run_project(
            settings, store, job["project_id"], client=client_factory(job) if client_factory else None,
            confirm_live=bool(job["confirmed"]), resume=job["kind"] == "resume",
            retry_escalated=bool(job["retry_escalated"]),
            should_stop=lambda: bool(store.get_job(job["id"])["stop_requested"]))
        status = RESULT_STATUS.get(result, "failed")
        message = _block_reason(store, job["project_id"])
        if status == "stopped" and store.get_job(job["id"])["stop_requested"]:
            message = "stopped by the operator after the current task; resume to continue"
    except (ConfigError, OpenRouterError) as exc:
        status, message = "failed", str(exc)
    except Exception as exc:                    # a job must never take the worker down with it
        status, message = "failed", f"{type(exc).__name__}: {exc}"
    store.finish_job(job["id"], status, redact(message, settings.openrouter_api_key))
    return status


def run_worker(settings: Settings, store: MemoryStore, *, client_factory: Callable[[dict], Any] | None = None,
               log: Callable[[str], None] = print) -> int:
    """Process the queue, one job at a time, until it is empty. Returns the number of jobs run.

    Only one worker can be active: a second one finds the lock taken and exits at once.
    """
    try:
        lock = HeldLock(settings.db_path, WORKER_LOCK).acquire()
    except LockBusy:
        log("another worker is already running; nothing to do")
        return 0
    processed = 0
    try:
        for job in store.jobs(("running",)):     # left over from a worker that died
            store.finish_job(job["id"], "interrupted", "the previous worker stopped before this job "
                                                       "finished; it can be resumed")
        while (job := store.claim_next_job()) is not None:
            log(f"job {job['id']}: {job['kind']} {job['project_id']}")
            status = run_job(settings, store, job, client_factory)
            log(f"job {job['id']}: {status}")
            processed += 1
    finally:
        lock.release()
    return processed


def start_worker_process(settings: Settings, env_file: str | Path = ".env") -> bool:
    """Start a detached worker process if none is alive. Returns True when one was started.

    The worker is independent of whoever launched it: closing the dashboard does not stop it.
    The command is an argument list (no shell) and carries no secrets; the worker reads its own
    configuration from the environment and the .env file.
    """
    store = MemoryStore(settings.db_path)
    try:
        if worker_alive(store) or not store.jobs(("queued",)):
            return False
    finally:
        store.close()
    package_root = str(Path(studio.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, env.get("PYTHONPATH", "")) if p)
    detach: dict[str, Any] = {"creationflags": 0x00000008 | 0x00000200} if os.name == "nt" \
        else {"start_new_session": True}      # Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.data_dir / "worker.log", "ab") as log:
        subprocess.Popen([sys.executable, "-m", "studio", "--env-file", str(env_file), "worker"],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env, **detach)
    return True
