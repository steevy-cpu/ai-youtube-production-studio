"""Deterministic workflow engine: explicit task states, dependencies, bounded retries,
feedback routing, checkpointing in SQLite and resume. No agent framework involved.

A task handler is a plain function `handler(ctx, feedback) -> summary`. It signals a
rejection by raising `Rejection`, naming the upstream task that must redo work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from studio.memory import MemoryStore
from studio.schemas import RevisionRequest
from studio.utils import now_iso

PENDING, RUNNING, COMPLETED = "PENDING", "RUNNING", "COMPLETED"
REJECTED, RETRYING, FAILED, NEEDS_REVIEW = "REJECTED", "RETRYING", "FAILED", "NEEDS_REVIEW"


class Rejection(Exception):
    """A reviewer rejected work. `target_task` is re-run with the requests as feedback."""

    def __init__(self, target_task: str, requests: list[RevisionRequest], reason: str = ""):
        super().__init__(reason or f"rejected; revision requested from {target_task}")
        self.target_task = target_task
        self.requests = requests
        self.reason = reason or str(self)


class Escalation(Exception):
    """The task cannot continue without a human decision."""


class StopRun(Exception):
    """Abort the whole run immediately (for example: budget exhausted)."""


@dataclass
class TaskSpec:
    name: str
    agent: str                       # responsible agent, for traceability
    handler: Callable[[Any, list[RevisionRequest]], str | None]
    depends_on: list[str] = field(default_factory=list)
    max_attempts: int = 1


class Engine:
    def __init__(self, store: MemoryStore, project_id: str, specs: list[TaskSpec], ctx: Any, log=None,
                 should_stop: Callable[[], bool] | None = None):
        self.store, self.project_id, self.ctx, self.log = store, project_id, ctx, log
        self.should_stop = should_stop        # polled between tasks; lets an operator stop cleanly
        self.specs = {spec.name: spec for spec in specs}
        names = set()
        for seq, spec in enumerate(specs):
            unknown = [d for d in spec.depends_on if d not in names]
            if unknown:
                raise ValueError(f"task {spec.name} depends on undefined/later tasks: {unknown}")
            names.add(spec.name)
            store.upsert_task(project_id, spec.name, seq, spec.agent, spec.max_attempts, spec.depends_on)

    # ---- helpers ------------------------------------------------------------
    def _info(self, message: str) -> None:
        if self.log:
            self.log.info(message)

    def _tasks(self) -> dict[str, dict]:
        return {t["name"]: t for t in self.store.tasks(self.project_id) if t["name"] in self.specs}

    def _downstream(self, name: str) -> list[str]:
        """Every task that depends on `name`, directly or transitively."""
        found: list[str] = []
        for spec in self.specs.values():          # specs are in dependency order
            if name in spec.depends_on or any(d in found for d in spec.depends_on):
                found.append(spec.name)
        return found

    def _next_runnable(self) -> dict | None:
        tasks = self._tasks()
        for task in tasks.values():
            if task["state"] in (PENDING, RETRYING) and all(
                    tasks[d]["state"] == COMPLETED for d in task["depends_on"]):
                return task
        return None

    # ---- public API ----------------------------------------------------------
    def prepare_resume(self, retry_escalated: bool = False) -> list[str]:
        """Make interrupted/failed tasks runnable again. Completed work is kept."""
        reset = []
        for task in self._tasks().values():
            if task["state"] in (RUNNING, FAILED, REJECTED) or (
                    retry_escalated and task["state"] == NEEDS_REVIEW):
                self.store.set_task(self.project_id, task["name"], state=PENDING, note="reset by resume",
                                    attempts=0, error="")
                reset.append(task["name"])
        return reset

    def run(self) -> str:
        """Run every runnable task. Returns the resulting project status."""
        self.store.update_project(self.project_id, status="running")
        try:
            while (task := self._next_runnable()) is not None:
                if self.should_stop is not None and self.should_stop():
                    raise StopRun("stop requested by the operator; completed tasks are kept")
                self._execute(task)
        except StopRun as stop:
            self.store.update_project(self.project_id, status="stopped")
            self._info(f"run stopped: {stop}")
            return "stopped"
        status = self.status()
        self.store.update_project(self.project_id, status=status)
        return status

    def status(self) -> str:
        states = [t["state"] for t in self._tasks().values()]
        if all(s == COMPLETED for s in states):
            return "completed"
        if NEEDS_REVIEW in states:
            return "needs_review"
        if FAILED in states:
            return "failed"
        return "incomplete"

    # ---- execution -------------------------------------------------------------
    def _execute(self, task: dict) -> None:
        name, spec, pid = task["name"], self.specs[task["name"]], self.project_id
        attempt = task["attempts"] + 1
        feedback = [RevisionRequest.model_validate(f) for f in task["feedback"]]
        self.store.set_task(pid, name, state=RUNNING, note=f"attempt {attempt}", attempts=attempt,
                            started_at=now_iso(), error="")
        self.store.remember(pid, "current_task", name, table="working_memory")
        if hasattr(self.ctx, "task"):
            self.ctx.task = name
        self._info(f"[{name}] RUNNING (attempt {attempt}/{spec.max_attempts}, agent {spec.agent})")
        try:
            summary = spec.handler(self.ctx, feedback) or ""
        except Rejection as rejection:
            self._handle_rejection(name, rejection)
        except Escalation as exc:
            self.store.add_error(pid, name, spec.agent, f"escalated: {exc}")
            self.store.set_task(pid, name, state=NEEDS_REVIEW, note=str(exc)[:300], error=str(exc),
                                finished_at=now_iso())
            self._info(f"[{name}] NEEDS_REVIEW: {exc}")
        except StopRun as exc:
            self.store.add_error(pid, name, spec.agent, str(exc))
            self.store.set_task(pid, name, state=FAILED, note=str(exc)[:300], error=str(exc),
                                finished_at=now_iso())
            raise
        except Exception as exc:  # any handler failure is recorded; other branches keep running
            message = f"{type(exc).__name__}: {exc}"
            self.store.add_error(pid, name, spec.agent, message)
            self.store.set_task(pid, name, state=FAILED, note=message[:300], error=message,
                                finished_at=now_iso())
            self._info(f"[{name}] FAILED: {message}")
        else:
            self.store.resolve_revisions(pid, name)
            self.store.set_task(pid, name, state=COMPLETED, note=summary[:300], summary=summary,
                                feedback=[], finished_at=now_iso())
            self._info(f"[{name}] COMPLETED {summary}")

    def _handle_rejection(self, name: str, rejection: Rejection) -> None:
        pid, target = self.project_id, rejection.target_task
        if target not in self.specs:
            raise ValueError(f"task {name} rejected towards unknown task {target}")
        for request in rejection.requests:
            request.project_id = pid
            self.store.add_revision(request)
        self.store.set_task(pid, name, state=REJECTED, note=rejection.reason[:300])
        target_task = self._tasks()[target]
        self._info(f"[{name}] REJECTED -> {target} must revise: {rejection.reason}")

        if target_task["attempts"] >= target_task["max_attempts"]:
            message = (f"{target} used all {target_task['max_attempts']} attempts and {name} still rejects: "
                       f"{rejection.reason}. Escalated for human review.")
            self.store.add_error(pid, name, self.specs[target].agent, message)
            for task_name in dict.fromkeys([target, name]):
                self.store.set_task(pid, task_name, state=NEEDS_REVIEW, note="attempts exhausted",
                                    error=message, finished_at=now_iso())
            self._info(f"[{name}] NEEDS_REVIEW: {message}")
            return

        feedback = [r.model_dump() for r in rejection.requests]
        self.store.set_task(pid, target, state=RETRYING, note=f"revision requested by {name}",
                            feedback=feedback)
        tasks = self._tasks()
        for task_name in self._downstream(target):
            if tasks[task_name]["state"] != PENDING:
                self.store.set_task(pid, task_name, state=PENDING, attempts=0,
                                    note=f"waiting for {target} revision")
