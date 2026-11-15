"""Run context shared by task handlers and agents: settings, memory, client, budget, paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studio.config import Mode, Settings
from studio.memory import MemoryStore
from studio.orchestration.engine import StopRun
from studio.services.openrouter import Usage
from studio.utils import safe_path

MEDIA_KINDS = ("image", "video", "tts")


class BudgetExceeded(StopRun):
    """Raised before a request that would break the budget or the media request cap."""


class Budget:
    """Checks the recorded spend before every external request and records usage after it."""

    def __init__(self, store: MemoryStore, project_id: str, max_media_requests: int, live: bool):
        self.store, self.project_id = store, project_id
        self.max_media_requests, self.live = max_media_requests, live

    @property
    def limit_usd(self) -> float:
        return float(self.store.get_project(self.project_id)["budget_usd"])

    def check(self, kind: str) -> None:
        summary = self.store.cost_summary(self.project_id)
        if self.live and summary["known_usd"] >= self.limit_usd:
            raise BudgetExceeded(f"budget reached: ${summary['known_usd']:.4f} recorded of "
                                 f"${self.limit_usd:.2f} allowed; refusing further '{kind}' requests. "
                                 "Resume with a higher --budget to continue.")
        if kind in MEDIA_KINDS and summary["media_requests"] >= self.max_media_requests:
            raise BudgetExceeded(f"media request cap reached ({self.max_media_requests}); refusing further "
                                 f"'{kind}' requests. Raise MAX_MEDIA_REQUESTS and resume to continue.")

    def record(self, task: str, agent: str, kind: str, usage: Usage) -> None:
        self.store.add_cost(self.project_id, task, agent, usage.model, kind, usage.cost_usd,
                            usage.note or ("" if usage.cost_usd is not None else "cost not reported"))


@dataclass
class ProjectDirs:
    root: Path
    output: Path

    def sub(self, name: str) -> Path:
        path = safe_path(self.root, name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def create(cls, settings: Settings, project_id: str) -> "ProjectDirs":
        root = safe_path(settings.data_dir / "projects", project_id)
        output = safe_path(settings.output_dir, project_id)
        root.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        return cls(root, output)


@dataclass
class Ctx:
    settings: Settings          # already adjusted for the run mode
    store: MemoryStore
    client: Any                 # OpenRouterClient or MockOpenRouterClient
    project_id: str
    mode: Mode
    dirs: ProjectDirs
    budget: Budget
    log: Any
    caps: dict[str, Any] = field(default_factory=dict)
    task: str = ""              # name of the task currently executing (set per handler)

    @property
    def project(self) -> dict:
        return self.store.get_project(self.project_id)
