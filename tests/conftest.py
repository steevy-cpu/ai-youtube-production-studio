"""Shared fixtures. No test makes a network request: OpenRouter is always mocked."""
import pytest

from studio.config import Mode, Settings
from studio.memory import MemoryStore
from studio.orchestration.pipeline import create_project, make_context, run_project
from studio.rendering.ffmpeg import ffmpeg_available
from studio.services.mock import MockOpenRouterClient

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not installed")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if anything tries to reach the network through httpx's default transport."""
    import httpx

    def blocked(self, request, *args, **kwargs):
        raise AssertionError(f"unexpected real network request to {request.url}")
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data", output_dir=tmp_path / "output", references_dir=tmp_path / "refs",
        target_duration_s=24, min_chapters=1, max_chapters=2, target_shot_seconds=6,
        dry_run_width=320, dry_run_height=180, dry_run_fps=10, dry_run_demo_rejections=False)


@pytest.fixture
def store(settings):
    memory = MemoryStore(settings.db_path)
    yield memory
    memory.close()


@pytest.fixture
def project(settings, store):
    """Factory: create a dry-run project and return (project_id, mock_client, run)."""
    def make(overrides=None, **setting_changes):
        cfg = settings.model_copy(update=setting_changes)
        project_id = create_project(cfg, store, "Cybersecurity", Mode.DRY_RUN)
        client = MockOpenRouterClient(cfg.for_mode(Mode.DRY_RUN), overrides)

        def run(**kwargs):
            return run_project(cfg, store, project_id, client=client, **kwargs)
        return project_id, client, run
    return make


@pytest.fixture
def ctx(settings, store):
    project_id = create_project(settings, store, "Cybersecurity", Mode.DRY_RUN)
    context = make_context(settings, store, project_id)
    context.task = "test"
    return context


REJECT = {"approved": False, "score": 0.2, "issues": ["subject is wrong"], "suggested_changes": ["fix it"]}
