"""Dashboard: read-only data layer (monitor.py) and the Streamlit page (app.py).

Projects are produced with the dry-run sample client; nothing here reaches the network.
"""
import hashlib
import sqlite3
from pathlib import Path

import pytest
from conftest import REJECT, needs_ffmpeg

import monitor
from studio.memory import MemoryStore

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
pytestmark = needs_ffmpeg


def digest(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def completed(project, store, settings):
    """A finished production whose sample judges rejected one prompt, one image and one thumbnail."""
    seen = []

    def judge(context):
        seen.append(context["purpose"])
        return REJECT if context["purpose"] == "image" and seen.count("image") == 1 else {
            "approved": True, "score": 0.9}
    project_id, client, run = project({"judge.image": [judge] * 50, "director.thumbnail": [
        {"approved": False, "reasons": ["character is off-model"]}]}, dry_run_demo_rejections=True)
    assert run() == "completed"
    return project_id


@pytest.fixture
def blocked(project, store, settings):
    """A production stopped at Final QA after rendering (narration longer than estimated)."""
    project_id, client, run = project()
    client.settings = client.settings.model_copy(update={"words_per_minute": 100})
    assert run() == "needs_review"
    return project_id


# ---- data layer -----------------------------------------------------------------------------
def test_overview_of_a_completed_project(completed, store, settings):
    snapshot = monitor.load_snapshot(settings, completed)
    o = monitor.overview(snapshot)
    assert (o["done"], o["total"], o["progress"], o["stage"]) == (29, 29, 1.0, "finished")
    assert o["status"] == "completed" and o["mode"] == "Dry run (sample data)" and not o["running"]
    assert o["images"] == len(store.assets(completed, "image")) and o["images_approved"] == o["images"] - 1
    assert o["narration"] == len(store.assets(completed, "audio")) and o["clips"] == 0
    assert o["cost_usd"] == 0 and o["unknown_cost_requests"] == 0 and o["elapsed_s"] >= 0
    assert o["sample_media"] is True and monitor.blockers(snapshot) == []
    assert monitor.format_duration(3725) == "1:02:05"


def test_agent_activity_reports_recorded_task_state(completed, blocked, settings):
    listed = monitor.agent_activity(monitor.load_snapshot(settings, completed))
    assert [r["Agent"] for r in listed[:13]] == [
        "Director", "Researcher", "Story Writer", "Scene Director", "Prompt Engineer", "Prompt Judge",
        "Visual Generation", "Visual Judge", "Voice Director", "Editor", "Final QA", "Compliance", "SEO & Growth"]
    rows = {r["id"]: r for r in listed}
    assert all(r["Status"] == "✅ completed" and r["Last updated"] != "-" for r in rows.values())
    assert rows["visual_generation"]["Tasks done"] == "4/4" and rows["director"]["Time spent"] != "-"

    rows = {r["id"]: r for r in monitor.agent_activity(monitor.load_snapshot(settings, blocked))}
    assert rows["final_qa"]["Status"] == "⚠️ needs review" and rows["final_qa"]["Attempt"] == "1/1"
    assert "accepted range" in rows["final_qa"]["Latest result or failure"]
    # Not invoked yet -> Pending, with no result and no time; never shown as active.
    assert rows["seo"]["Status"] == "⏳ pending" and rows["seo"]["Current task"] == "seo_package"
    assert rows["seo"]["Latest result or failure"] == "-" and rows["seo"]["Time spent"] == "-"
    assert rows["compliance"]["Status"] == "⏳ pending"
    # Invoked earlier, more work ahead -> waiting, not running.
    assert rows["director"]["Status"] == "⏸️ waiting for its next task"
    assert rows["editor"]["Status"] == "✅ completed"
    assert not any("running" in r["Status"] for r in rows.values())


def test_event_feed_is_chronological_and_complete(completed, store, settings):
    store.add_error(completed, "visual_generation", "visual_generation",
                    "OpenRouterError: HTTP 429 (rate limited): Rate limit exceeded")      # test setup only
    feed = monitor.event_feed(monitor.load_snapshot(settings, completed))
    kinds = {row["Event"] for row in feed}
    assert {"Task started", "Task completed", "Prompt rejected", "Image rejected", "Thumbnail rejected",
            "Regeneration requested", "Quality check passed", "Quality check failed", "API error", "Decision",
            "Project completed"} <= kinds
    times = [row["Time"] for row in feed if row["Event"] != "API error"]
    assert times == sorted(times) and feed[0]["Event"] == "Task started"
    assert [row["Event"] for row in feed if row["Event"] != "API error"][-1] == "Project completed"
    assert sum(row["Event"] == "Project completed" for row in feed) == 1
    rejected = next(row for row in feed if row["Event"] == "Image rejected")
    assert rejected["Subject"] == "visual_review" and rejected["Agent"] == "visual_judge"
    regeneration = [row for row in feed if row["Event"] == "Regeneration requested"]
    assert {"prompt_engineering", "visual_generation", "thumbnail_prompt"} <= {r["Subject"] for r in regeneration}


def test_blocked_project_shows_reason_next_action_and_rendered_video(blocked, settings):
    snapshot = monitor.load_snapshot(settings, blocked)
    o = monitor.overview(snapshot)
    assert o["status"] == "needs_review" and o["stage"] == "final_qa" and o["done"] == 18
    block, = monitor.blockers(snapshot)
    assert block["task"] == "final_qa" and "accepted range is 22-26s" in block["reason"]
    assert block["commands"][0] == f"python run.py resume {blocked} --retry-escalated"
    assert f"python run.py override {blocked} final_qa" in block["commands"][1]
    found = monitor.media(snapshot, settings)                      # visible although export never ran
    assert found["video"]["state"] == "preliminary" and Path(found["video"]["path"]).is_file()
    assert found["thumbnail"] is None and len(found["images"]) == o["images"] and found["audio"]


def test_other_blocking_states_have_specific_actions(project, store, settings):
    project_id, client, run = project(max_media_requests=2)
    snapshot = monitor.load_snapshot(settings, project_id)
    assert monitor.blockers(snapshot)[0]["commands"] == [f"python run.py run {project_id}"]   # not started
    assert monitor.overview(snapshot)["elapsed_s"] == 0 and monitor.event_feed(snapshot) == []
    assert run() == "stopped"
    block, = monitor.blockers(monitor.load_snapshot(settings, project_id))
    assert block["state"] == "failed" and "MAX_MEDIA_REQUESTS" in block["action"]
    assert monitor.load_snapshot(settings, "no-such-project") is None


def test_dashboard_reads_are_strictly_read_only(completed, settings):
    before = digest(settings.db_path)
    snapshot = monitor.load_snapshot(settings, completed)
    monitor.overview(snapshot), monitor.agent_activity(snapshot), monitor.event_feed(snapshot)
    monitor.media(snapshot, settings), monitor.list_projects(settings)
    assert digest(settings.db_path) == before
    reader = MemoryStore(settings.db_path, read_only=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.update_project(completed, status="running")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.add_decision(completed, "human", "compliance", "override", "from the dashboard")
    reader.close()
    assert digest(settings.db_path) == before


def test_locked_database_is_reported_not_crashed(completed, settings):
    writer = sqlite3.connect(settings.db_path)
    writer.execute("BEGIN EXCLUSIVE")                              # the pipeline in the middle of a write
    try:
        with pytest.raises(monitor.DatabaseBusy):
            monitor._read(settings, lambda s: s.list_projects(), attempts=2, timeout=0.05)
    finally:
        writer.rollback()
        writer.close()
    assert [p["id"] for p in monitor.list_projects(settings)] == [completed]


def test_no_database_means_no_projects(settings):
    assert monitor.list_projects(settings) == []
    assert not settings.db_path.exists()                           # and none was created


# ---- the Streamlit page -------------------------------------------------------------------------
def run_app(monkeypatch, settings):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(settings.output_dir))
    app = streamlit_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def metrics(app) -> dict:
    return {m.label: m.value for m in app.metric}


def test_page_without_projects(monkeypatch, settings):
    app = run_app(monkeypatch, settings)
    assert "No projects yet" in app.info[0].value and not app.metric


def test_page_for_a_completed_project(completed, monkeypatch, settings, store):
    before = digest(settings.db_path)
    app = run_app(monkeypatch, settings)
    assert app.sidebar.selectbox[0].value == completed
    shown = metrics(app)
    assert shown["Tasks completed"] == "29/29" and shown["Active agent"] == "none"
    assert shown["Mode"] == "Dry run (sample data)" and shown["Current stage"] == "finished"
    assert shown["Images"] == str(len(store.assets(completed, "image")))
    assert {"Elapsed time", "Recorded cost (USD)", "Unknown-cost requests", "Video clips", "Voiceovers",
            "Retries", "Rejections"} <= set(shown)
    assert int(shown["Retries"]) >= 3 and int(shown["Rejections"]) >= 3
    assert app.progress[0].value == 100 and "COMPLETED" in app.success[0].value
    tables = [d.value for d in app.dataframe]
    agents = tables[0]
    assert list(agents["Agent"])[:2] == ["Director", "Researcher"] and len(agents) == 18
    events = next(t for t in tables if "Event" in t.columns)
    assert {"Project completed", "Image rejected"} <= set(events["Event"])
    assert events.iloc[0]["Event"] == "Project completed"                            # newest first
    failed_checks = next(t for t in tables if "Suggested changes" in t.columns and "Evaluator" in t.columns)
    assert "FAILED" in set(failed_checks["Result"])                                  # detailed evaluation messages
    assert not app.button                                           # the Monitor view cannot change anything
    assert digest(settings.db_path) == before

    app.sidebar.radio[0].set_value("Media").run()                   # media viewer
    assert not app.exception
    assert len(app.get("video")) == 1 and len(app.get("audio")) == len(store.assets(completed, "audio"))
    assert any("Final video - APPROVED" in s.value for s in app.success)
    assert any("Thumbnail - APPROVED" in s.value for s in app.success)
    assert len(app.get("imgs")) + len(app.get("image")) >= len(store.assets(completed, "image", status=("approved",)))


def test_page_for_a_project_blocked_at_final_qa(blocked, monkeypatch, settings):
    before = digest(settings.db_path)
    app = run_app(monkeypatch, settings)
    shown = metrics(app)
    assert shown["Tasks completed"] == "18/29" and shown["Current stage"] == "final_qa"
    warnings = " ".join(w.value for w in app.warning)
    assert "NEEDS REVIEW" in warnings and "Blocked at `final_qa`" in warnings
    assert "accepted range is 22-26s" in warnings and "Next action" in warnings
    assert any(f"python run.py resume {blocked} --retry-escalated" in c.value for c in app.code)
    assert digest(settings.db_path) == before
    app.sidebar.radio[0].set_value("Media").run()
    warnings = " ".join(w.value for w in app.warning)
    assert "PRELIMINARY, not yet approved" in warnings and len(app.get("video")) == 1   # shown before export
    assert any("No thumbnail" in i.value for i in app.info)


def test_page_lets_you_choose_between_projects(completed, blocked, monkeypatch, settings):
    app = run_app(monkeypatch, settings)
    box = app.sidebar.selectbox[0]
    assert set(box.options) and {completed, blocked} == {o.split(" · ")[-1] for o in box.options}
    app = box.select(completed).run()
    assert not app.exception and metrics(app)["Tasks completed"] == "29/29"
    app = app.sidebar.selectbox[0].select(blocked).run()
    assert not app.exception and metrics(app)["Tasks completed"] == "18/29"


def test_paired_escalation_is_shown_once_with_the_reviewer_override(project, store, settings):
    def judge(context):
        return REJECT if context["purpose"] == "thumbnail" else {"approved": True, "score": 0.9}
    project_id, client, run = project({"judge.image": [judge] * 50}, max_thumbnail_attempts=2)
    assert run() == "needs_review"
    snapshot = monitor.load_snapshot(settings, project_id)
    block, = monitor.blockers(snapshot)                             # two tasks, one blocker
    assert block["task"] == "thumbnail_generation` and `thumbnail_review"
    assert "used all 2 attempts" in block["reason"]
    assert f"override {project_id} thumbnail_review " in block["commands"][1]
    found = monitor.media(snapshot, settings)
    assert found["thumbnail"] is None and [c.status for c in found["thumbnail_candidates"]] == ["rejected"] * 2
    assert found["video"]["state"] == "approved"                    # Director approved; export pending
