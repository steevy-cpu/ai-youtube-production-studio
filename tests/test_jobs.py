"""Local job queue, background worker, dashboard launch and recovery actions.

Every production here runs with the sample client; the conftest fixture fails any test that
touches the network, so no paid request can be made.
"""
import os
import sqlite3
import time
from pathlib import Path

import pytest
from conftest import REJECT, needs_ffmpeg

import launcher
import monitor
from studio.config import ConfigError, Mode, Settings, load_settings
from studio.memory import MemoryStore
from studio.orchestration import jobs
from studio.orchestration.locks import HeldLock
from studio.orchestration.pipeline import create_project, run_project
from studio.services.mock import MockOpenRouterClient

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
KEY = "sk-or-v1-test-key-do-not-use"


@pytest.fixture
def no_worker(monkeypatch):
    """Dashboard actions must not spawn real worker processes in tests; record the requests."""
    started = []
    monkeypatch.setattr(launcher, "start_worker", lambda settings, env_file=".env": started.append(1) or True)
    return started


def sample_factory(settings, overrides=None):
    """One sample client shared by every job, standing in for OpenRouter."""
    clients = {}

    def factory(job):
        store = MemoryStore(settings.db_path, read_only=True)
        mode = Mode(store.get_project(job["project_id"])["mode"])
        store.close()
        return clients.setdefault("client", MockOpenRouterClient(settings.for_mode(mode), overrides))
    factory.clients = clients
    return factory


# ---- creating projects from dashboard inputs, mode selection, confirmation ---------------------
def test_dashboard_inputs_create_a_project_and_queue_it(settings, store, no_worker):
    job = launcher.launch(settings, "  Software   Engineering ", Mode.DRY_RUN, 3.5, enable_video=True,
                          confirmed=True, request_id="req-1")
    project = store.get_project(job["project_id"])
    assert project["title"] == "Your Life as Every Rank in Software Engineering"
    assert launcher.title_for(settings, "Cybersecurity") == "Your Life as Every Rank in Cybersecurity"
    assert project["topic"] == "Software Engineering" and project["budget_usd"] == 3.5
    assert project["brief"]["enable_video"] is True and project["status"] == "created"
    assert (job["status"], job["kind"], job["request_id"]) == ("queued", "run", "req-1")
    assert no_worker == [1]                                        # a worker was requested, not run in-page
    assert store.cost_summary(job["project_id"])["requests"] == 0  # creating and queueing costs nothing


def test_mode_selection_uses_the_same_pipeline_with_different_targets(store, no_worker, tmp_path):
    settings = Settings(data_dir=tmp_path / "d", output_dir=tmp_path / "o", openrouter_api_key=KEY)
    choices = launcher.mode_choices(settings)
    assert list(choices) == ["Demo mode (60 seconds, paid)", "Full production mode (600 seconds, paid)",
                             "Offline test (sample media, no API calls, free)"]
    local = MemoryStore(settings.db_path)
    demo = launcher.launch(settings, "Nursing", choices["Demo mode (60 seconds, paid)"], 2.0,
                           enable_video=False, confirmed=True, request_id="demo")
    full = launcher.launch(settings, "Nursing", choices["Full production mode (600 seconds, paid)"], 25.0,
                           enable_video=False, confirmed=True, request_id="full")
    briefs = {j["request_id"]: local.get_project(j["project_id"]) for j in (demo, full)}
    assert (briefs["demo"]["mode"], briefs["demo"]["brief"]["target_duration_s"]) == ("preview", 60)
    assert (briefs["full"]["mode"], briefs["full"]["brief"]["target_duration_s"]) == ("full", 600)
    assert briefs["demo"]["budget_usd"] == 2.0 and briefs["full"]["budget_usd"] == 25.0
    scope = {row["Item"]: row["Value"] for row in launcher.scope(settings, Mode.PREVIEW, enable_video=True)}
    assert scope["Target length"].startswith("60 s") and "hard stop" in scope["Cost"]
    assert scope["AI video clips"].startswith("up to 1")
    assert {r["Item"]: r["Value"] for r in launcher.scope(settings, Mode.FULL, False)}["AI video clips"] == "none"
    assert KEY not in str(launcher.scope(settings, Mode.FULL, True))
    local.close()


def test_paid_launch_needs_confirmation_key_and_budget(settings, store, no_worker):
    def launch(cfg, **changes):
        args = {"enable_video": False, "confirmed": True, "request_id": "r"} | changes
        return launcher.launch(cfg, "Cybersecurity", Mode.PREVIEW, args.pop("budget", 2.0), **args)

    keyed = settings.model_copy(update={"openrouter_api_key": KEY})
    with pytest.raises(ConfigError, match="explicitly confirmed"):
        launch(keyed, confirmed=False)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        launch(settings)                                            # confirmed, but no key configured
    with pytest.raises(ConfigError, match="budget above 0"):
        launch(keyed, budget=0.0)
    assert store.list_projects() == [] and store.jobs() == [] and no_worker == []   # nothing was created

    job = launch(keyed)
    assert job["confirmed"] == 1 and store.get_project(job["project_id"])["mode"] == "preview"
    assert launcher.api_key_configured(keyed) is True and launcher.api_key_configured(settings) is False


# ---- job submission, duplicates, queue order ------------------------------------------------------
def test_duplicate_launches_are_prevented(settings, store, no_worker):
    first = launcher.launch(settings, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                            request_id="same-click")
    again = launcher.launch(settings, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                            request_id="same-click")               # a Streamlit rerun repeating the action
    assert again["id"] == first["id"] and len(store.list_projects()) == 1 and len(store.jobs()) == 1

    with pytest.raises(jobs.DuplicateJob):                          # a second job for a busy project
        jobs.submit_job(store, settings, first["project_id"], request_id="other-click")
    with pytest.raises(sqlite3.IntegrityError):                     # refused by the database itself
        store.add_job(first["project_id"], "run", confirmed=True)
    assert len(store.jobs()) == 1

    batch = launcher.launch_batch(settings, "A topic\nB topic", Mode.DRY_RUN, 1.0, enable_video=False,
                                  confirmed=True, request_id="batch-click")
    repeat = launcher.launch_batch(settings, "A topic\nB topic", Mode.DRY_RUN, 1.0, enable_video=False,
                                   confirmed=True, request_id="batch-click")
    assert [j["id"] for j in repeat] == [j["id"] for j in batch] and len(store.list_projects()) == 3


def test_only_one_worker_and_one_run_per_project(settings, store):
    project_id = create_project(settings, store, "Cybersecurity", Mode.DRY_RUN)
    jobs.submit_job(store, settings, project_id)
    with HeldLock(settings.db_path, jobs.WORKER_LOCK):              # a worker is already alive
        assert jobs.worker_alive(store) is True
        assert jobs.run_worker(settings, store, log=lambda m: None) == 0
        assert jobs.start_worker_process(settings) is False         # no second process is spawned
    assert store.jobs()[0]["status"] == "queued" and jobs.worker_alive(store) is False

    with HeldLock(settings.db_path, f"project:{project_id}"):       # e.g. the CLI is running this project
        with pytest.raises(ConfigError, match="already being run"):
            run_project(settings, store, project_id, client=MockOpenRouterClient(settings))
    assert store.cost_summary(project_id)["requests"] == 0


def test_stale_lock_is_taken_over(settings, store):
    assert store.acquire_lock("worker", "dead-process") is True
    assert store.acquire_lock("worker", "newcomer") is False        # still fresh
    store.conn.execute("UPDATE locks SET heartbeat = heartbeat - 120")
    store.conn.commit()
    assert store.lock_alive("worker") is False and store.acquire_lock("worker", "newcomer") is True


@needs_ffmpeg
def test_batch_runs_in_order_one_at_a_time_with_independent_budgets(settings, store, no_worker):
    assert jobs.parse_topics(" Cybersecurity \n\nSoftware Engineering\ncybersecurity\nMilitary ") == [
        "Cybersecurity", "Software Engineering", "Military"]
    queued = launcher.launch_batch(settings, "Cybersecurity\nSoftware Engineering\nMilitary", Mode.DRY_RUN,
                                   4.0, enable_video=False, confirmed=True, request_id="b1")
    assert [store.get_project(j["project_id"])["topic"] for j in queued] == [
        "Cybersecurity", "Software Engineering", "Military"]
    assert len({j["project_id"] for j in queued}) == 3 and len({j["batch_id"] for j in queued}) == 1
    assert all(store.get_project(j["project_id"])["budget_usd"] == 4.0 for j in queued)
    assert launcher.cancel(settings, queued[1]["id"]) is True       # remove the middle one
    assert launcher.cancel(settings, queued[1]["id"]) is False

    view = monitor.queue(settings)
    assert [r["Project"] for r in view["queued"]] == [queued[0]["project_id"], queued[2]["project_id"]]
    assert view["worker_alive"] is False and len(view["cancelled"]) == 1

    order = []
    assert jobs.run_worker(settings, store, log=order.append) == 2
    assert [line.split()[1].rstrip(":") for line in order if "run " in line] == [
        str(queued[0]["id"]), str(queued[2]["id"])]                 # first in, first out
    finished = {j["id"]: j for j in store.jobs()}
    first, third = finished[queued[0]["id"]], finished[queued[2]["id"]]
    assert first["status"] == third["status"] == "completed" and finished[queued[1]["id"]]["status"] == "cancelled"
    assert first["finished_at"] <= third["started_at"]              # never two at once
    assert store.get_project(queued[1]["project_id"])["status"] == "created"       # the cancelled one never ran
    costs = {j["project_id"]: store.cost_summary(j["project_id"])["requests"] for j in queued}
    assert costs[queued[0]["project_id"]] > 0 and costs[queued[1]["project_id"]] == 0   # recorded per project
    view = monitor.queue(settings)
    assert len(view["completed"]) == 2 and view["queued"] == [] and view["running"] == []


# ---- status retrieval while a job runs --------------------------------------------------------------
@needs_ffmpeg
def test_status_is_read_from_sqlite_while_the_worker_runs(settings, store, no_worker):
    keyed = settings.model_copy(update={"openrouter_api_key": KEY})
    job = launcher.launch(keyed, "Cybersecurity", Mode.PREVIEW, 2.0, enable_video=False, confirmed=True,
                          request_id="live")
    pid = job["project_id"]
    waiting = monitor.load_snapshot(keyed, pid)
    assert monitor.overview(waiting)["done"] == 0 and waiting["jobs"][0]["status"] == "queued"
    rows = {r["id"]: r for r in monitor.agent_activity(waiting)}
    assert len(rows) == 13 and all(r["Status"] == "⏳ pending" for r in rows.values())   # nobody invoked yet

    seen = []

    def factory(j):                                                 # observe the database mid-run
        client = MockOpenRouterClient(keyed.for_mode(Mode.PREVIEW))
        original = client.text_to_speech

        def speak(*args, **kwargs):
            if not seen:
                snap = monitor.load_snapshot(keyed, pid)
                seen.append((monitor.overview(snap), {r["id"]: r for r in monitor.agent_activity(snap)},
                             monitor.queue(keyed)))
            return original(*args, **kwargs)
        client.text_to_speech = speak
        return client

    assert jobs.run_worker(keyed, store, client_factory=factory, log=lambda m: None) == 1
    o, agents, queue = seen[0]
    assert o["status"] == "running" and o["stage"] == "narration_tts" and o["active_agent"] == "Voice Director"
    assert 0 < o["done"] < o["total"] and 0 < o["progress"] < 1
    assert agents["voice_director"]["Status"] == "🔄 running" and agents["seo"]["Status"] == "⏳ pending"
    assert [r["Project"] for r in queue["running"]] == [pid] and queue["worker_alive"] is True

    done = monitor.load_snapshot(keyed, pid)
    assert monitor.overview(done)["status"] == "completed" and done["jobs"][0]["status"] == "completed"
    assert store.get_project(pid)["mode"] == "preview"              # the paid mode ran on the sample client
    details = monitor.agent_details(done, "prompt_judge")
    assert details["made"] and details["runs"] and details["tasks"][0]["Time spent"]
    assert monitor.agent_outputs(keyed, pid, "story_writer")["script"]["chapters"]
    assert monitor.agent_details(done, "visual_generation")["assets"]


def test_queue_read_reports_a_locked_database(settings, store):
    writer = sqlite3.connect(settings.db_path)
    writer.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(monitor.DatabaseBusy):
            monitor._read(settings, lambda s: s.jobs(), attempts=2, timeout=0.05)
    finally:
        writer.rollback()
        writer.close()
    assert monitor.queue(settings)["jobs"] == []


# ---- stopping and recovery ---------------------------------------------------------------------------
@needs_ffmpeg
def test_stop_request_ends_the_job_cleanly_and_it_can_be_resumed(settings, store, no_worker):
    job = launcher.launch(settings, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                          request_id="stop-me")

    def factory(j):
        client = MockOpenRouterClient(settings.for_mode(Mode.DRY_RUN))
        original = client.text_to_speech

        def speak(*args, **kwargs):
            launcher.stop_after_current_task(settings, job["id"])   # the operator asks to stop mid-run
            return original(*args, **kwargs)
        client.text_to_speech = speak
        return client

    jobs.run_worker(settings, store, client_factory=factory, log=lambda m: None)
    stopped = store.get_job(job["id"])
    assert stopped["status"] == "stopped" and "stopped by the operator" in stopped["message"]
    tasks = {t["name"]: t["state"] for t in store.tasks(job["project_id"])}
    assert tasks["narration_tts"] == "COMPLETED" and tasks["audio_qa"] == "PENDING"   # the task in progress finished

    snapshot = monitor.load_snapshot(settings, job["project_id"])
    options = launcher.recovery_options(snapshot["project"], snapshot["tasks"], snapshot["jobs"])
    assert list(options["actions"]) == ["resume"]
    with pytest.raises(ConfigError, match="explicit confirmation"):
        launcher.recover(settings, job["project_id"], "resume", confirmed=False, request_id="r1")
    resumed = launcher.recover(settings, job["project_id"], "resume", confirmed=True, request_id="r1")
    assert resumed["kind"] == "resume" and len(store.jobs()) == 2
    images = len(store.assets(job["project_id"], "image"))
    jobs.run_worker(settings, store, log=lambda m: None)
    assert store.get_job(resumed["id"])["status"] == "completed"
    assert len(store.assets(job["project_id"], "image")) == images  # approved images were reused


def test_dead_worker_leaves_a_resumable_job(settings, store):
    project_id = create_project(settings, store, "Cybersecurity", Mode.DRY_RUN)
    job = jobs.submit_job(store, settings, project_id)
    store.claim_next_job()                                          # a worker took it ...
    store.update_project(project_id, status="running")              # ... and died without cleaning up
    assert jobs.reconcile(store) == [job["id"]]
    assert store.get_job(job["id"])["status"] == "interrupted" and "can be resumed" in store.get_job(job["id"])["message"]
    assert store.get_project(project_id)["status"] == "incomplete"
    snapshot = monitor.load_snapshot(settings, project_id)
    assert "resume" in launcher.recovery_options(snapshot["project"], snapshot["tasks"], snapshot["jobs"])["actions"]


@needs_ffmpeg
def test_failed_project_is_recovered_through_the_queue(settings, store, no_worker):
    factory = sample_factory(settings, {"seo.package": ["not json"] * 3})
    job = launcher.launch(settings, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                          request_id="will-fail")
    jobs.run_worker(settings, store, client_factory=factory, log=lambda m: None)
    failed = store.get_job(job["id"])
    assert failed["status"] == "failed" and failed["message"].startswith("seo_package: AgentOutputError")
    view = monitor.queue(settings)
    assert [r["Project"] for r in view["problem"]] == [job["project_id"]]
    snapshot = monitor.load_snapshot(settings, job["project_id"])
    assert "AgentOutputError" in monitor.blockers(snapshot)[0]["reason"]             # full reason available

    resumed = launcher.recover(settings, job["project_id"], "resume", confirmed=True, request_id="fix",
                               new_budget=7.0)
    with pytest.raises(jobs.DuplicateJob):                          # no second recovery while one is queued
        launcher.recover(settings, job["project_id"], "resume", confirmed=True, request_id="fix-again")
    writer_calls = factory.clients["client"].calls.count("writer.chapter")
    jobs.run_worker(settings, store, client_factory=factory, log=lambda m: None)
    assert store.get_job(resumed["id"])["status"] == "completed"
    assert store.get_project(job["project_id"])["budget_usd"] == 7.0
    assert factory.clients["client"].calls.count("writer.chapter") == writer_calls   # nothing was redone


@needs_ffmpeg
def test_blocked_project_needs_a_recorded_human_decision(settings, store, no_worker):
    def judge(context):
        return REJECT if context["purpose"] == "thumbnail" else {"approved": True, "score": 0.9}
    cfg = settings.model_copy(update={"max_thumbnail_attempts": 2})
    factory = sample_factory(cfg, {"judge.image": [judge] * 50})
    job = launcher.launch(cfg, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                          request_id="blocked")
    jobs.run_worker(cfg, store, client_factory=factory, log=lambda m: None)
    assert store.get_job(job["id"])["status"] == "blocked"
    snapshot = monitor.load_snapshot(cfg, job["project_id"])
    options = launcher.recovery_options(snapshot["project"], snapshot["tasks"], snapshot["jobs"])
    assert set(options["actions"]) == {"retry", "override"} and options["override_tasks"] == ["thumbnail_review"]

    with pytest.raises(ConfigError, match="written reason"):        # no silent approval
        launcher.recover(cfg, job["project_id"], "override", confirmed=True, request_id="o1",
                         task="thumbnail_review", reason=" ")
    assert store.decisions(job["project_id"], actor="human") == [] and len(store.jobs()) == 1
    accepted = launcher.recover(cfg, job["project_id"], "override", confirmed=True, request_id="o2",
                                task="thumbnail_review", reason="good enough for the class demo")
    jobs.run_worker(cfg, store, client_factory=factory, log=lambda m: None)
    assert store.get_job(accepted["id"])["status"] == "completed"
    assert store.decisions(job["project_id"], actor="human")[0]["reason"] == "good enough for the class demo"


@needs_ffmpeg
def test_compliance_block_cannot_be_accepted_from_the_dashboard(settings, store, no_worker):
    blocked = {"findings": [{"check": "misleading_claims", "status": "fail", "detail": "false statistic"}]}
    factory = sample_factory(settings, {"compliance.screen": [blocked]})
    job = launcher.launch(settings, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                          request_id="c")
    jobs.run_worker(settings, store, client_factory=factory, log=lambda m: None)
    snapshot = monitor.load_snapshot(settings, job["project_id"])
    options = launcher.recovery_options(snapshot["project"], snapshot["tasks"], snapshot["jobs"])
    assert "override" not in options["actions"] and options["cli_only"] == ["compliance"]
    with pytest.raises(ConfigError, match="command line"):
        launcher.recover(settings, job["project_id"], "override", confirmed=True, request_id="x",
                         task="compliance", reason="trust me")
    assert store.decisions(job["project_id"], actor="human") == []
    assert store.get_task(job["project_id"], "compliance")["state"] == "NEEDS_REVIEW"


# ---- the real detached worker process ---------------------------------------------------------------------
@needs_ffmpeg
def test_detached_worker_process_runs_the_queue(tmp_path, monkeypatch):
    for name, value in {"DATA_DIR": tmp_path / "data", "OUTPUT_DIR": tmp_path / "output",
                        "TARGET_DURATION_S": 24, "MIN_CHAPTERS": 1, "MAX_CHAPTERS": 2, "TARGET_SHOT_SECONDS": 6,
                        "DRY_RUN_WIDTH": 320, "DRY_RUN_HEIGHT": 180, "DRY_RUN_FPS": 10,
                        "OPENROUTER_API_KEY": KEY}.items():
        monkeypatch.setenv(name, str(value))
    settings = load_settings(env_file=None)
    store = MemoryStore(settings.db_path)
    job = jobs.create_and_submit(store, settings, "Cybersecurity", Mode.DRY_RUN, budget_usd=1.0, confirmed=True)
    assert jobs.start_worker_process(settings, env_file=tmp_path / "none.env") is True
    deadline = time.monotonic() + 120
    while store.get_job(job["id"])["status"] in jobs.ACTIVE and time.monotonic() < deadline:
        time.sleep(0.5)                                             # the page thread would simply re-render
    assert store.get_job(job["id"])["status"] == "completed"
    assert (settings.output_dir / job["project_id"] / "video.mp4").is_file()
    while jobs.worker_alive(store) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert jobs.worker_alive(store) is False                        # it exits when the queue is empty
    log = (settings.data_dir / "worker.log").read_text()
    assert "completed" in log and KEY not in log
    assert KEY not in (settings.data_dir / "projects" / job["project_id"] / "logs" / "run.log").read_text()
    store.close()


# ---- the Streamlit page -----------------------------------------------------------------------------------
def open_app(monkeypatch, settings, view, key=""):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(settings.output_dir))
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    for name in ("TARGET_DURATION_S", "MIN_CHAPTERS", "MAX_CHAPTERS"):
        monkeypatch.setenv(name, str(getattr(settings, name.lower())))
    app = testing.AppTest.from_file(APP, default_timeout=120).run()
    app = app.sidebar.radio[0].set_value(view).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def page_text(app) -> str:
    return " ".join(str(e.value) for group in (app.markdown, app.caption, app.info, app.warning, app.success,
                                               app.error) for e in group)


def test_page_starts_a_production_and_opens_its_monitor(settings, store, no_worker, monkeypatch):
    app = open_app(monkeypatch, settings, "New production")
    assert app.button(key="new_start").disabled                    # no topic yet
    app.text_input(key="new_topic").set_value("Cybersecurity")
    app.radio(key="new_mode").set_value("Offline test (sample media, no API calls, free)")
    app = app.run()
    assert "Your Life as Every Rank in Cybersecurity" in page_text(app)
    assert not app.button(key="new_start").disabled
    app = app.button(key="new_start").click().run()
    assert not app.exception
    project, = store.list_projects()
    assert project["mode"] == "dry_run" and [j["status"] for j in store.jobs()] == ["queued"] and no_worker == [1]
    assert app.sidebar.radio[0].value == "Monitor" and app.sidebar.selectbox[0].value == project["id"]
    assert {m.label: m.value for m in app.metric}["Tasks completed"] == "0/0"      # its monitor, real data only
    app = app.run()                                                 # a plain rerun must not launch again
    assert len(store.list_projects()) == 1 and len(store.jobs()) == 1 and no_worker == [1]


def test_page_requires_confirmation_for_paid_modes_and_hides_the_key(settings, store, no_worker, monkeypatch):
    app = open_app(monkeypatch, settings, "New production", key=KEY)
    app.text_input(key="new_topic").set_value("Cybersecurity")
    app = app.run()                                                 # Demo mode (paid) is the default
    assert app.radio(key="new_mode").value.startswith("Demo mode")
    assert app.button(key="new_start").disabled and "configured" in page_text(app)
    assert KEY not in page_text(app) and KEY not in str([t.value for t in app.table])
    app.checkbox(key="new_confirm").check()
    app = app.run()
    assert not app.button(key="new_start").disabled
    app = app.button(key="new_start").click().run()
    job, = store.jobs()
    assert job["confirmed"] == 1 and store.get_project(job["project_id"])["mode"] == "preview"
    assert store.cost_summary(job["project_id"])["requests"] == 0  # queued only; nothing was sent

    app = open_app(monkeypatch, settings, "New production", key="")
    app.text_input(key="new_topic").set_value("Nursing")
    app = app.run()
    app.checkbox(key="new_confirm").check()
    app = app.run()
    assert app.button(key="new_start").disabled and "MISSING" in page_text(app)     # no key, no paid launch


def test_page_queues_a_batch_and_shows_the_queue(settings, store, no_worker, monkeypatch):
    app = open_app(monkeypatch, settings, "Batch & queue")
    app.text_area(key="batch_topics").set_value("Cybersecurity\nSoftware Engineering\nMilitary")
    app.radio(key="batch_mode").set_value("Offline test (sample media, no API calls, free)")
    app = app.run()
    assert "3 independent production(s)" in page_text(app)
    app = app.button(key="batch_start").click().run()
    assert not app.exception
    assert [store.get_project(j["project_id"])["topic"] for j in store.jobs()] == [
        "Cybersecurity", "Software Engineering", "Military"]
    queued = next(d.value for d in app.dataframe if "Position" in d.value.columns)
    assert list(queued["Position"]) == [1, 2, 3] and set(queued["Status"]) == {"queued"}
    app = app.run()
    assert len(store.jobs()) == 3                                   # reruns do not queue it again


@needs_ffmpeg
def test_page_recovery_needs_explicit_confirmation(settings, store, no_worker, monkeypatch):
    factory = sample_factory(settings, {"seo.package": ["not json"] * 3})
    job = launcher.launch(settings, "Cybersecurity", Mode.DRY_RUN, 1.0, enable_video=False, confirmed=True,
                          request_id="will-fail")
    jobs.run_worker(settings, store, client_factory=factory, log=lambda m: None)
    app = open_app(monkeypatch, settings, "Recovery")
    assert "AgentOutputError" in " ".join(t.value for t in app.text)                # the full failure reason
    assert app.button(key="recover_start").disabled and len(store.jobs()) == 1
    app.checkbox(key="recover_confirm").check()
    app = app.run()
    app = app.button(key="recover_start").click().run()
    assert not app.exception and [j["kind"] for j in store.jobs()] == ["run", "resume"]
    assert app.sidebar.radio[0].value == "Monitor"
