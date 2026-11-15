"""SQLite memory persistence and the orchestration engine's state machine."""
import pytest

from studio.memory import MemoryStore
from studio.orchestration.context import Budget, BudgetExceeded
from studio.orchestration.engine import Engine, Escalation, Rejection, TaskSpec
from studio.schemas import Chapter, EvaluationResult, MediaAsset, RevisionRequest
from studio.services.openrouter import Usage


def _request(task="write"):
    return RevisionRequest(parent_id="x", target_kind="thing", target_task=task, responsible_agent="writer",
                           requested_by="reviewer", issues=["too short"], suggested_changes=["make it longer"])


def _new_project(store, budget=1.0):
    store.create_project("p1", "Topic", "Title", "dry_run", budget, {})
    return "p1"


# ---- memory ---------------------------------------------------------------------
def test_memory_persists_across_connections(tmp_path):
    path = tmp_path / "studio.db"
    store = MemoryStore(path)
    pid = _new_project(store)
    first = store.save_artifact(pid, "chapter", Chapter(index=1, title="A", narration="one two three"), key="1")
    second = store.save_artifact(pid, "chapter", Chapter(index=1, title="A", narration="four five six"), key="1")
    store.add_asset(MediaAsset(project_id=pid, parent_id="shot", kind="image", path="/tmp/x.png",
                               provenance="test", status="approved"))
    store.add_evaluation(EvaluationResult(project_id=pid, parent_id="shot", target_kind="image",
                                          evaluator="visual_judge", passed=False, score=0.4))
    store.add_cost(pid, "t", "agent", "model-a", "llm", 0.25)
    store.add_cost(pid, "t", "agent", "model-b", "tts", None)
    store.remember(pid, "tone", "playful")
    store.add_limitation(pid, "no references")
    store.close()

    reopened = MemoryStore(path)
    assert (first.version, second.version) == (1, 2)
    assert reopened.load(pid, "chapter", Chapter, key="1").narration == "four five six"   # latest version
    assert [v["version"] for v in reopened.artifact_versions(pid, "chapter", "1")] == [1, 2]
    asset = reopened.latest_asset(pid, "image", "shot", ("approved",))
    assert asset.path == "/tmp/x.png" and asset.provenance == "test"
    assert reopened.evaluation_summary(pid)[0]["passed"] == 0
    cost = reopened.cost_summary(pid)
    assert cost["known_usd"] == 0.25 and cost["unknown_cost_requests"] == 1 and cost["media_requests"] == 1
    assert reopened.recall(pid, "tone") == "playful"
    assert "no references" in reopened.progress_summary(pid)
    reopened.close()


def test_media_bytes_are_not_stored_in_sqlite(store):
    pid = _new_project(store)
    store.add_asset(MediaAsset(project_id=pid, kind="image", path="/somewhere/img.png", provenance="x"))
    columns = [r["name"] for r in store.conn.execute("PRAGMA table_info(assets)")]
    assert "path" in columns and not any(c in columns for c in ("blob", "bytes", "content"))


# ---- engine ----------------------------------------------------------------------
def _engine(store, handlers, max_write=3):
    pid = _new_project(store)
    specs = [TaskSpec("write", "writer", handlers["write"], [], max_write),
             TaskSpec("review", "reviewer", handlers["review"], ["write"]),
             TaskSpec("publish", "publisher", handlers.get("publish", lambda c, f: "ok"), ["review"])]
    return pid, Engine(store, pid, specs, ctx=None)


def test_dependencies_and_state_transitions(store):
    order = []
    pid, engine = _engine(store, {"write": lambda c, f: order.append("write"),
                                  "review": lambda c, f: order.append("review"),
                                  "publish": lambda c, f: order.append("publish")})
    assert engine.run() == "completed" and order == ["write", "review", "publish"]
    transitions = [(e["from_state"], e["to_state"]) for e in store.task_events(pid, "write")]
    assert transitions == [("PENDING", "RUNNING"), ("RUNNING", "COMPLETED")]
    with pytest.raises(ValueError, match="undefined"):
        Engine(store, pid, [TaskSpec("b", "x", lambda c, f: None, ["missing"])], ctx=None)


def test_rejection_routes_feedback_to_the_responsible_task(store):
    seen = []

    def write(ctx, feedback):
        seen.append([r.issues for r in feedback])

    def review(ctx, feedback):
        if len(seen) < 2:
            raise Rejection("write", [_request()], "too short")

    pid, engine = _engine(store, {"write": write, "review": review})
    assert engine.run() == "completed"
    assert seen == [[], [["too short"]]]                       # second pass received the feedback
    states = [e["to_state"] for e in store.task_events(pid, "write")]
    assert states == ["RUNNING", "COMPLETED", "RETRYING", "RUNNING", "COMPLETED"]
    assert "REJECTED" in [e["to_state"] for e in store.task_events(pid, "review")]
    assert [r["status"] for r in store.revisions(pid)] == ["resolved"]
    assert store.get_task(pid, "write")["attempts"] == 2


def test_retries_are_bounded_and_escalate(store):
    calls = []

    def review(ctx, feedback):
        raise Rejection("write", [_request()], "never good enough")

    pid, engine = _engine(store, {"write": lambda c, f: calls.append(1), "review": review}, max_write=3)
    assert engine.run() == "needs_review"
    assert len(calls) == 3                                      # exactly max_attempts, no infinite loop
    assert store.get_task(pid, "write")["state"] == "NEEDS_REVIEW"
    assert store.get_task(pid, "review")["state"] == "NEEDS_REVIEW"
    assert store.get_task(pid, "publish")["state"] == "PENDING"  # never ran
    assert "Escalated" in store.errors(pid)[-1]["message"]


def test_failed_task_recovery_keeps_completed_work(store):
    runs = {"write": 0, "review": 0}

    def write(ctx, feedback):
        runs["write"] += 1

    def review(ctx, feedback):
        runs["review"] += 1
        if runs["review"] == 1:
            raise RuntimeError("temporary outage")

    pid, engine = _engine(store, {"write": write, "review": review})
    assert engine.run() == "failed"
    assert store.get_task(pid, "review")["state"] == "FAILED"
    assert "temporary outage" in store.errors(pid)[0]["message"]

    assert engine.prepare_resume() == ["review"]
    assert engine.run() == "completed"
    assert runs == {"write": 1, "review": 2}                    # completed task was not repeated


def test_escalation_marks_needs_review(store):
    def review(ctx, feedback):
        raise Escalation("a human must decide")
    pid, engine = _engine(store, {"write": lambda c, f: None, "review": review})
    assert engine.run() == "needs_review"
    assert store.get_task(pid, "review")["error"] == "a human must decide"
    assert engine.prepare_resume() == []                        # not retried unless explicitly asked
    assert engine.prepare_resume(retry_escalated=True) == ["review"]


# ---- budget ----------------------------------------------------------------------
def test_budget_blocks_requests_once_the_limit_is_reached(store):
    pid = _new_project(store, budget=0.50)
    budget = Budget(store, pid, max_media_requests=2, live=True)
    budget.check("llm")
    budget.record("task", "agent", "llm", Usage(model="m", cost_usd=0.30))
    budget.check("image")
    budget.record("task", "agent", "image", Usage(model="m", cost_usd=0.25))
    with pytest.raises(BudgetExceeded, match="budget reached"):
        budget.check("llm")
    store.update_project(pid, budget_usd=5.0)                   # raising the budget allows continuing
    budget.check("llm")
    budget.record("task", "agent", "tts", Usage(model="m", cost_usd=None))   # unknown cost
    with pytest.raises(BudgetExceeded, match="media request cap"):
        budget.check("image")
    assert store.cost_summary(pid)["unknown_cost_requests"] == 1


def test_budget_stop_halts_the_run(store):
    def write(ctx, feedback):
        raise BudgetExceeded("budget reached")
    pid, engine = _engine(store, {"write": write, "review": lambda c, f: None})
    assert engine.run() == "stopped"
    assert store.get_task(pid, "write")["state"] == "FAILED"
    assert store.get_task(pid, "review")["state"] == "PENDING"
