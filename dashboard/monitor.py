"""Read-only data access for the dashboard. No Streamlit code in this module.

Everything shown by the dashboard comes from here, and everything here comes from the
existing SQLite memory (opened read-only), the run log and the media files on disk.
Nothing is written, started, retried or approved here; actions live in launcher.py.
"""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from studio.config import Settings
from studio.memory import APPROVED, MemoryStore

STATE_LABEL = {"PENDING": "⏳ pending", "RUNNING": "🔄 running", "COMPLETED": "✅ completed",
               "FAILED": "❌ failed", "RETRYING": "↩️ retrying", "REJECTED": "↩️ rejected",
               "NEEDS_REVIEW": "⚠️ needs review"}
MODE_LABEL = {"dry_run": "Dry run (sample data)", "preview": "Live preview", "full": "Full production"}
REJECTION_LABEL = {
    "prompt_review": "Prompt rejected", "thumbnail_prompt_review": "Prompt rejected",
    "visual_review": "Image rejected", "thumbnail_review": "Thumbnail rejected",
    "director_thumbnail_approval": "Thumbnail rejected", "script_guard": "Script rejected",
    "director_script_approval": "Script rejected", "timeline_guard": "Storyboard rejected",
    "audio_qa": "Narration rejected"}
# Which state of an agent's tasks is the most informative one to show as its "current task".
STATE_PRIORITY = ["RUNNING", "NEEDS_REVIEW", "FAILED", "RETRYING", "REJECTED"]
# Every agent of the studio, in pipeline order, followed by the deterministic services.
AGENTS = {"director": "Director", "research": "Researcher", "story_writer": "Story Writer",
          "scene_director": "Scene Director", "prompt_engineer": "Prompt Engineer",
          "prompt_judge": "Prompt Judge", "visual_generation": "Visual Generation",
          "visual_judge": "Visual Judge", "voice_director": "Voice Director", "editor": "Editor",
          "final_qa": "Final QA", "compliance": "Compliance", "seo": "SEO & Growth"}
SERVICES = {"script_guard": "Script Guard", "timeline_guard": "Timeline Guard", "audio_qa": "Audio QA",
            "renderer": "FFmpeg Renderer", "orchestrator": "Export"}
# Which stored records belong to which agent (used for the per-agent detail view).
EVALUATOR_AGENT = {"prompt_judge": "prompt_judge", "prompt_guard": "prompt_judge", "visual_judge": "visual_judge",
                   "media_guard": "visual_judge", "seo_thumbnail_review": "seo", "final_video_guard": "final_qa",
                   "final_av_check": "final_qa", "final_qa_continuity": "final_qa", "audio_qa": "audio_qa",
                   "script_guard": "script_guard", "timeline_guard": "timeline_guard"}
WORK_OWNER = {"prompt": "prompt_engineer", "thumbnail_prompt": "prompt_engineer", "image": "visual_generation",
              "thumbnail": "visual_generation", "reference": "visual_generation", "video": "visual_generation",
              "audio": "voice_director", "script": "story_writer", "storyboard": "scene_director",
              "final_video": "editor"}
AGENT_IO = {  # agent -> (what it reads, artifact kinds it writes)
    "director": ("project brief; later the script, QA and compliance reports, thumbnail reviews",
                 ["plan", "style_bible"]),
    "research": ("topic, production plan, local source files", ["research"]),
    "story_writer": ("production plan, research brief, style bible, reviewer feedback", ["script"]),
    "scene_director": ("approved script, style bible", ["storyboard"]),
    "prompt_engineer": ("storyboard shots, style bible, thumbnail brief, judge feedback",
                        ["image_prompt", "animation_prompt", "thumbnail_prompt"]),
    "prompt_judge": ("shots and their prompts, style bible, thumbnail brief", []),
    "visual_generation": ("approved prompts, style bible, character references, judge feedback", []),
    "visual_judge": ("generated images, their prompts, style bible", []),
    "voice_director": ("storyboard narration, production plan", ["narration"]),
    "editor": ("storyboard, approved images/clips, measured narration audio", ["timeline"]),
    "final_qa": ("rendered video, timeline, storyboard, script", []),
    "compliance": ("script, research claims, asset records, SEO metadata", ["compliance"]),
    "seo": ("approved script, production plan, thumbnail candidates", ["seo", "thumbnail_brief"]),
    "orchestrator": ("all approved outputs", ["report"])}
STALE_AFTER_S = 15 * 60


class DatabaseBusy(Exception):
    """The database could not be read right now (the pipeline is writing). Try again shortly."""


def _parse(timestamp: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp) if timestamp else None
    except ValueError:
        return None


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def list_projects(settings: Settings) -> list[dict]:
    """All projects in the local database, newest first. Empty when no database exists yet."""
    if not settings.db_path.is_file():
        return []
    return _read(settings, lambda store: store.list_projects())


def _read(settings: Settings, query, attempts: int = 3, timeout: float = 2.0) -> Any:
    """Run `query(store)` on a read-only connection inside one transaction (a consistent view).

    A writer holding the database for a moment is normal while a production runs, so a locked
    database is retried a few times and then reported as DatabaseBusy instead of crashing.
    """
    last = ""
    for attempt in range(attempts):
        store = None
        try:
            store = MemoryStore(settings.db_path, read_only=True, timeout=timeout)
            store.conn.execute("BEGIN")
            return query(store)
        except sqlite3.OperationalError as exc:
            last = str(exc)
            time.sleep(0.2 * (attempt + 1))
        finally:
            if store is not None:
                store.close()
    raise DatabaseBusy(last)


def load_snapshot(settings: Settings, project_id: str, timeout: float = 2.0) -> dict | None:
    """Everything the live part of the dashboard needs for one project, or None if it is unknown."""
    def query(store: MemoryStore) -> dict | None:
        project = store.get_project(project_id)
        if project is None:
            return None
        return {"project": project, "tasks": store.tasks(project_id), "events": store.task_events(project_id),
                "runs": store.agent_runs(project_id), "errors": store.errors(project_id),
                "evaluations": store.evaluations(project_id), "revisions": store.revisions(project_id),
                "decisions": store.decisions(project_id), "cost": store.cost_summary(project_id),
                "assets": store.assets(project_id), "jobs": store.jobs(project_id=project_id),
                "limitations": store.recall(project_id, "limitations", [])}
    snapshot = _read(settings, query, timeout=timeout)
    if snapshot is not None:
        snapshot["log_tail"] = log_tail(settings, project_id)
    return snapshot


def log_tail(settings: Settings, project_id: str, lines: int = 40) -> list[str]:
    path = settings.data_dir / "projects" / project_id / "logs" / "run.log"
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


# ---- overview -----------------------------------------------------------------------
def overview(snapshot: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    project, tasks, assets = snapshot["project"], snapshot["tasks"], snapshot["assets"]
    done = sum(t["state"] == "COMPLETED" for t in tasks)
    current = next((t for t in tasks if t["state"] == "RUNNING"), None) or \
        next((t for t in tasks if t["state"] != "COMPLETED"), None)
    running = project["status"] == "running"

    # Execution time = total time tasks actually spent RUNNING, from the recorded state changes.
    elapsed, open_since = 0.0, {}
    for event in snapshot["events"]:
        when = _parse(event["created_at"])
        if when is None:
            continue
        if event["to_state"] == "RUNNING":
            open_since[event["task"]] = when
        elif event["from_state"] == "RUNNING" and event["task"] in open_since:
            elapsed += (when - open_since.pop(event["task"])).total_seconds()
    if running:                                  # a task still executing counts up to now
        elapsed += sum((now - since).total_seconds() for since in open_since.values())

    active = next((t for t in tasks if t["state"] == "RUNNING"), None)
    completed = [e for e in snapshot["events"] if e["to_state"] == "COMPLETED"]
    stamps = [e["created_at"] for e in snapshot["events"]] + [r["created_at"] for r in snapshot["runs"]]
    last = _parse(max(stamps)) if stamps else None
    idle = (now - last).total_seconds() if last else None
    kinds = Counter(a.kind for a in assets)
    approved = Counter(a.kind for a in assets if a.status in APPROVED)
    return {
        "id": project["id"], "title": project["title"], "mode": MODE_LABEL.get(project["mode"], project["mode"]),
        "status": project["status"], "final_approval": project["final_approval"], "running": running,
        "stage": current["name"] if current else "finished", "done": done, "total": len(tasks),
        "active_agent": label(active["agent"]) if active else "none",
        "latest_result": f"{completed[-1]['task']}: {completed[-1]['note']}" if completed else "nothing yet",
        "retries": sum(e["to_state"] == "RETRYING" for e in snapshot["events"]),
        "rejections": sum(e["to_state"] == "REJECTED" for e in snapshot["events"]),
        "problem_tasks": [t["name"] for t in tasks if t["state"] in ("FAILED", "NEEDS_REVIEW", "REJECTED")],
        "progress": done / len(tasks) if tasks else 0.0, "elapsed_s": elapsed,
        "last_activity": last, "idle_s": idle, "stale": bool(running and idle and idle > STALE_AFTER_S),
        "cost_usd": snapshot["cost"]["known_usd"], "budget_usd": project["budget_usd"],
        "unknown_cost_requests": snapshot["cost"]["unknown_cost_requests"],
        "requests": snapshot["cost"]["requests"],
        "images": kinds["image"], "images_approved": approved["image"],
        "clips": kinds["video"], "clips_approved": approved["video"],
        "narration": kinds["audio"], "narration_approved": approved["audio"],
        "sample_media": any(a.is_sample for a in assets)}


# ---- agents ---------------------------------------------------------------------------
def label(agent: str) -> str:
    return AGENTS.get(agent) or SERVICES.get(agent) or agent


def task_durations(snapshot: dict) -> dict[str, float]:
    """Seconds each task has spent RUNNING (all attempts), from the recorded state changes."""
    totals: dict[str, float] = {}
    started: dict[str, datetime] = {}
    for event in snapshot["events"]:
        when = _parse(event["created_at"])
        if when is None:
            continue
        if event["to_state"] == "RUNNING":
            started[event["task"]] = when
        elif event["from_state"] == "RUNNING" and event["task"] in started:
            totals[event["task"]] = totals.get(event["task"], 0.0) + \
                (when - started.pop(event["task"])).total_seconds()
    return totals


def agent_activity(snapshot: dict) -> list[dict]:
    """One row per agent, every agent always listed. An agent that has not been invoked yet is
    Pending; nothing is shown as active unless a task record says it is running."""
    by_agent: dict[str, list[dict]] = {}
    for task in snapshot["tasks"]:
        by_agent.setdefault(task["agent"], []).append(task)
    invoked_tasks = {e["task"] for e in snapshot["events"] if e["to_state"] == "RUNNING"}
    last_event: dict[str, str] = {e["task"]: e["created_at"] for e in snapshot["events"]}
    last_run: dict[str, str] = {r["agent"]: r["created_at"] for r in snapshot["runs"]}
    durations = task_durations(snapshot)

    rows = []
    for agent in list(AGENTS) + [a for a in SERVICES if a in by_agent] + \
            [a for a in by_agent if a not in AGENTS and a not in SERVICES]:
        tasks = by_agent.get(agent, [])
        invoked = any(t["name"] in invoked_tasks for t in tasks) or agent in last_run
        urgent = next((t for state in STATE_PRIORITY for t in tasks if t["state"] == state), None)
        done = [t for t in tasks if t["state"] == "COMPLETED"]
        pending = [t for t in tasks if t["state"] == "PENDING"]
        if urgent:
            current, status = urgent, STATE_LABEL[urgent["state"]]
        elif not invoked:
            current, status = (pending[0] if pending else None), "⏳ pending"
        elif pending:
            current, status = pending[0], "⏸️ waiting for its next task"
        else:
            current, status = (done[-1] if done else None), "✅ completed"
        if urgent and urgent["state"] in ("FAILED", "NEEDS_REVIEW"):
            result = urgent["error"]
        elif urgent and urgent["state"] == "RUNNING":
            result = "working..."
        else:
            result = done[-1]["summary"] if done and invoked else ""
        stamps = [s for s in ([last_event.get(t["name"]) for t in tasks] + [last_run.get(agent)]) if s]
        rows.append({
            "Agent": label(agent), "Kind": "agent" if agent in AGENTS else "service", "id": agent,
            "Current task": current["name"] if current else "-", "Status": status,
            "Attempt": f"{current['attempts']}/{current['max_attempts']}" if current else "-",
            "Latest result or failure": result or "-",
            "Tasks done": f"{len(done)}/{len(tasks)}",
            "Time spent": format_duration(sum(durations.get(t["name"], 0.0) for t in tasks)) if invoked else "-",
            "Last updated": max(stamps) if stamps else "-"})
    return rows


def agent_details(snapshot: dict, agent: str) -> dict:
    """Everything recorded about one agent: tasks, model calls, evaluations, feedback, errors."""
    durations = task_durations(snapshot)
    tasks = [t for t in snapshot["tasks"] if t["agent"] == agent]

    def evaluation_row(e) -> dict:
        return {"Time": e.created_at, "Evaluator": e.evaluator, "Target": e.target_kind,
                "Result": "passed" if e.passed else "FAILED", "Score": "" if e.score is None else f"{e.score:.2f}",
                "Issues": "; ".join(e.issues), "Suggested changes": "; ".join(e.suggested_changes),
                "Explanation": e.explanation}
    return {
        "reads": AGENT_IO.get(agent, ("", []))[0], "writes": AGENT_IO.get(agent, ("", []))[1],
        "tasks": [{"Task": t["name"], "Status": STATE_LABEL[t["state"]],
                   "Attempts": f"{t['attempts']}/{t['max_attempts']}",
                   "Time spent": format_duration(durations.get(t["name"], 0.0)),
                   "Result or failure": (t["error"] if t["state"] in ("FAILED", "NEEDS_REVIEW") else t["summary"])
                   or "-", "Pending feedback": len(t["feedback"])} for t in tasks],
        "runs": [{"Time": r["created_at"], "Task": r["task"], "Action": r["action"], "Model": r["model"] or "-",
                  "Status": r["status"], "Seconds": r["duration_s"], "Detail": r["detail"]}
                 for r in snapshot["runs"] if r["agent"] == agent],
        "made": [evaluation_row(e) for e in snapshot["evaluations"] if EVALUATOR_AGENT.get(e.evaluator) == agent],
        "received": [evaluation_row(e) for e in snapshot["evaluations"]
                     if WORK_OWNER.get(e.target_kind) == agent and EVALUATOR_AGENT.get(e.evaluator) != agent],
        "revisions": [{"Time": r["created_at"], "Requested by": r["requested_by"], "For": r["responsible_agent"],
                       "About": r["target_kind"], "Task to redo": r["target_task"], "Status": r["status"]}
                      for r in snapshot["revisions"] if agent in (r["responsible_agent"], r["requested_by"])],
        "decisions": [d for d in snapshot["decisions"] if d["actor"] == agent],
        "errors": [{"Time": e["created_at"], "Task": e["task"], "Message": e["message"]}
                   for e in snapshot["errors"] if e["agent"] == agent],
        "assets": [{"File": Path(a.path).name, "Kind": a.kind, "Status": a.status, "Model": a.model or "-",
                    "Provenance": a.provenance} for a in snapshot["assets"]
                   if (agent == "visual_generation" and a.kind in ("image", "thumbnail", "reference", "video"))
                   or (agent == "voice_director" and a.kind == "audio")
                   or (agent == "renderer" and a.kind == "final_video")]}


def agent_outputs(settings: Settings, project_id: str, agent: str) -> dict[str, Any]:
    """Latest stored artifacts written by one agent (read on demand; they can be large)."""
    import json
    kinds = AGENT_IO.get(agent, ("", []))[1]
    if not kinds:
        return {}

    def query(store: MemoryStore) -> dict[str, Any]:
        found: dict[str, list] = {}
        for row in store.latest_artifacts(project_id):
            if row["kind"] in kinds:
                found.setdefault(row["kind"], []).append(json.loads(row["data"]))
        return {kind: records[0] if len(records) == 1 else records for kind, records in found.items()}
    return _read(settings, query)


def evaluation_messages(snapshot: dict, failed_only: bool = True) -> list[dict]:
    """Detailed judge and guard messages, newest first."""
    rows = [{"Time": e.created_at, "Evaluator": e.evaluator, "Target": e.target_kind,
             "Result": "passed" if e.passed else "FAILED", "Score": "" if e.score is None else f"{e.score:.2f}",
             "Issues": "; ".join(e.issues) or "-", "Suggested changes": "; ".join(e.suggested_changes) or "-",
             "Explanation": e.explanation}
            for e in snapshot["evaluations"] if not (failed_only and e.passed)]
    return rows[::-1]


def queue(settings: Settings) -> dict:
    """The local job queue grouped for display, plus whether a worker is alive."""
    def query(store: MemoryStore) -> dict:
        titles = {p["id"]: p for p in store.list_projects()}
        from studio.orchestration.jobs import worker_alive
        rows = []
        for job in store.jobs():
            project = titles.get(job["project_id"], {})
            rows.append({"Job": job["id"], "Status": job["status"], "Project": job["project_id"],
                         "Mode": MODE_LABEL.get(project.get("mode", ""), "-"),
                         "Budget (USD)": project.get("budget_usd"), "Kind": job["kind"],
                         "Batch": job["batch_id"] or "-", "Queued": job["created_at"],
                         "Started": job["started_at"] or "-", "Finished": job["finished_at"] or "-",
                         "Message": job["message"] or "-"})
        return {"worker_alive": worker_alive(store), "jobs": rows}
    result = _read(settings, query)
    rows = result["jobs"]
    result["queued"] = [r for r in rows if r["Status"] == "queued"]
    result["running"] = [r for r in rows if r["Status"] == "running"]
    result["completed"] = [r for r in rows if r["Status"] == "completed"][::-1]
    result["problem"] = [r for r in rows if r["Status"] in ("failed", "blocked", "stopped", "interrupted")][::-1]
    result["cancelled"] = [r for r in rows if r["Status"] == "cancelled"][::-1]
    return result


# ---- event feed --------------------------------------------------------------------------
def event_feed(snapshot: dict) -> list[dict]:
    """Chronological list of what happened, merged from the recorded tables (oldest first)."""
    agent_of = {t["name"]: t["agent"] for t in snapshot["tasks"]}
    feed: list[tuple[str, int, dict]] = []

    def add(when: str, kind: str, agent: str, subject: str, detail: str) -> None:
        feed.append((when, len(feed), {"Time": when, "Event": kind, "Agent": agent, "Subject": subject,
                                       "Detail": detail}))

    for e in snapshot["events"]:
        task, agent, note = e["task"], agent_of.get(e["task"], ""), e["note"]
        if e["to_state"] == "RUNNING":
            add(e["created_at"], "Task started", agent, task, note)
        elif e["to_state"] == "COMPLETED":
            add(e["created_at"], "Human override" if note.startswith("human override") else "Task completed",
                agent, task, note)
            if task == "export" and not note.startswith("human override"):
                add(e["created_at"], "Project completed", "orchestrator", snapshot["project"]["id"],
                    "all approvals passed and the final files were exported")
        elif e["to_state"] == "REJECTED":
            add(e["created_at"], REJECTION_LABEL.get(task, "Work rejected"), agent, task, note)
        elif e["to_state"] == "RETRYING":
            add(e["created_at"], "Regeneration requested", agent, task, note)
        elif e["to_state"] == "FAILED":
            add(e["created_at"], "Task failed", agent, task, note)
        elif e["to_state"] == "NEEDS_REVIEW":
            add(e["created_at"], "Needs human review", agent, task, note)
        elif e["to_state"] == "PENDING" and note == "reset by resume":
            add(e["created_at"], "Resumed", agent, task, note)
    for ev in snapshot["evaluations"]:
        score = "" if ev.score is None else f"score {ev.score:.2f}. "
        add(ev.created_at, "Quality check passed" if ev.passed else "Quality check failed", ev.evaluator,
            ev.target_kind, score + ("; ".join(ev.issues[:2]) or ev.explanation)[:300])
    for err in snapshot["errors"]:
        api = "OpenRouter" in err["message"] or "HTTP " in err["message"] or "network error" in err["message"]
        add(err["created_at"], "API error" if api else "Error", err["agent"], err["task"], err["message"][:300])
    for d in snapshot["decisions"]:
        add(d["created_at"], "Decision", d["actor"], d["subject"], f"{d['decision']}: {d['reason']}"[:300])
    # Timestamps have one-second precision, so records from different tables that share a second
    # keep their table order; the completion marker is always placed last within its second.
    return [row for _, _, row in sorted(
        feed, key=lambda item: (item[0], item[2]["Event"] == "Project completed", item[1]))]


# ---- what is blocking the production, and what to do about it ---------------------------------
def blockers(snapshot: dict) -> list[dict]:
    project, pid = snapshot["project"], snapshot["project"]["id"]
    found = []
    for task in snapshot["tasks"]:
        reason = task["error"] or "no reason was recorded"
        same = next((b for b in found if b["reason"] == reason and task["state"] == "NEEDS_REVIEW"), None)
        if same:
            # One escalation marks both the task that ran out of attempts and its reviewer. Show it
            # once; the override belongs on the reviewer (the later task), which releases both.
            same["task"] = f"{same['task']}` and `{task['name']}"
            same["agent"] = f"{same['agent']}, {task['agent']}"
            same["commands"][1] = f'python run.py override {pid} {task["name"]} --reason "..."   (then resume)'
        elif task["state"] == "NEEDS_REVIEW":
            if "accepted range" in reason or "target" in reason and "Final Video Guard" in reason:
                first = ("The rendered video is kept (see the media section). If the accepted duration "
                         "range in .env now covers it, retry Final QA:")
            elif task["name"] in ("compliance", "metadata_compliance"):
                first = ("A person must read the compliance findings. After fixing the cause, retry; or "
                         "record an explicit decision with the override command:")
            else:
                first = "A person must decide. Fix the cause and retry, or accept the current result:"
            found.append({"task": task["name"], "agent": task["agent"], "state": "needs review",
                          "reason": reason, "action": first, "commands": [
                              f"python run.py resume {pid} --retry-escalated",
                              f'python run.py override {pid} {task["name"]} --reason "..."   (then resume)']})
        elif task["state"] == "FAILED":
            if "budget reached" in reason:
                action, command = "The budget is used up. Continue with a higher one:", \
                    f"python run.py resume {pid} --budget <USD>"
            elif "media request cap" in reason:
                action, command = "Raise MAX_MEDIA_REQUESTS in .env, then:", f"python run.py resume {pid}"
            else:
                action, command = "Fix the cause shown above, then continue from this task:", \
                    f"python run.py resume {pid}"
            found.append({"task": task["name"], "agent": task["agent"], "state": "failed", "reason": reason,
                          "action": action, "commands": [command]})
    if not found and project["status"] == "created":
        found.append({"task": "-", "agent": "-", "state": "not started",
                      "reason": "The project exists but has not been run.",
                      "action": "Start it from the command line:", "commands": [f"python run.py run {pid}"]})
    elif not found and project["status"] in ("incomplete", "stopped"):
        found.append({"task": "-", "agent": "-", "state": project["status"],
                      "reason": "The last run ended before every task completed.",
                      "action": "Continue it from the command line:",
                      "commands": [f"python run.py resume {pid}"]})
    return found


# ---- media -------------------------------------------------------------------------------------
def media(snapshot: dict, settings: Settings) -> dict:
    """Paths of media that exist on disk. The rendered video is listed as soon as rendering is
    done; it does not depend on the export task or on output/ having been filled."""
    assets, pid = snapshot["assets"], snapshot["project"]["id"]

    def existing(kind: str) -> list:
        found = [a for a in assets if a.kind == kind and Path(a.path).is_file()]
        return sorted(found, key=lambda a: (Path(a.path).name, a.created_at))

    final = settings.output_dir / pid / "video.mp4"
    rendered = existing("final_video")
    approval = snapshot["project"]["final_approval"]
    if final.is_file() and approval == "approved":
        video = {"path": str(final), "approved": True, "state": "final"}             # exported
    elif rendered and approval in ("video_approved", "approved"):
        video = {"path": rendered[-1].path, "approved": True, "state": "approved"}   # export still pending
    elif rendered:
        video = {"path": rendered[-1].path, "approved": False, "state": "preliminary"}
    else:
        video = None
    thumbnails = existing("thumbnail")
    approved = [t for t in thumbnails if t.status in APPROVED]
    return {"images": existing("image"), "audio": existing("audio"), "clips": existing("video"),
            "references": existing("reference"), "video": video, "thumbnail": approved[-1] if approved else None,
            "thumbnail_candidates": thumbnails}


def media_signature(snapshot: dict) -> tuple:
    """Changes only when media was added or its review status changed (used to avoid redrawing)."""
    assets = snapshot["assets"]
    return (snapshot["project"]["id"], snapshot["project"]["final_approval"], len(assets),
            sum(a.status in APPROVED for a in assets), max((a.created_at for a in assets), default=""))
