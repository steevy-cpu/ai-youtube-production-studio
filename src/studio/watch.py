"""Live production log: follow what every agent is doing and saying, in the terminal.

Read-only. It polls the SQLite memory that the running production writes to (task state changes,
model calls with a digest of each answer, failed reviews, revision requests, decisions, errors)
and prints new records as they appear. It never starts, stops or changes a production.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from pydantic import BaseModel

AGENT_LABEL = {
    "director": "Director", "research": "Researcher", "story_writer": "Story Writer",
    "scene_director": "Scene Director", "prompt_engineer": "Prompt Engineer", "prompt_judge": "Prompt Judge",
    "visual_generation": "Visual Generation", "visual_judge": "Visual Judge",
    "voice_director": "Voice Director", "editor": "Editor", "final_qa": "Final QA",
    "compliance": "Compliance", "seo": "SEO & Growth", "seo_thumbnail_review": "SEO & Growth",
    "script_guard": "Script Guard", "timeline_guard": "Timeline Guard", "prompt_guard": "Prompt Guard",
    "media_guard": "Media Guard", "audio_qa": "Audio QA", "renderer": "Renderer",
    "orchestrator": "Orchestrator"}
COLOURS = {"dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
           "cyan": "36", "bold": "1"}
STATE_COLOUR = {"RUNNING": "cyan", "COMPLETED": "green", "REJECTED": "yellow", "RETRYING": "yellow",
                "NEEDS_REVIEW": "red", "FAILED": "red", "PENDING": "dim"}


def label(agent: str) -> str:
    return AGENT_LABEL.get(agent, agent.replace("_", " ").title())


def digest(result: BaseModel, limit: int = 900) -> str:
    """A short, readable account of what an agent answered (stored with every model call)."""
    def short(value, width: int) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= width else text[:width - 1] + "…"

    def describe(value) -> str:
        if isinstance(value, BaseModel):
            return digest(value, 300)
        if isinstance(value, dict):
            return short(json.dumps(value, ensure_ascii=False), 200)
        if isinstance(value, list):
            if not value:
                return ""
            shown = "; ".join(describe(v) for v in value[:3])
            return shown + (f" (+{len(value) - 3} more)" if len(value) > 3 else "")
        return short(value, 260)

    data = {name: getattr(result, name) for name in type(result).model_fields
            if name not in ("id", "project_id", "parent_id", "created_at", "status", "version")}
    parts = []
    if isinstance(data.get("approved"), bool):          # every judge and approval answer
        score = data.get("score")
        parts.append(("APPROVED" if data["approved"] else "REJECTED")
                     + (f" score {score:.2f}" if isinstance(score, (int, float)) else ""))
        for name in ("issues", "reasons", "explanation", "suggested_changes"):
            if data.get(name):
                parts.append(f"{name.replace('_', ' ')}: {describe(data[name])}")
    else:
        for name, value in data.items():
            text = describe(value)
            if text and text not in ("False", "0", "0.0"):
                parts.append(f"{name}: {text}")
    return short(" | ".join(parts), limit)


class Watcher:
    """Yields the new records of one project as printable (time, who, what, text, colour) events."""

    # table -> SQL returning rows newer than the cursor (rowid), oldest first
    QUERIES = {
        "task_events": "SELECT rowid AS _row, * FROM task_events WHERE project_id = ? AND rowid > ? ORDER BY rowid",
        "agent_runs": "SELECT rowid AS _row, * FROM agent_runs WHERE project_id = ? AND rowid > ? ORDER BY rowid",
        "evaluations": "SELECT rowid AS _row, * FROM evaluations WHERE project_id = ? AND rowid > ? ORDER BY rowid",
        "revisions": "SELECT rowid AS _row, * FROM revisions WHERE project_id = ? AND rowid > ? ORDER BY rowid",
        "decisions": "SELECT rowid AS _row, * FROM decisions WHERE project_id = ? AND rowid > ? ORDER BY rowid",
        "errors": "SELECT rowid AS _row, * FROM errors WHERE project_id = ? AND rowid > ? ORDER BY rowid"}

    ORDER = {"agent_runs": 1, "evaluations": 2, "revisions": 3, "decisions": 4, "task_events": 5, "errors": 6}

    def __init__(self, db_path: Path, project_id: str):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        self.conn.row_factory = sqlite3.Row
        self.project_id = project_id
        self.cursors = {table: 0 for table in self.QUERIES}

    def close(self) -> None:
        self.conn.close()

    def project(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM projects WHERE id = ?", (self.project_id,)).fetchone()
        return dict(row) if row else None

    def cost(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(amount_usd), 0) AS usd FROM costs WHERE project_id = ?",
                                (self.project_id,)).fetchone()
        return float(row["usd"] or 0)

    def running(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT name, agent, attempts, max_attempts, started_at FROM tasks WHERE project_id = ? "
            "AND state = 'RUNNING' ORDER BY seq", (self.project_id,))]

    def progress(self) -> tuple[int, int]:
        row = self.conn.execute("SELECT SUM(state = 'COMPLETED') AS done, COUNT(*) AS total FROM tasks "
                                "WHERE project_id = ?", (self.project_id,)).fetchone()
        return int(row["done"] or 0), int(row["total"] or 0)

    def poll(self) -> list[dict]:
        """New events since the last poll, in time order."""
        events = []
        for table, sql in self.QUERIES.items():
            for row in self.conn.execute(sql, (self.project_id, self.cursors[table])):
                self.cursors[table] = row["_row"]
                event = getattr(self, "_" + table)(dict(row))
                if event:
                    events.append(event | {"order": self.ORDER.get(table, 5) if not event.get("rule") else 0})
        # Timestamps have one-second resolution: within a second show the task start, then what the
        # agents said, then reviews and requests, then the task's outcome.
        return sorted(events, key=lambda e: (e["at"], e["order"]))

    # ---- one formatter per table ---------------------------------------------------
    def _task_events(self, r: dict) -> dict | None:
        state = r["to_state"]
        if state == "PENDING":
            return None                               # resets are implied by the rejection line above them
        agent = self.conn.execute("SELECT agent FROM tasks WHERE project_id = ? AND name = ?",
                                  (self.project_id, r["task"])).fetchone()
        return {"at": r["created_at"], "who": label(agent["agent"]) if agent else "",
                "what": f"{r['task']} {state}", "text": r["note"], "colour": STATE_COLOUR.get(state, "dim"),
                "rule": state == "RUNNING"}

    def _agent_runs(self, r: dict) -> dict:
        model = r["model"].split("/")[-1]
        timing = f" [{model}, {r['duration_s']:.1f}s]" if model else ""
        failed = r["status"] != "ok"
        return {"at": r["created_at"], "who": label(r["agent"]), "what": r["action"] + timing,
                "text": (f"{r['status']}: " if failed else "") + r["detail"],
                "colour": "red" if failed else ""}

    def _evaluations(self, r: dict) -> dict | None:
        if r["passed"]:
            return None                               # approvals already show in the judge's own answer
        data = json.loads(r["data"])
        score = "" if r["score"] is None else f" (score {r['score']:.2f}, needs {data.get('threshold')})"
        return {"at": r["created_at"], "who": label(r["evaluator"]),
                "what": f"FAILED CHECK on {r['target_kind']}{score}",
                "text": "\n".join(f"- {i}" for i in data.get("issues", [])) or data.get("explanation", ""),
                "colour": "yellow"}

    def _revisions(self, r: dict) -> dict:
        data = json.loads(r["data"])
        fixes = data.get("suggested_changes") or []
        return {"at": r["created_at"], "who": label(r["requested_by"]),
                "what": f"asks {label(r['responsible_agent'])} to redo the {r['target_kind'].replace('_', ' ')}",
                "text": "\n".join(f"- fix: {f}" for f in fixes[:4]), "colour": "magenta"}

    def _decisions(self, r: dict) -> dict:
        return {"at": r["created_at"], "who": label(r["actor"]), "what": f"decision on {r['subject']}: "
                f"{r['decision']}", "text": r["reason"], "colour": "blue"}

    def _errors(self, r: dict) -> dict:
        return {"at": r["created_at"], "who": label(r["agent"]), "what": f"ERROR in {r['task']}",
                "text": r["message"][:400] + ("…" if len(r["message"]) > 400 else ""), "colour": "red"}


def choose_project(db_path: Path) -> str | None:
    """The project that is running now, otherwise the most recently active one."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        row = conn.execute(
            "SELECT p.id FROM projects p LEFT JOIN task_events e ON e.project_id = p.id GROUP BY p.id "
            "ORDER BY (p.status = 'running') DESC, MAX(e.created_at) DESC LIMIT 1").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def render(event: dict, colour: bool, width: int = 110) -> str:
    def paint(text: str, name: str) -> str:
        return f"\033[{COLOURS[name]}m{text}\033[0m" if colour and name else text
    try:
        clock = datetime.fromisoformat(event["at"]).astimezone().strftime("%H:%M:%S")
    except ValueError:
        clock = event["at"][11:19]
    head = f"{paint(clock, 'dim')}  {paint(f'{event['who']:<17}', 'bold')} {paint(event['what'], event['colour'])}"
    lines = [("\n" if event.get("rule") else "") + head]
    for paragraph in (event["text"] or "").splitlines():
        lines += textwrap.wrap(paragraph, width, initial_indent=" " * 12, subsequent_indent=" " * 14) or []
    return "\n".join(lines)


def watch(db_path: Path, project_id: str | None = None, *, history: bool = False, tail: int = 25,
          follow: bool = True, interval: float = 1.0, out: Callable[[str], None] = print,
          colour: bool | None = None, sleep: Callable[[float], None] = time.sleep,
          keep_going: Callable[[], bool] = lambda: True) -> int:
    """Print the production log of a project and keep following it until interrupted."""
    if not Path(db_path).is_file():
        out("No productions yet (the studio database does not exist).")
        return 1
    project_id = project_id or choose_project(db_path)
    if project_id is None:
        out("No productions yet. Start one first, then run `watch` in a second terminal.")
        return 1
    colour = sys.stdout.isatty() if colour is None else colour
    watcher = Watcher(db_path, project_id)
    try:
        project = watcher.project()
        if project is None:
            out(f"Unknown project '{project_id}' (see `python run.py list`).")
            return 1
        done, total = watcher.progress()
        out(f"Watching {project_id}: {project['title']}  [{project['mode']}, {project['status']}, "
            f"{done}/{total} tasks done, ${watcher.cost():.2f} spent]")
        events = watcher.poll()
        if not history and len(events) > tail:
            out(f"... {len(events) - tail} earlier events hidden (use --history to see everything)")
            events = events[-tail:]
        for event in events:
            out(render(event, colour))
        for task in watcher.running():
            out(f"\nNow working: {label(task['agent'])} on {task['name']} "
                f"(attempt {task['attempts']}/{task['max_attempts']})")
        if not follow:
            return 0
        out("\nFollowing live. Press Ctrl+C to stop watching (the production is not affected).")
        status = project["status"]
        while keep_going():
            sleep(interval)
            for event in watcher.poll():
                out(render(event, colour))
            current = (watcher.project() or {}).get("status", status)
            if current != status:
                done, total = watcher.progress()
                out(f"\n=== production is now {current.upper()}: {done}/{total} tasks done, "
                    f"${watcher.cost():.2f} spent ===")
                status = current
        return 0
    except KeyboardInterrupt:
        out("\nStopped watching.")
        return 0
    finally:
        watcher.close()
