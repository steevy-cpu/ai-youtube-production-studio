"""Unified shared memory backed by one local SQLite file.

Memory categories and the tables behind them:
  working   -> tasks, working_memory            (what is happening now)
  episodic  -> agent_runs, task_events, evaluations, revisions, errors, decisions, costs
  semantic  -> knowledge                        (durable project facts and limitations)
  artifact  -> artifacts (versioned JSON), assets (metadata pointing at files on disk)
  queue     -> jobs (background production queue), locks (one worker, one run per project)

Media bytes are never stored here; `assets.path` references the local file.
All statements are parameterised.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from studio.schemas import EvaluationResult, MediaAsset, Record, RevisionRequest
from studio.utils import new_id, now_iso

T = TypeVar("T", bound=BaseModel)

APPROVED = ("approved", "human_approved")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY, topic TEXT NOT NULL, title TEXT NOT NULL, mode TEXT NOT NULL,
    status TEXT NOT NULL, budget_usd REAL NOT NULL, final_approval TEXT NOT NULL DEFAULT 'pending',
    brief TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
    project_id TEXT NOT NULL, name TEXT NOT NULL, seq INTEGER NOT NULL, agent TEXT NOT NULL,
    state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
    depends_on TEXT NOT NULL, feedback TEXT NOT NULL DEFAULT '[]', summary TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '', started_at TEXT, finished_at TEXT,
    PRIMARY KEY (project_id, name));
CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, task TEXT NOT NULL,
    from_state TEXT NOT NULL, to_state TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, task TEXT NOT NULL, agent TEXT NOT NULL,
    action TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL,
    duration_s REAL NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL,
    version INTEGER NOT NULL, status TEXT NOT NULL, parent_id TEXT, data TEXT NOT NULL,
    created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_artifacts ON artifacts (project_id, kind, key, version);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL, parent_id TEXT,
    path TEXT NOT NULL, status TEXT NOT NULL, is_sample INTEGER NOT NULL, model TEXT NOT NULL,
    data TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_assets ON assets (project_id, kind, parent_id);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, target_id TEXT, target_kind TEXT NOT NULL,
    evaluator TEXT NOT NULL, passed INTEGER NOT NULL, score REAL, data TEXT NOT NULL,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions (
    id TEXT PRIMARY KEY, project_id TEXT NOT NULL, target_id TEXT, target_kind TEXT NOT NULL,
    target_task TEXT NOT NULL, responsible_agent TEXT NOT NULL, requested_by TEXT NOT NULL,
    status TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, task TEXT NOT NULL,
    agent TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, task TEXT NOT NULL,
    agent TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL, amount_usd REAL,
    known INTEGER NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, actor TEXT NOT NULL,
    subject TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS knowledge (
    project_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (project_id, key));
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, kind TEXT NOT NULL,
    status TEXT NOT NULL, request_id TEXT UNIQUE, batch_id TEXT NOT NULL DEFAULT '',
    confirmed INTEGER NOT NULL DEFAULT 0, retry_escalated INTEGER NOT NULL DEFAULT 0,
    stop_requested INTEGER NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active ON jobs (project_id)
    WHERE status IN ('queued', 'running');
CREATE TABLE IF NOT EXISTS locks (
    name TEXT PRIMARY KEY, owner TEXT NOT NULL, started_at TEXT NOT NULL, heartbeat REAL NOT NULL);
CREATE TABLE IF NOT EXISTS working_memory (
    project_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (project_id, key));
"""


class MemoryStore:
    def __init__(self, db_path: Path | str, read_only: bool = False, timeout: float = 5.0):
        """read_only=True opens an existing database so that SQLite itself refuses every write
        (used by monitoring tools such as the dashboard)."""
        self.db_path = Path(db_path)
        if read_only:
            self.conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True,
                                        timeout=timeout)
            self.conn.row_factory = sqlite3.Row
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=timeout)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        try:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
        except sqlite3.Error:
            self.conn.rollback()        # never leave a failed write holding the database lock
            raise
        return cur

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    # ---- projects -----------------------------------------------------------
    def create_project(self, project_id: str, topic: str, title: str, mode: str,
                       budget_usd: float, brief: dict) -> None:
        now = now_iso()
        self._exec("INSERT INTO projects (id, topic, title, mode, status, budget_usd, brief, created_at, "
                   "updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (project_id, topic, title, mode, "created", budget_usd, json.dumps(brief), now, now))

    def get_project(self, project_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not rows:
            return None
        rows[0]["brief"] = json.loads(rows[0]["brief"])
        return rows[0]

    def list_projects(self) -> list[dict]:
        return self._rows("SELECT id, topic, title, mode, status, final_approval, budget_usd, created_at, "
                          "updated_at FROM projects ORDER BY created_at DESC")

    def update_project(self, project_id: str, **fields: Any) -> None:
        allowed = {"status", "budget_usd", "final_approval", "title"}
        for name, value in fields.items():
            if name not in allowed:
                raise ValueError(f"cannot update project field {name}")
            self._exec(f"UPDATE projects SET {name} = ?, updated_at = ? WHERE id = ?",  # name is whitelisted
                       (value, now_iso(), project_id))

    # ---- tasks (working memory / workflow state) ------------------------------
    def upsert_task(self, project_id: str, name: str, seq: int, agent: str, max_attempts: int,
                    depends_on: list[str]) -> None:
        self._exec("INSERT INTO tasks (project_id, name, seq, agent, state, max_attempts, depends_on) "
                   "VALUES (?,?,?,?,?,?,?) ON CONFLICT(project_id, name) DO UPDATE SET seq = excluded.seq, "
                   "agent = excluded.agent, max_attempts = excluded.max_attempts, "
                   "depends_on = excluded.depends_on",
                   (project_id, name, seq, agent, "PENDING", max_attempts, json.dumps(depends_on)))

    def tasks(self, project_id: str) -> list[dict]:
        rows = self._rows("SELECT * FROM tasks WHERE project_id = ? ORDER BY seq", (project_id,))
        for row in rows:
            row["depends_on"] = json.loads(row["depends_on"])
            row["feedback"] = json.loads(row["feedback"])
        return rows

    def get_task(self, project_id: str, name: str) -> dict:
        for task in self.tasks(project_id):
            if task["name"] == name:
                return task
        raise KeyError(f"unknown task {name}")

    def set_task(self, project_id: str, name: str, *, state: str | None = None, note: str = "",
                 **fields: Any) -> None:
        """Update a task; a state change is also appended to the task_events trace."""
        allowed = {"attempts", "feedback", "summary", "error", "started_at", "finished_at"}
        if state is not None:
            old = self.get_task(project_id, name)["state"]
            self._exec("UPDATE tasks SET state = ? WHERE project_id = ? AND name = ?",
                       (state, project_id, name))
            self._exec("INSERT INTO task_events (project_id, task, from_state, to_state, note, created_at) "
                       "VALUES (?,?,?,?,?,?)", (project_id, name, old, state, note, now_iso()))
        for field, value in fields.items():
            if field not in allowed:
                raise ValueError(f"cannot update task field {field}")
            if field == "feedback":
                value = json.dumps(value)
            self._exec(f"UPDATE tasks SET {field} = ? WHERE project_id = ? AND name = ?",  # whitelisted
                       (value, project_id, name))

    def task_events(self, project_id: str, task: str | None = None) -> list[dict]:
        if task:
            return self._rows("SELECT * FROM task_events WHERE project_id = ? AND task = ? ORDER BY id",
                              (project_id, task))
        return self._rows("SELECT * FROM task_events WHERE project_id = ? ORDER BY id", (project_id,))

    # ---- artifacts (versioned structured outputs) ------------------------------
    def save_artifact(self, project_id: str, kind: str, record: Record, key: str = "") -> Record:
        """Store a new version of (kind, key). Sets project_id and version on the record."""
        row = self.conn.execute("SELECT MAX(version) AS v FROM artifacts WHERE project_id = ? AND kind = ? "
                                "AND key = ?", (project_id, kind, key)).fetchone()
        record.project_id = project_id
        record.version = (row["v"] or 0) + 1
        self._exec("INSERT OR REPLACE INTO artifacts (id, project_id, kind, key, version, status, parent_id, "
                   "data, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (record.id, project_id, kind, key, record.version, record.status, record.parent_id,
                    record.model_dump_json(), record.created_at))
        return record

    def update_artifact(self, record: Record) -> None:
        """Rewrite an existing version in place (status changes, small annotations)."""
        self._exec("UPDATE artifacts SET status = ?, data = ? WHERE id = ?",
                   (record.status, record.model_dump_json(), record.id))

    def load(self, project_id: str, kind: str, model: type[T], key: str = "") -> T | None:
        row = self.conn.execute("SELECT data FROM artifacts WHERE project_id = ? AND kind = ? AND key = ? "
                                "ORDER BY version DESC LIMIT 1", (project_id, kind, key)).fetchone()
        return model.model_validate_json(row["data"]) if row else None

    def load_all(self, project_id: str, kind: str, model: type[T]) -> dict[str, T]:
        """Latest version of every key of a kind, as {key: record}."""
        rows = self.conn.execute(
            "SELECT a.key, a.data FROM artifacts a JOIN (SELECT key, MAX(version) AS v FROM artifacts "
            "WHERE project_id = ? AND kind = ? GROUP BY key) m ON a.key = m.key AND a.version = m.v "
            "WHERE a.project_id = ? AND a.kind = ?", (project_id, kind, project_id, kind)).fetchall()
        return {r["key"]: model.model_validate_json(r["data"]) for r in rows}

    def latest_artifacts(self, project_id: str) -> list[dict]:
        """Latest version of every artifact as raw rows (kind, key, version, status, data)."""
        return self._rows(
            "SELECT a.kind, a.key, a.version, a.status, a.data FROM artifacts a JOIN (SELECT kind, key, "
            "MAX(version) AS v FROM artifacts WHERE project_id = ? GROUP BY kind, key) m ON a.kind = m.kind "
            "AND a.key = m.key AND a.version = m.v WHERE a.project_id = ? ORDER BY a.kind, a.created_at, "
            "a.rowid", (project_id, project_id))

    def artifact_versions(self, project_id: str, kind: str, key: str = "") -> list[dict]:
        return self._rows("SELECT id, version, status, created_at FROM artifacts WHERE project_id = ? "
                          "AND kind = ? AND key = ? ORDER BY version", (project_id, kind, key))

    def artifact_counts(self, project_id: str) -> dict[str, int]:
        rows = self._rows("SELECT kind, COUNT(*) AS n FROM artifacts WHERE project_id = ? GROUP BY kind",
                          (project_id,))
        return {r["kind"]: r["n"] for r in rows}

    # ---- assets (media metadata; files live on disk) ----------------------------
    def add_asset(self, asset: MediaAsset) -> MediaAsset:
        self._exec("INSERT INTO assets (id, project_id, kind, parent_id, path, status, is_sample, model, "
                   "data, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (asset.id, asset.project_id, asset.kind, asset.parent_id, asset.path, asset.status,
                    int(asset.is_sample), asset.model, asset.model_dump_json(), asset.created_at))
        return asset

    def update_asset(self, asset: MediaAsset) -> None:
        self._exec("UPDATE assets SET status = ?, data = ? WHERE id = ?",
                   (asset.status, asset.model_dump_json(), asset.id))

    def assets(self, project_id: str, kind: str | None = None, parent_id: str | None = None,
               status: tuple[str, ...] | None = None) -> list[MediaAsset]:
        sql, params = "SELECT data FROM assets WHERE project_id = ?", [project_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if parent_id:
            sql += " AND parent_id = ?"
            params.append(parent_id)
        if status:
            sql += f" AND status IN ({','.join('?' * len(status))})"
            params.extend(status)
        rows = self.conn.execute(sql + " ORDER BY created_at, rowid", tuple(params)).fetchall()
        return [MediaAsset.model_validate_json(r["data"]) for r in rows]

    def latest_asset(self, project_id: str, kind: str, parent_id: str,
                     status: tuple[str, ...] | None = None) -> MediaAsset | None:
        found = self.assets(project_id, kind, parent_id, status)
        return found[-1] if found else None

    # ---- episodic records -------------------------------------------------------
    def add_evaluation(self, evaluation: EvaluationResult) -> EvaluationResult:
        self._exec("INSERT INTO evaluations (id, project_id, target_id, target_kind, evaluator, passed, score, "
                   "data, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (evaluation.id, evaluation.project_id, evaluation.parent_id, evaluation.target_kind,
                    evaluation.evaluator, int(evaluation.passed), evaluation.score,
                    evaluation.model_dump_json(), evaluation.created_at))
        return evaluation

    def evaluations(self, project_id: str, evaluator: str | None = None,
                    target_id: str | None = None) -> list[EvaluationResult]:
        sql, params = "SELECT data FROM evaluations WHERE project_id = ?", [project_id]
        if evaluator:
            sql += " AND evaluator = ?"
            params.append(evaluator)
        if target_id:
            sql += " AND target_id = ?"
            params.append(target_id)
        rows = self.conn.execute(sql + " ORDER BY created_at, rowid", tuple(params)).fetchall()
        return [EvaluationResult.model_validate_json(r["data"]) for r in rows]

    def evaluation_summary(self, project_id: str) -> list[dict]:
        return self._rows("SELECT evaluator, COUNT(*) AS total, SUM(passed) AS passed, "
                          "ROUND(AVG(score), 3) AS avg_score FROM evaluations WHERE project_id = ? "
                          "GROUP BY evaluator ORDER BY evaluator", (project_id,))

    def add_revision(self, request: RevisionRequest) -> None:
        self._exec("INSERT INTO revisions (id, project_id, target_id, target_kind, target_task, "
                   "responsible_agent, requested_by, status, data, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (request.id, request.project_id, request.parent_id, request.target_kind,
                    request.target_task, request.responsible_agent, request.requested_by, "open",
                    request.model_dump_json(), request.created_at))

    def resolve_revisions(self, project_id: str, target_task: str) -> None:
        self._exec("UPDATE revisions SET status = 'resolved' WHERE project_id = ? AND target_task = ? "
                   "AND status = 'open'", (project_id, target_task))

    def revisions(self, project_id: str) -> list[dict]:
        return self._rows("SELECT id, target_id, target_kind, target_task, responsible_agent, requested_by, "
                          "status, created_at FROM revisions WHERE project_id = ? ORDER BY created_at, rowid",
                          (project_id,))

    def add_agent_run(self, project_id: str, task: str, agent: str, action: str, model: str,
                      status: str, detail: str, duration_s: float) -> None:
        self._exec("INSERT INTO agent_runs (id, project_id, task, agent, action, model, status, detail, "
                   "duration_s, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (new_id("run"), project_id, task, agent, action, model, status, detail[:2000],
                    round(duration_s, 3), now_iso()))

    def agent_runs(self, project_id: str) -> list[dict]:
        return self._rows("SELECT * FROM agent_runs WHERE project_id = ? ORDER BY created_at, rowid",
                          (project_id,))

    def agent_status(self, project_id: str) -> list[dict]:
        return self._rows("SELECT agent, COUNT(*) AS runs, SUM(status = 'ok') AS ok, "
                          "SUM(status != 'ok') AS failed, MAX(created_at) AS last_run FROM agent_runs "
                          "WHERE project_id = ? GROUP BY agent ORDER BY agent", (project_id,))

    def add_error(self, project_id: str, task: str, agent: str, message: str) -> None:
        self._exec("INSERT INTO errors (project_id, task, agent, message, created_at) VALUES (?,?,?,?,?)",
                   (project_id, task, agent, message[:4000], now_iso()))

    def errors(self, project_id: str) -> list[dict]:
        return self._rows("SELECT * FROM errors WHERE project_id = ? ORDER BY id", (project_id,))

    def add_decision(self, project_id: str, actor: str, subject: str, decision: str, reason: str) -> None:
        self._exec("INSERT INTO decisions (project_id, actor, subject, decision, reason, created_at) "
                   "VALUES (?,?,?,?,?,?)", (project_id, actor, subject, decision, reason, now_iso()))

    def decisions(self, project_id: str, actor: str | None = None, subject: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM decisions WHERE project_id = ?", [project_id]
        if actor:
            sql += " AND actor = ?"
            params.append(actor)
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        return self._rows(sql + " ORDER BY id", tuple(params))

    # ---- costs --------------------------------------------------------------------
    def add_cost(self, project_id: str, task: str, agent: str, model: str, kind: str,
                 amount_usd: float | None, note: str = "") -> None:
        self._exec("INSERT INTO costs (project_id, task, agent, model, kind, amount_usd, known, note, "
                   "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                   (project_id, task, agent, model, kind, amount_usd, int(amount_usd is not None), note,
                    now_iso()))

    def cost_summary(self, project_id: str) -> dict:
        """Recorded spend. Requests without provider cost data are counted as `unknown`, never as 0."""
        rows = self._rows("SELECT kind, model, COUNT(*) AS requests, COALESCE(SUM(amount_usd), 0) AS usd, "
                          "SUM(known = 0) AS unknown FROM costs WHERE project_id = ? GROUP BY kind, model "
                          "ORDER BY kind, model", (project_id,))
        return {"known_usd": round(sum(r["usd"] for r in rows), 6),
                "unknown_cost_requests": sum(r["unknown"] for r in rows),
                "requests": sum(r["requests"] for r in rows),
                "media_requests": sum(r["requests"] for r in rows if r["kind"] in ("image", "video", "tts")),
                "by_kind": rows}

    # ---- semantic and working memory ---------------------------------------------
    def remember(self, project_id: str, key: str, value: Any, table: str = "knowledge") -> None:
        if table not in ("knowledge", "working_memory"):
            raise ValueError("unknown memory table")
        self._exec(f"INSERT INTO {table} (project_id, key, value, updated_at) VALUES (?,?,?,?) "
                   "ON CONFLICT(project_id, key) DO UPDATE SET value = excluded.value, "
                   "updated_at = excluded.updated_at", (project_id, key, json.dumps(value), now_iso()))

    def recall(self, project_id: str, key: str, default: Any = None, table: str = "knowledge") -> Any:
        if table not in ("knowledge", "working_memory"):
            raise ValueError("unknown memory table")
        row = self.conn.execute(f"SELECT value FROM {table} WHERE project_id = ? AND key = ?",
                                (project_id, key)).fetchone()
        return json.loads(row["value"]) if row else default

    def recall_all(self, project_id: str, table: str = "knowledge") -> dict[str, Any]:
        if table not in ("knowledge", "working_memory"):
            raise ValueError("unknown memory table")
        rows = self._rows(f"SELECT key, value FROM {table} WHERE project_id = ? ORDER BY key", (project_id,))
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def add_limitation(self, project_id: str, text: str) -> None:
        current = self.recall(project_id, "limitations", [])
        if text not in current:
            self.remember(project_id, "limitations", current + [text])

    # ---- background jobs (local production queue) ---------------------------------
    def add_job(self, project_id: str, kind: str, *, confirmed: bool, retry_escalated: bool = False,
                request_id: str | None = None, batch_id: str = "") -> dict:
        """Queue a job. The unique index refuses a second queued/running job for the same project."""
        cur = self._exec("INSERT INTO jobs (project_id, kind, status, request_id, batch_id, confirmed, "
                         "retry_escalated, created_at) VALUES (?,?,?,?,?,?,?,?)",
                         (project_id, kind, "queued", request_id, batch_id, int(confirmed),
                          int(retry_escalated), now_iso()))
        return self.get_job(cur.lastrowid)

    def get_job(self, job_id: int) -> dict | None:
        rows = self._rows("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0] if rows else None

    def job_by_request(self, request_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM jobs WHERE request_id = ?", (request_id,))
        return rows[0] if rows else None

    def jobs(self, statuses: tuple[str, ...] | None = None, project_id: str | None = None) -> list[dict]:
        sql, params = "SELECT * FROM jobs WHERE 1 = 1", []
        if statuses:
            sql += f" AND status IN ({','.join('?' * len(statuses))})"
            params.extend(statuses)
        if project_id:
            sql += " AND project_id = ?"
            params.append(project_id)
        return self._rows(sql + " ORDER BY id", tuple(params))

    def claim_next_job(self) -> dict | None:
        """Atomically take the oldest queued job and mark it running (first in, first out)."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
            if row:
                self.conn.execute("UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                                  (now_iso(), row["id"]))
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        return self.get_job(row["id"]) if row else None

    def finish_job(self, job_id: int, status: str, message: str = "") -> None:
        self._exec("UPDATE jobs SET status = ?, message = ?, finished_at = ? WHERE id = ?",
                   (status, message[:1000], now_iso(), job_id))

    def set_job_stop(self, job_id: int) -> None:
        self._exec("UPDATE jobs SET stop_requested = 1 WHERE id = ? AND status = 'running'", (job_id,))

    # ---- locks with a heartbeat (one worker, one run per project) -------------------
    def acquire_lock(self, name: str, owner: str, stale_after: float = 30.0) -> bool:
        """Take the lock unless another owner holds it and refreshed it within `stale_after` s."""
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute("SELECT owner, heartbeat FROM locks WHERE name = ?", (name,)).fetchone()
            held = row is not None and row["owner"] != owner and now - row["heartbeat"] < stale_after
            if not held:
                self.conn.execute("INSERT INTO locks (name, owner, started_at, heartbeat) VALUES (?,?,?,?) "
                                  "ON CONFLICT(name) DO UPDATE SET owner = excluded.owner, "
                                  "started_at = excluded.started_at, heartbeat = excluded.heartbeat",
                                  (name, owner, now_iso(), now))
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        return not held

    def touch_lock(self, name: str, owner: str) -> None:
        self._exec("UPDATE locks SET heartbeat = ? WHERE name = ? AND owner = ?", (time.time(), name, owner))

    def release_lock(self, name: str, owner: str) -> None:
        self._exec("DELETE FROM locks WHERE name = ? AND owner = ?", (name, owner))

    def lock_alive(self, name: str, stale_after: float = 30.0) -> bool:
        row = self.conn.execute("SELECT heartbeat FROM locks WHERE name = ?", (name,)).fetchone()
        return row is not None and time.time() - row["heartbeat"] < stale_after

    # ---- concise progress summary (what the Director reads) ------------------------
    def progress_summary(self, project_id: str) -> str:
        project = self.get_project(project_id) or {}
        tasks = self.tasks(project_id)
        done = [t["name"] for t in tasks if t["state"] == "COMPLETED"]
        open_ = [f"{t['name']}={t['state']}" for t in tasks if t["state"] != "COMPLETED"]
        evals = "; ".join(f"{e['evaluator']} {e['passed']}/{e['total']} passed"
                          for e in self.evaluation_summary(project_id)) or "none yet"
        cost = self.cost_summary(project_id)
        lines = [
            f"Project: {project.get('title', '?')} (mode {project.get('mode', '?')})",
            f"Completed tasks ({len(done)}/{len(tasks)}): {', '.join(done) or 'none'}",
            f"Open tasks: {', '.join(open_[:8]) or 'none'}",
            f"Evaluations: {evals}",
            f"Recorded cost: ${cost['known_usd']:.4f} of ${project.get('budget_usd', 0):.2f} budget "
            f"({cost['unknown_cost_requests']} requests with unknown cost)",
            f"Errors logged: {len(self.errors(project_id))}",
        ]
        limitations = self.recall(project_id, "limitations", [])
        if limitations:
            lines.append("Known limitations: " + " | ".join(limitations))
        return "\n".join(lines)
