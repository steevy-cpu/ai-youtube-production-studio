"""Render animated agents using activity recorded by the dashboard."""

from html import escape
from pathlib import Path

import monitor


def state_for(status: str, project_status: str = "running") -> str:
    if "running" in status:
        return "working" if project_status == "running" else "paused"
    if any(word in status for word in ("failed", "rejected", "review")):
        return "blocked"
    if "retrying" in status:
        return "retrying" if project_status == "running" else "paused"
    if "completed" in status:
        return "done"
    return "waiting"


def render_stage(snapshot: dict | None = None, *, demo: bool = False) -> str:
    rows = monitor.agent_activity(snapshot) if snapshot is not None else []
    by_id = {row["id"]: row for row in rows}
    cards = []

    for index, (agent_id, name) in enumerate(monitor.AGENTS.items()):
        row = by_id.get(agent_id, {})
        project_status = snapshot["project"]["status"] if snapshot else "created"
        state = state_for(row.get("Status", "pending"), project_status)
        task = row.get("Current task", "Waiting for a production")
        result = row.get("Latest result or failure", "No activity recorded yet.")
        done = row.get("Tasks done", "0/0")

        cards.append(
            f'''
            <button class="agent {state}"
                data-state="{state}"
                data-name="{escape(name, quote=True)}"
                data-task="{escape(task, quote=True)}"
                data-result="{escape(result, quote=True)}"
                style="--delay:{index * -.19}s">

                <div class="card-top">
                    <span class="number">{index + 1:02}</span>
                    <span class="status">{state}</span>
                </div>

                <div class="scene" aria-hidden="true">
                    <div class="platform"></div>
                    <div class="robot">
                        <div class="antenna"></div>
                        <div class="head">
                            <div class="visor"><i></i><i></i></div>
                        </div>
                        <div class="body"><span></span></div>
                        <div class="arm left"></div>
                        <div class="arm right"></div>
                        <div class="leg left"></div>
                        <div class="leg right"></div>
                    </div>
                    <div class="desk">
                        <div class="screen"><b></b><b></b><b></b></div>
                        <div class="keyboard"></div>
                    </div>
                </div>

                <strong>{escape(name)}</strong>
                <span class="task">{escape(task)}</span>
                <div class="card-bottom">
                    <span>Tasks</span><span>{escape(done)}</span>
                </div>
            </button>
            '''
        )

    template = Path(__file__).with_name("agent_stage.html").read_text(
        encoding="utf-8"
    )
    mode = (
        "PRESENTATION DEMO · SIMULATED"
        if demo
        else "RECORDED PRODUCTION ACTIVITY"
    )

    return (
        template.replace("__CARDS__", "".join(cards))
        .replace("__DEMO__", "true" if demo else "false")
        .replace("__MODE__", mode)
    )