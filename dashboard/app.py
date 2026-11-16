"""AI YouTube Production Studio dashboard.   Run:  streamlit run dashboard/app.py

Presentation only. Reading is done by monitor.py (SQLite opened read-only) and actions by
launcher.py, which queues jobs for a separate worker process. Nothing in this file runs a
production, so the page stays responsive and a production survives reruns and page reloads.
"""
import io
import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "dashboard"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import launcher  # noqa: E402
import monitor  # noqa: E402
import agent_stage  # noqa: E402
from studio.config import ConfigError, load_settings  # noqa: E402
from studio.services.openrouter import OpenRouterError  # noqa: E402

VIEWS = ["Monitor", "Presentation demo", "Media", "New production", "Batch & queue", "Recovery"]
STATUS_BOX = {
    "completed": (st.success, "Completed: every approval passed and the final files were exported."),
    "running": (st.info, "Running: the pipeline is executing right now."),
    "needs_review": (st.warning, "Needs review: the pipeline is waiting for a human decision."),
    "failed": (st.error, "Failed: a task stopped with an error."),
    "stopped": (st.warning, "Stopped: the run was halted before finishing."),
    "incomplete": (st.warning, "Incomplete: the last run ended before every task completed."),
    "created": (st.info, "Created: this project is waiting in the queue or has not been run yet."),
}
STATUS_MARK = {"completed": "✅", "running": "🔄", "needs_review": "⚠️", "failed": "❌", "stopped": "⏹️",
               "incomplete": "⏸️", "created": "🆕"}
RUNNING_REFRESH_S, IDLE_REFRESH_S = 3, 10

st.set_page_config(page_title="AI YouTube Production Studio", layout="wide")


# ---- small helpers ---------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=512)
def preview_image(path: str, modified: float, width: int = 480) -> bytes:
    """Small JPEG preview, cached per file version so large originals are read only once."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((width, width))
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=80)
    return buffer.getvalue()


def show_image(column, asset, caption: str) -> None:
    path = Path(asset.path)
    column.image(preview_image(str(path), path.stat().st_mtime), caption=caption)


def token(name: str) -> str:
    """One id per intended action. A rerun re-uses it, so the same click can never act twice."""
    return st.session_state.setdefault(f"token:{name}", uuid.uuid4().hex)


def new_token(name: str) -> None:
    st.session_state[f"token:{name}"] = uuid.uuid4().hex


def go_to(view: str, project_id: str | None = None) -> None:
    """Switch view (and project) on the next run; widget state cannot be changed mid-run."""
    st.session_state["goto"] = (view, project_id)
    st.rerun()


def table(rows: list[dict], **kwargs) -> None:
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch", **kwargs)
    else:
        st.caption("Nothing recorded.")


def load(settings, project_id: str) -> dict | None:
    """Latest snapshot; when the database is momentarily locked, keep showing the previous one."""
    try:
        snapshot = monitor.load_snapshot(settings, project_id)
    except monitor.DatabaseBusy:
        st.caption("The database is busy (the pipeline is writing). Showing the last data read.")
        return st.session_state.get(f"snapshot:{project_id}")
    st.session_state[f"snapshot:{project_id}"] = snapshot
    return snapshot

# Display the animated crew using recorded activity or the free demo.
def stage_section(snapshot=None, *, demo=False) -> None:
    components.html(
        agent_stage.render_stage(snapshot, demo=demo),
        height=760,
        scrolling=True,
    )
    if not demo:
        st.caption(
            "Characters follow recorded task states. "
            "Enable auto-refresh to see updates."
        )


# ---- Monitor ---------------------------------------------------------------------------
def overview_section(snapshot: dict) -> None:
    o = monitor.overview(snapshot)
    box, text = STATUS_BOX.get(o["status"], (st.info, o["status"]))
    box(f"**{o['status'].replace('_', ' ').upper()}** - {text}")
    if o["stale"]:
        st.warning(f"No activity has been recorded for {monitor.format_duration(o['idle_s'])}. Long renders "
                   "are normal, but the worker may have been interrupted; see Batch & queue.")

    a, b, c, d = st.columns(4)
    a.metric("Mode", o["mode"])
    b.metric("Current stage", o["stage"])
    c.metric("Active agent", o["active_agent"])
    d.metric("Tasks completed", f"{o['done']}/{o['total']}")
    st.progress(o["progress"], text=f"{o['progress']:.0%} complete ({o['done']} of {o['total']} tasks)")

    e, f, g, h, i, j = st.columns(6)
    e.metric("Elapsed time", monitor.format_duration(o["elapsed_s"]),
             help="Total time tasks spent running, from the recorded task state changes.")
    f.metric("Recorded cost (USD)", f"{o['cost_usd']:.4f}", help=f"Budget: ${o['budget_usd']:.2f}. "
             "From provider usage data. Offline tests make no API calls.")
    g.metric("Unknown-cost requests", o["unknown_cost_requests"],
             help="Requests for which the provider returned no cost. They are not counted as zero.")
    h.metric("Images", o["images"], help=f"{o['images_approved']} approved of {o['images']} generated")
    i.metric("Voiceovers", o["narration"],
             help=f"{o['narration_approved']} approved of {o['narration']} generated")
    j.metric("Video clips", o["clips"], help=f"{o['clips_approved']} approved of {o['clips']} generated")

    k, m, n = st.columns([1, 1, 4])
    k.metric("Retries", o["retries"], help="Times a task was sent back to redo work.")
    m.metric("Rejections", o["rejections"], help="Times a guard, judge or the Director rejected work.")
    n.markdown(f"**Latest agent result**  \n{o['latest_result']}")
    last = o["last_activity"].strftime("%Y-%m-%d %H:%M:%S UTC") if o["last_activity"] else "none recorded"
    problems = ", ".join(o["problem_tasks"]) or "none"
    st.caption(f"Status: {o['status'].replace('_', ' ')} · final approval: {o['final_approval']} · budget "
               f"${o['budget_usd']:.2f} · failed or rejected tasks now: {problems} · last activity: {last}")

    for block in monitor.blockers(snapshot):
        notice = st.error if block["state"] == "failed" else st.warning
        where = f"at `{block['task']}` ({block['agent']})" if block["task"] != "-" else ""
        notice(f"**Blocked {where} - {block['state']}**\n\n**Reason:** {block['reason']}\n\n"
               f"**Next action:** {block['action']} Use the Recovery view, or the command line:")
        st.code("\n".join(block["commands"]), language="bash")


def agents_section(snapshot: dict, settings) -> None:
    st.subheader("Agent activity")
    rows = monitor.agent_activity(snapshot)
    table([{k: v for k, v in row.items() if k != "id"} for row in rows])

    names = {row["Agent"]: row["id"] for row in rows}
    chosen = st.selectbox("Agent details", list(names), key="agent_detail")
    details = monitor.agent_details(snapshot, names[chosen])
    with st.expander(f"{chosen}: tasks, attempts and execution time ({len(details['tasks'])})", expanded=True):
        table(details["tasks"])
    with st.expander(f"{chosen}: model calls ({len(details['runs'])})"):
        table(details["runs"])
    with st.expander(f"{chosen}: evaluations it made ({len(details['made'])})"):
        table(details["made"])
    with st.expander(f"{chosen}: evaluations of its work ({len(details['received'])})"):
        table(details["received"])
    with st.expander(f"{chosen}: feedback and revision requests ({len(details['revisions'])})"):
        table(details["revisions"])
    with st.expander(f"{chosen}: errors ({len(details['errors'])})"):
        table(details["errors"])
    with st.expander(f"{chosen}: inputs and outputs"):
        st.markdown(f"**Reads:** {details['reads'] or 'not recorded for this component'}")
        if details["decisions"]:
            st.markdown("**Decisions**")
            table([{"Time": d["created_at"], "About": d["subject"], "Decision": d["decision"],
                    "Reason": d["reason"]} for d in details["decisions"]])
        if details["assets"]:
            st.markdown("**Media produced**")
            table(details["assets"])
        if details["writes"] and st.checkbox("Load the stored outputs (can be large)", key="load_outputs"):
            try:
                outputs = monitor.agent_outputs(settings, snapshot["project"]["id"], names[chosen])
            except monitor.DatabaseBusy:
                outputs = {}
                st.caption("The database is busy; try again in a moment.")
            for kind, content in outputs.items():
                st.markdown(f"**{kind}**")
                st.json(content, expanded=False)
            if not outputs:
                st.caption("No stored output yet.")
        elif not details["writes"] and not details["assets"]:
            st.caption("This component stores its results as evaluations, shown above.")


def events_section(snapshot: dict, newest_first: bool) -> None:
    failed = monitor.evaluation_messages(snapshot, failed_only=True)
    st.subheader("Evaluation messages")
    with st.expander(f"Failed checks with full feedback ({len(failed)})", expanded=bool(failed)):
        table(failed)
    with st.expander(f"All guard and judge results ({len(snapshot['evaluations'])})"):
        table(monitor.evaluation_messages(snapshot, failed_only=False), height=320)

    st.subheader("Event feed")
    feed = monitor.event_feed(snapshot)
    kinds = sorted({row["Event"] for row in feed})
    chosen = st.multiselect("Show only these event types (empty = all)", kinds, key="event_filter")
    rows = [row for row in feed if not chosen or row["Event"] in chosen]
    st.caption(f"{len(rows)} of {len(feed)} recorded events, {'newest' if newest_first else 'oldest'} first. "
               "Times are UTC with one-second precision; events within the same second may be reordered.")
    table(rows[::-1] if newest_first else rows, height=380)
    with st.expander("Execution log (last 40 lines)"):
        st.code("\n".join(snapshot["log_tail"]) or "No log file yet.", language="text")
    if snapshot["limitations"]:
        with st.expander(f"Known limitations of this production ({len(snapshot['limitations'])})"):
            for text in snapshot["limitations"]:
                st.markdown(f"- {text}")


def monitor_view(settings, project_id: str, newest_first: bool) -> None:
    snapshot = load(settings, project_id)
    if snapshot is None:
        st.info("This project is not in the database (anymore).")
        return
    # A status change alters the refresh rate and the other views: redraw the whole page once.
    marker = (monitor.media_signature(snapshot), snapshot["project"]["status"])
    if st.session_state.get("page_marker") not in (None, marker):
        st.session_state["page_marker"] = marker
        st.rerun()
    st.session_state["page_marker"] = marker
    stage_section(snapshot)
    overview_section(snapshot)
    agents_section(snapshot, settings)
    events_section(snapshot, newest_first)


# ---- Media -----------------------------------------------------------------------------
def media_view(snapshot: dict, settings) -> None:
    st.subheader("Media")
    found = monitor.media(snapshot, settings)
    if any(a.is_sample for a in snapshot["assets"]):
        st.caption("Media marked SAMPLE are locally generated placeholders, not AI-generated output.")

    left, right = st.columns([2, 1])
    with left:
        video = found["video"]
        if video is None:
            st.info("No video has been rendered yet.")
        elif video["state"] == "final":
            st.success("**Final video - APPROVED and exported.**")
        elif video["state"] == "approved":
            st.success("**Rendered video - APPROVED by the Director.** The thumbnail or the final "
                       "package export is not complete yet.")
        else:
            st.warning("**Rendered video - PRELIMINARY, not yet approved.** Final review has not passed.")
        if video:
            st.video(video["path"])
    with right:
        if found["thumbnail"] is not None:
            st.success("**Thumbnail - APPROVED.**")
            show_image(st, found["thumbnail"], Path(found["thumbnail"].path).name)
        elif found["thumbnail_candidates"]:
            latest = found["thumbnail_candidates"][-1]
            st.warning("**Thumbnail candidate - PRELIMINARY, not approved.**")
            show_image(st, latest, f"{Path(latest.path).name} · {latest.status}")
        else:
            st.info("No thumbnail has been generated yet.")

    def label(asset) -> str:
        state = "APPROVED" if asset.status in ("approved", "human_approved") else f"preliminary ({asset.status})"
        return f"{Path(asset.path).stem} · {state}{' · SAMPLE' if asset.is_sample else ''}"

    images = found["images"]
    approved = [a for a in images if a.status in ("approved", "human_approved")]
    with st.expander(f"Images ({len(approved)} approved of {len(images)} generated)",
                     expanded=bool(images) and len(images) <= 12):
        show_all = st.checkbox("Include rejected and unreviewed attempts", key="show_all_images")
        shown = images if show_all else approved
        columns = st.columns(4)
        for n, asset in enumerate(shown):
            show_image(columns[n % 4], asset, label(asset))
        if not shown:
            st.caption("Nothing to show yet.")
        extra = found["references"] + (found["thumbnail_candidates"] if show_all else [])
        if extra:
            st.caption("Character references" + (" and all thumbnail candidates" if show_all else ""))
            columns = st.columns(4)
            for n, asset in enumerate(extra):
                show_image(columns[n % 4], asset, label(asset))

    with st.expander(f"Audio ({len(found['audio'])} narration segments)"):
        for asset in found["audio"]:
            duration = f" · {asset.duration_s:.1f}s" if asset.duration_s else ""
            st.caption(label(asset) + duration)
            st.audio(asset.path)
        if not found["audio"]:
            st.caption("No narration has been generated yet.")

    with st.expander(f"Video clips ({len(found['clips'])})"):
        for asset in found["clips"]:
            st.caption(label(asset))
            st.video(asset.path)
        if not found["clips"]:
            st.caption("No clips (AI video generation is optional and off unless enabled for the project).")


# ---- New production --------------------------------------------------------------------
def launch_inputs(settings, prefix: str) -> tuple:
    """Mode, budget, clip option and the paid-run confirmation shared by single and batch launch."""
    choices = launcher.mode_choices(settings)
    mode = choices[st.radio("Mode", list(choices), key=f"{prefix}_mode")]
    budget = st.number_input("Maximum budget per production (USD)", min_value=0.0, step=0.5, format="%.2f",
                             value=float(settings.max_budget_usd), key=f"{prefix}_budget", disabled=not mode.live)
    video = st.checkbox("Enable AI-generated video clips (optional, more expensive)", key=f"{prefix}_video")
    return mode, budget, video


def new_production_view(settings) -> None:
    st.subheader("Start a production")
    topic = st.text_input("Video topic", placeholder="Cybersecurity", key="new_topic")
    title = launcher.title_for(settings, topic)
    st.markdown(f"**Title:** {title or '(enter a topic)'}")
    mode, budget, video = launch_inputs(settings, "new")

    st.markdown("**Estimated scope**")
    st.table(launcher.scope(settings, mode, video))
    key_ok = launcher.api_key_configured(settings)
    confirmed = True
    if mode.live:
        st.caption(f"OpenRouter API key: {'configured' if key_ok else 'MISSING - add it to .env first'}")
        confirmed = st.checkbox(f"I confirm paid OpenRouter requests up to ${budget:.2f} for this production.",
                                key="new_confirm")
    ready = bool(title) and confirmed and (not mode.live or (key_ok and budget > 0))
    if st.button("Start Production", type="primary", disabled=not ready, key="new_start"):
        try:
            job = launcher.launch(settings, topic, mode, budget, enable_video=video, confirmed=confirmed,
                                  request_id=token("new"))
        except (ConfigError, OpenRouterError) as exc:
            st.error(str(exc))
            return
        new_token("new")
        go_to("Monitor", job["project_id"])
    st.caption("The production runs in a separate worker process. You can switch views, reload this page or "
               "close the dashboard; it keeps running. One production runs at a time; others wait in the queue.")


# ---- Batch & queue -----------------------------------------------------------------------
def queue_section(settings) -> None:
    try:
        q = monitor.queue(settings)
    except monitor.DatabaseBusy:
        st.caption("The database is busy; the queue will reappear on the next refresh.")
        return
    waiting = len(q["queued"])
    if q["worker_alive"]:
        st.info(f"Worker: running. {waiting} production(s) waiting.")
    elif waiting:
        st.warning(f"Worker: not running, but {waiting} production(s) are waiting.")
        if st.button("Start the worker", key="start_worker"):
            launcher.start_worker(settings)
            st.rerun()
    else:
        st.caption("Worker: idle (it starts when a production is queued and exits when the queue is empty).")

    st.markdown("**Running now**")
    table(q["running"])
    st.markdown(f"**Queued ({waiting}, runs in this order)**")
    table([{"Position": n, **row} for n, row in enumerate(q["queued"], start=1)])
    st.markdown(f"**Failed, blocked or stopped ({len(q['problem'])})** - see the Recovery view")
    table(q["problem"])
    with st.expander(f"Completed ({len(q['completed'])}) and cancelled ({len(q['cancelled'])})"):
        table(q["completed"] + q["cancelled"])

    if q["queued"]:
        with st.expander("Cancel a queued production"):
            job_id = st.selectbox("Queued job", [r["Job"] for r in q["queued"]], key="cancel_job",
                                  format_func=lambda j: f"#{j} · " + next(
                                      r["Project"] for r in q["queued"] if r["Job"] == j))
            if st.checkbox("Yes, remove this job from the queue", key="cancel_confirm") and \
                    st.button("Cancel job", key="cancel_button"):
                launcher.cancel(settings, job_id)
                st.rerun()
    if q["running"]:
        with st.expander("Stop the running production"):
            st.caption("The worker finishes the task in progress, then stops. Completed tasks and approved "
                       "assets are kept and the production can be resumed from the Recovery view.")
            if st.checkbox("Yes, stop after the current task", key="stop_confirm") and \
                    st.button("Stop production", key="stop_button"):
                launcher.stop_after_current_task(settings, q["running"][0]["Job"])
                st.rerun()


def batch_view(settings, auto: bool) -> None:
    st.subheader("Batch production")
    text = st.text_area("Topics, one per line", placeholder="Cybersecurity\nSoftware Engineering\nMilitary",
                        key="batch_topics", height=120)
    topics = launcher.jobs.parse_topics(text)
    mode, budget, video = launch_inputs(settings, "batch")
    if topics:
        st.markdown(f"**{len(topics)} independent production(s), run one at a time in this order:**  \n"
                    + "  \n".join(f"{n}. {launcher.title_for(settings, t)}" for n, t in enumerate(topics, 1)))
    key_ok = launcher.api_key_configured(settings)
    confirmed = True
    if mode.live:
        st.caption(f"OpenRouter API key: {'configured' if key_ok else 'MISSING - add it to .env first'}")
        confirmed = st.checkbox(
            f"I confirm paid OpenRouter requests up to ${budget:.2f} for EACH of the {len(topics)} productions "
            f"(at most ${budget * len(topics):.2f} in total).", key="batch_confirm")
    ready = bool(topics) and confirmed and (not mode.live or (key_ok and budget > 0))
    if st.button("Queue batch", type="primary", disabled=not ready, key="batch_start"):
        try:
            launcher.launch_batch(settings, text, mode, budget, enable_video=video, confirmed=confirmed,
                                  request_id=token("batch"))
        except (ConfigError, OpenRouterError) as exc:
            st.error(str(exc))
            return
        new_token("batch")
        st.rerun()

    st.divider()
    st.subheader("Production queue")
    st.fragment(run_every=RUNNING_REFRESH_S if auto else None)(queue_section)(settings)


# ---- Recovery ----------------------------------------------------------------------------
def recovery_view(snapshot: dict, settings) -> None:
    st.subheader("Recovery")
    project = snapshot["project"]
    blocks = monitor.blockers(snapshot)
    for block in blocks:
        st.markdown(f"**`{block['task']}` - {block['state']}** ({block['agent']})")
        st.text(block["reason"])                           # the full recorded reason, unshortened
    interrupted = [j for j in snapshot["jobs"] if j["status"] in ("interrupted", "stopped")][-1:]
    for job in interrupted:
        st.markdown(f"**Job #{job['id']} - {job['status']}**")
        st.text(job["message"])

    options = launcher.recovery_options(project, snapshot["tasks"], snapshot["jobs"])
    if options["busy"]:
        st.info("This project already has a queued or running job. Follow it in the Monitor view.")
        return
    if not options["actions"]:
        st.success("Nothing to recover: this project has no failed or blocked task."
                   if project["status"] == "completed" or not blocks else
                   "No recovery action is available from the dashboard for this state.")
        if options.get("cli_only"):
            st.warning("Compliance decisions are made from the command line only.")
        return
    if options.get("cli_only"):
        st.warning(f"`{', '.join(options['cli_only'])}` is a compliance block. It cannot be accepted from the "
                   "dashboard; a person must record that decision with `python run.py override ...`.")

    action = st.radio("Recovery action", list(options["actions"]), format_func=options["actions"].get,
                      key="recover_action")
    task, reason = None, ""
    if action == "override":
        task = options["override_tasks"][0]
        st.caption(f"This records a human decision for `{task}`. Rejected work is not approved automatically: "
                   "you take responsibility for accepting it, and the decision is stored in the reports.")
        reason = st.text_input("Reason for accepting the current result (required)", key="recover_reason")
    cost = snapshot["cost"]["known_usd"]
    budget = st.number_input(f"Budget for this project (USD) - ${cost:.4f} recorded so far", min_value=0.0,
                             step=0.5, format="%.2f", value=float(project["budget_usd"]), key="recover_budget")
    paid = project["mode"] != "dry_run"
    confirmed = st.checkbox(
        f"Yes, {'continue this paid production (up to $' + format(budget, '.2f') + ' in total)' if paid else 'continue this production'}.",
        key="recover_confirm")
    ready = confirmed and (action != "override" or bool(reason.strip()))
    if st.button("Run recovery action", type="primary", disabled=not ready, key="recover_start"):
        try:
            launcher.recover(settings, project["id"], action, confirmed=confirmed, request_id=token("recover"),
                             new_budget=budget, task=task, reason=reason)
        except (ConfigError, OpenRouterError) as exc:
            st.error(str(exc))
            return
        new_token("recover")
        go_to("Monitor", project["id"])


# ---- page ---------------------------------------------------------------------------------
def main() -> None:
    st.title("AI YouTube Production Studio")
    try:
        settings = load_settings()
        launcher.housekeeping(settings)
        projects = monitor.list_projects(settings)
    except ConfigError as exc:
        st.error(str(exc))
        return
    except (monitor.DatabaseBusy, OSError):
        st.warning("The database is busy right now. Reload the page in a moment.")
        return

    by_id = {p["id"]: p for p in projects}
    if "goto" in st.session_state:                   # requested by an action on the previous run
        view, project_id = st.session_state.pop("goto")
        st.session_state["view"] = view
        if project_id in by_id:
            st.session_state["project"] = project_id
    if st.session_state.get("project") not in by_id:
        st.session_state.pop("project", None)

    view = st.sidebar.radio("View", VIEWS, key="view")
    project_id = st.sidebar.selectbox(
        "Production project", list(by_id), key="project",
        format_func=lambda pid: f"{STATUS_MARK.get(by_id[pid]['status'], '•')} {by_id[pid]['status']} · "
                                f"{by_id[pid]['mode']} · {pid}") if by_id else None
    auto = st.sidebar.toggle("Auto-refresh", value=True)
    newest_first = st.sidebar.toggle("Newest events first", value=True)
    st.sidebar.caption(f"Database: {settings.db_path}")

    # Run the presentation animation without making API requests.
    if view == "Presentation demo":
        st.info(
            "Simulated presentation demo. "
            "No API calls, credits, or production records are used."
        )
        stage_section(demo=True)
        return

    if view == "New production":
        new_production_view(settings)
        return
    if view == "Batch & queue":
        batch_view(settings, auto)
        return
    if project_id is None:
        st.info("No projects yet. Open the New production view to start one.")
        return

    snapshot = load(settings, project_id)
    if snapshot is None:
        st.info("This project could not be read. Reload the page in a moment.")
        return
    project = snapshot["project"]
    st.header(project["title"])
    st.caption(f"Project ID: `{project['id']}` · topic: {project['topic']} · created {project['created_at']}")

    if view == "Monitor":
        running = project["status"] == "running" or any(j["status"] in ("queued", "running")
                                                        for j in snapshot["jobs"])
        every = (RUNNING_REFRESH_S if running else IDLE_REFRESH_S) if auto else None
        st.sidebar.caption("Auto-refresh is off." if not auto else
                           f"Refreshing every {every} s ({'production running' if running else 'idle'}).")
        st.session_state["page_marker"] = (monitor.media_signature(snapshot), project["status"])
        st.fragment(run_every=every)(monitor_view)(settings, project_id, newest_first)
    elif view == "Media":
        media_view(snapshot, settings)
    else:
        recovery_view(snapshot, settings)


main()