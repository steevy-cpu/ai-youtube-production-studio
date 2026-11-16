"""Command line interface (primary user interface). Run `python -m studio --help`."""
from __future__ import annotations

import argparse
import sys

from studio.config import ConfigError, Mode, Settings, load_settings
from studio.memory import MemoryStore
from studio.orchestration import jobs
from studio.orchestration.outputs import is_approved, publish_work_files
from studio.orchestration.pipeline import apply_override, create_project, run_project
from studio.rendering.ffmpeg import ffmpeg_available
from studio.services.openrouter import OpenRouterClient, OpenRouterError


def _settings(args: argparse.Namespace) -> Settings:
    overrides = {}
    if getattr(args, "enable_video", False):
        overrides["enable_video"] = True
    if getattr(args, "duration", None):
        overrides["target_duration_s"] = args.duration
    return load_settings(args.env_file, **overrides)


def _confirm_live(settings: Settings, project: dict, assume_yes: bool) -> None:
    """Budget confirmation gate in front of every paid run."""
    s = settings.for_mode(Mode(project["mode"]))
    s.require_api_key()
    if project["budget_usd"] <= 0:
        raise ConfigError("a live run needs a budget above 0 (use --budget USD)")
    print(f"\nLIVE RUN - this makes paid OpenRouter requests.\n"
          f"  project        : {project['id']}  ({project['title']})\n"
          f"  mode           : {project['mode']}  target {s.target_duration_s:.0f}s at {s.width}x{s.height}\n"
          f"  budget         : ${project['budget_usd']:.2f} (hard stop; max {s.max_media_requests} media "
          f"requests)\n"
          f"  LLM / director : {s.llm_model} / {s.director_model}\n"
          f"  vision judge   : {s.vision_model}\n"
          f"  thumbnail      : {s.thumbnail_llm} (prompt + judge) / {s.thumbnail_vision} (reviews) / "
          f"{s.thumbnail_image if s.enable_images else 'placeholder'} (image)\n"
          f"  images         : {s.image_model if s.enable_images else 'DISABLED (placeholders)'}"
          f"{f' (rejected images redrawn by {s.image_retry})' if s.enable_images and s.image_retry != s.image_model else ''}\n"
          f"  narration      : {f'{s.tts_model} voice {s.tts_voice}' if s.enable_tts else 'DISABLED (placeholder tone)'}\n"
          f"  video clips    : {f'{s.video_model} (max {s.max_video_clips})' if s.enable_video else 'disabled'}")
    if assume_yes:
        print("  confirmed with --yes")
        return
    if input("Type 'yes' to confirm this budget and start: ").strip().lower() != "yes":
        raise ConfigError("cancelled: live run not confirmed")


def _run(settings: Settings, store: MemoryStore, project_id: str, args: argparse.Namespace,
         resume: bool = False) -> int:
    project = store.get_project(project_id)
    if project is None:
        raise ConfigError(f"unknown project '{project_id}' (see `studio list`)")
    live = Mode(project["mode"]).live
    if live:
        _confirm_live(settings, project, args.yes)
    print(f"Running project {project_id} in {project['mode']} mode ...")
    status = run_project(settings, store, project_id, confirm_live=live, resume=resume,
                         retry_escalated=getattr(args, "retry_escalated", False), verbose=args.verbose)
    _print_status(settings, store, project_id)
    if status != "completed":
        print(f"\nWork so far is kept in {settings.output_dir / project_id} (see STATUS.txt; not approved).")
        print("Not finished. Fix the cause, then `studio resume " + project_id + "`. For NEEDS_REVIEW "
              "tasks either `resume --retry-escalated` or record a decision with `studio override`.")
    return 0 if status == "completed" else 1


def _print_status(settings: Settings, store: MemoryStore, project_id: str) -> None:
    project = store.get_project(project_id)
    if project is None:
        raise ConfigError(f"unknown project '{project_id}'")
    tasks = store.tasks(project_id)
    done = sum(t["state"] == "COMPLETED" for t in tasks)
    running = next((t["name"] for t in tasks if t["state"] != "COMPLETED"), "finished")
    print(f"\nProject {project['id']}: {project['title']}\n"
          f"  mode {project['mode']} | status {project['status']} | final approval "
          f"{project['final_approval']} | progress {done}/{len(tasks)} tasks | current stage: {running}")
    for t in tasks:
        line = f"  {t['state']:<12} {t['name']:<28} {t['agent']:<18} attempts {t['attempts']}/{t['max_attempts']}"
        detail = t["error"] if t["state"] in ("FAILED", "NEEDS_REVIEW") else t["summary"]
        print(line + (f"  {detail[:110]}" if detail else ""))
    cost = store.cost_summary(project_id)
    print(f"  cost: ${cost['known_usd']:.4f} recorded of ${project['budget_usd']:.2f} budget; "
          f"{cost['unknown_cost_requests']} request(s) with UNKNOWN cost; {cost['media_requests']} media "
          f"request(s)")
    print(f"  output: {settings.output_dir / project_id}  "
          f"({'APPROVED final files' if is_approved(store, project_id) else 'incomplete, not approved'})")


def cmd_new(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    project_id = create_project(settings, store, args.topic, Mode(args.mode), title=args.title,
                                budget_usd=args.budget, sources_dir=args.sources)
    print(f"Created project {project_id} ({args.mode}). Start it with: studio run {project_id}")
    return 0


def _create_and_run(mode: Mode):
    def command(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
        project_id = create_project(settings, store, args.topic, mode, title=args.title,
                                    budget_usd=getattr(args, "budget", None), sources_dir=args.sources,
                                    enable_video=getattr(args, "enable_video", False))
        return _run(settings, store, project_id, args)
    return command


def cmd_run(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    return _run(settings, store, args.project, args)


def cmd_resume(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    if args.budget is not None:
        if store.get_project(args.project) is None:
            raise ConfigError(f"unknown project '{args.project}'")
        store.update_project(args.project, budget_usd=args.budget)
    return _run(settings, store, args.project, args, resume=True)


def cmd_status(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    _print_status(settings, store, args.project)
    errors = store.errors(args.project)
    if errors:
        print("  recent errors:")
        for e in errors[-5:]:
            print(f"    [{e['task']}/{e['agent']}] {e['message'][:160]}")
    return 0


def cmd_watch(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    from studio.watch import watch
    return watch(settings.db_path, args.project, history=args.history, tail=args.tail,
                 follow=not args.once)


def cmd_list(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    projects = store.list_projects()
    for p in projects:
        print(f"{p['id']:<42} {p['mode']:<8} {p['status']:<13} {p['title']}")
    if not projects:
        print("No projects yet. Try: studio dry-run \"Cybersecurity\"")
    return 0


def cmd_report(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    project = store.get_project(args.project)
    if project is None:
        raise ConfigError(f"unknown project '{args.project}'")
    cost = store.cost_summary(args.project)
    print(f"Costs for {args.project} (budget ${project['budget_usd']:.2f})")
    for row in cost["by_kind"]:
        unknown = f" + {row['unknown']} with UNKNOWN cost" if row["unknown"] else ""
        print(f"  {row['kind']:<6} {row['model']:<44} {row['requests']:>4} requests  ${row['usd']:.4f}{unknown}")
    print(f"  total recorded ${cost['known_usd']:.4f}; requests with unknown cost: "
          f"{cost['unknown_cost_requests']} (not counted as zero)")
    if project["mode"] == Mode.DRY_RUN.value:
        print("  (dry run: no API calls were made, so nothing was spent)")
    print("\nEvaluations")
    for row in store.evaluation_summary(args.project):
        score = "-" if row["avg_score"] is None else f"{row['avg_score']:.2f}"
        print(f"  {row['evaluator']:<22} {row['passed']}/{row['total']} passed  avg score {score}")
    revisions = store.revisions(args.project)
    print(f"\nRevision requests: {len(revisions)}")
    for r in revisions[-10:]:
        print(f"  {r['requested_by']} -> {r['responsible_agent']} ({r['target_kind']}, {r['status']})")
    decisions = store.decisions(args.project)
    print(f"\nDecisions: {len(decisions)}")
    for d in decisions:
        print(f"  {d['actor']:<9} {d['subject']:<20} {d['decision']:<24} {d['reason'][:80]}")
    limitations = store.recall(args.project, "limitations", [])
    if limitations:
        print("\nKnown limitations of this production")
        for text in limitations:
            print(f"  - {text}")
    return 0


def cmd_override(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    released = apply_override(store, args.project, args.task, args.reason)
    print(f"Recorded human decision for {', '.join(released)}. Continue with: studio resume {args.project}")
    return 0


def cmd_worker(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    """Process the local job queue (jobs are queued and confirmed from the dashboard)."""
    processed = jobs.run_worker(settings, store, log=lambda message: print(message, flush=True))
    print(f"worker finished: {processed} job(s) processed", flush=True)
    return 0


def cmd_queue(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    jobs.reconcile(store)
    rows = store.jobs()
    print(f"worker: {'running' if jobs.worker_alive(store) else 'not running'}")
    for job in rows:
        print(f"  #{job['id']:<4} {job['status']:<12} {job['kind']:<7} {job['project_id']:<42} "
              f"{job['message'][:80]}")
    if not rows:
        print("  the queue is empty")
    return 0


def cmd_collect(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    if store.get_project(args.project) is None:
        raise ConfigError(f"unknown project '{args.project}'")
    out = publish_work_files(store, settings, args.project)       # local copies only, no API call
    state = "APPROVED" if is_approved(store, args.project) else "INCOMPLETE - NOT APPROVED"
    print(f"Collected the project's files into {out} ({state}). See {out / 'STATUS.txt'}")
    return 0


def cmd_check(args: argparse.Namespace, settings: Settings, store: MemoryStore) -> int:
    print(f"Configuration OK. Target {settings.target_duration_s:.0f}s, {settings.width}x{settings.height}, "
          f"~{settings.target_words} words.")
    print(f"ffmpeg/ffprobe: {'found' if ffmpeg_available() else 'MISSING (required)'}")
    print(f"OPENROUTER_API_KEY: {'set' if settings.openrouter_api_key else 'not set (dry run only)'}")
    if args.models:
        caps, problems = OpenRouterClient(settings).capabilities()    # free catalog requests only
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        for note in caps["notes"]:
            print(f"  note: {note}")
        print("Model check: " + ("FAILED" if problems else "all configured models look usable"))
        return 1 if problems else 0
    return 0 if ffmpeg_available() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studio", description="Local multi-agent AI video production studio")
    parser.add_argument("--env-file", default=".env", help="path to the .env file (default: .env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the run log to the terminal")
    sub = parser.add_subparsers(dest="command", required=True)

    def topic_command(name: str, text: str, handler, live: bool):
        p = sub.add_parser(name, help=text)
        p.add_argument("topic", help='video topic, e.g. "Cybersecurity"')
        p.add_argument("--title", help="override the generated video title")
        p.add_argument("--sources", help="directory of local .txt/.md research sources")
        p.add_argument("--duration", type=float, help="target duration in seconds (overrides the config)")
        if live:
            p.add_argument("--budget", type=float, help="maximum spend in USD (default MAX_BUDGET_USD)")
            p.add_argument("--yes", action="store_true", help="confirm the budget without the prompt")
            p.add_argument("--enable-video", action="store_true", help="allow AI video clip generation")
        else:
            p.set_defaults(yes=True)
        p.set_defaults(func=handler)
        return p

    topic_command("dry-run", "full pipeline with labelled sample data, no API calls", _create_and_run(Mode.DRY_RUN), False)
    topic_command("preview", "short live sample (paid, needs confirmation)", _create_and_run(Mode.PREVIEW), True)
    topic_command("produce", "full live production (paid, needs confirmation)", _create_and_run(Mode.FULL), True)

    p = sub.add_parser("new", help="create a project without running it")
    p.add_argument("topic")
    p.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.DRY_RUN.value)
    p.add_argument("--title")
    p.add_argument("--sources")
    p.add_argument("--budget", type=float)
    p.add_argument("--duration", type=float)
    p.set_defaults(func=cmd_new)

    for name, handler, text in (("run", cmd_run, "run a created project"),
                                ("resume", cmd_resume, "resume a failed or stopped project")):
        p = sub.add_parser(name, help=text)
        p.add_argument("project")
        p.add_argument("--yes", action="store_true", help="confirm the budget without the prompt")
        p.add_argument("--enable-video", action="store_true")
        if name == "resume":
            p.add_argument("--budget", type=float, help="set a new budget before resuming")
            p.add_argument("--retry-escalated", action="store_true",
                           help="also retry NEEDS_REVIEW tasks with fresh attempts")
        p.set_defaults(func=handler)

    for name, handler, text in (("status", cmd_status, "show stage, task states and errors"),
                                ("report", cmd_report, "show cost and evaluation summaries"),
                                ("collect", cmd_collect, "copy the project's files so far into output/")):
        p = sub.add_parser(name, help=text)
        p.add_argument("project")
        p.set_defaults(func=handler)

    p = sub.add_parser("watch", help="live log: follow what each agent is doing and saying (read-only)")
    p.add_argument("project", nargs="?", help="default: the production running now, else the latest one")
    p.add_argument("--history", action="store_true", help="print the whole log from the start first")
    p.add_argument("--tail", type=int, default=25, help="how many past events to show first (default 25)")
    p.add_argument("--once", action="store_true", help="print and exit instead of following")
    p.set_defaults(func=cmd_watch)

    sub.add_parser("list", help="list local projects").set_defaults(func=cmd_list)
    sub.add_parser("worker", help="run queued jobs one at a time, then exit (started by the dashboard)"
                   ).set_defaults(func=cmd_worker)
    sub.add_parser("queue", help="show the local job queue").set_defaults(func=cmd_queue)

    p = sub.add_parser("override", help="record a human decision accepting a NEEDS_REVIEW task")
    p.add_argument("project")
    p.add_argument("task")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_override)

    p = sub.add_parser("check", help="validate configuration (add --models to verify model ids, free)")
    p.add_argument("--models", action="store_true")
    p.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = _settings(args)
        store = MemoryStore(settings.db_path)
        try:
            return args.func(args, settings, store)
        finally:
            store.close()
    except (ConfigError, OpenRouterError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted - state is saved; continue with `studio resume <project>`", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
