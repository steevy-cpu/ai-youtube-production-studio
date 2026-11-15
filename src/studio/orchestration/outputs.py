"""Keeps `output/<project>/` useful at every stage, not only after a successful export.

Layout:
  STATUS.txt                         APPROVED or INCOMPLETE, with the reason
  preliminary/UNAPPROVED_video.mp4   the rendered video, until the production is approved
  assets/images|audio|references|thumbnails|video/   media generated so far
  artifacts/*.json, script.txt       plan, script, storyboard, prompts, timeline, reports
  video.mp4, thumbnail.png, ...      FINAL files, written only by the export task

Everything here is a copy made from local files and SQLite records. It never calls a model.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from studio.config import Settings
from studio.memory import APPROVED, MemoryStore
from studio.schemas import Script
from studio.utils import now_iso, safe_path

PRELIMINARY_VIDEO = "UNAPPROVED_video.mp4"
ASSET_DIRS = {"image": "images", "audio": "audio", "reference": "references", "video": "video"}


def _copy(source: str, target: Path) -> bool:
    if not Path(source).is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def is_approved(store: MemoryStore, project_id: str) -> bool:
    project = store.get_project(project_id)
    return project["final_approval"] == "approved" and all(
        t["state"] == "COMPLETED" for t in store.tasks(project_id))


def publish_work_files(store: MemoryStore, settings: Settings, project_id: str) -> Path:
    """Mirror the project's current media and artifacts into its output directory."""
    out = safe_path(settings.output_dir, project_id)
    out.mkdir(parents=True, exist_ok=True)
    approved = is_approved(store, project_id)

    # ---- media: approved assets only, except thumbnail candidates (all, labelled) ----
    assets_dir = out / "assets"
    shutil.rmtree(assets_dir, ignore_errors=True)        # rebuilt each time, so nothing stale remains
    counts: dict[str, int] = {}
    for kind, folder in ASSET_DIRS.items():
        for asset in store.assets(project_id, kind, status=APPROVED):
            counts[folder] = counts.get(folder, 0) + _copy(asset.path, assets_dir / folder / Path(asset.path).name)
    for asset in store.assets(project_id, "thumbnail"):
        name = f"{Path(asset.path).stem}__{asset.status}.png"
        counts["thumbnails"] = counts.get("thumbnails", 0) + _copy(asset.path, assets_dir / "thumbnails" / name)

    # ---- structured artifacts (latest version of each) ----
    artifacts_dir = out / "artifacts"
    shutil.rmtree(artifacts_dir, ignore_errors=True)
    artifacts_dir.mkdir(parents=True)
    grouped: dict[str, list] = {}
    for row in store.latest_artifacts(project_id):
        grouped.setdefault(row["kind"], []).append(json.loads(row["data"]))
    for kind, records in grouped.items():
        content = records[0] if len(records) == 1 and kind not in ("chapter", "image_prompt", "narration") \
            else records
        (artifacts_dir / f"{kind}.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    script = store.load(project_id, "script", Script)
    if script:
        (artifacts_dir / "script.txt").write_text(
            f"{script.title}\n\n" + "\n\n".join(f"## {c.index}. {c.title}\n{c.narration}"
                                                for c in script.chapters) + "\n", encoding="utf-8")

    # ---- rendered video: preliminary copy until the production is approved ----
    preliminary = out / "preliminary"
    videos = store.assets(project_id, "final_video")
    if approved:
        shutil.rmtree(preliminary, ignore_errors=True)   # the approved video.mp4 now sits at the top level
    elif videos and _copy(videos[-1].path, preliminary / PRELIMINARY_VIDEO):
        _copy(str(Path(videos[-1].path).with_suffix(".srt")), preliminary / "UNAPPROVED_captions.srt")

    _write_status(store, project_id, out, approved, counts, bool(videos))
    return out


def _write_status(store: MemoryStore, project_id: str, out: Path, approved: bool, counts: dict[str, int],
                  rendered: bool) -> None:
    project, tasks = store.get_project(project_id), store.tasks(project_id)
    done = sum(t["state"] == "COMPLETED" for t in tasks)
    lines = [f"Project: {project['id']} - {project['title']}", f"Mode: {project['mode']}",
             f"Updated: {now_iso()}", f"Tasks completed: {done}/{len(tasks)}", ""]
    if approved:
        lines += ["STATUS: APPROVED", "",
                  "All reviews and approvals passed. The final files are in this folder:",
                  "video.mp4, thumbnail.png, metadata.json, description.txt, captions.srt and the reports."]
    else:
        lines += ["STATUS: INCOMPLETE - NOT APPROVED", "",
                  "This production has not passed every review and approval. Nothing in this folder is an",
                  "approved final output. Files under preliminary/, assets/ and artifacts/ are work in",
                  "progress, kept so that nothing is lost when a task fails or waits for human review.",
                  "The final video.mp4, thumbnail.png and metadata.json appear here only after export.", ""]
        if rendered:
            lines.append(f"Rendered but unapproved video: preliminary/{PRELIMINARY_VIDEO}")
        lines.append("Tasks not completed:")
        for task in tasks:
            if task["state"] != "COMPLETED":
                reason = f" - {task['error'][:300]}" if task["error"] else ""
                lines.append(f"  {task['state']:<12} {task['name']}{reason}")
    lines += ["", "Files collected: " + (", ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "none")]
    (out / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
