"""End-to-end dry-run pipeline, feedback loops, recovery, overrides and FFmpeg rendering."""
import json
import sqlite3
from pathlib import Path

import pytest
from conftest import REJECT, needs_ffmpeg

from studio.config import ConfigError, Mode
from studio.orchestration.pipeline import apply_override, build_specs, create_project, run_project
from studio.rendering.ffmpeg import probe
from studio.rendering.renderer import render
from studio.schemas import AudioClip, Caption, EditingTimeline, ImagePrompt, Storyboard, ThumbnailBrief, TimelineClip
from studio.services.placeholders import sample_audio, sample_image

pytestmark = needs_ffmpeg
EXPECTED_OUTPUTS = {"video.mp4", "thumbnail.png", "metadata.json", "description.txt", "quality_report.json",
                    "compliance_report.json", "production_report.json", "captions.srt"}


def states(store, project_id):
    return {t["name"]: t["state"] for t in store.tasks(project_id)}


def test_workflow_graph_contains_every_stage_and_the_thumbnail_branch(settings):
    specs = {s.name: s for s in build_specs(settings)}
    for name in ("director_plan", "research", "story_writing", "script_guard", "scene_direction",
                 "timeline_guard", "prompt_review", "visual_review", "audio_qa", "rendering", "final_qa",
                 "compliance", "seo_package", "thumbnail_prompt", "thumbnail_prompt_review",
                 "thumbnail_generation", "thumbnail_review", "director_thumbnail_approval", "export"):
        assert name in specs
    assert "visual_review" not in specs["voice_direction"].depends_on        # narration is independent
    assert specs["thumbnail_generation"].max_attempts == settings.max_thumbnail_attempts


def test_end_to_end_dry_run(project, store, settings):
    project_id, client, run = project(dry_run_demo_rejections=True)
    assert run() == "completed"
    assert set(states(store, project_id).values()) == {"COMPLETED"}

    # output directory structure
    out = settings.output_dir / project_id
    assert {p.name for p in out.iterdir()} == EXPECTED_OUTPUTS | {"STATUS.txt", "assets", "artifacts"}
    assert "STATUS: APPROVED" in (out / "STATUS.txt").read_text()
    assert not (out / "preliminary").exists()                # only the approved video remains
    assert len(list((out / "assets" / "images").iterdir())) == len(store.assets(project_id, "image", status=("approved",)))
    assert {"script.txt", "storyboard.json", "image_prompt.json", "timeline.json"} <= {
        p.name for p in (out / "artifacts").iterdir()}
    info = probe(out / "video.mp4")
    assert (info["video"]["width"], info["video"]["height"]) == (320, 180) and info["audio"] is not None
    assert abs(info["duration_s"] - 24) <= 24 * settings.duration_tolerance
    from PIL import Image
    assert Image.open(out / "thumbnail.png").size == (1280, 720)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["published"] is False and "SAMPLE OUTPUT" in metadata["sample_notice"]

    # sample assets are labelled, never presented as AI output
    assets = store.assets(project_id)
    assert assets and all(a.is_sample for a in assets)
    assert all("no AI model" in a.provenance or "placeholder" in a.provenance for a in assets
               if a.kind in ("image", "audio", "thumbnail", "reference"))
    assert all(Path(a.path).is_file() for a in assets)                        # files on disk, paths in DB

    # SQLite records exist for every memory category
    db = sqlite3.connect(settings.db_path)
    for table in ("projects", "tasks", "task_events", "agent_runs", "artifacts", "assets", "evaluations",
                  "revisions", "costs", "decisions", "knowledge", "working_memory"):
        assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0, table
    db.close()
    for kind in ("brief", "plan", "style_bible", "research", "script", "storyboard", "image_prompt",
                 "narration", "timeline", "compliance", "seo", "thumbnail_brief", "thumbnail_prompt", "report"):
        assert store.artifact_counts(project_id).get(kind), kind

    # demo rejections exercised both feedback loops
    brief = store.load(project_id, "thumbnail_brief", ThumbnailBrief)
    candidates = store.assets(project_id, "thumbnail", brief.id)
    assert [c.status for c in candidates] == ["rejected", "approved"]
    assert {r["target_task"] for r in store.revisions(project_id)} == {"prompt_engineering", "thumbnail_generation"}
    assert client.calls.count("prompt.revise") == 1

    # every agent left a trace; dry run recorded no spend
    agents = {r["agent"] for r in store.agent_runs(project_id)}
    assert {"director", "research", "story_writer", "scene_director", "prompt_engineer", "prompt_judge",
            "visual_generation", "visual_judge", "voice_director", "editor", "final_qa", "seo",
            "compliance"} <= agents
    assert store.cost_summary(project_id)["known_usd"] == 0
    assert store.get_project(project_id)["final_approval"] == "approved"


def test_prompt_judge_rejection_revises_only_the_rejected_prompt(project, store):
    def reject_first(context):
        first = context["shot_indexes"][0]
        return {"verdicts": [dict(REJECT if i == first else {"approved": True, "score": 0.9}, shot_index=i)
                             for i in context["shot_indexes"]]}
    project_id, client, run = project({"judge.prompts": [reject_first]})
    assert run() == "completed"
    board = store.load(project_id, "storyboard", Storyboard)
    versions = {s.index: len(store.artifact_versions(project_id, "image_prompt", s.id)) for s in board.shots}
    assert versions[1] == 2 and all(v == 1 for i, v in versions.items() if i != 1)
    assert client.calls.count("prompt.revise") == 1 and client.calls.count("prompt.scene") == len(board.scenes)
    request = [r for r in store.revisions(project_id) if r["target_task"] == "prompt_engineering"]
    assert len(request) == 1 and request[0]["responsible_agent"] == "prompt_engineer"
    assert all(p.status == "approved" for p in store.load_all(project_id, "image_prompt", ImagePrompt).values())


def test_visual_judge_rejection_regenerates_only_that_image(project, store):
    seen = []

    def judge(context):
        seen.append(context["purpose"])
        reject = context["purpose"] == "image" and seen.count("image") == 1
        return REJECT if reject else {"approved": True, "score": 0.9}
    project_id, client, run = project({"judge.image": [judge] * 50})
    assert run() == "completed"
    board = store.load(project_id, "storyboard", Storyboard)
    per_shot = {s.index: [a.status for a in store.assets(project_id, "image", s.id)] for s in board.shots}
    assert per_shot[1] == ["rejected", "approved"]
    assert all(v == ["approved"] for i, v in per_shot.items() if i != 1)     # approved assets preserved
    regenerated = store.assets(project_id, "image", board.shots[0].id)[-1]
    assert "Corrections required" in regenerated.metadata["prompt"] and "fix it" in regenerated.metadata["prompt"]


def test_thumbnail_loop_is_bounded_and_escalates(project, store, settings):
    def judge(context):
        return REJECT if context["purpose"] == "thumbnail" else {"approved": True, "score": 0.9}
    project_id, client, run = project({"judge.image": [judge] * 50}, max_thumbnail_attempts=2)
    assert run() == "needs_review"
    tasks = states(store, project_id)
    assert tasks["thumbnail_generation"] == tasks["thumbnail_review"] == "NEEDS_REVIEW"
    assert tasks["export"] == "PENDING" and tasks["rendering"] == "COMPLETED"
    brief = store.load(project_id, "thumbnail_brief", ThumbnailBrief)
    assert len(store.assets(project_id, "thumbnail", brief.id)) == 2          # exactly max attempts
    assert not (settings.output_dir / project_id / "thumbnail.png").exists()

    # an explicitly recorded human decision releases it; resume finishes without redoing the video
    with pytest.raises(ConfigError, match="reason"):
        apply_override(store, project_id, "thumbnail_review", "  ")
    apply_override(store, project_id, "thumbnail_review", "acceptable for the class demo")
    renders_before = client.calls.count("qa.continuity")
    assert run(resume=True) == "completed"
    assert client.calls.count("qa.continuity") == renders_before
    assert store.decisions(project_id, actor="human")[0]["reason"] == "acceptable for the class demo"
    assert (settings.output_dir / project_id / "thumbnail.png").exists()


def test_director_thumbnail_rejection_triggers_targeted_regeneration(project, store):
    project_id, client, run = project({"director.thumbnail": [
        {"approved": False, "reasons": ["character is off-model"]}]})
    assert run() == "completed"
    brief = store.load(project_id, "thumbnail_brief", ThumbnailBrief)
    candidates = store.assets(project_id, "thumbnail", brief.id)
    assert [c.status for c in candidates] == ["rejected", "approved"]
    # The Director's concept feedback goes to the Prompt Engineer; the new image follows the new prompt.
    assert client.calls.count("prompt.thumbnail") == 2 and client.calls.count("judge.thumbnail_prompt") == 2
    versions = store.artifact_versions(project_id, "thumbnail_prompt")
    assert [v["version"] for v in versions] == [1, 2] and candidates[1].prompt_id == versions[1]["id"]
    assert [r["responsible_agent"] for r in store.revisions(project_id)] == ["prompt_engineer"]


def test_compliance_failure_blocks_approval_until_a_human_decides(project, store, settings):
    blocked = {"findings": [{"check": "misleading_claims", "status": "fail", "detail": "false statistic"}]}
    project_id, client, run = project({"compliance.screen": [blocked]})
    assert run() == "needs_review"
    tasks = states(store, project_id)
    assert tasks["compliance"] == "NEEDS_REVIEW" and tasks["director_final_approval"] == "PENDING"
    assert "director.final" not in client.calls                               # Director never got to approve
    assert not (settings.output_dir / project_id / "video.mp4").exists()
    assert run(resume=True) == "needs_review"                                 # resume alone cannot bypass it

    apply_override(store, project_id, "compliance", "statistic checked manually and corrected in notes")
    assert run(resume=True) == "completed"
    decisions = [(d["actor"], d["decision"]) for d in store.decisions(project_id)]
    assert ("human", "override") in decisions and ("director", "noted_human_override") in decisions
    report = json.loads((settings.output_dir / project_id / "compliance_report.json").read_text())
    assert report["blocked"] is True and report["human_decisions"][0]["subject"] == "compliance"


def test_failed_task_is_recovered_by_resume(project, store):
    project_id, client, run = project({"seo.package": ["not json"] * 3})
    assert run() == "failed"
    assert states(store, project_id)["seo_package"] == "FAILED"
    assert "AgentOutputError" in store.errors(project_id)[-1]["message"]
    writer_calls = client.calls.count("writer.chapter")
    assert run(resume=True) == "completed"
    assert client.calls.count("writer.chapter") == writer_calls               # earlier work untouched


def test_narration_branch_survives_a_visual_branch_failure(project, store):
    project_id, client, run = project({"prompt.scene": ["broken"] * 3})
    assert run() == "failed"
    tasks = states(store, project_id)
    assert tasks["prompt_engineering"] == "FAILED" and tasks["audio_qa"] == "COMPLETED"
    assert tasks["editing"] == "PENDING"


def test_media_request_cap_stops_the_run(project, store):
    project_id, client, run = project(max_media_requests=2)
    assert run() == "stopped"
    assert store.cost_summary(project_id)["media_requests"] == 2
    assert "media request cap" in store.errors(project_id)[-1]["message"]


def test_live_modes_need_confirmation_and_a_key(settings, store):
    project_id = create_project(settings, store, "Cybersecurity", Mode.PREVIEW, budget_usd=1.0)
    with pytest.raises(ConfigError, match="explicit confirmation"):
        run_project(settings, store, project_id)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        run_project(settings, store, project_id, confirm_live=True)
    assert store.cost_summary(project_id)["requests"] == 0
    with pytest.raises(ConfigError, match="too short"):
        create_project(settings.model_copy(update={"min_chapters": 5}), store, "X", Mode.DRY_RUN)


def test_ffmpeg_renders_a_timeline_from_synthetic_assets(tmp_path):
    images, audio = [], tmp_path / "narration.wav"
    for n in range(2):
        path = tmp_path / f"img{n}.png"
        path.write_bytes(sample_image(f"synthetic {n}", 640, 360, seed=n))
        images.append(path)
    audio.write_bytes(sample_audio(3.5))
    timeline = EditingTimeline(width=320, height=180, fps=10, total_duration_s=4.0, clips=[
        TimelineClip(shot_id="a", scene_id="s", asset_path=str(images[0]), asset_kind="image", start_s=0,
                     duration_s=2.0, camera_move="zoom_in", fade_in=True,
                     captions=[Caption(text="Hello there", start_s=0.2, end_s=1.8)]),
        TimelineClip(shot_id="b", scene_id="s", asset_path=str(images[1]), asset_kind="image", start_s=2.0,
                     duration_s=2.0, camera_move="pan_left", fade_out=True)],
        audio=[AudioClip(scene_id="s", path=str(audio), start_s=0, slot_s=4.0, audio_s=3.5)])
    out = render(timeline, tmp_path / "work", tmp_path / "out" / "video.mp4")
    info = probe(out)
    assert "mp4" in info["format"] and info["video"]["codec"] == "h264" and info["audio"]["codec"] == "aac"
    assert (info["video"]["width"], info["video"]["height"]) == (320, 180)
    assert abs(info["duration_s"] - 4.0) < 0.15
    assert "Hello there" in out.with_suffix(".srt").read_text()


# ---- regression: work is preserved when Final QA fails, and resume reuses it ----------------------
def test_rendered_video_survives_final_qa_failure_and_resume_reuses_assets(project, store, settings):
    project_id, client, run = project()
    # The sample narrator speaks slower than estimated, as real TTS did: 24 s planned -> ~36 s rendered.
    client.settings = client.settings.model_copy(update={"words_per_minute": 100})
    assert run() == "needs_review"
    tasks = states(store, project_id)
    assert tasks["rendering"] == "COMPLETED" and tasks["final_qa"] == "NEEDS_REVIEW"
    assert "accepted range is 22-26s" in store.get_task(project_id, "final_qa")["error"]

    out = settings.output_dir / project_id
    preliminary = out / "preliminary" / "UNAPPROVED_video.mp4"
    assert probe(preliminary)["duration_s"] > 30                             # the rendered MP4 is kept
    status = (out / "STATUS.txt").read_text()
    assert "INCOMPLETE - NOT APPROVED" in status and "NEEDS_REVIEW" in status and "final_qa" in status
    images = sorted(p.name for p in (out / "assets" / "images").iterdir())
    assert len(images) == len(store.load(project_id, "storyboard", Storyboard).shots)
    assert list((out / "assets" / "audio").iterdir()) and (out / "artifacts" / "script.txt").is_file()
    for final_name in ("video.mp4", "thumbnail.png", "metadata.json"):       # nothing posing as final
        assert not (out / final_name).exists()

    # Widen the accepted duration and resume from Final QA: no media or script work is repeated.
    before = {a.id: a.status for a in store.assets(project_id) if a.kind in ("image", "audio", "final_video")}
    calls_before = list(client.calls)
    relaxed = settings.model_copy(update={"duration_tolerance": 0.6})
    assert run_project(relaxed, store, project_id, client=client, resume=True,
                       retry_escalated=True) == "completed"
    after = {a.id: a.status for a in store.assets(project_id) if a.kind in ("image", "audio", "final_video")}
    assert after == before                                                   # same approved assets, none new
    new_calls = client.calls[len(calls_before):]
    assert not {"visual_generation", "narration_tts", "writer.chapter", "scene.chapter", "prompt.scene",
                "qa.continuity"} & set(new_calls)                            # Final QA retry made no LLM call
    assert new_calls.count("thumbnail_generation") == 1                      # only the missing thumbnail

    assert EXPECTED_OUTPUTS <= {p.name for p in out.iterdir()}               # final export is complete
    assert not (out / "preliminary").exists() and "STATUS: APPROVED" in (out / "STATUS.txt").read_text()
    assert probe(out / "video.mp4")["duration_s"] == pytest.approx(probe(
        store.assets(project_id, "final_video")[0].path)["duration_s"])      # exported, not re-rendered
    assert len(store.assets(project_id, "final_video")) == 1


def test_collect_marks_unfinished_projects_incomplete(project, store, settings):
    from studio.orchestration.outputs import publish_work_files
    project_id, client, run = project({"prompt.scene": ["broken"] * 3})      # visual branch fails early
    assert run() == "failed"
    out = publish_work_files(store, settings, project_id)
    status = (out / "STATUS.txt").read_text()
    assert "INCOMPLETE - NOT APPROVED" in status and "FAILED       prompt_engineering" in status
    assert not (out / "preliminary").exists()                                # nothing was rendered
    assert list((out / "assets" / "audio").iterdir())                        # narration branch output kept


# ---- regression: thumbnail text is drawn once, by software ------------------------------------------
def test_thumbnail_text_is_never_requested_from_the_image_model(project, store):
    def judge(context):                      # a reviewer complains about the title text, as happened live
        if context["purpose"] == "thumbnail" and not getattr(judge, "done", False):
            judge.done = True
            return {"approved": False, "score": 0.3, "issues": ["Overlay text 'EVERY RANK' looks doubled"],
                    "suggested_changes": ["Enlarge the 'EVERY RANK' text and add it at the top"]}
        return {"approved": True, "score": 0.9}
    project_id, client, run = project({"judge.image": [judge] * 50})
    assert run() == "completed"
    brief = store.load(project_id, "thumbnail_brief", ThumbnailBrief)
    first, second = store.assets(project_id, "thumbnail", brief.id)
    for candidate in (first, second):
        sent = candidate.metadata["prompt"]
        assert sent.rstrip().endswith("free of faces and important details.")       # the no-text rule is last
        assert "must contain no text of any kind" in sent
        assert Path(candidate.metadata["raw_path"]).is_file()                        # artwork without text kept
    assert "Enlarge the 'EVERY RANK' text" in second.metadata["prompt"]              # feedback still recorded
    assert second.metadata["prompt"].index("Enlarge") < second.metadata["prompt"].index("IMPORTANT")
    assert second.metadata["text_rendering"].startswith("added locally")


def test_title_banner_is_drawn_once_inside_the_bottom_band(tmp_path):
    from PIL import Image

    from studio.agents.visual_generation import finish_thumbnail
    source, plain, titled = tmp_path / "raw.png", tmp_path / "plain.png", tmp_path / "titled.png"
    Image.new("RGB", (1600, 900), (40, 160, 90)).save(source)
    finish_thumbnail(source, plain, (1280, 720), "")
    finish_thumbnail(source, titled, (1280, 720), "rank to rank")
    a, b = Image.open(plain).convert("RGB"), Image.open(titled).convert("RGB")
    assert a.size == b.size == (1280, 720)
    changed = [y for y in range(720) if a.crop((0, y, 1280, y + 1)).tobytes() != b.crop((0, y, 1280, y + 1)).tobytes()]
    assert changed and min(changed) > 720 * 0.6 and max(changed) == 719              # only the bottom band
    assert b.getpixel((640, 100)) == (40, 160, 90)                                    # artwork untouched
    band = b.crop((0, min(changed), 1280, 720)).tobytes()
    white = sum(1 for i in range(0, len(band), 3) if band[i:i + 3] == b"\xff\xff\xff")
    assert white > 2000                                                               # the title is there
