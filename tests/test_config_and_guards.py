"""Configuration errors and the deterministic software guards."""
import pytest
from PIL import Image

from studio.config import ConfigError, Mode, Settings, load_settings
from studio import guards
from studio.guards import (audio_guard, final_video_guard, image_guard, prompt_guard, script_guard,
                           timeline_guard)
from studio.schemas import (AudioClip, Chapter, ChapterPlan, EditingTimeline, ImagePrompt, ProductionPlan,
                            Scene, Script, Shot, Storyboard, TimelineClip)
from studio.services.placeholders import sample_audio, sample_image
from studio.utils import redact, safe_path


# ---- configuration ------------------------------------------------------------
def test_missing_api_key_is_a_clear_error():
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        Settings().require_api_key()


def test_invalid_configuration_values(monkeypatch):
    monkeypatch.setenv("WIDTH", "1000")                 # 1000x1080 is not 16:9
    with pytest.raises(ConfigError, match="WIDTH/HEIGHT"):
        load_settings(env_file=None)
    monkeypatch.delenv("WIDTH")
    monkeypatch.setenv("PROMPT_JUDGE_THRESHOLD", "7")
    with pytest.raises(ConfigError, match="PROMPT_JUDGE_THRESHOLD"):
        load_settings(env_file=None)


def test_music_requires_a_license(monkeypatch):
    monkeypatch.setenv("MUSIC_PATH", "song.mp3")
    with pytest.raises(ConfigError, match="MUSIC_LICENSE"):
        load_settings(env_file=None)


def test_environment_overrides_and_mode_profiles(monkeypatch):
    monkeypatch.setenv("TARGET_DURATION_S", "300")
    monkeypatch.setenv("ENABLE_VIDEO", "true")
    s = load_settings(env_file=None)
    assert s.target_duration_s == 300 and s.enable_video is True and s.target_words == 750
    assert s.for_mode(Mode.PREVIEW).target_duration_s == s.preview_duration_s
    assert (s.for_mode(Mode.DRY_RUN).width, s.for_mode(Mode.DRY_RUN).height) == (640, 360)
    assert s.for_mode(Mode.FULL) is s


def test_secrets_are_redacted_and_paths_confined(tmp_path):
    key = "sk-or-v1-abcdef1234567890"
    assert key not in redact(f"Authorization: Bearer {key}")
    assert key not in redact(f"failed with api_key={key}", key)
    assert "abcdef" not in repr(Settings(openrouter_api_key=key))
    with pytest.raises(ValueError):
        safe_path(tmp_path, "..", "outside.txt")


# ---- script guard ----------------------------------------------------------------
def _plan(n=2):
    return ProductionPlan(creative_goals=["g"], audience="a", tone="t", chapters=[
        ChapterPlan(index=i, title=f"Rank {i}", target_seconds=12) for i in range(1, n + 1)])


def _script(words_per_chapter, n=2):
    return Script(title="T", hook="Hook.", chapters=[
        Chapter(index=i, title=f"Rank {i}", narration=("word " * words_per_chapter).strip() + ".")
        for i in range(1, n + 1)])


def test_script_guard_accepts_a_script_on_target(settings):
    result = script_guard(_script(30), _plan(), settings)          # 60 words == 24s at 150 wpm
    assert result.passed and result.metrics["words"] == 60


def test_script_guard_flags_length_structure_and_empty_content(settings):
    too_long = script_guard(_script(60), _plan(), settings)
    assert not too_long.passed and any("words" in i for i in too_long.issues)
    assert any("shorten" in s for s in too_long.suggested_changes)
    missing_chapter = script_guard(_script(60, n=1), _plan(), settings)
    assert any("plan requires" in i for i in missing_chapter.issues)
    thin = script_guard(_script(3), _plan(), settings)
    assert any("only 3 words" in i for i in thin.issues)


# ---- timeline guard ---------------------------------------------------------------
def _storyboard(script, durations=(6.0, 6.0)):
    scenes, clock, index = [], 0.0, 0
    for chapter in script.chapters:
        half = len(chapter.narration.split()) // 2
        words = chapter.narration.split()
        shots = []
        for text, duration in ((" ".join(words[:half]), durations[0]), (" ".join(words[half:]), durations[1])):
            index += 1
            shots.append(Shot(index=index, chapter_index=chapter.index, narration=text,
                              visual_description="a visual description", start_s=clock, duration_s=duration))
            clock += duration
        scenes.append(Scene(index=chapter.index, chapter_index=chapter.index, title="s",
                            start_s=shots[0].start_s, duration_s=sum(durations), shots=shots))
    return Storyboard(scenes=scenes)


def test_timeline_guard_accepts_contiguous_storyboard(settings):
    script = _script(30)
    assert timeline_guard(_storyboard(script), script, settings).passed


def test_timeline_guard_detects_overlap_gap_and_bad_durations(settings):
    script = _script(30)
    board = _storyboard(script)
    board.scenes[1].shots[0].start_s -= 2                 # overlaps the previous shot
    result = timeline_guard(board, script, settings)
    assert not result.passed and any("expected" in i for i in result.issues)

    board = _storyboard(script, durations=(0.2, 11.8))    # first shot shorter than the minimum
    result = timeline_guard(board, script, settings)
    assert any("lasts 0.2s" in i for i in result.issues) and result.metrics["bad_chapters"] == [1, 2]

    board = _storyboard(script)
    board.scenes[0].shots[0].narration = "changed words"  # narration no longer covered exactly
    result = timeline_guard(board, script, settings)
    assert any("do not cover" in i for i in result.issues) and 1 in result.metrics["bad_chapters"]

    board = _storyboard(script, durations=(12.0, 12.0))   # 48s instead of 24s
    assert any("target 24s" in i for i in timeline_guard(board, script, settings).issues)


# ---- prompt / media / audio guards ---------------------------------------------------
def test_prompt_guard():
    good = ImagePrompt(parent_id="shot1", prompt="A cartoon analyst at a desk, wide shot", negative_prompt="text")
    assert prompt_guard(good, {"shot1"}).passed
    orphan = prompt_guard(good, {"other"})
    assert not orphan.passed and "unknown parent" in orphan.issues[0]
    no_negative = ImagePrompt(parent_id="shot1", prompt="A cartoon analyst at a desk, wide shot")
    assert not prompt_guard(no_negative, {"shot1"}).passed
    with pytest.raises(ValueError):
        ImagePrompt(parent_id="shot1", prompt="   ")       # empty prompts never validate


def test_image_guard_validates_metadata_and_detects_corruption(tmp_path):
    good = tmp_path / "ok.png"
    good.write_bytes(sample_image("hello"))
    result = image_guard(good, aspect_ratio=16 / 9)
    assert result.passed and result.metrics == {"format": "PNG", "width": 1280, "height": 720}
    assert not image_guard(good, aspect_ratio=16 / 9, exact_size=(1920, 1080)).passed

    square = tmp_path / "square.png"
    Image.new("RGB", (800, 800)).save(square)
    assert any("aspect ratio" in i for i in image_guard(square, aspect_ratio=16 / 9).issues)

    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"this is not an image")
    assert any("cannot be decoded" in i for i in image_guard(corrupt, aspect_ratio=16 / 9).issues)
    assert any("missing" in i for i in image_guard(tmp_path / "none.png", aspect_ratio=16 / 9).issues)


def test_audio_guard_measures_duration_and_timing(tmp_path):
    clip = tmp_path / "a.wav"
    clip.write_bytes(sample_audio(3.0))
    ok = audio_guard(clip, estimated_s=3.0)
    assert ok.passed and abs(ok.metrics["measured_s"] - 3.0) < 0.05
    assert any("suggests about" in i for i in audio_guard(clip, estimated_s=30.0).issues)
    broken = tmp_path / "b.wav"
    broken.write_bytes(b"RIFFnope")
    assert not audio_guard(broken, estimated_s=3.0).passed


# ---- final video duration: preview window vs full-production tolerance ---------------------------
def _final_guard(monkeypatch, tmp_path, settings, duration):
    """Run the Final Video Guard on a stand-in file whose probed duration is `duration`."""
    media = tmp_path / "media.bin"
    media.write_bytes(b"x")
    monkeypatch.setattr(guards, "probe", lambda path: {
        "format": "mov,mp4,m4a,3gp,3g2,mj2", "duration_s": duration, "size_bytes": 1,
        "video": {"codec": "h264", "width": settings.width, "height": settings.height},
        "audio": {"codec": "aac", "sample_rate": "44100", "channels": 2}})
    script = _script(30, n=1)
    board = _storyboard(script)
    timeline = EditingTimeline(
        width=settings.width, height=settings.height, fps=settings.fps, total_duration_s=duration,
        clips=[TimelineClip(shot_id=s.id, scene_id="s", asset_path=str(media), asset_kind="image",
                            start_s=0, duration_s=1) for s in board.shots],
        audio=[AudioClip(scene_id="s", path=str(media), start_s=0, slot_s=1, audio_s=1)])
    return final_video_guard(media, timeline, board, settings)


@pytest.mark.parametrize("duration, passes", [(68.0, True), (68.033, True), (50.0, True), (80.0, True),
                                              (80.03, True), (81.0, False), (49.0, False)])
def test_preview_accepts_50_to_80_seconds(monkeypatch, tmp_path, duration, passes):
    preview = Settings().for_mode(Mode.PREVIEW)
    assert preview.target_duration_s == 60 and preview.duration_bounds == (50, 80)
    result = _final_guard(monkeypatch, tmp_path, preview, duration)
    assert result.passed is passes
    assert [c.passed for c in result.criteria if c.name != "duration within accepted range"].count(False) == 0
    if not passes:
        assert "accepted range is 50-80s (target 60s)" in result.issues[0]


@pytest.mark.parametrize("duration, passes", [(600.0, True), (540.0, True), (660.0, True), (661.0, False),
                                              (539.0, False), (68.0, False)])
def test_full_production_tolerance_is_unchanged(monkeypatch, tmp_path, duration, passes):
    full = Settings().for_mode(Mode.FULL)
    assert full.duration_window is None and full.duration_bounds == pytest.approx((540, 660))
    assert _final_guard(monkeypatch, tmp_path, full, duration).passed is passes


def test_preview_window_is_configurable_and_does_not_leak_into_other_modes(monkeypatch):
    monkeypatch.setenv("PREVIEW_MAX_DURATION_S", "75")
    monkeypatch.setenv("DURATION_WINDOW", "[1, 9999]")                      # not an accepted setting
    s = load_settings(env_file=None)
    assert s.for_mode(Mode.PREVIEW).duration_bounds == (50, 75)
    assert s.duration_window is None and s.for_mode(Mode.DRY_RUN).duration_window is None
    assert s.for_mode(Mode.DRY_RUN).duration_bounds == pytest.approx((540, 660))
    monkeypatch.setenv("PREVIEW_MIN_DURATION_S", "70")                      # 60 s preview outside 70-75
    with pytest.raises(ConfigError, match="PREVIEW_DURATION_S"):
        load_settings(env_file=None)


def test_other_final_checks_still_apply_in_preview(monkeypatch, tmp_path):
    preview = Settings().for_mode(Mode.PREVIEW)
    wrong_size = preview.model_copy(update={"width": 1280, "height": 720})   # video is probed as 1920x1080
    result = _final_guard(monkeypatch, tmp_path, preview, 68.0)
    assert result.passed
    monkeypatch.setattr(guards, "probe", lambda path: {
        "format": "mp4", "duration_s": 68.0, "size_bytes": 1, "audio": None,
        "video": {"codec": "h264", "width": 1920, "height": 1080}})
    media = tmp_path / "media.bin"
    timeline = EditingTimeline(width=1280, height=720, fps=30, total_duration_s=60.0, clips=[
        TimelineClip(shot_id="x", scene_id="s", asset_path=str(media), asset_kind="image", start_s=0,
                     duration_s=1)], audio=[AudioClip(scene_id="s", path=str(media), start_s=0, slot_s=1,
                                                      audio_s=1)])
    issues = " | ".join(final_video_guard(media, timeline, _storyboard(_script(30, n=1)), wrong_size).issues)
    for expected in ("no audio stream", "expected 1280x720", "timeline specifies 60.0s", "shots missing"):
        assert expected in issues
