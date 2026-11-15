"""Deterministic software guards. No LLM is involved in any check in this module.

Each guard returns an EvaluationResult (passed, criteria, issues, suggested changes, metrics)
so guards and LLM judges share one report format.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

from studio.config import Settings
from studio.rendering.ffmpeg import FFmpegError, audio_duration, probe
from studio.schemas import (AnimationPrompt, Criterion, EditingTimeline, EvaluationResult, ImagePrompt,
                            ProductionPlan, Script, Storyboard)
from studio.utils import normalize_text

IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
AUDIO_SUFFIXES = {".mp3", ".wav"}


class _Report:
    """Collects criteria while a guard runs."""

    def __init__(self, guard: str, target_kind: str, target_id: str | None):
        self.guard, self.target_kind, self.target_id = guard, target_kind, target_id
        self.criteria: list[Criterion] = []
        self.issues: list[str] = []
        self.fixes: list[str] = []
        self.metrics: dict = {}

    def check(self, name: str, ok: bool, issue: str = "", fix: str = "") -> bool:
        self.criteria.append(Criterion(name=name, passed=ok, note="" if ok else issue))
        if not ok:
            self.issues.append(issue or name)
            if fix:
                self.fixes.append(fix)
        return ok

    def result(self) -> EvaluationResult:
        passed = all(c.passed for c in self.criteria)
        return EvaluationResult(parent_id=self.target_id, target_kind=self.target_kind, evaluator=self.guard,
                                passed=passed, criteria=self.criteria, issues=self.issues,
                                suggested_changes=self.fixes, metrics=self.metrics,
                                status="passed" if passed else "failed",
                                explanation="all checks passed" if passed else "; ".join(self.issues))


# ---- Script Guard ---------------------------------------------------------------
def chapter_word_targets(plan: ProductionPlan, settings: Settings) -> dict[int, int]:
    """Words each chapter should have, scaled so the whole script fits the target duration."""
    total_seconds = sum(c.target_seconds for c in plan.chapters)
    return {c.index: max(round(settings.target_words * c.target_seconds / total_seconds), 15)
            for c in plan.chapters}


def script_guard(script: Script, plan: ProductionPlan, settings: Settings) -> EvaluationResult:
    r = _Report("script_guard", "script", script.id)
    target, tol = settings.target_words, settings.duration_tolerance
    words = script.words
    estimated = words / settings.words_per_minute * 60
    r.metrics = {"words": words, "target_words": target, "estimated_duration_s": round(estimated, 1),
                 "chapters": len(script.chapters), "chapter_words": {c.index: c.words for c in script.chapters}}
    r.check("hook present", bool(script.hook.strip()), "the opening hook is empty", "write an opening hook")
    r.check("chapter structure", [c.index for c in script.chapters] == [c.index for c in plan.chapters],
            f"script has chapters {[c.index for c in script.chapters]} but the plan requires "
            f"{[c.index for c in plan.chapters]}", "write exactly the planned chapters, in order")
    r.check("chapter count in range", settings.min_chapters <= len(script.chapters) <= settings.max_chapters,
            f"{len(script.chapters)} chapters is outside {settings.min_chapters}-{settings.max_chapters}")
    for chapter in script.chapters:
        r.check(f"chapter {chapter.index} not empty", chapter.words >= 15,
                f"chapter {chapter.index} has only {chapter.words} words", "write full narration")
    r.check("total length", abs(words - target) <= target * tol,
            f"script has {words} words (~{estimated:.0f}s); target is {target} words "
            f"(~{settings.target_duration_s:.0f}s) within {tol:.0%}",
            f"{'shorten' if words > target else 'lengthen'} the script to about {target} words")
    return r.result()


def chapters_off_target(script: Script, plan: ProductionPlan, settings: Settings) -> dict[int, tuple[int, int]]:
    """Chapters whose length is furthest from plan: {index: (actual_words, target_words)}."""
    targets = chapter_word_targets(plan, settings)
    off = {c.index: (c.words, targets.get(c.index, c.words)) for c in script.chapters
           if abs(c.words - targets.get(c.index, c.words)) > targets.get(c.index, c.words)
           * settings.duration_tolerance}
    return off


# ---- Timeline Guard -----------------------------------------------------------------
def timeline_guard(storyboard: Storyboard, script: Script, settings: Settings) -> EvaluationResult:
    r = _Report("timeline_guard", "storyboard", storyboard.id)
    shots = storyboard.shots
    total, target, tol = storyboard.duration_s, settings.target_duration_s, settings.duration_tolerance
    r.metrics = {"scenes": len(storyboard.scenes), "shots": len(shots), "estimated_duration_s": round(total, 1),
                 "bad_chapters": []}
    r.check("scene order", [s.index for s in storyboard.scenes] == list(range(1, len(storyboard.scenes) + 1)),
            "scene indexes are not consecutive starting at 1")
    r.check("shot order", [s.index for s in shots] == list(range(1, len(shots) + 1)),
            "shot indexes are not consecutive starting at 1")
    chapters = [s.chapter_index for s in storyboard.scenes]
    r.check("chapters in order", chapters == sorted(chapters), "scenes are not ordered by chapter")

    cursor, gaps = 0.0, []
    for scene in storyboard.scenes:
        if abs(scene.start_s - cursor) > 0.01:
            gaps.append(f"scene {scene.index} starts at {scene.start_s:.2f}s, expected {cursor:.2f}s")
        inner = scene.start_s
        for shot in scene.shots:
            if abs(shot.start_s - inner) > 0.01:
                gaps.append(f"shot {shot.index} starts at {shot.start_s:.2f}s, expected {inner:.2f}s")
            inner += shot.duration_s
        if abs(inner - (scene.start_s + scene.duration_s)) > 0.01:
            gaps.append(f"scene {scene.index} duration does not equal the sum of its shots")
        cursor = scene.start_s + scene.duration_s
    r.check("no gaps or overlaps", not gaps, "; ".join(gaps[:5]), "recompute timestamps contiguously")

    bad_chapters: set[int] = set()
    for shot in shots:
        ok = settings.min_shot_seconds <= shot.duration_s <= settings.max_shot_seconds
        if not ok:
            bad_chapters.add(shot.chapter_index)
        r.check(f"shot {shot.index} duration", ok,
                f"shot {shot.index} (chapter {shot.chapter_index}) lasts {shot.duration_s:.1f}s; allowed "
                f"{settings.min_shot_seconds:.0f}-{settings.max_shot_seconds:.0f}s",
                "split long shots into several shots / merge very short shots")
    for chapter in script.chapters:
        covered = normalize_text(" ".join(s.narration for s in shots if s.chapter_index == chapter.index))
        ok = covered == normalize_text(chapter.narration)
        if not ok:
            bad_chapters.add(chapter.index)
        r.check(f"chapter {chapter.index} narration covered", ok,
                f"shots of chapter {chapter.index} do not cover its narration exactly",
                "assign every sentence of the chapter to exactly one shot, in order")
    r.check("total duration", abs(total - target) <= target * tol,
            f"storyboard is {total:.0f}s; target {target:.0f}s within {tol:.0%}")
    r.metrics["bad_chapters"] = sorted(bad_chapters)
    return r.result()


# ---- Prompt Guard ---------------------------------------------------------------------
def prompt_guard(prompt: ImagePrompt | AnimationPrompt, valid_parent_ids: set[str]) -> EvaluationResult:
    r = _Report("prompt_guard", "prompt", prompt.id)
    r.check("prompt text present", len(prompt.prompt.strip()) >= 15, "prompt text is missing or too short",
            "write a full descriptive prompt")
    r.check("references a known scene/shot", prompt.parent_id in valid_parent_ids,
            f"prompt refers to unknown parent '{prompt.parent_id}'")
    r.check("supported media type", prompt.media_type in ("image", "animation"),
            f"unsupported media type '{prompt.media_type}'")
    if isinstance(prompt, ImagePrompt):
        r.check("negative prompt present", bool(prompt.negative_prompt.strip()),
                "negative prompt is empty", "state what must not appear")
        r.check("aspect ratio set", bool(prompt.aspect_ratio.strip()), "aspect ratio is missing")
    else:
        r.check("positive duration", prompt.duration_s > 0, "animation duration must be positive")
    return r.result()


# ---- Media Guard (images, video clips) --------------------------------------------------
def image_guard(path: Path | str, *, aspect_ratio: float, min_width: int = 512,
                target_id: str | None = None, exact_size: tuple[int, int] | None = None) -> EvaluationResult:
    r, path = _Report("media_guard", "image", target_id), Path(path)
    if not r.check("file exists", path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}"):
        return r.result()
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            fmt, (width, height) = img.format, img.size
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        r.check("not corrupt", False, f"image cannot be decoded: {exc}", "regenerate the image")
        return r.result()
    r.metrics = {"format": fmt, "width": width, "height": height}
    r.check("supported format", fmt in IMAGE_FORMATS, f"unsupported image format {fmt}")
    r.check("minimum size", width >= min_width, f"image is only {width}px wide (minimum {min_width})")
    r.check("aspect ratio", abs(width / height - aspect_ratio) <= 0.12 * aspect_ratio,
            f"image is {width}x{height}; expected an aspect ratio near {aspect_ratio:.2f}",
            "regenerate with the correct aspect ratio")
    if exact_size:
        r.check("exact dimensions", (width, height) == exact_size,
                f"image is {width}x{height}; required {exact_size[0]}x{exact_size[1]}")
    return r.result()


def video_clip_guard(path: Path | str, *, expected_s: float, target_id: str | None = None) -> EvaluationResult:
    r, path = _Report("media_guard", "video", target_id), Path(path)
    if not r.check("file exists", path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}"):
        return r.result()
    try:
        info = probe(path)
    except FFmpegError as exc:
        r.check("not corrupt", False, str(exc), "regenerate the clip")
        return r.result()
    r.metrics = info
    if r.check("has video stream", info["video"] is not None, "file has no video stream"):
        r.check("duration", info["duration_s"] is not None and 0.5 <= info["duration_s"] <= expected_s * 2 + 2,
                f"clip lasts {info['duration_s']}s; expected about {expected_s}s")
    return r.result()


# ---- Audio Guard ---------------------------------------------------------------------------
def audio_guard(path: Path | str, *, estimated_s: float, target_id: str | None = None,
                min_ratio: float = 0.4, max_ratio: float = 2.2) -> EvaluationResult:
    r, path = _Report("audio_guard", "audio", target_id), Path(path)
    if not r.check("file exists", path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}"):
        return r.result()
    r.check("supported format", path.suffix.lower() in AUDIO_SUFFIXES, f"unsupported audio type {path.suffix}")
    try:
        measured = audio_duration(path)
    except FFmpegError as exc:
        r.check("valid audio", False, str(exc), "regenerate the narration segment")
        return r.result()
    r.metrics = {"measured_s": round(measured, 3), "estimated_s": round(estimated_s, 3)}
    r.check("valid audio", measured > 0.2, f"audio is only {measured:.2f}s long")
    r.check("timing consistent with text", estimated_s * min_ratio <= measured <= estimated_s * max_ratio,
            f"narration lasts {measured:.1f}s but the text suggests about {estimated_s:.1f}s "
            "(possible truncated or runaway speech)", "regenerate the narration segment")
    return r.result()


# ---- Final Video Guard ------------------------------------------------------------------------
def final_video_guard(path: Path | str, timeline: EditingTimeline, storyboard: Storyboard,
                      settings: Settings) -> EvaluationResult:
    r, path = _Report("final_video_guard", "final_video", timeline.id), Path(path)
    if not r.check("file exists", path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}"):
        return r.result()
    try:
        info = probe(path)
    except FFmpegError as exc:
        r.check("valid MP4", False, str(exc))
        return r.result()
    r.metrics = info
    r.check("valid MP4", "mp4" in info["format"], f"container is '{info['format']}', not MP4")
    r.check("audio stream present", info["audio"] is not None, "final video has no audio stream")
    if r.check("video stream present", info["video"] is not None, "final video has no video stream"):
        width, height = info["video"]["width"], info["video"]["height"]
        r.check("resolution", (width, height) == (settings.width, settings.height),
                f"video is {width}x{height}; expected {settings.width}x{settings.height}")
        a, b = (float(x) for x in settings.aspect_ratio.split(":"))
        r.check("aspect ratio", abs(width / height - a / b) < 0.02,
                f"aspect ratio {width / height:.3f} does not match {settings.aspect_ratio}")
    duration = info["duration_s"] or 0.0
    low, high = settings.duration_bounds
    r.metrics["accepted_duration_s"] = [round(low, 1), round(high, 1)]
    # Compared at 0.1 s precision (inclusive) so container/audio padding of a few ms is ignored.
    r.check("duration within accepted range", round(low, 1) <= round(duration, 1) <= round(high, 1),
            f"video lasts {duration:.1f}s; accepted range is {low:.0f}-{high:.0f}s "
            f"(target {settings.target_duration_s:.0f}s)",
            "adjust the script length (narration decides the final duration); the video is never "
            "stretched or trimmed to fit")
    r.check("matches timeline", abs(duration - timeline.total_duration_s) <= 1.5,
            f"video lasts {duration:.1f}s but the timeline specifies {timeline.total_duration_s:.1f}s")
    missing = sorted({s.index for s in storyboard.shots if s.id not in {c.shot_id for c in timeline.clips}})
    r.check("all shots covered", not missing, f"shots missing from the timeline: {missing}")
    absent = [c.asset_path for c in timeline.clips if not Path(c.asset_path).is_file()] + \
             [a.path for a in timeline.audio if not Path(a.path).is_file()]
    r.check("no missing assets", not absent, f"{len(absent)} timeline assets are missing on disk")
    return r.result()

