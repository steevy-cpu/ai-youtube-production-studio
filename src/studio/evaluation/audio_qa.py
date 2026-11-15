"""Audio QA: file validity, measured duration, silence, timing and optional transcription check."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from studio.guards import audio_guard
from studio.orchestration.context import Ctx
from studio.rendering.ffmpeg import FFmpegError, detect_silence, ffmpeg_available
from studio.schemas import Criterion, EvaluationResult, MediaAsset, NarrationSegment

MAX_SILENCE_RATIO, MAX_SILENCE_S, MIN_TRANSCRIPT_SIMILARITY = 0.35, 3.0, 0.8


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def check_narration(ctx: Ctx, segment: NarrationSegment, asset: MediaAsset) -> EvaluationResult:
    result = audio_guard(asset.path, estimated_s=segment.estimated_s, target_id=asset.id)
    result.evaluator, result.project_id = "audio_qa", ctx.project_id

    def add(name: str, ok: bool, issue: str, fix: str = "regenerate the narration segment") -> None:
        result.criteria.append(Criterion(name=name, passed=ok, note="" if ok else issue))
        if not ok:
            result.issues.append(issue)
            result.suggested_changes.append(fix)

    if result.passed and ffmpeg_available():
        try:
            silences = detect_silence(asset.path)
        except FFmpegError as exc:
            add("silence analysis", False, f"silence analysis failed: {exc}")
        else:
            measured = result.metrics["measured_s"]
            total, longest = sum(d for _, d in silences), max((d for _, d in silences), default=0.0)
            result.metrics.update(silence_s=round(total, 2), longest_silence_s=round(longest, 2))
            add("no excessive silence", total <= measured * MAX_SILENCE_RATIO and longest <= MAX_SILENCE_S,
                f"{total:.1f}s of silence in {measured:.1f}s of audio (longest gap {longest:.1f}s)")
    else:
        result.metrics["silence_check"] = "not performed"

    # Transcription comparison only when an STT model is configured and the audio is real speech.
    if result.passed and ctx.settings.stt_model and not asset.is_sample and not ctx.client.is_mock:
        ctx.budget.check("stt")
        text, usage = ctx.client.transcribe(ctx.settings.stt_model, Path(asset.path))
        ctx.budget.record(ctx.task, "audio_qa", "stt", usage)
        similarity = SequenceMatcher(None, _words(segment.text), _words(text)).ratio()
        result.metrics.update(transcription="performed", transcript_similarity=round(similarity, 3))
        add("spoken words match the script", similarity >= MIN_TRANSCRIPT_SIMILARITY,
            f"transcript matches only {similarity:.0%} of the script text")
    else:
        result.metrics["transcription"] = ("not performed (no STT_MODEL configured)"
                                           if not ctx.settings.stt_model else
                                           "not performed (placeholder or dry-run audio)")

    result.passed = all(c.passed for c in result.criteria)
    result.status = "passed" if result.passed else "failed"
    result.explanation = "all checks passed" if result.passed else "; ".join(result.issues)
    return ctx.store.add_evaluation(result)
