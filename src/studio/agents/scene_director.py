"""Scene Director Agent: turns each chapter into scenes and shots.

The model decides WHERE to cut and WHAT to show. It refers to narration by sentence number,
so it can never alter the approved wording; durations and timestamps are computed by code.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from studio.agents.base import Agent, AgentOutputError
from studio.config import Settings
from studio.schemas import CameraMove, Chapter, MediaType, Scene, Shot, Storyboard, StyleBible
from studio.utils import new_id, split_sentences, word_count


class ShotDraft(BaseModel):
    first_sentence: int = Field(ge=1)
    last_sentence: int = Field(ge=1)
    visual_description: str = Field(min_length=10)
    framing: str = "medium shot"
    camera_move: CameraMove = "zoom_in"
    media_type: MediaType = "image"


class SceneDraft(BaseModel):
    title: str
    summary: str = ""
    shots: list[ShotDraft] = Field(min_length=1)


class ChapterScenes(BaseModel):
    scenes: list[SceneDraft] = Field(min_length=1)


def coverage_problem(scenes: list[SceneDraft], sentence_count: int) -> str:
    """Empty string when shots cover sentences 1..N exactly once and in order."""
    expected = 1
    for shot in (s for scene in scenes for s in scene.shots):
        if shot.first_sentence != expected or shot.last_sentence < shot.first_sentence:
            return (f"a shot covers sentences {shot.first_sentence}-{shot.last_sentence} but the next "
                    f"uncovered sentence is {expected}")
        expected = shot.last_sentence + 1
    if expected != sentence_count + 1:
        return f"shots cover sentences 1-{expected - 1} but the chapter has {sentence_count} sentences"
    return ""


def retime(storyboard: Storyboard, settings: Settings) -> Storyboard:
    """Deterministically renumber scenes/shots and assign contiguous estimated timestamps."""
    clock, scene_no, shot_no = 0.0, 0, 0
    for scene in sorted(storyboard.scenes, key=lambda s: (s.chapter_index, s.index)):
        scene_no += 1
        scene.index, scene.start_s = scene_no, round(clock, 3)
        for shot in scene.shots:
            shot_no += 1
            shot.index, shot.start_s = shot_no, round(clock, 3)
            shot.duration_s = round(max(word_count(shot.narration) / settings.words_per_minute * 60, 0.5), 3)
            clock += shot.duration_s
        scene.duration_s = round(clock - scene.start_s, 3)
    storyboard.scenes.sort(key=lambda s: s.index)
    return storyboard


class SceneDirector(Agent):
    name = "scene_director"
    system = ("You are the Scene Director. You break narration into scenes and visual shots for a 2D "
              "cartoon video and describe exactly what is visible in each shot.")

    def chapter_scenes(self, chapter: Chapter, style: StyleBible, feedback: str = "") -> list[Scene]:
        s = self.ctx.settings
        sentences = split_sentences(chapter.narration)
        numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(sentences, start=1))
        per_shot = round(s.target_shot_seconds * s.words_per_minute / 60)
        max_words = int(s.max_shot_seconds * s.words_per_minute / 60)
        animation_rule = ("Mark at most one especially dynamic shot as media_type \"animation\"; every "
                          "other shot is \"image\"." if s.enable_video else
                          "Every shot must use media_type \"image\".")
        user = (
            f"Chapter {chapter.index}: \"{chapter.title}\". Narration, one numbered sentence per line:\n"
            f"<data>\n{numbered}\n</data>\n\n"
            f"Group the {len(sentences)} sentences into scenes (one location or idea each) and split every "
            "scene into shots. Each shot covers a consecutive range first_sentence..last_sentence. Every "
            "sentence must belong to exactly one shot, in order, with no gaps.\n"
            f"Aim for about {per_shot} words of narration per shot and never more than {max_words}.\n"
            "visual_description: one concrete still image (who, doing what, where, key props, expression) "
            "that illustrates those sentences. No on-screen text.\n"
            f"Main character: {style.characters[0].name}. Environment rules: {style.environment_guidelines}\n"
            f"Framing rules: {style.framing_guidelines}\n{animation_rule}")
        if feedback:
            user += f"\n\nYour previous storyboard for this chapter was rejected:\n{feedback}"
        context = {"chapter_index": chapter.index, "title": chapter.title,
                   "sentence_words": [word_count(t) for t in sentences],
                   "words_per_minute": s.words_per_minute, "target_shot_seconds": s.target_shot_seconds,
                   "allow_animation": s.enable_video}

        problem = ""
        for _ in range(2):          # one targeted repair if the sentence coverage is wrong
            extra = f"\n\nFix this error from your last answer: {problem}" if problem else ""
            draft = self.ask("scene.chapter", ChapterScenes, self.system, user + extra, context=context)
            problem = coverage_problem(draft.scenes, len(sentences))
            if not problem:
                break
        else:
            raise AgentOutputError(f"scene_director: chapter {chapter.index} storyboard is invalid: {problem}")

        scenes = []
        for number, scene_draft in enumerate(draft.scenes, start=1):
            scene_id = new_id()
            shots = [Shot(
                index=1, chapter_index=chapter.index, parent_id=scene_id,
                narration=" ".join(sentences[shot.first_sentence - 1:shot.last_sentence]),
                visual_description=shot.visual_description, framing=shot.framing,
                camera_move=shot.camera_move, media_type=shot.media_type if s.enable_video else "image",
                start_s=0, duration_s=1) for shot in scene_draft.shots]
            scenes.append(Scene(id=scene_id, index=number, chapter_index=chapter.index, parent_id=chapter.id,
                                title=scene_draft.title, summary=scene_draft.summary, start_s=0,
                                duration_s=1, shots=shots))
        return scenes
