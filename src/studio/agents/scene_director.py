"""Scene Director Agent: turns each chapter into scenes and shots.

The model decides WHERE to cut and WHAT to show. It refers to narration by sentence number,
so it can never alter the approved wording; durations and timestamps are computed by code.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from studio.agents.base import Agent, AgentOutputError
from studio.agents.prompt_engineer import SHOT_RULES, style_text
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


class ShotVisual(BaseModel):
    shot_index: int
    visual_description: str = Field(min_length=10)


class SceneVisuals(BaseModel):
    visuals: list[ShotVisual] = Field(min_length=1)


class TrimPlan(BaseModel):
    remove: list[int] = []
    reason: str = ""


MAX_CHAPTER_CUT = 0.5       # a trim never removes more than half of one chapter's words


def numbered_sentences(storyboard: Storyboard) -> list[dict]:
    """Every narration sentence in video order: number, chapter, shot id, text, words, protected.

    The first and the last sentence of each chapter are protected (hook and hand-over to the next rank).
    """
    rows = []
    for scene in storyboard.scenes:
        for shot in scene.shots:
            for text in split_sentences(shot.narration):
                rows.append({"n": len(rows) + 1, "chapter": scene.chapter_index, "shot": shot.id,
                             "text": text, "words": word_count(text), "protected": False})
    for chapter in {r["chapter"] for r in rows}:
        own = [r for r in rows if r["chapter"] == chapter]
        own[0]["protected"] = own[-1]["protected"] = True
    return rows


def allowed_removals(rows: list[dict], wanted: list[int], settings: Settings) -> set[int]:
    """The subset of `wanted` that code accepts: no protected sentence, at most half of a chapter,
    and no shot left shorter than the minimum shot length."""
    by_n = {r["n"]: r for r in rows}
    chapter_words = {c: sum(r["words"] for r in rows if r["chapter"] == c) for c in {r["chapter"] for r in rows}}
    min_words = settings.min_shot_seconds * settings.words_per_minute / 60
    removed: set[int] = set()
    for n in dict.fromkeys(wanted):
        row = by_n.get(n)
        if row is None or row["protected"]:
            continue
        cut = sum(by_n[m]["words"] for m in removed if by_n[m]["chapter"] == row["chapter"]) + row["words"]
        if cut > chapter_words[row["chapter"]] * MAX_CHAPTER_CUT:
            continue
        left = sum(r["words"] for r in rows if r["shot"] == row["shot"] and r["n"] not in removed | {n})
        if 0 < left < min_words:
            continue
        removed.add(n)
    return removed


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
            "that illustrates those sentences.\n"
            f"{animation_rule}\n\nEvery picture is drawn under this Style Bible:\n{style_text(style)}\n\n"
            f"{SHOT_RULES}\n\nWrite every visual_description so that it already respects these rules: "
            "never ask for words, digits, labels or symbols (describe the wordless stand-in yourself), "
            "never change a character's outfit, and name colours from the palette.")
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

    def revise_visuals(self, scene: Scene, rejected: list[tuple[Shot, str]], style: StyleBible
                       ) -> dict[str, str]:
        """A new picture for shots whose visual could not be turned into an acceptable prompt.

        Only `visual_description` changes: narration, timing and shot ids stay as they are, so
        nothing recorded is lost. Returns {shot_id: new visual_description} for the shots answered.
        """
        listing = "\n\n".join(
            f"shot_index {shot.index} ({shot.framing})\n  narration: {shot.narration}\n"
            f"  current visual: {shot.visual_description}\n  why it failed:\n{feedback}"
            for shot, feedback in rejected)
        user = (f"Style Bible:\n{style_text(style)}\n\nScene \"{scene.title}\": {scene.summary}\n"
                f"<data>\n{listing}\n</data>\n\n{SHOT_RULES}\n\n"
                "The Prompt Judge could not accept an image prompt for these shots because of what "
                "the picture is asked to show. Describe a DIFFERENT picture for each listed shot: one "
                "concrete still image (who, doing what, where, key props, expression) that illustrates "
                "the same narration, keeps the idea, and can be drawn under the rules without any "
                "stand-in being needed. Do not repeat the element that failed. Return `visuals`: one "
                "entry per listed shot_index.")
        result = self.ask("scene.visuals", SceneVisuals, self.system, user, context={
            "shots": [{"index": shot.index, "visual_description": shot.visual_description}
                      for shot, _ in rejected]})
        by_index = {v.shot_index: v.visual_description.strip() for v in result.visuals}
        return {shot.id: by_index[shot.index] for shot, _ in rejected
                if by_index.get(shot.index) and by_index[shot.index] != shot.visual_description}

    def trim(self, storyboard: Storyboard, words_to_cut: int, reason: str) -> set[int]:
        """Choose whole sentences to drop so the video gets shorter. Wording is never changed.

        Returns the accepted sentence numbers (see `numbered_sentences`); may be empty.
        """
        rows = numbered_sentences(storyboard)
        listing, chapter = [], 0
        for r in rows:
            if r["chapter"] != chapter:
                chapter = r["chapter"]
                listing.append(f"Chapter {chapter}:")
            listing.append(f"{r['n']}. [{r['words']} words{', KEEP' if r['protected'] else ''}] {r['text']}")
        user = (f"The finished video is too long: {reason}\nThe narration below is the whole video, one "
                "numbered sentence per line.\n<data>\n" + "\n".join(listing) + "\n</data>\n\n"
                f"Choose sentences to remove so that about {words_to_cut} words disappear (a little more "
                "is fine, less is not). Rules: never remove a sentence marked KEEP; never remove more "
                "than half of one chapter; remove asides, repetition and detail first; every rank must "
                "still be told and each remaining sentence must still follow from the one before it. "
                "You cannot reword anything. Return `remove`: the sentence numbers, and a short `reason`.")
        context = {"words_to_cut": words_to_cut,
                   "candidates": [[r["n"], r["words"]] for r in rows if not r["protected"]]}
        removed: set[int] = set()
        for _ in range(2):          # one more request if the first cut is clearly too small
            plan = self.ask("scene.trim", TrimPlan, self.system, user, context=context)
            removed = allowed_removals(rows, sorted(removed) + plan.remove, self.ctx.settings)
            cut = sum(r["words"] for r in rows if r["n"] in removed)
            if cut >= words_to_cut * 0.8:
                break
            user += (f"\n\nYour last answer removes only {cut} of the {words_to_cut} words needed "
                     f"(accepted so far: {sorted(removed)}). Name additional sentences to remove.")
        return removed
