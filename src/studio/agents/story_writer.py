"""Story Writer Agent: writes the narration one chapter at a time."""
from __future__ import annotations

from pydantic import BaseModel, field_validator

from studio.agents.base import Agent
from studio.schemas import Chapter, ChapterPlan, ProductionPlan, ResearchBrief, StyleBible


class ChapterDraft(BaseModel):
    title: str
    narration: str

    @field_validator("narration")
    @classmethod
    def _spoken_text(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value.split()) < 15:
            raise ValueError("narration is too short")
        return value


class StoryWriter(Agent):
    name = "story_writer"
    system = ("You are the Story Writer. You write original, funny, educational narration that will be "
              "read aloud by one narrator. Write only words to be spoken: no stage directions, headings, "
              "markdown, emojis or speaker labels. Address the viewer as 'you'.")

    def write_chapter(self, chapter: ChapterPlan, target_words: int, plan: ProductionPlan,
                      research: ResearchBrief, style: StyleBible, topic: str, previous_tail: str = "",
                      current: Chapter | None = None, feedback: str = "") -> Chapter:
        notes = next((c for c in research.chapters if c.index == chapter.index), None)
        uncertain = [c.text for c in research.claims if c.support == "uncertain"]
        position = ("This is the FIRST chapter: its first sentence is the opening hook of the whole video."
                    if chapter.index == plan.chapters[0].index else
                    "This is the LAST chapter: end the video with a satisfying, memorable close."
                    if chapter.index == plan.chapters[-1].index else
                    "Open with a short transition from the previous chapter.")
        user = (
            f"Video topic: {topic}. Tone: {plan.tone}. Audience: {plan.audience}.\n"
            f"Main character: {style.characters[0].name} ({style.characters[0].role or 'the viewer avatar'}).\n"
            f"All chapters: {', '.join(f'{c.index}. {c.title}' for c in plan.chapters)}\n\n"
            f"Write chapter {chapter.index}: \"{chapter.title}\". Purpose: {chapter.purpose}\n{position}\n"
            f"Length: about {target_words} words (between {round(target_words * 0.93)} and "
            f"{round(target_words * 1.07)}).\n"
            f"Research for this chapter: {notes.summary if notes else 'n/a'}\n"
            f"Key points: {'; '.join(notes.key_points) if notes else 'n/a'}\n"
            f"Do not state these uncertain claims as fact: {'; '.join(uncertain[:8]) or 'none'}\n"
            f"Previous chapter ended with: {previous_tail or '(this is the start)'}")
        action = "writer.chapter"
        if current and feedback:
            action = "writer.revise"
            user += (f"\n\nYou are REVISING this chapter. Current text:\n<data>\n{current.narration}\n</data>\n"
                     f"Required changes:\n{feedback}\nKeep what works; return the full revised chapter.")
        draft = self.ask(action, ChapterDraft, self.system, user, context={
            "index": chapter.index, "title": chapter.title, "target_words": target_words, "topic": topic})
        return Chapter(index=chapter.index, title=chapter.title, narration=draft.narration)
