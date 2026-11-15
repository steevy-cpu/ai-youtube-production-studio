"""Voice Director Agent: narration pacing, voice settings and TTS segments (one per scene)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from studio.agents.base import Agent
from studio.schemas import MediaAsset, NarrationSegment, ProductionPlan, Storyboard
from studio.services.generation import SpeechService
from studio.utils import word_count


class VoicePlan(BaseModel):
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    instructions: str = Field(min_length=5, max_length=600)
    pacing_notes: str = ""


class VoiceDirector(Agent):
    name = "voice_director"
    system = ("You are the Voice Director. You decide how one narrator should deliver the script: "
              "pace, energy and tone, kept consistent for the whole video.")

    def __init__(self, ctx):
        super().__init__(ctx)
        self.speech = SpeechService(ctx)

    def plan(self, storyboard: Storyboard, plan: ProductionPlan) -> tuple[VoicePlan, list[NarrationSegment]]:
        s = self.ctx.settings
        sample = " ".join(scene.narration for scene in storyboard.scenes[:2])[:700]
        user = (f"Video tone: {plan.tone}. Audience: {plan.audience}. Language: {s.language}.\n"
                f"Configured voice: {s.tts_voice} (a stock voice of the speech model; do not imitate any "
                f"real person).\nOpening narration:\n<data>\n{sample}\n</data>\n\n"
                "Write `instructions`: two or three sentences telling the speech model how to deliver "
                "the narration (tone, energy, pacing, pauses). Set `speed` (1.0 = normal; stay within "
                "0.9-1.1 unless the tone demands otherwise).")
        voice_plan = self.ask("voice.plan", VoicePlan, self.system, user)
        speed = round(s.tts_speed * voice_plan.speed, 2)
        segments = [NarrationSegment(
            parent_id=scene.id, index=scene.index, text=scene.narration, voice=s.tts_voice, speed=speed,
            instructions=voice_plan.instructions,
            estimated_s=round(word_count(scene.narration) / s.words_per_minute * 60 / speed, 2))
            for scene in storyboard.scenes]
        return voice_plan, segments

    def speak(self, segment: NarrationSegment) -> MediaAsset:
        asset = self.speech.generate(segment, agent=self.name)
        self.note("text_to_speech", f"segment {segment.index} -> {asset.duration_s:.1f}s")
        return asset
