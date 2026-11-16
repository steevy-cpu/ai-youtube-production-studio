"""Dry-run stand-in for the OpenRouter client. Makes no network calls.

It exposes the same methods as OpenRouterClient and answers each agent request with a
deterministic, clearly labelled sample built from the request's `task` and `context`.
Tests can script specific answers through `overrides`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from studio.config import Settings
from studio.services.openrouter import MediaResult, Usage
from studio.services.placeholders import sample_audio, sample_image, sample_video
from studio.utils import word_count

MOCK_MODEL = "dry-run-sample"
RANKS = ["The Curious Beginner", "Intern", "Junior", "Mid-Level", "Senior", "Team Lead", "Principal",
         "Director", "Executive", "Living Legend"]
SENTENCES = [
    "This is sample narration produced by the dry run, standing in for the real script about {topic}.",
    "At the {rank} stage of {topic}, every day brings a new puzzle and a slightly bigger coffee.",
    "You learn the tools, you break something small, and you quietly fix it before anyone notices.",
    "People start asking for your opinion, which is flattering right up until they actually follow it.",
    "The meetings get longer, the problems get stranger, and the stakes get a little more real.",
    "Somewhere along the way you realise the job was never about the tools, it was about people.",
    "Your calendar fills up, your snacks improve, and your to-do list becomes a living document.",
    "Every mistake becomes a story, and every story becomes advice you give to the next newcomer.",
    "You celebrate a small win, take a deep breath, and then the next challenge knocks on the door.",
    "And just when you feel comfortable, the next rank appears on the horizon, waving politely.",
]


def _verdict(approved: bool = True, score: float = 0.9, issue: str = "") -> dict:
    return {"approved": approved, "score": score,
            "criteria": [{"name": "matches request", "passed": approved, "note": "dry-run sample verdict"}],
            "explanation": "Dry-run sample verdict (no model was consulted).",
            "issues": [issue] if issue else [],
            "suggested_changes": ["Make the main subject clearer and larger in frame."] if issue else []}


class MockOpenRouterClient:
    is_mock = True

    def __init__(self, settings: Settings, overrides: dict[str, list[Any]] | None = None):
        self.settings = settings
        self.overrides = overrides or {}       # task -> queue of dicts or callables(context) -> dict
        self.calls: list[str] = []             # every task name requested, in order
        self._demo_done: set[str] = set()

    # ---- same surface as OpenRouterClient -------------------------------------
    def capabilities(self) -> tuple[dict[str, Any], list[str]]:
        return {"image_references": self.settings.image_use_references != "off", "tts_instructions": True,
                "video_first_frame": True, "image_reference_max": 3, "notes": ["Dry run: no model capabilities were checked."]}, []

    def chat(self, model: str, system: str, user: str, *, images: list[Path] | None = None,
             json_schema: dict | None = None, task: str = "", context: dict | None = None
             ) -> tuple[str, Usage]:
        self.calls.append(task)
        context = context or {}
        queue = self.overrides.get(task)
        if queue:
            answer = queue.pop(0)
            answer = answer(context) if callable(answer) else answer
        else:
            builder: Callable[[dict], dict] | None = getattr(self, "_" + task.replace(".", "_"), None)
            if builder is None:
                raise NotImplementedError(f"dry-run sample provider has no answer for task '{task}'")
            answer = builder(context)
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return text, Usage(model=MOCK_MODEL, cost_usd=0.0, note="dry run: no API call")

    def generate_image(self, model: str, prompt: str, *, aspect_ratio: str = "16:9",
                       references: list[Path] | None = None, task: str = "",
                       context: dict | None = None) -> MediaResult:
        self.calls.append(task or "image")
        label = (context or {}).get("label", prompt[:120])
        return MediaResult(sample_image(label, seed=len(self.calls)), "image/png",
                           Usage(model=MOCK_MODEL, cost_usd=0.0, note="dry run: no API call"),
                           {"references_used": len(references or [])})

    def generate_video(self, model: str, prompt: str, *, duration_s: int, resolution: str,
                       aspect_ratio: str, first_frame: Path | None = None, task: str = "",
                       context: dict | None = None) -> MediaResult:
        self.calls.append(task or "video")
        return MediaResult(sample_video(prompt[:100], duration_s), "video/mp4",
                           Usage(model=MOCK_MODEL, cost_usd=0.0, note="dry run: no API call"),
                           {"job_id": "dry-run", "first_frame_used": bool(first_frame)})

    def text_to_speech(self, model: str, text: str, *, voice: str, fmt: str = "mp3", speed: float = 1.0,
                       instructions: str = "", task: str = "", context: dict | None = None) -> MediaResult:
        self.calls.append(task or "tts")
        seconds = word_count(text) / self.settings.words_per_minute * 60 / max(speed, 0.1)
        return MediaResult(sample_audio(seconds), "audio/wav",
                           Usage(model=MOCK_MODEL, cost_usd=0.0, note="dry run: no API call"))

    def transcribe(self, model: str, audio_path: Path, *, language: str = "en") -> tuple[str, Usage]:
        raise NotImplementedError("dry run has no transcription")

    # ---- sample answers, one per agent task --------------------------------------
    def _demo_reject(self, key: str) -> bool:
        """True exactly once per key when demo rejections are enabled."""
        if not self.settings.dry_run_demo_rejections or key in self._demo_done:
            return False
        self._demo_done.add(key)
        return True

    def _director_plan(self, c: dict) -> dict:
        count = max(c["min_chapters"], min(c["max_chapters"], round(c["target_seconds"] / 75)))
        ranks = RANKS[:count] if count <= len(RANKS) else RANKS + [f"Rank {i}" for i in range(len(RANKS), count)]
        seconds = round(c["target_seconds"] / count, 1)
        return {
            "plan": {"creative_goals": [f"[SAMPLE] Entertain and teach viewers about careers in {c['topic']}",
                                        "[SAMPLE] Keep one recurring main character across every rank"],
                     "audience": "Curious beginners and students (sample)",
                     "tone": "Playful, warm, lightly sarcastic (sample)",
                     "chapters": [{"index": i + 1, "title": rank, "target_seconds": seconds,
                                   "purpose": f"Show daily life at the '{rank}' level (sample)"}
                                  for i, rank in enumerate(ranks)],
                     "notes": "Dry-run sample plan."},
            "style_bible": {
                "characters": [{"name": "Riley", "role": "main character",
                                "appearance": "round-faced cartoon person with short teal hair and big eyes",
                                "clothing": "orange hoodie, grey trousers, white sneakers"}],
                "art_style": "flat 2D cartoon, bold clean outlines, soft cel shading (sample)",
                "color_palette": ["teal", "warm orange", "cream", "charcoal"],
                "environment_guidelines": "simple stylised offices and city backdrops, minimal clutter",
                "framing_guidelines": "medium and wide shots, main character on a rule-of-thirds line",
                "negative_prompt": "photorealism, text, watermarks, logos, real people, extra fingers"}}

    def _research_brief(self, c: dict) -> dict:
        source = (c.get("source_ids") or [None])[0]
        return {"overview": f"[SAMPLE] Dry-run research overview for {c['topic']}. No model was consulted.",
                "chapters": [{"index": ch["index"], "title": ch["title"],
                              "summary": f"Sample summary of the '{ch['title']}' level in {c['topic']}.",
                              "key_points": ["Typical responsibilities (sample)", "Common mistakes (sample)"]}
                             for ch in c["chapters"]],
                "claims": [{"text": f"Sample claim about {c['topic']} careers.", "chapter_index": 1,
                            "support": "verified_source" if source else "background_knowledge",
                            "source_id": source},
                           {"text": "Sample claim that could not be verified.", "chapter_index": 1,
                            "support": "uncertain"}],
                "notes": ["Dry-run sample brief."]}

    def _writer_chapter(self, c: dict) -> dict:
        words, i = [], c["index"]
        while len(words) < c["target_words"]:
            words += SENTENCES[i % len(SENTENCES)].format(topic=c["topic"], rank=c["title"]).split()
            i += 1
        words = words[:c["target_words"]]            # hit the requested length exactly
        words[-1] = words[-1].rstrip(".,") + "."
        return {"title": c["title"], "narration": " ".join(words)}

    _writer_revise = _writer_chapter

    def _director_review_script(self, c: dict) -> dict:
        return {"approved": True, "reasons": ["Dry-run sample approval."], "chapter_feedback": []}

    def _director_final(self, c: dict) -> dict:
        return {"approved": True, "reasons": ["Dry-run sample approval."], "chapter_feedback": []}

    _director_thumbnail = _director_final

    def _scene_chapter(self, c: dict) -> dict:
        per_shot = c["target_shot_seconds"] * c["words_per_minute"] / 60
        shots, start, words = [], 1, 0
        for number, count in enumerate(c["sentence_words"], start=1):
            words += count
            if words >= per_shot or number == len(c["sentence_words"]):
                shots.append({"first_sentence": start, "last_sentence": number,
                              "visual_description": f"[SAMPLE] Riley at the '{c['title']}' stage, "
                                                    f"moment {len(shots) + 1}",
                              "framing": "medium shot", "media_type": "image",
                              "camera_move": ["zoom_in", "pan_right", "zoom_out", "pan_left"][len(shots) % 4]})
                start, words = number + 1, 0
        if len(shots) > 1:         # a trailing fragment of a few words would be shorter than the minimum shot
            tail = sum(c["sentence_words"][shots[-1]["first_sentence"] - 1:shots[-1]["last_sentence"]])
            if tail * 60 / c["words_per_minute"] < 2:
                last = shots.pop()
                shots[-1]["last_sentence"] = last["last_sentence"]
        if shots and c.get("allow_animation"):
            shots[0]["media_type"] = "animation"
        scenes = [{"title": f"{c['title']} - part {n // 3 + 1}", "summary": "Sample scene.",
                   "shots": shots[n:n + 3]} for n in range(0, len(shots), 3)]
        return {"scenes": scenes}

    def _scene_visuals(self, c: dict) -> dict:
        return {"visuals": [{"shot_index": shot["index"],
                             "visual_description": f"[SAMPLE, redrawn] {shot['visual_description']}"}
                            for shot in c["shots"]]}

    def _scene_trim(self, c: dict) -> dict:
        remove, cut = [], 0
        for number, words in c["candidates"]:
            if cut >= c["words_to_cut"]:
                break
            remove.append(number)
            cut += words
        return {"remove": remove, "reason": "Dry-run sample trim."}

    def _prompt_scene(self, c: dict) -> dict:
        return {"prompts": [{
            "shot_index": shot["index"],
            "prompt": f"[SAMPLE] {shot['visual_description']}, flat 2D cartoon, bold outlines",
            "negative_prompt": "photorealism, text, watermarks", "composition": "subject on left third",
            "lighting": "soft even daylight", "characters": ["Riley"],
            "animation_prompt": "gentle camera push-in, character blinks and waves"
            if shot.get("media_type") == "animation" else ""} for shot in c["shots"]]}

    def _prompt_revise(self, c: dict) -> dict:
        return {"prompt": f"[SAMPLE, revised] {c['visual_description']}, flat 2D cartoon, subject large "
                          "and clear", "negative_prompt": "photorealism, text, watermarks",
                "composition": "subject centred and large", "lighting": "soft even daylight",
                "characters": ["Riley"]}

    def _prompt_thumbnail(self, c: dict) -> dict:
        return {"prompt": f"[SAMPLE] Thumbnail: {c['concept']}, flat 2D cartoon, high contrast, no text",
                "negative_prompt": "text, letters, watermarks, logos, photorealism",
                "composition": "large expressive character on the right, empty space on the left",
                "lighting": "bright rim light", "characters": ["Riley"]}

    def _judge_prompts(self, c: dict) -> dict:
        reject = c["shot_indexes"][0] if self._demo_reject("prompt") else None
        return {"verdicts": [dict(_verdict(i != reject, 0.9 if i != reject else 0.4,
                                           "" if i != reject else "Demo rejection: subject is not prominent "
                                           "enough (dry-run sample)"), shot_index=i)
                             for i in c["shot_indexes"]]}

    def _judge_thumbnail_prompt(self, c: dict) -> dict:
        return _verdict()

    def _judge_image(self, c: dict) -> dict:
        if c.get("purpose") == "thumbnail" and self._demo_reject("thumbnail"):
            return _verdict(False, 0.45, "Demo rejection: thumbnail subject too small (dry-run sample)")
        return _verdict()

    def _seo_review_thumbnail(self, c: dict) -> dict:
        return _verdict()

    def _seo_package(self, c: dict) -> dict:
        title = c["title"]
        return {"seo": {"titles": [title, f"{title} (Explained)", f"Every Rank in {c['topic']}, Ranked"],
                        "selected_title": title,
                        "description": f"[SAMPLE] Dry-run description for '{title}'. Follow one character "
                                       f"through every rank in {c['topic']}.",
                        "tags": [c["topic"].lower(), "career", "explained", "animation"],
                        "keywords": [f"{c['topic'].lower()} career path"],
                        "hook": "Open on the funniest moment of the first rank (sample).",
                        "retention_suggestions": ["Tease the final rank in the first 15 seconds (sample)."]},
                "thumbnail_brief": {
                    "concept": f"Riley climbing a ladder of {c['topic']} ranks",
                    "main_subject": "Riley, surprised, halfway up a glowing ladder",
                    "composition": "character right, ladder diagonal, clear space top-left for text",
                    "color_palette": ["teal", "warm orange", "cream"], "emotion": "wide-eyed surprise",
                    "overlay_text": "EVERY RANK", "style_requirements": "flat 2D cartoon, bold outlines",
                    "avoid": ["copyrighted characters", "real logos", "misleading claims"]}}

    def _voice_plan(self, c: dict) -> dict:
        return {"speed": 1.0, "instructions": "Friendly, upbeat storyteller with clear pacing (sample).",
                "pacing_notes": "Short pause between scenes (sample)."}

    def _qa_continuity(self, c: dict) -> dict:
        return _verdict()

    def _compliance_screen(self, c: dict) -> dict:
        return {"findings": [{"check": name, "status": "pass", "detail": "Dry-run sample screening."}
                             for name in c["checks"]]}

    _compliance_metadata = _compliance_screen
