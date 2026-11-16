"""Pydantic contracts exchanged between agents, guards, memory and the renderer.

Every persisted object derives from `Record` (id, project id, parent id, timestamp,
status, version). Models used directly as LLM output accept missing bookkeeping
fields; the owning agent stamps them after validation.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from studio.utils import new_id, now_iso, word_count

CameraMove = Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "static"]
MediaType = Literal["image", "animation"]
Support = Literal["verified_source", "background_knowledge", "uncertain"]


class Record(BaseModel):
    id: str = Field(default_factory=new_id)
    project_id: str = ""
    parent_id: str | None = None
    created_at: str = Field(default_factory=now_iso)
    status: str = "draft"
    version: int = 1


def _non_empty(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("must not be empty")
    return value.strip()


# ---- project / planning ---------------------------------------------------
class ProjectBrief(Record):
    topic: str
    title: str
    mode: str
    target_duration_s: float
    width: int
    height: int
    language: str = "English"
    sources_dir: str = ""
    enable_video: bool = False      # per-project opt-in to AI video clips (in addition to ENABLE_VIDEO)

    _v = field_validator("topic", "title")(_non_empty)


class ChapterPlan(BaseModel):
    index: int = Field(ge=1)
    title: str
    purpose: str = ""
    target_seconds: float = Field(gt=0)

    _v = field_validator("title")(_non_empty)


class ProductionPlan(Record):
    creative_goals: list[str] = Field(min_length=1)
    audience: str
    tone: str
    chapters: list[ChapterPlan] = Field(min_length=1)
    notes: str = ""


class Character(BaseModel):
    name: str
    role: str = ""
    appearance: str
    clothing: str

    _v = field_validator("name", "appearance")(_non_empty)


class StyleBible(Record):
    characters: list[Character] = Field(min_length=1)
    art_style: str
    color_palette: list[str] = Field(min_length=1)
    environment_guidelines: str
    framing_guidelines: str
    negative_prompt: str
    reference_asset_ids: list[str] = []
    limitations: list[str] = []

    _v = field_validator("art_style", "negative_prompt")(_non_empty)


# ---- research ---------------------------------------------------------------
class Source(BaseModel):
    id: str
    title: str
    path: str


class Claim(BaseModel):
    text: str
    chapter_index: int | None = None
    support: Support = "background_knowledge"
    source_id: str | None = None
    note: str = ""


class ResearchChapter(BaseModel):
    index: int = Field(ge=1)
    title: str
    summary: str
    key_points: list[str] = []


class ResearchBrief(Record):
    topic: str = ""
    overview: str
    chapters: list[ResearchChapter] = Field(min_length=1)
    claims: list[Claim] = []
    sources: list[Source] = []
    notes: list[str] = []


# ---- script -----------------------------------------------------------------
class Chapter(Record):
    index: int = Field(ge=1)
    title: str
    narration: str

    _v = field_validator("title", "narration")(_non_empty)

    @property
    def words(self) -> int:
        return word_count(self.narration)


class Script(Record):
    title: str
    hook: str
    chapters: list[Chapter] = Field(min_length=1)

    @property
    def words(self) -> int:
        return sum(c.words for c in self.chapters)


# ---- storyboard ---------------------------------------------------------------
class Shot(Record):
    index: int = Field(ge=1)              # global order in the video
    chapter_index: int = Field(ge=1)
    narration: str
    visual_description: str
    framing: str = "medium shot"
    camera_move: CameraMove = "zoom_in"
    media_type: MediaType = "image"
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)


class Scene(Record):
    index: int = Field(ge=1)
    chapter_index: int = Field(ge=1)
    title: str
    summary: str = ""
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    shots: list[Shot] = Field(min_length=1)

    @property
    def narration(self) -> str:
        return " ".join(s.narration for s in self.shots)


class Storyboard(Record):
    scenes: list[Scene] = Field(min_length=1)

    @property
    def shots(self) -> list[Shot]:
        return [shot for scene in self.scenes for shot in scene.shots]

    @property
    def duration_s(self) -> float:
        return sum(scene.duration_s for scene in self.scenes)


# ---- prompts ------------------------------------------------------------------
class ImagePrompt(Record):
    """parent_id is the shot id, the thumbnail brief id, or the style bible id."""
    purpose: Literal["shot", "thumbnail", "reference"] = "shot"
    media_type: Literal["image"] = "image"
    prompt: str
    negative_prompt: str = ""
    composition: str = ""
    lighting: str = ""
    characters: list[str] = []
    aspect_ratio: str = "16:9"

    _v = field_validator("prompt")(_non_empty)


class AnimationPrompt(Record):
    """parent_id is the shot id."""
    media_type: Literal["animation"] = "animation"
    prompt: str
    motion: str = ""
    duration_s: float = Field(default=4, gt=0)

    _v = field_validator("prompt")(_non_empty)


# ---- narration ------------------------------------------------------------------
class NarrationSegment(Record):
    """One TTS request; parent_id is the scene id."""
    index: int = Field(ge=1)
    text: str
    voice: str
    speed: float = 1.0
    instructions: str = ""
    estimated_s: float = Field(gt=0)

    _v = field_validator("text", "voice")(_non_empty)


# ---- thumbnail / SEO --------------------------------------------------------------
class ThumbnailBrief(Record):
    concept: str
    main_subject: str
    composition: str
    color_palette: list[str] = Field(min_length=1)
    emotion: str
    overlay_text: str = Field(default="", max_length=40)
    style_requirements: str
    avoid: list[str] = []

    _v = field_validator("concept", "main_subject", "composition")(_non_empty)


class SEOPackage(Record):
    titles: list[str] = Field(min_length=1)
    selected_title: str
    description: str
    tags: list[str] = Field(min_length=1)
    keywords: list[str] = []
    hook: str = ""
    retention_suggestions: list[str] = []
    disclaimer: str = ""

    _v = field_validator("selected_title", "description")(_non_empty)


# ---- media ------------------------------------------------------------------------
class MediaAsset(Record):
    """A file on disk. parent_id is the shot / scene / brief it was made for."""
    kind: Literal["image", "video", "audio", "thumbnail", "reference", "final_video", "music"]
    path: str
    mime: str = ""
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    model: str = ""
    prompt_id: str | None = None
    provenance: str = ""        # e.g. "openrouter:<model>", "dry-run sample", "user supplied"
    license: str = ""
    is_sample: bool = False     # True for locally generated placeholders (never AI output)
    metadata: dict[str, Any] = {}


# ---- evaluation / revision ----------------------------------------------------------
class Criterion(BaseModel):
    name: str
    passed: bool
    note: str = ""


class EvaluationResult(Record):
    """Result of a software guard or an LLM judge. parent_id is the evaluated object."""
    target_kind: str
    evaluator: str
    passed: bool
    score: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = None
    criteria: list[Criterion] = []
    explanation: str = ""
    issues: list[str] = []
    suggested_changes: list[str] = []
    metrics: dict[str, Any] = {}


class RevisionRequest(Record):
    """Structured feedback routed by the orchestrator. parent_id is the artifact to redo."""
    target_kind: str
    target_task: str
    responsible_agent: str
    requested_by: str
    issues: list[str] = []
    suggested_changes: list[str] = []

    def as_feedback(self) -> str:
        lines = [f"- problem: {i}" for i in self.issues] + [f"- fix: {s}" for s in self.suggested_changes]
        return "\n".join(lines) or "- no details given"


# ---- editing --------------------------------------------------------------------------
class Caption(BaseModel):
    text: str
    start_s: float
    end_s: float


class TimelineClip(BaseModel):
    shot_id: str
    scene_id: str
    asset_path: str
    asset_kind: Literal["image", "video"]
    start_s: float
    duration_s: float = Field(gt=0)
    camera_move: CameraMove = "static"
    fade_in: bool = False
    fade_out: bool = False
    captions: list[Caption] = []


class AudioClip(BaseModel):
    scene_id: str
    path: str
    start_s: float
    slot_s: float = Field(gt=0)     # time reserved on the timeline (audio + pause)
    audio_s: float = Field(gt=0)    # measured narration length


class EditingTimeline(Record):
    width: int
    height: int
    fps: int
    clips: list[TimelineClip] = Field(min_length=1)
    audio: list[AudioClip] = Field(min_length=1)
    music_path: str = ""
    music_volume: float = 0.0
    music_title: str = ""            # "Title by Author" of a library track
    music_license: str = ""
    music_source: str = ""
    transition_s: float = 0.3
    total_duration_s: float
    notes: list[str] = []


# ---- reports ---------------------------------------------------------------------------
class ComplianceItem(BaseModel):
    check: str
    mandatory: bool
    status: Literal["pass", "fail", "needs_human_review"]
    detail: str = ""


class ComplianceReport(Record):
    items: list[ComplianceItem]
    blocked: bool
    human_review_required: list[str] = []
    synthetic_media_disclosure: str = ""
    disclaimer: str = ("Automated compliance checks are screening tools only. "
                       "They are not legal advice and do not guarantee compliance.")


class ProductionReport(Record):
    topic: str
    title: str
    mode: str
    final_status: str
    outputs: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []
    costs: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    limitations: list[str] = []
    human_decisions: list[dict[str, Any]] = []
