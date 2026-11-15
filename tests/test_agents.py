"""Agent output validation and agent-level rules, using scripted mock model answers."""
import json

import pytest

from studio.agents.base import Agent, AgentOutputError, Verdict, llm_schema
from studio.agents.compliance import Compliance
from studio.agents.editor import Editor, caption_chunks
from studio.agents.research import Research
from studio.agents.scene_director import SceneDirector, coverage_problem, retime
from studio.schemas import (Chapter, ChapterPlan, Character, MediaAsset, ProductionPlan, ResearchBrief,
                            Script, Storyboard, StyleBible)
from studio.services.placeholders import sample_audio, sample_image

VALID = {"approved": True, "score": 0.9}


def _style():
    return StyleBible(characters=[Character(name="Riley", appearance="teal hair", clothing="hoodie")],
                      art_style="flat cartoon", color_palette=["teal"], environment_guidelines="simple",
                      framing_guidelines="medium", negative_prompt="text")


def _plan():
    return ProductionPlan(creative_goals=["g"], audience="a", tone="t",
                          chapters=[ChapterPlan(index=1, title="Intern", target_seconds=24)])


# ---- schema validation of model output ------------------------------------------------
def test_valid_output_is_parsed_even_inside_code_fences(ctx):
    ctx.client.overrides["judge.x"] = ["```json\n" + json.dumps(VALID) + "\n```"]
    verdict = Agent(ctx).ask("judge.x", Verdict, "system", "user")
    assert verdict.approved and verdict.score == 0.9


def test_malformed_output_gets_a_bounded_repair_then_passes(ctx):
    ctx.client.overrides["judge.x"] = ["I think it looks great!", {"approved": True, "score": 7}, VALID]
    verdict = Agent(ctx).ask("judge.x", Verdict, "system", "user")
    assert verdict.score == 0.9
    statuses = [r["status"] for r in ctx.store.agent_runs(ctx.project_id)]
    assert statuses == ["invalid_output", "invalid_output", "ok"]         # every attempt is traceable


def test_persistently_invalid_output_never_passes_silently(ctx):
    ctx.client.overrides["judge.x"] = [{"approved": "maybe"}] * 5
    with pytest.raises(AgentOutputError, match="failed validation after 3 attempts"):
        Agent(ctx).ask("judge.x", Verdict, "system", "user")
    assert ctx.client.calls.count("judge.x") == 3                         # max_llm_repair_attempts + 1


def test_llm_schema_hides_bookkeeping_fields():
    props = llm_schema(Chapter)["properties"]
    assert set(props) == {"index", "title", "narration"}


def test_judge_needs_both_approval_and_score(ctx):
    agent = Agent(ctx)
    low = agent.evaluation(Verdict(approved=True, score=0.5), target_id="t", target_kind="prompt", threshold=0.7)
    assert not low.passed and "below the required" in low.issues[0]
    assert agent.evaluation(Verdict(approved=True, score=0.8), target_id="t", target_kind="prompt",
                            threshold=0.7).passed
    assert len(ctx.store.evaluations(ctx.project_id)) == 2


# ---- research: citations cannot be invented ----------------------------------------------
def test_research_downgrades_invented_citations(ctx):
    ctx.client.overrides["research.brief"] = [{
        "overview": "o", "chapters": [{"index": 1, "title": "Intern", "summary": "s"}],
        "claims": [{"text": "made up", "support": "verified_source", "source_id": "S9"},
                   {"text": "general", "support": "background_knowledge"}]}]
    brief = Research(ctx).brief("Cybersecurity", _plan(), sources_dir=None)
    assert [c.support for c in brief.claims] == ["uncertain", "background_knowledge"]
    assert brief.claims[0].source_id is None and brief.sources == []
    assert any("No local sources" in n for n in brief.notes)


def test_research_accepts_citations_of_local_files(ctx, tmp_path):
    (tmp_path / "handbook.md").write_text("SOC analysts triage alerts.")
    (tmp_path / "ignored.pdf").write_text("x")
    ctx.client.overrides["research.brief"] = [{
        "overview": "o", "chapters": [{"index": 1, "title": "Intern", "summary": "s"}],
        "claims": [{"text": "SOC analysts triage alerts", "support": "verified_source", "source_id": "S1"}]}]
    brief = Research(ctx).brief("Cybersecurity", _plan(), sources_dir=tmp_path)
    assert brief.claims[0].support == "verified_source" and [s.title for s in brief.sources] == ["handbook"]


# ---- scene director: narration cannot be altered, timing is computed -----------------------
def test_scene_director_rejects_storyboards_that_skip_sentences(ctx):
    chapter = Chapter(index=1, title="Intern", narration="One two three. Four five six. Seven eight nine.")
    bad = {"scenes": [{"title": "s", "shots": [{"first_sentence": 1, "last_sentence": 1,
                                                "visual_description": "Riley at a desk"}]}]}
    ctx.client.overrides["scene.chapter"] = [bad, bad]
    with pytest.raises(AgentOutputError, match="3 sentences"):
        SceneDirector(ctx).chapter_scenes(chapter, _style())

    good = {"scenes": [{"title": "s", "shots": [
        {"first_sentence": 1, "last_sentence": 2, "visual_description": "Riley at a desk"},
        {"first_sentence": 3, "last_sentence": 3, "visual_description": "Riley drinks coffee",
         "media_type": "animation"}]}]}
    ctx.client.overrides["scene.chapter"] = [bad, good]                  # one targeted repair
    scenes = SceneDirector(ctx).chapter_scenes(chapter, _style())
    board = retime(Storyboard(scenes=scenes), ctx.settings)
    assert " ".join(s.narration for s in board.shots) == chapter.narration
    assert [s.media_type for s in board.shots] == ["image", "image"]     # video disabled -> stills only
    assert board.shots[1].start_s == board.shots[0].duration_s and board.shots[0].duration_s == 2.4


def test_coverage_problem():
    from studio.agents.scene_director import SceneDraft, ShotDraft
    shot = lambda a, b: ShotDraft(first_sentence=a, last_sentence=b, visual_description="x" * 12)
    assert coverage_problem([SceneDraft(title="s", shots=[shot(1, 2), shot(3, 4)])], 4) == ""
    assert "next uncovered sentence is 3" in coverage_problem([SceneDraft(title="s", shots=[shot(1, 2), shot(2, 4)])], 4)


# ---- editor: timing follows measured audio ---------------------------------------------------
def test_editor_uses_measured_narration_and_never_shortens_it(ctx, tmp_path):
    chapter = Chapter(index=1, title="Intern", narration="One two three four. Five six seven eight.")
    ctx.client.overrides["scene.chapter"] = [{"scenes": [{"title": "s", "shots": [
        {"first_sentence": 1, "last_sentence": 1, "visual_description": "Riley at a desk"},
        {"first_sentence": 2, "last_sentence": 2, "visual_description": "Riley drinks coffee"}]}]}]
    board = retime(Storyboard(scenes=SceneDirector(ctx).chapter_scenes(chapter, _style())), ctx.settings)
    image, audio = tmp_path / "i.png", tmp_path / "a.wav"
    image.write_bytes(sample_image("x"))
    audio.write_bytes(sample_audio(10.0))                 # far longer than the 3.2s word estimate
    visuals = {s.id: MediaAsset(kind="image", path=str(image)) for s in board.shots}
    narration = {board.scenes[0].id: MediaAsset(kind="audio", path=str(audio), duration_s=10.0)}
    timeline = Editor(ctx).build_timeline(board, visuals, narration)
    assert timeline.audio[0].slot_s >= 10.0 and timeline.total_duration_s >= 10.0
    assert abs(timeline.clips[0].duration_s - 5.0) < 0.11             # half the words -> half the audio
    assert timeline.clips[1].start_s == timeline.clips[0].duration_s   # contiguous
    assert sum(c.duration_s for c in timeline.clips) == pytest.approx(timeline.total_duration_s)
    assert timeline.clips[0].captions[0].text == "One two three four."


def test_caption_chunks_are_short():
    chunks = caption_chunks("This sentence has quite a few more words than a single caption line should hold. Ok.")
    assert all(len(c.split()) <= 11 for c in chunks) and " ".join(chunks).endswith("Ok.")


# ---- compliance ---------------------------------------------------------------------------------
def test_compliance_blocks_on_mandatory_failure_and_flags_human_review(ctx):
    script = Script(title="T", hook="h", chapters=[Chapter(index=1, title="Intern", narration="word " * 20)])
    research = ResearchBrief(overview="o", chapters=[{"index": 1, "title": "Intern", "summary": "s"}],
                             claims=[{"text": "c", "support": "uncertain"}])
    assets = [MediaAsset(kind="image", path="x.png", provenance="")]          # provenance missing
    report = Compliance(ctx).review(script, research, assets)
    by_check = {i.check: i for i in report.items}
    assert report.blocked and by_check["asset_provenance"].status == "fail"
    assert by_check["factual_claims"].status == "needs_human_review"
    assert by_check["synthetic_media_disclosure"].status == "needs_human_review"
    assert "not legal advice" in report.disclaimer

    ctx.client.overrides["compliance.screen"] = [{"findings": [
        {"check": "impersonation", "status": "fail", "detail": "imitates a real presenter"}]}]
    good_assets = [MediaAsset(kind="image", path="x.png", provenance="test")]
    report = Compliance(ctx).review(script, research, good_assets)
    by_check = {i.check: i for i in report.items}
    assert report.blocked and by_check["impersonation"].status == "fail"
    assert by_check["privacy"].status == "needs_human_review"                 # screening gave no result


# ---- generation services validate what the provider returns ----------------------------------------
def test_speech_service_wraps_pcm_and_rejects_unknown_audio(ctx):
    from types import SimpleNamespace

    from studio.schemas import NarrationSegment
    from studio.services.generation import SpeechService
    from studio.services.openrouter import MediaResult, OpenRouterError, Usage

    def fake(media_type, metadata=None):
        return SimpleNamespace(is_mock=False, text_to_speech=lambda *a, **k: MediaResult(
            b"\x01\x00" * 16000, media_type, Usage(model="tts/model", cost_usd=None), metadata or {}))
    segment = NarrationSegment(parent_id="scene", index=1, text="hello there", voice="Kore", estimated_s=1.0)

    ctx.client = fake("audio/pcm", {"sample_rate": 16000})
    asset = SpeechService(ctx).generate(segment, agent="voice_director")
    assert asset.path.endswith(".wav") and asset.duration_s == pytest.approx(1.0, abs=0.01)
    assert asset.metadata["pcm_sample_rate_source"] == "response header" and not asset.is_sample
    assert "no voice cloning" in asset.provenance
    assert ctx.store.cost_summary(ctx.project_id)["unknown_cost_requests"] == 1    # unknown, not zero

    ctx.client = fake("text/html")
    with pytest.raises(OpenRouterError, match="unsupported content type"):
        SpeechService(ctx).generate(segment, agent="voice_director")


def test_image_service_rejects_bytes_that_are_not_an_image(ctx):
    from types import SimpleNamespace

    from studio.services.generation import ImageService
    from studio.services.openrouter import MediaResult, OpenRouterError, Usage
    ctx.client = SimpleNamespace(is_mock=False, generate_image=lambda *a, **k: MediaResult(
        b"<html>error page</html>", "image/png", Usage(model="img/model", cost_usd=0.04)))
    with pytest.raises(OpenRouterError, match="not a valid image"):
        ImageService(ctx).generate("a prompt", kind="image", parent_id="shot", prompt_id=None, name="shot_001",
                                   agent="visual_generation")
    assert ctx.store.assets(ctx.project_id) == []
