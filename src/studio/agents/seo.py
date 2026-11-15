"""SEO & Growth Agent: metadata package, thumbnail brief and thumbnail review.

It has no analytics data and makes no performance predictions.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from studio.agents.base import Agent, Verdict
from studio.schemas import EvaluationResult, ProductionPlan, Script, SEOPackage, ThumbnailBrief

DISCLAIMER = ("Suggestions are editorial ideas generated without any YouTube analytics data. "
              "They are not predictions of views, ranking or virality.")


class SEOOutput(BaseModel):
    seo: SEOPackage
    thumbnail_brief: ThumbnailBrief


class SEO(Agent):
    name = "seo"
    system = ("You are the SEO & Growth Agent for a YouTube channel. You write honest, original "
              "metadata and a thumbnail concept that accurately represent the video. No clickbait "
              "claims the video does not deliver, no performance promises.")

    def package(self, script: Script, plan: ProductionPlan, topic: str, art_style: str) -> SEOOutput:
        chapters = "\n".join(f"{c.index}. {c.title}: {c.narration[:220]}..." for c in script.chapters)
        user = (f"Working title: {script.title}\nTopic: {topic}\nAudience: {plan.audience}\n"
                f"Opening hook: {script.hook}\n<data>\n{chapters}\n</data>\n\n"
                "Produce:\n- seo.titles: 3-5 original title options (under 70 characters); "
                "seo.selected_title: the best one\n- seo.description: a YouTube description (2-3 short "
                "paragraphs plus a chapter list without timestamps)\n- seo.tags: 8-15 tags; seo.keywords: "
                "search phrases\n- seo.hook: how to sharpen the first 15 seconds; "
                "seo.retention_suggestions: 3-5 concrete ideas\n"
                "- thumbnail_brief for the selected title: concept, main_subject, composition, "
                "color_palette, emotion, optional overlay_text (at most 3 short words, or empty), "
                f"style_requirements (must fit the channel style: {art_style}) and a list of things to "
                "avoid (always include copyrighted characters, real logos and misleading claims).")
        result = self.ask("seo.package", SEOOutput, self.system, user,
                          context={"title": script.title, "topic": topic})
        result.seo.disclaimer = DISCLAIMER
        if result.seo.selected_title not in result.seo.titles:
            result.seo.titles.insert(0, result.seo.selected_title)
        return result

    def review_thumbnail(self, image: Path, brief: ThumbnailBrief, title: str, asset_id: str) -> EvaluationResult:
        user = (f"Video title: {title}\nYour thumbnail brief:\n{brief.model_dump_json(indent=1)}\n\n"
                "Look at the attached thumbnail candidate. Does the artwork follow the brief (concept, "
                "main subject, composition, colours, emotion), represent the video honestly and stay "
                "readable at small size?\n"
                "About the title text: it is drawn by software, once, in the dark banner along the bottom "
                "edge. Its wording, position and style are fixed and are not something the artwork can "
                "fix, so do not ask to move, restyle, enlarge or add text. Only report a text problem if "
                "the banner text is missing or misspelled, or if the artwork ABOVE the banner contains "
                "other text or letters (that is an artwork error).\n"
                "score 0.0-1.0; approved=true only without serious problems. List issues and fixes that "
                "an image generator can act on.")
        verdict = self.ask("seo.review_thumbnail", Verdict, self.system, user, images=[image],
                           model=self.ctx.settings.vision_model)
        return self.evaluation(verdict, target_id=asset_id, target_kind="thumbnail",
                               threshold=self.ctx.settings.thumbnail_threshold, evaluator="seo_thumbnail_review")
