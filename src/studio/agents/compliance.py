"""Compliance Agent: a per-production checklist mixing deterministic checks and LLM screening.

These are screening aids, not legal guarantees. A failed mandatory item blocks approval;
anything that cannot be automated is marked as needing human review.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from studio.agents.base import Agent
from studio.utils import new_id
from studio.schemas import (ComplianceItem, ComplianceReport, MediaAsset, ResearchBrief, Script, SEOPackage,
                            ThumbnailBrief)

SCRIPT_CHECKS = {
    "originality": "Is the script original, not copying existing shows, creators, characters or brands?",
    "impersonation": "Does it avoid impersonating or putting words in the mouth of real people?",
    "privacy": "Does it avoid personal data or allegations about private individuals?",
    "misleading_claims": "Is it free of false, dangerous or misleading factual claims?",
}
METADATA_CHECKS = {
    "metadata_misleading_claims": "Do the title, description and thumbnail concept honestly describe the video?",
    "metadata_rights": "Do they avoid trademarks, copyrighted characters and real people's names?",
}


class Finding(BaseModel):
    check: str
    status: Literal["pass", "fail", "needs_human_review"]
    detail: str = ""


class Screening(BaseModel):
    findings: list[Finding]


class Compliance(Agent):
    name = "compliance"
    system = ("You are the Compliance Agent. You screen content for originality, impersonation, "
              "privacy and misleading-claim risks. Use 'fail' for a clear problem, "
              "'needs_human_review' when unsure, 'pass' otherwise. Always explain in `detail`.")

    def _screen(self, action: str, checks: dict[str, str], material: str) -> list[ComplianceItem]:
        questions = "\n".join(f"- {name}: {question}" for name, question in checks.items())
        user = f"Screen this material.\n<data>\n{material}\n</data>\n\nReturn one finding per check:\n{questions}"
        found = {f.check: f for f in self.ask(action, Screening, self.system, user,
                                              context={"checks": list(checks)}).findings}
        return [ComplianceItem(check=name, mandatory=True, status=found[name].status, detail=found[name].detail)
                if name in found else
                ComplianceItem(check=name, mandatory=True, status="needs_human_review",
                               detail="The automated screening returned no result for this check.")
                for name in checks]

    @staticmethod
    def _finish(report: ComplianceReport) -> ComplianceReport:
        report.blocked = any(i.mandatory and i.status == "fail" for i in report.items)
        report.human_review_required = [f"{i.check}: {i.detail}" for i in report.items
                                        if i.status == "needs_human_review"]
        report.status = "blocked" if report.blocked else "screened"
        return report

    def review(self, script: Script, research: ResearchBrief, assets: list[MediaAsset]) -> ComplianceReport:
        s = self.ctx.settings
        items = []
        missing = [a.id for a in assets if not a.provenance.strip()]
        items.append(ComplianceItem(
            check="asset_provenance", mandatory=True, status="fail" if missing else "pass",
            detail=f"{len(missing)} asset(s) have no provenance record" if missing else
            f"all {len(assets)} assets record how they were produced"))
        music = [a for a in assets if a.kind == "music"]
        items.append(ComplianceItem(
            check="music_license", mandatory=True,
            status="fail" if any(not a.license.strip() for a in music) else "pass",
            detail="; ".join(f"{a.path}: {a.license or 'NO LICENSE RECORDED'}" for a in music)
            or "no background music used"))
        if music:
            items.append(ComplianceItem(
                check="music_license_verification", mandatory=False, status="needs_human_review",
                detail="The licence is the one recorded in the music library manifest (tracks.json) or "
                       "supplied by the user; the software has not verified it."))
        items.append(ComplianceItem(
            check="voice_cloning", mandatory=True, status="pass",
            detail=f"Narration uses the stock voice '{s.tts_voice}'; no reference audio is ever sent."))
        samples = [a for a in assets if a.is_sample]
        if samples:
            items.append(ComplianceItem(
                check="placeholder_media", mandatory=False, status="needs_human_review",
                detail=f"{len(samples)} asset(s) are local placeholders (dry run or disabled feature); "
                       "this output is a demonstration, not a publishable video."))
        unverified = [c for c in research.claims if c.support != "verified_source"]
        items.append(ComplianceItem(
            check="factual_claims", mandatory=False,
            status="needs_human_review" if unverified or not research.claims else "pass",
            detail=f"{len(unverified)} of {len(research.claims)} research claims are not backed by a "
                   "supplied source and need human fact-checking."))
        items.append(ComplianceItem(
            check="synthetic_media_disclosure", mandatory=False, status="needs_human_review",
            detail="Visuals and narration are synthetic. Apply the platform's altered/synthetic content "
                   "disclosure when publishing, if its policy requires it."))
        text = "\n\n".join(f"[{c.title}]\n{c.narration}" for c in script.chapters)
        items += self._screen("compliance.screen", SCRIPT_CHECKS, text)
        report = ComplianceReport(
            project_id=self.ctx.project_id, parent_id=script.id, items=items, blocked=False,
            synthetic_media_disclosure="This video contains AI-generated images and a synthetic voice.")
        return self._finish(report)

    def review_metadata(self, report: ComplianceReport, seo: SEOPackage, brief: ThumbnailBrief) -> ComplianceReport:
        material = (f"Titles: {seo.titles}\nSelected title: {seo.selected_title}\nDescription: "
                    f"{seo.description}\nTags: {seo.tags}\nThumbnail concept: {brief.concept}; subject: "
                    f"{brief.main_subject}; overlay text: {brief.overlay_text}")
        kept = [i for i in report.items if i.check not in METADATA_CHECKS]
        updated = report.model_copy(update={
            "id": new_id(), "items": kept + self._screen("compliance.metadata", METADATA_CHECKS, material)})
        return self._finish(updated)
