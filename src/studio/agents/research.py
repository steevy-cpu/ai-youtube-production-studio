"""Research Agent: structured brief built from local source files and labelled background knowledge.

There is no web access. Only files supplied locally can be cited; any claim marked as
verified without a valid local source is downgraded by code, so citations cannot be invented.
"""
from __future__ import annotations

from pathlib import Path

from studio.agents.base import Agent
from studio.schemas import ProductionPlan, ResearchBrief, Source

MAX_SOURCES, MAX_CHARS = 8, 6000


def load_sources(directory: Path | str | None) -> list[tuple[Source, str]]:
    """Read .txt/.md files from a local directory (non-recursive)."""
    if not directory or not Path(directory).is_dir():
        return []
    files = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in (".txt", ".md") and p.name.lower() != "readme.md")
    sources = []
    for number, path in enumerate(files[:MAX_SOURCES], start=1):
        text = path.read_text(encoding="utf-8", errors="replace")[:MAX_CHARS]
        sources.append((Source(id=f"S{number}", title=path.stem, path=str(path)), text))
    return sources


class Research(Agent):
    name = "research"
    system = ("You are the Research Agent. You organise accurate, useful knowledge for the script "
              "writer. You cannot browse the web. Never invent a citation.")

    def brief(self, topic: str, plan: ProductionPlan, sources_dir: Path | str | None) -> ResearchBrief:
        sources = load_sources(sources_dir)
        chapters = "\n".join(f"{c.index}. {c.title} - {c.purpose}" for c in plan.chapters)
        if sources:
            material = "\n\n".join(f"<data source_id=\"{s.id}\" title=\"{s.title}\">\n{text}\n</data>"
                                   for s, text in sources)
            source_rule = ("Local sources are provided below. A claim may use support=\"verified_source\" "
                           "only when one of them states it; set source_id to that source's id.")
        else:
            material = "(no local source files were provided)"
            source_rule = ("No verified sources are available, so no claim may use "
                           "support=\"verified_source\".")
        user = (
            f"Topic: {topic}\nChapters (one research chapter per entry, same index and title):\n{chapters}\n\n"
            "For each chapter give a summary and key points the writer can dramatise (typical work, "
            "responsibilities, struggles, what changes at that level).\n"
            f"List the factual claims the script may rely on. {source_rule} Use "
            "support=\"background_knowledge\" for general knowledge you are confident about and "
            "support=\"uncertain\" for anything you are unsure of. Leave the `sources` field empty.\n\n"
            f"{material}")
        brief = self.ask("research.brief", ResearchBrief, self.system, user, context={
            "topic": topic, "chapters": [{"index": c.index, "title": c.title} for c in plan.chapters],
            "source_ids": [s.id for s, _ in sources]})

        # Enforce citation discipline in code: the model's own claims about sources are not trusted.
        valid = {s.id for s, _ in sources}
        brief.topic, brief.sources = topic, [s for s, _ in sources]
        downgraded = 0
        for claim in brief.claims:
            if claim.source_id not in valid:
                if claim.support == "verified_source":
                    claim.support, downgraded = "uncertain", downgraded + 1
                    claim.note = (claim.note + " " if claim.note else "") + \
                        "Downgraded: cited source is not among the local source files."
                claim.source_id = None
        if not sources:
            brief.notes.append("No local sources were supplied: every claim is unverified background "
                               "knowledge of the language model and needs human fact-checking.")
        if downgraded:
            brief.notes.append(f"{downgraded} claim(s) cited a non-existent source and were marked uncertain.")
        return brief
