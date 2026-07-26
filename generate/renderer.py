"""
Renders a validated Quiz to markdown with inline citations.

Each citation shows the source file + page and a short quote of the
cited chunk, so a reader can verify the claim without opening the code.
"""
from generate.schema import QuizItem ,GuideSection,GuideClaim,CVBullet

QUOTE_LEN = 150


def render(items: list[QuizItem], chunks: list[dict], topic: str,
           struck: list[tuple[QuizItem, str]] | None = None) -> str:
    by_id = {c["chunk_id"]: c for c in chunks}

    lines = [f"# Quiz — {topic}", ""]
    for i, item in enumerate(items, 1):
        lines.append(f"### Q{i}. {item.question}")
        lines.append("")
        lines.append(f"**Answer:** {item.answer}")
        lines.append("")
        for cid in item.citations:
            chunk = by_id[cid]
            quote = chunk["text"][:QUOTE_LEN].replace("\n", " ").strip()
            lines.append(
                f"> 📖 `{cid}` — {chunk['source_file']}, p.{chunk['page']}: "
                f"“{quote}…”"
            )
        lines.append("")

    if struck:
        lines.append("---")
        lines.append("")
        lines.append("## ⚠️ Struck by the Critic (not supported by sources)")
        lines.append("")
        for item, reason in struck:
            lines.append(f"- ~~{item.question}~~ — **{item.answer}**")
            lines.append(f"  - *Struck because:* {reason}")
        lines.append("")

    return "\n".join(lines)


def render_bullets(bullets: list[CVBullet], chunks: list[dict], topic: str,
                   struck: list[tuple[CVBullet, str]] | None = None,
                   gap: str | None = None) -> str:
    by_id = {c["chunk_id"]: c for c in chunks}
    lines = [f"# CV Bullets — {topic}", ""]

    if gap:                       # honest gap reporting, not filler
        lines.append(f"> ⚠️ {gap}")
        lines.append("")
        return "\n".join(lines)

    # bullets read as a CV: claim + compact citation tags, evidence below
    used: list[str] = []
    for bullet in bullets:
        tags = " ".join(f"`[{cid}]`" for cid in bullet.citations)
        lines.append(f"- {bullet.text} {tags}")
        for cid in bullet.citations:
            if cid not in used:
                used.append(cid)
    lines.append("")

    if used:
        lines.append("### Sources")
        lines.append("")
        for cid in used:
            chunk = by_id[cid]
            quote = chunk["text"][:QUOTE_LEN].replace("\n", " ").strip()
            lines.append(f"- `[{cid}]` **{chunk['source_file']}** — “{quote}…”")
        lines.append("")

    if struck:
        lines.append("---")
        lines.append("")
        lines.append("## ⚠️ Struck by the Critic (not supported by evidence)")
        lines.append("")
        for bullet, reason in struck:
            lines.append(f"- ~~{bullet.text}~~")
            lines.append(f"  - *Struck because:* {reason}")
        lines.append("")

    return "\n".join(lines)


def render_guide(sections: list[GuideSection], chunks: list[dict], topic: str,
                  struck: list[tuple[GuideClaim, str]] | None = None) -> str:
    by_id = {c["chunk_id"]: c for c in chunks}
    lines = [f"# Study Guide — {topic}", ""]

    for section in sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        for claim in section.claims:
            lines.append(f"- {claim.text}")
            lines.append("")
            for cid in claim.citations:
                chunk = by_id[cid]
                quote = chunk["text"][:QUOTE_LEN].replace("\n", " ").strip()
                lines.append(
                    f"> 📖 `{cid}` — {chunk['source_file']}, p.{chunk['page']}: "
                    f"“{quote}…”"
                )

    if struck:
        lines.append("---")
        lines.append("")
        lines.append("## ⚠️ Struck by the Critic (not supported by sources)")
        lines.append("")
        for item, reason in struck:
            lines.append(f"- ~~{item.text}~~")
            lines.append(f"  - *Struck because:* {reason}")
        lines.append("")
    return "\n".join(lines)