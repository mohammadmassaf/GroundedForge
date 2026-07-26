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

    # bullets read as a CV: claim + numbered refs, evidence listed below.
    # Refs (not bare text) keep each claim traceable to ITS evidence; numbers
    # (not raw chunk_ids) keep the line readable.
    order: list[str] = []
    for bullet in bullets:
        for cid in bullet.citations:
            if cid not in order:
                order.append(cid)
    ref_no = {cid: i + 1 for i, cid in enumerate(order)}

    for bullet in bullets:
        refs = "".join(f"[{ref_no[cid]}]" for cid in bullet.citations)
        lines.append(f"- {bullet.text} {refs}")
    lines.append("")

    if order:
        lines.append("### Sources")
        lines.append("")
        for cid in order:
            chunk = by_id[cid]
            quote = chunk["text"][:QUOTE_LEN].replace("\n", " ").strip()
            # git chunks use the same string for both; don't print it twice
            label = (f"`{cid}`" if chunk["source_file"] == cid
                     else f"`{cid}` — **{chunk['source_file']}**")
            lines.append(f"{ref_no[cid]}. {label} — “{quote}…”")
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