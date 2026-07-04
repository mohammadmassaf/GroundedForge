"""
Renders a validated Quiz to markdown with inline citations.

Each citation shows the source file + page and a short quote of the
cited chunk, so a reader can verify the claim without opening the code.
"""
from generate.schema import Quiz

QUOTE_LEN = 150


def render(quiz: Quiz, chunks: list[dict], topic: str) -> str:
    by_id = {c["chunk_id"]: c for c in chunks}

    lines = [f"# Quiz — {topic}", ""]
    for i, item in enumerate(quiz.items, 1):
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
    return "\n".join(lines)
