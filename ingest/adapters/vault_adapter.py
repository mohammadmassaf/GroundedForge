"""
vault adapter: turn hand-picked Obsidian notes into chunks.

Same heading-section chunking as docs (via sections.split_sections), plus ONE
new step: strip the YAML frontmatter block at the top of each note before
chunking, so that metadata block never becomes a chunk.

Source config (from corpus.yaml):
    {type: vault, paths: [<note path>, ...], repo: <short name, optional>}

`repo` matches the field the git and docs adapters set, so notes ABOUT a
project are scoped to it at retrieval time. Optional here (unlike the
siblings, where it is required) because a note need not belong to any repo.

Note: this adapter reads ONLY the paths explicitly listed — it never crawls
the vault. Flow: read -> strip_frontmatter (returns (meta, body), lifting
`type`/`status` into chunk metadata) -> split_sections -> make_chunk.
"""
from pathlib import Path
import yaml

from ingest.adapters.base import register, make_chunk
from ingest.adapters.sections import split_sections


def strip_frontmatter(text: str) -> tuple[dict, str]:
    """
    Remove a leading YAML frontmatter block, if present, and return the body.

    Obsidian notes may open with a fenced block:

        ---
        type: project
        tags: [project, ai-engineering]
        status: active
        ---
        # Real content starts here
        ...

    Return everything AFTER the closing `---` (the "# Real content..." onward).

    Rules / edge cases:
      - If the note does NOT start with a `---` line, there's no frontmatter —
        return `text` unchanged.
      - The block is delimited by the FIRST two `---` lines only. A `---` later
        in the body is a horizontal rule, not a fence — don't strip on that.
      - Be reasonable about a missing closing `---` (a malformed note): decide
        whether to treat it as "no frontmatter" rather than eating the whole file.

    (Optional, later: you could also parse a field like `status:` out of the
     block and pass it as chunk metadata. Not required now.)
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.split("\n")          # lines[0] is "---"
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":         # the closing fence — stop here
            block = "\n".join(lines[1:i])     # between the fences → YAML
            body  = "\n".join(lines[i+1:])    # after the fence → body
            return yaml.safe_load(block) or {}, body

    return {}, text                   # no closing fence found → malformed → no frontmatter
    


@register("vault")
def load_vault(source: dict) -> list[dict]:
    paths = source["paths"]
    repo = source.get("repo")

    chunks = []
    for note_path in paths:
        p = Path(note_path)
        meta, body = strip_frontmatter(p.read_text(encoding="utf-8"))
        extra = {k: meta[k] for k in ("type", "status") if meta.get(k)}
        if repo:
            extra["repo"] = repo
        for i, sec in enumerate(split_sections(body)):
            chunks.append(make_chunk(
                text=sec["text"],
                source_file=f"{p.stem} § {sec['title']}",
                chunk_id=f"vault-{p.stem}-{i}",
                source_type="vault",
                section=sec["title"],
                char_start=sec["char_start"],
                char_end=sec["char_end"],
                **extra,
            ))
    return chunks
