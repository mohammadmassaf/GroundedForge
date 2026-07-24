"""
Shared markdown heading-splitter, used by docs_adapter and vault_adapter.

split_sections(text) walks the text and cuts it into heading sections: a
heading line (`#`..`######`) plus every line under it up to the next heading.
Content before the first heading — or a file with no headings at all — becomes
one "(intro)" section. This is exactly the logic worked out in docs_adapter;
it's factored out here so vault (and anything else markdown) can reuse it.

Each returned section is a dict:
    {
        "title":      "Setup",        # heading text, or "(intro)" if none
        "text":       "## Setup\n...", # heading + body, ready to embed
        "char_start": 120,             # offset of the section within `text`
        "char_end":   740,
    }

The caller turns each section into a chunk with its own source_file / chunk_id /
source_type (those differ per source, the splitting doesn't).
"""


def _is_heading(line: str) -> bool:
    return line.lstrip().startswith("#")


def _section(lines: list[str], start: int, end: int) -> dict:
    first = lines[0].lstrip()
    title = first.lstrip("#").strip() if first.startswith("#") else "(intro)"
    return {
        "title": title,
        "text": "\n".join(lines),
        "char_start": start,
        "char_end": end,
    }


def split_sections(text: str) -> list[dict]:
    """Split markdown `text` into heading sections (see module docstring)."""
    lines = text.split("\n")
    sections: list[dict] = []
    current: list[str] = []
    pos = 0
    sec_start = 0

    for line in lines:
        if _is_heading(line):
            if current:                       # close the open section
                sections.append(_section(current, sec_start, pos))
            current = [line]                  # this heading starts the next one
            sec_start = pos
        else:
            current.append(line)
        pos += len(line) + 1                  # +1 for the "\n" split() removed

    if current:                               # tail: the last open section
        sections.append(_section(current, sec_start, pos))

    # drop sections with no real content (e.g. a blank line between a note's
    # frontmatter and its first heading) — an empty chunk is retrieval noise
    return [s for s in sections if s["text"].strip()]
