"""
docs adapter: turn a repo's markdown docs into chunks. One heading section = one chunk.

Source config (from corpus.yaml):
    {type: docs, path: <repo path>, repo: <short name for citations>}

Which files get read: README.md, CLAUDE.md, and docs/*.md under `path`
(helper _find_doc_files does the discovery for you).

YOU implement load_docs. For each file, split its text into HEADING SECTIONS
and emit one chunk per section:

  1. A "section" = a markdown heading line (`#`, `##`, `###`, ...) plus every
     line under it up to (but not including) the next heading line.
     Example — this file:
         # Setup            ┐
         Install deps.      │ section 1  ("Setup")
         Run it.            ┘
         ## Config          ┐
         Edit the yaml.     ┘ section 2  ("Config")
     → 2 chunks.

  2. Build ONE chunk per section with make_chunk(...):
       - text        = the heading + its body (what a retriever matches on)
       - source_file = f"{repo} {stem} § {heading}"   e.g. "mealwise README § Auth"
                       (stem = filename without .md; this is the citation identity)
       - source_type = "docs"
       - chunk_id    = unique per section (see the duplicate-heading edge case)
       - metadata    = repo=..., section=<heading text>
       - page        = let it default (1)
       - char_start / char_end = the section's start/end offset WITHIN the file
         — these are meaningful again here (unlike git), so track positions as
         you split, and a citation can point at an exact span of the file.

  3. Collect all files' chunks into one list and return it.

Edge cases to decide on (this is the "chunk-unit design" skill again):
  - PREAMBLE: text before the first heading (a title line, an intro paragraph).
    Does that become its own chunk, or get dropped? Your call — just be deliberate.
  - NO HEADINGS at all in a file → one chunk for the whole file.
  - DUPLICATE headings (two `## Setup`) → your chunk_id must still be unique
    (a per-file running index is the usual fix).
  - EMPTY section (a heading immediately followed by another heading).
  - A real heading is a line that starts with 1-6 `#` then a space. A `#` inside
    a ``` code fence is NOT a heading — handling that is optional polish.

Return: list[chunk dict]. Empty list if the repo has no docs — don't crash.
"""
from pathlib import Path

from ingest.adapters.base import register, make_chunk


def _find_doc_files(repo_path: str) -> list[Path]:
    """Return the markdown files to ingest: README.md, CLAUDE.md, docs/*.md."""
    root = Path(repo_path)
    files: list[Path] = []
    for name in ("README.md", "CLAUDE.md"):
        p = root / name
        if p.exists():
            files.append(p)
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        files.extend(sorted(docs_dir.glob("*.md")))
    return files

def _section_chunk(section_lines, repo, stem, i, sec_start, sec_end):
    first = section_lines[0]
    if first.startswith("#"):
        title = first.lstrip("#").strip()     # normal heading
    else:
        title = "(intro)"                     # ← the edge-case fallback, ONE place
    return make_chunk(
        text="\n".join(section_lines),
        source_file=f"{repo} {stem} § {title}",
        chunk_id=f"{repo}-{stem}-{i}",
        source_type="docs",
        section=title,
        repo=repo,
        char_start=sec_start, char_end=sec_end,
    )


@register("docs")
def load_docs(source: dict) -> list[dict]:
    repo_path = source["path"]
    repo = source["repo"]
    files = _find_doc_files(repo_path)

    chunks =[]
    for file in files:
        text = file.read_text(encoding = "utf-8")
        lines = text.split("\n")
        section = []
        pos = 0
        sec_start = 0 
        i = 0
        for line in lines:
            if line.startswith("#") or line.lstrip().startswith("#"):
                if section:
                    chunks.append(_section_chunk(section, repo, file.stem, i, sec_start, pos))
                section = [line]
                sec_start = pos
                i += 1
            else:
                section.append(line)
            pos += len(line) + 1

      
        if section:
            chunks.append(_section_chunk(section, repo, file.stem, i, sec_start, pos))
    return chunks
                    
                    
    
