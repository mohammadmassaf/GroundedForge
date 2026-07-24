"""
docs adapter: turn a repo's markdown docs into chunks. One heading section = one chunk.

Source config (from corpus.yaml):
    {type: docs, path: <repo path>, repo: <short name for citations>}

Reads README.md, CLAUDE.md, and docs/*.md under `path` (see _find_doc_files),
then defers the heading-section splitting to sections.split_sections (shared
with vault_adapter). Each section becomes one chunk:
    source_file = "{repo} {stem} § {heading}"   citation identity
    chunk_id    = "{repo}-{stem}-{i}"            unique per file section
    section     = heading text (or "(intro)" for pre-heading / heading-less text)
    char_start/char_end = section span within the file (provenance)
"""
from pathlib import Path

from ingest.adapters.base import register, make_chunk
from ingest.adapters.sections import split_sections


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


@register("docs")
def load_docs(source: dict) -> list[dict]:
    repo_path = source["path"]
    repo = source["repo"]
    files = _find_doc_files(repo_path)

    chunks = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        for i, sec in enumerate(split_sections(text)):
            chunks.append(make_chunk(
                text=sec["text"],
                source_file=f"{repo} {file.stem} § {sec['title']}",
                chunk_id=f"{repo}-{file.stem}-{i}",
                source_type="docs",
                section=sec["title"],
                repo=repo,
                char_start=sec["char_start"],
                char_end=sec["char_end"],
            ))
    return chunks
                    
                    
    
