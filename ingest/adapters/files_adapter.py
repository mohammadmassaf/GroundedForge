"""
files adapter: wrap the v1 PDF/txt/md loader + character chunker as an adapter.

This is what keeps study mode working through the same corpus.yaml mechanism:
the 'default' corpus is just {type: files, path: data}. No new chunking logic —
it reuses loader.load_file + chunker.chunk_pages and tags each chunk with
source_type="files".

Source config (from corpus.yaml):
    {type: files, path: <dir>}
"""
from pathlib import Path

from ingest.adapters.base import register
from ingest.loader import load_file
from ingest.chunker import chunk_pages

SUPPORTED = {".pdf", ".txt", ".md"}


@register("files")
def load_files(source: dict) -> list[dict]:
    data_dir = Path(source["path"])
    if not data_dir.is_dir():
        return []

    files = [f for f in data_dir.iterdir() if f.suffix.lower() in SUPPORTED]
    chunks = []
    for f in sorted(files):
        pages = load_file(str(f))
        for chunk in chunk_pages(pages, source_file=f.name):
            chunk["source_type"] = "files"     # the v1 chunker predates this key
            chunks.append(chunk)
    return chunks
