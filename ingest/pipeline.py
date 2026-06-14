"""
Ingest pipeline: load all files in data/ → chunk → save as JSON.

Output: chunks/<corpus>.json
Each entry is a chunk dict from chunker.chunk_pages().
"""
import json
import os
from pathlib import Path

from ingest.loader import load_file
from ingest.chunker import chunk_pages

SUPPORTED = {".pdf", ".txt", ".md"}


def run(data_dir: str = "data", corpus: str = "default") -> list[dict]:
    data_path = Path(data_dir)
    files = [f for f in data_path.iterdir() if f.suffix.lower() in SUPPORTED]

    if not files:
        raise SystemExit(f"No supported files found in {data_dir}/")

    all_chunks = []
    for f in sorted(files):
        print(f"  loading {f.name}...")
        pages = load_file(str(f))
        chunks = chunk_pages(pages, source_file=f.name)
        print(f"    -> {len(pages)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_dir = Path("chunks")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{corpus}.json"
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")

    print(f"\nSaved {len(all_chunks)} chunks -> {out_path}")
    return all_chunks
