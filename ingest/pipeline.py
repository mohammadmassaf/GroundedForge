"""
Ingest pipeline: read corpus.yaml -> run each source through its adapter ->
save the combined chunks to chunks/<corpus>.json.

The per-source logic lives in ingest/adapters/*; this just orchestrates.
"""
import json
from pathlib import Path

from ingest.corpus_config import load_corpus
from ingest.adapters import load_source


def run(corpus: str = "default") -> list[dict]:
    sources = load_corpus(corpus)

    all_chunks = []
    for src in sources:
        where = src.get("path") or src.get("paths")
        print(f"  [{src['type']}] {where}")
        chunks = load_source(src)
        print(f"    -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_dir = Path("chunks")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{corpus}.json"
    out_path.write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nSaved {len(all_chunks)} chunks -> {out_path}")
    return all_chunks
