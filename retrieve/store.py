"""
Embeds all chunks from chunks/<corpus>.json and stores them in ChromaDB.
Run once per corpus (or re-run to rebuild).

ChromaDB persists to chroma_db/ (gitignored).
Each corpus gets its own collection so corpora never mix.
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality, ~90MB download on first use


def build(corpus: str = "default") -> None:
    chunks_path = Path(f"chunks/{corpus}.json")
    if not chunks_path.exists():
        raise SystemExit(f"No chunks found at {chunks_path} — run 'ingest' first")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks (this may take a minute)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name=corpus,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source_file": c["source_file"],
            "page":        c["page"],
            "char_start":  c["char_start"],
            "char_end":    c["char_end"],
        } for c in chunks],
    )

    print(f"Stored {len(chunks)} embeddings in ChromaDB collection '{corpus}'")
