"""
Embeds all chunks from chunks/<corpus>.json and stores them in ChromaDB.
Run once per corpus (or re-run to rebuild).

ChromaDB persists to chroma_db/ (gitignored).
Each corpus gets its own collection so corpora never mix.
"""
import json
from pathlib import Path


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality, ~90MB download on first use


def build(corpus: str = "default") -> None:
    chunks_path = Path(f"chunks/{corpus}.json")
    if not chunks_path.exists():
        raise SystemExit(f"No chunks found at {chunks_path} — run 'ingest' first")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading embedding model ({MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(model_name = MODEL_NAME)
    dcmts = []
    for chunk in chunks :
        dcmts.append(Document(page_content=chunk["text"], metadata={
            "source_file":chunk["source_file"],
            "page" :chunk["page"],
            "char_start" : chunk["char_start"],
            "char_end" : chunk["char_end"],
            "chunk_id" : chunk["chunk_id"]
        }))
    Chroma.from_documents(documents  = dcmts, embedding = embeddings , ids =[ c["chunk_id"] for c in chunks] ,
                        collection_name=corpus, persist_directory="chroma_db", collection_metadata={"hnsw:space": "cosine"})
    print(f"Stored {len(chunks)} embeddings in ChromaDB collection '{corpus}' (via LangChain)")
