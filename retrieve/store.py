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
from retrieve.model_pins import EMBEDDING_MODEL , EMBEDDING_KWARGS




def _chunk_metadata(chunk: dict) -> dict:
    """
    Build the ChromaDB metadata dict for one chunk.

    Everything on the chunk EXCEPT `text` (that becomes the document's
    page_content) should be stored as metadata, so any of it — source_type,
    repo, sha, date, section, type, status — can drive a `where=` filter later.

    Two rules ChromaDB imposes on metadata:
      - values must be scalars (str / int / float / bool) — ours already are
      - a value may NOT be None — Chroma rejects None, so a key with a None
        value (e.g. a vault chunk that has no `status`) must be DROPPED, not
        stored as None.
    """
    metadata = { k : v for k , v in chunk.items() if k != "text" and v is not  None }
    return metadata
    



def build(corpus: str = "default") -> None:
    chunks_path = Path(f"chunks/{corpus}.json")
    if not chunks_path.exists():
        raise SystemExit(f"No chunks found at {chunks_path} — run 'ingest' first")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL , model_kwargs= EMBEDDING_KWARGS)
    dcmts = []
    for chunk in chunks :
        dcmts.append(Document(page_content=chunk["text"], metadata=_chunk_metadata(chunk)))
    Chroma.from_documents(documents  = dcmts, embedding = embeddings , ids =[ c["chunk_id"] for c in chunks] ,
                        collection_name=corpus, persist_directory="chroma_db", collection_metadata={"hnsw:space": "cosine"})
    print(f"Stored {len(chunks)} embeddings in ChromaDB collection '{corpus}' (via LangChain)")
