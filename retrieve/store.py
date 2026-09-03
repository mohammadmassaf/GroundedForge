"""
Embeds all chunks from chunks/<corpus>.json and stores them in ChromaDB.
Run once per corpus (or re-run to rebuild).

ChromaDB persists to the directory `retrieve.paths.chroma_dir()` picks for the
corpus: chroma_demo/ (committed, public RFCs) for `demo`, chroma_db/
(gitignored) for everything else. Each corpus gets its own collection so
corpora never mix.
"""
import json
from pathlib import Path


from retrieve.model_pins import EMBEDDING_MODEL , EMBEDDING_KWARGS , EMBEDDING_REVISION
from retrieve.paths import chroma_dir

# langchain_huggingface is imported inside build(), not here. Importing it pulls
# in torch (~8s, and ~200MB resident), and ensure_store() -- the path the Space
# runs on every boot -- needs neither. A module-level import would make the
# "no model loaded" claim below false before the function was even called.




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
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

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
    persist_directory = chroma_dir(corpus)
    Chroma.from_documents(documents  = dcmts, embedding = embeddings , ids =[ c["chunk_id"] for c in chunks] ,
                        collection_name=corpus, persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"})
    print(f"Stored {len(chunks)} embeddings in ChromaDB collection '{corpus}' at {persist_directory}/ (via LangChain)")


def ensure_store(corpus: str = "demo") -> bool:
    """
    Make sure `corpus` has a queryable Chroma store, rebuilding it from the
    committed vector pack if it is missing. Returns True if it built one.

    This is what the Gradio Space calls on startup. The Space has no build
    step -- HF installs requirements.txt and runs app.py -- so anything not in
    the repo has to be reconstructed at boot, and a cold boot that re-embeds
    526 chunks takes ~34s against A4's under-three-seconds bar.

    The pack skips the expensive two thirds of that. Embedding is 15.3s of the
    34 and is already done; what is left is Chroma writing its index, ~2s, with
    NO model loaded at all. That last part matters beyond speed: rebuilding the
    index must not depend on downloading 90MB of weights, or a cold Space with
    a slow hub is a blank page.

    Idempotent by design ("running it twice changes nothing"): it returns early
    when the collection already has rows, so app.py can call it unconditionally
    and a warm container pays nothing.
    """
    import chromadb
    import numpy as np

    from retrieve.paths import vector_pack   # raises for a private corpus

    persist_directory = chroma_dir(corpus)
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        existing = client.get_collection(corpus)
        if existing.count() > 0:
            return False
    except Exception:
        pass   # no collection yet -- that is what we are here to fix

    pack_path = Path(vector_pack(corpus))
    if not pack_path.exists():
        raise SystemExit(
            f"No vector pack at {pack_path} and no store at {persist_directory}/ - "
            f"run 'python scripts/make_vector_pack.py {corpus}'"
        )

    chunks = json.loads(Path(f"chunks/{corpus}.json").read_text(encoding="utf-8"))
    pack = np.load(pack_path, allow_pickle=False)
    ids, vectors = [str(i) for i in pack["ids"]], pack["vectors"]

    # The pack carries the model revision it was produced with. Query vectors
    # come from whatever model_pins says TODAY, and mixing revisions is silent:
    # every score would be wrong and nothing would raise.
    packed_revision = str(pack["revision"])
    if packed_revision != EMBEDDING_REVISION:
        raise SystemExit(
            f"{pack_path} was built with revision {packed_revision[:7]}, but "
            f"model_pins says {EMBEDDING_REVISION[:7]} - rebuild the pack"
        )

    by_id = {c["chunk_id"]: c for c in chunks}
    if set(ids) != set(by_id):
        raise SystemExit(f"{pack_path} does not match chunks/{corpus}.json - rebuild the pack")

    collection = client.get_or_create_collection(
        corpus, metadata={"hnsw:space": "cosine"})   # must match build()'s space
    collection.add(
        ids=ids,
        embeddings=[v.tolist() for v in vectors],
        documents=[by_id[i]["text"] for i in ids],
        metadatas=[_chunk_metadata(by_id[i]) for i in ids],
    )
    print(f"Rebuilt '{corpus}' store from {pack_path} ({len(ids)} chunks, no model loaded)")
    return True
