"""
Retrieves the top-k most relevant chunks for a query string.

Returns a list of result dicts, ordered by relevance (best first):
{
    "chunk_id":    "I2208-Part-1_p7_c0",
    "source_file": "I2208-2024-2025-Part-1.pdf",
    "page":        7,
    "score":       0.87,   # cosine similarity, higher = more relevant
    "text":        "..."
}
"""
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path="chroma_db")
    return _client


def search(query_text: str, corpus: str = "default", k: int = 5) -> list[dict]:
    """
    TODO: implement this function.

    Steps:
    1. Embed `query_text` using _get_model().encode(...)
       - Pass convert_to_list=True so you get a plain Python list, not a tensor.
       - encode() normally takes a list of strings; for a single string wrap it: [query_text]
         and take index [0] of the result.

    2. Get the ChromaDB collection for this corpus:
           client = _get_client()
           collection = client.get_collection(name=corpus)

    3. Query the collection for the top-k nearest vectors:
           results = collection.query(
               query_embeddings=[query_embedding],
               n_results=k,
               include=["documents", "metadatas", "distances"],
           )
       ChromaDB returns distances (lower = closer). Convert to similarity score with:
           score = 1 - distance

    4. Build and return a list of result dicts (see module docstring for the shape).
       The results lists are nested: results["ids"][0], results["documents"][0], etc.
       Iterate with zip() over ids, documents, metadatas, distances.

    Raise a clear error if the collection doesn't exist yet
    (hint: client.get_collection raises chromadb.errors.InvalidCollectionException).
    """
    query_embedding = _get_model().encode([query_text])[0].tolist()
    client = _get_client()
    try:
        collection = client.get_collection(name=corpus)
    except Exception:
        raise SystemExit(f"No collection '{corpus}' found — run 'ingest' then 'build-index' first")
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    final = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        final.append({
            "chunk_id":    chunk_id,
            "source_file": meta["source_file"],
            "page":        meta["page"],
            "score":       1 - dist,
            "text":        doc,
        })
    return final