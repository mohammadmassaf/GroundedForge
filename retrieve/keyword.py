"""
BM25 keyword retrieval - the exact-token complement to vector search.

BM25 scores a chunk by: how often the query's tokens appear in it (term
frequency), how rare each token is across the corpus (IDF), normalized by
chunk length. No semantics: "car" and "automobile" are strangers, but
"CSMA/CD" matches "CSMA/CD" exactly - which is precisely what embeddings
fumble.

Index lives in memory, built lazily from chunks/<corpus>.json on first
query (214 chunks -> milliseconds), cached per corpus like the model
singleton in retrieve/query.py.
"""
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

_indexes: dict = {}   # corpus name -> (BM25Okapi, list of chunk dicts)


def _tokenize(text: str) -> list[str]:
    base =  text.lower().split()
    tokenized = []
    for elmt in base:
        token = elmt.strip('.,;:!?"\'()[]')
        if token:
            tokenized.append(token)
    return tokenized



def _get_index(corpus: str):
    """Lazy singleton per corpus - same pattern as _get_model()/_get_client().

    Loads chunks/<corpus>.json, tokenizes every chunk's text, builds
    BM25Okapi(list_of_token_lists), caches (index, chunks) in _indexes.
    """
    if corpus not in _indexes:
        chunks_path = Path(f"chunks/{corpus}.json")
        if not chunks_path.exists():
            raise SystemExit(f"No chunks at {chunks_path} - run 'ingest' first")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        index = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        _indexes[corpus] = (index, chunks)
    return _indexes[corpus]


def _matches(chunk: dict, where: dict | None) -> bool:
    """
    Python-side equivalent of Chroma's `where` filter (BM25 has no engine filter).
    Supports plain {key: value} conditions and Chroma's {"$and": [...]} /
    {"$or": [...]} wrappers.
    """
    if not where:
        return True
    if "$and" in where:
        return all(_matches(chunk, cond) for cond in where["$and"])
    if "$or" in where:
        return any(_matches(chunk, cond) for cond in where["$or"])
    return all(chunk.get(key) == value for key, value in where.items())


def search_bm25(query_text: str, corpus: str = "default", k: int = 5,
                where: dict | None = None) -> list[dict]:
    """
    Top-k chunks by BM25 score, in the same result shape as
    retrieve.query.search() so the two lists are interchangeable - which is
    what lets rrf_merge fuse them without knowing which retriever produced
    which.

    The query is tokenized with _tokenize, the same function the index was
    built with: query and corpus must pass through one tokenizer or "Configured"
    never matches the stored "configured" and the score silently reads zero.

    get_scores returns one score per chunk, positionally aligned with `chunks`
    (score[i] belongs to chunks[i]), so ranking is done over positions and the
    chunks are looked up afterwards.
    """
    query_tokens = _tokenize(query_text)
    index , chunks = _get_index(corpus)
    scores = index.get_scores(query_tokens)
    # filter BEFORE taking k (same reason as Chroma's where=: taking k first and
    # filtering after could return fewer than k — or zero — matching chunks)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_k = [i for i in ranked if _matches(chunks[i], where)][:k]

    results = []
    for i in top_k:
        results.append({**chunks[i] , "score" : scores[i]})
    return results
