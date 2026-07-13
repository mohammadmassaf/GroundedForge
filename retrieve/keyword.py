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
    return text.lower().split()


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


def search_bm25(query_text: str, corpus: str = "default", k: int = 5) -> list[dict]:
    """
    TODO(you): top-k chunks by BM25 score, same result shape as
    retrieve.query.search() so the two lists are interchangeable.

    What you have to work with:
      - index, chunks = _get_index(corpus)
      - index.get_scores(tokenized_query) returns one score per chunk,
        in the SAME ORDER as the chunks list (score[i] belongs to chunks[i])
      - you need the indices of the k highest scores, best first

    Steps in intent:
    1. Tokenize the query the same way the corpus was tokenized.
    2. Get the score array.
    3. Find the top-k chunk positions by score. (Hint: sorted() with a
       key, over range(len(scores)) or enumerate - or look up argsort.)
    4. Build the usual result dicts (chunk_id, source_file, page, score,
       text) from those chunks, best first, and return them.
    """
    query_tokens = _tokenize(query_text)
    index , chunks = _get_index(corpus)
    scores = index.get_scores(query_tokens)
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for i in top_k:
        results.append({**chunks[i] , "score" : scores[i]})
    return results
