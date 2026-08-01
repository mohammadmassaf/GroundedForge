"""
Hybrid retrieval - the M7 pipeline assembled from your three pieces:

    query --> vector top-N  --\
                               +-- rrf_merge --> N candidates --> (optional) cross-encoder --> top-k
    query --> BM25   top-N  --/

Every stage fetches MORE than it keeps (N=20 in, k out) because later
stages can only promote what earlier stages passed along - narrow too
early and the right chunk is gone before the smarter judge sees it.
"""
from retrieve.keyword import search_bm25
from retrieve.fusion import rrf_merge

# retrieve.query (langchain_chroma + HuggingFaceEmbeddings) and retrieve.rerank
# (sentence_transformers + torch) are imported INSIDE hybrid_search, not here.
# Importing a module executes its imports, so a module-level import would make
# ANY file that merely imports this one load torch -- that cost the unit tests
# ~3 minutes to run ten dict comparisons. Same deferred-import pattern main.py
# uses per CLI command; sys.modules caches the module, so the repeat cost is nil.

FETCH_N = 20   # candidates fetched per retriever and kept through the merge


def hybrid_search(query_text: str, corpus: str = "default", k: int = 5,
                  use_rerank: bool = False, where: dict | None = None) -> list[dict]:
    """
    TODO(you): chain the pipeline. No new tools - you wrote all four
    functions this calls.

    Intent:
    1. Fetch FETCH_N results from vector search and FETCH_N from BM25.
    2. Fuse them with RRF, keeping FETCH_N candidates (not k - the wide
       net stays wide until the last stage).
    3. If use_rerank: hand the candidates to the cross-encoder, keep k.
       Otherwise: keep the first k of the fused list.
    4. Return the final list.
    """
    from retrieve.query import search

    # both retrievers get the same filter — Chroma applies it in the engine,
    # BM25 in Python (see keyword._matches)
    vector = search(query_text,corpus,FETCH_N,where=where)
    bm = search_bm25(query_text,corpus,FETCH_N,where=where)
    fused  = rrf_merge(vector,bm,FETCH_N)
    if use_rerank:
        from retrieve.rerank import rerank
        result = rerank(query_text,fused,k)
    else:
        result = fused[:k]
    return result
        
