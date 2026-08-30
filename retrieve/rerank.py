"""
Cross-encoder re-ranking - stage two of two-stage retrieval.

A bi-encoder (our vector search) embeds query and chunk SEPARATELY and
compares the two vectors: fast, pre-indexable, but no token-level
interaction. A cross-encoder feeds (query, chunk) through the transformer
TOGETHER - every query token attends to every chunk token - which is far
more accurate but cannot be pre-computed: it runs per pair, per query.

Hence the pattern: a cheap wide net fetches ~20 candidates, the expensive
judge re-scores only those. The re-ranker can only promote what the
candidates contain - its ceiling is the candidate set's recall.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, local, free; trained
on MS MARCO passage-ranking - exactly this "given a query, order passages
by relevance" task).
"""
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from retrieve.model_pins import RERANK_MODEL, RERANK_REVISION

load_dotenv()  # HF_TOKEN from .env -> authenticated model downloads



_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL , revision = RERANK_REVISION)
    return _model


def rerank(query_text: str, candidates: list[dict], k: int = 5) -> list[dict]:
    """
    Re-order candidates by cross-encoder relevance and return the top k.

    The difference from the retrievers upstream: a bi-encoder embeds the query
    and the chunk SEPARATELY and compares the two vectors, so it never sees them
    together. A cross-encoder reads (query, chunk) as one input and scores the
    pair directly - far more accurate, and far too slow to run over a whole
    corpus. Hence the two-stage shape: retrieve wide and cheap, re-rank narrow
    and expensive.

    predict() returns one score per pair in input order (score[i] belongs to
    candidates[i]), so ranking runs over positions. The returned dicts carry
    "score" replaced by the cross-encoder score.
    """
    pairs  = [(query_text,c["text"] )for c in candidates]
    scores = _get_model().predict(pairs)
    top = sorted(range(len(scores)), key = lambda i : scores[i], reverse = True)[:k]
    results  = []
    for i in top:
        results.append({**candidates[i] , "score" : float(scores[i])})

    return results
        

    
