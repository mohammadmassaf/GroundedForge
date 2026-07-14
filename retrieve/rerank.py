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

load_dotenv()  # HF_TOKEN from .env -> authenticated model downloads

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query_text: str, candidates: list[dict], k: int = 5) -> list[dict]:
    """
    TODO(you): re-order the candidate results by cross-encoder relevance,
    return the top k.

    What you have to work with:
      - _get_model().predict(list_of_pairs) where each pair is
        (query_text, chunk_text); returns one relevance score per pair,
        in input order (score[i] belongs to candidates[i] - same
        parallel-arrays situation as BM25's get_scores).

    Intent:
    1. Build the (query, text) pair list from the candidates.
    2. Get the scores.
    3. Sort candidates by their score, best first; take top k.
    4. Return them with "score" replaced by the cross-encoder score.
    """
    pairs  = [(query_text,c["text"] )for c in candidates]
    scores = _get_model().predict(pairs)
    top = sorted(range(len(scores)), key = lambda i : scores[i], reverse = True)[:k]
    results  = []
    for i in top:
        results.append({**candidates[i] , "score" : scores[i]})

    return results
        

    
