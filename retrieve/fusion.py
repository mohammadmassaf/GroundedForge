"""
Reciprocal Rank Fusion - merge two ranked result lists without comparing
their scores.

Why not average scores: cosine similarity lives in 0..1, BM25 is unbounded.
Different currencies. RRF ignores scores and uses POSITIONS: a chunk's
fused score is the sum, over every list it appears in, of

    1 / (RRF_K + rank)        # rank starts at 1 for the best result

RRF_K = 60 (standard damping constant; flattens the difference between
rank 1 and rank 3 so one list can't dominate). A chunk ranked well in
BOTH lists collects two contributions and beats a chunk that's rank 1 in
only one. A chunk absent from a list simply collects nothing from it.

Worked example (RRF_K=60):
  vector: [A, B, C]   ->  A: 1/61, B: 1/62, C: 1/63
  bm25:   [B, D]      ->  B: 1/61, D: 1/62
  fused scores: B ≈ 0.0325 > A ≈ 0.0164 > D ≈ 0.0161 > C ≈ 0.0159
  B wins: decent in both lists beats best-in-one.
"""

RRF_K = 60


def rrf_merge(list_a: list[dict], list_b: list[dict], k: int = 5) -> list[dict]:
    """
    Fuse two ranked result lists into one, returning the top k.

    Both inputs are the usual result dicts (chunk_id, source_file, page, score,
    text), best first, and the same chunk_id may appear in both - that overlap
    is the point, since a chunk ranked decently by both retrievers should beat
    one ranked first by only one of them.

    Each list contributes 1/(RRF_K + rank) per chunk into a shared tally keyed
    by chunk_id, so only RANKS are combined, never the raw scores: a cosine
    distance and a BM25 score are not on the same scale and adding them would
    be meaningless.

    The returned dicts carry "score" replaced by the fused RRF value, so
    anything downstream that prints or thresholds a score still works.
    """
    points = {}
    rep ={}
    for rank , r in enumerate(list_a , start = 1):
        cid = r["chunk_id"]
        points[cid]= points.get(cid , 0)
        points[cid] += 1 / (RRF_K + rank)
        rep.setdefault(cid , r)
    for rank , r in enumerate(list_b , start = 1):
        cid = r["chunk_id"]
        points[cid]= points.get(cid , 0)
        points[cid] += 1 / (RRF_K + rank)
        rep.setdefault(cid , r)
    sorted_points = sorted(points , key = lambda i:  points[i] , reverse=True)[:k]
    results = []
    for cid in sorted_points:
        results.append({**rep[cid] , "score":points[cid]})
    return results
    
    
    
