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
    TODO(you): fuse two ranked result lists into one, return the top k.

    Both inputs are the usual result dicts (chunk_id, source_file, page,
    score, text), best first. The same chunk_id can appear in both lists.

    Intent:
    1. Walk each list, accumulating every chunk's RRF contribution into
       one shared tally keyed by chunk_id. (Accumulator pattern - same as
       hits/total_kept, but keyed. enumerate(list, start=1) gives ranks.)
    2. Keep one representative result dict per chunk_id so you can return
       real results, not just ids.
    3. Sort chunk_ids by fused score, best first; take the top k.
    4. Return their result dicts, with "score" replaced by the fused RRF
       score (so downstream printing still works).
    """
    points = {}
    rep ={}
    for rank , r in enumerate(list_a , start = 1):
        cid = r["chunk_id"]
        points[cid]= points.get(cid , 0)
        points[cid] += 1 / (RRF_K + rank)
        rep[cid] = r
    for rank , r in enumerate(list_b , start = 1):
        cid = r["chunk_id"]
        points[cid]= points.get(cid , 0)
        points[cid] += 1 / (RRF_K + rank)
        rep[cid] = r 
    sorted_points = sorted(points , key = lambda i:  points[i] , reverse=True)[:k]
    results = []
    for cid in sorted_points:
        results.append({**rep[cid] , "score":points[cid]})
    return results
    
    
    
