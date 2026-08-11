"""
Unit tests for reciprocal rank fusion (retrieve/fusion.py).

Pure arithmetic over two ranked lists -- no index, no model.

Beyond the ranking maths, one test here guards a subtler contract: `rrf_merge`
must not lose keys that only one of the two input lists carries. `vector_score`
is the case that matters. The gap check in run_bullets_loop reads it, and it
only exists on results from retrieve.query.search -- BM25 results have no such
key. Merge the two carelessly and it disappears for exactly the best chunks,
the ones both retrievers found.
"""
from retrieve.fusion import rrf_merge, RRF_K


def _r(cid, **extra):
    """A result dict with just enough shape to be merged."""
    return {"chunk_id": cid, "source_file": "f.md", "page": 1,
            "text": f"text of {cid}", "score": 0.5, **extra}


# --- the ranking maths ------------------------------------------------------

def test_decent_in_both_beats_best_in_one():
    """The module docstring's own worked example. B is rank 2 in the vector
    list and rank 1 in BM25, so it collects two contributions and beats A,
    which is rank 1 in one list only. That property -- agreement between
    retrievers outranks a single strong opinion -- is the whole reason to
    fuse rather than concatenate."""
    vector = [_r("A"), _r("B"), _r("C")]
    bm25 = [_r("B"), _r("D")]

    fused = rrf_merge(vector, bm25, k=4)

    assert [r["chunk_id"] for r in fused] == ["B", "A", "D", "C"]


def test_score_is_replaced_by_the_fused_score():
    """Downstream prints and sorts on `score`, so it has to carry the RRF
    value rather than whichever input score happened to survive."""
    fused = rrf_merge([_r("A", score=0.99)], [], k=1)

    assert fused[0]["score"] == 1 / (RRF_K + 1)


def test_k_limits_the_output():
    """The candidate net stays wide through the merge and narrows here."""
    vector = [_r("A"), _r("B"), _r("C")]

    assert len(rrf_merge(vector, [], k=2)) == 2


def test_a_chunk_in_both_lists_appears_once():
    """Fusion deduplicates on chunk_id -- the same chunk from two retrievers
    is one piece of evidence, not two."""
    fused = rrf_merge([_r("A")], [_r("A")], k=5)

    assert [r["chunk_id"] for r in fused] == ["A"]


# --- key preservation -------------------------------------------------------

def test_vector_score_survives_when_bm25_also_found_the_chunk():
    """The trap. BM25 results carry no vector_score, so if the BM25 dict is
    allowed to replace the vector one as the representative, the key vanishes
    for every chunk both retrievers agreed on -- and run_bullets_loop then
    reads 0 and declares a gap on the strongest possible evidence."""
    vector = [_r("A", vector_score=0.62)]
    bm25 = [_r("A")]                     # same chunk, no vector_score

    fused = rrf_merge(vector, bm25, k=5)

    assert fused[0]["vector_score"] == 0.62


def test_bm25_only_chunk_simply_has_no_vector_score():
    """A chunk the dense retriever never surfaced has no dense confidence to
    report. Callers read it with .get(..., 0), so absence is the honest
    value -- inventing one would fake evidence the embedder never found."""
    fused = rrf_merge([], [_r("Z")], k=5)

    assert "vector_score" not in fused[0]
