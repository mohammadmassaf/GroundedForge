"""
Unit tests for per-section evidence gathering (retrieve/star_evidence.py).

gather_evidence is pure orchestration: it decides WHICH filter, WHICH query
and WHAT to do when a pool comes back thin. The retrieval itself (Chroma,
BM25, the cross-encoder) is somebody else's tested code and is far too slow
and non-deterministic to run here -- so hybrid_search is mocked and we test
only the decisions.

Same rule as test_loop_integration: patch the name WHERE IT IS USED.
star_evidence did `from retrieve.hybrid import hybrid_search`, so the target
is "retrieve.star_evidence.hybrid_search".

New here vs your earlier fakes: this fake is a RECORDER. Your _fake_generate
just returned a canned value; this one also appends every call it received to
a list, so a test can assert on HOW gather_evidence called it (which filter,
which query, how many times) rather than only on what came back. That's the
only way to test a function whose interesting behaviour is its calls.
"""
from retrieve.star_evidence import (
    gather_evidence,
    _where_for,
    SECTION_SOURCES,
    WIDENED_SOURCES,
    SECTION_HINTS,
    MIN_POOL,
)


def _chunk(cid, source_type="docs"):
    """A result dict shaped like query.py builds them."""
    return {"chunk_id": cid, "source_file": "f.md", "page": 1,
            "source_type": source_type, "text": f"text of {cid}", "score": -5.0}


def _recorder(results_for):
    """
    Build (fake_hybrid_search, calls).

    `results_for(where)` decides what a call returns, so a test can make the
    FILTERED call come back thin and the WIDENED one come back rich -- that's
    how you drive the fallback without touching a real index.

    `calls` collects one dict per invocation: {"query", "k", "where"}.
    """
    calls = []

    def fake(query, corpus="job", k=5, use_rerank=False, where=None):
        calls.append({"query": query, "k": k, "where": where})
        return list(results_for(where))[:k]

    return fake, calls


# --- _where_for: the Chroma filter shape ------------------------------------

def test_where_for_single_source():
    """One source needs no $or -- Chroma takes the bare equality form."""
    assert _where_for(["git"]) == {"source_type": "git"}


def test_where_for_several_sources():
    """Several sources become an explicit $or, in the order given."""
    assert _where_for(["vault", "docs"]) == {
        "$or": [{"source_type": "vault"}, {"source_type": "docs"}]
    }


def test_where_for_scopes_a_single_source_to_a_repo():
    """Two conditions need an explicit $and -- Chroma will not read a second
    key in the same dict as a conjunction."""
    assert _where_for(["git"], repo="mealwise") == {
        "$and": [{"source_type": "git"}, {"repo": "mealwise"}]
    }


def test_where_for_nests_the_or_inside_the_and():
    """(vault OR docs) AND repo -- not (vault) OR (docs AND repo), which would
    let any vault note from any project through."""
    assert _where_for(["vault", "docs"], repo="mealwise") == {
        "$and": [
            {"$or": [{"source_type": "vault"}, {"source_type": "docs"}]},
            {"repo": "mealwise"},
        ]
    }


def test_where_for_without_a_repo_is_unchanged():
    """v1 study corpora have no repo metadata at all; passing None must leave
    the filter exactly as it was before scoping existed."""
    assert _where_for(["git"], repo=None) == {"source_type": "git"}


# --- gather_evidence: the per-section decisions -----------------------------

def test_returns_a_pool_for_every_section(monkeypatch):
    """Every STAR section gets a key, even before any content assertions --
    a missing key would blow up generate_star's pool loop."""
    fake, _ = _recorder(lambda where: [_chunk("a"), _chunk("b"), _chunk("c")])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    pools = gather_evidence("any question", corpus="job", k=3)

    assert set(pools) == set(SECTION_SOURCES)


def test_each_section_uses_its_own_filter(monkeypatch):
    """The filter passed for each section must match SECTION_SOURCES --
    this is what keeps Action on git and Situation off it."""
    fake, calls = _recorder(lambda where: [_chunk("a"), _chunk("b"), _chunk("c")])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    gather_evidence("any question", corpus="job", k=3)

    used = [c["where"] for c in calls]
    expected = [_where_for(src) for src in SECTION_SOURCES.values()]
    assert used == expected


def test_query_carries_the_section_hint(monkeypatch):
    """TODO(you): every recorded query must contain BOTH the question text and
    that section's hint from SECTION_HINTS. Without the hint the four sections
    issue identical queries; without the question the hint retrieves generic
    prose (both failure modes were measured on 2026-07-31)."""
    fake, calls = _recorder(lambda where: [_chunk("a"), _chunk("b"), _chunk("c")])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    gather_evidence("any question", corpus="job", k=3)
    sections = list(SECTION_SOURCES)

    assert len(calls) == len(sections)
    for i , name in  enumerate(sections):
        assert "any question" in calls[i]["query"]
        assert SECTION_HINTS[name]  in calls[i]["query"]


def test_thin_pool_triggers_widened_retry(monkeypatch):
    """TODO(you): make the filtered call return FEWER than MIN_POOL chunks and
    assert a second call happened for that section, carrying the WIDENED
    filter (WIDENED_SOURCES) rather than the original one.
    Hint: results_for receives `where`, so you can branch on it."""
    fake , calls = _recorder(lambda where: [_chunk(f"c{i}") for i in range(MIN_POOL - 1)])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    gather_evidence("any question", corpus="job", k=3)
    sections = list(SECTION_SOURCES)
    assert len(calls) == 2 * len(sections)
    for i , name in enumerate(sections):
        assert _where_for(WIDENED_SOURCES[name])  ==  calls[2*i + 1]["where"]
        


def test_fallback_does_not_duplicate_chunks(monkeypatch):
    """TODO(you): have the widened call return a chunk the filtered call
    ALREADY returned. Assert the final pool holds no repeated chunk_id.
    A duplicate reaches the Generator as two independent pieces of evidence
    when it is one."""
    WIDE = [_where_for(v) for v in WIDENED_SOURCES.values()]

    def results_for(where):
        if where in WIDE:
            return [_chunk("dup") , _chunk("fresh")]
        return [_chunk("dup")]
    fake , calls = _recorder(results_for)
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    pools =gather_evidence("any question", corpus="job", k=3)
    for pool in pools.values():
        ids = [c["chunk_id"] for c in pool]
        assert len(ids) == len(set(ids))
        assert "fresh" in ids

def test_fallback_stops_at_k(monkeypatch):
    """TODO(you): have the widened call return far more than k fresh chunks.
    Assert the pool never exceeds k -- the loop is bounded by the pool being
    full, not by how many results came back."""
    WIDE = [_where_for(v) for v in WIDENED_SOURCES.values()]
    
    def results_for(where):
        if where in WIDE:
            return [_chunk("dup") , _chunk("fresh"),_chunk("new")]
        return [_chunk("dup")]
    
    fake , calls = _recorder(results_for)
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    k = 3
    pools =gather_evidence("any question", corpus="job", k=k)
    for pool in pools.values():
        assert len(pool) == k


def test_healthy_pool_skips_the_fallback(monkeypatch):
    """TODO(you): when the filtered call already returns >= MIN_POOL, that
    section must make exactly ONE call. Widening a pool that isn't thin would
    quietly let vault notes into a git-only section."""
    fake, calls = _recorder(lambda where: [_chunk("a"), _chunk("b"), _chunk("c")])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)
    
    gather_evidence("any question", corpus="job", k=3)

    assert len(calls) == len(SECTION_SOURCES)




def test_pool_may_end_short_when_evidence_is_thin(monkeypatch):
    """TODO(you): both calls return almost nothing -> the section's pool is
    allowed to come back under k (even empty) without raising. A short pool
    is the honest signal that drives "insufficient evidence in corpus"; the
    bug would be manufacturing chunks to fill it."""
    def results_for(where):
        return []
    fake , calls = _recorder(results_for)

    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    pools =gather_evidence("any question", corpus="job", k=3)
    assert set(pools) == set(SECTION_SOURCES)
    for pool in pools.values():
        assert pool == []


# --- repo scoping -----------------------------------------------------------
#
# The job corpus holds two projects, and one of them discusses the other in its
# commit messages. Relevance scoring alone will not keep them apart: asked "how
# is authentication implemented in MealWise?", the cross-encoder ranked a
# Grounded Forge commit first -- it held "implemented" and a quoted mealwise
# sha, and no mention of auth. Scope is a filter, not a hint.

def test_every_section_is_scoped_to_the_repo(monkeypatch):
    """A repo passed to gather_evidence must reach every section's filter, not
    just the first -- Action is the section most likely to pull a neighbouring
    project's commits."""
    fake, calls = _recorder(lambda where: [_chunk("a"), _chunk("b"), _chunk("c")])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    gather_evidence("any question", corpus="job", k=3, repo="mealwise")

    assert len(calls) == len(SECTION_SOURCES)
    for call in calls:
        assert {"repo": "mealwise"} in call["where"]["$and"]


def test_widening_relaxes_source_type_but_never_the_repo(monkeypatch):
    """The regression this whole change exists to prevent. A thin pool widens
    the SOURCE TYPE, because evidence may live somewhere unexpected -- it must
    never widen the PROJECT, because another project's history is not weak
    evidence for this one, it is wrong evidence."""
    fake, calls = _recorder(lambda where: [_chunk(f"c{i}") for i in range(MIN_POOL - 1)])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    gather_evidence("any question", corpus="job", k=3, repo="mealwise")

    sections = list(SECTION_SOURCES)
    assert len(calls) == 2 * len(sections)          # every section fell back
    for i, name in enumerate(sections):
        widened = calls[2 * i + 1]["where"]
        assert widened == _where_for(WIDENED_SOURCES[name], repo="mealwise")
        assert {"repo": "mealwise"} in widened["$and"]


def test_no_repo_leaves_every_filter_unscoped(monkeypatch):
    """Omitting repo must reproduce the pre-scoping filters exactly, or the v1
    study corpus (which has no repo metadata) would retrieve nothing at all."""
    fake, calls = _recorder(lambda where: [_chunk("a"), _chunk("b"), _chunk("c")])
    monkeypatch.setattr("retrieve.star_evidence.hybrid_search", fake)

    gather_evidence("any question", corpus="job", k=3)

    assert [c["where"] for c in calls] == [_where_for(s) for s in SECTION_SOURCES.values()]
    assert all("$and" not in c["where"] for c in calls)

