"""
Unit tests for strike attribution (eval/strikes.py).

WHY THIS MODULE GETS EXACT ASSERTIONS
-------------------------------------
Testing an LLM system means splitting it into layers that can be asserted and
layers that can only be measured:

  deterministic   load_strikes parses JSONL, uncited is set arithmetic,
                  rank_uncited is BM25 over a fixed list, attribute is two
                  comparisons against two constants. Same input -> same output,
                  every run. There is a known-correct answer, so: assert it.

  stochastic      the Generator and the Critic are LLM calls. Ask twice, get
                  two strings. No expected value exists, so they are not tested
                  here at all -- they are measured in aggregate by
                  eval/run_eval.py (grounding %, recall@k) against an eval set,
                  and a regression shows up as a moving number, not a failing
                  assert.

strikes.py is entirely in the first column: no network, no model, every input a
file. That is why it is the right place to spend test effort -- it backs a
published finding (notes.md, Finding 8), and a silent bug here changes a
conclusion rather than crashing.

WHAT THE FIXTURES DO NOT TOUCH
------------------------------
The real traces/ directory is measurement INPUT (see conftest.py -- the suite
once wrote 81 fake traces into it and the strike count read 62 instead of 36).
That autouse fixture redirects `critic.trace.TRACE_DIR`, which is a DIFFERENT
global from `strikes.TRACE_DIR`. It stops tests writing traces; it does not
stop them reading. Any test touching sweep() must patch strikes.TRACE_DIR
itself.

Likewise the real chunks/ corpora: every test here scores against a six-chunk
corpus built in-process. Real BM25 arithmetic, controlled inputs.
"""
import json

import pytest
from rank_bm25 import BM25Okapi

from eval import strikes as S
from retrieve import keyword
from retrieve.keyword import _tokenize


# --- the tiny corpus -------------------------------------------------------
#
# Six chunks, chosen so the three verdicts and the SECRET_key pathology are all
# reachable with the REAL thresholds (MIN_MARGIN=2.0, MIN_BASELINE=0.2). Scores
# verified against rank_bm25:
#
#   claim "create database tables automatically on startup"
#       t-alembic 6.12   t-db 0.72   everything else 0.00
#   claim "adding database services layer"
#       t-db 5.47   t-alembic 0.51
#   claim "stored the jwt secret key and create database tables automatically"
#       t-alembic 3.88   t-auth 1.85   t-readme 0.00  <-- the pathology
#
# That last line is the whole reason `attribute` checks unmeasurable first:
# t-readme CONTAINS the words secret, key and jwt, but as the single token
# `SECRET_key=your_jwt_secret`, so it scores zero against a claim about them
# while an unrelated chunk clears MIN_MARGIN.
TINY_CHUNKS = [
    {"chunk_id": "t-alembic",
     "text": "create database tables automatically on startup with alembic migrations"},
    {"chunk_id": "t-db",
     "text": "adding database services layer"},
    {"chunk_id": "t-gemini",
     "text": "configure the gemini environment variable scaffolding"},
    {"chunk_id": "t-auth",
     "text": "jwt auth login and refresh token endpoints"},
    {"chunk_id": "t-readme",
     "text": "SECRET_key=your_jwt_secret goes in the env file"},
    {"chunk_id": "t-noise",
     "text": "and the and of and the project and"},
]

TINY = "tiny"   # corpus name the tests pass around

ALL_IDS = [c["chunk_id"] for c in TINY_CHUNKS]

ALEMBIC_CLAIM = "create database tables automatically on startup"
DB_CLAIM = "adding database services layer"
SECRET_CLAIM = "stored the jwt secret key and create database tables automatically"


@pytest.fixture(autouse=True)
def tiny_corpus(monkeypatch):
    """
    Make _get_index("tiny") return a real BM25 index over TINY_CHUNKS, without
    writing chunks/tiny.json and without mocking _get_index.

    _get_index is a lazy singleton keyed by corpus name, so seeding its cache
    is enough -- the real function still runs, the real index is still built,
    and the real positional alignment between get_scores() and `chunks` is
    still exercised. Mocking _get_index instead would replace exactly the code
    that rank_uncited's chunk_id -> position bridge depends on, and the test
    would then prove only that the mock agrees with itself.

    monkeypatch.setitem reverts the cache entry after each test, so a stale
    "tiny" index cannot leak into another test file.
    """
    index = BM25Okapi([_tokenize(c["text"]) for c in TINY_CHUNKS])
    monkeypatch.setitem(keyword._indexes, TINY, (index, TINY_CHUNKS))


# --- trace-file helpers ----------------------------------------------------

def write_trace(tmp_path, *events, name="bullets_20260830_120000.jsonl"):
    """
    Write `events` as one JSONL file in tmp_path and return its Path.

    Events are dicts in the exact shape the real Tracer emits, so the tests
    exercise the real parse. Callers build them with the three constructors
    below.
    """
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return path


def generated(pool, round_=1):
    """A `generated` event -- the ONLY line that carries the pool."""
    return {"ts": "2026-08-30T12:00:00", "event": "generated",
            "round": round_, "pool": list(pool)}


def generated_pre_22fb586(round_=1):
    """
    A `generated` event as the traces on disk before 22fb586 wrote it: the
    event IS there, it just carries no `pool` key. star traces still look like
    this today -- they log `pool_sizes` and a per-section `pools` dict, neither
    of which is the flat list this module reads.

    Distinct from omitting the line entirely, and the more dangerous of the two
    shapes: `.get("pool")` returns None here, but `.get("pool", [])` -- a
    one-character-plausible bug -- returns [] and turns 83 unknowns into 83
    findings.
    """
    return {"ts": "2026-07-31T21:30:12", "event": "generated",
            "round": round_, "pool_sizes": {"situation": 6, "task": 6}}


def quant(bullet, citations, passed, round_=1, reason=""):
    """A `quant_check` verdict. Note the key is `passed`, and the claim
    text rides under `bullet` -- the bullets generator's spelling."""
    return {"ts": "2026-08-30T12:00:01", "event": "quant_check", "round": round_,
            "bullet": bullet, "citations": list(citations),
            "passed": passed, "reason": reason}


def critic(claim_key, claim, citations, supported, round_=1, reason=""):
    """A `critic_verdict`. The key is `supported`, and the claim text rides
    under whichever of bullet/question/section the generator uses --
    claim_key is that spelling, so a test can prove all three normalize."""
    return {"ts": "2026-08-30T12:00:02", "event": "critic_verdict", "round": round_,
            claim_key: claim, "citations": list(citations),
            "supported": supported, "reason": reason}


def a_strike(claim, citations, pool, **over):
    """
    A strike dict as load_strikes would have produced it, for testing the
    functions downstream of it without going through a file.

    `pool=None` builds the unattributable shape.
    """
    return {"trace": "t.jsonl", "stage": "critic", "round": 1,
            "claim": claim, "citations": list(citations),
            "pool": None if pool is None else list(pool),
            "attributable": pool is not None, "reason": "", **over}


# ===========================================================================
# load_strikes -- replaying an event log to reconstruct state
# ===========================================================================

def test_pool_is_carried_forward_to_later_verdicts(tmp_path):
    """The pool rides on `generated`, the verdict rides on a LATER line. A
    strike must come back holding the pool from the generated event above it."""
    path = write_trace(
        tmp_path,
        generated(["t-alembic", "t-db", "t-gemini"]),
        critic("bullet", "some claim", ["t-db"], supported=False),
    )
    struck = S.load_strikes(path)
    assert len(struck) == 1
    assert struck[0]["pool"] == ["t-alembic", "t-db", "t-gemini"]
    assert struck[0]["attributable"] is True


def test_a_later_pool_replaces_the_earlier_one(tmp_path):
    """Multi-round runs log `generated` once per round. A round-2 strike must
    be attributed against round 2's pool -- carrying round 1's forward would
    score the claim against chunks the generator never saw."""
    path = write_trace(
        tmp_path,
        generated(["t-alembic", "t-db"], round_=1),
        critic("bullet", "round one claim", ["t-db"], supported=False, round_=1),
        generated(["t-auth", "t-readme"], round_=2),
        critic("bullet", "round two claim", ["t-auth"], supported=False, round_=2),
    )
    first, second = S.load_strikes(path)
    assert first["pool"] == ["t-alembic", "t-db"]
    assert second["pool"] == ["t-auth", "t-readme"]


def test_both_verdict_events_are_recognized(tmp_path):
    """quant_check spells its outcome `passed`, critic_verdict spells it
    `supported`. Both are strikes when false, and downstream code must not have
    to know which -- only `stage` distinguishes them."""
    path = write_trace(
        tmp_path,
        generated(["t-alembic", "t-db"]),
        quant("a fabricated figure", ["t-db"], passed=False),
        critic("bullet", "an unsupported claim", ["t-db"], supported=False),
    )
    struck = S.load_strikes(path)
    assert [s["stage"] for s in struck] == ["quant", "critic"]


def test_passing_verdicts_are_not_strikes(tmp_path):
    """The obvious direction, and the one whose absence would be silent: a file
    of healthy claims must yield nothing. If this ever fails, every count in
    Finding 8 is inflated."""
    path = write_trace(
        tmp_path,
        generated(["t-alembic", "t-db"]),
        quant("a real figure", ["t-db"], passed=True),
        critic("bullet", "a supported claim", ["t-db"], supported=True),
    )
    assert S.load_strikes(path) == []


@pytest.mark.parametrize("claim_key", ["bullet", "question", "section"])
def test_claim_text_is_read_under_every_generator_spelling(tmp_path, claim_key):
    """bullets logs `bullet`, quiz logs `question`, star logs `section`. All
    three must land in `claim`, or a strike from one generator arrives with an
    empty claim and then scores 0.00 against everything -- a silent
    unmeasurable."""
    path = write_trace(
        tmp_path,
        generated(["t-alembic", "t-db"]),
        critic(claim_key, "the claim text", ["t-db"], supported=False),
    )
    assert S.load_strikes(path)[0]["claim"] == "the claim text"


def test_a_pre_pool_era_strike_is_unattributable_not_empty(tmp_path):
    """
    THE distinction the module exists to protect. A trace with no `generated`
    pool (83 of the ones on disk predate 22fb586) must come back with
    pool=None and attributable=False.

    pool=[] would read as "there was nowhere else the support could have been",
    which is a FINDING. pool=None reads as "we never recorded it", which is
    ignorance. Collapsing them turns 83 unknowns into 83 pieces of evidence for
    over-reach. Hence `is None`, not a falsy check -- [] is falsy too, and [] is
    the bug.

    Both shapes are pinned. The `generated`-with-no-pool case is the one that
    actually exists on disk and the one a `.get("pool", [])` default silently
    breaks; the no-`generated`-at-all case only exercises the initializer.
    A mutation run proved the second case alone passes against that bug.
    """
    with_event = write_trace(
        tmp_path,
        generated_pre_22fb586(),
        critic("bullet", "a claim from before the pool was logged",
               ["t-db"], supported=False),
        name="pre_era_with_generated.jsonl",
    )
    without_event = write_trace(
        tmp_path,
        critic("bullet", "a claim from before the pool was logged",
               ["t-db"], supported=False),
        name="pre_era_no_generated.jsonl",
    )
    for path in (with_event, without_event):
        strike = S.load_strikes(path)[0]
        assert strike["pool"] is None, path.name
        assert strike["attributable"] is False, path.name


def test_blank_lines_are_skipped(tmp_path):
    """A trailing newline is normal in an append-written JSONL. json.loads("")
    raises, so a file ending in one would take out the whole sweep."""
    path = tmp_path / "bullets_blank.jsonl"
    path.write_text(
        json.dumps(generated(["t-alembic", "t-db"])) + "\n"
        + "\n"
        + json.dumps(critic("bullet", "a claim", ["t-db"], supported=False)) + "\n"
        + "\n",
        encoding="utf-8",
    )
    assert len(S.load_strikes(path)) == 1


# ===========================================================================
# uncited -- set arithmetic that must not become a set
# ===========================================================================

def test_uncited_is_pool_minus_citations():
    strike = a_strike("a claim", ["t-db"],
                      ["t-alembic", "t-db", "t-gemini", "t-auth"])
    assert S.uncited(strike) == ["t-alembic", "t-gemini", "t-auth"]


def test_uncited_preserves_pool_order():
    """
    Pool order is retrieval rank. "it skipped the top-ranked chunk" and "it
    skipped the 9th" are different stories, and a set loses that. The pool here
    is deliberately not in sorted order, and long enough that a set-based
    implementation cannot pass by luck.
    """
    pool = ["t-noise", "t-alembic", "t-readme", "t-db", "t-auth", "t-gemini"]
    strike = a_strike("a claim", ["t-db"], pool)
    assert S.uncited(strike) == ["t-noise", "t-alembic", "t-readme",
                                 "t-auth", "t-gemini"]


def test_uncited_is_empty_when_the_whole_pool_was_cited():
    """Legitimately empty: nowhere else the support could have been. This is a
    no_lead by definition, and it must NOT raise -- only a missing pool does."""
    strike = a_strike("a claim", ["t-alembic", "t-db"], ["t-alembic", "t-db"])
    assert S.uncited(strike) == []


def test_uncited_raises_on_an_unattributable_strike():
    """Contract: the caller holds `attributable` and is expected to check it.
    Returning [] here would silently feed "nothing was left uncited" -- a
    finding -- into the report."""
    with pytest.raises(ValueError):
        S.uncited(a_strike("a claim", ["t-db"], None))


# ===========================================================================
# rank_uncited -- real BM25, purpose-built corpus
# ===========================================================================

def test_ranked_best_first():
    ranked = S.rank_uncited(ALEMBIC_CLAIM, ALL_IDS, TINY)
    assert ranked[0][0] == "t-alembic"
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_every_requested_id_is_returned_exactly_once():
    """Ranking must not drop the zero-scoring chunks. They are the evidence for
    a no_lead."""
    ranked = S.rank_uncited(ALEMBIC_CLAIM, ALL_IDS, TINY)
    assert sorted(cid for cid, _ in ranked) == sorted(ALL_IDS)


def test_a_score_belongs_to_its_own_chunk():
    """
    The bridge under test is `positions = {chunk_id: i}` against a scores array
    aligned with the FULL corpus. An off-by-one there does not crash -- it
    silently hands each claim its neighbour's score, and every verdict in
    Finding 8 becomes noise. So this pins scores by chunk identity, never by
    position in the result.
    """
    scores = dict(S.rank_uncited(ALEMBIC_CLAIM, ["t-gemini", "t-alembic"], TINY))
    assert scores["t-alembic"] > 5.0     # the chunk that says exactly this
    assert scores["t-gemini"] == 0.0     # shares no token with the claim


def test_scoring_a_subset_does_not_change_the_scores():
    """
    Scored against the full-corpus index, never a fresh BM25 over the few
    candidate chunks -- IDF measured over 6 documents is noise, and a chunk
    could win on the word "and". A local-index implementation fails this.
    """
    subset = dict(S.rank_uncited(ALEMBIC_CLAIM, ["t-alembic", "t-db"], TINY))
    whole = dict(S.rank_uncited(ALEMBIC_CLAIM, ALL_IDS, TINY))
    assert subset["t-alembic"] == whole["t-alembic"]
    assert subset["t-db"] == whole["t-db"]


def test_an_unknown_chunk_id_raises():
    """The next re-ingest will shift chunk ids. An id that does not resolve
    must blow up loudly rather than score 0.00 and be read as evidence."""
    with pytest.raises(KeyError):
        S.rank_uncited(ALEMBIC_CLAIM, ["t-alembic", "gone-after-reingest"], TINY)


# ===========================================================================
# attribute -- the three-way verdict
# ===========================================================================

def test_a_better_scoring_uncited_chunk_is_a_lead():
    """
    Cites t-db (0.72) with t-alembic (6.12) sitting uncited in the same pool.
    Margin 5.40, baseline above MIN_BASELINE. Finding 8's first job lead in
    miniature.

    The shortlist is the product, so a lead pointing at the wrong chunk is
    worse than no lead -- hence the second assert.
    """
    strike = a_strike(ALEMBIC_CLAIM, ["t-db"], ["t-alembic", "t-db", "t-gemini"])
    result = S.attribute(strike, TINY)
    assert result["verdict"] == "lead"
    assert result["best_uncited"][0] == "t-alembic"
    assert result["best_cited"][0] == "t-db"


def test_citing_the_best_available_chunk_is_no_lead():
    """Cites t-db (5.47); the best uncited is t-alembic at 0.51. The claim went
    further than its evidence -- over-reach, not misdirection. 27 of 36 strikes
    look like this."""
    strike = a_strike(DB_CLAIM, ["t-db"], ["t-alembic", "t-db", "t-gemini"])
    assert S.attribute(strike, TINY)["verdict"] == "no_lead"


def test_unmeasurable_is_checked_before_the_margin():
    """
    THE ordering test, and the one that would have reversed Finding 8.

    The claim cites t-readme, which CONTAINS secret/key/jwt but as the single
    token `SECRET_key=your_jwt_secret`, so it scores 0.00. Uncited t-alembic
    scores 3.88 -- a margin comfortably ABOVE MIN_MARGIN. A margin-first
    implementation therefore returns "lead" here, ranking the four least
    trustworthy strikes in the set highest.

    The margin assert is not decoration: without it this test would still pass
    if the fixture scores drifted below MIN_MARGIN, i.e. for the wrong reason.
    """
    strike = a_strike(SECRET_CLAIM, ["t-readme"],
                      ["t-alembic", "t-readme", "t-auth"])
    result = S.attribute(strike, TINY)

    assert result["best_cited"][1] < S.MIN_BASELINE
    assert result["best_uncited"][1] - result["best_cited"][1] >= S.MIN_MARGIN
    assert result["verdict"] == "unmeasurable"


def test_citing_the_whole_pool_is_no_lead_not_a_crash():
    """Empty shortlist: there was nowhere else the support could have been.
    Indexing shortlist[0] unguarded is an IndexError that takes out the
    sweep."""
    strike = a_strike(ALEMBIC_CLAIM, ["t-alembic", "t-db"], ["t-alembic", "t-db"])
    result = S.attribute(strike, TINY)
    assert result["shortlist"] == []
    assert result["best_uncited"] is None
    assert result["verdict"] == "no_lead"


def test_the_original_strike_fields_survive():
    """attribute RETURNS the strike with fields added -- the report prints
    a["trace"], a["stage"], a["claim"], a["reason"] straight off the result."""
    strike = a_strike(ALEMBIC_CLAIM, ["t-db"], ["t-alembic", "t-db"],
                      trace="bullets_x.jsonl", stage="quant",
                      reason="not in evidence")
    result = S.attribute(strike, TINY)
    assert result["trace"] == "bullets_x.jsonl"
    assert result["stage"] == "quant"
    assert result["claim"] == ALEMBIC_CLAIM
    assert result["reason"] == "not in evidence"
    assert {"verdict", "best_cited", "best_uncited", "shortlist"} <= result.keys()


# ===========================================================================
# belongs_to -- measure, don't infer
# ===========================================================================

def test_a_strike_whose_ids_resolve_belongs():
    assert S.belongs_to(a_strike("c", ["t-db"], ["t-alembic", "t-db"]), TINY)


def test_a_strike_with_a_foreign_id_does_not_belong():
    """Decided by resolution, not by filename convention (quiz_* runs exist
    over both networks and demo). One unresolvable id is enough: it means the
    trace was written against a different chunk set, so NONE of its scores mean
    what they appear to."""
    strike = a_strike("c", ["t-db"], ["t-alembic", "t-db", "Part-3_p22"])
    assert S.belongs_to(strike, TINY) is False


def test_an_unattributable_strike_does_not_belong():
    """pool=None must be False, not a TypeError from iterating None. And an
    empty pool resolves vacuously under all() -- "belongs" is not the right
    answer for a strike with nothing to resolve."""
    assert S.belongs_to(a_strike("c", ["t-db"], None), TINY) is False
    assert S.belongs_to(a_strike("c", ["t-db"], []), TINY) is False


# ===========================================================================
# sweep -- the three buckets, and the traces/ boundary
# ===========================================================================

def test_sweep_separates_the_three_buckets(tmp_path, monkeypatch):
    """
    unattributable / foreign / attributed, counted apart. Folding any two
    together is how a lead count over 21 strikes gets quoted as one over 198.

    NOTE the fixture boundary: conftest.py's autouse fixture patches
    critic.trace.TRACE_DIR, a DIFFERENT global. sweep reads strikes.TRACE_DIR,
    so without the patch below this test walks the real 245-file traces/
    directory and asserts against live measurement data.
    """
    monkeypatch.setattr(S, "TRACE_DIR", tmp_path)

    write_trace(tmp_path,
                critic("bullet", "pre-22fb586 claim", ["t-db"], supported=False),
                name="a_no_pool.jsonl")
    write_trace(tmp_path,
                generated(["t-alembic", "Part-3_p22"]),
                critic("bullet", "networks claim", ["t-alembic"], supported=False),
                name="b_foreign.jsonl")
    write_trace(tmp_path,
                generated(["t-alembic", "t-db", "t-gemini"]),
                critic("bullet", ALEMBIC_CLAIM, ["t-db"], supported=False),
                name="c_ours.jsonl")

    all_strikes, attributed, foreign = S.sweep(TINY)

    assert len(all_strikes) == 3
    assert foreign == 1
    assert len(attributed) == 1
    assert attributed[0]["trace"] == "c_ours.jsonl"
    # the unattributable one is in the denominator but nowhere else
    assert any(s["attributable"] is False for s in all_strikes)
    assert all(a["attributable"] for a in attributed)


def test_sweep_reads_only_matching_files(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "TRACE_DIR", tmp_path)
    write_trace(tmp_path,
                generated(["t-alembic", "t-db"]),
                critic("bullet", ALEMBIC_CLAIM, ["t-db"], supported=False),
                name="bullets_x.jsonl")
    write_trace(tmp_path,
                generated(["t-alembic", "t-db"]),
                critic("question", DB_CLAIM, ["t-db"], supported=False),
                name="quiz_x.jsonl")

    all_strikes, _, _ = S.sweep(TINY, pattern="bullets_*.jsonl")
    assert [s["trace"] for s in all_strikes] == ["bullets_x.jsonl"]


# ===========================================================================
# strike_report -- wording is a behaviour
# ===========================================================================

def _attributed(verdict, claim="a claim", stage="critic"):
    """The minimum an attributed strike needs to survive strike_report. Only
    the lead branch reads best_cited/shortlist/reason, but supplying them for
    every verdict keeps the stub honest."""
    return {"trace": "t.jsonl", "stage": stage, "claim": claim,
            "reason": "not in evidence", "verdict": verdict,
            "best_cited": ("t-db", 0.72), "best_uncited": ("t-alembic", 6.12),
            "shortlist": [("t-alembic", 6.12), ("t-gemini", 0.0)]}


def test_the_report_never_claims_mis_citation():
    """
    The headline is a LEAD count. The worked example in the module docstring is
    why: the top-scoring uncited chunk (mealwise@0cf0a59, 8.61 vs 0.12) turned
    out to be about Gemini scaffolding and supported nothing. Printing that
    number as a mis-citation count publishes a measurement never made.
    """
    strike = a_strike(ALEMBIC_CLAIM, ["t-db"], ["t-alembic", "t-db"])
    report = S.strike_report([strike], [S.attribute(strike, TINY)])
    assert "mis-cit" not in report.lower()
    assert "better-scoring uncited chunk" in report


def test_the_report_shows_the_denominator():
    """len(strikes) and len(attributed) both, always. 5 leads out of 36 and 5
    out of 198 are different claims."""
    strikes = [a_strike("c", ["t-db"], None) for _ in range(3)]
    report = S.strike_report(strikes, [_attributed("lead")])
    assert "strikes loaded     : 3" in report
    assert "attributed         : 1" in report


def test_unmeasurable_is_reported_as_its_own_line():
    """
    Not folded into either answer. Counting the four SECRET_key strikes as
    leads takes mis-citation from 2 to 6 and reverses the conclusion; counting
    them as no_lead manufactures evidence for over-reach.
    """
    attributed = [_attributed("lead"), _attributed("no_lead"),
                  _attributed("unmeasurable")]
    report = S.strike_report([], attributed)
    assert "had a better-scoring uncited chunk : 1" in report
    assert "cited the best available           : 1" in report
    assert "not measurable (see MIN_BASELINE)  : 1" in report


def test_the_pre_pool_era_bucket_is_named_when_present():
    """The line is conditional -- present with its count and the commit when
    there are unattributable strikes, absent when there are none."""
    with_none = S.strike_report(
        [a_strike("c", ["t-db"], ["t-alembic", "t-db"])], [_attributed("lead")])
    assert "no pool recorded" not in with_none

    with_some = S.strike_report(
        [a_strike("c", ["t-db"], None)], [_attributed("lead")])
    assert "no pool recorded" in with_some
    assert S.POOL_ERA in with_some


def test_the_foreign_bucket_is_named_when_present():
    """Same conditional shape. A foreign count of 0 must not print a line
    saying zero traces came from another corpus."""
    assert "another corpus" not in S.strike_report([], [_attributed("lead")], 0)
    assert "another corpus" in S.strike_report([], [_attributed("lead")], 4)
