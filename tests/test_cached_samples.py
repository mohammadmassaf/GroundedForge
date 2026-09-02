"""
Consistency tests for the committed demo artifacts (samples/*.json).

These files are what the Space renders on load, before any model is loaded and
without spending a Groq call. They are DATA, not code, which is exactly why
they need tests: nothing else checks them. A hand-edited artifact still parses,
still renders, and quietly stops being a record of a real run -- and the whole
claim of this project is that its outputs are evidence rather than assertions.

So these tests do not check the wording of any answer. They check the
properties that make the file trustworthy: that every citation resolves, that
the counts match the rows, and that the known trap escape is still recorded as
an escape rather than silently tidied away.

Deterministic and offline -- they read two JSON files.
"""
import json
from pathlib import Path

import pytest

QUIZ = Path("samples/cached_quiz.json")
TRAPS = Path("samples/cached_traps.json")


@pytest.fixture(scope="module")
def quiz():
    return json.loads(QUIZ.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def traps():
    return json.loads(TRAPS.read_text(encoding="utf-8"))


# --- the cached quiz -------------------------------------------------------

def test_every_citation_resolves_within_the_file(quiz):
    """
    The artifact must be self-contained: the Space renders a citation by looking
    its chunk_id up in this same file's `chunks`. A citation pointing outside it
    renders as a blank source box -- a claim with no visible evidence, which is
    the one thing this project must never show.
    """
    pool = {c["chunk_id"] for c in quiz["chunks"]}
    for item in quiz["kept"] + quiz["struck"]:
        missing = set(item["citations"]) - pool
        assert not missing, f"{item['question'][:50]!r} cites {missing}"


def test_every_claim_cites_something(quiz):
    """Field(min_length=1) on the schema says a validated claim always cites.
    Pin it here too: the cached file is read straight to the UI without passing
    back through Pydantic."""
    for item in quiz["kept"] + quiz["struck"]:
        assert item["citations"], f"uncited claim: {item['question'][:60]!r}"


def test_the_quiz_came_from_the_demo_corpus(quiz):
    """Public RFCs only. A cached artifact built against `job` or `networks`
    would put vault notes or copyrighted course text on a public page."""
    assert quiz["corpus"] == "demo"
    sources = {c["source_file"] for c in quiz["chunks"]}
    assert sources <= {"rfc768.txt", "rfc791.txt", "rfc793.txt"}, sources


def test_the_run_metadata_is_present(quiz):
    """topic/model/k/n are shown next to the artifact so a reader knows what
    produced it. A cached run with no provenance is just text on a page."""
    for key in ("topic", "model", "k", "n", "generated_at"):
        assert quiz.get(key), f"missing {key}"


# --- the cached traps ------------------------------------------------------

def test_caught_count_matches_the_rows(traps):
    """The headline number the panel prints must be derived from the rows, not
    stored beside them and allowed to drift."""
    assert traps["caught"] == sum(1 for t in traps["traps"] if t["struck"])
    assert traps["total"] == len(traps["traps"])


def test_a_struck_trap_names_the_stage_that_caught_it(traps):
    """struck and caught_by have to agree. A trap marked struck with no stage
    is a panel row claiming a catch nothing performed."""
    for t in traps["traps"]:
        if t["struck"]:
            assert t["caught_by"] in {"quant", "critic"}, t["id"]
        else:
            assert t["caught_by"] is None, t["id"]


def test_every_trap_carries_its_evidence(traps):
    """The panel puts the planted claim next to the sentence that contradicts
    it. Without the evidence text the strike is an assertion, which is the
    failure mode this whole project exists to avoid."""
    for t in traps["traps"]:
        assert t["evidence"], t["id"]
        assert {e["chunk_id"] for e in t["evidence"]} == set(t["citations"]), t["id"]
        assert all(e["text"].strip() for e in t["evidence"]), t["id"]


def test_the_known_escape_is_still_recorded_as_an_escape(traps):
    """
    dt4 is not caught (notes.md Finding 9): the Critic quotes "units of 8 octets
    (64 bits)" and calls it a match. Two prompt fixes failed and the prompt was
    reverted, so the honest number is 5/6.

    This test exists to stop the escape being quietly removed. If a future model
    or prompt actually catches dt4, this test SHOULD fail -- that is the signal
    to go update Finding 9 and the headline, not to delete the trap.
    """
    dt4 = next(t for t in traps["traps"] if t["id"] == "dt4")
    assert dt4["struck"] is False
    assert traps["caught"] == 5 and traps["total"] == 6


def test_the_traps_match_the_authored_set(traps):
    """The cached verdicts must correspond to eval/eval_set_demo.json, or the
    panel is showing results for claims that are no longer the ones on file."""
    authored = json.loads(Path("eval/eval_set_demo.json").read_text(encoding="utf-8"))["traps"]
    by_id = {t["id"]: t for t in authored}
    assert {t["id"] for t in traps["traps"]} == set(by_id)
    for cached in traps["traps"]:
        source = by_id[cached["id"]]
        assert cached["claim"] == source["claim"], cached["id"]
        assert cached["citations"] == source["citations"], cached["id"]
        assert cached["expected_stage"] == source["stage"], cached["id"]
