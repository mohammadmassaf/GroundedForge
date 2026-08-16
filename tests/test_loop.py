"""
Integration tests for the STAR orchestration policy (critic/loop.py::run_star_loop),
with BOTH LLM calls mocked out.

run_star_loop chains generate_star -> a scope check -> check_quantities ->
check_claim, and decides what gets flagged. We are not testing whether the
model writes good answers (that's the eval set's job) -- we're testing the
POLICY: given a known answer and known verdicts, is the right section flagged,
for the right reason, and in the right order?

Three things make this different from test_loop_integration:

1. THREE stages, not two. A section can be flagged by the scope check, by the
   deterministic quant check, or by the LLM Critic -- and the first two
   `continue`, so a section they catch must never reach the LLM. Order is a
   behaviour worth testing, not an implementation detail: the whole point of a
   deterministic pre-check is that it runs BEFORE you pay for judgment.

2. Flagged sections are KEPT. run_bullets_loop drops a struck bullet; a STAR
   answer with no Result is broken, so this loop returns the section and marks
   it untrustworthy instead.

3. Tracer writes real files to traces/. Tests patch it out -- a unit test
   should not litter the run history it's meant to be independent of.

Same rule as always: patch the name WHERE IT IS USED (critic.loop.X).
"""
from critic.loop import run_star_loop
from generate.schema import STARAnswer, StarSection
from critic.schema import Verdict


def _chunk(cid):
    """Minimal pooled chunk. `text` carries no digits, so any number in a
    section's text is automatically unsupported unless a test puts it here."""
    return {"chunk_id": cid, "source_file": "f.md", "page": 1,
            "source_type": "docs", "text": f"evidence text for {cid}"}


# Distinct ids per pool, so a citation crossing pools is constructible.
POOLS = {
    "situation": [_chunk("s1"), _chunk("s2")],
    "task":      [_chunk("t1")],
    "action":    [_chunk("a1")],
    "result":    [_chunk("r1")],
}


def _answer(**overrides):
    """
    Build a STARAnswer. Each kwarg is (text, citations) for that section, e.g.
    _answer(action=("Improved it by 40%", ["a1"])). Anything not overridden
    gets a clean, well-cited default.
    """
    sections = {
        "situation": ("the context this work happened in", ["s1"]),
        "task":      ("what specifically had to be solved", ["t1"]),
        "action":    ("what I implemented to solve it", ["a1"]),
        "result":    ("what came of the work in the end", ["r1"]),
    }
    sections.update(overrides)
    return STARAnswer(
        question="an interview question",
        **{name: StarSection(text=t, citations=c) for name, (t, c) in sections.items()},
    )


class _NullTracer:
    """Swallows trace writes so tests don't create files in traces/."""
    path = "(test)"

    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def _critic_recorder(supported=True, reason="found in the cited evidence"):
    """
    Build (fake_check_claim, seen) where `seen` records the section text of
    every LLM call. An EMPTY `seen` entry is how a test proves an earlier
    stage short-circuited before reaching the model.
    """
    seen = []

    def fake(question, text, cited_chunks):
        seen.append(text)
        return Verdict(supported=supported, reason=reason)

    return fake, seen


def _patch(monkeypatch, answer, check_claim):
    """Wire the three seams: the generator, the Critic, and the tracer."""
    monkeypatch.setattr("critic.loop.Tracer", _NullTracer)
    monkeypatch.setattr("critic.loop.generate_star", lambda question, pools: answer)
    monkeypatch.setattr("critic.loop.check_claim", check_claim)


# --- the clean path ---------------------------------------------------------

def test_clean_answer_flags_nothing(monkeypatch):
    """Every section cites its own pool, has no figures, and the Critic
    supports it -> nothing flagged, and all four sections saw the Critic."""
    fake_critic, seen = _critic_recorder(supported=True)
    _patch(monkeypatch, _answer(), fake_critic)

    answer, flagged = run_star_loop("a question", POOLS)

    assert flagged == []
    assert len(seen) == 4


def test_flagged_section_is_kept_not_dropped(monkeypatch):
    """Unlike bullets, a failing section stays in the answer -- the artifact
    shows it WITH a warning rather than silently omitting a STAR part."""
    fake_critic, _ = _critic_recorder(supported=True)
    _patch(monkeypatch,
           _answer(action=("Improved grounding by 40% overall", ["a1"])),
           fake_critic)

    answer, flagged = run_star_loop("a question", POOLS)

    assert len(flagged) == 1
    assert [name for name, _ in answer.sections()] == ["Situation", "Task", "Action", "Result"]
    assert "40%" in answer.action.text


# --- stage 0: the scope check ----------------------------------------------

def test_citation_outside_its_pool_is_flagged(monkeypatch):
    """TODO(you): give Action a citation belonging to the SITUATION pool (the
    ids in POOLS are distinct for exactly this). Assert Action is flagged and
    the reason names the stray id.

    This is the line that makes an evidence pool an authorization boundary
    rather than a bucket of context -- without it, "what I did" can be
    answered from a planning note and nothing objects."""
    fake_critic , _ = _critic_recorder(supported = True)
    _patch(monkeypatch,
               _answer(action=("what I implemented to solve it", ["s1"])),
               fake_critic)
    answer , flagged = run_star_loop("a question" , POOLS)
    assert len(flagged) == 1 
    name , reason = flagged[0]
    assert name == "Action"
    assert "s1" in reason


def test_scope_violation_skips_the_llm_critic(monkeypatch):
    """TODO(you): same setup, but assert on `seen` from the recorder -- the
    out-of-scope section's text must NEVER appear, because the loop
    `continue`s before check_claim. Sending a citation you already know is
    invalid to a paid model is waste, and its verdict would be meaningless
    anyway: it would be judging text against the wrong evidence."""
    fake_critic , seen= _critic_recorder(supported = True)
    _patch(monkeypatch,
                   _answer(action=("what I implemented to solve it", ["s1"])),
                   fake_critic)
    answer , flagged = run_star_loop("a question" , POOLS)
    assert answer.action.text not in seen

# --- stage 1: the deterministic quant check --------------------------------

def test_inflated_number_is_struck_before_the_llm(monkeypatch):
    """TODO(you): the integration test the v2 spec asks for. Give a section a
    figure that appears in NO cited chunk, and assert two things: it is
    flagged with a "[quant]" reason, and its text never reached the Critic.

    test_quant.py already proves check_quantities strikes in isolation. This
    proves run_star_loop actually calls it, and calls it FIRST -- the wiring,
    not the rule."""
    fake_critic , seen= _critic_recorder(supported = True)
    _patch(monkeypatch,
                       _answer(action=("Improved grounding by 40% overall", ["a1"])),
                       fake_critic)
    answer , flagged = run_star_loop("a question" , POOLS)
    name , reason = flagged[0]
    assert answer.action.text not in seen
    assert "[quant]" in reason


# --- stage 2: the LLM Critic -----------------------------------------------

def test_unsupported_section_is_flagged_by_the_critic(monkeypatch):
    """TODO(you): a section that passes scope and quant but that the Critic
    rejects. Build the recorder with supported=False and assert the flagged
    reason is the Critic's own reason string -- not a "[quant]" prefix, which
    would mean the wrong stage caught it."""
    seen = []
    def fake_critic(question, text, cited_chunks):
        seen.append(text)
        if text == "the outcome of my work":
            return Verdict(supported=False, reason="the evidence does not mention this")
        return Verdict(supported=True, reason="found in the cited evidence")
    _patch(monkeypatch,
                           _answer(action=("the outcome of my work", ["a1"])),
                           fake_critic)
    answer , flagged = run_star_loop("a question" , POOLS)
    name , reason = flagged[0]
    assert reason == "the evidence does not mention this"
    assert answer.action.text in seen

# --- run_bullets_loop: honest gap reporting ---------------------------------
#
# This branch shipped broken and unnoticed for weeks: MIN_EVIDENCE_SCORE is
# calibrated for cosine (0..1) but the loop was reading `score`, which by then
# holds a cross-encoder logit (unbounded, usually negative). Every job-mode
# topic looked like "insufficient evidence". Nothing caught it because the gap
# path had no test and every manual run happened to have strong evidence.
#
# It now reads `vector_score`, which stays cosine all the way through fusion
# and rerank. These tests pin both directions of that threshold.

from critic.loop import run_bullets_loop, MIN_EVIDENCE_SCORE
from generate.schema import Bullets, CVBullet


def _scored(cid, vector_score):
    """A retrieved chunk carrying the cosine score the gap check reads."""
    return {"chunk_id": cid, "source_file": "f.md", "page": 1,
            "source_type": "docs", "text": f"evidence text for {cid}",
            "score": -7.0,                    # a plausible reranked logit
            "vector_score": vector_score}


def _generator_spy(bullets=None):
    """Records calls. An empty `calls` proves the gap short-circuited before
    the generator -- which is the point: no filler, and no LLM spend."""
    calls = []

    def fake(topic, chunks, n=5, avoid=None):
        calls.append(topic)
        return bullets or Bullets(bullets=[
            CVBullet(text="Implemented the thing", citations=[chunks[0]["chunk_id"]])
        ])

    return fake, calls


def _patch_bullets(monkeypatch, generate, check_claim):
    monkeypatch.setattr("critic.loop.Tracer", _NullTracer)
    monkeypatch.setattr("critic.loop.generate_bullets", generate)
    monkeypatch.setattr("critic.loop.check_claim", check_claim)
    monkeypatch.setattr("critic.loop.MAX_ROUNDS", 1)


def test_weak_evidence_returns_a_gap(monkeypatch):
    """Every chunk below the threshold -> no bullets, no strikes, a gap
    message naming the topic. This is the "insufficient evidence in corpus"
    promise, and it must fire on the COSINE score, not the reranked logit."""
    fake_gen, called = _generator_spy()
    fake_critic, _ = _critic_recorder(supported=True)
    _patch_bullets(monkeypatch, fake_gen, fake_critic)
    weak = [_scored("c1", MIN_EVIDENCE_SCORE - 0.1)]

    kept, struck, gap = run_bullets_loop("a topic nothing covers", weak, n=3)

    assert kept == [] and struck == []
    assert gap and "a topic nothing covers" in gap
    assert called == []          # the generator was never asked


def test_gap_is_decided_before_any_generation(monkeypatch):
    """The regression guard for the bug itself: a reranked logit of -7.0 sits
    on every chunk, but vector_score is healthy, so this must NOT gap. Read
    the wrong key and this test fails."""
    fake_gen, called = _generator_spy()
    fake_critic, _ = _critic_recorder(supported=True)
    _patch_bullets(monkeypatch, fake_gen, fake_critic)
    healthy = [_scored("c1", 0.62)]

    kept, struck, gap = run_bullets_loop("a covered topic", healthy, n=1)

    assert gap is None
    assert called == ["a covered topic"]
    assert len(kept) == 1


def test_no_chunks_at_all_is_a_gap(monkeypatch):
    """Empty retrieval must gap rather than crash on max() of nothing."""
    fake_gen, called = _generator_spy()
    fake_critic, _ = _critic_recorder(supported=True)
    _patch_bullets(monkeypatch, fake_gen, fake_critic)

    kept, struck, gap = run_bullets_loop("anything", [], n=3)

    assert gap and kept == [] and struck == []
    assert called == []


def test_chunk_without_a_vector_score_counts_as_no_evidence(monkeypatch):
    """A BM25-only chunk never got a cosine score. Reading it with a default
    of 0 is the honest reading -- the dense retriever didn't find it, so it
    contributes no dense confidence."""
    fake_gen, called = _generator_spy()
    fake_critic, _ = _critic_recorder(supported=True)
    _patch_bullets(monkeypatch, fake_gen, fake_critic)
    bm25_only = [{"chunk_id": "c1", "source_file": "f.md", "page": 1,
                  "source_type": "docs", "text": "evidence", "score": 8.4}]

    kept, struck, gap = run_bullets_loop("a topic", bm25_only, n=3)

    assert gap and called == []


# --- trace observability ----------------------------------------------------
#
# A strike reason like "the evidence does not mention X" has two very different
# causes that read identically: the support was NOWHERE in the pool (retrieval
# miss), or it sat in the pool under a chunk the claim didn't cite
# (mis-citation). They need opposite fixes. Diagnosing a real strike meant
# grepping the corpus by hand and getting it wrong first, so the trace now
# records both the pool and each claim's citations.

class _RecordingTracer:
    """Captures trace records instead of writing them to traces/."""
    path = "(test)"
    records = []

    def __init__(self, *args, **kwargs):
        _RecordingTracer.records = []

    def log(self, event, **fields):
        _RecordingTracer.records.append((event, fields))


def _events(name):
    return [f for e, f in _RecordingTracer.records if e == name]


def test_trace_records_the_pool_it_generated_from(monkeypatch):
    """Without the pool you cannot tell a retrieval miss from a mis-citation,
    because the Critic's reason string looks the same either way."""
    fake_gen, _ = _generator_spy()
    fake_critic, _ = _critic_recorder(supported=True)
    monkeypatch.setattr("critic.loop.Tracer", _RecordingTracer)
    monkeypatch.setattr("critic.loop.generate_bullets", fake_gen)
    monkeypatch.setattr("critic.loop.check_claim", fake_critic)
    monkeypatch.setattr("critic.loop.MAX_ROUNDS", 1)

    run_bullets_loop("a topic", [_scored("c1", 0.62), _scored("c2", 0.61)], n=1)

    assert _events("generated")[0]["pool"] == ["c1", "c2"]


def test_trace_records_what_each_claim_cited(monkeypatch):
    """The other half: which chunk the claim pointed at. The Critic judges a
    claim against ONLY these, so a verdict is unreadable without them."""
    fake_gen, _ = _generator_spy()
    fake_critic, _ = _critic_recorder(supported=False, reason="not in the evidence")
    monkeypatch.setattr("critic.loop.Tracer", _RecordingTracer)
    monkeypatch.setattr("critic.loop.generate_bullets", fake_gen)
    monkeypatch.setattr("critic.loop.check_claim", fake_critic)
    monkeypatch.setattr("critic.loop.MAX_ROUNDS", 1)

    run_bullets_loop("a topic", [_scored("c1", 0.62), _scored("c2", 0.61)], n=1)

    verdict = _events("critic_verdict")[0]
    assert verdict["citations"] == ["c1"]
    assert verdict["supported"] is False
    # pool minus citations = where the support could have been instead
    assert set(_events("generated")[0]["pool"]) - set(verdict["citations"]) == {"c2"}
