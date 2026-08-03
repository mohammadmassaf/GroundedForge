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