"""
Integration test for the orchestration policy (critic/loop.py::run_loop),
with the LLM MOCKED OUT.

run_loop chains two LLM calls (generate -> check_claim) and applies the
cite-or-strike policy on top: keep supported claims, strike unsupported
ones. We are NOT testing whether the model is any good (that's the eval
set's job). We're testing the POLICY: given a known generator output and
known verdicts, does the loop keep/strike correctly?

To make it deterministic and free, we replace generate + check_claim with
fakes that return canned values. monkeypatch swaps them for one test and
restores them automatically.

Key rule: patch the name WHERE IT IS USED. run_loop imported these into
critic.loop's namespace, so we patch "critic.loop.generate" and
"critic.loop.check_claim" (not critic.critic.check_claim).
"""
from critic.loop import run_loop
from generate.schema import Quiz, QuizItem
from critic.schema import Verdict


# Two chunks the canned quiz will cite. run_loop builds by_id from these
# and looks up each item's citations, so the chunk_ids must match.
CHUNKS = [
    {"chunk_id": "c1", "source_file": "f.pdf", "page": 1, "text": "supported evidence"},
    {"chunk_id": "c2", "source_file": "f.pdf", "page": 2, "text": "unrelated text"},
]


def _fake_generate(topic, chunks, n=5):
    """A canned Quiz of two items, one citing c1 and one citing c2, so the
    fake Critic below can support exactly one of them. Ignores its arguments:
    no LLM is called, which is what makes this test free and deterministic."""
    itema = QuizItem(question = "question 1" , answer = "answer " , citations = ["c1"])
    itemb = QuizItem(question = "question 2 " , answer = "answer " , citations = ["c2"])
    return Quiz(items = [itema,itemb])


def _fake_check_claim(question, answer, cited_chunks):
    """A Verdict decided by the evidence rather than at random: c1 is
    supported, anything else is not. Deciding on the CITED chunk rather than
    on the item is what makes this test exercise the real wiring - the loop
    has to resolve citations to chunks correctly for the verdicts to land."""
    if cited_chunks[0]["chunk_id"] == "c1":
        return Verdict(supported=True, reason="found in evidence")
    return Verdict(supported=False, reason="not in evidence")


def test_unsupported_claim_is_struck(monkeypatch):
    """The policy end to end: of two generated items the supported one is
    kept and the unsupported one is struck, each landing on the right side.

    MAX_ROUNDS is pinned to 1 so the top-up round doesn't re-run the fake
    generator and double the counts - the test is about the keep/strike
    decision, not about the regeneration loop.
    """
    monkeypatch.setattr("critic.loop.generate",_fake_generate)
    monkeypatch.setattr("critic.loop.check_claim" , _fake_check_claim)
    monkeypatch.setattr("critic.loop.MAX_ROUNDS" , 1)

    kept, struck = run_loop("any topic" , CHUNKS , n = 2)
    assert len(kept) == 1 
    assert len(struck) == 1
    assert kept[0].citations == ["c1"]
    assert struck[0][0].citations == ["c2"]

