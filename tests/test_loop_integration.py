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
    """TODO(you): return a canned Quiz with TWO items:
       - item A: question/answer + citations=["c1"]   (will be judged supported)
       - item B: question/answer + citations=["c2"]   (will be judged NOT supported)
    Build it with Quiz(items=[QuizItem(...), QuizItem(...)]). Ignore the
    args — this fake doesn't call any LLM."""
    itema = QuizItem(question = "question 1" , answer = "answer " , citations = ["c1"])
    itemb = QuizItem(question = "question 2 " , answer = "answer " , citations = ["c2"])
    return Quiz(items = [itema,itemb])


def _fake_check_claim(question, answer, cited_chunks):
    """TODO(you): return a Verdict. Make it deterministic on the evidence:
    if the cited chunk is c1 -> supported=True; otherwise supported=False.
    You can look at cited_chunks[0]["chunk_id"]. reason must be >=5 chars."""
    if cited_chunks[0]["chunk_id"] == "c1":
        return Verdict(supported=True, reason="found in evidence")
    return Verdict(supported=False, reason="not in evidence")


def test_unsupported_claim_is_struck(monkeypatch):
    """TODO(you):
    1. monkeypatch.setattr("critic.loop.generate", _fake_generate)
       monkeypatch.setattr("critic.loop.check_claim", _fake_check_claim)
       monkeypatch.setattr("critic.loop.MAX_ROUNDS", 1)   # one round, so the
           # regeneration path doesn't re-run _fake_generate and double the counts
    2. kept, struck = run_loop("any topic", CHUNKS, n=2)
    3. Assert: len(kept) == 1 and len(struck) == 1.
       Assert the kept item is the c1 one, the struck one is the c2 one.
       struck is a list of (QuizItem, reason) tuples — check struck[0][0]
       and that struck[0][1] is the reason string.
    """
    monkeypatch.setattr("critic.loop.generate",_fake_generate)
    monkeypatch.setattr("critic.loop.check_claim" , _fake_check_claim)
    monkeypatch.setattr("critic.loop.MAX_ROUNDS" , 1)

    kept, struck = run_loop("any topic" , CHUNKS , n = 2)
    assert len(kept) == 1 
    assert len(struck) == 1
    assert kept[0].citations == ["c1"]
    assert struck[0][0].citations == ["c2"]

