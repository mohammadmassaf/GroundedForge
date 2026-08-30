"""
Tracer's own guarantees, tested against the real class.

These deliberately do NOT go through the loops. critic/loop.py's tests replace
Tracer with a recording double, so anything Tracer does in its CONSTRUCTOR is
invisible there -- the double would have to re-implement it to be observed,
which tests the double rather than the code. Writing a real file to tmp_path is
the honest level for that.
"""
import json

from critic.trace import Tracer

# TRACE_DIR is already redirected into tmp_path by the autouse fixture in
# conftest.py, and Tracer exposes the file it chose as .path -- so these tests
# never need to know where it went.


def _lines(tracer) -> list[dict]:
    return [json.loads(l) for l in tracer.path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_trace_opens_by_naming_its_corpus():
    """A trace stores chunk IDs, never chunk text, so resolving them means
    loading chunks/<corpus>.json. Without this line the corpus is inferred from
    the FILENAME (bullets_* => job) and a guess decides whether the IDs resolve
    at all."""
    tracer = Tracer("bullets", corpus="job")

    first = _lines(tracer)[0]
    assert first["event"] == "run_start"
    assert first["corpus"] == "job"


def test_a_run_with_no_corpus_records_that_too():
    """None is written, not omitted. "no corpus" and "the field didn't exist
    yet" are different states, and 222 of the traces on disk are the second --
    a reader has to be able to tell them apart."""
    tracer = Tracer("run")

    first = _lines(tracer)[0]
    assert first["event"] == "run_start"
    assert "corpus" in first
    assert first["corpus"] is None


def test_events_append_after_the_header():
    """run_start is a header, not a replacement: normal logging still appends
    in order behind it."""
    tracer = Tracer("quiz", corpus="networks")
    tracer.log("generated", round=1, count=2)
    tracer.log("done", kept=2, struck=0)

    assert [l["event"] for l in _lines(tracer)] == ["run_start", "generated", "done"]
