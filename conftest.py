# Its presence puts the repo root on sys.path so tests can
# `from generate... import` / `from critic... import` without PYTHONPATH.
import pytest


@pytest.fixture(autouse=True)
def _no_real_traces(tmp_path, monkeypatch):
    """
    No test may write into the real traces/ directory. Autouse, so it applies
    whether or not a test remembers to patch Tracer.

    This is not tidiness. traces/ is the DATASET the strike analysis reads, and
    test_loop_integration.py ran the real Tracer with canned content
    ("question 1", citations=["c2"], "not in evidence"), which is
    indistinguishable from a real strike to anything sweeping traces/*.jsonl.
    81 such files had accumulated -- 28 of them carrying a pool, so they landed
    in the ATTRIBUTABLE bucket and made the real count read 62 instead of 36.
    Worse, the number grew by one on every pytest run, so it was not even
    stable between two invocations.

    Patching one test file would have fixed today's leak and left the next one
    to be discovered the same way. Redirecting the directory itself closes the
    class of bug: measurement input cannot be contaminated by the test suite.
    """
    monkeypatch.setattr("critic.trace.TRACE_DIR", tmp_path / "traces")
