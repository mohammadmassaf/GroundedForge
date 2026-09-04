# Its presence puts the repo root on sys.path so tests can
# `from generate... import` / `from critic... import` without PYTHONPATH.
import atexit
import os
import shutil
import tempfile

import pytest

from retrieve.paths import STORE_ROOT_ENV

# --- no test may open the real vector store --------------------------------
#
# Same lesson as the traces/ fixture below, and found the same way: importing
# app.py runs _boot(), which opens the demo store, so two pytest runs at once
# had several processes on one sqlite file and a 22-second suite took 51
# minutes waiting on locks.
#
# This runs at conftest IMPORT time, not in a fixture, and that is the whole
# point. pytest imports test modules during COLLECTION, before any fixture --
# even a session-scoped autouse one -- has run, and tests/test_app_demo.py
# imports app at module level. A fixture was tried first and the real store was
# still touched; the mtime proved it.
#
# An environment variable rather than monkeypatching, so it reaches the
# SUBPROCESSES two tests spawn: they inherit the environment, not this
# interpreter's patches.
_STORE_ROOT = tempfile.mkdtemp(prefix="gf_test_store_")
os.environ[STORE_ROOT_ENV] = _STORE_ROOT
atexit.register(shutil.rmtree, _STORE_ROOT, ignore_errors=True)


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
