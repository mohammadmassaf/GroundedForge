"""
Tests for the demo layer's own logic (app.py).

app.py is deliberately thin -- it calls the same search -> run_loop -> render
path main.py does -- so the only things worth testing here are the decisions it
makes that the engine does not: which questions reach the page, and whether the
cached artifacts render without reaching for a chunk they do not carry.

Importing app runs ensure_store() and reads both cached files, which is itself
the assertion that the Space's boot path works. It must not import torch.
"""
import json
from pathlib import Path

import pytest

import app


# --- distinct(): the duplicate-question filter ------------------------------

def test_a_restated_question_is_dropped():
    """
    The observed pair, from a live run on an off-corpus topic: the pool
    narrowed to one chunk and the Generator asked about it twice. Word order
    differs, so nothing string-based would have caught it.
    """
    items = [
        {"question": "What is the purpose of the TCP PUSH flag?", "answer": "a", "citations": ["c1"]},
        {"question": "What is the purpose of the PUSH flag in TCP?", "answer": "b", "citations": ["c1"]},
    ]
    assert [i["question"] for i in app.distinct(items)] == [items[0]["question"]]


def test_the_first_occurrence_wins():
    """Order matters: the kept one should be the one the Generator produced
    first, not whichever happens to survive a set operation."""
    items = [
        {"question": "How is the UDP checksum calculated?", "answer": "a", "citations": ["c1"]},
        {"question": "How is the checksum calculated for UDP?", "answer": "b", "citations": ["c2"]},
    ]
    kept = app.distinct(items)
    assert len(kept) == 1
    assert kept[0]["answer"] == "a"


def test_different_questions_about_one_field_are_all_kept():
    """
    The failure that would matter more than the duplicate: over-filtering.
    These four are the cached UDP run, which shares heavy vocabulary
    ("UDP", "checksum", "pseudo header") and is NOT repetitive. If the
    threshold ever eats these, the demo silently shows fewer claims than the
    run produced.
    """
    cached = json.loads(Path("samples/cached_quiz.json").read_text(encoding="utf-8"))
    assert len(app.distinct(cached["kept"])) == len(cached["kept"])


def test_an_empty_list_survives():
    """A run where everything was struck reaches here with nothing kept."""
    assert app.distinct([]) == []


@pytest.mark.parametrize("question", [
    "What is the purpose of the TCP PUSH flag?",
    "WHAT IS THE PURPOSE OF THE TCP PUSH FLAG",
    "What is the purpose of the TCP PUSH flag???",
])
def test_case_and_punctuation_do_not_hide_a_duplicate(question):
    """Normalization has to survive the model's formatting whims, or the filter
    passes duplicates that differ only in a question mark."""
    items = [{"question": "What is the purpose of the TCP PUSH flag?", "answer": "a",
              "citations": ["c1"]},
             {"question": question, "answer": "b", "citations": ["c1"]}]
    assert len(app.distinct(items)) == 1


# --- the cached artifacts render --------------------------------------------

def test_the_cached_quiz_renders_with_every_citation_resolved():
    """
    The page's opening state. A citation whose chunk is missing renders as
    "source not in this artifact" -- a claim with no visible evidence, which is
    the one thing this page must never show.
    """
    markdown = app.render_quiz(app.CACHED)
    assert "source not in this artifact" not in markdown
    for item in app.CACHED["kept"]:
        assert item["question"] in markdown
        for cid in item["citations"]:
            assert cid in markdown


def test_the_trap_panel_reports_the_real_number(monkeypatch):
    """5 of 6, not 6 of 6. The panel must print what was measured, including
    the known dt4 escape -- see notes.md Finding 9."""
    markdown = app.render_traps()
    assert f"{app.TRAPS['caught']} of {app.TRAPS['total']}" in markdown
    assert "Not caught" in markdown, "the known escape must stay visible"


def test_every_trap_shows_its_contradicting_evidence():
    """A strike the reader cannot check is an assertion. Each row must carry
    the source quote next to the planted claim."""
    markdown = app.render_traps()
    for trap in app.TRAPS["traps"]:
        assert trap["claim"] in markdown
        assert trap["evidence"][0]["chunk_id"] in markdown


# --- guardrails --------------------------------------------------------------

def test_an_empty_topic_costs_nothing(monkeypatch):
    """A stray Enter on an empty box must not reach Groq."""
    def explode(*a, **k):
        raise AssertionError("generation attempted for an empty topic")
    monkeypatch.setattr(app, "_budget_left", explode)

    status, markdown, last = app.generate("   ", 0.0)
    assert "Type a topic" in status
    assert last == 0.0
    assert markdown == app.render_quiz(app.CACHED)


def test_the_cooldown_serves_the_cached_run(monkeypatch):
    """Inside the cooldown the page still shows something real, and the clock
    is not reset -- otherwise repeated clicking would extend the lockout."""
    import time
    now = time.time()
    status, markdown, last = app.generate("TCP", now - 1)
    assert "One live run every" in status
    assert last == now - 1
    assert markdown == app.render_quiz(app.CACHED)


def test_an_exhausted_budget_serves_the_cached_run(monkeypatch):
    """Constraint 3's endpoint: the demo degrades to the saved run rather than
    erroring, and never by spending a call to find out."""
    monkeypatch.setattr(app, "_budget_left", lambda: 0)
    status, markdown, _ = app.generate("TCP", 0.0)
    assert "resting" in status
    assert markdown == app.render_quiz(app.CACHED)


def test_live_runs_are_capped_below_the_token_cliff():
    """
    n=5 measured 1995 of 2000 output tokens, because gpt-oss-20b bills
    reasoning against the same budget. The public box must sit below that.
    """
    assert app.N_ITEMS < 5


def test_the_app_boots_from_any_working_directory(tmp_path):
    """
    Regression, 2026-09-03: the first deploy died with
    `FileNotFoundError: 'chunks/demo.json'` -- a file that was deployed
    correctly, one directory away.

    Every path in this project is relative to the repo root. main.py gets away
    with that because a CLI runs from the directory you cloned into; HF starts
    app.py from somewhere else. app.py now chdir's to its own location, and
    this asserts it, because nothing else in the suite would: pytest runs from
    the repo root, so the bug is invisible to every other test here.
    """
    import subprocess
    import sys

    repo = Path.cwd().resolve()
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, r'{repo}'); import app; print('BOOTED', len(app.CACHED['kept']))"],
        capture_output=True, text=True, cwd=tmp_path)      # <- deliberately NOT the repo root
    assert "BOOTED" in out.stdout, out.stdout + out.stderr


def test_importing_the_app_does_not_import_torch():
    """
    The whole point of the cached-first design: the page is useful before the
    model exists. If an import creeps back to module scope, the page starts
    waiting ~16s for torch and nothing else fails to warn you.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import app, sys; print('torch' in sys.modules)"],
        capture_output=True, text=True, cwd=".")
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr
