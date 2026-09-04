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


# --- saying so when the corpus never saw the question ----------------------

@pytest.mark.parametrize("topic", [
    "TCP connection establishment and the three-way handshake",
    "UDP checksum and the pseudo header",
    "IP fragmentation and reassembly",
    "The Time to Live field",
    "TCP sequence numbers",
])
def test_a_covered_topic_raises_no_notice(topic):
    """
    The direction that must not produce false positives. A warning on a topic
    the RFCs genuinely cover would undercut the whole page: it tells a reader
    the answer is off-target when it is not.

    "The Time to Live field" is here deliberately. It scores 0.301 on vector
    similarity, LOWER than off-corpus "BGP route reflection" at 0.444, which is
    why the notice cannot be built on a similarity floor.
    """
    assert app.unknown_terms(topic) == []
    assert app._subject_changed(topic) == ""


@pytest.mark.parametrize("topic,expected", [
    ("HTTP/2 server push and header compression", {"http/2", "compression"}),
    ("Kubernetes pod autoscaling", {"kubernetes", "pod", "autoscaling"}),
    ("photosynthesis in C4 plants", {"photosynthesis", "c4", "plants"}),
])
def test_an_uncovered_topic_names_the_missing_words(topic, expected):
    """Names the words rather than asserting a verdict, so a reader can check
    the claim: these are absent from RFC 768, 791 and 793."""
    assert set(app.unknown_terms(topic)) == expected
    notice = app._subject_changed(topic)
    assert "contains nothing about" in notice
    for word in expected:
        assert word in notice


def test_the_notice_survives_words_the_corpus_shares_with_the_question():
    """
    The case a score-based test gets wrong. "React hooks and state management"
    scores 7.54 on BM25 because `state` and `management` are TCP vocabulary, so
    a keyword floor would call it covered. Membership still catches `react` and
    `hooks`.
    """
    missing = app.unknown_terms("React hooks and state management")
    assert set(missing) == {"react", "hooks"}


def test_stopwords_are_never_reported_missing():
    """A notice reading "contains nothing about the, and" would be absurd and
    would fire on every topic."""
    assert app.unknown_terms("what is the purpose of the checksum") == []


def test_source_text_is_escaped_before_it_reaches_the_page():
    """
    RFC 793 is full of `<SEQ=100><ACK=301><CTL=SYN,ACK>` and RFC 791 of `+-+-+`
    header diagrams. Rendered as HTML without escaping, a browser reads those
    angle brackets as tags and silently swallows the text between them -- so the
    evidence quote, the one thing a reader is invited to verify, disappears.

    Escaping is also the injection guard, but the corpus breaks this on its own
    without anyone being hostile.
    """
    payload = {
        "chunks": [{"chunk_id": "c1", "source_file": "rfc793.txt", "page": 37,
                    "text": "2. SYN-SENT --> <SEQ=100><CTL=SYN> --> SYN-RECEIVED & more"}],
        "kept": [{"question": "What does <SYN> do?", "answer": "A & B <tag>",
                  "citations": ["c1"]}],
        "struck": [],
    }
    out = app.render_quiz(payload)
    assert "&lt;SEQ=100&gt;" in out
    assert "<SEQ=100>" not in out
    assert "&amp; more" in out
    assert "&lt;SYN&gt;" in out and "<SYN>" not in out


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
