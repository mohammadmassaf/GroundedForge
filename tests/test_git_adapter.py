"""
Unit tests for commit parsing (ingest/adapters/git_adapter.py::load_git).

The parser is the risky part, not the subprocess call. So `_run_git_log` is
monkeypatched with canned output and the real `git` is never invoked -- tests
that shell out depend on the repo's actual history and start failing the moment
someone commits.

The canned strings mirror the real format the adapter asks for:

    format:%x00%h%n%ad%n%s%n%b   then a blank line, then --numstat rows

so \\x00 separates commits, the first three lines are sha / date / subject, the
body runs until the blank line, and tab-separated "added deleted path" rows
follow.
"""
from ingest.adapters import git_adapter


SOURCE = {"path": ".", "repo": "mealwise"}


def _log(monkeypatch, raw):
    monkeypatch.setattr(git_adapter, "_run_git_log", lambda repo_path, pretty: raw)


def test_one_commit_becomes_one_chunk(monkeypatch):
    """The chunk unit for git is a whole commit -- no character windowing, so
    a commit is never split mid-message."""
    _log(monkeypatch,
         "\x00abc1234\n2026-07-01\nfeat: add the planner\nwhy it mattered\n\n"
         "3\t1\tapp/main.py\n")

    chunks = git_adapter.load_git(SOURCE)

    assert len(chunks) == 1


def test_citation_identity_is_repo_at_sha(monkeypatch):
    """`mealwise@abc1234` is what a reader sees next to a claim, so it is both
    the chunk_id and the source_file."""
    _log(monkeypatch, "\x00abc1234\n2026-07-01\nfeat: add it\nbody\n\n1\t0\tf.py\n")

    chunk = git_adapter.load_git(SOURCE)[0]

    assert chunk["chunk_id"] == "mealwise@abc1234"
    assert chunk["source_file"] == "mealwise@abc1234"
    assert chunk["source_type"] == "git"


def test_metadata_carries_repo_sha_and_date(monkeypatch):
    """These reach ChromaDB via store._chunk_metadata and are what a `where=`
    filter can select on -- --repo on make-bullets depends on this."""
    _log(monkeypatch, "\x00abc1234\n2026-07-01\nfeat: add it\nbody\n\n1\t0\tf.py\n")

    chunk = git_adapter.load_git(SOURCE)[0]

    assert chunk["repo"] == "mealwise"
    assert chunk["sha"] == "abc1234"
    assert chunk["date"] == "2026-07-01"


def test_text_holds_message_and_change_summary(monkeypatch):
    """The embedded text is subject + body + what changed. Commit subjects are
    terse, so the file list is often the only place a retrievable term like a
    module name appears."""
    _log(monkeypatch,
         "\x00abc1234\n2026-07-01\nfeat: add the planner\nwhy it mattered\n\n"
         "3\t1\tapp/main.py\n2\t0\tapp/util.py\n")

    text = git_adapter.load_git(SOURCE)[0]["text"]

    assert "feat: add the planner" in text
    assert "why it mattered" in text
    assert "Changed: app/main.py, app/util.py (+5/-1)" in text


def test_multiline_body_with_blank_lines_is_kept(monkeypatch):
    """The parser splits body from numstat on the LAST blank line, so a body
    containing its own blank lines survives -- this is the classic git-log
    parsing bug."""
    _log(monkeypatch,
         "\x00abc1234\n2026-07-01\nfix: the thing\nfirst para\n\nsecond para\n\n"
         "1\t1\tf.py\n")

    text = git_adapter.load_git(SOURCE)[0]["text"]

    assert "first para" in text
    assert "second para" in text
    assert "Changed: f.py (+1/-1)" in text


def test_binary_files_count_as_zero(monkeypatch):
    """--numstat prints "-\\t-" for binaries. int("-") raises, so this is a
    crash the first time anyone commits an image."""
    _log(monkeypatch,
         "\x00abc1234\n2026-07-01\nfeat: add a logo\nbody\n\n"
         "-\t-\tlogo.png\n4\t2\tapp/main.py\n")

    text = git_adapter.load_git(SOURCE)[0]["text"]

    assert "Changed: logo.png, app/main.py (+4/-2)" in text


def test_several_commits_each_become_a_chunk(monkeypatch):
    """\\x00 is the record separator; commit messages can contain anything
    else, which is why a control character was chosen."""
    _log(monkeypatch,
         "\x00aaa1111\n2026-07-01\nfeat: one\nbody\n\n1\t0\ta.py\n"
         "\x00bbb2222\n2026-07-02\nfeat: two\nbody\n\n2\t0\tb.py\n")

    chunks = git_adapter.load_git(SOURCE)

    assert [c["sha"] for c in chunks] == ["aaa1111", "bbb2222"]


def test_empty_log_yields_no_chunks(monkeypatch):
    """An empty repo, or one with only merge commits, is valid input."""
    _log(monkeypatch, "")

    assert git_adapter.load_git(SOURCE) == []
