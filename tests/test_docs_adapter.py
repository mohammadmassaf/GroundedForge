"""
Unit tests for doc discovery (ingest/adapters/docs_adapter.py::_find_doc_files).

NEW TECHNIQUE HERE: `tmp_path`. It's a pytest built-in fixture -- ask for it as
an argument and pytest hands you a fresh empty directory (a pathlib.Path),
unique per test, deleted afterwards. That's how you test filesystem code
without touching the repo and without tests leaking into each other.

The alternative -- pointing tests at the real repo -- makes them fail the day
someone adds a file, which is a test that reports the wrong thing.
"""
from ingest.adapters.docs_adapter import _find_doc_files


def _repo(tmp_path, *names):
    """Create empty files (nested paths allowed) and return the repo root."""
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# doc\nbody", encoding="utf-8")
    return tmp_path


def test_finds_readme_and_claude_md(tmp_path):
    """The two root docs the adapter knows about by name."""
    root = _repo(tmp_path, "README.md", "CLAUDE.md")

    found = {p.name for p in _find_doc_files(str(root))}

    assert found == {"README.md", "CLAUDE.md"}


def test_finds_markdown_under_docs(tmp_path):
    """docs/*.md is globbed, and sorted so chunk_ids are stable between runs --
    an unstable order would renumber chunk_ids and break saved citations."""
    root = _repo(tmp_path, "docs/zeta.md", "docs/alpha.md")

    found = [p.name for p in _find_doc_files(str(root))]

    assert found == ["alpha.md", "zeta.md"]


def test_missing_files_are_skipped(tmp_path):
    """A repo with no CLAUDE.md and no docs/ is normal, not an error."""
    root = _repo(tmp_path, "README.md")

    assert [p.name for p in _find_doc_files(str(root))] == ["README.md"]


def test_excluded_file_is_not_ingested(tmp_path):
    """corpus.yaml can drop a file by name. CLAUDE.md is instructions to an
    agent about how to work, not a record of work done -- its sections were
    topping the situation/task/result pools for every question, including ones
    the corpus cannot answer at all."""
    root = _repo(tmp_path, "README.md", "CLAUDE.md")

    found = {p.name for p in _find_doc_files(str(root), exclude={"CLAUDE.md"})}

    assert found == {"README.md"}


def test_exclude_also_applies_under_docs(tmp_path):
    """The filter must cover the globbed files too, not just the two named
    ones -- otherwise excluding a name works at the root and silently fails a
    directory down."""
    root = _repo(tmp_path, "docs/keep.md", "docs/drop.md")

    found = {p.name for p in _find_doc_files(str(root), exclude={"drop.md"})}

    assert found == {"keep.md"}


