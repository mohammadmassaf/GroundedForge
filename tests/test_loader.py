"""
Unit tests for file loading (ingest/loader.py).

load_file returns [(page_number, text)]. The page number is what a citation
shows the reader, so getting it wrong doesn't corrupt retrieval -- it makes a
claim untraceable, which for this project is the same as unsupported.
"""
import pytest

from ingest.loader import load_file


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_plain_text_is_one_page(tmp_path):
    """No page breaks, no pagination to infer."""
    path = _write(tmp_path, "notes.txt", "line one\nline two")

    assert load_file(path) == [(1, "line one\nline two")]


def test_form_feeds_become_pages(tmp_path):
    """Paginated plain text (IETF RFCs use \\f) keeps page granularity, so a
    citation points at a page rather than at a 90-page file."""
    path = _write(tmp_path, "rfc.txt", "page one\fpage two\fpage three")

    pages = load_file(path)

    assert [n for n, _ in pages] == [1, 2, 3]
    assert pages[1][1] == "page two"


def test_blank_pages_are_dropped_but_numbering_holds(tmp_path):
    """A trailing form feed shouldn't emit an empty page -- and the surviving
    pages must keep their ORIGINAL numbers, or every citation after a blank
    page silently points one page early."""
    path = _write(tmp_path, "rfc.txt", "page one\f   \fpage three\f")

    pages = load_file(path)

    assert [n for n, _ in pages] == [1, 3]


def test_markdown_is_loaded_as_text(tmp_path):
    """.md goes through the same path as .txt."""
    path = _write(tmp_path, "readme.md", "# Heading\nbody")

    assert load_file(path) == [(1, "# Heading\nbody")]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(str(tmp_path / "absent.txt"))


def test_unsupported_extension_raises(tmp_path):
    """Fail loudly rather than silently ingesting nothing."""
    path = _write(tmp_path, "data.csv", "a,b\n1,2")

    with pytest.raises(ValueError):
        load_file(path)
