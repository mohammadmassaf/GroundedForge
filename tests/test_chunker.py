"""
Unit tests for the chunker (ingest/chunker.py).

Pure logic, no I/O: given known page text, do the chunk boundaries obey
the overlap contract, cover the whole text, and attach right metadata?
"""
from ingest.chunker import chunk_pages, _chunk_text



pages = [(3, "Transmission occurs between transmitter and receiver.")]
source_file = "I2208-Part-1.pdf"

# --- _chunk_text boundaries ------------------------------------------------

def test_overlap_between_consecutive_chunks():
    """TODO(you): use the docstring's own example — text of 20 chars,
    chunk_size=10, overlap=3. Assert _chunk_text returns
    [(0,10),(7,17),(14,20)] (the overlap is start[n+1] == end[n]-overlap)."""

    result = _chunk_text("ABCDEFGHIJKLMNOPQRST", chunk_size=10, overlap=3)
    assert result == [(0, 10), (7, 17), (14, 20)]
    


def test_last_chunk_reaches_end():
    """TODO(you): assert the final tuple's char_end equals len(text) — the
    last chunk always runs to the end, even if shorter than chunk_size."""
    result = _chunk_text("ABCDEFGHIJKLMNOPQRST", chunk_size=10, overlap=3)
    assert result[-1][1] == 20


def test_short_text_is_one_chunk():
    """TODO(you): text shorter than chunk_size -> exactly one chunk (0, len)."""
    assert _chunk_text("ABC", chunk_size=10, overlap=3) == [(0, 3)]


def test_empty_text_no_chunks():
    """TODO(you): empty string -> empty list (no zero-length chunks emitted)."""
    assert _chunk_text("", chunk_size=10, overlap=3) == []


# --- chunk_pages metadata --------------------------------------------------

def test_chunk_pages_attaches_metadata():
    """TODO(you): pass pages=[(3, "some page text")] and a source_file.
    Assert each returned chunk dict has the right page, source_file, a
    chunk_id built from them, and text that matches text[char_start:char_end]."""
    chunks = chunk_pages(pages,source_file)
    assert len(chunks) ==  1
    assert chunks[0]["chunk_id"] =="I2208-Part-1_p3_c0"
    assert chunks[0]["source_file"]	== "I2208-Part-1.pdf"
    assert chunks[0]["page"] ==	3
    assert chunks[0]["text"] == pages[0][1][chunks[0]["char_start"]:chunks[0]["char_end"]]