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
    """Consecutive chunks overlap by `overlap` characters:
    start[n+1] == end[n] - overlap. Overlap exists so a sentence spanning a
    boundary survives in at least one chunk intact."""

    result = _chunk_text("ABCDEFGHIJKLMNOPQRST", chunk_size=10, overlap=3)
    assert result == [(0, 10), (7, 17), (14, 20)]
    


def test_last_chunk_reaches_end():
    """The last chunk always runs to the end of the text, even when
    shorter than chunk_size. No trailing characters are silently dropped."""
    result = _chunk_text("ABCDEFGHIJKLMNOPQRST", chunk_size=10, overlap=3)
    assert result[-1][1] == 20


def test_short_text_is_one_chunk():
    """Text shorter than chunk_size is one chunk, not a padded one."""
    assert _chunk_text("ABC", chunk_size=10, overlap=3) == [(0, 3)]


def test_empty_text_no_chunks():
    """An empty string yields no chunks at all - a zero-length chunk would
    embed to a meaningless vector and could be retrieved and cited."""
    assert _chunk_text("", chunk_size=10, overlap=3) == []


# --- chunk_pages metadata --------------------------------------------------

def test_chunk_pages_attaches_metadata():
    """Each chunk carries the page and source_file it came from, a
    chunk_id built from both, and text matching its own char_start:char_end
    span. This is the metadata every citation is later traced through."""
    chunks = chunk_pages(pages,source_file)
    assert len(chunks) ==  1
    assert chunks[0]["chunk_id"] =="I2208-Part-1_p3_c0"
    assert chunks[0]["source_file"]	== "I2208-Part-1.pdf"
    assert chunks[0]["page"] ==	3
    assert chunks[0]["text"] == pages[0][1][chunks[0]["char_start"]:chunks[0]["char_end"]]