"""
Splits page text into overlapping chunks with metadata attached.

Each chunk is a dict:
{
    "chunk_id":    "Part-1_p3_c0",   # unique: filename + page + chunk index
    "source_file": "I2208-Part-1.pdf",
    "page":        3,
    "char_start":  120,
    "char_end":    740,
    "text":        "..."
}
"""

CHUNK_SIZE = 700    # target characters per chunk  (~500-600 tokens for English prose)
OVERLAP    = 100    # characters shared between consecutive chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    source_file: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[dict]:
    """
    pages       : output of loader.load_file()  [(page_num, text), ...]
    source_file : filename string to embed in metadata
    Returns     : list of chunk dicts
    """
    all_chunks = []

    for page_num, text in pages:
        page_chunks = _chunk_text(text, chunk_size, overlap)

        for idx, (char_start, char_end) in enumerate(page_chunks):
            all_chunks.append({
                "chunk_id":    _make_id(source_file, page_num, idx),
                "source_file": source_file,
                "page":        page_num,
                "char_start":  char_start,
                "char_end":    char_end,
                "text":        text[char_start:char_end],
            })

    return all_chunks


def _make_id(source_file: str, page: int, idx: int) -> str:
    stem = source_file.replace(" ", "_").replace(".pdf", "").replace(".txt", "")
    return f"{stem}_p{page}_c{idx}"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """
    TODO: implement this function.

    Given a string `text`, return a list of (char_start, char_end) tuples
    that slice `text` into overlapping chunks.

    Rules:
    - Each chunk should be roughly `chunk_size` characters long.
    - Consecutive chunks should share `overlap` characters at their boundary
      (the end of chunk N and the start of chunk N+1 overlap by `overlap` chars).
    - The last chunk runs to the end of the text (even if shorter than chunk_size).
    - Don't emit empty chunks (skip if char_start >= len(text)).

    Example with chunk_size=10, overlap=3, text="ABCDEFGHIJKLMNOPQRST" (20 chars):
        chunk 0: (0, 10)   → "ABCDEFGHIJ"
        chunk 1: (7, 17)   → "HIJKLMNOPQ"   ← starts 10-3=7 chars after previous start
        chunk 2: (14, 20)  → "NOPQRST"      ← last chunk hits end of text
    """
    chunks = []
    length = len(text)
    i = 0
    while i < length:
        char_end = min(i + chunk_size, length)
        chunks.append((i, char_end))
        i += chunk_size - overlap
    return chunks
