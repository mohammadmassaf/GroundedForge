"""
Loads raw text from a single file.
Returns a list of (page_number, text) tuples.
Page number is 1-indexed for PDFs; always 1 for plain text files.
"""
from pathlib import Path
from pypdf import PdfReader


def load_file(path: str) -> list[tuple[int, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".pdf":
        return _load_pdf(p)
    elif p.suffix.lower() in (".txt", ".md"):
        return _load_text(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")


def _load_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _load_text(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    return [(1, text)]
