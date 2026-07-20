"""
The adapter contract for v2 (job mode).

An *adapter* turns one source (a git repo, a folder of docs, a list of vault
notes, a data/ folder of files) into a list of **chunk dicts** — the same shape
the v1 chunker produced, so everything downstream (store -> query -> render)
keeps working unchanged.

Every chunk MUST carry these keys (store.py reads them to build ChromaDB
metadata; renderer.py reads them to show citations):

    chunk_id     : str   unique id, also the ChromaDB id
    source_file  : str   human-readable source (filename, "repo@sha", note name)
    page         : int   page/section index; 1 when not meaningful (e.g. commits)
    char_start   : int   offset of `text` within its source unit
    char_end     : int   offset end
    text         : str   the chunk content that gets embedded
    source_type  : str   which adapter produced it: git | docs | vault | files

Adapters may attach any extra metadata the source has (repo, sha, date,
section, ...). Those extra keys ride along in the chunk dict; from M2 the store
layer passes them into ChromaDB so retrieval can filter on `source_type`.

To add an adapter (M1): write a function that takes a source-config dict and
returns list[chunk], and register it with @register("<type>").
"""
from typing import Callable

# A chunk that reaches the vector store must have all of these.
REQUIRED_KEYS = (
    "chunk_id",
    "source_file",
    "page",
    "char_start",
    "char_end",
    "text",
    "source_type",
)

# An adapter maps a source-config dict -> a list of chunk dicts.
Adapter = Callable[[dict], list[dict]]

_REGISTRY: dict[str, Adapter] = {}


def register(source_type: str) -> Callable[[Adapter], Adapter]:
    """Decorator: register an adapter under a source `type` from corpus.yaml."""
    def decorator(fn: Adapter) -> Adapter:
        _REGISTRY[source_type] = fn
        return fn
    return decorator


def get_adapter(source_type: str) -> Adapter:
    try:
        return _REGISTRY[source_type]
    except KeyError:
        raise ValueError(
            f"No adapter registered for source type '{source_type}'. "
            f"Known types: {sorted(_REGISTRY)}"
        )


def load_source(source: dict) -> list[dict]:
    """Dispatch one corpus.yaml source entry to its adapter."""
    source_type = source.get("type")
    if not source_type:
        raise ValueError(f"Source is missing a 'type' field: {source}")
    chunks = get_adapter(source_type)(source)
    for chunk in chunks:
        validate_chunk(chunk)
    return chunks


def make_chunk(
    *,
    chunk_id: str,
    text: str,
    source_file: str,
    source_type: str,
    page: int = 1,
    char_start: int = 0,
    char_end: int | None = None,
    **metadata,
) -> dict:
    """
    Build a chunk dict with the required keys filled in.

    Adapters call this so they only have to think about *what* a chunk is
    (a commit? a heading section?) and *what metadata* it deserves (repo, sha,
    date, section) — passed as keyword args in **metadata. The bookkeeping keys
    are handled here.
    """
    if char_end is None:
        char_end = len(text)
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "page": page,
        "char_start": char_start,
        "char_end": char_end,
        "text": text,
        "source_type": source_type,
        **metadata,
    }


def validate_chunk(chunk: dict) -> None:
    """Fail loudly if a chunk is missing a required key (catches adapter bugs)."""
    missing = [k for k in REQUIRED_KEYS if k not in chunk]
    if missing:
        raise ValueError(
            f"Chunk missing required key(s) {missing}: "
            f"{ {k: chunk.get(k) for k in ('chunk_id', 'source_type')} }"
        )
