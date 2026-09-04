"""
Where each corpus's vector store lives.

ONE SOURCE OF TRUTH
-------------------
The persist directory is read by the writer (`store.build`) and by the reader
(`query.search`), and they must agree. A disagreement does not raise: Chroma
happily opens an empty store at the wrong path and `search` returns [], which
reads downstream as "the corpus has nothing about this" — a grounding finding
rather than a configuration bug. So the path is computed here and nowhere else.

WHY THIS IS A FUNCTION OF THE CORPUS, NOT A CONSTANT
----------------------------------------------------
The demo store is COMMITTED, because the Gradio Space has no build step: HF
runs `app.py` and nothing else, so an index that is not in the repo has to be
rebuilt on every cold boot (~34s measured for 526 chunks — see the shipping
plan). Committing it is the only way the page is up in under three seconds.

That makes the path a privacy boundary rather than a preference.
`chroma_db/chroma.sqlite3` is a single 8.9 MB file holding EVERY corpus at
once — including `job` (vault notes) and `networks` (course PDFs under
copyright). It can never be committed, and one mistyped env var pointing a
`build-index --corpus job` at the public directory would put vault text in a
public repo with no error and no obvious symptom.

Keying the directory off the corpus name removes the chance to get it wrong:
`demo` writes to the committed store, everything else writes to the gitignored
one, and no argument, flag or environment variable can swap them. Same reasoning
as `belongs_to()` deciding a corpus by whether ids resolve rather than by
reading a filename, and as conftest redirecting TRACE_DIR rather than asking
each test to remember.
"""
import os
from pathlib import Path

# Where both stores live, relative to the working directory by default.
#
# GF_STORE_ROOT relocates them for tests. Note what it CANNOT do: it moves the
# public and private directories together, so the split below survives any value
# it is given. That is the whole reason it is a root and not a per-corpus
# override -- an override could aim `job` at the public directory, which is the
# one mistake this module exists to make impossible.
#
# It exists because the test suite was opening the REAL demo store: importing
# app.py runs _boot(), and two pytest runs at once then contended on one sqlite
# file -- a 22-second suite took 51 minutes. Same shape as the traces/ problem
# conftest already fixes: measurement state and test state sharing a directory.
STORE_ROOT_ENV = "GF_STORE_ROOT"

# Everything except the demo. Gitignored, holds all corpora in one sqlite file.
PRIVATE_CHROMA_DIR = Path("chroma_db")

# The demo's store. ALSO gitignored, despite being public: Chroma rewrites its
# HNSW files and sqlite pages on every read, so committing it meant an ordinary
# query left 4.7 MB of binary diff behind. What ships instead is the vector
# pack below, and this directory is rebuilt from it -- see ensure_store().
PUBLIC_CHROMA_DIR = Path("chroma_demo")

# Committed, immutable: the embeddings themselves. Same chunks + same pinned
# model revision = same floats, so this file changes only on a re-ingest.
VECTOR_PACK_DIR = Path("index")

# Corpora whose store is committed. A set, not a bool on the corpus, because
# the question "is this public?" is answered here rather than in nine call sites.
PUBLIC_CORPORA = {"demo"}


def store_root() -> Path:
    """
    The parent of both stores. Read on every call, not captured at import, so a
    test can set it after this module is already loaded.
    """
    return Path(os.environ.get(STORE_ROOT_ENV, "."))


def chroma_dir(corpus: str) -> str:
    """
    The persist directory for `corpus`, as the string Chroma wants.

    The public store for `demo`, the private one for everything else. There is
    deliberately no per-corpus override -- only the shared root can move, and
    moving it takes both directories along, so `job` can never resolve to the
    directory the demo publishes from.
    """
    directory = PUBLIC_CHROMA_DIR if corpus in PUBLIC_CORPORA else PRIVATE_CHROMA_DIR
    return str(store_root() / directory)


def vector_pack(corpus: str) -> str:
    """
    The committed embeddings for `corpus`, as `index/<corpus>_vectors.npz`.

    Only public corpora have one, and only public corpora may: the pack is
    committed, and an embedding is not anonymous -- vectors of vault notes are
    still derived from vault notes, and they sit next to the chunk_ids that
    name them.
    """
    if corpus not in PUBLIC_CORPORA:
        raise ValueError(
            f"{corpus!r} is not a public corpus; its embeddings must not be committed"
        )
    return str(VECTOR_PACK_DIR / f"{corpus}_vectors.npz")
