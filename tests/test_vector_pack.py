"""
Tests for the committed embeddings and the store rebuilt from them
(index/demo_vectors.npz + retrieve.store.ensure_store).

This is the Space's entire boot path: HF has no build step, so the demo index
does not exist until ensure_store() reconstructs it from the pack. If that is
wrong, the Space serves a page whose every search returns nothing — and an
empty result set does not raise. It reads downstream as "the corpus has nothing
on this", which is a grounding FINDING rather than a broken deployment.

Everything here runs without loading the embedding model, which is also the
property under test: rebuilding must not depend on 90 MB of weights arriving
from the hub, or a cold Space with a slow network is a blank page.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from retrieve.model_pins import EMBEDDING_MODEL, EMBEDDING_REVISION
from retrieve.paths import vector_pack

PACK = Path("index/demo_vectors.npz")
CHUNKS = Path("chunks/demo.json")


@pytest.fixture(scope="module")
def pack():
    return np.load(PACK, allow_pickle=False)


@pytest.fixture(scope="module")
def chunks():
    return json.loads(CHUNKS.read_text(encoding="utf-8"))


# --- which corpora may have a committed pack -------------------------------

def test_only_public_corpora_have_a_pack():
    """
    An embedding is not anonymous. Vectors of vault notes are still derived
    from vault notes, and they ship beside the chunk_ids that name them — so
    the same public/private split that governs the store governs the pack.
    """
    assert vector_pack("demo") == str(Path("index/demo_vectors.npz"))
    for private in ("job", "networks", "default"):
        with pytest.raises(ValueError):
            vector_pack(private)


# --- the pack itself -------------------------------------------------------

def test_pack_covers_exactly_the_committed_chunks(pack, chunks):
    """Same ids, same order. Order matters: the rebuild pairs vectors to text
    positionally, so a permutation would attach every chunk's text to somebody
    else's vector — and retrieval would still return results, just wrong ones."""
    assert [str(i) for i in pack["ids"]] == [c["chunk_id"] for c in chunks]


def test_pack_is_float32_and_the_model_dimension(pack, chunks):
    """all-MiniLM-L6-v2 emits 384 dimensions. A shape mismatch means the pack
    was built with a different model than the one that will embed queries."""
    assert pack["vectors"].dtype == np.float32
    assert pack["vectors"].shape == (len(chunks), 384)


def test_pack_records_the_revision_it_was_built_with(pack):
    """
    The pin travels WITH the vectors, and must match what model_pins says now.

    Mixing revisions is the quiet failure this guards: stored vectors from one
    revision, query vectors from another, every score subtly wrong, nothing
    raised. Finding 5 already paid for the general version of this lesson —
    a model changing under the system without the numbers changing loudly.
    """
    assert str(pack["revision"]) == EMBEDDING_REVISION
    assert str(pack["model"]) == EMBEDDING_MODEL


def test_vectors_are_normalized(pack):
    """
    Sentence-transformers returns unit vectors, and the store is built with
    cosine space. Near-unit norms are a cheap check that the floats survived
    the float32 round-trip into the npz intact.
    """
    norms = np.linalg.norm(pack["vectors"], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"norms range {norms.min()}..{norms.max()}"


# --- the rebuild -----------------------------------------------------------

def test_rebuild_produces_a_queryable_store_without_the_model(tmp_path, monkeypatch):
    """
    The boot path, end to end, into a throwaway directory.

    Queried with a vector taken straight from the pack rather than one the
    model produces: that keeps the test offline and deterministic, and it still
    proves the thing that matters — that a chunk's own vector finds that chunk.
    If ids and documents were misaligned, the nearest neighbour would be
    somebody else.
    """
    import chromadb
    from retrieve import store

    monkeypatch.setattr(store, "chroma_dir", lambda corpus: str(tmp_path))
    assert store.ensure_store("demo") is True

    pack = np.load(PACK, allow_pickle=False)
    ids = [str(i) for i in pack["ids"]]

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("demo")
    assert collection.count() == len(ids)

    probe = 7
    hit = collection.query(query_embeddings=[pack["vectors"][probe].tolist()], n_results=1)
    assert hit["ids"][0][0] == ids[probe]


def test_rebuild_is_idempotent(tmp_path, monkeypatch):
    """app.py calls this unconditionally on startup, so a warm container must
    pay nothing and must not double-insert."""
    import chromadb
    from retrieve import store

    monkeypatch.setattr(store, "chroma_dir", lambda corpus: str(tmp_path))
    assert store.ensure_store("demo") is True
    assert store.ensure_store("demo") is False

    collection = chromadb.PersistentClient(path=str(tmp_path)).get_collection("demo")
    assert collection.count() == len(json.loads(CHUNKS.read_text(encoding="utf-8")))


def test_importing_the_store_module_does_not_import_torch():
    """
    Regression guard. `retrieve.store` imported HuggingFaceEmbeddings at module
    level, which pulls torch in (~8s, ~200MB) before ensure_store() is even
    called — on the one path that needs neither. The heavy imports now live
    inside build(). Moving them back out would silently restore the cost.
    """
    import subprocess
    import sys

    code = "import retrieve.store, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.stdout.strip() == "False", out.stdout + out.stderr
