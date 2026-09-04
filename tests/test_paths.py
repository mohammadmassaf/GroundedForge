"""
Unit tests for the vector-store path rule (retrieve/paths.py).

This is a five-line module and it still gets tests, because what it decides is
not a preference but a privacy boundary: chroma_demo/ is COMMITTED, and
chroma_db/ holds the job corpus (vault notes) and networks (course PDFs under
copyright) in one sqlite file. A bug that sends a private corpus to the public
directory does not raise, does not look wrong in `git status` (a 9 MB binary
among other binaries), and is discovered by a stranger reading the repo.

Deterministic and offline, like every other test here — the function is a dict
lookup on a string. No model, no store, nothing touched on disk.
"""
from pathlib import Path

import pytest

from retrieve.paths import (chroma_dir, store_root, PUBLIC_CHROMA_DIR,
                            PRIVATE_CHROMA_DIR, PUBLIC_CORPORA)


def test_demo_uses_the_committed_store():
    """The whole point of the split: the Space boots against a store that is
    already in the repo, because the Gradio SDK has no build step."""
    assert chroma_dir("demo") == str(store_root() / PUBLIC_CHROMA_DIR)


@pytest.mark.parametrize("corpus", ["job", "networks", "default"])
def test_private_corpora_use_the_gitignored_store(corpus):
    """job carries vault notes, networks carries copyrighted course PDFs, and
    default is whatever is in data/. None of the three may resolve to the
    committed directory."""
    assert chroma_dir(corpus) == str(store_root() / PRIVATE_CHROMA_DIR)


def test_an_unknown_corpus_defaults_to_private():
    """Fail closed. A corpus added later — a second job corpus, a client's
    documents — must land in the gitignored store until someone deliberately
    adds it to PUBLIC_CORPORA. The dangerous default is the public one."""
    assert chroma_dir("some-corpus-invented-next-month") == str(store_root() / PRIVATE_CHROMA_DIR)


def test_the_two_stores_are_never_the_same_directory():
    """If these ever collapse to one path, every guarantee above is void and
    every test here passes anyway — each would still equal the other."""
    assert PUBLIC_CHROMA_DIR != PRIVATE_CHROMA_DIR


def test_only_demo_is_public():
    """Pins the membership list itself. Adding a corpus to PUBLIC_CORPORA means
    committing its embeddings, so it should be a deliberate edit that breaks a
    test and makes someone think, not a one-word change that slips through."""
    assert PUBLIC_CORPORA == {"demo"}


def test_moving_the_root_moves_both_stores_together(monkeypatch):
    """
    The property that makes GF_STORE_ROOT safe to have at all.

    A per-corpus override could aim `job` at the directory the demo publishes
    from. A shared root cannot: whatever it is set to, the public and private
    directories move together and stay distinct, so the split survives any
    value -- including a mistyped one.
    """
    monkeypatch.setenv("GF_STORE_ROOT", "/tmp/somewhere-else")
    public, private = chroma_dir("demo"), chroma_dir("job")
    assert public != private
    assert public.startswith(str(Path("/tmp/somewhere-else")))
    assert private.startswith(str(Path("/tmp/somewhere-else")))


def test_the_root_defaults_to_the_working_directory(monkeypatch):
    """Unset, the stores sit where the app and the CLI expect them: beside the
    repo. The env var is a test affordance, not the normal path."""
    monkeypatch.delenv("GF_STORE_ROOT", raising=False)
    assert chroma_dir("demo") == str(Path(".") / PUBLIC_CHROMA_DIR)
