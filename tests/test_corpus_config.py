"""
Unit tests for corpus.yaml parsing (ingest/corpus_config.py).

Uses tmp_path (see test_docs_adapter for what it is) to write throwaway config
files -- the real corpus.yaml is gitignored and holds personal vault paths, so
tests must never read it.

Most of what's tested here is the FAILURE messages. load_corpus raises
SystemExit rather than returning None, because a typo'd corpus name should stop
the CLI with something a human can act on, not produce an empty ingest that
looks like it worked.
"""
import pytest

from ingest.corpus_config import load_corpus


CONFIG = """\
corpora:
  default:
    - type: files
      path: data
  job:
    - type: git
      path: ../mealwise
      repo: mealwise
    - type: docs
      path: .
      repo: grounded-forge
      exclude: [CLAUDE.md]
"""


def _config(tmp_path, text=CONFIG):
    path = tmp_path / "corpus.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_returns_the_sources_for_a_named_corpus(tmp_path):
    """Each source keeps its own fields; the adapter layer reads them."""
    sources = load_corpus("job", _config(tmp_path))

    assert [s["type"] for s in sources] == ["git", "docs"]
    assert sources[0]["repo"] == "mealwise"


def test_corpora_are_isolated_from_each_other(tmp_path):
    """Asking for one corpus must not leak sources from another -- study mode
    and job mode share a config file but never a chunk set."""
    sources = load_corpus("default", _config(tmp_path))

    assert len(sources) == 1
    assert sources[0]["type"] == "files"


def test_optional_fields_survive_parsing(tmp_path):
    """`exclude` is read by docs_adapter, not by the loader -- the loader must
    pass unknown keys through rather than filtering to a known schema."""
    sources = load_corpus("job", _config(tmp_path))

    assert sources[1]["exclude"] == ["CLAUDE.md"]


def test_unknown_corpus_exits_with_the_known_names(tmp_path):
    """The error names what IS defined -- a typo'd corpus is the likeliest
    reason to hit this, and the fix is visible in the message."""
    with pytest.raises(SystemExit) as excinfo:
        load_corpus("nope", _config(tmp_path))

    assert "nope" in str(excinfo.value)
    assert "default" in str(excinfo.value)


def test_missing_config_file_points_at_the_template(tmp_path):
    """First-run experience: corpus.yaml is gitignored, so a fresh clone has
    none. The message has to name corpus.example.yaml or the user is stuck."""
    with pytest.raises(SystemExit) as excinfo:
        load_corpus("job", str(tmp_path / "absent.yaml"))

    assert "corpus.example.yaml" in str(excinfo.value)


def test_corpus_with_no_sources_exits(tmp_path):
    """An empty corpus would otherwise ingest zero chunks and report success --
    the silent-failure shape this project exists to avoid."""
    path = _config(tmp_path, "corpora:\n  empty:\n")

    with pytest.raises(SystemExit):
        load_corpus("empty", path)
