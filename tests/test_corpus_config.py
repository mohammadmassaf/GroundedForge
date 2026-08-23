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


ROOTS_CONFIG = """\
roots:
  repos: ..
  vault: ../../myvault.obsd

corpora:
  job:
    - type: git
      path: ${repos}/mealwise
      repo: mealwise
    - type: docs
      path: ${repos}/mealwise
      repo: mealwise
      chunk_size: 800
    - type: vault
      repo: grounded-forge
      paths:
        - ${vault}/notes/one.md
        - ${vault}/notes/two.md
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


# --- ${root} expansion -------------------------------------------------------
#
# The config names locations symbolically and a `roots:` block says where they
# actually are, so the same corpus definition works on this laptop and inside a
# container that mounts the vault at /mnt/vault. Config states intent, the
# environment supplies the addresses -- the same split already made for
# GROQ_API_KEY, applied to paths.


def test_root_expands_in_a_path(tmp_path):
    """The plain case: one placeholder, one string field."""
    sources = load_corpus("job", _config(tmp_path, ROOTS_CONFIG))

    assert sources[0]["path"] == "../mealwise"


def test_root_expands_inside_a_list_of_paths(tmp_path):
    """The vault adapter takes `paths` (a list), not `path`. Expansion walks
    the parsed structure rather than a list of known field names, so a string
    nested in a list is reached without the loader knowing what `paths` is --
    the case a field-by-field implementation silently misses."""
    sources = load_corpus("job", _config(tmp_path, ROOTS_CONFIG))

    assert sources[2]["paths"] == [
        "../../myvault.obsd/notes/one.md",
        "../../myvault.obsd/notes/two.md",
    ]


def test_one_root_serves_every_source_that_names_it(tmp_path):
    """The whole point of the block: the environmental assumption is written
    once and overridden in one place, not repeated per entry."""
    sources = load_corpus("job", _config(tmp_path, ROOTS_CONFIG))

    assert sources[0]["path"] == sources[1]["path"] == "../mealwise"


def test_config_without_roots_is_unchanged(tmp_path):
    """Regression guard: study mode's config has no roots and no placeholders,
    and must behave exactly as it did before expansion existed. Ordinary
    strings pass through substitution untouched."""
    sources = load_corpus("job", _config(tmp_path))

    assert sources[0]["path"] == "../mealwise"
    assert sources[1]["exclude"] == ["CLAUDE.md"]


def test_unknown_root_exits_naming_it_and_the_defined_roots(tmp_path):
    """An unresolved placeholder is a request that could not be filled -- if it
    passed through, `${valut}` would reach Path() and surface three layers down
    as a file-not-found naming a path that looks like nothing. The message is
    the feature: it names the typo AND what was available."""
    broken = ROOTS_CONFIG.replace("${vault}/notes/one.md", "${valut}/notes/one.md")

    with pytest.raises(SystemExit) as excinfo:
        load_corpus("job", _config(tmp_path, broken))

    message = str(excinfo.value)
    assert "valut" in message
    assert "vault" in message and "repos" in message


def test_non_string_values_survive_expansion(tmp_path):
    """Expansion is a substitution, not a schema check. A numeric option has no
    placeholder to resolve, so it is not an error -- and it must come back as
    itself rather than falling off the end of the branch chain as None."""
    sources = load_corpus("job", _config(tmp_path, ROOTS_CONFIG))

    assert sources[1]["chunk_size"] == 800
