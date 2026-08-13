"""
Unit tests for frontmatter stripping (ingest/adapters/vault_adapter.py).

strip_frontmatter decides what a vault note's chunks are made of. Get it wrong
in one direction and a YAML block becomes an embedded chunk (retrieval noise
that matches nothing); wrong in the other and a malformed note loses its whole
body.

It returns (meta, body): the parsed YAML as a dict, and everything after the
closing fence. load_vault lifts `type` and `status` out of meta into chunk
metadata, so the dict is not decoration -- it drives what you can filter on.
"""
from ingest.adapters.vault_adapter import strip_frontmatter, load_vault


NOTE = """---
type: project
status: active
---
# Real content
body text
"""


def test_frontmatter_is_removed_from_the_body():
    """The YAML block must not reach the chunker -- it would embed as a chunk
    of key/value noise sitting at the top of every note."""
    _, body = strip_frontmatter(NOTE)

    assert body.startswith("# Real content")
    assert "type: project" not in body


def test_frontmatter_fields_are_parsed_out():
    """meta feeds chunk metadata, which is what `where=` filters on later."""
    meta, _ = strip_frontmatter(NOTE)

    assert meta["type"] == "project"
    assert meta["status"] == "active"


def test_note_without_frontmatter_is_untouched():
    """Not every note has a block; the body must survive verbatim."""
    text = "# Just a heading\nand body"
    meta, body = strip_frontmatter(text)

    assert meta == {}
    assert body == text


def test_horizontal_rule_after_the_block_is_not_a_fence():
    """Only the FIRST two `---` lines delimit frontmatter. A later one is a
    Markdown horizontal rule, and treating it as a fence would eat every
    section between the two."""
    text = "---\ntype: project\n---\n# Heading\nintro\n\n---\n\nmore body"
    meta, body = strip_frontmatter(text)

    assert meta == {"type": "project"}
    assert body.startswith("# Heading")
    assert "more body" in body       # nothing after the rule was swallowed
    assert "---" in body             # the rule itself survives


def test_missing_closing_fence_is_treated_as_no_frontmatter():
    """A malformed note keeps its content. The alternative -- consuming the
    whole file looking for a fence that never comes -- loses the note
    entirely, and silently."""
    text = "---\ntype: project\n# never closed\nbody text"
    meta, body = strip_frontmatter(text)

    assert meta == {}
    assert body == text


def test_empty_frontmatter_block_yields_empty_meta():
    """`---\\n---` parses to nothing, not to None -- callers do meta.get(...)
    and would crash on None."""
    meta, body = strip_frontmatter("---\n---\n# Heading\nbody")

    assert meta == {}
    assert body.startswith("# Heading")


# --- load_vault: repo scoping ----------------------------------------------
#
# git and docs entries already carry `repo`; vault notes did not, so they
# belonged to no project. That is the worst of both worlds: a repo-filtered
# retrieval drops them, and an unfiltered one lets a Grounded Forge planning
# note answer a question about MealWise.

def _note(tmp_path, name="grounded-forge.md"):
    p = tmp_path / name
    p.write_text("---\ntype: project\n---\n# Deliverable\nship the thing\n"
                 "## Scope\nstudy mode only\n", encoding="utf-8")
    return p


def test_repo_reaches_every_chunk_from_the_note(tmp_path):
    """One `repo:` on the source config, not per note path -- and it has to
    land on every section, since retrieval filters chunk by chunk."""
    p = _note(tmp_path)

    chunks = load_vault({"paths": [str(p)], "repo": "grounded-forge"})

    assert len(chunks) > 1
    assert all(c["repo"] == "grounded-forge" for c in chunks)


def test_repo_is_absent_when_the_config_omits_it(tmp_path):
    """Optional, unlike the git/docs adapters where it is required. A note need
    not belong to a project, and store.py drops None-valued metadata anyway --
    so the key must simply not be there rather than be present and empty."""
    p = _note(tmp_path)

    chunks = load_vault({"paths": [str(p)]})

    assert all("repo" not in c for c in chunks)


def test_adding_a_repo_does_not_change_chunk_ids(tmp_path):
    """Citations are chunk_ids, and eval traps cite vault chunks by id. If
    scoping shifted ids, re-ingesting would silently invalidate every stored
    citation -- so ids must depend on the note and its section order alone."""
    p = _note(tmp_path)

    without = [c["chunk_id"] for c in load_vault({"paths": [str(p)]})]
    with_repo = [c["chunk_id"] for c in load_vault({"paths": [str(p)], "repo": "gf"})]

    assert without == with_repo
