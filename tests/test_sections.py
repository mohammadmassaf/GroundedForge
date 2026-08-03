"""
Unit tests for the shared heading splitter (ingest/adapters/sections.py).

Pure string logic, no I/O. Both docs_adapter and vault_adapter chunk through
this, so a bug here changes the shape of two thirds of the job corpus at once.

Note on offsets: char_start/char_end are computed per LINE and include the
newline that split("\\n") removed, so a slice of the original text carries a
trailing newline the section's own text doesn't. The round-trip test compares
stripped, which is the contract that actually holds.
"""
from ingest.adapters.sections import split_sections


def test_single_heading_becomes_one_section():
    """A heading plus its body is one section, titled by the heading text."""
    sections = split_sections("## Setup\ninstall the thing\nthen run it")

    assert len(sections) == 1
    assert sections[0]["title"] == "Setup"
    assert sections[0]["text"] == "## Setup\ninstall the thing\nthen run it"


def test_each_heading_starts_a_new_section():
    """Consecutive headings cut the text; the heading line belongs to the
    section it opens, not the one it closes."""
    sections = split_sections("# One\nalpha\n# Two\nbeta")

    assert [s["title"] for s in sections] == ["One", "Two"]
    assert sections[0]["text"] == "# One\nalpha"
    assert sections[1]["text"] == "# Two\nbeta"


def test_text_before_the_first_heading_is_intro():
    """Preamble can't be dropped -- a README's opening paragraph is often the
    most descriptive text in the file."""
    sections = split_sections("some preamble\n# Real Heading\nbody")

    assert sections[0]["title"] == "(intro)"
    assert sections[0]["text"] == "some preamble"
    assert sections[1]["title"] == "Real Heading"


def test_file_without_headings_is_one_intro_section():
    """A heading-less note still produces a chunk rather than vanishing."""
    sections = split_sections("just prose\nand more prose")

    assert len(sections) == 1
    assert sections[0]["title"] == "(intro)"


def test_heading_level_does_not_matter():
    """#### is as much a section boundary as #; the title drops all hashes."""
    sections = split_sections("#### Deep\nbody")

    assert sections[0]["title"] == "Deep"


def test_offsets_locate_the_section_in_the_original_text():
    """char_start/char_end are provenance -- they must actually point at the
    section, or a citation can't be traced back to its source."""
    text = "# One\nalpha\n# Two\nbeta"
    sections = split_sections(text)

    for section in sections:
        located = text[section["char_start"]:section["char_end"]]
        assert located.strip() == section["text"].strip()


def test_blank_sections_are_dropped():
    """A blank line between frontmatter and the first heading would otherwise
    become an empty chunk -- pure retrieval noise with a real chunk_id."""
    sections = split_sections("\n\n# Heading\nbody")

    assert [s["title"] for s in sections] == ["Heading"]


def test_empty_text_yields_nothing():
    """No content, no chunks -- and no crash."""
    assert split_sections("") == []
