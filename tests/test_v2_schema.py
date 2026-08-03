"""
Schema tests for the v2 output types (generate/schema.py).

Same job as test_schema.py does for Quiz/Guide: these are the Pydantic shapes
the Generator's reply must validate against, and _parse_and_validate re-prompts
the model when they fail. So every constraint here is a retry the model is
forced through rather than a bad artifact reaching the reader.

The one that matters most is `citations` being non-empty. A claim with no
citation is precisely what this whole project exists to prevent, and the schema
is the cheapest place to make it impossible.
"""
import pytest
from pydantic import ValidationError

from generate.schema import CVBullet, Bullets, StarSection, STARAnswer


def _section(text="a section of an answer", citations=("c1",)):
    return StarSection(text=text, citations=list(citations))


def _answer(**overrides):
    fields = {name: _section(citations=[f"{name[0]}1"])
              for name in ("situation", "task", "action", "result")}
    fields.update(overrides)
    return STARAnswer(question="an interview question", **fields)


# --- CVBullet / Bullets -----------------------------------------------------

def test_valid_bullet_parses():
    bullet = CVBullet(text="Built JWT authentication", citations=["mealwise@abc1234"])

    assert bullet.citations == ["mealwise@abc1234"]


def test_bullet_needs_a_citation():
    """An uncited bullet is an unsupported claim -- rejected at the shape
    layer, before the Critic ever has to judge it."""
    with pytest.raises(ValidationError):
        CVBullet(text="Built JWT authentication", citations=[])


def test_bullet_text_has_a_floor():
    """min_length=10 catches a model that emits a stub like "auth" and calls
    it a resume bullet."""
    with pytest.raises(ValidationError):
        CVBullet(text="auth", citations=["c1"])


def test_empty_bullet_list_rejected():
    """Zero bullets means the generation failed; it must not validate as a
    successful empty artifact."""
    with pytest.raises(ValidationError):
        Bullets(bullets=[])


def test_bullets_all_citations_flattens():
    """render_bullets numbers the sources off this list."""
    bullets = Bullets(bullets=[
        CVBullet(text="Built the auth layer", citations=["c1", "c2"]),
        CVBullet(text="Added the meal planner", citations=["c3"]),
    ])

    assert bullets.all_citations() == ["c1", "c2", "c3"]


# --- StarSection / STARAnswer ----------------------------------------------

def test_valid_star_answer_parses():
    answer = _answer()

    assert answer.action.citations == ["a1"]


def test_star_section_needs_a_citation():
    """Per-section citations are what the scope check in run_star_loop tests
    against; an empty list would make that check vacuous for that section."""
    with pytest.raises(ValidationError):
        StarSection(text="what I did on the project", citations=[])


def test_all_four_sections_are_required():
    """A STAR answer missing its Result isn't a STAR answer. This is also why
    run_star_loop FLAGS a bad section instead of dropping it -- the schema
    leaves it nowhere to go."""
    with pytest.raises(ValidationError):
        STARAnswer(question="an interview question",
                   situation=_section(), task=_section(), action=_section())


def test_sections_are_returned_in_star_order():
    """run_star_loop lowercases these names to index the evidence pools, so the
    order and the spelling are load-bearing, not cosmetic."""
    names = [name for name, _ in _answer().sections()]

    assert names == ["Situation", "Task", "Action", "Result"]


def test_star_all_citations_flattens_in_section_order():
    """render_star numbers sources off this, so a reader's [1] must be the
    Situation's first citation."""
    assert _answer().all_citations() == ["s1", "t1", "a1", "r1"]
