"""
Unit tests for the output schemas (generate/schema.py).

Pure logic, no LLM: does Pydantic accept well-formed output and reject
every malformed shape the cite-or-strike floors are supposed to catch?
"""
import pytest
from pydantic import ValidationError

from generate.schema import Quiz, QuizItem, Guide, GuideSection, GuideClaim


# --- happy paths -----------------------------------------------------------

VALID_QUIZ = {"items": [
    {"question": "What does TCP guarantee?", "answer": "reliable ordered delivery", "citations": ["c1"]}
]}

VALID_GUIDE = {"sections": [
    {"heading": "TCP basics", "claims": [
        {"text": "TCP guarantees ordered delivery", "citations": ["c1"]}
    ]}
]}

MULTI_GUIDE = {"sections": [
    {"heading": "S1", "claims": [
        {"text": "first claim text here", "citations": ["c1", "c2"]},
        {"text": "second claim text here", "citations": ["c3"]},
    ]},
    {"heading": "S2", "claims": [
        {"text": "third claim text here", "citations": ["c4"]},
    ]},
]}   # all_citations() -> ["c1", "c2", "c3", "c4"]  

TWO_ITEM_QUIZ = {"items": [
    {"question": "first long enough question", "answer": "a", "citations": ["c1", "c2"]},
    {"question": "second long enough question", "answer": "b", "citations": ["c3"]}
]}   

NO_ITEMS        = {"items": []}                                   # Quiz needs >=1 item
NO_CITATION     = {"items": [{"question": "long enough question", "answer": "a", "citations": []}]}
SHORT_QUESTION  = {"items": [{"question": "hi", "answer": "a", "citations": ["c1"]}]}   # <10 chars

NO_SECTIONS     = {"sections": []}                               # Guide needs >=1 section
GUIDE_NO_CITE   = {"sections": [{"heading": "H", "claims": [{"text": "long enough claim", "citations": []}]}]}


def test_valid_quiz_parses():
    """TODO(you): model_validate a well-formed quiz dict (one item, valid
    question/answer/citations). Assert it has 1 item and the fields match."""
    
    quiz = Quiz.model_validate(VALID_QUIZ)
    assert len(quiz.items) == 1
    assert quiz.items[0].citations == ["c1"]
    assert quiz.items[0].question == "What does TCP guarantee?"

    


def test_valid_guide_parses():
    """TODO(you): build a valid Guide (one section, one claim). Assert the
    nesting came through: 1 section, 1 claim, right heading."""
    
    guide = Guide.model_validate(VALID_GUIDE)
    assert len(guide.sections) == 1
    assert guide.sections[0].heading == "TCP basics"
    assert len(guide.sections[0].claims) == 1 
    assert guide.sections[0].claims[0].text == "TCP guarantees ordered delivery"
    assert guide.sections[0].claims[0].citations ==  ["c1"]


# --- reject paths (each should raise ValidationError) ----------------------

def test_empty_quiz_items_rejected():
    """TODO(you): a Quiz with items=[] must raise. Use:
        with pytest.raises(ValidationError):
            Quiz.model_validate({"items": []})
    """
    with pytest.raises(ValidationError):
        Quiz.model_validate(NO_ITEMS)
    


def test_quiz_item_needs_a_citation():
    """TODO(you): a QuizItem with citations=[] must raise."""
    with pytest.raises(ValidationError):
        Quiz.model_validate(NO_CITATION)


def test_short_question_rejected():
    """TODO(you): a QuizItem whose question is under 10 chars must raise."""
    with pytest.raises(ValidationError):
        Quiz.model_validate(SHORT_QUESTION)



def test_empty_guide_sections_rejected():
    """TODO(you): a Guide with sections=[] must raise."""
    
    with pytest.raises(ValidationError):
        Guide.model_validate(NO_SECTIONS)


def test_guide_claim_needs_a_citation():
    """TODO(you): a GuideClaim with citations=[] must raise (nested three
    levels deep — proves Pydantic validates recursively)."""
    with pytest.raises(ValidationError):
        Guide.model_validate(GUIDE_NO_CITE)


# --- all_citations() flattening -------------------------------------------

def test_quiz_all_citations_flattens():
    """TODO(you): build a Quiz with two items whose citations are e.g.
    ["c1","c2"] and ["c3"]. Assert all_citations() == ["c1","c2","c3"]."""

    assert Quiz.model_validate(TWO_ITEM_QUIZ).all_citations() == ["c1", "c2", "c3"]

def test_guide_all_citations_flattens():
    """TODO(you): build a Guide with citations spread across sections/claims.
    Assert all_citations() returns them all, flattened."""
    
    assert Guide.model_validate(MULTI_GUIDE).all_citations() == ["c1", "c2", "c3", "c4"]