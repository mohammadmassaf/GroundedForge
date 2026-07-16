"""
Unit tests for the parse/validate layer (generate/generator.py).

No LLM: _parse_and_validate takes a raw string (as if from the model) plus
the set of valid chunk_ids, and must (a) parse good JSON into the right
schema, (b) reject bad JSON / bad shape, (c) catch citations of chunk_ids
that weren't in the context. That last check is the grounding guard.
"""
import json

import pytest

from generate.generator import _parse_and_validate
from generate.schema import Quiz, Guide

VALID_IDS = {"c1", "c2"}


def _quiz_json(citations):
    return json.dumps({"items": [
        {"question": "a long enough question", "answer": "ans", "citations": citations}
    ]})


# --- happy path ------------------------------------------------------------

def test_valid_quiz_json_parses():
    """TODO(you): feed _quiz_json(["c1"]) with VALID_IDS, default model=Quiz.
    Assert you get a Quiz back with one item."""
    
    raw = _quiz_json(["c1"])
    quiz = _parse_and_validate(raw , VALID_IDS)
    assert len(quiz.items) == 1



def test_fenced_json_is_stripped():
    """TODO(you): wrap a valid quiz json in ```json ... ``` fences and assert
    it STILL parses (the fence-stripping path). Build the fenced string with
    an f-string or concatenation."""
    
    raw = f"```json\n{_quiz_json(['c1'])}\n```"
    quiz = _parse_and_validate(raw, VALID_IDS)
    assert len(quiz.items) == 1


# --- reject paths (raise ValueError) --------------------------------------

def test_malformed_json_rejected():
    """TODO(you): pass 'not json at all' -> pytest.raises(ValueError)."""
    
    with pytest.raises(ValueError):
        _parse_and_validate("not json" , VALID_IDS)


def test_unknown_citation_rejected():
    """TODO(you): _quiz_json(["c9"]) cites a chunk_id NOT in VALID_IDS ->
    must raise ValueError (the semantic grounding check)."""

    raw = _quiz_json(["c9"])
    with pytest.raises(ValueError):
        _parse_and_validate(raw , VALID_IDS)


# --- schema routing via model= --------------------------------------------

def test_model_param_routes_to_guide():
    """TODO(you): a valid GUIDE json passed with model=Guide returns a Guide;
    the SAME guide json passed with model=Quiz (default) must raise, because
    it doesn't match the quiz shape. Two asserts / one raises."""
    ...
    raw = json.dumps({"sections": [
    {"heading": "H", "claims": [
        {"text": "a long enough claim", "citations": ["c1"]}
    ]}
    ]})
    guide = _parse_and_validate(raw ,VALID_IDS , model = Guide)
    assert len(guide.sections) == 1

    with pytest.raises(ValueError):
        _parse_and_validate(raw,VALID_IDS)