"""
Tests for the reasoning-exhaustion guard (generate.generator._reasoning_exhausted
and the two loops that use it).

The bug: gpt-oss-20b bills reasoning tokens as OUTPUT, against the same cap as
the answer. Given a thin or off-topic pool it reasons until the cap and returns
finish_reason "length" with empty content. The retry loop then asked twice more
and got the same nothing, so one unanswerable question cost ~6000 tokens.

Measured 60% of the time (3 of 5 runs) on an off-corpus topic, which on a public
demo is not an edge case -- it is what happens when a visitor types something the
RFCs do not cover.

The guard is pure logic over a response object, so these tests fake the response
and cost nothing. What they must pin is BOTH directions: that exhaustion is
detected, and that nothing else is mistaken for it -- treating a normal reply as
exhausted would turn working generations into GenerationError.
"""
from types import SimpleNamespace

import pytest

from generate import generator
from generate.generator import _reasoning_exhausted, GenerationError


def choice(finish_reason, content):
    """The two fields of a completion choice the guard reads."""
    return SimpleNamespace(finish_reason=finish_reason,
                           message=SimpleNamespace(content=content))


# --- what counts as exhausted -------------------------------------------------

def test_truncated_with_no_content_is_exhaustion():
    """The observed shape: usage showed 1998 of 2000 tokens spent on reasoning
    and the content came back empty."""
    assert _reasoning_exhausted(choice("length", "")) is True


def test_truncated_with_only_whitespace_is_exhaustion():
    """A couple of stray tokens after the reasoning is still nothing to parse."""
    assert _reasoning_exhausted(choice("length", "   \n ")) is True


def test_a_none_content_is_exhaustion():
    """The SDK may hand back None rather than "". `None.strip()` would raise,
    so the guard coalesces first — an AttributeError here would surface as a
    mystery crash on the one path that is supposed to explain itself."""
    assert _reasoning_exhausted(choice("length", None)) is True


# --- what must NOT count ------------------------------------------------------

def test_a_normal_reply_is_not_exhaustion():
    """finish_reason "stop" with JSON is the happy path. If this ever returns
    True, every successful generation becomes a GenerationError."""
    assert _reasoning_exhausted(choice("stop", '{"items": []}')) is False


def test_truncation_that_still_produced_text_is_not_exhaustion():
    """
    A reply cut off mid-JSON is a DIFFERENT failure: the model was answering and
    ran out of room. That one the retry loop can genuinely fix by showing the
    model its malformed output, so it must keep reaching the retry path.
    """
    assert _reasoning_exhausted(choice("length", '{"items": [{"question": "wh')) is False


def test_an_empty_reply_that_was_not_truncated_is_not_exhaustion():
    """Empty content with finish_reason "stop" is a different bug again, and
    mislabelling it would hide it behind a "corpus cannot answer" message."""
    assert _reasoning_exhausted(choice("stop", "")) is False


# --- the loops fail fast ------------------------------------------------------

def _fake_response(finish_reason, content):
    return SimpleNamespace(choices=[choice(finish_reason, content)])


def test_generation_does_not_retry_an_exhausted_response(monkeypatch):
    """
    The whole point: one call, not three.

    Retrying is not merely useless here, it is expensive and deterministic --
    the model refills whatever budget it is given (1998/2000, then 3998/4000
    when the cap was raised to check). Counting calls is the assertion because
    the token cost IS the bug.
    """
    calls = []

    def fake_complete(messages, temperature, max_tokens):
        calls.append(1)
        return _fake_response("length", "")

    monkeypatch.setattr(generator, "_complete", fake_complete)

    chunks = [{"chunk_id": "rfc791_p20_c0", "source_file": "rfc791.txt",
               "page": 20, "text": "the fragment offset is measured in units of 8 octets"}]
    with pytest.raises(GenerationError):
        generator.generate("something the corpus cannot answer", chunks, n=4)

    assert len(calls) == 1, f"expected one call, made {len(calls)}"


def test_malformed_json_is_still_retried(monkeypatch):
    """
    The guard must not shrink the retry loop's real job. Invalid JSON with
    content present is exactly what the loop was built for -- the model is shown
    its own bad output and asked again -- so it must still get its attempts.
    """
    calls = []

    def fake_complete(messages, temperature, max_tokens):
        calls.append(1)
        return _fake_response("stop", "not json at all")

    monkeypatch.setattr(generator, "_complete", fake_complete)

    chunks = [{"chunk_id": "rfc791_p20_c0", "source_file": "rfc791.txt",
               "page": 20, "text": "some evidence"}]
    with pytest.raises(GenerationError):
        generator.generate("a topic", chunks, n=4)

    assert len(calls) == generator.MAX_RETRIES + 1, (
        f"malformed JSON should use all {generator.MAX_RETRIES + 1} attempts, "
        f"used {len(calls)}")
