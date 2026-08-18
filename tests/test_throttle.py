"""
Unit tests for the per-minute throttle retry (generate/generator.py::_complete).

Groq raises the SAME RateLimitError for two limits that mean opposite things:

  TPM (per minute)  a throttle. gpt-oss-20b allows 8000 and one generate call is
                    ~4800, so back-to-back calls hit it constantly. Wait: SECONDS.
  TPD (per day)     exhaustion. Wait: hours. The run should stop and report
                    honest partial tallies.

The harness originally treated every RateLimitError as "stop the run", and a
7.5-second throttle threw away five unevaluated questions. Retrying both would
be worse -- it would spin for hours against a daily cap while reporting nothing.

So the whole behaviour under test is: which error do we wait out, and which do we
let through. time.sleep is patched everywhere so the suite stays instant.
"""
import pytest

from generate import generator


class _FakeResponse:
    """Stands in for a Groq completion. Only identity matters here."""


def _rate_limit(message):
    """
    Build a RateLimitError without a live HTTP exchange. Groq's constructor wants
    a response and body, so this fakes the minimum: `str(err)` must carry the
    message, since that is what _is_tpm and _retry_after read.
    """
    class _Err(generator.RateLimitError):
        def __init__(self, msg):
            Exception.__init__(self, msg)
            self.response = None

    return _Err(message)


TPM = ("Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute "
       "(TPM): Limit 8000, Used 4209, Requested 4786. Please try again in 7.4625s")
TPD = ("Rate limit reached for model `openai/gpt-oss-20b` on tokens per day "
       "(TPD): Limit 100000, Used 99513, Requested 2607. Please try again in 55m43s")


def _client_raising(*errors):
    """A fake client that raises the given errors in order, then succeeds."""
    calls = {"n": 0}
    queue = list(errors)

    class _Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            if queue:
                raise queue.pop(0)
            return _FakeResponse()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    return _Client(), calls


# --- telling the two limits apart -------------------------------------------

def test_tpm_and_tpd_are_distinguished():
    """The only signal is the message text; both are the same exception class."""
    assert generator._is_tpm(_rate_limit(TPM)) is True
    assert generator._is_tpm(_rate_limit(TPD)) is False


def test_wait_is_read_from_the_message():
    """Groq states the wait inline. A buffer is added because sleeping the exact
    figure lands on the boundary and fails again."""
    assert 8.0 <= generator._retry_after(_rate_limit(TPM)) <= 9.0


def test_unparseable_wait_falls_back_rather_than_crashing():
    """A message format change must slow the run down, not break it."""
    wait = generator._retry_after(_rate_limit("tokens per minute (TPM): slow down"))

    assert wait == generator.TPM_FALLBACK_SLEEP


def test_an_absurd_wait_is_capped():
    """A multi-minute 'throttle' is not a throttle. The cap stops a runaway sleep
    from silently stalling a run for an hour."""
    huge = _rate_limit("tokens per minute (TPM): try again in 99999.0s")

    assert generator._retry_after(huge) == generator.TPM_MAX_SLEEP


# --- the retry policy --------------------------------------------------------

def test_a_throttle_is_waited_out_and_the_call_succeeds(monkeypatch):
    """The point of the whole change: a TPM error costs seconds, not the run."""
    client, calls = _client_raising(_rate_limit(TPM))
    slept = []
    monkeypatch.setattr(generator, "_get_client", lambda: client)
    monkeypatch.setattr(generator.time, "sleep", slept.append)

    resp = generator._complete([{"role": "user", "content": "x"}], 0.3, 100)

    assert isinstance(resp, _FakeResponse)
    assert calls["n"] == 2          # failed once, retried, succeeded
    assert len(slept) == 1


def test_a_daily_cap_is_raised_immediately(monkeypatch):
    """TPD must propagate so grounding_eval can stop early and keep partial
    tallies. Waiting it out would spin for hours and report nothing -- and
    asserting that nothing slept is how we prove it didn't try."""
    client, calls = _client_raising(_rate_limit(TPD))
    slept = []
    monkeypatch.setattr(generator, "_get_client", lambda: client)
    monkeypatch.setattr(generator.time, "sleep", slept.append)

    with pytest.raises(generator.RateLimitError):
        generator._complete([{"role": "user", "content": "x"}], 0.3, 100)

    assert calls["n"] == 1
    assert slept == []


def test_repeated_throttles_eventually_give_up(monkeypatch):
    """Without a cap a persistent throttle is an infinite loop. Giving up raises,
    which the harness reports honestly, rather than hanging forever."""
    client, calls = _client_raising(*[_rate_limit(TPM)] * 20)
    monkeypatch.setattr(generator, "_get_client", lambda: client)
    monkeypatch.setattr(generator.time, "sleep", lambda s: None)

    with pytest.raises(generator.RateLimitError):
        generator._complete([{"role": "user", "content": "x"}], 0.3, 100)

    assert calls["n"] == generator.TPM_MAX_ATTEMPTS
