"""
The quantitative pre-check: a DETERMINISTIC guard that runs BEFORE the LLM Critic.

Why this exists
---------------
CV bullets are where a model inflates: "improved grounding by 40%", "built 12
endpoints", "cut latency in half". Those are exactly the claims a hiring manager
will probe, and exactly the ones an LLM judge is least reliable at policing
(it tends to accept a plausible-sounding number).

So we don't ask the model. Every number in a claim must appear LITERALLY in the
evidence the claim cites — a pure string check, no LLM call, no judgment. If a
number isn't there, the claim is struck before the Critic ever runs. Cheapest
possible check for the highest-risk failure.

The policy (decided 2026-07-19): STRICT EXACT MATCH
--------------------------------------------------
A number passes only if it appears in a cited chunk after *simple* normalization
(whitespace, `%` vs "percent", comma separators, trailing .0) — nothing semantic.
  - claim "92.7% grounding" + evidence "92.7% of claims"  -> PASS
  - claim "~93% grounding"  + evidence "92.7% ..."        -> STRIKE (rounding is
    a semantic judgment; the Refiner rewrites it to the exact figure)
  - claim "41 claims"       + evidence "38/41 claims"     -> PASS (41 appears)
Rejected: tolerant rounding (the rounding rule becomes code that itself needs
testing) — see the spec for the full rationale.
"""
import re

# Matches numeric tokens with an optional trailing unit/symbol:
#   92.7%   41   1,200   3x   $50   10ms
NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*\s*(?:%|percent|x|ms|s|k|m)?", re.IGNORECASE)


def normalize(token: str) -> str:
    """
    Canonicalize a numeric token so trivial formatting differences don't cause
    a false strike. Mechanical only — no rounding, no unit conversion.

        " 92.7 % " -> "92.7%"      "1,200" -> "1200"
        "92.7 percent" -> "92.7%"  "41.0"  -> "41"
    """
    t = token.strip().lower().replace(" ", "").replace(",", "")
    t = t.replace("percent", "%")
    if "." in t:                      # 41.0 -> 41, but keep 92.7
        head = t.rstrip("%x")
        suffix = t[len(head):]
        if head.endswith(".0"):
            t = head[:-2] + suffix
    return t


def extract_numbers(text: str) -> list[str]:
    """
    Return every numeric token in `text`, normalized.

    TODO(you) — V2-M3:
      1. Use NUMBER_RE.findall(text) to pull out the raw numeric tokens.
      2. Run each through normalize().
      3. Return the list (duplicates are fine; empty list if the text has no
         numbers — that's the common, healthy case).
    """
    raise NotImplementedError("quant.extract_numbers — your turn (V2-M3)")


def check_quantities(claim_text: str, cited_chunks: list[dict]) -> tuple[bool, str]:
    """
    Deterministic quantitative check for ONE claim.

    Returns (ok, reason):
        (True,  "")                      every number in the claim is in evidence
        (False, "<why>")                 at least one number is unsupported

    TODO(you) — V2-M3:
      1. numbers = extract_numbers(claim_text)
         If there are none -> return (True, "") immediately. A bullet with no
         figures can't inflate a figure; the LLM Critic still checks its meaning.
      2. Build the evidence haystack: the normalized numbers found across ALL
         cited chunks (extract_numbers over each chunk's "text").
         Hint: a set makes the membership test cheap and reads clearly.
      3. Every claim number must be present in that haystack. Collect the ones
         that are missing.
      4. If any are missing -> return (False, <a reason naming the exact
         unsupported figure(s)>). That reason lands in the artifact and the run
         trace, so make it specific: the Refiner and the reader both need to
         know WHICH number failed, not just that one did.
      5. Otherwise -> return (True, "").

    Edge cases worth a thought:
      - the same number appearing in a different context ("41 chunks" in the
        claim vs "41 files" in evidence) still PASSES — this check is about
        fabricated figures, not about semantics. The LLM Critic covers meaning;
        these two layers are deliberately different jobs.
      - a claim citing several chunks: the number may come from ANY of them.
    """
    raise NotImplementedError("quant.check_quantities — your turn (V2-M3)")
