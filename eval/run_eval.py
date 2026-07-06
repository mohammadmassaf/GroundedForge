"""
M5: the evaluation harness.

Two independent measurements, one report:

  retrieval eval  -> recall@k: for each eval question, does top-k contain
                     a chunk from a known-right (source_file, page)?
                     Grades RETRIEVAL alone.
  grounding eval  -> grounding %: run the full cite-or-strike pipeline on
                     eval questions; % of generated claims the Critic
                     supported. Grades GENERATION alone.

Separating them is the point: a bad quiz item is either a retrieval miss
(right chunk never surfaced) or a generation failure (chunk surfaced,
generator ignored it) - two different bugs, two different fixes.

Run:  python main.py eval --corpus networks
"""
import json
from pathlib import Path

from retrieve.query import search
from critic.loop import run_loop

EVAL_SET = Path("eval/eval_set.json")

KS = (3, 5, 10)


def load_eval_set() -> list[dict]:
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    return data["items"]


def _hit(results: list[dict], expected: list[dict]) -> bool:
    """
    TODO(you): does any retrieved result match any expected location?

    A result matches when its source file equals an expected entry's file
    AND its page is one of that entry's pages. Return True/False.
    (You've done this kind of cross-checking before - the semantic
    citation check in _parse_and_validate.)
    """
    for result in results:
        for actual in expected:
            if (result["source_file"] == actual["source_file"] and
                result["page"] in actual["pages"]): 
                return True

    return False


def retrieval_eval(items: list[dict], corpus: str) -> dict:
    """
    TODO(you): compute recall@k for each k in KS.

    Idea: retrieve once per eval question with the LARGEST k. The top-3
    results are just the first 3 of those - slice, don't re-query.
    For each k, count the questions where _hit() is true over the first
    k results, divide by the number of questions.

    Return {"recall@3": 0.8, "recall@5": ..., "recall@10": ...} plus a
    list of the questions that missed at the largest k (for the report).
    """
    raise NotImplementedError


def grounding_eval(items: list[dict], corpus: str, n: int = 2) -> dict:
    """
    TODO(you): compute the grounding score over the eval set.

    Idea: for each eval question, run the full pipeline (you already have
    run_loop) asking for a small n. Tally supported vs struck across all
    questions - the counts are just the lengths of what run_loop returns.
    grounding % = supported / total claims generated.

    Return {"grounded": <float 0..1>, "total_claims": int,
            "struck_examples": [(question, reason), ...]}.

    Note: this makes 2 LLM calls per claim - on the free tier, run it on
    a SUBSET first (items[:5]) while debugging.
    """
    raise NotImplementedError


def report(retrieval: dict, grounding: dict) -> str:
    lines = ["", "=" * 52, "GROUNDED FORGE - EVAL REPORT", "=" * 52]
    for k in KS:
        pct = retrieval[f"recall@{k}"] * 100
        lines.append(f"  recall@{k:<2} : {pct:5.1f}%")
    if retrieval.get("misses"):
        lines.append("  retrieval misses:")
        for q in retrieval["misses"]:
            lines.append(f"    - {q}")
    lines.append("-" * 52)
    pct = grounding["grounded"] * 100
    lines.append(f"  grounding : {pct:5.1f}%  ({grounding['total_claims']} claims)")
    for q, reason in grounding.get("struck_examples", []):
        lines.append(f"    struck: {q[:60]} - {reason[:80]}")
    lines.append("=" * 52)
    return "\n".join(lines)
