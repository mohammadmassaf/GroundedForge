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
from groq import RateLimitError 

from retrieve.query import search
from critic.loop import run_loop
from generate.generator import GenerationError
from retrieve.hybrid import hybrid_search

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


def retrieval_eval(items: list[dict], corpus: str , mode = "vector") -> dict:
    """
    TODO(you): compute recall@k for each k in KS.

    Idea: retrieve once per eval question with the LARGEST k. The top-3
    results are just the first 3 of those - slice, don't re-query.
    For each k, count the questions where _hit() is true over the first
    k results, divide by the number of questions.

    Return {"recall@3": 0.8, "recall@5": ..., "recall@10": ...} plus a
    list of the questions that missed at the largest k (for the report).
    """
    hits = {k:0 for k in KS}
    misses = []
    for item in items:
        if mode == "vector":
            result = search(item["question"], corpus , k = max(KS))
        elif mode == "hybrid":
            result = hybrid_search(item["question"] , corpus ,  k = max(KS) , use_rerank=False)
        else:
            result = hybrid_search(item["question"] , corpus ,  k = max(KS) , use_rerank=True)
        if not _hit(result, item["expected"]):
            misses.append(item["question"])
        for k in KS:
            if _hit(result[:k],item["expected"]):
                hits[k] +=1
    out = {f"recall@{k}": hits[k] / len(items) for k in KS}
    out["misses"] = misses
    return out





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
    total_kept = total_struck = failed = 0
    struck_examples = []
    grounding = {}
    for item in items:
        try:
            chunks = search(item["question"], corpus , k = 8)
            kept , struck  = run_loop(item["question"] , chunks , n=n )
        except  GenerationError as e:
            failed += 1
            print(f"  skipped '{item['question'][:50]}': {e}")
            continue
        except RateLimitError as e:
            print(f"  rate limit hit - stopping early: {e}")
            break

        total_kept += len(kept)
        total_struck += len(struck)
        
        for quiz_item , reason in struck:
            struck_examples.append((quiz_item.question,reason))
    total_claims = total_kept + total_struck
    grounded = total_kept / total_claims if total_claims else 0.0

    grounding["grounded"] = grounded                    
    grounding["total_claims"] = total_claims               
    grounding["struck_examples"] = struck_examples 
    grounding["failed"] = failed
     
    return grounding


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
    if grounding.get("failed"):
        lines.append(f"  !! {grounding['failed']} question(s) failed generation and were excluded")
    for q, reason in grounding.get("struck_examples", []):
        lines.append(f"    struck: {q[:60]} - {reason[:80]}")
    lines.append("=" * 52)
    return "\n".join(lines)
