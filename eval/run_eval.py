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
from critic.loop import run_loop, run_bullets_loop
from critic.quant import check_quantities
from critic.critic import check_claim
from generate.generator import GenerationError
from retrieve.hybrid import hybrid_search

EVAL_SET = Path("eval/eval_set.json")
EVAL_SET_JOB = Path("eval/eval_set_job.json")

KS = (3, 5, 10)


def _eval_path(corpus: str) -> Path:
    """Job mode has its own eval set; every other corpus uses the v1 one."""
    return EVAL_SET_JOB if corpus == "job" else EVAL_SET


def load_eval_set(corpus: str = "default") -> list[dict]:
    data = json.loads(_eval_path(corpus).read_text(encoding="utf-8"))
    return data["items"]


def load_traps(corpus: str = "default") -> list[dict]:
    """Adversarial items (v2). Absent from the v1 set -> empty list."""
    data = json.loads(_eval_path(corpus).read_text(encoding="utf-8"))
    return data.get("traps", [])


def _scope(item: dict) -> dict | None:
    """
    The retrieval filter for one eval item.

    Job-mode items declare the repo their evidence must come from. Without this
    the corpus's two projects compete for every question, and a relevance
    scorer is free to prefer the wrong one: for "How is authentication
    implemented in MealWise?" the cross-encoder ranked a Grounded Forge commit
    first, because it contained "implemented" and the literal string
    "mealwise@8912ac4" (quoted as an example) but no mention of auth.

    v1 study items carry no repo -> None -> unscoped, exactly as before.
    """
    return {"repo": item["repo"]} if item.get("repo") else None


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
        where = _scope(item)
        if mode == "vector":
            result = search(item["question"], corpus , k = max(KS) , where=where)
        elif mode == "hybrid":
            result = hybrid_search(item["question"] , corpus ,  k = max(KS) , use_rerank=False , where=where)
        else:
            result = hybrid_search(item["question"] , corpus ,  k = max(KS) , use_rerank=True , where=where)
        if not _hit(result, item["expected"]):
            misses.append(item["question"])
        for k in KS:
            if _hit(result[:k],item["expected"]):
                hits[k] +=1
    out = {f"recall@{k}": hits[k] / len(items) for k in KS}
    out["misses"] = misses
    return out





def grounding_eval(items: list[dict], corpus: str, n: int = 2,
                   generator: str = "quiz") -> dict:
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
    evaluated = gapped = 0
    stopped_early = False
    gap_examples = []
    struck_examples = []
    grounding = {}
    for item in items:
        try:
            if generator == "bullets":
                # job mode: quiz items over a commit history are meaningless.
                # Same cite-or-strike policy, different output type.
                chunks = hybrid_search(item["question"], corpus, k=8, use_rerank=True,
                                       where=_scope(item))
                kept, struck, gap = run_bullets_loop(item["question"], chunks, n=n)
            else:
                chunks = search(item["question"], corpus , k = 8)
                kept , struck  = run_loop(item["question"] , chunks , n=n )
                gap = None
        except  GenerationError as e:
            failed += 1
            print(f"  skipped '{item['question'][:50]}': {e}")
            continue
        except RateLimitError as e:
            print(f"  rate limit hit - stopping early: {e}")
            stopped_early = True
            break

        evaluated += 1
        # A gapped question produces no claims at all, so it cannot move the
        # grounding ratio in either direction -- it silently leaves the
        # denominator. Count it, or a run where 13 of 15 questions declined to
        # answer still reports a confident percentage.
        if gap:
            gapped += 1
            gap_examples.append((item["question"], gap))

        total_kept += len(kept)
        total_struck += len(struck)

        for claim , reason in struck:
            # QuizItem carries .question, CVBullet carries .text
            struck_examples.append((getattr(claim, "question", None) or claim.text, reason))
    total_claims = total_kept + total_struck
    grounded = total_kept / total_claims if total_claims else 0.0

    grounding["grounded"] = grounded
    grounding["total_claims"] = total_claims
    grounding["struck_examples"] = struck_examples
    grounding["failed"] = failed
    grounding["evaluated"] = evaluated
    grounding["total_questions"] = len(items)
    grounding["gapped"] = gapped
    grounding["gap_examples"] = gap_examples
    grounding["stopped_early"] = stopped_early

    return grounding


def trap_eval(traps: list[dict], corpus: str = "job") -> dict:
    """
    Adversarial eval (V2-M5): does the Critic strike claims we KNOW are wrong?

    Why this exists. recall@k and grounding % both measure the system on good
    inputs, so neither can see a broken Critic — one that answers "supported"
    to everything scores 100% grounding, the best possible number from the
    worst possible judge. Traps measure the other direction: authored claims
    with a known defect, counted by whether they were caught.

    Each trap carries {claim, citations, stage, why}. `stage` is which check
    SHOULD catch it: "quant" (a figure absent from the cited chunks) or
    "critic" (semantically wrong, but every number in it does appear — so the
    deterministic check cannot see it).

    TODO(you) — V2-M5:
      1. Load chunks/<corpus>.json and index it by chunk_id, so a trap's
         citations can be resolved to real chunk dicts (same by_id shape
         run_star_loop builds).
      2. For each trap, run the SAME two stages the loop runs, in the same
         order: check_quantities first, then check_claim only if it passes.
         Record whether the trap was struck and which stage struck it.
      3. Return the catch rate plus enough detail for the report: which traps
         escaped, and which were caught by a stage other than the expected one.

    DECISION — what counts as a catch. A trap struck by the "wrong" stage was
    still struck, so the artifact would be safe; but a t5-style trap caught by
    quant means the haystack isn't what we think it is. Decide whether the
    headline rate counts stage-correctness or only strike/no-strike, and make
    the other one visible rather than silently folded in.

    Note the asymmetry with grounding %: there, struck claims are the bad
    outcome. Here, struck claims are the ONLY good outcome. A trap that
    survives is a hole in the guard.
    """
    chunks = json.loads(Path(f"chunks/{corpus}.json").read_text(encoding="utf-8"))

    by_id = {c["chunk_id"]: c  for c in chunks}
    caught = 0 
    total = len(traps)
    stage_mismatch = []
    by_stage= {"quant": 0, "critic": 0}
    escaped = []
    for trap in traps:
        caught_by = None
        cited = [by_id[cid] for cid in trap["citations"]]
        ok , reason = check_quantities(trap["claim"] , cited)
        if not ok:
            caught_by = "quant"
        else:
            verdict = check_claim("" , trap["claim"] , cited)
            if not verdict.supported:
                caught_by = "critic"
        if caught_by :
            caught +=1  
            by_stage[caught_by] +=1 
            if caught_by != trap["stage"]:       
                stage_mismatch.append((trap["id"] , trap["stage"] , caught_by))
        else:
            escaped.append((trap["id"] , trap["claim"]))
    catch_rate = caught / total 

    return {
    "catch_rate": caught / total if total else 0.0,
    "caught": caught,
    "total": total,
    "by_stage": by_stage,
    "escaped": escaped,
    "stage_mismatch": stage_mismatch,
    }


        

        
    
        


def report(retrieval: dict, grounding: dict, traps: dict | None = None) -> str:
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
    # The denominator travels WITH the percentage. A grounding score can only be
    # lowered by a claim that was generated and then struck, so every question
    # that produced nothing -- gapped, failed, or never reached -- is invisible
    # in the ratio and has to be reported beside it.
    scope = f"{grounding['total_claims']} claims"
    if grounding.get("total_questions"):
        scope += f" from {grounding.get('evaluated', '?')}/{grounding['total_questions']} questions"
    lines.append(f"  grounding : {pct:5.1f}%  ({scope})")
    if grounding.get("stopped_early"):
        lines.append("  !! STOPPED EARLY (rate limit) - remaining questions were never evaluated")
    if grounding.get("failed"):
        lines.append(f"  !! {grounding['failed']} question(s) failed generation and were excluded")
    if grounding.get("gapped"):
        lines.append(f"  !! {grounding['gapped']} question(s) returned a gap (no claims, excluded from the ratio)")
        for q, gap in grounding.get("gap_examples", []):
            lines.append(f"    gap: {q[:60]} - {gap[:70]}")
    for q, reason in grounding.get("struck_examples", []):
        lines.append(f"    struck: {q[:60]} - {reason[:80]}")
    if traps:
        lines.append("-" * 52)
        pct = traps["catch_rate"] * 100
        lines.append(f"  inflation-catch : {pct:5.1f}%  ({traps['caught']}/{traps['total']} traps struck)")
        for stage, count in sorted(traps.get("by_stage", {}).items()):
            lines.append(f"      caught by {stage:<7}: {count}")
        for tid, claim in traps.get("escaped", []):
            lines.append(f"    ESCAPED {tid}: {claim[:70]}")
        for tid, expected, actual in traps.get("stage_mismatch", []):
            lines.append(f"    stage mismatch {tid}: expected {expected}, caught by {actual}")
    lines.append("=" * 52)
    return "\n".join(lines)
