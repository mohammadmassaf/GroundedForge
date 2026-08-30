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

# The pool size the Generator reads, and the default for --gen-k. Not decoration:
# it is the only recall figure that predicts what the Generator can actually
# cite. j1 showed why -- its evidence sat at rank 9, so recall@10 read 100% for a
# question the Generator never saw the evidence for.
GEN_K = 8

# 3 and 5 are the ranking-quality probes, 10 is the wide net. gen_k joins them so
# the recall table ALWAYS contains the k in use, whatever it is set to.
BASE_KS = (3, 5, 10)


def _ks(gen_k: int = GEN_K) -> tuple[int, ...]:
    return tuple(sorted(set(BASE_KS) | {gen_k}))


KS = _ks()


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
    Did retrieval surface any of the known-right locations?

    A result matches when its source file equals an expected entry's file AND
    its page is one of that entry's pages. Matching on (file, page) rather than
    on chunk_id is deliberate: chunk ids change whenever chunk size, overlap or
    the ingest order changes, and an eval set that has to be rewritten after
    every re-chunk is not a fixed measuring stick.
    """
    for result in results:
        for actual in expected:
            if (result["source_file"] == actual["source_file"] and
                result["page"] in actual["pages"]): 
                return True

    return False


def retrieval_eval(items: list[dict], corpus: str , mode = "vector",
                   gen_k: int = GEN_K) -> dict:
    """
    recall@k for each k in KS: of the eval questions, what fraction had a
    known-right chunk somewhere in the top k?

    Grades RETRIEVAL alone, with no LLM involved, which is why it is free and
    reproducible. It is also the CEILING on grounding - the generator cannot
    cite evidence that never reached it, so a recall miss and a generation
    failure look identical in the output and need opposite fixes.

    One retrieval per question, at the largest k; the smaller ks are slices of
    that same ranked list rather than fresh queries, so every row of the table
    describes the same retrieval.
    """
    ks = _ks(gen_k)
    hits = {k:0 for k in ks}
    misses = []
    for item in items:
        result = _retrieve(item["question"], corpus, max(ks), mode, _scope(item))
        if not _hit(result, item["expected"]):
            misses.append(item["question"])
        for k in ks:
            if _hit(result[:k],item["expected"]):
                hits[k] +=1
    out = {f"recall@{k}": hits[k] / len(items) for k in ks}
    out["misses"] = misses
    # the report reads these back rather than the module constant, so a swept
    # --gen-k still prints a table containing the k that fed generation
    out["ks"] = ks
    out["gen_k"] = gen_k
    return out





def _retrieve(question: str, corpus: str, k: int, mode: str,
              where: dict | None) -> list[dict]:
    """One retrieval, in whichever mode the run was asked for. Shared by both
    evals so the report's two halves can never grade different retrievers."""
    if mode == "vector":
        return search(question, corpus, k=k, where=where)
    return hybrid_search(question, corpus, k=k, use_rerank=(mode == "rerank"),
                         where=where)


def grounding_eval(items: list[dict], corpus: str, n: int = 2,
                   generator: str = "quiz", mode: str = "rerank",
                   gen_k: int = GEN_K) -> dict:
    """
    The grounding score: of every claim generated over the eval set, what
    fraction did the Critic support against the sources it cited?

    Grades GENERATION alone. Two LLM calls per claim, so a full 15-question job
    run costs ~90-96k tokens against a ~100k rolling-day cap - hence --limit and
    --offset, and hence the raw kept/struck counts in the return value, so two
    windowed runs can be combined correctly by summing counts. Averaging their
    percentages is wrong whenever the windows produced different numbers of
    claims, which they always do.

    Questions that produce no claims - gapped, failed, or never reached - are
    counted separately rather than folded into the ratio. A claim can only lower
    the score by being generated and struck, so a run where 13 of 15 questions
    declined would otherwise report a confident percentage over the two that
    answered.
    """
    total_kept = total_struck = failed = 0
    evaluated = gapped = 0
    stopped_early = False
    gap_examples = []
    struck_examples = []
    grounding = {}
    for item in items:
        try:
            chunks = _retrieve(item["question"], corpus, gen_k, mode, _scope(item))
            if generator == "bullets":
                # job mode: quiz items over a commit history are meaningless.
                # Same cite-or-strike policy, different output type.
                kept, struck, gap = run_bullets_loop(item["question"], chunks, n=n,
                                                     corpus=corpus)
            else:
                kept , struck  = run_loop(item["question"] , chunks , n=n ,
                                          corpus=corpus)
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
    # kept/struck are reported raw so two windowed runs can be COMBINED
    # correctly: grounding is (kept_a + kept_b) / (claims_a + claims_b).
    # Averaging the two percentages is wrong whenever the halves produced
    # different numbers of claims, which they always do.
    grounding["kept"] = total_kept
    grounding["struck"] = total_struck
    grounding["struck_examples"] = struck_examples
    grounding["failed"] = failed
    grounding["evaluated"] = evaluated
    grounding["total_questions"] = len(items)
    grounding["gapped"] = gapped
    grounding["gap_examples"] = gap_examples
    grounding["stopped_early"] = stopped_early
    # both recorded so a run is reproducible from its own report
    grounding["mode"] = mode
    grounding["gen_k"] = gen_k

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

    Each trap carries {claim, citations, stage, why}. `stage` is which check
    SHOULD catch it: "quant" (a figure absent from the cited chunks) or
    "critic" (semantically wrong, but every number in it does appear - so the
    deterministic check cannot see it).

    Traps run the same two stages as the real loop, in the same order, so this
    measures the guard as shipped rather than a reimplementation of it.

    The headline rate counts strike / no-strike, because a trap struck by the
    "wrong" stage still would not have reached the artifact. Stage mismatches
    are reported separately rather than folded in: a critic-stage trap caught
    by quant means the evidence is not what the trap author thought, which is a
    problem with the trap, not the guard.

    Note the asymmetry with grounding %: there, struck claims are the bad
    outcome. Here they are the ONLY good outcome, and a trap that survives is a
    hole in the guard.
    """
    chunks = json.loads(Path(f"chunks/{corpus}.json").read_text(encoding="utf-8"))

    by_id = {c["chunk_id"]: c  for c in chunks}
    caught = 0 
    total = len(traps)
    stage_mismatch = []
    by_stage= {"quant": 0, "critic": 0}
    escaped = []
    evaluated = 0
    stopped_early = False
    for trap in traps:
        caught_by = None
        cited = [by_id[cid] for cid in trap["citations"]]
        ok , reason = check_quantities(trap["claim"] , cited)
        if not ok:
            caught_by = "quant"
        else:
            try:
                verdict = check_claim("" , trap["claim"] , cited)
            except RateLimitError as e:
                # Same policy as grounding_eval. This ran unguarded once and the
                # 429 escaped all the way out of main(), so a run that had
                # already spent 8 questions' worth of tokens printed NO report
                # at all. Partial numbers beat losing the whole run.
                print(f"  trap rate limit hit - stopping early: {e}")
                stopped_early = True
                break
            if not verdict.supported:
                caught_by = "critic"
        evaluated += 1
        if caught_by :
            caught +=1  
            by_stage[caught_by] +=1 
            if caught_by != trap["stage"]:       
                stage_mismatch.append((trap["id"] , trap["stage"] , caught_by))
        else:
            escaped.append((trap["id"] , trap["claim"]))
    # the rate is over what was actually CHECKED, not over what was authored --
    # a trap never reached is neither caught nor escaped
    return {
    "catch_rate": caught / evaluated if evaluated else 0.0,
    "caught": caught,
    "evaluated": evaluated,
    "total": total,
    "stopped_early": stopped_early,
    "by_stage": by_stage,
    "escaped": escaped,
    "stage_mismatch": stage_mismatch,
    }


        

        
    
        


def report(retrieval: dict, grounding: dict, traps: dict | None = None) -> str:
    lines = ["", "=" * 52, "GROUNDED FORGE - EVAL REPORT", "=" * 52]
    gen_k = retrieval.get("gen_k", grounding.get("gen_k", GEN_K))
    for k in retrieval.get("ks", KS):
        pct = retrieval[f"recall@{k}"] * 100
        # the row the Generator actually reads is the one that bounds grounding;
        # the others describe ranking quality only
        tag = "  <- feeds generation" if k == gen_k else ""
        lines.append(f"  recall@{k:<2} : {pct:5.1f}%{tag}")
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
    if "kept" in grounding:
        scope = f"{grounding['kept']} kept / {grounding['struck']} struck"
    else:
        scope = f"{grounding['total_claims']} claims"
    if grounding.get("total_questions"):
        scope += f" from {grounding.get('evaluated', '?')}/{grounding['total_questions']} questions"
    if grounding.get("mode"):
        scope += f", {grounding['mode']} k={grounding.get('gen_k', GEN_K)}"
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
        checked = traps.get("evaluated", traps["total"])
        scope = f"{traps['caught']}/{checked} traps struck"
        if checked != traps["total"]:
            scope += f", {traps['total'] - checked} never checked"
        lines.append(f"  inflation-catch : {pct:5.1f}%  ({scope})")
        if traps.get("stopped_early"):
            lines.append("  !! traps STOPPED EARLY (rate limit)")
        for stage, count in sorted(traps.get("by_stage", {}).items()):
            lines.append(f"      caught by {stage:<7}: {count}")
        for tid, claim in traps.get("escaped", []):
            lines.append(f"    ESCAPED {tid}: {claim[:70]}")
        for tid, expected, actual in traps.get("stage_mismatch", []):
            lines.append(f"    stage mismatch {tid}: expected {expected}, caught by {actual}")
    lines.append("=" * 52)
    return "\n".join(lines)
