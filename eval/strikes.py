"""
Strike attribution: turning "I think it mis-cited" into a number.

WHY THIS EXISTS
---------------
grounding % says how many claims were struck. It cannot say WHY the system
failed, and there are two causes that read identically in the Critic's reason
string:

  retrieval miss  - the support was NOWHERE in the pool. The generator had
                    nothing to work with. Fix: retrieval (gen_k, mode, scoping).
  mis-citation    - the support WAS in the pool, under a chunk the claim did
                    not cite. The generator had it and pointed elsewhere.
                    Fix: citation discipline in the prompt.

Opposite fixes. Guessing wrong costs a free-tier day tuning the wrong half.

THE LEVER
---------
Since 22fb586 every `generated` event logs `pool` and every verdict logs
`citations`, so `pool - citations` is computable after the fact, offline, for
free: "where the support could have been instead". Ranking that leftover set by
BM25 against the struck claim says which of those chunks LOOK like they should
have been cited.

WHAT THIS IS NOT
----------------
Not a classifier. Worked example (bullets_20260818_103903, a j1 auth strike):
the top-scoring uncited chunk was `mealwise@0cf0a59` at 8.61 vs the cited
chunk's 0.12 - a textbook "mis-citation" shape. Reading it, 0cf0a59 is about
Gemini scaffolding and matched only on "configure/environment/variable". It
does not support the claim either. The real cause was over-reach.

So the headline is a LEAD COUNT - "N strikes had a better-scoring uncited
chunk" - and it becomes a mis-citation count only after a human reads the
shortlists. Word the report that way or it will be quoted as a measurement it
is not.

Run:  python main.py strikes --corpus job
"""
import json
from collections import Counter
from pathlib import Path

from retrieve.keyword import _get_index, _tokenize

TRACE_DIR = Path("traces")

# Traces predating 22fb586 carry neither `pool` nor `citations`, so a strike
# from one is unattributable rather than unattributed. Counted and reported
# separately - silently dropping them would inflate the denominator's quality.
POOL_ERA = "22fb586"

# How much better an uncited chunk must score before it counts as a lead.
# Read off the data, not chosen: across the 21 attributable job strikes the
# margins fall into two groups with a clean gap between them -- 8.11 and above,
# then nothing until 1.75. Anything in 2..8 splits them identically.
MIN_MARGIN = 2.0

# Below this, the CITED chunk scored ~nothing, and that is a statement about the
# tokenizer rather than about the citation. _tokenize splits on whitespace and
# strips only .,;:!?"'()[] -- so the README's `SECRET_key=your_jwt_secret`
# survives as ONE token and the words "secret", "key" and "jwt" can never match
# it. Four strikes cite that chunk and score 0.00-0.12 against it. Their true
# margins are unknowable, so they are reported as unmeasurable rather than as
# the largest leads in the set, which is what a bare margin test made them.
MIN_BASELINE = 0.2



# Which stage each verdict event represents. Also serves as the membership
# test for "is this line a verdict at all".
_STAGES = {"quant_check": "quant", "critic_verdict": "critic"}


def _strike(data: dict, pool: list[str] | None, trace_path: Path) -> dict:
    """One struck claim, normalized so nothing downstream knows which
    generator or which stage produced it."""
    return {
        "trace": trace_path.name,
        "stage": _STAGES[data["event"]],
        "round": data.get("round"),
        # bullets log `bullet`, quiz logs `question`, star logs `section`
        "claim": data.get("bullet") or data.get("question") or data.get("section") or "",
        "citations": data.get("citations", []),
        # None = the run predates 22fb586 and never recorded a pool. NOT the
        # same as "nothing was left uncited" - one is ignorance, the other is
        # a finding, and they must not land in the same bucket.
        "pool": pool,
        "attributable": pool is not None,
        "reason": data.get("reason", ""),
    }

def load_strikes(trace_path: Path) -> list[dict]:
      """
      TODO(you): pull every struck claim out of one trace file.

      A trace is JSONL - one event per line, in the order they happened. Two
      things you need travel on DIFFERENT lines, which is the whole difficulty:

        - `pool` rides on the `generated` event, once per round
        - `citations` + the verdict ride on `quant_check` / `critic_verdict`

      So you have to carry the most recent pool forward as you read down the
      file, and attach it to each verdict you meet after it. (A round-2
      regeneration logs a fresh `generated` - the pool is the same list today,
      but read it per-round anyway rather than assuming.)

      A claim was struck when a `critic_verdict` has supported == False, or a
      `quant_check` has passed == False. Note the two events use DIFFERENT key
      names for the same idea, and different key names for the claim text too
      (`bullet` for bullets, `question` for quiz, `section` for star) - normalize
      to one shape here so nothing downstream has to know which generator ran.

      Return a list of dicts, one per strike, each carrying at least: the claim
      text, its citations, the pool it came from, the reason, and which stage
      struck it ("quant" or "critic"). Include the trace filename too - when the
      report shows you a suspicious shortlist you will want to open the source.

      Guard: a pool-era check. If a verdict arrives with no pool recorded, mark
      it unattributable rather than pretending the uncited set is empty.
      """
      current_pool = None 
      strikes = []
      with open(trace_path , "r" , encoding = "utf-8") as  file :
          for line in file:
              if not line.strip():
                  continue
              data = json.loads(line)
              if data["event"] == "generated":
                  current_pool = data.get("pool")
              elif data["event"] in _STAGES:
                  ok = data.get("supported", data.get("passed"))
                  if not ok :
                      strikes.append(_strike(data,current_pool, trace_path))

          return strikes
                      
                  
              

                  
          

def uncited(strike: dict) -> list[str]:
    """
    TODO(you): the chunks that were available and not cited.

    Set arithmetic - but return a LIST, ordered as the pool was. Pool order is
    retrieval rank, and "the generator skipped the top-ranked chunk" is a
    different story from "it skipped the 9th", so throwing that order away
    throws away a signal you will want.
    """

    
    if strike["pool"] is None: 
      raise ValueError(f"{strike['trace']}: strike has no pool {strike['claim'][:60]  }")
    result = []
    citations = set(strike["citations"])
    for p in strike["pool"]:
        if p  not in citations:
            result.append(p)

    return result
            
        


def rank_uncited(claim: str, ids: list[str], corpus: str) -> list[tuple[str, float]]:
    """
    TODO(you): score each uncited chunk against the claim, best first.

    What you have to work with:
      - index, chunks = _get_index(corpus)
      - index.get_scores(tokens) -> one score per CORPUS chunk, positionally
        aligned with `chunks` (score[i] belongs to chunks[i]) - exactly the
        alignment you relied on in search_bm25()
      - _tokenize(text) -> the same tokenizer the corpus was built with

    The trap worth naming: do NOT build a fresh BM25Okapi over just these ~9
    chunks. BM25 weights a term by how RARE it is across the collection, and
    "rare across 9 documents" is noise. Score against the full-corpus index and
    select the positions you care about.

    You need chunk_id -> position. `chunks` is a list; build the lookup once.

    Return [(chunk_id, score), ...] sorted best first.
    """
    result = []
    query_tokens = _tokenize(claim)
    index , chunks = _get_index(corpus)
    scores = index.get_scores(query_tokens)
    positions = {c["chunk_id"]: i for i , c in enumerate(chunks)}
    for cid in ids:
        position = positions[cid]
        result.append((cid , scores[position]))
    ranked = sorted(result, key=lambda x: x[1], reverse=True)

    return ranked
    


def attribute(strike: dict, corpus: str) -> dict:
    """
    TODO(you): decide what this one strike LOOKS like, and say how confidently.

    Add to the strike dict:
      - the ranked uncited shortlist (from rank_uncited)
      - the best score among its CITED chunks, for comparison
      - a lead flag: did some uncited chunk outscore everything it cited?

    DECISION - where you put the bar. `best_uncited > best_cited` is the
    loosest possible rule and will flag the 0.12-vs-0.15 cases, which are two
    chunks that both say nothing. Consider requiring a margin, or a floor on
    the uncited score, so "a better chunk existed" means something. Whatever
    you pick, put the threshold in a named constant with a comment saying what
    you observed, not a magic number inline - same reason MIN_EVIDENCE_SCORE
    is named in critic/loop.py.
    """
    shortlist = rank_uncited(strike["claim"] , uncited(strike) , corpus)
    cited = rank_uncited(strike["claim"] , strike["citations"] , corpus)
    # `cited` is never empty: Field(min_length=1) on every citations list means a
    # validated claim always cited something. `shortlist` CAN be, when the claim
    # cited the whole pool -- and that is a no_lead by definition, since there
    # was nowhere else the support could have been.
    best_uncited = shortlist[0] if shortlist else None
    best_cited = cited[0]

    c = best_cited[1]

    if best_uncited is None:     verdict = "no_lead"
    elif c <= MIN_BASELINE:      verdict = "unmeasurable"
    elif best_uncited[1] - c >= MIN_MARGIN: verdict = "lead"
    else:                        verdict = "no_lead"

    return {
        **strike,                      # everything load_strikes captured, unchanged
        "verdict": verdict,            # "lead" | "no_lead" | "unmeasurable"
        "best_cited": best_cited,      # (chunk_id, score) - the baseline it's judged against
        "best_uncited": best_uncited,  # (chunk_id, score) or None if it cited the whole pool
        "shortlist": shortlist,        # every uncited chunk, ranked - what a human reads
    }


def strike_report(strikes: list[dict], attributed: list[dict]) -> str:
    """
    The summary, worded so it cannot be misquoted.

    Takes both lists because the two say different things: `strikes` is every
    strike loaded, `attributed` only the ones that had a pool to work from. The
    difference is the part a reader has to see -- a lead count over 21 strikes
    means something different from the same count over 198.

    Every count here is a LEAD count. "lead" means an uncited chunk scored
    materially better than the cited one, which is grounds to go read it, not a
    finding that the claim was mis-cited. The distinction is not pedantry: the
    top-scoring uncited chunk in the worked example (mealwise@0cf0a59, 8.61 vs
    0.12) turned out to be about Gemini scaffolding and supported nothing.
    """
    lines = ["", "=" * 52, "GROUNDED FORGE - STRIKE ATTRIBUTION", "=" * 52]

    # The denominator first. A strike from a pre-22fb586 trace recorded no pool,
    # so it cannot be attributed at all -- reporting only the attributable ones
    # would quietly present a third of the data as the whole of it.
    skipped = len(strikes) - len(attributed)
    lines.append(f"  strikes loaded     : {len(strikes)}")
    lines.append(f"  attributable       : {len(attributed)}")
    if skipped:
        lines.append(f"  not attributable   : {skipped}  (no pool recorded - traces predate {POOL_ERA})")

    stages = Counter(s["stage"] for s in attributed)
    if stages:
        lines.append("-" * 52)
        for stage, n in sorted(stages.items()):
            lines.append(f"  struck by {stage:<8}: {n}")

    verdicts = Counter(a["verdict"] for a in attributed)
    lines.append("-" * 52)
    lines.append(f"  had a better-scoring uncited chunk : {verdicts['lead']}")
    lines.append(f"  cited the best available           : {verdicts['no_lead']}")
    # Not folded into either number above. The cited chunk scored below
    # MIN_BASELINE, which means the scorer could not read it -- so no comparison
    # was possible. Counting these as leads overstates mis-citation; counting
    # them as no_lead manufactures evidence for over-reach. They are neither.
    lines.append(f"  not measurable (see MIN_BASELINE)  : {verdicts['unmeasurable']}")

    # The actual product. The counts say whether a prompt fix is worth a run;
    # these shortlists are what you read to find out whether it is real.
    for a in attributed:
        if a["verdict"] != "lead":
            continue
        lines.append("-" * 52)
        lines.append(f"  LEAD  {a['claim'][:70]}")
        lines.append(f"    trace   {a['trace']}  [{a['stage']}]")
        lines.append(f"    cited   {a['best_cited'][0]:<28} {a['best_cited'][1]:6.2f}")
        for cid, score in a["shortlist"][:3]:
            lines.append(f"    uncited {cid:<28} {score:6.2f}")
        lines.append(f"    struck because: {a['reason'][:80]}")

    lines.append("=" * 52)
    return "\n".join(lines)
