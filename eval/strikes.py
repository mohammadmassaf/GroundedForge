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
      Every struck claim in one trace file, normalized.

      The difficulty is that the two halves travel on DIFFERENT lines: `pool`
      rides on the `generated` event once per round, while `citations` and the
      verdict ride on each `quant_check` / `critic_verdict`. Neither line is
      enough alone, so the most recent pool is carried forward as the file is
      read and attached to every verdict met after it - replaying an event log
      in order to reconstruct state.

      A verdict with no pool recorded is marked unattributable rather than
      given an empty one. 222 of the traces on disk predate 22fb586 and never
      logged it, and "we did not record this" must not read as "there was
      nowhere else to look".
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
    The chunks that were available to the generator and not cited - i.e.
    where the support could have been instead.

    Returns a LIST, in pool order, not a set: pool order is retrieval rank, and
    "it skipped the top-ranked chunk" is a different story from "it skipped the
    9th".

    Raises rather than returning [] for an unattributable strike. The caller
    already holds `attributable` and is expected to check it; reaching here
    without doing so is a bug at the call site, which is why this is a plain
    ValueError nobody is meant to catch.
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
    Score each of `ids` against the claim, best first, as [(chunk_id, score)].

    Scoring, not retrieval: the candidate set is already fixed by the caller,
    so unlike search_bm25 this ranks a given list rather than searching for one.
    Used for both sides of the comparison - hand it the uncited chunks for the
    shortlist, hand it the citations for the baseline.

    Scored against the FULL-corpus index, never a fresh BM25 over just these
    few chunks. BM25 weights a term by how rare it is across the collection,
    and rarity measured over nine documents is noise - `and` appears in 117 of
    244 chunks here and is correctly worth almost nothing, but over nine it
    would look distinctive and a chunk could win on it.

    get_scores returns one score per corpus chunk, positionally aligned with
    `chunks`, so chunk_id -> position is the bridge between an id and its score.
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
    Score what the claim cited, score what it ignored, compare, and label -
    returning the strike with the shortlist, both baselines and a verdict added.

    The verdict is three-way rather than a flag, because there are three real
    states and only two of them are answers:

      lead         an uncited chunk scored materially better - go read it
      no_lead      it cited the best thing available - the strike is over-reach
      unmeasurable the cited chunk scored below MIN_BASELINE, so the scorer
                   could not read it and no comparison was possible

    Forcing that third state into a boolean breaks in both directions: as a
    lead it overstates mis-citation, as a no_lead it manufactures evidence for
    over-reach out of cases never measured. Same call already made for gapped
    questions in grounding_eval, and for EmptyGeneration being a sibling of
    GenerationError rather than a subclass.

    Checks unmeasurable FIRST. The four SECRET_key strikes carry the LARGEST
    margins in the set precisely because their baseline is broken, so a margin
    test run first would rank the least trustworthy cases highest.
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


def belongs_to(strike: dict, corpus: str) -> bool:
    """
    Does this strike come from a run over `corpus`?

    Decided by whether its chunk ids actually resolve against
    chunks/<corpus>.json - a measurement rather than an inference. The
    alternative is reading the corpus off the filename (`quiz_*` => networks),
    which is a guess encoded in a naming convention and ambiguous anyway, since
    quiz runs exist over both `networks` and `demo`.

    Traces written after the run_start header landed say so outright, but 243
    of the traces on disk predate it, so this has to work without them.

    It also catches the case that will arrive with the next re-ingest: if chunk
    ids shift, an old trace stops resolving and is reported as foreign instead
    of scoring its claims against chunks that no longer mean the same thing.
    """
    ids = {c["chunk_id"] for c in _get_index(corpus)[1]}
    return bool(strike["pool"]) and all(cid in ids for cid in strike["pool"])


def sweep(corpus: str, pattern: str = "*.jsonl") -> tuple[list, list, int]:
    """
    Load every trace matching `pattern`, attribute the ones belonging to
    `corpus`, and return (all_strikes, attributed, foreign_count).

    Three buckets, kept apart on purpose: a strike can be unattributable (no
    pool recorded), foreign (a different corpus), or attributed. Only the third
    is measurable, and the report has to show the other two or a lead count
    over 21 strikes gets read as a lead count over 198.
    """
    all_strikes, attributed, foreign = [], [], 0
    for path in sorted(TRACE_DIR.glob(pattern)):
        for strike in load_strikes(path):
            all_strikes.append(strike)
            if not strike["attributable"]:
                continue
            if not belongs_to(strike, corpus):
                foreign += 1
                continue
            attributed.append(attribute(strike, corpus))
    return all_strikes, attributed, foreign


def strike_report(strikes: list[dict], attributed: list[dict],
                  foreign: int = 0) -> str:
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
    no_pool = sum(1 for s in strikes if not s["attributable"])
    lines.append(f"  strikes loaded     : {len(strikes)}")
    lines.append(f"  attributed         : {len(attributed)}")
    if no_pool:
        lines.append(f"  no pool recorded   : {no_pool}  (traces predate {POOL_ERA})")
    if foreign:
        lines.append(f"  another corpus     : {foreign}  (chunk ids do not resolve here)")

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
