"""
Per-section evidence gathering for STAR answers (V2-M4).

A STAR answer needs FOUR different evidence pools, because its four parts live
in different sources:

    Situation / Task  -> the "why": constraints, goals   -> vault + docs
    Action            -> what you actually did           -> git
    Result            -> outcomes, eval numbers, ship    -> vault + docs
 
So instead of one retrieval, we run one retrieval PER SECTION, each biased
toward the source type where that kind of evidence lives.

YOU implement gather_evidence. Two real decisions live in it:

  DECISION 1 — how to bias each section's query.
    Two levers, and you can use either or both:
      (a) the FILTER: restrict source_type via `where=` (see SECTION_SOURCES)
      (b) the QUERY TEXT: the raw interview question is about the whole story;
          each section wants a different slant on it. Retrieving the Action
          pool with "...what did you build/change/fix" pulls different chunks
          than the bare question does.
    Think about what each section is actually looking for, then decide whether
    a filter alone is enough or the query text should shift too.

  DECISION 2 — the fallback when a filtered pool comes back thin.
    Filtering is a blade (you scored 9/10 on why). If a section's filtered
    retrieval returns nothing useful — no chunks, or nothing above a usable
    score — the section should NOT just fail: the evidence may simply live in
    an unexpected source. Decide what "thin" means and what to do about it
    (retry unfiltered? widen the source list? accept an empty pool and let the
    Generator report a gap?). Whatever you choose, be deliberate — this is the
    precision/recall tradeoff from M2 showing up in a real decision.

Return shape (what the Generator consumes):
    {
      "situation": [chunk, chunk, ...],
      "task":      [...],
      "action":    [...],
      "result":    [...],
    }
Each list is that section's evidence pool. A section may cite ONLY chunks from
its own pool, so pools decide what each part of the answer can say.

Note: `hybrid_search(query, corpus=..., k=..., use_rerank=True, where=...)` is
the retrieval entry point — you already threaded `where` through it in M3.
Chroma wants {"source_type": "git"} for one value, or
{"$or": [{"source_type": "vault"}, {"source_type": "docs"}]} for several.
"""
from retrieve.hybrid import hybrid_search


# Where each section's evidence usually lives.
SECTION_SOURCES = {
    "situation": ["vault", "docs"],
    "task":      ["vault", "docs"],
    "action":    ["git"],
    "result":    ["vault", "docs"],
}
SECTION_HINTS = {
    "situation": "the goal",
    "task":      "the constraint",
    "action":    "what I did",
    "result":    "the outcome",
}
WIDENED_SOURCES = {
    "situation": ["vault", "docs" , "git"],
        "task":      ["vault", "docs" , "git"],
        "action":    ["git" , "docs"],
        "result":    ["vault", "docs" , "git"],

}

# A pool this small (or this weak) means the filter probably hid the evidence.
MIN_POOL = 2


def _where_for(sources: list[str], repo: str | None = None) -> dict:
    """
    Build the Chroma filter for one or several source types, optionally scoped
    to one repo.

    Chroma takes a single condition bare but needs an explicit $and to combine
    two, so the repo clause can't just be another key in the same dict.
    """
    clause = ({"source_type": sources[0]} if len(sources) == 1
              else {"$or": [{"source_type": s} for s in sources]})
    if repo:
        return {"$and": [clause, {"repo": repo}]}
    return clause


def gather_evidence(question: str, corpus: str = "job", k: int = 6,
                    repo: str | None = None) -> dict[str, list[dict]]:
    """
    Retrieve one evidence pool per STAR section. See module docstring for the
    two decisions you need to make. 

    TODO(you) — V2-M4:
      1. For each of the four sections in SECTION_SOURCES:
           - build the filter with _where_for(...)
           - decide the query text for this section (DECISION 1)
           - retrieve with hybrid_search(..., where=..., use_rerank=True)
      2. Apply your fallback rule when the pool comes back thin (DECISION 2).
      3. Return the dict of four pools.
    """
    pools= {}
    
    
    for section , sources in SECTION_SOURCES.items():
        where = _where_for(sources, repo)
        query = question + " " +  SECTION_HINTS[section]
        pool = hybrid_search(query,corpus, k = k , use_rerank=True , where = where)
        seen = {c["chunk_id"] for c in pool}
        if len(pool) < MIN_POOL:
            # widening relaxes the SOURCE TYPE only -- repo stays pinned. A thin
            # pool means the evidence lived in an unexpected source, never that
            # another project's history is fair game.
            where = _where_for(WIDENED_SOURCES[section], repo)
            result = hybrid_search(query,corpus, k = k  , use_rerank=True , where = where)
            for chunk in result:
                if len(pool) >= k: 
                    break
                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"]) 
                    pool.append(chunk)
                
        pools[section] = pool
    return pools

