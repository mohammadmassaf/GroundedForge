"""
The multi-agent loop: Generator -> Critic -> Refiner.

run_loop() orchestrates one full cite-or-strike run:

    round 1: generate n items
             critique each item          -> kept / struck
    round 2: if short and rounds remain: regenerate the missing count
             critique the new items too  -> kept / struck
    ...
    stop when: enough kept items, OR max_rounds exhausted.

Returns (kept, struck):
    kept   = list[QuizItem]                    — survived the Critic
    struck = list[tuple[QuizItem, str reason]] — failed, with why

The Refiner here is POLICY CODE, not an LLM: deciding what to do with
verdicts (drop, regenerate, stop) is deterministic logic — another case of
"move correctness out of the LLM and into code."

Every step logs to the Tracer, so a run is reconstructable from its trace.
"""
from generate.generator import generate , generate_guide , generate_bullets , generate_star ,  EmptyGeneration
from generate.schema import QuizItem , GuideSection
from critic.critic import check_claim
from critic.quant import check_quantities
from critic.trace import Tracer

MAX_ROUNDS = 2

# Honest gap reporting: if the best retrieved chunk is weaker than this, the
# corpus doesn't really cover the topic — say so instead of generating filler.
# Threshold gets tuned against the eval set in M5.
MIN_EVIDENCE_SCORE = 0.25


def run_loop(topic: str, chunks: list[dict], n: int = 5,
             corpus: str | None = None) -> tuple[list, list]:
   """
    Cite-or-strike for quiz items: generate, critique each item against ONLY
    the chunks it cited, keep the supported ones and strike the rest.

    Runs up to MAX_ROUNDS, asking each round only for the shortfall (n minus
    what survived), so a round that loses three items regenerates three rather
    than restarting. Stops early once enough items are kept.

    Unlike bullets, an empty generation here is still a hard failure rather
    than a gap - see the note in the open-work list. v1 claims and measures
    that an off-corpus topic yields zero items loudly, and that behaviour is
    published.
    """
   tracer = Tracer("quiz", corpus=corpus)
   by_id = {c["chunk_id"]: c for c in chunks}
   kept ,struck = [] , []
   for round_num in range(1, MAX_ROUNDS + 1):
       need = n - len(kept)
       if need == 0: break
       quiz = generate(topic,chunks,n = need)
       tracer.log("generated" , round = round_num , count = len(quiz.items),
                  pool=[c["chunk_id"] for c in chunks])
       for item in quiz.items:
           cited_chunks = [by_id[cid] for cid in item.citations]
           verdict = check_claim(item.question, item.answer,cited_chunks)
           tracer.log("critic_verdict", round=round_num,
                         question=item.question, citations=item.citations,
                         supported=verdict.supported, reason=verdict.reason)
           if verdict.supported: kept.append(item)
           else: struck.append((item, verdict.reason))
   tracer.log("done", kept=len(kept), struck=len(struck))
   print(f"Trace: {tracer.path}")

   return kept,struck
   
def run_bullets_loop(topic: str, chunks: list[dict], n: int = 5,
                     corpus: str | None = None) -> tuple[list, list, str | None]:
    """
    Cite-or-strike for CV bullets. Same shape as run_loop, with TWO differences:

    1. TWO-STAGE verification per bullet:
         stage 1 (deterministic): check_quantities — any figure not literally in
                  the cited evidence is struck immediately, WITHOUT an LLM call
         stage 2 (LLM): check_claim — semantic support, only for bullets that
                  survived stage 1
       Cheap+certain first, expensive+fuzzy second.

    2. Honest gap reporting: if retrieval is too weak to support the topic at
       all, return a gap message instead of generating filler.

    Returns (kept, struck, gap): gap is None on a normal run, or a message when
    the corpus doesn't cover the topic.
    """
    tracer = Tracer("bullets", corpus=corpus)
    by_id = {c["chunk_id"]: c for c in chunks}

    # --- honest gap reporting: is there any real evidence to work from? ---
    best = max((c.get("vector_score", 0) for c in chunks), default=0)
    if not chunks or best < MIN_EVIDENCE_SCORE:
        gap = (f"Insufficient evidence in corpus for '{topic}' "
               f"(best match {best:.3f} < {MIN_EVIDENCE_SCORE}).")
        tracer.log("gap", topic=topic, best_score=best)
        print(f"Trace: {tracer.path}")
        return [], [], gap

    kept, struck = [], []
    seen: set[str] = set()          # normalized bullet text, for dedupe
    for round_num in range(1, MAX_ROUNDS + 1):
        need = n - len(kept)
        if need == 0:
            break
        # tell the Generator what it already produced, or a top-up round just
        # re-emits its best bullet
        try :
            result = generate_bullets(topic, chunks, n=need,
                                  avoid=[b.text for b in kept])
        except EmptyGeneration  :
            if kept:
                tracer.log("no_material" , round = round_num , kept = len(kept) , requested = need)
                break
            else:
                gap = f"no material for '{topic}'"
                tracer.log("no_material" , round = round_num , kept = len(kept) , requested = need)
                print(f"Trace: {tracer.path}")
                return [] , struck , gap
        # The pool is logged alongside the verdicts because a strike has two very
        # different causes that read identically in the reason string: the
        # support was NOWHERE in the pool (a retrieval miss), or it was in the
        # pool under a chunk the claim didn't cite (a mis-citation). Without the
        # pool you cannot tell them apart, and they need opposite fixes.
        tracer.log("generated", round=round_num, count=len(result.bullets),
                   pool=[c["chunk_id"] for c in chunks])

        for bullet in result.bullets:
            key = " ".join(bullet.text.lower().split())
            if key in seen:         # safety net if the model ignores `avoid`
                tracer.log("duplicate_dropped", round=round_num, bullet=bullet.text)
                continue
            seen.add(key)

            cited_chunks = [by_id[cid] for cid in bullet.citations]

            # stage 1 — deterministic, no LLM
            ok, reason = check_quantities(bullet.text, cited_chunks)
            tracer.log("quant_check", round=round_num, bullet=bullet.text,
                       citations=bullet.citations, passed=ok, reason=reason)
            if not ok:
                struck.append((bullet, f"[quant] {reason}"))
                continue

            # stage 2 — LLM Critic, only for survivors
            verdict = check_claim("", bullet.text, cited_chunks)
            tracer.log("critic_verdict", round=round_num, bullet=bullet.text,
                       citations=bullet.citations,
                       supported=verdict.supported, reason=verdict.reason)
            if verdict.supported:
                kept.append(bullet)
            else:
                struck.append((bullet, verdict.reason))

    tracer.log("done", kept=len(kept), struck=len(struck))
    print(f"Trace: {tracer.path}")
    return kept, struck, None


def run_star_loop(question: str, pools: dict[str, list[dict]],
                  corpus: str | None = None) -> tuple[object, list]:
    """
    Cite-or-strike for a STAR answer. Verifies each of the four sections
    independently, with the same two-stage check as bullets.

    Returns (answer, flagged): `flagged` is a list of (section_name, reason) for
    sections that failed verification. Unlike bullets we DON'T drop a failed
    section (a STAR answer with no Result is broken) — we keep it and flag it,
    so the artifact shows what isn't trustworthy.
    """
    tracer = Tracer("star", corpus=corpus)
    by_id = {c["chunk_id"]: c for pool in pools.values() for c in pool}

    answer = generate_star(question, pools)
    tracer.log("generated", question=question,
               pool_sizes={k: len(v) for k, v in pools.items()},
               pools={k: [c["chunk_id"] for c in v] for k, v in pools.items()})

    flagged = []
    for name, section in answer.sections():
        cited_chunks = [by_id[cid] for cid in section.citations]

        # scope check: did this section cite outside its own pool?
        own_ids = {c["chunk_id"] for c in pools.get(name.lower(), [])}
        stray = [cid for cid in section.citations if cid not in own_ids]
        if stray:
            reason = f"cited outside its evidence pool: {stray}"
            flagged.append((name, reason))
            # Logged in the SAME field shape as a critic_verdict, deliberately.
            # This event used to carry only `stray`, so the trace recorded
            # strictly less than the function returned: a replay of the trace
            # could not see the strike at all, and it surfaced only as a
            # disagreement with this run's own `done` count. `stray` is kept
            # alongside `citations` because it says WHICH of them broke scope --
            # unifying the shape must not cost information.
            tracer.log("scope_violation", section=name,
                       citations=section.citations, stray=stray,
                       supported=False, reason=reason)
            continue

        # stage 1 — deterministic
        ok, reason = check_quantities(section.text, cited_chunks)
        tracer.log("quant_check", section=name, citations=section.citations,
                   passed=ok, reason=reason)
        if not ok:
            flagged.append((name, f"[quant] {reason}"))
            continue

        # stage 2 — LLM Critic
        verdict = check_claim("", section.text, cited_chunks)
        tracer.log("critic_verdict", section=name, citations=section.citations,
                   supported=verdict.supported, reason=verdict.reason)
        if not verdict.supported:
            flagged.append((name, verdict.reason))

    tracer.log("done", flagged=len(flagged))
    print(f"Trace: {tracer.path}")
    return answer, flagged


def run_guide_loop(topic, chunks, corpus: str | None = None) -> tuple[list, list]:
   tracer = Tracer("guide", corpus=corpus)
   by_id = {c["chunk_id"]: c for c in chunks}
   struck = [] 
   kept_sections = []
   guide = generate_guide(topic, chunks) 
   tracer.log("generated", sections=len(guide.sections))
   for section in guide.sections:
       survivors = []
       for claim in section.claims:
           cited_chunks = [by_id[cid] for cid in claim.citations]
           verdict = check_claim("" , claim.text , cited_chunks)
           tracer.log("critic_verdict", heading=section.heading, claim=claim.text,
                      citations=claim.citations,
                      supported=verdict.supported, reason=verdict.reason)
           if verdict.supported: survivors.append(claim)
           else : struck.append((claim , verdict.reason))
       if survivors:
          kept_sections.append(GuideSection(heading=section.heading, claims=survivors))
   tracer.log("done", kept=len(kept_sections), struck=len(struck))
   
   print(f"Trace: {tracer.path}")
   return kept_sections , struck