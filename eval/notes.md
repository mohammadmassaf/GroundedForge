# Eval notes — v1 (study mode, networks corpus)

Corpus: 3 course PDFs (I2208 Computer Networks), 214 chunks of ~700 chars / 100 overlap.
Eval set: 20 fixed questions with known-right (source_file, page) locations — `eval/eval_set.json`.
Run: `python main.py eval --corpus networks` (add `--limit N` to cap the LLM-costed grounding half).

## Results

| Metric | Score | Notes |
|---|---|---|
| recall@3 | 90% | full 20-question set |
| recall@5 | 95% | full set |
| recall@10 | **100%** | full set — every eval answer is findable |
| grounding | **92.7%** | 41 claims over 19/20 questions (1 question failed generation and was excluded); 3 struck |

All three full-run strikes follow the patterns below: two are the formula-vs-worked-result
strictness case, one is an evidence-coverage case. No fabricated claim passed the Critic.

recall@10 = 100% means quality problems downstream are never "the chunk isn't findable" —
they are ranking (the @3↔@10 gap, the M7 re-ranker's target) or generation issues.

## The two struck claims, read closely

1. **"Propagation time = Distance × 1000 / speed"** — the slide's `×1000` is a km→m unit
   conversion inside a worked example, not part of the formula. The Generator's phrasing
   didn't match the evidence's exact form and the Critic (rule 2: computed values need the
   shown result) struck it. Arguably over-strict — the claim is pedagogically fine.
2. **"data vs communication ... not explicitly discussed"** — an eval question (e5) that
   slightly overreaches what p.7 of Part-1 actually states. An honest strike caused by an
   imperfect eval question rather than a generation failure.

Both strikes are the **Critic's precision/recall tradeoff** made visible:

- Stricter Critic → lower grounding %, stronger guarantee (false positives cost valid items)
- Looser Critic → higher grounding %, weaker guarantee (false negatives leak hallucinations)

The score is only meaningful **alongside the strictness policy that produced it** — a higher
number from a looser judge certifies less. Calibrating the policy = hand-checking the claims
whose verdicts *flip* when a rule changes, against the actual slides.

## Retrieval-confidence threshold

Observed cosine-similarity bands (query → top-1 score):

- On-corpus topics ("transmission and propagation delay"): comfortably retrieved, quiz fully grounded
- Off-corpus topic ("TCP vs UDP"): top score **0.387**, all chunks only loosely related

v1 does **not** enforce a hard score threshold before generation. The guard is behavioral
instead: the Generator's "never invent" rule produced **zero items** on the off-corpus topic
(schema's min-1-item floor turned that into a loud failure), and the Critic independently
strikes anything unsupported. A threshold near ~0.45 would have cleanly separated the two
cases above, but with n=1 off-corpus observation we note the tradeoff rather than tune:

- threshold too high → real questions phrased unlike the slides get refused (recall cost)
- threshold too low → junk context reaches the Generator and the Critic carries the load (precision cost)

Revisit with more off-corpus probes in M7, where hybrid retrieval changes the score
distribution anyway.

## M7: hybrid retrieval (BM25 + RRF + cross-encoder re-ranking)

Setup: same 20-question eval set, same networks corpus, three retrieval modes via the new
`--retrieval` flag on `eval`. The hybrid pipeline fetches 20 candidates from vector search
and 20 from BM25 (rank_bm25), fuses them with Reciprocal Rank Fusion (positions, not scores —
cosine and BM25 aren't comparable currencies), and optionally re-scores all fused candidates
with a cross-encoder (ms-marco-MiniLM-L-6-v2) before taking top-k. Wide-then-narrow: every
stage fetches more than it keeps.

| mode | recall@3 | recall@5 | recall@10 |
|---|---|---|---|
| vector (v1 baseline) | **90%** | 95% | 100% |
| hybrid (BM25 + RRF) | 80% | 90% | 100% |
| hybrid + rerank | **90%** | 95% | 100% |

### Analysis

- **Hybrid alone dropped recall@3 (90% → 80%).** Per-question diff: hybrid flipped 4
  questions — lost 3, gained 1. The 3 losses share a shape: abstract, wordy eval questions
  ("what characteristics of a network system must be evaluated…"). BM25 matches exact
  tokens, and words like "characteristics" / "evaluated" appear all over a textbook, so
  BM25's top ranks were topically scattered and RRF — which trusts both lists equally —
  diluted vector's correct top-3 down to fused ranks 4–6.
- **Re-ranking recovered the full 90%.** recall@10 stayed 100% in every mode: the right
  chunk never left the 20-candidate pool, and the cross-encoder (which reads query and
  chunk *together*, token-level) promoted it back up. This is two-stage retrieval doing
  exactly its job — fusion widens the net, the re-ranker sharpens the order. The
  re-ranker's ceiling is the candidate set's recall, and here the ceiling was 100%.
- **The one hybrid win is the most instructive case:** the eval question misspelled
  "sheilded twisted pair". Vector@3 missed (embeddings drifted to p.60–61); hybrid@3 hit
  p.62 at rank 2 because BM25 anchored on the exact tokens "twisted pair". That's the
  exact-token complementarity hybrid retrieval exists for — this eval set just barely
  exercises it.

### Conclusion

On this corpus there was no headroom: recall@10 was already 100%, and the paraphrase-style
questions are vector search's home turf. Hybrid + rerank **matched** the baseline rather
than beating it — the real gain is robustness to keyword-shaped queries (acronyms, error
codes, formula names, misspellings), which the headline number doesn't reward. A negative-ish
result, but only an eval harness makes it visible at all; without one, "added hybrid
retrieval" would have shipped as an assumed improvement while silently costing recall@3.

## Harness behavior worth keeping

- Per-item generation failures (`GenerationError`) are skipped and counted (`failed` in the
  report) — one flaky item costs one question, not the run.
- Rate limits (`RateLimitError`) stop the run early but keep partial tallies — the report's
  denominators stay honest about coverage.

---

# Eval notes — v2 (job mode, repo corpus)

Corpus: two of my own repos (`mealwise`, `grounded-forge`) ingested as commit messages,
README sections, and project notes — not prose documents. Eval set: 15 questions with
known-right locations plus 6 adversarial traps — `eval/eval_set_job.json`.
Run: `python main.py eval --corpus job --retrieval rerank` (`--no-traps` to skip the
adversarial half, `--limit N` to cap the LLM-costed grounding half).

## Study mode vs job mode, side by side

| Metric | study (networks, 20 q) | job (repo, 15 q) |
|---|---|---|
| recall@3 | 90% | 60% vector · **73.3%** rerank |
| recall@5 | 95% | 73.3% vector · 80% rerank |
| recall@10 | **100%** | 93.3% (all modes) |
| grounding | **92.7%** (41 claims, 19/20 q) | **87.1%** (31 claims, 15/15 q) |
| inflation-catch | not measured in v1 | **100%** (6/6 traps) |

Same engine, same Critic, different corpus shape. Job mode is harder on both axes, and the
two axes fail for unrelated reasons — which is the entire argument for measuring them
separately.

## Traps: the measurement that makes the other two trustworthy

recall@k and grounding % both grade the system on good inputs, so neither can detect a
broken Critic. A Critic that answers "supported" to everything scores **100% grounding** —
the best possible number from the worst possible judge. The traps close that hole: 6
authored claims with known defects, each tagged with the stage that should catch it.

All 6 were struck, each by its authored stage — 3 by `quant`, 3 by `critic`, no stage
mismatches:

- `quant` caught the rounding inflation (`~93%` where evidence says `92.7%`), a fabricated
  `40%` latency figure, and a right-number-wrong-chunk citation (`188`).
- `critic` caught an unused-skill claim (Kubernetes, absent from the corpus), a
  misattribution whose every *number* is real (`100% recall@3`, which quant must pass by
  design), and a premise leaked from the question itself.

The stage split is the useful part, not just the rate. A t5-style trap caught by `quant`
would mean the haystack isn't what we think it is, so the report prints stage mismatches
separately rather than folding them into the headline.

## Finding 1: job-mode strikes are abstraction, not fabrication

First full job run scored **70.4%**, well under study mode's 92.7%. Reading the strike
reasons, it was not eight different problems — **9 of 10 strikes said "does not explicitly
state."** Every one was the same move: the bullet sat one level of abstraction above its
evidence.

```
evidence : "feat: add auth with JWT"  +  ".env holds SECRET_key for JWT"
struck   : "Implemented JWT authentication"
```

Nothing there is false. The Critic's standard is presence in the cited evidence, not truth,
so a fair inference is still a strike. What survived were near-verbatim lifts:

```
evidence : "stop retrying on 429, show a rate limit message to the user"
kept     : "Stopped retrying on 429 errors and displayed rate limit messages"
```

**Why this is structural, not a bug.** Course PDFs are explanatory prose that already states
things at claim level, so a grounded quiz answer never has to abstract. Commit subjects are
*labels* — `feat: add auth with JWT` is seven words covering a week of work. A CV bullet that
stays at that level isn't a CV bullet. So the artifact type and the verifier want opposite
things: bullets are supposed to generalise, and a grounding checker punishes generalisation.
70.4% was the harness correctly reporting that tension, not a regression.

Worth noting the prompt was arguing with itself: rule 6 said *"never just reword a commit
subject"*, which is exactly what every surviving bullet did. The Critic wins that argument
because it runs second.

## Finding 2: fixing it at the Generator, and the injection that fix caused

The wrong lever is loosening the Critic — relax it enough to accept
"SECRET_key → implemented JWT authentication" and it very likely also accepts the Kubernetes
trap. Strictness is what the traps are measuring; trading it for a prettier grounding score
trades the guarantee for the number.

The right lever is the Generator. Added rule 7 ("stay at the evidence's level" — combine
facts across cited chunks, never rename or upgrade them), with a carve-out disambiguating it
from rule 6: **rule 6 governs substance, rule 7 governs vocabulary.**

Measured, same corpus, prompt as the only variable:

| | before | after |
|---|---|---|
| headline | 70.4% (27 claims, 11/15 q, rate-limited) | **87.1%** (31 claims, 15/15 q) |
| first 5 items, like-for-like | 53.8% (7/13) | **80.0%** (8/10) |
| questions with zero strikes | — | 12 of 15 |

The headline pair isn't a clean comparison — the first run stopped early at 11 questions —
so the first-5 like-for-like row is the defensible one. Both runs retrieved with
hybrid+rerank.

**The instructive failure:** I wrote a worked example into rule 7 whose "RIGHT" answer was a
finished, plausible MealWise bullet. The generator emitted it character-for-character:

```
prompt : RIGHT: "Added JWT auth in a dedicated auth router, keyed by a SECRET_key"
output :        "Added JWT auth in a dedicated auth router, keyed by a SECRET_key"
```

Rule 1 says *use ONLY the context provided by the user* — but the system prompt is context
too. On the one question whose evidence hadn't been retrieved (see Finding 3), the closest
thing to an answer anywhere in the model's context was the example, so it copied it. A
few-shot example in a grounded generator is an injection vector.

Rule now in force: **examples must use a fictional domain and must not introduce any entity
absent from the evidence line shown beside them**, and one example's correct answer is
*write nothing* — thin evidence should produce fewer bullets, not a well-hedged one. The
earlier version taught the opposite by demonstrating a rescue.

## Finding 3: the re-ranker discards the right evidence (j1)

j1 — *"How is authentication implemented in MealWise?"* — is both the sole recall@10 miss in
rerank mode and the source of 2 of the 4 remaining strikes. Excluding it, the run is 27 kept
/ 2 struck = **93.1%**.

It is a *retrieval* fault, and specifically a re-ranking fault. Rank of the two expected
chunks (`mealwise@aa2b173`, `mealwise README § API reference`):

| mode | ranks of expected evidence |
|---|---|
| vector | **3 and 4** |
| hybrid (RRF) | 5 and 7 |
| hybrid + rerank | **absent from top-10** |

The cross-encoder's rank 1 was `grounded-forge@7f87ab2` at −0.146, while every genuine
MealWise chunk scored below −3.0. That chunk is a Grounded Forge commit about the quant
pre-check. It contains the word "implemented" and the literal string `mealwise@8912ac4`
(quoted as a sha-pollution example) — two of the query's three content words — and **zero
mention of authentication**. It outranked the actual auth commit.

It is also neatly self-referential: the commit that stopped shas polluting the *quant*
haystack is now polluting the *retrieval* haystack.

**Root cause.** The job corpus mixes two repos, and one of them talks about the other in its
commit messages. Nothing in the pipeline treats "about MealWise" as a constraint — it is a
soft topical hint that a relevance scorer is free to overrule. This is the same argument as
the per-section STAR pools one level up: an evidence pool should be an authorization
boundary, not a bag of context. A bullet about MealWise must not be evidenced by a Grounded
Forge commit, however well it scores.

**Correction to an earlier hypothesis.** I first attributed j1 to a vocabulary gap (`auth` in
the commit vs `authentication` in the question) and queued prefix expansion as the fix.
Measurement says otherwise: vector search ranks both expected chunks at 3 and 4. The evidence
is findable. Prefix expansion would not have fixed j1.

**Mode tradeoff, honestly.** rerank is the better mode overall (73.3% vs 60.0% recall@3) and
it fixes the "deliverable" miss that vector and hybrid both fail. It is also the only mode
that loses j1. Averages hide that; the per-question miss list is what surfaced it.

## Harness caveat found while investigating

`grounding_eval` hardcodes hybrid+rerank at k=8, while `retrieval_eval` honours
`--retrieval`. A single report can therefore describe **two different retrievers** — the
recall rows from the flagged mode, the grounding row always from rerank. The numbers above
are consistent only because the reported run used `--retrieval rerank`. Threading the mode
through `grounding_eval` is open.

## Open items

- Project-scoped retrieval for job mode (metadata filter by source repo) — the fix Finding 3
  points at.
- Thread `--retrieval` through `grounding_eval` so both halves of the report describe one
  system.
- j6 *"Where is MealWise deployed?"* failed generation in the first run (empty bullet list
  twice → `GenerationError`) while the gap check passed. Two distinct no-answer paths exist —
  pre-generation gap and post-generation empty — and only one is reported as a gap.
