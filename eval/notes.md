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

## Harness behavior worth keeping

- Per-item generation failures (`GenerationError`) are skipped and counted (`failed` in the
  report) — one flaky item costs one question, not the run.
- Rate limits (`RateLimitError`) stop the run early but keep partial tallies — the report's
  denominators stay honest about coverage.
