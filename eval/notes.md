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
