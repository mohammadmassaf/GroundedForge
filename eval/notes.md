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

> **Superseded 2026-08-20.** That 92.7% was measured on `llama-3.3-70b-versatile`, retired by
> Groq. Re-measured on `gpt-oss-20b`: **70.0%** (35 kept / 15 struck, 50 claims, 20/20
> questions). See "Finding 7" at the end of the v2 notes — the drop is the same defect rule 7a
> fixed in job mode, which quiz mode never received.

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

Job-mode retrieval below is measured **after** repo scoping (Finding 3) and the vault
re-ingest; the pre-scoping figures live in that finding.

| Metric | study (networks, 20 q) | job (repo, 15 q) |
|---|---|---|
| recall@3 | 90% | 60% vector · 60% hybrid · **73.3%** rerank |
| recall@5 | 95% | 66.7% vector · 73.3% hybrid · **80%** rerank |
| recall@8 *(k the Generator reads)* | **100%** | 86.7% vector · **93.3%** hybrid · 86.7% rerank |
| recall@10 | **100%** | 93.3% vector · 93.3% hybrid · **100%** rerank |
| grounding | **92.7%** (41 claims, 19/20 q) | **87.1%** (31 claims, 15/15 q) |
| inflation-catch | not measured in v1 | **100%** (6/6 traps) |

**Both grounding figures above were produced by `llama-3.3-70b-versatile`, which Groq retired
on 2026-08-17.** They are historical. Job mode on `gpt-oss-20b` scores **96.7%** (29 kept /
1 struck, 30 claims) over all 15 questions. On the 8-question window that isolates the prompt
change, rule 7a took grounding from 50.0% to 93.8% with nothing else varying. Finding 5 has the full story. recall@k is unaffected in every
mode: retrieval is entirely local and reproduced exactly through the model swap.

The 87.1% grounding figure predates repo scoping and the k=8 finding — it is the best
*measured* job-mode number, not the current system's. Re-measuring is blocked on the Groq
daily token limit, not on anything unknown.

Same engine, same Critic, different corpus shape. Job mode is harder on both axes, and the
two axes fail for unrelated reasons — which is the entire argument for measuring them
separately.

The **recall@8 row is the sharpest single comparison**: on the study corpus the Generator
receives every piece of evidence it needs (100%), on the job corpus it does not (86.7% in the
mode actually used). Any grounding gap between the two modes therefore starts with retrieval
already behind, before the Generator writes a word.

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

## Finding 4: the eval measured a k the system does not use

Scoping fixed the cross-project contamination — recall@10 reached **100%**, no misses in
any question. j1 still failed. Scoped, its expected evidence ranks:

| mode (scoped to `repo: mealwise`) | rank of j1's evidence |
|---|---|
| vector | 3 and 4 |
| hybrid (RRF) | 3 and 9 |
| hybrid + rerank | 9 only |

The Generator retrieves **k=8**. The evidence sat at rank 9. So `recall@10 = 100%` was true
and useless at the same time: the harness reported success at a k the system never reads.
Two conclusions that look identical in a report and are not — *the evidence is findable* vs
*the Generator can cite it*.

`KS` now includes 8, the k `grounding_eval` actually retrieves with, and that row inverts
the ranking:

| mode | @3 | @5 | **@8 (feeds generation)** | @10 |
|---|---|---|---|---|
| vector | 60.0% | 66.7% | 86.7% | 93.3% |
| hybrid | 60.0% | 73.3% | **93.3%** | 93.3% |
| hybrid + rerank | **73.3%** | 80.0% | 86.7% | **100%** |

The cross-encoder wins the two numbers a README would quote and loses the one that decides
what the Generator can cite. Same pattern as the v1 M7 result — re-ranking sharpens the very
top of the list — except here the artifact reads eight chunks deep, so a sharper top-3 is
worth less than a fuller top-8.

**Consequence:** job mode should generate with `--retrieval hybrid`, not `rerank`. Untested
against grounding % so far (Groq TPD exhausted); the mode is now threaded through
`grounding_eval`, so it is a one-flag experiment.

## Finding 5: a prompt tuned to one model does not transfer

Groq retired `llama-3.3-70b-versatile` on 2026-08-17 (404 `model_not_found`, mid-run) and no
Llama chat model remains in the catalogue. Switched to `openai/gpt-oss-20b`, which keeps its
reasoning in a separate field so `content` stays clean JSON and the parser needed no changes.

Validated the swap on the **traps first** — 6/6 struck, 3 quant / 3 critic, no stage
mismatches, identical to the retired model. That cost ~5k tokens against ~48k for a grounding
run, and it is the only measurement that can catch a rubber-stamp Critic. Cheap adversarial
check first, expensive measurement second, is worth keeping as a habit.

Then grounding, same 8 questions, same `rerank --gen-k 10`: **50.0%** (11 kept / 11 struck),
against 84.0% on the retired model. Both agents changed at once, so the number alone cannot
say whether the Generator got worse or the Critic got stricter. The traces can.

**It was neither.** Every one of the 11 strikes has the same shape — the Critic concedes the
main claim and rejects an appended clause:

```
"confirms cross-checking against USDA FDC   BUT does not explicitly state that USDA_KEY is used..."
"confirms tables are auto-created on startup BUT does not specify PostgreSQL..."
"shows generate_meal_plan was made async     BUT does not explicitly state that..."
```

Not mis-citation either: 10 of 11 strikes cite 2–3 chunks, and no reason says the support sat
elsewhere in the pool. The bullets are "X, enabling Y" where only X is in the evidence.

Then the word counts explained why:

| | mean words/bullet | grounding |
|---|---|---|
| llama-3.3-70b | **8.1** | 84.0% |
| gpt-oss-20b | **13.4** | 50.0% |

**Rule 6 asks for 12–25 words. The old model was violating that floor, and the 84% depended
on it.** Eight-word bullets say one thing, and one thing is easy to ground. The old traces
show the same gradient internally: struck bullets averaged 10.2 words, kept ones 7.6. Today
every bullet sits in the compliant band (kept 13.2, struck 13.5), so length no longer
discriminates — nothing is short any more.

### The real defect: rules 6 and 7 contradict each other

- **Rule 6** — 12–25 words, "name the SPECIFIC technical content"
- **Rule 7** — stay at the evidence's level, never add what it does not state

A commit subject is seven words. There is often no 12–25 words of *grounded* content to be
had, so obeying rule 6 forces padding, and padding is unsupported by construction. The two
rules were never compatible; llama-3.3-70b hid it by under-complying with rule 6, and
gpt-oss-20b exposed it by obeying.

This is the third time a prompt written against observed behaviour has misfired (after rule
7's example becoming an answer, and rule 5's becoming a prohibition) — but the first caused by
a model swap rather than by wording. **A prompt is tuned to a model, and that tuning is not
portable.** The eval is the only reason this was visible at all; without it the swap would
have shipped as "same system, new model" while grounding fell by 34 points.

Fix: drop rule 6's word floor so a bullet may be as long as the evidence supports and no
longer, AND name the padding shapes explicitly in rule 7a ("enabling X", "via Y", "to ensure
Z") so the failure mode is forbidden by description rather than only by principle. Rule 7a
ends with a procedure rather than a principle: read the bullet one clause at a time, name the
chunk supporting each, and DELETE any clause without one — delete rather than soften, because
a vaguer unsupported clause is still unsupported.

### Result: 50.0% → 93.8%, and the diagnosis was half wrong

Same 8 questions, same model, same flags, prompt as the only variable — the cleanest A/B in
the project:

| | claims | kept / struck | grounding | mean words |
|---|---|---|---|---|
| gpt-oss-20b, old prompt | 22 | 11 / 11 | **50.0%** | 13.6 |
| gpt-oss-20b, new prompt | 16 | **15 / 1** | **93.8%** | **12.1** |

The claim count falling is a consequence, not a confound: fewer strikes means fewer top-up
rounds, so the loop reached `n` instead of regenerating.

**But length is not what fixed it.** 13.6 → 12.1 words is a small move, nowhere near
llama-3.3-70b's 8.1. If length were the mechanism, 12.1-word bullets should still be failing
at close to the old rate. They aren't — strikes fell by a factor of eleven while length
barely moved.

So **rule 7a did the work, not rule 6's floor.** The bullets are still long; they simply stop
carrying an unsupported trailing clause. Same length, different content.

Which corrects the reading above: word count was a **symptom mistaken for a cause**. The old
model's 8.1-word bullets were not safe *because* they were short — they were safe because a
short bullet has no room for a purpose clause. Length correlated with the real variable and
was easy to measure, which is exactly how a proxy gets mistaken for a mechanism.

Worth keeping as a method note: the word-count table was what made the padding hypothesis
visible in the first place, and it was still the wrong causal story. The A/B is what settled
it. A correlation strong enough to generate the right fix can still be the wrong explanation.

The one remaining strike has the same shape, smaller: *"Implemented SQLAlchemy ORM **for data
modeling**"* — the trailing purpose clause again, once in sixteen rather than eleven in
twenty-two.

**Final M5 number: 96.7%** (29 kept / 1 struck, 30 claims) over **15/15 questions**,
`gpt-oss-20b`, `rerank --gen-k 10`, commit `59cf8e7`. Run in three windows on one commit with
identical flags, so the counts add:

| window | kept / struck | questions |
|---|---|---|
| q1–8 | 15 / 1 | 8 |
| q9 | 2 / 0 | 1 |
| q10–15 | 12 / 0 | 6 |
| **total** | **29 / 1** | **15** |

No failures, no gaps, no early stop — the first complete job-mode run the project has produced,
and only possible after the throttle and network retries in Finding 6. The single strike is the
trailing-purpose-clause shape ("Implemented SQLAlchemy ORM *for data modeling*"), one in thirty.

Higher than the 84.0% the retired model reached on the first 8 questions, but that comparison
crosses a model change *and* three prompt changes. Only the 50.0% → 93.8% pair on those same 8
questions is clean.

## Finding 6: two rate limits, one exception class

Every eval run since the model swap died partway, for three different reasons that all looked
like "the run stopped".

**TPM is not TPD.** Groq raises the same `RateLimitError` for a per-minute throttle and a
per-day exhaustion, and the harness treated both as "stop the run". gpt-oss-20b allows
**8000 TPM** (down from llama-3.3-70b's 12000) while one generate call is now ~4800, so
back-to-back calls hit it constantly. A message reading *"try again in 7.4625s"* was throwing
away five unevaluated questions. They are now split on the message text: TPM sleeps the stated
wait and retries, TPD propagates so `grounding_eval` can stop early with honest partial
tallies. Retrying both would be worse than stopping on both — it would spin for hours against
a daily cap and report nothing.

**Then the network became the bottleneck.** At ~4800 tokens per call against 8000/minute the
pipeline sustains roughly 1.5 calls a minute, so an 8-question window is a ~30 minute run
making ~50 calls — long enough that ordinary flakiness matters. One dropped TCP connect
(`APITimeoutError`) ended a run two questions in. Now retried with a 2/5/15s backoff, on a
budget separate from the throttle budget: they are unrelated failures, and a call that has
waited out four throttles should still get its full network allowance.

All three call sites route through one `_complete()` helper, so the policy is defined once.

The general shape, worth remembering: **"the run stopped" is not a diagnosis.** Three
different causes produced the same symptom, and two of them were recoverable in seconds. The
fix in each case was to stop treating a category of error as a single thing.

## Harness caveat, now fixed

`grounding_eval` hardcoded hybrid+rerank at k=8 while `retrieval_eval` honoured
`--retrieval`, so a single report could describe **two different retrievers** — the recall
rows from the flagged mode, the grounding row always from rerank. Both halves now route
through one `_retrieve()` helper, so the flag governs the whole run.

## The eval does not fit in a day, and that is an eval problem only

Measured cost, job corpus:

| | tokens | per 100k free-tier day |
|---|---|---|
| one eval question (k=10, n=2) | ~6,400 | — |
| **full 15-question run** | **~96,000** | 1 run, using 96% of the cap |
| `make-bullets` (n=5, k=8) | ~8,700 | ~12 artifacts |
| `make-star` (k=6 × 4 pools) | ~6,800 | ~15 artifacts |

Both artifact figures are upper bounds: `MAX_ROUNDS = 2` and round two only runs when the
Critic struck something, so a clean generation costs roughly half.

**Real usage never hits the cap.** Preparing for interviews you might generate five bullet
sets and three STAR answers, which is a third of a day. The eval is the pathological
workload, being ~15 real invocations fired back to back plus 6 traps. Groq's TPD is also a
rolling 24-hour window rather than a midnight reset, so a run started hours after the last
one still begins part-spent.

That distinction matters because it rules out the tempting fixes. Shrinking the 1,116-token
generator system prompt (~33k per run, a third of the budget), halving `n`, or swapping in a
cheaper model would each **degrade the product to fit the measurement** — and rule 7's worked
examples are most of that prompt, i.e. exactly the thing that moved grounding 53.8% → 80%.
The system is fine; only the measurement is expensive, so the measurement is what bends.

Hence `--offset`: the set is evaluated in windows across days and the halves are added.
Grounding is `(kept_a + kept_b) / (claims_a + claims_b)`, which is why the report now prints
`kept / struck` raw. **Averaging the two percentages is wrong** whenever the halves produced
different claim counts, which they always do. The halves are only one measurement if nothing
changed between them, so record the commit sha with each.

### Upgrade path: checkpoint results to disk (option G)

`--offset` is manual windowing. The better version is to append each question's result to a
JSONL as it completes and have the report aggregate whatever is on disk:

- a run **resumes** rather than restarting, so the rolling token window stops dictating what
  gets measured
- a crash or a 429 costs **one question**, not the run (the same instinct as the existing
  `RateLimitError` handling, one level up)
- questions already evaluated under the current config are skipped automatically, so there is
  no manual offset arithmetic to get wrong

The cost is cache invalidation: the checkpoint key has to cover corpus, retrieval mode,
`gen_k`, `n`, and the generator prompt, or a stale entry silently contributes a claim measured
under a different system. That is the whole difficulty, and it is why this is a post-M5 task
rather than a quick win — `--offset` needs no invalidation because the human is holding the
config fixed.

## Open items

- ~~Project-scoped retrieval for job mode~~ — done; recall@10 100%, no misses in any mode.
- ~~Thread `--retrieval` through `grounding_eval`~~ — done; one `_retrieve()` for both halves.
- Measure grounding % under `--retrieval hybrid` vs `rerank`. recall@8 says hybrid should
  win; that prediction is untested against the Critic. Two windowed runs per config, so a
  full A/B is four days on the free tier.
- Questions 9-15 have never been evaluated once: every run so far took the front slice, which
  is what `--offset` exists to fix.
- Checkpointed results (option G above), once M5 has shipped.
- `j5` intermittently fails generation with `Extra data: line 3 column 1` - the model emits
  valid JSON then keeps writing. Distinct from the empty-list failure; costs a question per
  run when it hits.
- j6 *"Where is MealWise deployed?"* failed generation in the first run (empty bullet list
  twice → `GenerationError`) while the gap check passed. Two distinct no-answer paths exist —
  pre-generation gap and post-generation empty — and only one is reported as a gap.


## Finding 7: the defect was the model's, not the bullets prompt's

Study mode is the **clean model-only comparison** the job-mode numbers could never be:
`SYSTEM_PROMPT` (quiz) has not changed all project, while rules 6 and 7a live in
`BULLETS_SYSTEM_PROMPT`, which quiz generation never touches. Same corpus, same 20 questions,
same vector retrieval at k=8, same Critic. Only the model differs.

| | llama-3.3-70b | gpt-oss-20b |
|---|---|---|
| grounding | **92.7%** | **70.0%** |
| claims | 41 | 50 |
| struck | 3 | 15 |
| questions completed | 19/20 | **20/20** |
| recall@3 / @8 / @10 | 90% / 100% / 100% | identical, to the decimal |

**All 15 strikes concede part of the claim and reject an unsupported extension.** Not one is a
fabrication, and not one is a retrieval miss — recall@8 is 100%, so every answer had its
evidence in the prompt. Exactly the shape rule 7a was written for in job mode.

So the padding behaviour is **a property of gpt-oss-20b, not of the bullets prompt**. Job mode
found it first only because that is where the measuring was happening. Two supporting signals:
it produced *more* claims (50 vs 41) and completed all 20 questions where the old model failed
one. It is a more willing generator, and the willingness is what over-reaches.

### Quiz mode fails one level earlier than bullets

The bullets fix targeted a padded *clause*. Here, **10 of 15 struck questions begin with "Why"
or "How"**, and the rest ask for roles or consequences:

```
"Why is Shielded Twisted Pair generally more expensive and harder to install?"
"How does a bad medium cause each of the three main transmission impairments?"
"How do redundant bits help the receiver determine if an error has occurred?"
```

The source states *that* STP costs more; it never says *why*. So the question itself is
unanswerable from the evidence, and any answer invents a mechanism. Bullets has no question, so
this failure mode could not appear there.

Fix: two new rules in `SYSTEM_PROMPT`. Rule 4 governs **question selection** — find the sentence
that IS the answer before writing the item, and prefer a "what" the evidence answers outright to
a "why" it only gestures at. Rule 5 is the bullets rule ported: do not extend a supported answer,
delete an unsupported clause rather than softening it.

Cost: the quiz prompt grows 159 → 568 tokens, about +16k across a 20-question run.

`GUIDE_SYSTEM_PROMPT` deliberately left alone — the guide generator has no eval set, so a change
there would be unmeasured.

**Baseline to beat: 70.0% (35 kept / 15 struck, 50 claims, 20/20 questions),** `gpt-oss-20b`,
bare defaults, commit `af23583`.
