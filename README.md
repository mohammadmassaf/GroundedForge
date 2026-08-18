# Grounded Forge

![tests](https://github.com/mohammadmassaf/GroundedForge/actions/workflows/tests.yml/badge.svg)

**A cite-or-strike artifact generator: every claim cites your own documents, an independent Critic agent strikes anything the sources don't support, and the no-hallucination guarantee is measured, not asserted.**

Two modes on one engine. **Study mode** turns course PDFs into a cited quiz. **Job mode** turns your git history and project notes into cited CV bullets and STAR interview answers. Swapping the domain changed ingestion and the output schema; retrieval, the Critic, the orchestration loop and the eval harness are the same code.

> **92.7% grounding** (38/41 claims independently verified against source) · **100% recall@10** on a fixed 20-question eval set
>
> *Grounding measured on `llama-3.3-70b-versatile`, which Groq retired on 2026-08-17; the pipeline now runs `openai/gpt-oss-20b` and that figure is being re-measured. recall@k is unaffected — retrieval is entirely local.*

Point it at a folder of course PDFs, ask for a quiz on any topic, and get a markdown artifact where every question and answer links to the exact page it came from, with unsupported claims struck out, visibly, never silently emitted.

## Why this isn't just ChatGPT

A plain chat model silently blends your notes with its training knowledge. It answers confidently from general knowledge even when your course frames a topic differently, and you can't tell which is which. Grounded Forge builds the grounding machinery explicitly:

- **Context-only generation:** the Generator sees ONLY retrieved chunks from your documents and must cite the chunk_ids it used. Asked about a topic your corpus doesn't cover, it produces *nothing* rather than inventing (verified: an off-corpus topic yields zero items, loudly).
- **Independent verification:** a separate Critic agent, with no memory of generation, checks each claim against only its cited evidence: *is this actually in the source?* True statements that aren't in your notes get struck. Truth isn't the standard, presence in the evidence is.
- **A measured guarantee:** a fixed eval set produces a reproducible grounding score. The number above isn't a promise; re-run `eval` and check.

## How it works

```
ingest            retrieve              generate                critic                 eval
PDFs → chunks  →  embed (local)      →  Generator drafts    →  Critic verifies     →  grounding %
  + metadata      ChromaDB, cosine      quiz w/ citations      each claim vs its      recall@k
                  top-k per query       (schema-validated,     cited chunks only;
                                        retry on invalid)      Refiner drops/regens
```

- **Multi-agent loop is hand-rolled:** no LangChain/LlamaIndex. Generator (temp 0.3) → Critic (temp 0.0, strict) → Refiner (deterministic policy: keep verified, strike failed, regenerate shortfall, hard round cap).
- **Every run writes a JSONL trace:** every generation and every verdict with its reason. "Why was this claim struck?" is answerable from the trace alone.
- **Free tier end to end:** Groq (`openai/gpt-oss-20b`), local sentence-transformers embeddings, local ChromaDB. No paid services.
- **The local half survives a vendor pulling a model.** When Groq retired the LLM this was built on, every recall@k figure reproduced exactly — embeddings, BM25, the cross-encoder and the vector store all run here. Only the grounding score, which measures a specific model, had to be re-measured.

## Setup

```bash
git clone https://github.com/mohammadmassaf/GroundedForge.git
cd GroundedForge
python -m venv venv
venv\Scripts\pip install -r requirements.txt     # Windows (use venv/bin/pip on Unix)
copy .env.example .env                            # add your free Groq API key (console.groq.com)
```

**Try it without supplying anything.** The repo ships a small public-domain
corpus (`demo_corpus/` — the IP, TCP and UDP RFCs) so a fresh clone is runnable:

```bash
python main.py ingest --corpus demo              # 526 chunks with source metadata
python main.py build-index --corpus demo         # embed chunks -> ChromaDB
python main.py make-quiz "TCP connection establishment" --corpus demo -n 5
```

**On your own material.** Drop PDFs (or .txt/.md) into `data/`, add a corpus
entry to `corpus.yaml` (copy `corpus.example.yaml`), then:

```bash
python main.py ingest --corpus mycourse          # PDFs -> chunks with source metadata
python main.py build-index --corpus mycourse     # embed chunks -> ChromaDB
python main.py make-quiz "your topic" --corpus mycourse -n 5
```

## Real output

From an actual run against the shipped demo corpus
(`make-quiz "TCP connection establishment and the three-way handshake" --corpus demo`):

> ### Q1. What is the procedure used to establish a connection in TCP?
>
> **Answer:** The three-way handshake
>
> 📖 `rfc793_p36_c3` — rfc793.txt, p.36: "…The \"three-way handshake\" is the procedure used to establish a connection. This procedure normally is initiated by one TCP and responded to…"

Full sample artifact: [quiz_demo.md](quiz_demo.md) — reproducible from a clean
clone with the three commands above. When the Critic strikes a claim, it appears
struck-through in a separate section with the reason. See the sample trace in
[samples/](samples/) for the verdicts behind a full eval run.

The measured numbers below come from a 20-question eval set over three
networking-course PDFs, which stay out of the repo for copyright reasons — the
demo corpus is what's reproducible, the course corpus is what's measured.

## Evaluation

`python main.py eval --corpus mycourse` runs two independent measurements over a fixed eval set ([eval/eval_set.json](eval/eval_set.json)):

| Metric | Result | What it grades |
|---|---|---|
| recall@3 | 90% | retrieval ranking |
| recall@5 | 95% | retrieval ranking |
| recall@10 | 100% | is the right chunk findable at all |
| **grounding** | **92.7%** | % of generated claims the Critic verified against their cited sources |

Separating the two localizes every failure: a bad quiz item is either a retrieval miss or a generation failure, and those are different bugs with different fixes. Full analysis, struck-claim case studies, and the Critic strictness tradeoff: [eval/notes.md](eval/notes.md).

### Hybrid retrieval (M7)

Retrieval also runs in hybrid mode: BM25 keyword search fused with vector search via Reciprocal Rank Fusion, then re-ranked by a cross-encoder (`--retrieval vector|hybrid|rerank` on `eval`). Measured head-to-head on the same eval set:

| mode | recall@3 | recall@5 | recall@10 |
|---|---|---|---|
| vector (baseline) | **90%** | 95% | 100% |
| hybrid (BM25 + RRF) | 80% | 90% | 100% |
| hybrid + rerank | **90%** | 95% | 100% |

The honest finding: on a paraphrase-style eval set the pipeline **matched** the baseline rather than beating it. Fusion alone diluted the top-3 with keyword noise, and the re-ranker won it back. What hybrid buys is robustness the headline number doesn't reward: the one question vector missed (a misspelled "sheilded twisted pair") is the one hybrid fixed, via exact-token matching. Full per-question diff in [eval/notes.md](eval/notes.md).


## v2 — job mode: the same engine, a different corpus

The domain-agnostic claim above is only worth making if it survives a real swap. v2 points the
identical pipeline at **my own git history, repo docs and project notes** and produces **cited
CV bullets** and **cited STAR interview answers**. Ingestion and the output schemas changed;
retrieval, the Critic, the orchestration loop and the eval harness did not.

```bash
python main.py make-bullets "MealWise authentication" --repo mealwise -n 5
python main.py make-star "Tell me about improving retrieval quality" --repo grounded-forge
```

Sample artifact: [bullets_job.md](bullets_job.md) — real bullets with commit-level citations,
and a struck section showing what the Critic rejected and why.

### What a repo corpus breaks that a textbook doesn't

| | study mode (course PDFs) | job mode (repos) |
|---|---|---|
| recall@3 | 90% | 73.3% |
| recall@8 *(what the Generator reads)* | **100%** | 86.7% |
| recall@10 | 100% | 100% |
| grounding | 92.7%\* | **94.4%** (17 kept / 1 struck, 9/15 q) |
| inflation-catch | not measured | **100%** (6/6 traps) |

\* *study grounding was measured on `llama-3.3-70b-versatile`, retired by Groq mid-project;
job grounding is current on `openai/gpt-oss-20b`. See below.*

**recall@8 is the row that matters** — it is the k the Generator actually reads, and it is the
ceiling on grounding, because a claim cannot cite evidence that never reached the prompt. The
study corpus hands the Generator everything it needs; the repo corpus does not. Job mode starts
behind before a word is written.

Three failures that only a repo corpus produces:

- **Prose states things; commits label them.** `feat: add auth with JWT` is seven words
  covering a week of work, so a CV bullet has to generalise — and a grounding checker punishes
  generalisation. The artifact type and the verifier want opposite things. That tension *is*
  job mode.
- **Evidence pools are an authorization boundary, not a bag of context.** Asked "how is
  authentication implemented in MealWise?", the cross-encoder ranked a *Grounded Forge* commit
  first: it contained "implemented" and a quoted `mealwise@…` sha, and no mention of auth.
  Scoping retrieval by repo took recall@10 from 93.3% to 100%.
- **A number cited from the wrong chunk is still wrong.** Every verdict logs its citations and
  the pool it was drawn from, so `pool − citations` separates "the evidence was never
  retrieved" from "it was there and the claim pointed elsewhere". Those read identically in a
  strike reason and need opposite fixes.

### Adversarial evaluation: measuring the Critic itself

recall@k and grounding % both grade the system on *good* inputs, so neither can detect a broken
Critic — one that answers "supported" to everything scores **100% grounding**, the best possible
number from the worst possible judge.

So the job eval set carries **6 authored traps**, each a claim with a known defect and a tag for
which check should catch it: a rounded `~93%` where the source says `92.7%`, a fabricated `40%`
latency gain, a skill absent from the corpus, a real number cited from the wrong chunk. All 6
were struck, each by its intended stage — 3 by the deterministic quantity check, 3 by the LLM
Critic, no mismatches.

That is the measurement that makes the other two trustworthy, and it costs a tenth of a
grounding run.

### Two-stage verification

Job mode adds a deterministic pre-check before any LLM judgment: **every figure in a claim must
appear literally in its cited evidence, or it is struck without an API call.** Inflation is the
cheapest hallucination to catch and the most damaging on a CV, so it never reaches a model that
might be talked round. The Critic only sees claims that already survived arithmetic.

### When the vendor deleted the model

Groq retired `llama-3.3-70b-versatile` mid-project. Every LLM-dependent number became
historical overnight — and **every recall figure reproduced exactly**, because embeddings, BM25,
the re-ranker and the vector store all run locally.

Re-measuring on `gpt-oss-20b` then exposed a defect the old model had been hiding: two prompt
rules contradicted each other. One demanded 12–25 words per bullet, another forbade saying
anything the evidence doesn't state — and a seven-word commit often cannot pay for 12 grounded
words, so obeying the first forced padding. The retired model had quietly averaged 8.1 words and
ignored the rule. The new one obeyed it, and grounding fell to 50%.

Naming the padding shape explicitly (`", enabling X"`, `", via Y"`) took the same 8 questions
from **50.0% to 93.8%**, prompt as the only variable. Instructive detail: mean bullet length
barely moved, 13.6 → 12.1 words. Length was a symptom I mistook for the cause, and only the A/B
told the difference.

**A grounding score measures a specific model, on a specific date.** Both are now published
beside every number here.

Full analysis — six findings, each with the measurement that produced it: [eval/notes.md](eval/notes.md).

## Tech stack

Python · Groq (`openai/gpt-oss-20b`) · sentence-transformers (all-MiniLM-L6-v2 embeddings + ms-marco-MiniLM-L-6-v2 cross-encoder, both local) · ChromaDB (local, cosine) · rank_bm25 · Pydantic v2 (schema-validated LLM output with retry-on-invalid) · pypdf

## Project structure

```
ingest/     loader (pypdf), chunker (~700 chars, 100 overlap), pipeline -> chunks/<corpus>.json
retrieve/   embed + store (ChromaDB, one collection per corpus), top-k query with scores,
            BM25 keyword search, RRF fusion, cross-encoder re-ranking, hybrid pipeline
generate/   Generator agent, Pydantic schema, validation + retry loop, markdown renderer
critic/     Critic agent (claim vs cited evidence), orchestration loop, JSONL run tracer
eval/       fixed eval set, recall@k + grounding harness, findings (notes.md)
```

Corpora are fully isolated: one ChromaDB collection per `--corpus`, so subjects never cross-contaminate. Swapping domains = new files in `data/` + a new corpus name, no code changes. v2 does exactly that: the same engine over a git/docs/notes corpus, adding only adapters and output schemas — see [v2 — job mode](#v2--job-mode-the-same-engine-a-different-corpus).

## How this was built

This was built as a deliberate learning project, not an autogenerated one. I wrote the core logic myself: the grounded Generator and its validate-and-retry loop, the independent Critic, the multi-agent orchestration loop, the BM25 keyword search, Reciprocal Rank Fusion, and cross-encoder re-ranking, plus the recall@k and grounding evaluation harness and the unit tests.

I used Claude as a Socratic tutor and code reviewer. It explained unfamiliar concepts, scaffolded function signatures with intent-level TODOs, and reviewed my implementations for bugs. It did not write the core functions for me. Every concept behind the code was checkpointed and quizzed before I moved on, so I can explain and defend any line in this repository. The commit history reflects that workflow: small, reasoned steps rather than bulk generation.
