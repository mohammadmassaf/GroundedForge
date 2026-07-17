# Grounded Forge

![tests](https://github.com/mohammadmassaf/GroundedForge/actions/workflows/tests.yml/badge.svg)

**A cite-or-strike quiz generator: every answer cites your own course material, an independent Critic agent strikes any claim the sources don't support, and the no-hallucination guarantee is measured, not asserted.**

> **92.7% grounding** (38/41 claims independently verified against source) · **100% recall@10** on a fixed 20-question eval set

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
- **Free tier end to end:** Groq (llama-3.3-70b), local sentence-transformers embeddings, local ChromaDB. No paid services.

## Setup

```bash
git clone https://github.com/mohammadmassaf/GroundedForge.git
cd GroundedForge
python -m venv venv
venv\Scripts\pip install -r requirements.txt     # Windows (use venv/bin/pip on Unix)
copy .env.example .env                            # add your free Groq API key (console.groq.com)
```

Drop your course PDFs (or .txt/.md) into `data/`, then:

```bash
python main.py ingest --corpus mycourse          # PDFs -> chunks with source metadata
python main.py build-index --corpus mycourse     # embed chunks -> ChromaDB
python main.py make-quiz "your topic" --corpus mycourse -n 5
```

## Real output

From an actual run against three networking-course PDFs (`make-quiz "transmission and propagation delay" --corpus networks`):

> ### Q1. What is the formula for propagation delay?
>
> **Answer:** Propagation Delay = Distance / Propagation speed
>
> 📖 `I2208-2024-2025-Part-3_p53_c0` — I2208-2024-2025-Part-3.pdf, p.53: "…𝑃𝑟𝑜𝑝𝑎𝑔𝑎𝑡𝑖𝑜𝑛 𝐷𝑒𝑙𝑎𝑦 = 𝐷𝑖𝑠𝑡𝑎𝑛𝑐𝑒 / 𝑃𝑟𝑜𝑝𝑎𝑔𝑎𝑡𝑖𝑜𝑛 𝑠𝑝𝑒𝑒𝑑…"

Full sample artifact: [quiz_networks.md](quiz_networks.md). When the Critic strikes a claim, it appears struck-through in a separate section with the reason. See the sample trace in [samples/](samples/) for the verdicts behind a full eval run.

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

## Tech stack

Python · Groq (llama-3.3-70b-versatile) · sentence-transformers (all-MiniLM-L6-v2 embeddings + ms-marco-MiniLM-L-6-v2 cross-encoder, both local) · ChromaDB (local, cosine) · rank_bm25 · Pydantic v2 (schema-validated LLM output with retry-on-invalid) · pypdf

## Project structure

```
ingest/     loader (pypdf), chunker (~700 chars, 100 overlap), pipeline -> chunks/<corpus>.json
retrieve/   embed + store (ChromaDB, one collection per corpus), top-k query with scores,
            BM25 keyword search, RRF fusion, cross-encoder re-ranking, hybrid pipeline
generate/   Generator agent, Pydantic schema, validation + retry loop, markdown renderer
critic/     Critic agent (claim vs cited evidence), orchestration loop, JSONL run tracer
eval/       fixed eval set, recall@k + grounding harness, findings (notes.md)
```

Corpora are fully isolated: one ChromaDB collection per `--corpus`, so subjects never cross-contaminate. Swapping domains = new files in `data/` + a new corpus name, no code changes (v2 will point the same engine at a repo/commit corpus for CV & interview prep artifacts).

## How this was built

This was built as a deliberate learning project, not an autogenerated one. I wrote the core logic myself: the grounded Generator and its validate-and-retry loop, the independent Critic, the multi-agent orchestration loop, the BM25 keyword search, Reciprocal Rank Fusion, and cross-encoder re-ranking, plus the recall@k and grounding evaluation harness and the unit tests.

I used Claude as a Socratic tutor and code reviewer. It explained unfamiliar concepts, scaffolded function signatures with intent-level TODOs, and reviewed my implementations for bugs. It did not write the core functions for me. Every concept behind the code was checkpointed and quizzed before I moved on, so I can explain and defend any line in this repository. The commit history reflects that workflow: small, reasoned steps rather than bulk generation.
