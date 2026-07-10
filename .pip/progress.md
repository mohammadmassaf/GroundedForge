# PIP Progress — Grounded Forge

_Source: C:\Users\MsiPc\Desktop\projects\GroundedForge. Last updated: 2026-07-04._

## Concept tree

### M0–M2: Setup + Ingest + Retrieve
- [x] **CLI subcommand pattern with argparse** — `built` · prereqs: none · code: `main.py:31`
- [x] **PDF text extraction** — `built` · prereqs: none · code: `ingest/loader.py:23`
- [x] **Overlapping text chunking** — `built` · prereqs: none · code: `ingest/chunker.py:53`
- [x] **Document ingestion pipeline** — `built` · prereqs: pdf-text-extraction, text-chunking · code: `ingest/pipeline.py:17`
- [x] **Vector embeddings and semantic similarity** — `built` · prereqs: none · code: `retrieve/store.py:26`
- [x] **Vector store with cosine distance** — `built` · prereqs: vector-embeddings · code: `retrieve/store.py:32`
- [x] **Nearest-neighbour retrieval** — `built` · prereqs: vector-embeddings, vector-store · code: `retrieve/query.py:36`
- [x] **Singleton pattern for shared resources** — `known` (also in Mealwise) · code: `retrieve/query.py:18`

### M3: Grounded Generation
- [x] **Grounded generation (context-only prompting)** — `checkpointed` (8/10) · prereqs: rag-pipeline, prompt-engineering · code: `generate/generator.py:31`
- [x] **Structured LLM output with schema validation** — `checkpointed` (7/10) · prereqs: prompt-engineering · code: `generate/generator.py:85` · gap: static-schema vs runtime-validation boundary
- [x] **Retrieval-augmented generation (RAG)** — `built` · prereqs: prompt-engineering, vector-embeddings · code: `retrieve/query.py:36`

### M4: The Critic
- [x] **Multi-agent loop (Generator → Critic → Refiner)** — `checkpointed` (8/10) · prereqs: grounded-generation, claim-verification · code: `critic/loop.py:31` · gap: termination as budget
- [x] **Claim verification against source chunks** — `checkpointed` (7/10) · prereqs: nearest-neighbor-retrieval · code: `critic/critic.py:96` · gap: independence vs scope
- [x] **Agent run tracing and observability** — `checkpointed` (8/10) · prereqs: multi-agent-loop · code: `critic/trace.py:21`

### M5: Grounding Eval
- [x] **Grounding evaluation and hallucination measurement** — `checkpointed` (7/10) · prereqs: claim-verification, multi-agent-loop · code: `eval/run_eval.py:84` · gap: calibrating the evaluator
- [x] **Retrieval quality metrics (recall@k)** — `checkpointed` (8/10) · prereqs: nearest-neighbor-retrieval · code: `eval/run_eval.py:55`

### M7–M9: v1.5 extensions
- [ ] **Hybrid retrieval (BM25 + vector fusion)** — `new` · prereqs: nearest-neighbor-retrieval, retrieval-metrics
- [ ] **Cross-encoder re-ranking** — `new` · prereqs: hybrid-retrieval
- [ ] **Testing LLM systems** — `new` · prereqs: structured-llm-output, multi-agent-loop
- [ ] **Framework tradeoffs (hand-rolled vs LangChain)** — `new` · prereqs: rag-pipeline

## Mastery list (deduped — known across all projects)
- (none yet ≥9 — grounded-generation at 8, passed)

## Dropped prerequisite edges (log)
- (none)
