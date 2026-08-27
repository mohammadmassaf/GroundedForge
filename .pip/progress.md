# PIP Progress — Grounded Forge

_Source: the GroundedForge repo root. Last updated: 2026-08-24 (D-M1 + D-M2 shipped; new stack **devops-containers** opened with 3 of 16 concepts done)._

> **Recurring gap, two concepts running:** on a trade-off, the concrete/visible cost gets named and the abstract one doesn't (token budget but not measurement integrity; silent failure but not misplaced strictness). Worth targeting directly rather than waiting for it to surface a third time.

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
- [x] **Grounding evaluation and hallucination measurement** — `checkpointed` (7/10) · prereqs: claim-verification, multi-agent-loop · code: `eval/run_eval.py:84` · ~~gap: calibrating the evaluator~~ **closed 2026-08-23** by adversarial-eval-traps — a judge can't be calibrated by watching it agree with itself
- [x] **Retrieval quality metrics (recall@k)** — `checkpointed` (8/10) · prereqs: nearest-neighbor-retrieval · code: `eval/run_eval.py:55`

### M7–M9: v1.5 extensions
- [x] **Hybrid retrieval (BM25 + vector fusion)** — `checkpointed` (8/10) · prereqs: nearest-neighbor-retrieval, retrieval-metrics · code: `retrieve/fusion.py:26`
- [x] **Cross-encoder re-ranking** — `checkpointed` (7/10) · prereqs: hybrid-retrieval · code: `retrieve/rerank.py:35` · gap: ceiling = candidate-set recall, not k
- [x] **Testing LLM systems** — `checkpointed` (8/10) · prereqs: structured-llm-output, multi-agent-loop · code: `tests/test_loop_integration.py:52` · gap: non-determinism (not just cost) as why model judgment resists unit tests
- [x] **Framework tradeoffs (hand-rolled vs LangChain)** — `checkpointed` (8/10) · prereqs: rag-pipeline · code: `docs/langchain-notes.md` · gap: naming the forward conditions under which a hand-rolled layer should move onto a framework

### V2-M0–M1: Corpus adapters (job mode)
- [x] **Corpus adapters for semi-structured sources** — `checkpointed` (8/10) · prereqs: ingestion-pipeline, text-chunking · code: `ingest/adapters/git_adapter.py:62` + `ingest/adapters/base.py:76` · gap: naming the concrete failure of sha-in-text-only (citation must regex-parse free text; can't metadata-filter on it)

### V2-M2: Job-corpus retrieval
- [x] **Metadata-filtered retrieval** — `checkpointed` (9/10 · mastered) · prereqs: vector-store, nearest-neighbor-retrieval · code: `retrieve/store.py:19` (_chunk_metadata) + `retrieve/query.py:33` (where=/filter=)

### V2-M3: make-bullets + quant pre-check
- [x] **Deterministic pre-checks before LLM judgment** — `checkpointed` (7/10) · prereqs: claim-verification, structured-llm-output · code: `critic/quant.py:66` (check_quantities) + `critic/loop.py:78` (two-stage loop) · gap: silent-failure asymmetry — judged permissive vs strict guards equally bad by frequency; the deciding factor is observability (a weakened guard reports success while not checking)

### V2-M4: make-star synthesis
- [x] **Cross-source evidence synthesis** — `checkpointed` (8/10) · prereqs: metadata-filtered-retrieval, multi-agent-loop · code: `retrieve/star_evidence.py:85` (gather_evidence) + `critic/loop.py:176` (own_ids scope check) · gap: per-section retrieval is what *creates* the citation-scope boundary — named the starvation argument but not that a single pool makes `own_ids` vacuous

### V2-M5: Job eval + ship
- [x] **Adversarial eval traps** — `built` (8/10) · prereqs: grounding-eval · code: `eval/run_eval.py:228` (trap_eval) + `eval/eval_set.json` (s1–s6) · gaps: (1) costed only the token direction of growing the set, not measurement integrity — tuning the Critic prompt against traps turns them into training data; (2) 6/6 is recall over an *authored* population, so unauthored shapes are unmeasured by construction; (3) s6 (regression guard, derived from the defect rules 4–5 already fixed) vs s5 (independent probe) are two instruments folded into one number

### D-M0: Docker prep
- [x] **Configuration indirection (late-bound locations)** — `checkpointed` (8/10) · prereqs: corpus-adapters · code: `ingest/corpus_config.py:21` (_expand) + `:71` (roots table) + `:82` (resolution point) · gaps: (1) costed only the silent-failure direction, not the cost of misplaced strictness; (2) placed the silent failure on the write side — the write succeeds, which is what hides it

### D-M1: Dockerfile + offline image
- [x] **Container image layering and build cache** — `checkpointed` (8/10) · stack: devops-containers · prereqs: none · code: `Dockerfile:30` (torch on its own line) → `:41` (bake) → `:44` (`COPY . .`) · measured both directions: source edit rebuilt in 10 s, an edit above the bake cost 210 s + 186 MB
- [x] **Build/runtime pin symmetry for reproducibility** — `checkpointed` (8/10, raised from 6 by a targeted pass) · stack: devops-containers · prereqs: container-image-layering · code: `retrieve/model_pins.py:8` + `docker/bake_models.py:13` + `retrieve/store.py:50` · ~~gap: incidental vs guaranteed~~ **closed 2026-08-24** — carries the test *"which line would have to change for the wrong version to arrive?"*, and transferred it to a mutable-tag/digest case in CI · remaining gap: naming what makes a defect invisible (absence of a **trigger**, not of a bug — correct behaviour under the one condition where it cannot fail is not evidence). `recheck: true`

### D-M2: Volumes and state
- [x] **Container state persistence: volumes, bind mounts, mount-point ownership** — `checkpointed` (8/10) · stack: devops-containers · prereqs: container-image-layering · code: `Dockerfile:49` (`mkdir -p` before `chown -R`) · a volume mounted where the image has nothing is created by the daemon as root; `chown -R` cannot reach a path that does not exist yet

## Mastery list (deduped — known across all projects)
- **metadata-filtered-retrieval** — 9/10 (grounded-forge, V2-M2, 2026-07-24) — first mastered concept

## Dropped prerequisite edges (log)
- (none)
