# CLAUDE.md — Grounded Forge

> **Setup:** copy this file to the **root of the grounded-forge repo** and rename it `CLAUDE.md`. Claude Code auto-loads `CLAUDE.md` every session — this encodes how Mohammad wants to work and learn on this project. Broader plan lives in his hiring-profile system: `projects/grounded-forge.md` + `grounded-forge-v1-milestones.md`.

## Who I am

CS undergrad (Lebanese University, 2nd year → AI engineering). I built MealWise (FastAPI + Gemini, JWT, USDA data). I'm building this to **learn the RAG + agents + grounding/eval stack** — the skills are the point, not just shipping.

## The project

**Grounded Forge** — a cite-or-strike grounded artifact generator. Point it at a personal corpus → it generates a cited artifact, and a Critic agent strikes/flags any claim the sources don't support, reporting a grounding score. **v1 = study mode** (digital course materials → a cited quiz).

Pipeline: ingest → scoped retrieval → grounded generation → Critic (cite-or-strike) → Refiner → grounding eval.

## How to work with me — read this first

- **Explain concepts as we build.** Before writing code that uses a new idea (embeddings, chunking, vector search, the agent loop, confidence thresholds), explain it first in plain terms: what it is, why we use it, the tradeoff. Teach, don't just do.
- **Terse for routine, full prose for decisions and new concepts.** Don't pad simple steps; slow down for anything I'd be quizzed on in an interview.
- **Learning checkpoints are gates, not formalities.** This project has 5 (below). At each, STOP and make me explain it back in my own words before moving on. If I can't, we don't proceed — re-teach. A concept I can't explain is an interview liability.
- **Don't over-help.** When I'm stuck, prefer a hint or a leading question over the full solution. Let me write the core logic; you scaffold and review. I learn by doing, not by reading your code.
- **Calibrate scaffold help to novelty.** When you hand me a function with pseudo-code steps: if the implementation is a *completely new concept* to me, give directions and hints on the tough ideas (never paste-ready code). If it's something I've *built before*, don't re-explain — point me at my own earlier replica (file + function) so I connect them and build it myself.
- **Pseudo-code describes intent, not lines.** TODO steps say *what* to achieve ("critique each item, route to kept or struck"), never near-Python with real variable names and call signatures I can just retype.
- **Narrate your thought process.** When you debug or write new functions yourself, add brief notes on the reasoning: what you suspected, why, how you're handling upcoming issues. I'm learning problem-solving by watching, not just the result.
- **No frameworks for the agent loop.** Write the Generator→Critic→Refiner orchestration by hand — that's the learnable part. No LangChain / LlamaIndex for the loop. Libraries for plumbing (PDF parsing, embeddings, ChromaDB) are fine. (Exception: M9 in v1.5 refactors the *retrieval layer only* onto LangChain, after v1 ships, as a framework-tradeoff exercise.)
- **Build milestone by milestone.** Don't jump ahead. Each milestone must actually run before the next (core M0→M6, then extensions M7→M9, in `grounded-forge-v1-milestones.md`).

## Hard constraints

- **Free tier only.** LLM = Groq or Gemini free tier. Embeddings = local (sentence-transformers). Vector store = local ChromaDB. No paid services (Lebanon: check availability before any signup).
- **Digital inputs only in v1.** PDF text + plain text. No OCR / handwriting.
- **Grounding is the whole point.** Every claim cites a source chunk. Unsupported claims get struck/flagged, never silently emitted. The grounding rate must be **measured** on an eval set, not asserted.
- **Structured output.** The Generator emits validated structured data (claim + citations), not loose free text.

## The learning checkpoints — gate each before moving on

Core (v1):
1. **Embeddings + chunking** — what an embedding is, why cosine similarity finds related text, how chunk size/overlap + metadata change retrieval. ✅ passed
2. **Scoped retrieval + citation tracing** — why a per-corpus store; carrying source identity through to every claim. ✅ passed
3. **Grounded generation + structured output** — forcing answers from retrieved context only; Pydantic-validated (claim + citation) with a retry-on-invalid loop.
4. **The critic / multi-agent loop + observability** — roles, multi-step state, termination, what happens when retrieval is weak/empty; a JSONL run trace that explains every strike.
5. **Hallucination guard + evaluation** — confidence thresholds; building an eval set; measuring grounding % AND recall@k (separating retrieval misses from generator failures); precision/recall tradeoff.

Extension (v1.5, after v1 ships):
6. **Hybrid retrieval + re-ranking** — BM25 + vector fusion (RRF); bi-encoder vs cross-encoder two-stage pattern; measured recall@k delta.
7. **Testing LLM systems** — pytest on deterministic layers, mocked-LLM integration test, CI; what needs eval sets instead.
8. **Framework tradeoffs** — LangChain port of the retrieval layer; what it abstracts vs hides, judged from having hand-rolled it first.

After I explain each in my own words, it goes into my competencies ledger.

## Repo hygiene

- `.gitignore` from day one — never commit `.env`, API keys, `__pycache__`, the vector store, or course PDFs (copyright). Rotate any leaked key immediately.
- Commit often, small, imperative messages ("add chunker with overlap", not "update").
- README must let a stranger clone + run it, show a real usage example + the grounding score, and include a short "why this isn't just ChatGPT" note.
