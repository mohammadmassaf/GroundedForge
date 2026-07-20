# Grounded Forge v2 — Job Mode — Design Spec

Date: 2026-07-18
Status: approved (brainstormed + section-by-section approval)
Companion plan in the hiring-profile vault: `projects/groundedForge/grounded-forge-v2-milestones.md`

## Goal

Point the same cite-or-strike engine at a new corpus — my git repos, their docs, and my hiring-profile vault notes — and generate **grounded CV bullets** and **grounded STAR answers**, where every claim (especially every number) traces to real evidence or gets struck. This is the domain-agnostic proof promised in v1's scope: swapping domains touches only ingestion and output schemas, nothing else.

## Decisions made during brainstorming

- **Corpus:** git history (commits) + repo docs (README/CLAUDE.md/docs) + hiring-profile vault notes. Commit messages alone are too thin for STAR: Situation lives in the vault, Action in commits, Result in eval numbers and ship notes.
- **Output driver:** question/topic-driven like v1 (`make-bullets "MealWise"`, `make-star "<interview question>"`). Job-description-driven generation is **v2.5** (December, when real applications exist).
- **Window:** 2026-07-19 → ~2026-08-01, ~2 h/day. Estimates are **assisted-pace** (~11 h total), recalibrated per the M8 retro rule — solo-pace numbers (~20 h) ran 2–3× high across v1. Surplus time = depth, never skipped PIP checkpoints. In December: re-run ingestion on newer commits + build the JD extension.
- **Approach:** source adapters inside the engine (chosen over external preprocessing scripts, which skip the learning and shred metadata; and over extracting the engine into a package, which spends the window on packaging instead of RAG skills).

## Architecture

```
ingest/adapters/*  ──►  chunks/<corpus>.json  ──►  retrieve/ (untouched)
                                                      │
eval/ (new job eval set)  ◄──  critic/ (+quant check)  ◄──  generate/ (+2 schemas)
```

Everything lives in the existing repo. The five engine stages keep their current structure; v2 adds an adapter layer at the front and two output schemas near the back.

## Ingestion: the adapter layer

**Adapter contract.** Every adapter emits the same chunk dict the chunker already produces (`chunk_id`, `source_file`, `page`, `text`, char range) plus new optional metadata: `source_type` (`git` | `docs` | `vault` | `files`), `repo`, `sha`, `date`. Existing file ingestion (loader.py + chunker.py) is wrapped as `files_adapter` — study mode keeps working unchanged.

**Adapters and their chunk units:**

| Adapter | Source | Chunk unit | Citation identity |
|---|---|---|---|
| `git_adapter` | `git log` on a local repo path: message + `--numstat` diff stats + files touched | 1 commit = 1 chunk (no char windowing); merge/trivial commits (e.g. lockfile-only) filtered | `repo@short-sha` |
| `docs_adapter` | README, CLAUDE.md, `docs/*.md` per repo | markdown heading section | `repo README § Section` |
| `vault_adapter` | explicitly listed hiring-profile vault notes; frontmatter-aware | heading section | `note § Section` |
| `files_adapter` | v1 behavior (`data/` PDFs/txt/md) | char windows (unchanged) | file + page |

`vault_adapter` only reads paths explicitly listed in config — it never crawls the vault.

**Corpus config.** `corpus.yaml` at repo root defines named corpora → list of sources (type + path). `ingest --corpus job` reads it. `data/` folder behavior remains for study mode.

**Privacy guardrail.** Vault text lands in `chunks/*.json` and `chroma_db/` — both must be verified gitignored (M0 checkbox). Committed sample artifacts quote vault notes → manual read-before-commit rule.

## Generation: two new outputs

**CLI:** `make-bullets "<project/skill>"` → N resume bullets; `make-star "<interview question>"` → one STAR answer.

**Schemas (Pydantic, same validate-retry loop as v1 — unknown chunk_id → re-prompt with error, max 2 retries, fail loudly):**

```python
CVBullet   { text: str, citations: list[chunk_id] }          # artifact = list of bullets

STARAnswer { question: str,
             situation: Section, task: Section,
             action: Section,    result: Section }
# Section  { text: str, citations: list[chunk_id] }          # per-section citations
```

**Cross-source synthesis (`make-star`).** One retrieval per STAR section, biased by Chroma metadata filter on `source_type`:

- Situation / Task → prefer `vault` + `docs` (the why, constraints, goals)
- Action → prefer `git` (what I actually did)
- Result → prefer `vault` + `docs` (eval numbers, ship notes, outcomes)

"Prefer" = filtered query first, unfiltered fallback if results are thin. The Generator receives the four labeled evidence pools; each section may only cite its own pool. `make-bullets` stays single-retrieval like v1.

**Generator prompt rules (job mode):** first person, resume/interview voice, strong verbs, past tense; a quantity may appear only if literally present in a retrieved chunk; no skill-name-dropping without evidence of use; thin evidence → say so rather than pad (Critic enforces).

**Rendering:** same markdown artifact style as v1 — claim text + inline citation (quote + human-readable source id, e.g. `[mealwise@8912ac4]`, `[grounded-forge README § Eval]`).

## Critic upgrade: quantitative claims + honest gaps

- **Deterministic quant pre-check (new, runs before the LLM check):** regex-extract every number/metric from a claim (percentages, counts, "N endpoints"); each must literally appear (after normalization) in the cited chunks. Failure = automatic strike, no LLM call. Inflation becomes the cheapest thing to catch.
  - **Policy (decided 2026-07-19): strict exact match.** A number passes only if it appears in a cited chunk after *simple* normalization — whitespace, `%` vs "percent", digit forms — nothing semantic. "~93%" citing a 92.7% chunk is a **strike**; the Refiner rewrites to the exact figure. Rationale: simplest to implement, zero inflation risk, matches the honest-gap ethos. Rejected alternatives: tolerant rounding (the rounding rule is itself code that needs testing) and deferring the decision to M3 (would ambush the milestone). One inflation trap must cover the rounding edge case.
- **Semantic check:** v1 Critic unchanged (claim supported / not-supported + reason, against cited chunks).
- **Refiner:** rewrites the claim without the unsupported quantity, or drops it. Same termination bounds as v1.
- **Honest gap reporting:** retrieval confidence below threshold for a requested bullet/section → artifact states "insufficient evidence in corpus" instead of generating filler.

## Evaluation

Same harness, new eval set:

- **~15 grounding items** — facts about my own projects where I know the right chunk (e.g. "MealWise auth uses JWT" → the commit/README chunk). Report grounding % + recall@k on the job corpus.
- **~5 inflation traps (new)** — claims with wrong numbers or unused skills; measure **inflation-catch rate** (did the Critic strike them?).
- **Side-by-side table:** study-mode vs job-mode grounding % and recall@k — this is the domain-agnostic proof and the v2 README/LinkedIn headline.

## Testing

Extends existing pytest + GitHub Actions CI:

- **Unit (pure logic, no LLM):** git-log parsing (canned `git log` output fixture), heading-section chunking, frontmatter stripping, `corpus.yaml` parsing, quant pre-check.
- **Integration (mocked LLM):** canned Generator response containing an inflated number → assert the Critic strikes it.

## Milestones (2026-07-19 → ~2026-08-01, ~11 h assisted-pace)

- **V2-M0** (~0.5 h) — `corpus.yaml` schema; verify `chunks/` + `chroma_db/` gitignored; adapter interface skeleton
- **V2-M1** (~2.5 h) — four adapters in risk order (git → docs → vault → files-wrap), mid-checkpoint after git_adapter; `ingest --corpus job` produces chunks from both repos + vault. Draft eval items while the corpus is in hand.
- **V2-M2** (~1.5 h) — index + `query` on job corpus with `source_type` filtering. "Retrieval untouched" means the *algorithm* is unchanged (embedding, fusion, rerank); M2's only changes to `retrieve/` are **additive** — pass the adapter metadata (`source_type`, `repo`, `sha`, `date`, `section`) through `store.py` into ChromaDB, and add an optional `where=` filter arg to `search()`. Spot-check retrieval against git chunks (terse commit messages may embed poorly)
- **V2-M3** (~2 h) — `make-bullets` end-to-end: schema, prompts, quant pre-check (strict policy above), gap reporting
- **V2-M4** (~2 h) — `make-star`: per-section retrieval + synthesis
- **V2-M5** (~2.5 h) — job eval set (~15 items + ~5 inflation traps incl. rounding edge); side-by-side numbers; README v2 section; tests green; read every committed sample for private vault content before pushing; LinkedIn draft → **ship**

**Cut order if behind:** (1) drop per-section source filtering — single retrieval for STAR; (2) merge `vault_adapter` into `docs_adapter`; (3) eval set to 10 + traps to 3. **Never cut:** the quant pre-check or the side-by-side grounding table.

## Learning checkpoints (PIP teaches + grades)

Aligned with PIP's ledger (`.pip/progress.md`) — one concept per milestone.

v2 (this phase):

9. **Corpus adapters for semi-structured sources** — per-source chunk units + metadata design (→ V2-M1); prereqs: ingestion-pipeline, text-chunking
10. **Metadata-filtered retrieval** — filter the vector store by `source_type` etc. (→ V2-M2); prereqs: vector-store, nearest-neighbor-retrieval
11. **Deterministic pre-checks before LLM judgment** — quant pre-check + honest gap reporting / decline-to-claim (→ V2-M3); prereqs: claim-verification, structured-llm-output (targets v1 gaps on both)
12. **Cross-source evidence synthesis** — per-section retrieval + evidence pools per output section (→ V2-M4); prereqs: metadata-filtered-retrieval, multi-agent-loop
13. **Adversarial eval traps** — deliberate inflation / wrong-number items measuring the Critic (→ V2-M5); prereqs: grounding-eval (targets v1 gap: calibrating the evaluator)

v2.5 (December, out of scope now):

14. **Query decomposition / structured extraction** — JD → structured requirements → per-requirement retrieval
15. **Requirement–evidence matching (gap analysis)** — map each JD requirement to strongest evidence; report "no evidence" honestly

## Out of scope (v2)

- JD-driven generation (v2.5, December)
- OCR, browser UI, deployment, accounts (unchanged from v1)
- Any change to retrieve/, the agent-loop structure, or the eval harness beyond the new eval set
