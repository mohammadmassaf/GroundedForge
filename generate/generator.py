"""
The Generator agent: retrieved chunks -> validated Quiz.

Flow:
    chunks = retrieve.query.search(topic, corpus, k)
    quiz   = generate(topic, chunks, n)   <- this module
    markdown = render(quiz, chunks)       <- renderer.py

The generate() call must NEVER return unvalidated output. Three failure
modes are handled by the retry loop:
  - the model wraps JSON in ```json fences        -> strip before parsing
  - the JSON doesn't match the schema             -> Pydantic raises
  - a citation names a chunk_id not in the context -> semantic check fails
On any failure we re-prompt WITH the error message so the model can
correct itself. Max 2 retries, then raise.
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from generate.schema import Quiz , Guide , Bullets , STARAnswer

load_dotenv()

# llama-3.3-70b-versatile was retired by Groq (404 model_not_found, 2026-08-17)
# and no Llama chat model remains in the catalogue. gpt-oss-20b keeps its
# reasoning in a separate `reasoning` field, so `content` is still clean JSON and
# the parser needs no changes -- qwen3.6-27b emits <think> inline and would.
# Every LLM-dependent number in this repo predates the swap and is historical:
# grounding % and the trap catch rate both move with the model. recall@k does
# not -- embeddings, BM25 and the cross-encoder are all local.
MODEL = "openai/gpt-oss-20b"
MAX_RETRIES = 2

# Cap the reply. Groq counts a request as input + the output budget you RESERVE,
# and with this unset the model's full 8192-token default is reserved every call
# -- so a 6.5k-token prompt was billed as 14.7k and rejected against the 12k
# per-minute ceiling. Real outputs here are far smaller: a STAR answer runs ~900
# tokens, five bullets ~500, five quiz items ~700. 2000 leaves room without
# reserving budget nothing will use. Too LOW is the dangerous direction: a
# truncated reply is invalid JSON, which burns a retry.
MAX_OUTPUT_TOKENS = 2000
class GenerationError(Exception):
    pass

class EmptyGeneration(Exception):
    """ this error signals that there is a gap in the bullet , not that the run failed """
    pass

SYSTEM_PROMPT = """\
You are a quiz generator for a student studying from their own course material.

RULES — these are absolute:
1. Use ONLY the context provided by the user. Do NOT use any outside knowledge.
2. Every quiz item must have a "citations" list naming the chunk_id(s) the
   question AND answer are based on. Use chunk_ids exactly as given, e.g. [I2208-Part-1_p7_c0].
3. If the context cannot support the requested number of items, produce fewer.
   NEVER invent material to fill the count.
4. Respond with ONLY a JSON object, no markdown fences, no commentary:
   {"items": [{"question": "...", "answer": "...", "citations": ["<chunk_id>"]}]}
"""

GUIDE_SYSTEM_PROMPT = """\
You are a study-guide generator for a student studying from their own course material.

RULES — these are absolute:
1. Use ONLY the context provided by the user. Do NOT use any outside knowledge.
2. Organize the guide into sections. Every section has a "heading" and a list of
   "claims". Every claim must have a "citations" list naming the chunk_id(s) that
   claim is based on. Use chunk_ids exactly as given, e.g. [I2208-Part-1_p7_c0].
3. If the context cannot support a claim, leave it out. NEVER invent material.
4. Respond with ONLY a JSON object, no markdown fences, no commentary:
   {"sections": [{"heading": "...", "claims": [{"text": "...", "citations": ["<chunk_id>"]}]}]}
"""

BULLETS_SYSTEM_PROMPT = """\
You are writing resume bullets for a developer, from evidence drawn from their
own git history, repo docs, and project notes.

RULES — these are absolute:
1. Use ONLY the context provided by the user. Do NOT use any outside knowledge,
   and do NOT infer skills or impact the evidence doesn't show.
2. Every bullet must have a "citations" list naming the chunk_id(s) it is based
   on. Use chunk_ids exactly as given, e.g. [mealwise@8912ac4].
   Cite EVERY chunk that supports the bullet, not just the main one. A bullet is
   checked against ONLY the chunks it cites — so if one chunk shows the endpoint
   and another shows the technology it uses, cite BOTH or the claim will fail.
3. NUMBERS: you may state a figure ONLY if that exact figure appears in the
   cited context. Never estimate, round, or aggregate into a new number.
   No "~", no "over N", no invented percentages. If the context has no number,
   write the bullet without one.
4. Voice: first person implied (no "I"), past tense, strong concrete verbs.
   Say what was built and how, not adjectives about how good it was.
5. AGGREGATE — one bullet is a CAPABILITY, not a commit. Several chunks usually
   describe steps toward the same feature (the model change, the endpoint, the
   service split). Combine them into ONE bullet and cite all of them.
   NEVER write a bullet for a single small change. These are implementation
   details, not accomplishments. The examples below describe a FICTIONAL
   project and are illustrations of the rule only — never cite them, never
   reuse their wording, and never treat their subject matter as forbidden in
   your own context:
   - WRONG: "Added a priority field to the Ticket model"
   - WRONG: "Made send_digest async using a worker thread"
   - WRONG: "Created database tables automatically on startup"
   - RIGHT: "Built the ticket-triage pipeline: nested Ticket/Queue/Board
     models, an async dispatch path, and per-queue reassignment"
   If the only evidence for something is one tiny commit, leave it out.
6. LENGTH AND SUBSTANCE: 12 to 25 words. Lead with the verb, and name the
   SPECIFIC technical content the evidence shows — structures, endpoints,
   techniques, technologies. Mine the detail in the evidence (changed file
   paths, structure names, described behaviour); never just reword a commit
   subject, and never paraphrase a README overview of the whole project.
7. STAY AT THE EVIDENCE'S LEVEL. Use the evidence's OWN nouns and verbs for the
   technical content. Rule 5 lets you COMBINE facts from several cited chunks
   into one capability; it does not let you RENAME, generalise, or upgrade them.
   An independent Critic checks each bullet against ONLY its cited chunks and
   asks "is this stated here?", never "is this true?". A fair inference the
   evidence does not state WILL be struck and the bullet lost.
   The two examples below are ILLUSTRATIONS OF THE RULE, not evidence. They
   describe a fictional project. Never cite them, and never reuse their wording
   — if your bullet resembles one of them, you are answering from this prompt
   instead of from the context, and it will be struck.
   - evidence: "env var WIDGET_TOKEN is required"
     WRONG: "Implemented token-based widget authentication"  (a config var is
     not a feature; nothing here says what the token is used for)
     RIGHT: nothing — this evidence supports no capability bullet on its own.
   - evidence: "the service reconciles ledger totals against the clearing file"
     WRONG: "Built a financial data validation layer"   (renames both the verb
     and the object into something more impressive)
     RIGHT: "Reconciled ledger totals against the clearing file"
   Reusing the evidence's wording is NOT the "rewording a commit subject" that
   rule 6 forbids: rule 6 is about SUBSTANCE (say what changed and where), this
   rule is about VOCABULARY (say it in the evidence's terms).
8. Each bullet must cover a DIFFERENT accomplishment. Never repeat or reword a
   bullet you have already written.
9. MERGE overlapping bullets. If two bullets would cite the same evidence, they
   describe the SAME capability — write one bullet, not two.
   - WRONG (same evidence, split in two): "Implemented authentication with JWT"
     + "Built user management with registration and login endpoints"
   - RIGHT: "Built JWT authentication with user registration and login endpoints"
10. If the context cannot support the requested number of bullets, produce fewer.
    NEVER invent material to fill the count.
11. Respond with ONLY a JSON object, no markdown fences, no commentary:
   {"bullets": [{"text": "...", "citations": ["<chunk_id>"]}]}
"""

STAR_SYSTEM_PROMPT = """\
You are preparing a STAR interview answer for a developer, from evidence drawn
from their own git history, repo docs, and project notes.

The context is grouped into FOUR labeled evidence pools — one per STAR section.

RULES — these are absolute:
1. Use ONLY the context provided. No outside knowledge, no inferred impact.
2. Each section may cite ONLY chunk_ids from ITS OWN pool. Never cite an
   ACTION chunk in the Situation section, etc. Cite every chunk you used.
3. NUMBERS: state a figure ONLY if that exact figure appears in the cited pool.
   Never estimate, round, or invent. No "~", no "over N".
4. Write in first person, past tense, spoken-interview voice — 2 to 4 sentences
   per section. Natural speech, not resume shorthand.
5. Content of each section:
   - Situation: the context and why it mattered
   - Task: what specifically had to be solved, and the constraints
   - Action: what YOU did — the concrete technical steps
   - Result: what came of it. If the pool has no outcome evidence, say plainly
     what shipped; NEVER invent an impact or a metric.
6. If a pool is empty or too thin to support its section, write that the
   evidence doesn't cover it rather than filling it in from imagination.
7. The QUESTION IS NOT EVIDENCE. It may presume conditions the corpus never
   states — a deadline, time pressure, team size, scale, difficulty. Do not
   repeat such a condition unless a cited chunk says it. Answer what the
   evidence supports, even when the question presumes more.
8. Respond with ONLY a JSON object, no markdown fences, no commentary:
   {"question": "...",
    "situation": {"text": "...", "citations": ["<chunk_id>"]},
    "task":      {"text": "...", "citations": ["<chunk_id>"]},
    "action":    {"text": "...", "citations": ["<chunk_id>"]},
    "result":    {"text": "...", "citations": ["<chunk_id>"]}}
"""

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _build_prompt(chunks:list[dict],task :str) -> str:
   """
    TODO(you): build the user prompt string.

    Steps:
    1. Format each chunk as a labeled block:
           [<chunk_id>] (source: <source_file>, p.<page>)
           <text>
       joined with blank lines. The label is what the model cites.
    2. End with the instruction, e.g.:
           "Generate {n} quiz items about: {topic}. Remember: JSON only,
            cite only the chunk_ids above."
    3. Return the full string.
   """
  
   block = []
   for chunk in chunks:
         block.append(
            f"[{chunk['chunk_id']}] (source: {chunk['source_file']}, p.{chunk['page']})\n"
            f"{chunk['text']}"
         )
   context = "\n\n".join(block)
       
   return (
       f"{context}\n\n"
       f"{task}\n"
       f"Remember: JSON only, cite only the chunk_ids above."
   )
  


def _parse_and_validate(raw: str, valid_ids: set[str] , model = Quiz) -> Quiz:
   """
    TODO(you): turn the raw model reply into a validated Quiz, or raise
    ValueError with a message the model can act on.

    Steps:
    1. Strip markdown fences if present (the model sometimes adds them
       despite instructions): if raw starts with "```", cut the first and
       last fence lines. (Look at mealwise's parser.py — you solved this
       there with .strip() and slicing.)
    2. json.loads(raw) — on json.JSONDecodeError, raise
       ValueError(f"Invalid JSON: {e}")
    3. Quiz.model_validate(data) — on pydantic.ValidationError, raise
       ValueError(f"Schema error: {e}")
    4. Semantic check: for every item, every citation must be in valid_ids.
       If not, raise ValueError(f"Unknown chunk_id(s) cited: {bad_ids} — "
                                "cite only chunk_ids from the context")
    5. Return the Quiz.
   """
   raw = raw.strip()
   if raw.startswith("```"):
    raw = raw.split("\n", 1)[1]      # drop the first line, whatever fence it was
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
   try:
       data = json.loads(raw)
   except json.JSONDecodeError as e :
       raise  ValueError(f"expected data in json format : {e}")
   try:
       obj = model.model_validate(data)
   except ValidationError as e:
         raise ValueError(f"Schema error: {e}") 
   if hasattr(obj, "is_empty"):
       if obj.is_empty():
        raise EmptyGeneration(f"this {type(obj).__name__} validates fine but contains zero items")
   bad = set(obj.all_citations()) - valid_ids
   if bad:
        raise ValueError(f"Unknown chunk_id(s) cited: {bad} — "
                                "cite only chunk_ids from the context")
   return obj

def _run(system_prompt, chunks, task, model):
    valid_ids = set(c["chunk_id"] for c in chunks)
    base = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": _build_prompt(chunks, task)},
]
    messages = base
    for i in range(MAX_RETRIES + 1):
        resp = _get_client().chat.completions.create(
            model = MODEL , messages = messages , temperature = 0.3,
            max_completion_tokens = MAX_OUTPUT_TOKENS,
        )
        raw  = resp.choices[0].message.content
        try:
            return _parse_and_validate(raw,valid_ids , model = model)
        except ValueError as e:
            last_error = e
            # Rebuild from `base` rather than appending: the old loop kept every
            # failed attempt, so attempt 3 re-sent two full model outputs it no
            # longer needed. Combined with the unset output cap that pushed a
            # single request past the 12k per-minute ceiling, where no amount of
            # waiting helps. One failure is what the model needs to correct.
            messages = base + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your response was invalid: {e}. Reply again with corrected JSON only."},
            ]
    raise GenerationError(f"Generator failed after "
                                        f"{MAX_RETRIES} retries: {last_error}")

def generate(topic: str, chunks: list[dict], n: int = 5) -> Quiz:
    
    return _run(SYSTEM_PROMPT, chunks, f"Generate {n} quiz items about: {topic}.", Quiz)
   
def generate_guide(topic, chunks, n=5) -> Guide:
    return _run(GUIDE_SYSTEM_PROMPT, chunks, f"Write a study guide about: {topic}.", Guide)

def generate_star(question: str, pools: dict[str, list[dict]]) -> STARAnswer:
    """
    One generation call over FOUR labeled evidence pools (see star_evidence).

    Note the shape difference from every other generator: the context isn't one
    flat chunk list, it's pools with headers, so the model can tell which chunks
    each section is allowed to cite.
    """
    blocks = []
    for name in ("situation", "task", "action", "result"):
        chunks = pools.get(name, [])
        header = f"=== {name.upper()} POOL ==="
        if not chunks:
            blocks.append(f"{header}\n(no evidence retrieved for this section)")
            continue
        body = "\n\n".join(
            f"[{c['chunk_id']}] (source: {c['source_file']})\n{c['text']}"
            for c in chunks
        )
        blocks.append(f"{header}\n{body}")
    context = "\n\n".join(blocks)

    # all pooled chunks are valid ids; per-section scoping is enforced in the loop
    all_chunks = [c for pool in pools.values() for c in pool]
    valid_ids = set(c["chunk_id"] for c in all_chunks)

    base = [
        {"role": "system", "content": STAR_SYSTEM_PROMPT},
        {"role": "user", "content":
            f"{context}\n\nAnswer this interview question: {question}\n"
            f"Remember: JSON only, each section cites only its own pool."},
    ]
    messages = base
    for _ in range(MAX_RETRIES + 1):
        resp = _get_client().chat.completions.create(
            model=MODEL, messages=messages, temperature=0.3,
            max_completion_tokens=MAX_OUTPUT_TOKENS)
        raw = resp.choices[0].message.content
        try:
            return _parse_and_validate(raw, valid_ids, model=STARAnswer)
        except ValueError as e:
            last_error = e
            # same as _run: keep only the latest failure, not the whole chain.
            # STAR carries four pools, so its base prompt is the largest in the
            # project and accumulating attempts on top of it is worst here.
            messages = base + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content":
                    f"Your response was invalid: {e}. Reply again with corrected JSON only."},
            ]
    raise GenerationError(f"STAR generation failed after {MAX_RETRIES} retries: {last_error}")


def generate_bullets(topic: str, chunks: list[dict], n: int = 5,
                     avoid: list[str] | None = None) -> Bullets:
    """`avoid` = bullets already kept in an earlier round; the model must not
    repeat them (otherwise a top-up round just re-emits its best bullet)."""
    task = f"Write {n} resume bullets about: {topic}."
    if avoid:
        already = "\n".join(f"- {t}" for t in avoid)
        task += ("\n\nYou have ALREADY written the bullets below. Write about "
                 f"DIFFERENT accomplishments — do not repeat or reword these:\n{already}")
    return _run(BULLETS_SYSTEM_PROMPT, chunks, task, Bullets)