"""
The Critic agent: one claim + its cited chunks -> Verdict.

Design rules (why this looks the way it does):
- INDEPENDENT: a fresh LLM call with no memory of generation. The Critic
  never sees the Generator's conversation — only the claim and the evidence.
- NARROW CONTEXT: only the chunks the claim CITED, not all retrieved
  chunks. If the answer needed a chunk it didn't cite, that's a grounding
  failure too — the citation is the claim's whole case.
- STRICT: temperature 0.0 — verification wants determinism, not creativity.
  "Partially supported" = not supported.
"""
import json

from pydantic import ValidationError

from critic.schema import Verdict
from generate.generator import _complete, MODEL, GenerationError

CRITIC_SYSTEM_PROMPT = """\
You are a strict fact-checker. You will be given a CLAIM (a quiz question
and its answer) and the EVIDENCE (the exact source excerpts the claim cites).

Decide: is every part of the claim directly supported by the evidence?

RULES:
1. Judge ONLY against the evidence text. Your own knowledge is irrelevant —
   a true statement that is not in the evidence is NOT supported.
2. Computed or derived values count as supported ONLY if the evidence shows
   the result (not merely the formula or the exercise statement).
3. Partially supported = not supported.
4. Respond with ONLY a JSON object, no markdown fences:
   {"supported": true/false, "reason": "<one or two sentences>"}
"""

MAX_RETRIES = 1

# A verdict is {"supported": bool, "reason": "..."} -- ~100 tokens. Reserving the
# model's full default on every one of these (2 per claim, dozens per eval run)
# is what pushed requests past the 12k/minute ceiling. See the note on
# generator.MAX_OUTPUT_TOKENS.
#
# Raised from 400 for gpt-oss-20b: reasoning tokens are billed as OUTPUT and share
# this budget, so the cap now covers thinking plus the verdict. Truncation is the
# expensive failure here -- a cut-off reply is invalid JSON, which costs a whole
# retry, while an unused reservation costs nothing against the daily limit.
MAX_OUTPUT_TOKENS = 900


def _build_critic_prompt(question: str, answer: str, cited_chunks: list[dict]) -> str:
   """
    Build the Critic's user prompt: the cited chunks as labeled EVIDENCE
    blocks, then the claim, then the question to answer.

    Only the CITED chunks go in, never the whole pool. The Critic's job is
    "does this claim follow from what it points at", not "is this claim true
    somewhere in the corpus" - so a claim supported by a chunk it failed to
    cite is struck, correctly.
   """
   block = []
   for chunk in cited_chunks:
       block.append(
           f"[{chunk['chunk_id']}] (source: {chunk['source_file']}, p.{chunk['page']})\n"
           f"{chunk['text']}"
       )
   context = "\n\n".join(block)
   return (
       f"EVIDENCE\n{context}\n\n"
       f"CLAIM\nQuestion: {question}\nAnswer: {answer}\n\n"
       f"Is the claim fully supported by the evidence? JSON only."
   )


def _parse_verdict(raw: str) -> Verdict:
   """
    Raw model reply -> Verdict, or ValueError.

    Two layers, the same shape as _parse_and_validate in the generator minus
    its semantic layer (a Verdict carries no chunk ids to cross-check): strip
    any code fence the model added despite instructions, then json.loads, then
    Verdict.model_validate. Both failures surface as ValueError so
    check_claim's retry loop only has to catch one thing.
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
      verdict = Verdict.model_validate(data)
   except ValidationError as e:
      raise ValueError(f"Schema error: {e}") 
   return verdict


def check_claim(question: str, answer: str, cited_chunks: list[dict]) -> Verdict:
   """
    Ask the Critic whether `answer` is supported by `cited_chunks`, with a
    retry-on-invalid loop.

    temperature=0.0: a verdict must be reproducible. The same claim against the
    same evidence cannot be supported on one run and struck on the next, or the
    grounding score is measuring sampling noise.

    On a malformed reply the bad output and a correction are appended to the
    message list and the model is asked again, so it sees its own mistake -
    same pattern as generate(). Exhausting the retries is fatal: a Critic that
    cannot answer must never be defaulted to "supported".
    """
   base = [
      {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
      {"role": "user",   "content": _build_critic_prompt(question,answer,cited_chunks)},
]
   messages = base
   for i in range(MAX_RETRIES + 1):
        resp = _complete(messages, 0.0, MAX_OUTPUT_TOKENS)
        raw  = resp.choices[0].message.content
        try:
            return _parse_verdict(raw)
        except ValueError as e:
            last_error = e
            messages = base + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your response was invalid: {e}. Reply again with corrected JSON only."},
            ]
   raise GenerationError(f"Critic failed: {last_error}")
   
   
