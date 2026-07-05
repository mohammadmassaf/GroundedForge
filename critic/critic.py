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
from generate.generator import _get_client, MODEL

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


def _build_critic_prompt(question: str, answer: str, cited_chunks: list[dict]) -> str:
   """
    TODO(you): build the Critic's user prompt.

    Steps:
    1. Format the cited chunks as labeled EVIDENCE blocks — same pattern as
       _build_user_prompt in the generator ([chunk_id] header + text).
    2. Present the claim, e.g.:
           CLAIM
           Question: <question>
           Answer: <answer>
    3. End with: "Is the claim fully supported by the evidence? JSON only."
    4. Return the full string.
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
    TODO(you): raw model reply -> Verdict, or raise ValueError.

    Same three-layer idea as _parse_and_validate, minus the semantic layer
    (a Verdict has no runtime ids to cross-check):
    1. strip fences if present
    2. json.loads -> ValueError(f"Invalid JSON: {e}")
    3. Verdict.model_validate -> ValueError(f"Schema error: {e}")
    4. return the Verdict
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
    TODO(you): the Critic call with a retry loop.

    Same shape as generate(), smaller:
    1. messages = [critic system prompt, user prompt from _build_critic_prompt]
    2. loop MAX_RETRIES + 1 times:
       call the LLM with temperature=0.0
       try: return _parse_verdict(raw)
       except ValueError as e: keep last_error, append the assistant reply
           + a correction message (same pattern as generate())
    3. after the loop: raise SystemExit(f"Critic failed: {last_error}")
    """
   messages = [
      {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
      {"role": "user",   "content": _build_critic_prompt(question,answer,cited_chunks)},
]
   for i in range(MAX_RETRIES + 1):
        resp = _get_client().chat.completions.create(
            model = MODEL , messages = messages , temperature = 0.0
        )
        raw  = resp.choices[0].message.content
        try:
            return _parse_verdict(raw)
        except ValueError as e:
            last_error = e 
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Your response was invalid: {e}. Reply again with corrected JSON only."})
   raise SystemExit(f"Critic failed: {last_error}")
   
   
