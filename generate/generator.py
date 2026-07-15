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

from generate.schema import Quiz , Guide

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 2
class GenerationError(Exception):
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

   bad = set(obj.all_citations()) - valid_ids
   if bad:
        raise ValueError(f"Unknown chunk_id(s) cited: {bad} — "
                                "cite only chunk_ids from the context")
   return obj

def _run(system_prompt, chunks, task, model):
    valid_ids = set(c["chunk_id"] for c in chunks)
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": _build_prompt(chunks, task)},
]
    for i in range(MAX_RETRIES + 1):
        resp = _get_client().chat.completions.create(
            model = MODEL , messages = messages , temperature = 0.3
        )
        raw  = resp.choices[0].message.content
        try:
            return _parse_and_validate(raw,valid_ids , model = model)
        except ValueError as e:
            last_error = e 
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Your response was invalid: {e}. Reply again with corrected JSON only."})
    raise GenerationError(f"Generator failed after "
                                        f"{MAX_RETRIES} retries: {last_error}")
def generate(topic: str, chunks: list[dict], n: int = 5) -> Quiz:
    
    
    
    
    return _run(SYSTEM_PROMPT, chunks, f"Generate {n} quiz items about: {topic}.", Quiz)
   
def generate_guide(topic, chunks, n=5) -> Guide:
    return _run(GUIDE_SYSTEM_PROMPT, chunks, f"Write a study guide about: {topic}.", Guide)