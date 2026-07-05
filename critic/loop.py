"""
The multi-agent loop: Generator -> Critic -> Refiner.

run_loop() orchestrates one full cite-or-strike run:

    round 1: generate n items
             critique each item          -> kept / struck
    round 2: if short and rounds remain: regenerate the missing count
             critique the new items too  -> kept / struck
    ...
    stop when: enough kept items, OR max_rounds exhausted.

Returns (kept, struck):
    kept   = list[QuizItem]                    — survived the Critic
    struck = list[tuple[QuizItem, str reason]] — failed, with why

The Refiner here is POLICY CODE, not an LLM: deciding what to do with
verdicts (drop, regenerate, stop) is deterministic logic — another case of
"move correctness out of the LLM and into code."

Every step logs to the Tracer, so a run is reconstructable from its trace.
"""
from generate.generator import generate
from generate.schema import QuizItem
from critic.critic import check_claim
from critic.trace import Tracer

MAX_ROUNDS = 2


def run_loop(topic: str, chunks: list[dict], n: int = 5) -> tuple[list, list]:
   """
    TODO(you): the orchestration.

    Steps:
    1. tracer = Tracer("quiz"); by_id = {c["chunk_id"]: c for c in chunks}
       kept, struck = [], []
    2. for round_num in range(1, MAX_ROUNDS + 1):
         a. how many items are still needed? need = n - len(kept)
            if need == 0: break
         b. quiz = generate(topic, chunks, n=need)
            tracer.log("generated", round=round_num, count=len(quiz.items))
         c. for each item in quiz.items:
              cited_chunks = [by_id[cid] for cid in item.citations]
              verdict = check_claim(item.question, item.answer, cited_chunks)
              tracer.log("critic_verdict", round=round_num,
                         question=item.question,
                         supported=verdict.supported, reason=verdict.reason)
              if verdict.supported: kept.append(item)
              else: struck.append((item, verdict.reason))
    3. tracer.log("done", kept=len(kept), struck=len(struck))
       print(f"Trace: {tracer.path}")
    4. return kept, struck
    """
   tracer = Tracer("quiz")
   by_id = {c["chunk_id"]: c for c in chunks}
   kept ,struck = [] , []
   for round_num in range(1, MAX_ROUNDS + 1):
       need = n - len(kept)
       if need == 0: break
       quiz = generate(topic,chunks,n = need)
       tracer.log("generated" , round = round_num , count = len(quiz.items))
       for item in quiz.items:
           cited_chunks = [by_id[cid] for cid in item.citations]
           verdict = check_claim(item.question, item.answer,cited_chunks)
           tracer.log("critic_verdict", round=round_num,
                         question=item.question,
                         supported=verdict.supported, reason=verdict.reason)
           if verdict.supported: kept.append(item)
           else: struck.append((item, verdict.reason))
   tracer.log("done", kept=len(kept), struck=len(struck))
   print(f"Trace: {tracer.path}")

   return kept,struck
   
   
