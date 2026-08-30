"""
Run trace: append-only JSONL observability for the agent loop.

One Tracer per run. Every agent step appends one line to
traces/run_<timestamp>.jsonl, e.g.:

  {"ts": "2026-07-04T18:02:11", "event": "critic_verdict",
   "question": "What is the formula for propagation delay?",
   "supported": false, "reason": "cited chunk poses the exercise but ..."}

Line 1 of every trace is a `run_start` carrying the corpus, so the file says
what it was measuring without the reader inferring it from the filename.

The rule this file exists to satisfy: "why was this claim struck?" must be
answerable from the trace alone, without re-running anything. Its corollary,
learned the hard way: a trace must record at least what the traced function
RETURNS. `scope_violation` once logged less than run_star_loop returned, and
the missing strike was invisible until a replay disagreed with the run's own
`done` count.
"""
import json
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path("traces")


class Tracer:
    def __init__(self, run_name: str = "run", corpus: str | None = None):
        TRACE_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = TRACE_DIR / f"{run_name}_{stamp}.jsonl"
        self.corpus = corpus
        # Always the first line: which corpus this run read. A trace stores
        # chunk IDs, never chunk text, so resolving those IDs means loading
        # chunks/<corpus>.json -- and without this field a consumer has to
        # infer the corpus from the FILENAME (bullets_* => job), which is a
        # guess dressed up as a naming convention. `corpus` may be None for
        # runs that genuinely have none; that is still recorded, because
        # "unknown" and "never asked" are different states.
        self.log("run_start", run=run_name, corpus=corpus)

    def log(self, event: str, **fields) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False , default = str) + "\n")
