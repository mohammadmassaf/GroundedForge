"""
Run trace: append-only JSONL observability for the agent loop.

One Tracer per run. Every agent step appends one line to
traces/run_<timestamp>.jsonl, e.g.:

  {"ts": "2026-07-04T18:02:11", "event": "critic_verdict",
   "question": "What is the formula for propagation delay?",
   "supported": false, "reason": "cited chunk poses the exercise but ..."}

The rule this file exists to satisfy: "why was this claim struck?" must be
answerable from the trace alone, without re-running anything.
"""
import json
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path("traces")


class Tracer:
    def __init__(self, run_name: str = "run"):
        TRACE_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = TRACE_DIR / f"{run_name}_{stamp}.jsonl"

    def log(self, event: str, **fields) -> None:
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
