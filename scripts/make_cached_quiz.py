"""
Run the real quiz path once over the demo corpus and freeze the result as JSON.

Why JSON and not the markdown `make-quiz` already writes: the Space has to
render kept items, their citations, AND the struck panel as separate UI pieces.
Markdown is the rendered form -- taking it apart again in app.py would mean
parsing our own output. The cached file carries the same data the renderer
gets, so app.py can call `render()` on it or lay it out itself.

Same call path as cmd_make_quiz (search -> run_loop), so the cached artifact is
a genuine run, not a mock-up.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

TOPIC = sys.argv[1] if len(sys.argv) > 1 else \
    "TCP connection establishment, window management and congestion control"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
K = 8
OUT = Path("samples/cached_quiz.json")

from retrieve.query import search
from critic.loop import run_loop
from generate.generator import MODEL

t0 = time.perf_counter()
chunks = search(TOPIC, corpus="demo", k=K)
print(f"retrieved {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s", flush=True)

kept, struck = run_loop(TOPIC, chunks, n=N, corpus="demo")
print(f"\nkept {len(kept)}, struck {len(struck)}", flush=True)

for item, reason in struck:
    print(f"  STRUCK: {item.question[:70]}")
    print(f"          {reason[:100]}")

payload = {
    "_comment": (
        "A real run of the demo pipeline, frozen so the Space renders instantly "
        "on load and costs no Groq quota. Regenerate with scripts, not by hand -- "
        "an edited artifact is no longer evidence of anything."
    ),
    "topic": TOPIC,
    "corpus": "demo",
    "generated_at": datetime.now().strftime("%Y-%m-%d"),
    "model": MODEL,
    "k": K,
    "n": N,
    # The whole retrieved pool, not just cited chunks: the UI shows a citation's
    # source text, and "what was available but not used" is the same signal the
    # strike analysis reads.
    "chunks": [
        {"chunk_id": c["chunk_id"], "source_file": c["source_file"],
         "page": c["page"], "text": c["text"], "score": round(c["score"], 4)}
        for c in chunks
    ],
    "kept": [
        {"question": i.question, "answer": i.answer, "citations": i.citations}
        for i in kept
    ],
    "struck": [
        {"question": i.question, "answer": i.answer,
         "citations": i.citations, "reason": r}
        for i, r in struck
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1000:.1f} KB)")
