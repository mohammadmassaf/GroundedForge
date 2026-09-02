"""
Freeze the demo trap verdicts so the Space's strike panel costs nothing to show.

The cached quiz proves the system produces cited claims. It does NOT show the
Critic striking anything -- that run kept 5 of 5, and so does the committed
quiz_demo.md. A strike panel that only fills when the generator happens to
over-reach is empty for some visitors, and the strike is the product.

The traps are the fix: authored claims with a KNOWN defect, so the panel always
has something real and explained. Runs the same two stages as the live loop, in
the same order, so what is cached is the guard as shipped.

Cost is three Groq calls, not six -- the quant-stage traps are caught
deterministically before any LLM is reached.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from critic.quant import check_quantities
from critic.critic import check_claim
from generate.generator import MODEL

OUT = Path("samples/cached_traps.json")

traps = json.loads(Path("eval/eval_set_demo.json").read_text(encoding="utf-8"))["traps"]
by_id = {c["chunk_id"]: c
         for c in json.loads(Path("chunks/demo.json").read_text(encoding="utf-8"))}

rows = []
for t in traps:
    cited = [by_id[c] for c in t["citations"]]

    # Same order as the real loop: the deterministic figure check first, and the
    # LLM only for what survives it.
    ok, reason = check_quantities(t["claim"], cited)
    if not ok:
        caught_by, struck = "quant", True
    else:
        verdict = check_claim("", t["claim"], cited)
        caught_by = "critic" if not verdict.supported else None
        struck = not verdict.supported
        reason = verdict.reason

    print(f"{t['id']}  expected={t['stage']:6} caught_by={str(caught_by):6} struck={struck}", flush=True)

    rows.append({
        "id": t["id"],
        "claim": t["claim"],
        "citations": t["citations"],
        "expected_stage": t["stage"],
        "caught_by": caught_by,
        "struck": struck,
        "reason": reason,
        "why": t["why"],
        # The source text, so the UI can put the planted claim next to the
        # sentence that contradicts it. Without this the panel is an assertion.
        "evidence": [{"chunk_id": c["chunk_id"], "source_file": c["source_file"],
                      "page": c["page"], "text": c["text"]} for c in cited],
    })

caught = sum(1 for r in rows if r["struck"])
payload = {
    "_comment": (
        "Cached verdicts for the demo corpus's adversarial traps, so the Space can "
        "show the Critic striking a planted claim with no Groq call. Traps are "
        "authored in eval/eval_set_demo.json; regenerate this file from there rather "
        "than editing it. dt4 is a KNOWN escape, kept visible on purpose -- see "
        "eval/notes.md Finding 9."
    ),
    "corpus": "demo",
    "generated_at": datetime.now().strftime("%Y-%m-%d"),
    "model": MODEL,
    "caught": caught,
    "total": len(rows),
    "traps": rows,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\ncaught {caught}/{len(rows)} -> wrote {OUT} ({OUT.stat().st_size / 1000:.1f} KB)")
