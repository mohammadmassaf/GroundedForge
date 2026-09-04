"""
Grounded Forge — the public demo (Hugging Face Space, Gradio SDK).

WHAT THIS IS
------------
A thin UI over the same functions `main.py` calls: `search` -> `run_loop` ->
render. No engine code lives here. If this file starts making decisions about
grounding, the decision is in the wrong place.

THREE CONSTRAINTS SHAPE EVERY CHOICE BELOW
------------------------------------------
1. The page must be useful with ZERO model loaded and ZERO Groq spent. A cold
   Space needs ~16s to load torch and the embedder, and a recruiter will not
   wait. So the page opens on a CACHED real run (samples/cached_quiz.json) and
   only pays for the model if someone actually presses Generate.

2. A public URL can drain the free tier. Generation is rate limited per session
   AND process-wide per day; when the budget is gone the button serves the
   cached run instead of failing.

3. The strike is the product, and a live run may not produce one — the cached
   run kept 5 of 5, and so does the committed quiz_demo.md. So the struck panel
   is fed by the ADVERSARIAL TRAPS (samples/cached_traps.json), which are
   authored claims with known defects. It is always populated, always
   explained, and costs nothing.

DEPLOY NOTE
-----------
Docker is a paid SDK on HF's free tier, so this is a Gradio Space on ZeroGPU.
Gradio Spaces have no build step: HF installs requirements.txt and runs this
file. Anything not committed has to be reconstructed here at boot, which is
what `ensure_store` does — rebuilding the Chroma index from the committed
vector pack in ~3s without loading a model.
"""
import json
import os
import time
from datetime import date
from pathlib import Path

# Every path in this project is relative to the repo root -- chunks/<corpus>.json,
# chroma_db/, index/, samples/, traces/. main.py gets away with it because a CLI
# is run from the directory you cloned into. A Space is not: HF starts app.py
# with some other working directory, and the first thing that touched disk died
# with FileNotFoundError: 'chunks/demo.json' -- a file that WAS deployed, one
# directory away.
#
# Establishing the invariant here rather than rewriting every path to be
# absolute: app.py is an entrypoint, and an entrypoint is exactly where a
# process-wide assumption like "cwd is the repo root" belongs. Before the
# project imports, so nothing can read a file first.
os.chdir(Path(__file__).resolve().parent)

import gradio as gr                                        # noqa: E402

from retrieve.store import ensure_store                    # noqa: E402

# --- tunables, all of them constraint 2 --------------------------------------

CORPUS = "demo"
K = 8                      # matches the eval; do not diverge without re-measuring

# Four, not five. A live run at n=5 measured 1995 of 2000 output tokens, because
# gpt-oss-20b bills reasoning against the same budget. Five tokens of headroom is
# not a margin to hand to strangers typing arbitrary topics.
N_ITEMS = 4

COOLDOWN_SECONDS = 20      # per browser session
DAILY_RUN_BUDGET = 40      # process-wide; resets on restart, which is fine

CACHED_QUIZ = Path("samples/cached_quiz.json")
CACHED_TRAPS = Path("samples/cached_traps.json")

REPO_URL = "https://github.com/mohammadmassaf/GroundedForge"


# --- boot ---------------------------------------------------------------------

def _boot():
    """Rebuild the vector store if this container has never done it.

    Deliberately NOT wrapped in a try: a Space that cannot build its index
    should fail loudly at startup rather than serve a page where every search
    silently returns nothing. An empty result set reads downstream as "the
    corpus has nothing on this" — a grounding finding — which is exactly the
    wrong story to tell about a broken deployment.
    """
    try:
        built = ensure_store(CORPUS)
    except FileNotFoundError as e:
        # A missing input here is a DEPLOYMENT fault, not a bug in the code, and
        # the bare FileNotFoundError names one relative path while saying nothing
        # about what did arrive. On a host whose logs you cannot stream, the
        # traceback is the only channel there is -- so put the evidence in it.
        here = Path.cwd()
        listing = sorted(p.name + ("/" if p.is_dir() else "") for p in here.iterdir())
        raise SystemExit(
            f"boot failed: {e}\n"
            f"  cwd        : {here}\n"
            f"  contents   : {listing}\n"
            f"  chunks/    : {sorted(p.name for p in (here / 'chunks').iterdir()) if (here / 'chunks').is_dir() else 'MISSING'}\n"
            f"  index/     : {sorted(p.name for p in (here / 'index').iterdir()) if (here / 'index').is_dir() else 'MISSING'}\n"
            f"  samples/   : {sorted(p.name for p in (here / 'samples').iterdir()) if (here / 'samples').is_dir() else 'MISSING'}"
        ) from e
    print(f"[boot] store ready (rebuilt={built})", flush=True)


_boot()


# ZeroGPU refuses to start a Space that declares no @spaces.GPU function:
# "No @spaces.GPU function detected during startup". This app wants no GPU at
# all -- retrieval is MiniLM on CPU over 526 chunks, and generation is a Groq
# API call -- but ZeroGPU is the only hardware a free account can run, since
# Docker is a paid SDK and CPU Basic is PRO-gated. So the requirement is
# satisfied honestly rather than pretended away: this declares a GPU function,
# and the docstring says plainly that nothing in the demo needs one.
#
# Guarded by ImportError so the app still runs locally and in CI, where the
# `spaces` package is a Space-only dependency and nothing is decorated.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_declaration():
        """Exists only to satisfy ZeroGPU's startup check.

        Not called on any request path. If this demo ever earns a GPU, the
        honest place to spend it is embedding the user's query -- currently the
        ~16s of model load on the first Generate click.
        """
        return "ok"

except ImportError:
    pass    # not on a ZeroGPU Space; nothing to declare


CACHED = json.loads(CACHED_QUIZ.read_text(encoding="utf-8"))
TRAPS = json.loads(CACHED_TRAPS.read_text(encoding="utf-8"))

_spent = {"date": date.today(), "runs": 0}


def _budget_left() -> int:
    if _spent["date"] != date.today():
        _spent.update(date=date.today(), runs=0)
    return max(0, DAILY_RUN_BUDGET - _spent["runs"])


# --- rendering ----------------------------------------------------------------
#
# ONE renderer for both the cached artifact and a live run. A live QuizItem is
# converted to the cached file's shape before it gets here, so the two paths
# cannot drift apart and the cached file keeps looking like what it is: a real
# run, not a mock-up.

def _quote(chunk: dict, limit: int = 240) -> str:
    return " ".join(chunk["text"].split())[:limit]


# How much of two questions' wording must overlap before the second is treated
# as a restatement of the first. 0.8 separates the observed case -- "What is the
# purpose of the TCP PUSH flag?" against "What is the purpose of the PUSH flag
# in TCP?", which share 8 of 9 words (0.89) -- from genuinely different
# questions about one field, which in the cached UDP run peak around 0.4.
DUPLICATE_OVERLAP = 0.8


def _words(question: str) -> set:
    return {w.strip(".,;:!?\"'()[]").lower() for w in question.split() if w.strip()}


def distinct(items: list[dict]) -> list[dict]:
    """
    Drop questions that restate one already kept, first occurrence winning.

    WHY THIS IS IN THE DEMO AND NOT IN run_loop
    -------------------------------------------
    The repetition is a real Generator weakness -- an off-corpus topic narrowed
    the pool to one chunk and it asked about that chunk twice, in two phrasings.
    The place to fix that is the Generator prompt, and that is exactly why it is
    not being fixed here: the prompt is what the published grounding number was
    measured on, and run_loop is what grounding_eval calls. Deduplicating there
    would change the denominator of a number the README quotes, for a
    presentation problem.

    So this trims what the PAGE shows and nothing else. The underlying
    repetition is still in the trace, still visible to the eval, and still open
    as a prompt fix with a before/after -- the same call made for the Critic's
    dt4 miss.

    Compares word sets rather than strings, because the duplicate pair differed
    only in word order and had no string overlap worth matching on.
    """
    kept: list[dict] = []
    for item in items:
        words = _words(item["question"])
        if any(len(words & _words(k["question"])) / max(1, len(words | _words(k["question"])))
               >= DUPLICATE_OVERLAP for k in kept):
            continue
        kept.append(item)
    return kept


def render_quiz(payload: dict) -> str:
    by_id = {c["chunk_id"]: c for c in payload["chunks"]}
    lines = []
    for i, item in enumerate(payload["kept"], 1):
        lines.append(f"### Q{i}. {item['question']}")
        lines.append("")
        lines.append(f"**Answer:** {item['answer']}")
        lines.append("")
        for cid in item["citations"]:
            chunk = by_id.get(cid)
            if chunk is None:            # cannot happen for a validated run; see tests
                lines.append(f"> `{cid}` — source not in this artifact")
                continue
            lines.append(
                f"> **`{cid}`** · {chunk['source_file']} p.{chunk['page']}  \n"
                f"> *“{_quote(chunk)}…”*"
            )
        lines.append("")

    if payload.get("struck"):
        lines.append("---")
        lines.append("#### Struck by the Critic in this run")
        for item in payload["struck"]:
            lines.append(f"- ~~{item['question']}~~ — **{item['answer']}**")
            lines.append(f"  - *Struck because:* {item.get('reason', '')}")
        lines.append("")

    return "\n".join(lines)


def render_traps() -> str:
    """
    The strike panel. Authored claims with known defects, each shown beside the
    sentence that contradicts it — a strike the reader can check rather than
    take on trust.
    """
    caught, total = TRAPS["caught"], TRAPS["total"]
    lines = [
        f"**{caught} of {total} planted claims were struck.** Each claim below was "
        "written to be wrong in a specific way, then put through the same two "
        "checks a generated claim faces.",
        "",
    ]
    for t in TRAPS["traps"]:
        if t["struck"]:
            stage = ("a deterministic figure check, no LLM call"
                     if t["caught_by"] == "quant" else "the Critic, reading for meaning")
            head = f"✅ **Struck** by {stage}"
        else:
            head = "❌ **Not caught** — a known gap, kept visible on purpose"
        lines.append(f"**{t['id']}** · {head}")
        lines.append(f"> **Claim:** {t['claim']}")
        evidence = t["evidence"][0]
        lines.append(
            f"> **Evidence** (`{evidence['chunk_id']}`, {evidence['source_file']} "
            f"p.{evidence['page']}): *“{_quote(evidence, 300)}…”*"
        )
        if t["struck"]:
            lines.append(f"> **Verdict:** {t['reason']}")
        else:
            lines.append(
                "> **Verdict:** the Critic answered *supported*, quoting the very "
                "sentence that contradicts the claim. Two prompt fixes failed and "
                "were reverted, so this is reported rather than hidden."
            )
        lines.append("")
    return "\n".join(lines)


# --- the live path ------------------------------------------------------------

def generate(topic: str, last_run: float):
    """
    Run the real pipeline for `topic`, or explain why it did not.

    Returns (status_markdown, quiz_markdown, new_last_run). Every failure path
    falls back to the cached run so the page is never empty — a blank page
    teaches a visitor nothing about grounding.
    """
    topic = (topic or "").strip()
    now = time.time()

    if not topic:
        return ("Type a topic, or pick one of the examples.",
                render_quiz(CACHED), last_run)

    waited = now - last_run
    if last_run and waited < COOLDOWN_SECONDS:
        return (f"⏳ One live run every {COOLDOWN_SECONDS}s — "
                f"{COOLDOWN_SECONDS - waited:.0f}s to go. Showing the saved run.",
                render_quiz(CACHED), last_run)

    if _budget_left() <= 0:
        return ("🌙 The live demo is resting — today's generation budget is spent. "
                "Below is a saved run over the same corpus.",
                render_quiz(CACHED), last_run)

    # Imports here, not at module scope: this is the first thing that pulls in
    # torch and the embedder (~16s on a cold container). Doing it at import time
    # would make the PAGE wait for it, which is the cost this whole design exists
    # to avoid.
    from retrieve.query import search
    from critic.loop import run_loop
    from generate.generator import GenerationError, EmptyGeneration

    _spent["runs"] += 1
    started = time.time()
    try:
        chunks = search(topic, corpus=CORPUS, k=K)
        kept, struck = run_loop(topic, chunks, n=N_ITEMS, corpus=CORPUS)
    except EmptyGeneration:
        return (_no_coverage(topic), render_quiz(CACHED), now)
    except GenerationError:
        # The measured failure mode for an off-corpus topic: the model spends its
        # whole output budget reasoning (1998 of 2000 tokens) and returns nothing,
        # three times over. Presenting that as a crash would be both ugly and
        # misleading — the honest reading is that the corpus cannot answer.
        return (_no_coverage(topic), render_quiz(CACHED), now)
    except Exception as e:                      # noqa: BLE001 - the UI must not 500
        # The message, not just the type. On a host whose container logs cannot
        # be streamed, this line is the only diagnostic channel there is -- the
        # first live run on the Space failed as a bare "RuntimeError", which
        # says nothing about which of a dozen causes it was. Truncated, because
        # an exception string is not a place to trust with unbounded output.
        print(f"[generate] {type(e).__name__}: {e}", flush=True)
        return (f"⚠️ The run failed: `{type(e).__name__}: {str(e)[:300]}`. "
                f"Showing the saved run.",
                render_quiz(CACHED), now)

    if not kept and not struck:
        return (_no_coverage(topic), render_quiz(CACHED), now)

    shown = distinct([{"question": i.question, "answer": i.answer,
                       "citations": i.citations} for i in kept])
    payload = {
        "chunks": chunks,
        "kept": shown,
        "struck": [{"question": i.question, "answer": i.answer,
                    "citations": i.citations, "reason": r} for i, r in struck],
    }
    took = time.time() - started
    status = (f"**{len(kept)} kept, {len(struck)} struck** in {took:.1f}s "
              f"· {len(chunks)} chunks retrieved · {_budget_left()} live runs left today")
    repeats = len(kept) - len(shown)
    if repeats:
        # Said out loud rather than hidden: the run really did produce them, and
        # a count that does not match the questions on screen is the kind of
        # small dishonesty this project exists to avoid.
        status += (f"  \n{repeats} near-duplicate question"
                   f"{'s' if repeats > 1 else ''} hidden — the corpus is thin on this topic.")
    if not struck:
        status += "  \nNothing was struck this time — the panel below shows the Critic on planted claims."
    return (status, render_quiz(payload), now)


def _no_coverage(topic: str) -> str:
    """The honest-gap message. This is not an error state — it is the system
    working: the RFCs do not cover the topic, so nothing is invented to fill
    the space."""
    return (
        f"**Nothing in the corpus supports “{topic}”.**  \n"
        "The demo reads only RFC 768, 791 and 793 (UDP, IP, TCP). Rather than "
        "inventing an answer, it says so — that refusal is the point of the "
        "project. Try one of the examples, and see the saved run below."
    )


# --- UI -----------------------------------------------------------------------

PRESETS = [
    "TCP connection establishment and the three-way handshake",
    "UDP checksum and the pseudo header",
    "IP fragmentation and reassembly",
    "The Time to Live field",
]

INTRO = f"""
# Grounded Forge — cite-or-strike, on public RFCs

Every claim below cites a specific chunk of a source document. An **independent
Critic** then re-reads each claim against *only the chunks it cited* and strikes
anything the sources do not support. Nothing unsupported reaches the page.

The corpus is three public-domain IETF RFCs — **768 (UDP), 791 (IP), 793 (TCP)** —
so you can check any citation yourself. [Source and measurements]({REPO_URL}).
"""

FOOTER = f"""
---
Shown on load: a real run, frozen so the page costs nothing to open —
*“{CACHED['topic']}”*, {CACHED['model']}, k={CACHED['k']}, {CACHED['generated_at']}.
Live runs are capped at {N_ITEMS} items and rate limited; this is a free-tier demo.
"""

with gr.Blocks(title="Grounded Forge") as demo:   # theme goes to launch() in Gradio 6
    last_run = gr.State(0.0)

    gr.Markdown(INTRO)

    with gr.Row():
        topic_box = gr.Textbox(
            label="Topic",
            placeholder="e.g. TCP connection establishment and the three-way handshake",
            scale=4,
            # Gradio 6 grows a Textbox to fill its row unless the line count is
            # pinned, which on the Space rendered a one-line input as a textarea
            # tall enough to push the quiz below the fold.
            lines=1,
            max_lines=2,
        )
        go = gr.Button("Generate cited quiz", variant="primary", scale=1)

    with gr.Row():
        for preset in PRESETS:
            btn = gr.Button(preset, size="sm")
            btn.click(lambda p=preset: p, outputs=topic_box)

    status = gr.Markdown("Showing a saved run. Press **Generate** for a live one.")
    quiz_out = gr.Markdown(render_quiz(CACHED))

    with gr.Accordion("Does the Critic actually catch anything? — planted claims",
                      open=False):
        gr.Markdown(render_traps())

    gr.Markdown(FOOTER)

    go.click(generate, inputs=[topic_box, last_run],
             outputs=[status, quiz_out, last_run])
    topic_box.submit(generate, inputs=[topic_box, last_run],
                     outputs=[status, quiz_out, last_run])


if __name__ == "__main__":
    # HF Spaces sets PORT; locally default to Gradio's usual 7860.
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)),
                theme=gr.themes.Soft())
