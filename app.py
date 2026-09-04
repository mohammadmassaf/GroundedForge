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
import html
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


def esc(text: str) -> str:
    """
    HTML-escape before interpolating anything into the page.

    Everything rendered here is either model output or corpus text -- neither is
    written by us. RFC 791 is full of ASCII header diagrams made of `+-+-+` and
    angle brackets (`<SEQ=100><CTL=SYN>`), which a browser will happily read as
    tags and swallow. So this is a correctness fix before it is a security one,
    though it is both.
    """
    return html.escape(str(text), quote=False)


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


def _citation(cid: str, by_id: dict) -> str:
    chunk = by_id.get(cid)
    if chunk is None:                 # cannot happen for a validated run; see tests
        return (f'<div class="gf-cite gf-cite-missing">'
                f'<span class="gf-cid">{esc(cid)}</span> — source not in this artifact</div>')
    return (
        f'<div class="gf-cite">'
        f'<span class="gf-cid">{esc(cid)}</span>'
        f'<span class="gf-src">{esc(chunk["source_file"])} · p.{esc(chunk["page"])}</span>'
        f'<span class="gf-quote">“{esc(_quote(chunk))}…”</span>'
        f'</div>'
    )


def render_quiz(payload: dict) -> str:
    """
    One artifact as HTML.

    HTML rather than markdown because the page has to make three things
    structurally different at a glance -- the question asked, the answer
    claimed, and the evidence it rests on. In markdown those are all just
    paragraphs, which is why the first version read as an undifferentiated wall.
    """
    by_id = {c["chunk_id"]: c for c in payload["chunks"]}
    out = []

    for i, item in enumerate(payload["kept"], 1):
        cites = "".join(_citation(cid, by_id) for cid in item["citations"])
        out.append(
            f'<article class="gf-item">'
            f'  <div class="gf-head"><span class="gf-n">Q{i}</span>'
            f'    <h3 class="gf-q">{esc(item["question"])}</h3></div>'
            f'  <p class="gf-a"><span class="gf-a-label">Answer</span>{esc(item["answer"])}</p>'
            f'  {cites}'
            f'</article>'
        )

    if payload.get("struck"):
        rows = "".join(
            f'<div class="gf-struck-row">'
            f'  <p class="gf-struck-claim">{esc(s["question"])} — {esc(s["answer"])}</p>'
            f'  <p class="gf-struck-why">Struck because: {esc(s.get("reason", ""))}</p>'
            f'</div>'
            for s in payload["struck"]
        )
        out.append(
            f'<section class="gf-struck">'
            f'  <h4>Struck by the Critic in this run</h4>{rows}</section>'
        )

    return f'<div class="gf-artifact">{"".join(out)}</div>'


def render_traps() -> str:
    """
    The strike panel. Authored claims with known defects, each shown beside the
    sentence that contradicts it — a strike the reader can check rather than
    take on trust.
    """
    caught, total = TRAPS["caught"], TRAPS["total"]
    rows = []
    for t in TRAPS["traps"]:
        if t["struck"]:
            stage = ("a deterministic figure check — no LLM call"
                     if t["caught_by"] == "quant" else "the Critic, reading for meaning")
            badge = f'<span class="gf-badge gf-ok">Struck</span> by {esc(stage)}'
            verdict = esc(t["reason"])
        else:
            badge = ('<span class="gf-badge gf-bad">Not caught</span> '
                     "— a known gap, kept visible on purpose")
            verdict = ("the Critic answered <em>supported</em>, quoting the very "
                       "sentence that contradicts the claim. Two prompt fixes failed "
                       "and were reverted, so this is reported rather than hidden.")
        ev = t["evidence"][0]
        rows.append(
            f'<article class="gf-trap">'
            f'  <div class="gf-trap-head"><span class="gf-cid">{esc(t["id"])}</span>{badge}</div>'
            f'  <p class="gf-trap-claim">{esc(t["claim"])}</p>'
            f'  <div class="gf-cite">'
            f'    <span class="gf-cid">{esc(ev["chunk_id"])}</span>'
            f'    <span class="gf-src">{esc(ev["source_file"])} · p.{esc(ev["page"])}</span>'
            f'    <span class="gf-quote">“{esc(_quote(ev, 300))}…”</span></div>'
            f'  <p class="gf-verdict">{verdict}</p>'
            f'</article>'
        )
    return (
        f'<div class="gf-artifact">'
        f'<p class="gf-trap-intro"><strong>{caught} of {total} planted claims were '
        f'struck.</strong> Each was written to be wrong in a specific way, then put '
        f'through the same two checks a generated claim faces.</p>'
        f'{"".join(rows)}</div>'
    )


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
    status = (_subject_changed(topic)
              + f"**{len(kept)} kept, {len(struck)} struck** in {took:.1f}s "
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


# Words too common to be evidence of anything. Deliberately short: the test is
# whether a word appears in the corpus AT ALL, so only words that would be
# present in any English text need excluding.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "is", "are",
    "was", "were", "be", "with", "how", "what", "why", "when", "which", "does",
    "do", "did", "its", "it", "this", "that", "from", "by", "as", "at",
}

_vocab: set | None = None


def corpus_vocabulary() -> set:
    """Every distinct token in the demo corpus. Built once, on first use."""
    global _vocab
    if _vocab is None:
        from retrieve.keyword import _tokenize
        chunks = json.loads(Path(f"chunks/{CORPUS}.json").read_text(encoding="utf-8"))
        _vocab = set()
        for chunk in chunks:
            _vocab.update(_tokenize(chunk["text"]))
    return _vocab


def unknown_terms(topic: str) -> list[str]:
    """
    Words in the topic that appear NOWHERE in the corpus.

    WHY THIS AND NOT A SCORE
    ------------------------
    Retrieval always returns its nearest neighbour, so asking for "HTTP/2 server
    push" returns TCP's PUSH flag and the page answered a question nobody asked.
    The obvious fix is a similarity floor, and it does not work: measured over
    six on-corpus and six off-corpus topics, the two distributions OVERLAP.
    "The Time to Live field" scores 0.301, a real topic covered on three pages,
    while "BGP route reflection" scores 0.444. BM25 overlaps too, because
    off-corpus questions reuse networking words: "React hooks and state
    management" scores 7.54 on `state` and `management` alone.

    Vocabulary membership separates them exactly. Across those twelve topics
    every on-corpus one had ZERO unknown words and every off-corpus one had two
    or three. It is not a threshold read off a distribution, it is a fact about
    the corpus, and it can name the words to the reader instead of asserting a
    verdict about them.
    """
    from retrieve.keyword import _tokenize

    vocab = corpus_vocabulary()
    seen, out = set(), []
    for word in _tokenize(topic):
        if word in _STOPWORDS or len(word) < 2 or word in seen:
            continue
        seen.add(word)
        if word not in vocab:
            out.append(word)
    return out


def _subject_changed(topic: str) -> str:
    """The notice shown when the corpus has never seen part of the question.

    Names the missing words rather than saying "off topic", because the reader
    can check the claim: these words are absent from RFC 768, 791 and 793, and
    the questions below therefore answer something adjacent to what was asked.
    """
    missing = unknown_terms(topic)
    if not missing:
        return ""
    words = ", ".join(f"**{esc(w)}**" for w in missing[:4])
    return (f"⚠️ The corpus contains nothing about {words}. "
            f"Retrieval returned the nearest material it does have, so the questions "
            f"below answer a different question than the one you asked.  \n")


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

GF_CSS = """
/* --- Ink & Paper -----------------------------------------------------------
   A cite-or-strike tool is a printed brief with footnoted evidence, so it is
   set like one. Also: every other demo in a Spaces gallery is dark, and looking
   deliberate is most of the job here. Three roles, three treatments -- the
   question is the heading, the answer sits behind a green rail, the evidence is
   demoted to a footnote block. -------------------------------------------- */
:root {
  --paper:#f7f4ee; --card:#ffffff; --ink:#1f1d1a; --ink-soft:#5d574c;
  --rule:#e5ded1; --rule-soft:#efe9dd;
  --verified:#2f6f5e; --strike:#8b2f28; --cite-bg:#faf7f1; --cid-bg:#ece5d8;
  --cid-ink:#6b4f1d;
}
.gradio-container, .gradio-container .prose {
  background:var(--paper) !important; color:var(--ink) !important;
  max-width:880px !important; margin:0 auto !important;
}
.gradio-container h1 { font-family:Georgia,'Iowan Old Style',serif !important;
  font-weight:700 !important; letter-spacing:-.01em; color:var(--ink) !important; }
footer { display:none !important; }

/* preset buttons: a grid of equal cells. The old row let a two-line label make
   its own button taller than the rest -- the ragged edge was the tell. */
#gf-presets { display:grid !important; grid-template-columns:repeat(4,1fr) !important;
  gap:8px !important; }
#gf-presets button {
  min-height:52px !important; height:100% !important; white-space:normal !important;
  line-height:1.25 !important; font-size:12.5px !important; padding:8px 10px !important;
  background:var(--card) !important; color:var(--ink-soft) !important;
  border:1px solid var(--rule) !important; border-radius:8px !important;
}
#gf-presets button:hover { border-color:var(--ink) !important; color:var(--ink) !important; }

/* Every colour below is !important, and that is load-bearing rather than lazy.
   Gradio re-applies `dark` to <body> AFTER the on-load js runs -- the deployed
   page came back with body.className "theme-loaded dark" -- so its dark palette
   wins any specificity contest these rules could otherwise mount, and the
   question headings, ANSWER labels and chunk-id chips all rendered near-white
   on cream. Winning on !important is deterministic; winning on a class removal
   that Gradio undoes is not. Local looked fine only because of load timing. */
.gf-artifact { font-size:15px; line-height:1.55; }
.gf-item { background:var(--card) !important; border:1px solid var(--rule);
  border-radius:10px; padding:16px 18px; margin:0 0 12px; }
.gf-head { display:flex; align-items:baseline; gap:10px; }
.gf-n { font:700 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--ink) !important; color:var(--paper) !important; padding:4px 7px;
  border-radius:4px; letter-spacing:.06em; flex:none; }
.gf-q { font-family:Georgia,'Iowan Old Style',serif; font-size:17px !important;
  font-weight:700; margin:0 !important; color:var(--ink) !important; line-height:1.35; }
.gf-a { margin:11px 0 0 !important; padding:2px 0 2px 13px;
  border-left:3px solid var(--verified); color:#33302b !important; }
.gf-a-label { display:block; font:700 10px/1 ui-sans-serif,system-ui,sans-serif;
  letter-spacing:.13em; text-transform:uppercase; color:var(--verified) !important;
  margin-bottom:4px; }
.gf-cite { background:var(--cite-bg) !important; border:1px solid var(--rule-soft);
  border-radius:7px; padding:9px 11px; margin-top:10px; font-size:13px;
  color:var(--ink-soft) !important; }
.gf-cid { font:600 11.5px ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--cid-bg) !important; color:var(--cid-ink) !important; padding:2px 6px;
  border-radius:4px; margin-right:7px; }
/* Colour stated on the spans, not inherited from .gf-cite: Gradio has its own
   rule for bare <span> and it wins over inheritance, which left the source line
   and the quoted evidence near-white on cream -- the one part of the page a
   reader is invited to check. */
.gf-src { font-size:12px; color:var(--ink-soft) !important; }
.gf-quote { display:block; margin-top:6px; font-style:italic;
  color:#4a453c !important; }
.gf-cite-missing { border-color:var(--strike); color:var(--strike); }

.gf-struck { margin-top:16px; padding:14px 16px; background:#fdf3f2 !important;
  border:1px solid #f0d5d2; border-radius:10px; }
.gf-struck h4 { margin:0 0 8px !important; color:var(--strike) !important;
  font-size:13px !important; letter-spacing:.04em; text-transform:uppercase; }
.gf-struck-claim { margin:0 !important; text-decoration:line-through;
  color:#6f6a61 !important; }
.gf-struck-why { margin:3px 0 10px !important; font-size:13px;
  color:var(--strike) !important; }

.gf-trap-intro { margin:0 0 12px !important; color:var(--ink-soft) !important; }
.gf-trap { background:var(--card) !important; border:1px solid var(--rule);
  border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.gf-trap-head { font-size:13px; color:var(--ink-soft) !important; margin-bottom:7px; }
.gf-trap-claim { margin:0 !important; font-family:Georgia,serif; font-size:15px;
  color:var(--ink) !important; }
.gf-verdict { margin:9px 0 0 !important; font-size:13px; color:var(--ink-soft) !important;
  padding-left:11px; border-left:3px solid var(--rule); }
.gf-badge { font:700 10px/1 ui-sans-serif,system-ui,sans-serif; letter-spacing:.1em;
  text-transform:uppercase; padding:3px 6px; border-radius:4px; margin-right:6px; }
.gf-ok  { background:#e3efe9 !important; color:var(--verified) !important; }
.gf-bad { background:#fbe6e4 !important; color:var(--strike) !important; }

/* Gradio's own chrome, brought into the palette. Written against body.dark too,
   so a viewer whose OS is dark still gets a readable page even if the class
   removal above is ever defeated -- the failure mode otherwise is invisible
   text, which looks like a broken deploy rather than a theme problem. */
body, body.dark, .gradio-container, body.dark .gradio-container {
  background:var(--paper) !important; color:var(--ink) !important;
}
body.dark .prose, body.dark .prose p, body.dark .prose li,
.gradio-container .prose p, .gradio-container .prose li,
.gradio-container label, body.dark label { color:var(--ink) !important; }
/* <strong> carries its own colour in Gradio's dark palette, so the emphasised
   half of every sentence went white-on-paper -- the intro read with holes in
   it, which is worse than being wholly unstyled. */
.gradio-container .prose strong, .gradio-container .prose b,
body.dark .prose strong, body.dark .prose b,
.gradio-container strong, body.dark strong { color:var(--ink) !important; }
.gradio-container .prose em, .gradio-container .prose i,
body.dark .prose em, body.dark .prose i,
.gradio-container em, body.dark em { color:var(--ink-soft) !important; }
.gradio-container .prose a, body.dark .prose a { color:var(--strike) !important; }

/* the primary button: ink, not the theme's violet */
button.primary, body.dark button.primary {
  background:var(--ink) !important; color:var(--paper) !important;
  border:1px solid var(--ink) !important; font-weight:600 !important;
}
button.primary:hover { background:#3a352e !important; }

/* the topic field. Gradio wraps every input in .block inside .form, both of
   which carry their own dark surface colour -- so the field sat in a navy slab
   on a paper page. Flattened to the page rather than restyled: the input's own
   border is the only edge this needs. */
.gradio-container .block, body.dark .block,
.gradio-container .form, body.dark .form {
  background:transparent !important; border:none !important; box-shadow:none !important;
}
#gf-topic textarea, #gf-topic input, body.dark #gf-topic textarea {
  background:var(--card) !important; color:var(--ink) !important;
  border:1px solid var(--rule) !important; font-size:14px !important;
}
#gf-topic textarea::placeholder { color:#a49c8e !important; }
#gf-topic label span, #gf-topic .block-label {
  background:transparent !important; color:var(--ink-soft) !important; }

/* accordion header */
.gradio-container .label-wrap, body.dark .label-wrap {
  background:var(--card) !important; color:var(--ink) !important;
  border:1px solid var(--rule) !important; border-radius:8px !important; }
"""

# Gradio follows the viewer's OS dark-mode preference by putting `dark` on
# <body>, and every one of its own text colours keys off that. Left alone it
# paints light text onto this light theme -- the first attempt rendered the
# intro and every source quote invisible.
#
# It marks the mode with a class, so the class is what gets removed. An earlier
# attempt set ?__theme=light and reloaded; it never fired, and a redirect on
# load is a worse mechanism anyway -- it costs a round trip and flashes.
FORCE_LIGHT = """
() => {
  document.body.classList.remove('dark');
  document.documentElement.classList.remove('dark');
}
"""

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

with gr.Blocks(title="Grounded Forge") as demo:   # css/js/theme go to launch() in Gradio 6
    last_run = gr.State(0.0)

    gr.Markdown(INTRO)

    with gr.Row():
        topic_box = gr.Textbox(
            elem_id="gf-topic",
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

    preset_buttons = []
    with gr.Row(elem_id="gf-presets"):
        for preset in PRESETS:
            preset_buttons.append((gr.Button(preset, size="sm"), preset))

    status = gr.Markdown("Showing a saved run. Press **Generate** for a live one.")
    quiz_out = gr.HTML(render_quiz(CACHED))

    with gr.Accordion("Does the Critic actually catch anything? — planted claims",
                      open=False):
        gr.HTML(render_traps())

    gr.Markdown(FOOTER)

    # Handlers last, once every component they write to exists.
    go.click(generate, inputs=[topic_box, last_run],
             outputs=[status, quiz_out, last_run])
    topic_box.submit(generate, inputs=[topic_box, last_run],
                     outputs=[status, quiz_out, last_run])

    for btn, preset in preset_buttons:
        # Fill the box AND run, chained with .then(). One click, not two, for
        # two reasons.
        #
        # A preset that only fills a textbox asks a visitor to find and press a
        # second button before anything happens, which is the whole reason
        # example buttons exist.
        #
        # And filling the box is itself a server round-trip in Gradio, so
        # "click preset, then Generate" is a race: pressing Generate before the
        # value lands runs with an EMPTY topic. Seen on the live Space -- the
        # status came back "Type a topic, or pick one of the examples" with the
        # box visibly full. .then() sequences them, so it cannot happen.
        btn.click(lambda p=preset: p, outputs=topic_box).then(
            generate, inputs=[topic_box, last_run],
            outputs=[status, quiz_out, last_run])


if __name__ == "__main__":
    # HF Spaces sets PORT; locally default to Gradio's usual 7860.
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)),
                theme=gr.themes.Soft(), css=GF_CSS, js=FORCE_LIGHT)
