import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def cmd_ingest(args):
    from ingest.pipeline import run
    run(corpus=args.corpus)


def cmd_build_index(args):
    from retrieve.store import build
    build(corpus=args.corpus)


def cmd_query(args):
    from retrieve.query import search
    results = search(args.query_text, corpus=args.corpus, k=args.k)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r['score']:.3f} | {r['source_file']} p.{r['page']} | {r['chunk_id']}")
        print(r["text"][:300])


def cmd_make_quiz(args):
    from pathlib import Path

    from retrieve.query import search
    from critic.loop import run_loop
    from generate.renderer import render

    print(f"Retrieving chunks for: {args.topic}")
    chunks = search(args.topic, corpus=args.corpus, k=args.k)
    print(f"Generating {args.n} quiz items from {len(chunks)} chunks (cite-or-strike)...")
    kept, struck = run_loop(args.topic, chunks, n=args.n)
    print(f"Critic: {len(kept)} kept, {len(struck)} struck")
    markdown = render(kept, chunks, args.topic, struck=struck)

    out = Path(args.out) if args.out else Path(f"quiz_{args.corpus}.md")
    out.write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}")
    print(f"\nSaved to {out}")

def cmd_make_guide(args):
    from pathlib import Path

    from retrieve.hybrid import hybrid_search
    from critic.loop import run_guide_loop
    from generate.renderer import render_guide

    print(f"Retrieving chunks for: {args.topic}")
    chunks = hybrid_search(args.topic, corpus=args.corpus, k=args.k , use_rerank = True)
    print(f"Generating a guide from  {len(chunks)} chunks (cite-or-strike)...")
    kept, struck = run_guide_loop(args.topic, chunks)
    print(f"Critic: {len(kept)} kept, {len(struck)} struck")
    markdown = render_guide(kept, chunks, args.topic, struck=struck)

    out = Path(args.out) if args.out else Path(f"guide_{args.corpus}.md")
    out.write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}")
    print(f"\nSaved to {out}")


def cmd_make_bullets(args):
    from pathlib import Path

    from retrieve.hybrid import hybrid_search
    from critic.loop import run_bullets_loop
    from generate.renderer import render_bullets

    where = {}
    if args.source_type:
        where["source_type"] = args.source_type
    if args.repo:
        where["repo"] = args.repo
    # Chroma wants a single condition or an explicit $and for several
    if len(where) > 1:
        where = {"$and": [{k: v} for k, v in where.items()]}
    where = where or None
    print(f"Retrieving chunks for: {args.topic}" + (f" [filter: {where}]" if where else ""))
    chunks = hybrid_search(args.topic, corpus=args.corpus, k=args.k, use_rerank=True, where=where)
    print(f"Generating {args.n} bullets from {len(chunks)} chunks (cite-or-strike)...")
    kept, struck, gap = run_bullets_loop(args.topic, chunks, n=args.n)
    if gap:
        print(f"GAP: {gap}")
    else:
        print(f"Critic: {len(kept)} kept, {len(struck)} struck")
    markdown = render_bullets(kept, chunks, args.topic, struck=struck, gap=gap)

    out = Path(args.out) if args.out else Path(f"bullets_{args.corpus}.md")
    out.write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}")
    print(f"\nSaved to {out}")


def cmd_make_star(args):
    from pathlib import Path

    from retrieve.star_evidence import gather_evidence
    from critic.loop import run_star_loop
    from generate.renderer import render_star

    print(f"Gathering per-section evidence for: {args.question}"
          + (f" [repo: {args.repo}]" if args.repo else ""))
    pools = gather_evidence(args.question, corpus=args.corpus, k=args.k, repo=args.repo)
    for name, pool in pools.items():
        print(f"  {name:10} {len(pool)} chunks")
    print("Generating STAR answer (cite-or-strike)...")
    answer, flagged = run_star_loop(args.question, pools)
    print(f"Critic: {len(flagged)} section(s) flagged")
    markdown = render_star(answer, pools, flagged=flagged)

    out = Path(args.out) if args.out else Path(f"star_{args.corpus}.md")
    out.write_text(markdown, encoding="utf-8")
    print(f"\n{markdown}")
    print(f"\nSaved to {out}")


def cmd_eval(args):
    from eval.run_eval import (load_eval_set, load_traps, retrieval_eval,
                               grounding_eval, trap_eval, report, GEN_K)

    gen_k = args.gen_k if args.gen_k is not None else GEN_K
    items = load_eval_set(args.corpus)
    traps = load_traps(args.corpus)
    # job mode generates bullets; quiz items over a commit history are meaningless
    generator = "bullets" if args.corpus == "job" else "quiz"
    print(f"Eval set: {len(items)} questions, {len(traps)} traps ({generator})")

    retrieval = retrieval_eval(items, corpus=args.corpus, mode=args.retrieval,
                               gen_k=gen_k)
    # A full 15-question run costs ~90-96k tokens and the free tier caps at 100k
    # per rolling day, so the set has to be evaluable in windows. --offset picks
    # where the window starts; without it every run re-measures the same front
    # slice and the tail is never evaluated at all.
    # `is not None`, not truthiness: --limit 0 means "no LLM half at all", and
    # `if args.limit` read 0 as "unset" and ran the whole set instead.
    start = args.offset
    window = items[start:start + args.limit] if args.limit is not None else items[start:]
    if window:
        print(f"Grounding on questions {start + 1}-{start + len(window)} of {len(items)}")
    grounding = grounding_eval(window,
                               corpus=args.corpus, generator=generator,
                               mode=args.retrieval, gen_k=gen_k)
    trap_results = None
    if traps and not args.no_traps:
        try:
            trap_results = trap_eval(traps, corpus=args.corpus)
        except Exception as e:
            # Deliberately broad. The report IS the product of the run, and the
            # grounding half above may have cost thousands of tokens by now;
            # losing all of it to a failure in the adversarial half is the worse
            # outcome by far. Report what we have and say what broke.
            print(f"  trap eval failed ({type(e).__name__}: {e}) - reporting without traps")
    print(report(retrieval, grounding, trap_results))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="grounded-forge",
        description="Cite-or-strike grounded artifact generator.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_ingest = sub.add_parser("ingest", help="Ingest a corpus (sources defined in corpus.yaml)")
    p_ingest.add_argument("--corpus", default="default", help="Corpus name from corpus.yaml (default: default)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_build = sub.add_parser("build-index", help="Embed chunks and store in ChromaDB")
    p_build.add_argument("--corpus", default="default")
    p_build.set_defaults(func=cmd_build_index)

    p_query = sub.add_parser("query", help="Retrieve top-k chunks for a query")
    p_query.add_argument("query_text", help="Query string")
    p_query.add_argument("--corpus", default="default")
    p_query.add_argument("-k", type=int, default=5, help="Number of chunks to return")
    p_query.set_defaults(func=cmd_query)

    p_quiz = sub.add_parser("make-quiz", help="Generate a cited quiz from retrieved chunks")
    p_quiz.add_argument("topic", help="Topic or chapter to quiz on")
    p_quiz.add_argument("--corpus", default="default")
    p_quiz.add_argument("-n", type=int, default=5, help="Number of quiz items")
    p_quiz.add_argument("-k", type=int, default=8, help="Number of chunks to retrieve as context")
    p_quiz.add_argument("--out", default=None, help="Output markdown file (default: quiz_<corpus>.md)")
    p_quiz.set_defaults(func=cmd_make_quiz)

    p_quiz = sub.add_parser("make-guide", help="Generate a cited guide from retrieved chunks")
    p_quiz.add_argument("topic", help="Topic or chapter to guide on")
    p_quiz.add_argument("--corpus", default="default")
    p_quiz.add_argument("-k", type=int, default=8, help="Number of chunks to retrieve as context")
    p_quiz.add_argument("--out", default=None, help="Output markdown file (default: guide_<corpus>.md)")
    p_quiz.set_defaults(func=cmd_make_guide)

    p_bullets = sub.add_parser("make-bullets", help="Generate cited CV bullets from your work corpus")
    p_bullets.add_argument("topic", help="Project or skill area to write bullets about")
    p_bullets.add_argument("--corpus", default="job")
    p_bullets.add_argument("-n", type=int, default=5, help="Number of bullets")
    p_bullets.add_argument("-k", type=int, default=8, help="Number of chunks to retrieve as context")
    p_bullets.add_argument("--source-type", choices=["git", "docs", "vault", "files"], default=None,
                           help="Restrict evidence to one source type")
    p_bullets.add_argument("--repo", default=None,
                           help="Restrict evidence to one repo (e.g. mealwise)")
    p_bullets.add_argument("--out", default=None, help="Output markdown file (default: bullets_<corpus>.md)")
    p_bullets.set_defaults(func=cmd_make_bullets)

    p_star = sub.add_parser("make-star", help="Generate a cited STAR interview answer")
    p_star.add_argument("question", help="The interview question")
    p_star.add_argument("--corpus", default="job")
    p_star.add_argument("-k", type=int, default=6, help="Chunks retrieved per STAR section")
    p_star.add_argument("--repo", default=None,
                        help="Restrict evidence to one repo (e.g. mealwise)")
    p_star.add_argument("--out", default=None, help="Output markdown file (default: star_<corpus>.md)")
    p_star.set_defaults(func=cmd_make_star)

    p_eval = sub.add_parser("eval", help="Run retrieval (recall@k) + grounding evals")
    p_eval.add_argument("--corpus", default="default")
    p_eval.add_argument("--limit", type=int, default=None,
                        help="Grounding eval on N questions only (LLM cost control)")
    p_eval.add_argument("--offset", type=int, default=0,
                        help="Skip the first N questions. With --limit, evaluates a window, "
                             "so a set too big for one day's tokens can be run in halves and "
                             "the claim counts added together.")
    p_eval.add_argument("--retrieval" , choices=["vector", "hybrid", "rerank"] , default="vector")
    p_eval.add_argument("--no-traps", action="store_true",
                        help="Skip the adversarial traps (they cost one LLM call each)")
    # default resolved in cmd_eval, not here: importing GEN_K would pull
    # eval.run_eval (and langchain + torch behind it) into every CLI command.
    p_eval.add_argument("--gen-k", type=int, default=None,
                        help="Chunks the Generator reads per question (default 8). "
                             "recall@<gen-k> is always added to the report, since that "
                             "row is the ceiling on grounding.")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
