import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def cmd_ingest(args):
    from ingest.pipeline import run
    run(data_dir="data", corpus=args.corpus)


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


def cmd_eval(args):
    from eval.run_eval import load_eval_set, retrieval_eval, grounding_eval, report

    items = load_eval_set()
    print(f"Eval set: {len(items)} questions")
    retrieval = retrieval_eval(items, corpus=args.corpus)
    grounding = grounding_eval(items[:args.limit] if args.limit else items, corpus=args.corpus)
    print(report(retrieval, grounding))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="grounded-forge",
        description="Cite-or-strike grounded artifact generator.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_ingest = sub.add_parser("ingest", help="Ingest documents from data/ into the vector store")
    p_ingest.add_argument("--corpus", default="default", help="Corpus name (default: default)")
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

    p_eval = sub.add_parser("eval", help="Run retrieval (recall@k) + grounding evals")
    p_eval.add_argument("--corpus", default="default")
    p_eval.add_argument("--limit", type=int, default=None,
                        help="Grounding eval on first N questions only (LLM cost control)")
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
