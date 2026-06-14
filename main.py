import argparse
import sys


def cmd_ingest(args):
    print("ingest: not implemented yet (coming in M1)")


def cmd_query(args):
    print("query: not implemented yet (coming in M2)")


def cmd_make_quiz(args):
    print("make-quiz: not implemented yet (coming in M3)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="grounded-forge",
        description="Cite-or-strike grounded artifact generator.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_ingest = sub.add_parser("ingest", help="Ingest documents from data/ into the vector store")
    p_ingest.add_argument("--corpus", default="default", help="Corpus name (default: default)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Retrieve top-k chunks for a query")
    p_query.add_argument("query_text", help="Query string")
    p_query.add_argument("--corpus", default="default")
    p_query.add_argument("-k", type=int, default=5, help="Number of chunks to return")
    p_query.set_defaults(func=cmd_query)

    p_quiz = sub.add_parser("make-quiz", help="Generate a cited quiz from retrieved chunks")
    p_quiz.add_argument("topic", help="Topic or chapter to quiz on")
    p_quiz.add_argument("--corpus", default="default")
    p_quiz.add_argument("-n", type=int, default=5, help="Number of quiz items")
    p_quiz.set_defaults(func=cmd_make_quiz)

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
