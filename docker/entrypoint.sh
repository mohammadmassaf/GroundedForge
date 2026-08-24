#!/bin/sh
# ENTRYPOINT is the CLI (see grounded-forge-docker-plan.md D9). With no
# arguments, run the offline-safe demo path end to end: ingest the baked
# RFC corpus, build its index, then a retrieval query -- no GROQ_API_KEY
# needed, so a bare `docker run` works for anyone who pulls the image.
# Any other arguments pass straight through to the CLI unchanged, so
# `docker run <image> make-bullets "..." --corpus job` (once job mode is
# mounted in, per D-M3) still works through the same image.
set -e

if [ "$#" -eq 0 ]; then
    python main.py ingest --corpus demo
    python main.py build-index --corpus demo
    exec python main.py query "TCP connection establishment" --corpus demo
fi

exec python main.py "$@"
