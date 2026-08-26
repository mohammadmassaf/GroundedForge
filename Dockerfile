# syntax=docker/dockerfile:1
#
# D-M1: builds and runs the demo corpus offline. No mounts yet -- chroma_db/
# and chunks/ live in the container's writable layer and are lost when the
# container is removed. D-M2 adds named volumes; D-M3 mounts a real corpus.
# See grounded-forge-docker-plan.md for the decisions (D1-D13) behind every
# choice below.

FROM python:3.13-slim

# libgomp1: torch's CPU kernels use OpenMP and slim Debian doesn't ship it.
# ca-certificates: HTTPS for the model bake below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf_home

COPY requirements.txt .

# Torch on its own index line (D7): --extra-index-url would let pip fall back
# to PyPI's Linux default, which is the CUDA build -- multi-GB, no error, easy
# to miss. --index-url makes the CPU wheel the only option pip can resolve.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Bake both MiniLM models at a pinned revision (D2/D3) -- before COPY . . so
# editing source never invalidates this layer and re-triggers a ~180MB
# re-download on this connection.
COPY retrieve/__init__.py retrieve/model_pins.py retrieve/
COPY docker/bake_models.py .
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p "$HF_HOME" && python bake_models.py && rm bake_models.py

# git: the git adapter shells out to `git log`, and slim has no git binary.
# Deliberately NOT with the other system packages at the top -- that layer sits
# above torch and the model bake, so touching it would re-download ~1.1GB. Here
# it invalidates nothing (C1: bandwidth is the binding constraint).
#
# safe.directory '*': git refuses to run in a repo owned by another user, since
# a foreign .git/config can name commands git will execute. Mounted host repos
# arrive with an owner uid that is not appuser, so every `git log` fails with
# "dubious ownership". Blanket-allowing is right HERE specifically: this image
# exists to read repos it does not own, at paths the invoker chooses, with one
# user inside and the mounts read-only. Listing paths instead would mean editing
# the guard on every use, which is friction, not safety.
RUN apt-get update && apt-get install git -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

COPY . .
# The real corpus.yaml is gitignored (D0) and never enters the build context;
# ship the example config so `demo` and `default` work out of the box.
COPY corpus.example.yaml ./corpus.yaml

RUN mkdir -p chunks chroma_db traces out

RUN chown -R appuser:appuser /app "$HF_HOME"
USER appuser

ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
