"""
<docstring: these four constants are the single source of truth for which
weights the pipeline loads — the app at runtime and docker/bake_models.py at
build time read the same values, so the image cannot bake one revision and
load another. Point at D3 for why pinned.>
"""

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"     # sha, verified via https://huggingface.co/api/models/<repo> on <date>
EMBEDDING_KWARGS = {"revision" : EMBEDDING_REVISION}

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
