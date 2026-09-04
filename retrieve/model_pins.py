"""
<docstring: these four constants are the single source of truth for which
weights the pipeline loads — the app at runtime and docker/bake_models.py at
build time read the same values, so the image cannot bake one revision and
load another. Point at D3 for why pinned.>
"""

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"     # sha, verified via https://huggingface.co/api/models/<repo> on <date>

# device is pinned as well as the revision. sentence-transformers picks CUDA
# whenever it sees a GPU, and on a ZeroGPU Space touching CUDA outside a
# @spaces.GPU function raises -- the first live run there died on a bare
# RuntimeError. This pipeline is CPU-only by design (526 chunks, MiniLM), and
# index/demo_vectors.npz was produced on CPU, so asking for CPU explicitly also
# keeps query vectors on the same device as the stored ones.
EMBEDDING_KWARGS = {"revision": EMBEDDING_REVISION, "device": "cpu"}

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
