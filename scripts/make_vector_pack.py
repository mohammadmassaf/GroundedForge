"""
Embed a corpus once and commit the VECTORS, not the database.

WHY NOT JUST COMMIT THE CHROMA STORE
------------------------------------
The Gradio Space has no build step, so the demo index has to ship with the
repo or be rebuilt on every cold boot (~34s for 526 chunks). Committing
chroma_demo/ solved that and introduced a worse problem: Chroma rewrites its
HNSW files and sqlite pages on every READ, so an ordinary `query` left 4.7 MB
of binary diff in `git status`. On a repo people clone, committing that churn
repeatedly bloats history for no gain.

The vectors themselves never change. Same chunks + same pinned model revision
= same floats, so they are exactly the kind of thing that belongs in git: one
immutable file, no churn, and a diff that is either "nothing" or "the corpus
was re-ingested".

WHAT IT COSTS
-------------
Rebuilding the store from the pack at boot is the cheap third of the work --
embedding is 15.3s of the 34s, and this skips it. What remains is Chroma
writing the index, ~2s, and it needs no model at all.

Run: python scripts/make_vector_pack.py [corpus]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from langchain_huggingface import HuggingFaceEmbeddings
from retrieve.model_pins import EMBEDDING_MODEL, EMBEDDING_REVISION, EMBEDDING_KWARGS
from retrieve.paths import vector_pack

corpus = sys.argv[1] if len(sys.argv) > 1 else "demo"

chunks = json.loads(Path(f"chunks/{corpus}.json").read_text(encoding="utf-8"))
print(f"{len(chunks)} chunks from chunks/{corpus}.json")

print(f"loading {EMBEDDING_MODEL} @ {EMBEDDING_REVISION[:7]}...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=EMBEDDING_KWARGS)

vectors = np.asarray(embeddings.embed_documents([c["text"] for c in chunks]),
                     dtype=np.float32)
ids = np.asarray([c["chunk_id"] for c in chunks])

out = Path(vector_pack(corpus))
out.parent.mkdir(parents=True, exist_ok=True)
# The model revision travels WITH the vectors. Rebuilding a store from floats
# produced by a different revision than the one that will embed the queries is
# silent nonsense -- every score would be wrong and nothing would raise.
np.savez_compressed(out, ids=ids, vectors=vectors,
                    model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION)

print(f"wrote {out}  {vectors.shape[0]}x{vectors.shape[1]} float32  "
      f"({out.stat().st_size / 1e6:.2f} MB)")
