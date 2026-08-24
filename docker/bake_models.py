"""
Pre-downloads the embedding and cross-encoder models into HF_HOME at build
time, pinned to a fixed revision.

Why pin: an unpinned rebuild months from now could silently fetch different
weights with nothing in the diff to show it. Two constants buy reproducibility
and make HF_HUB_OFFLINE=1 work at runtime with zero network access.
See grounded-forge-docker-plan.md D2/D3.

Revisions confirmed via https://huggingface.co/api/models/<repo> on 2026-08-23.
"""
from sentence_transformers import CrossEncoder, SentenceTransformer
from retrieve.model_pins import EMBEDDING_REVISION , EMBEDDING_MODEL , RERANK_MODEL , RERANK_REVISION



if __name__ == "__main__":
    SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
    CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION)
