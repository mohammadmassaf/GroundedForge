"""
Retrieves the top-k most relevant chunks for a query string.

Returns a list of result dicts, ordered by relevance (best first):
{
    "chunk_id":    "I2208-Part-1_p7_c0",
    "source_file": "I2208-2024-2025-Part-1.pdf",
    "page":        7,
    "score":       0.87,   # cosine similarity, higher = more relevant
    "text":        "..."
}
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from retrieve.model_pins import EMBEDDING_MODEL , EMBEDDING_KWARGS
from retrieve.paths import chroma_dir





_embeddings = None



def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL , model_kwargs= EMBEDDING_KWARGS)
    return _embeddings





def search(query_text: str, corpus: str = "default", k: int = 5,
           where: dict | None = None) -> list[dict]:
    """
    Top-k retrieval via LangChain Chroma. Returns result dicts (see module docstring).

    `where` is an optional ChromaDB metadata filter, e.g. {"source_type": "git"}:
    results are still ranked by embedding similarity, but only chunks matching
    the filter are considered. None = search the whole corpus (v1 behavior).
    """
    store = Chroma(collection_name=corpus, persist_directory=chroma_dir(corpus), embedding_function=_get_embeddings())
    results = store.similarity_search_with_score(query_text, k =k , filter = where)
    final = []
    for doc , dist in results:
        final.append({
            "chunk_id" : doc.metadata["chunk_id"],
            "source_file" : doc.metadata["source_file"],
            "page" : doc.metadata["page"],
            "source_type" : doc.metadata.get("source_type"),
            "text": doc.page_content,
            "score" : 1-dist,
            "vector_score": 1 - dist

        })
    return final
