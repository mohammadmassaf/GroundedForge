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


MODEL_NAME = "all-MiniLM-L6-v2"

_embeddings = None



def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embeddings





def search(query_text: str, corpus: str = "default", k: int = 5) -> list[dict]:
    """Top-k retrieval via LangChain Chroma. Returns result dicts (see module docstring)."""
    store = Chroma(collection_name=corpus, persist_directory="chroma_db", embedding_function=_get_embeddings())
    results = store.similarity_search_with_score(query_text, k =k)
    final = []
    for doc , dist in results:
        final.append({
            "chunk_id" : doc.metadata["chunk_id"],
            "source_file" : doc.metadata["source_file"],
            "page" : doc.metadata["page"],
            "text": doc.page_content,
            "score" : 1-dist
        })
    return final
