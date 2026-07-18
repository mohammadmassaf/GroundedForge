# LangChain refactor: what it bought, what it hid

In M9 the retrieval layer (embedding + vector store) was ported from a
hand-rolled implementation to LangChain, while the multi-agent loop was
deliberately left hand-rolled. This note records what changed, what the
framework abstracted, what it hid, and when reaching for it is the right
call. The refactor was behavior-preserving: recall@k was identical before
and after (90 / 95 / 100), so the two versions are directly comparable.

## The scope decision: framework for plumbing, hand-rolled for the point

The guiding rule was to use the framework only where the work is commodity,
and keep it out of the part that is the project's actual value.

- **Retrieval moved to LangChain.** Vector-store access is standardized
  work: embed, store, query by nearest neighbor. Many libraries do it the
  same way, so a shared interface is a real win.
- **The agent loop stayed hand-rolled.** Generator, Critic, and Refiner are
  where the grounding guarantee lives. Wrapping that in a framework would
  hide the exact reasoning that makes the project defensible and that was
  the whole point of building it.

## What the framework abstracted (did for us)

| Concern | Hand-rolled | LangChain |
|---|---|---|
| Embedding | call `SentenceTransformer.encode()` directly | `HuggingFaceEmbeddings`, invoked internally by the store |
| Build index | `chromadb.PersistentClient` -> `get_or_create_collection` -> `collection.add(ids, embeddings, documents, metadatas)` | `Chroma.from_documents(docs, embedding, ...)`, one call |
| Query | `encode(query)` -> `get_collection` -> `collection.query(...)` -> unpack nested parallel arrays | `store.similarity_search_with_score(q, k)` -> loop `(Document, score)` tuples |

Net effect: the embedding step disappears from view, connection and
collection management collapse into one `Chroma` object, and the awkward
`results["ids"][0]` / `results["documents"][0]` parallel-array unpacking
becomes clean tuples. The retrieval code shrank by roughly half.

## What the framework hid (the traps)

These are the reasons "it just works" is not the same as "I understand it."
Each one could silently change behavior if missed.

1. **Distance metric default.** LangChain's Chroma wrapper defaults to L2
   (Euclidean), not cosine. Matching the original store required passing
   `collection_metadata={"hnsw:space": "cosine"}` explicitly. A wrong
   default here changes ranking with no error.
2. **Misleading score semantics.** `similarity_search_with_score` returns
   the raw distance (lower is closer), even though the name says "score."
   The original `score = 1 - distance` conversion had to be kept to produce
   a similarity. The field name does not match the value.
3. **The row id is not surfaced.** A LangChain `Document` exposes
   `page_content` and `metadata`, but not Chroma's row id. The `chunk_id`
   had to be stored inside `metadata` so the query side could read it back,
   since every citation depends on it.
4. **API inconsistency.** `Chroma.from_documents(embedding=...)` versus the
   `Chroma(embedding_function=...)` constructor: the same object passed
   under two different keyword names.

## When to use a framework here, and when not to

Reach for the framework when the layer is commodity and you benefit from a
shared interface: swapping Chroma for FAISS or Pinecone becomes a one-line
change, and the retriever plugs into larger pipelines. It removes real
boilerplate, and once the traps above are known, it hides nothing that
matters.

Keep it hand-rolled when the logic is the product. The agent loop's control
flow (regenerate, strike, terminate) is the differentiator and the thing to
be able to explain line by line. A framework abstraction there would trade
understanding for convenience that is not needed.

The short version: LangChain earned its place on the retrieval layer because
that layer is plumbing. It stays off the agent loop because that logic is
the product.
