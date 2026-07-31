"""
Query de-framing - strip interview scaffolding off a question before retrieval.

An interview prompt is mostly *frame*, not content:

    "Tell me about a time you had to improve retrieval quality under a deadline."
     |---------------- frame ----------------|  |------- content -------|

The frame describes the interview. The content describes your corpus. Only the
second half should steer retrieval.

WHY THIS ISN'T FOR BM25. BM25 already weights every query term by its IDF, so
frame words self-cancel at scoring time - stripping them by hand buys the
lexical retriever nothing. The de-framer exists for the OTHER two consumers of
the query string:

    embedding (retrieve/query.py)  - pools every token into ONE vector, with no
                                     notion that a word is uninformative here.
                                     Seven frame words genuinely rotate the
                                     vector toward interview-speak.
    cross-encoder (retrieve/rerank.py) - attends over the literal query text,
                                     and runs LAST, so it decides the final k.

So: borrow the lexical retriever's corpus statistics to de-noise the query for
the dense retriever, which has no statistics of its own.

WHY IDF AND NOT A STOPWORD LIST. A hardcoded list of opener phrases is a
blacklist you author - it fails silently on the one phrasing you didn't
imagine. IDF is a whitelist the corpus derives: it never enumerates what to
remove, it asks which tokens actually discriminate between chunks *in this
corpus*. Same question against a different corpus keeps a different set. That
adaptivity is the whole point, which is why `corpus` is a parameter and not a
default you can ignore.

Tools: `_get_index(corpus)` returns (BM25Okapi, chunks); the index carries an
`.idf` dict mapping token -> float, built from document frequencies when the
index was constructed. High = rare = discriminating.
"""
from retrieve.keyword import _get_index, _tokenize






def deframe(question: str, corpus: str = "job") -> str:
    """
    Return `question` with its low-information tokens dropped, judged against
    the IDF table of `corpus`.

    TODO(you) - V2-M4:
      1. Tokenize the question the same way the corpus was tokenized. Reuse
         _tokenize; do not roll a second tokenizer, or the two alphabets drift.
      2. Get this corpus's IDF table off the index.
      3. Score each query token by how much it discriminates, handling tokens
         the table has never seen (DECISION 2).
      4. Keep the informative ones (DECISION 4), and make sure a question that
         survives nothing still produces a usable query rather than "".
      5. Reassemble into a string.

         Order matters here and the reason is worth knowing: BM25 is a bag of
         words and cannot see order at all, but the embedding and the
         cross-encoder are sequence models that read position. Reordering by
         IDF would hand two of your three consumers word salad.
    """
    tokenized = _tokenize(question)
    index , _ = _get_index(corpus)
    result = []
    avrg_idf = index.average_idf
    for token in tokenized:
        score = index.idf.get(token)
        ## keep rule 
        if token not in index.idf:
            result.append(token)      # unseen: no corpus evidence, keep on benefit of the doubt
            continue
        if score >= avrg_idf:
            result.append(token)
    if result:
        return " ".join(result)
    return question
        
