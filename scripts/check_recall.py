"""
CI gate: fail the build if retrieval recall drops.

WHY THIS AND NOT THE GROUNDING SCORE
------------------------------------
The eval has two halves and only one of them can be a build gate.

Recall@k is local, free and deterministic: the same corpus, the same pinned
embedding revision and the same questions produce the same number every run
(verified by running it twice before this gate was written). A drop means
something in the repo changed for the worse -- chunking, the model pin, the
vector pack, the eval set.

Grounding % is none of those things. It costs Groq tokens, it moves with the
model, and Groq has already retired one model mid-project. Gating it would make
the build fail for reasons that are not in the repository, which trains everyone
to ignore a red build. It stays a thing you run deliberately and publish with a
model and a date beside it.

WHY THE FLOOR IS WHERE IT IS
----------------------------
83.3% is 5 of 6 on the demo set, measured 2026-09-05 on `vector` mode, which is
what the demo actually runs. The sixth is a known vocabulary-gap miss written up
in eval/notes.md Finding 11: MiniLM does not bridge "router" to "gateway" or
"packet" to "datagram", and the RFCs are 1981 English.

The floor is the CURRENT value, not a comfortable margin below it. A gate set
below what the system does today only fires after a regression has already been
released twice. If a change genuinely improves recall, raise the floor in the
same commit -- that is the point of a ratchet.

Run: python scripts/check_recall.py
"""
import sys

sys.path.insert(0, ".")

from eval.run_eval import load_eval_set, retrieval_eval          # noqa: E402
from retrieve.store import ensure_store                           # noqa: E402

CORPUS = "demo"           # the only corpus a clone or a CI runner has
MODE = "vector"           # what cmd_make_quiz runs; gate what ships
GATED_K = 8               # the k that feeds generation
FLOOR = 0.833             # 5 of 6, measured 2026-09-05


def main() -> int:
    # The store is gitignored and rebuilt from the committed vector pack. On a
    # fresh runner it does not exist yet, and this costs ~3s with no model load.
    ensure_store(CORPUS)

    items = load_eval_set(CORPUS)
    result = retrieval_eval(items, corpus=CORPUS, mode=MODE)

    print(f"retrieval gate: {CORPUS} corpus, {MODE} mode, {len(items)} questions")
    for k in result["ks"]:
        value = result[f"recall@{k}"]
        marker = "  <- gated" if k == GATED_K else ""
        print(f"  recall@{k:<3}: {value:6.1%}{marker}")

    for miss in result["misses"]:
        print(f"  miss: {miss}")

    actual = result[f"recall@{GATED_K}"]
    if actual + 1e-9 < FLOOR:
        print(f"\nFAIL: recall@{GATED_K} is {actual:.1%}, below the {FLOOR:.1%} floor.")
        print("Retrieval got worse. Find out why before raising or lowering this.")
        return 1

    print(f"\nOK: recall@{GATED_K} is {actual:.1%}, floor {FLOOR:.1%}.")
    if actual > FLOOR + 0.01:
        # Said out loud rather than left as a quiet win: a floor that lags the
        # real number stops protecting the difference between them.
        print(f"Recall is now ABOVE the floor. Raise FLOOR to {actual:.3f} in this commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
