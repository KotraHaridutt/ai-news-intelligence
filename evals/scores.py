"""
Feature 2, Section G (#52, #53): scores against the labels of Section F. Plain code, no LLM.

  retrieval_scores    which chunks retrieval picked vs the chunk grades (0/1/2) and dated flags
  fact_check_scores   what the fact check decided per story vs the expected outcome, and whether
                      the text it showed repeats what the reader was already told
"""
import math
import re

WORD = re.compile(r"[a-z0-9]+(?:[.,'’][a-z0-9]+)*")
NGRAM = 5
# A shown text whose 5-word phrases are mostly already in "told" repeats it (instead of new facts)
REPEAT_SHARE = 0.30


def _ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def retrieval_scores(retrieved: list[str], grades: dict[str, int], dated=(), k: int = 10) -> dict:
    """
    retrieved: chunk ids in the order sent to the model. grades: every chunk's grade.
      recall@k            share of the grade-2 chunks that were retrieved
      precision@k         share of the retrieved chunks that are grade 2
      useful_precision@k  share of the retrieved chunks that are grade 1 or 2
      mrr                 1 / rank of the first grade-2 chunk
      ndcg@k              ranking quality with gains 2^grade - 1 (1.0 = the best possible order)
      dated_recall@k      share of the dated chunks retrieved (what the timeline needs)
    """
    top = retrieved[:k]
    relevant = {c for c, g in grades.items() if g == 2}
    useful = {c for c, g in grades.items() if g >= 1}
    hits = [c for c in top if c in relevant]
    first = next((rank for rank, c in enumerate(top, 1) if c in relevant), None)
    gain = lambda c: 2 ** grades.get(c, 0) - 1
    dcg = sum(gain(c) / math.log2(rank + 1) for rank, c in enumerate(top, 1))
    ideal = sorted((2 ** g - 1 for g in grades.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal, 1))
    dated = set(dated)
    return {
        f"recall@{k}": _ratio(len(hits), len(relevant)),
        f"precision@{k}": _ratio(len(hits), len(top)),
        f"useful_precision@{k}": _ratio(sum(1 for c in top if c in useful), len(top)),
        "mrr": round(1 / first, 4) if first else 0.0,
        f"ndcg@{k}": _ratio(dcg, idcg),
        f"dated_recall@{k}": _ratio(sum(1 for c in top if c in dated), len(dated)),
        "retrieved": len(top), "relevant": len(relevant),
    }


def _ngrams(text: str) -> set[tuple]:
    words = WORD.findall((text or "").lower())
    return {tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def repeat_share(shown: str, told: str) -> float:
    """Share of the shown text's 5-word phrases that already appear in what was told."""
    grams = _ngrams(shown)
    return round(len(grams & _ngrams(told)) / len(grams), 4) if grams else 0.0


def fact_check_scores(outcomes: dict[str, str], expected: dict[str, str], texts: dict[str, str], told: str) -> dict:
    """
    outcomes: the app's decision per story ("shown", "nothing_new", "unjudged"); expected: the label
    ("new", "nothing_new"); texts: what it showed. Scores:
      accuracy             shown for a new story, hidden for a nothing-new one (unjudged counts as wrong)
      null_precision       of the stories it hid, the share that really had nothing new
      null_recall          of the stories with nothing new, the share it hid
      repeats              stories whose shown text is mostly (> 30%) what was told
    """
    ids = sorted(expected)
    right = [i for i in ids if (outcomes.get(i) == "shown" and expected[i] == "new")
             or (outcomes.get(i) == "nothing_new" and expected[i] == "nothing_new")]
    hidden = {i for i in ids if outcomes.get(i) == "nothing_new"}
    nothing = {i for i in ids if expected[i] == "nothing_new"}
    shares = {i: repeat_share(texts[i], told) for i in ids if outcomes.get(i) == "shown" and texts.get(i)}
    return {
        "accuracy": _ratio(len(right), len(ids)),
        "null_precision": _ratio(len(hidden & nothing), len(hidden)),
        "null_recall": _ratio(len(hidden & nothing), len(nothing)),
        "unjudged": sum(1 for i in ids if outcomes.get(i, "unjudged") == "unjudged"),
        "repeat_shares": shares,
        "repeats": sorted(i for i, s in shares.items() if s > REPEAT_SHARE),
        "wrong": sorted(set(ids) - set(right)),
    }
