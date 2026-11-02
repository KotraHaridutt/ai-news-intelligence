from typing import Callable

from langchain_core.documents import Document

from services import observability
from services.safety import questions, settings
from services.safety.engine import Decision, judge
from services.safety.guard import check

# --- Retrieved chunks that aren't about the query (feature 3, Section G #52a) ---
# A planted or SEO page can rank among the top chunks by sharing the query's words without being
# about its topic. Before the answer is written, Clef is asked about each chunk the answer would get:
# noul("`chunk_03` is about the topic in `query`"), all of them in one call, with the query and the
# chunks in `state` (never in the question's wording, #15c), as calibrated in A2 §5.4. A chunk with
# P below 0.3 is in the block band (1 of 50 own chunks, 37 of 50 other topics' chunks).
#
#   monitor  every chunk gets its score (metadata "relevance"); nothing changes
#   enforce  a chunk in the block band is swapped for the next one the retrieval ranks, under the
#            same per-story cap, so the answer still gets k chunks when the index has them (like the
#            refill in D #34). The replacements are judged too, once (MAX_ROUNDS).
# No fallback classifier: when Clef can't run the check fails open with a flag (#18), chunks unchanged.
# Verdicts are cached per (chunk text, query): another view of the same topic pays only for its new chunks.

RELEVANCE = questions.get("relevance")
# Judged rounds: the k chunks, then their replacements. A replacement's own replacement is kept
# unjudged (marked so on the trace): a third round would cost another call and ~1.5 s for a case
# where most of the index is off-topic, and the answer then has few good chunks either way.
MAX_ROUNDS = 2


def _key(i: int) -> str:
    # The state key Clef was calibrated with (rule 6): chunk_00, chunk_01, ...
    return f"chunk_{i:02d}"


def p_on_topic(decision: Decision | None) -> float | None:
    """P(the chunk is about the query's topic): 1 - the block score, None when it wasn't judged."""
    if decision is None or decision.score is None:
        return None
    return round(1.0 - decision.score, 4)


async def screen(query: str, ranked: list, chosen: list, pick: Callable[[list], list], where: str) -> list:
    """
    The chunks for one answer. `ranked`: the whole index, best first; `chosen`: the k picked from it;
    `pick(pool)`: picks k from a pool the same way (the per-story cap). Returns copies of the chunks
    carrying metadata "relevance" (P on topic, None if unjudged), or `chosen` as it is when the check
    is off or failed. One safety event per answer; the counts go on the retriever span.
    """
    if not chosen:
        return chosen
    judged: dict[int, Decision] = {}
    box: dict = {}

    async def run() -> Decision:
        current, removed, rounds = list(chosen), set(), 0
        while True:
            new = [doc for doc in current if id(doc) not in judged]
            if new and rounds < MAX_ROUNDS:
                decisions = await judge(RELEVANCE, {_key(i): doc.page_content for i, doc in enumerate(new)},
                                        shared={"query": query}, label=f"{where}: relevance")
                judged.update({id(doc): decisions[_key(i)] for i, doc in enumerate(new)})
                rounds += 1
            off = [doc for doc in current if id(doc) in judged and judged[id(doc)].verdict == "block"]
            if not off or settings.mode("relevance") != "enforce":
                break
            removed |= {id(doc) for doc in off}
            current = pick([doc for doc in ranked if id(doc) not in removed])
        box.update(current=current, removed=len(removed), rounds=rounds)
        return _decision([judged[id(doc)] for doc in chosen], len(removed), sum(1 for d in current if id(d) not in judged))

    decision = await check("relevance", run, where=where)
    if decision is None or "current" not in box:
        return chosen
    final = box["current"] if decision.enforced else list(chosen)
    scores = [p_on_topic(judged.get(id(doc))) for doc in chosen]
    observability.annotate(relevance={
        "scores": scores, "off_topic": sum(1 for d in chosen if judged[id(d)].verdict == "block"),
        "swapped": box["removed"] if decision.enforced else 0, "rounds": box["rounds"], "enforced": decision.enforced,
        "unjudged_in_answer": sum(1 for d in final if id(d) not in judged)})
    observability.add_totals(relevance_off_topic=sum(1 for d in chosen if judged[id(d)].verdict == "block"),
                             relevance_swapped=box["removed"] if decision.enforced else 0)
    return [Document(page_content=doc.page_content, metadata={**doc.metadata, "relevance": p_on_topic(judged.get(id(doc)))})
            for doc in final]


# --- Not enough coverage (feature 3, J #79) ---
# When none of the chunks an answer would get is about the query, the model would have to fill the
# answer in from what it knows. In enforce mode the user gets "not enough coverage" (404) instead, before
# Gemini is called (a chat call saved). Decided by the relevance scores above when Clef judged the chunks;
# otherwise by embedding similarity: the query's vector against each chunk's, both from the pipeline's
# cache (a chunk was embedded for the index, the query for ranking), so usually no embedding call.
# Needs no Clef of its own. Fails open: an unjudged chunk counts as covered, and an embedding failure allows.
COVERAGE_MIN_SIMILARITY = 0.60      # the fallback: the best chunk must be at least this close (J §9)


def _cosines(query_vector, vectors) -> list[float]:
    import numpy as np
    q = np.asarray(query_vector, dtype=np.float32)
    m = np.asarray(vectors, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return [round(float(x), 4) for x in m @ q]


async def coverage(query: str, documents: list, where: str, refuse: Callable[[Decision], Exception]) -> Decision | None:
    """Raises refuse(decision) in enforce mode when no chunk is about the query. Returns the decision."""
    if not documents:
        return None
    scores = [doc.metadata.get("relevance") for doc in documents]

    async def run() -> Decision:
        if any(p is not None for p in scores):
            # Relevance judged them (G #52a): covered when any chunk is outside the block band, or unjudged
            on_topic = [p for p in scores if p is None or 1.0 - p < RELEVANCE.block_at]
            best = max((p for p in scores if p is not None), default=None)
            return Decision(verdict="allow" if on_topic else "block", score=None if best is None else round(1.0 - best, 4),
                            threshold=RELEVANCE.block_at, engine="clef",
                            label=f"{len(on_topic)} of {len(scores)} chunks on topic")
        from services import rag_service       # rag_service imports this module
        query_vector = await rag_service.get_query_vector(query, purpose="safety")
        sims = _cosines(query_vector, await rag_service.get_vectors([doc.page_content for doc in documents], purpose="safety"))
        best = max(sims)
        return Decision(verdict="allow" if best >= COVERAGE_MIN_SIMILARITY else "block", score=round(1.0 - best, 4),
                        threshold=round(1.0 - COVERAGE_MIN_SIMILARITY, 4), engine="embedding",
                        label=f"best chunk similarity {best:.3f}")
    return await check("coverage", run, where=where, refuse=refuse, text=query)


def _decision(first: list[Decision], removed: int, unjudged: int) -> Decision:
    """One decision for the answer's chunks: block when any chunk was off-topic, flag when any was
    uncertain or couldn't be judged, else allow. The score is the worst chunk's."""
    off = sum(1 for d in first if d.verdict == "block")
    failed = [d for d in first if d.engine == "none"]
    unsure = sum(1 for d in first if d.verdict == "flag") - len(failed)
    scored = [d.score for d in first if d.score is not None]
    verdict = "block" if off else "flag" if (unsure or failed) else "allow"
    parts = [f"{off} of {len(first)} chunks off-topic"]
    if removed:
        parts.append(f"{removed} swapped" + (f", {unjudged} replacement(s) unjudged" if unjudged else ""))
    if unsure:
        parts.append(f"{unsure} uncertain")
    if failed:
        parts.append(f"{len(failed)} unjudged (no engine)")
    return Decision(verdict=verdict, score=max(scored) if scored else None, threshold=RELEVANCE.block_at,
                    engine="clef" if len(failed) < len(first) else "none", label="; ".join(parts),
                    cached=bool(first) and all(d.cached for d in first),
                    reason=failed[0].reason if failed else None)
