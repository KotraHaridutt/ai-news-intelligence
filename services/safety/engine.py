import asyncio
import hashlib
import json
from dataclasses import dataclass

from services import cache_service
from services.safety import classifier, clef, settings
from services.safety.questions import Question

# --- Which engine decides (feature 3, Section B #15b, #18, #19) ---
# Clef first (SAFETY_ENGINE=clef, the default). The same check runs on the embedding classifier
# when Clef is off, times out (the question's timeout_s), answers 5xx or 429 after its one retry,
# refuses the call (401/402/...), or today's neurons passed clef.FALLBACK_AT_NEURONS.
# When neither can run (no classifier for the question, or the embedding API fails too), the check
# fails OPEN with a flag (decided 2026-10-31, #18): the request continues, and the other layers
# (prompt isolation, Gemini's filter, the output checks) still protect the answer. A safety check
# that is down must not take the whole app down.

# The fallback's embedding call: a fresh query takes ~1.2 s; a 429 is retried with 2, 4 and 8 s
# waits (rag_service.call_with_retry), which a safety check shouldn't sit through. Past this,
# the check fails open (the pipeline still gets its vector: it embeds the query itself).
EMBED_TIMEOUT_SECONDS = 5.0


@dataclass
class Decision:
    """What one check decided about one item."""
    verdict: str                     # allow / flag / block
    score: float | None              # block score 0-1; None when it failed open
    threshold: float | None          # the score from which it blocks
    engine: str                      # clef / embedding / none
    label: str | None = None         # the most likely option or class
    probabilities: dict | None = None
    cached: bool = False
    reason: str | None = None        # why the fallback decided, or why it failed open
    enforced: bool = False           # set by safety.check: a block acted on in enforce mode

    @classmethod
    def failed_open(cls, reason: str, threshold: float | None = None) -> "Decision":
        return cls(verdict="flag", score=None, threshold=threshold, engine="none", reason=reason)


def _call_of(question: Question, key: str, items: dict[str, str], shared: dict[str, str]) -> tuple[dict, str]:
    """The state and the key one item is asked about, as the question was calibrated (rule 6):
    one item per call under the question's own state key, or, batched, under the item's key."""
    if question.batch:
        return {**shared, **items}, key
    return {**shared, question.state_key: items[key]}, question.state_key


def content_hash(question: Question, key: str, items: dict[str, str], shared: dict[str, str] | None = None) -> str:
    """What a verdict depends on: the question's version, the pinned model, and the texts it reads."""
    state, asked = _call_of(question, key, items, shared or {})
    texts = [state[k] for k in question.reads(asked)]
    raw = json.dumps([question.version, clef.MODEL, texts], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _from_clef(question: Question, answer: dict, cached: bool) -> Decision:
    score = question.block_score(answer)
    probabilities = answer.get("probabilities")
    label = answer.get("choice") or (max(probabilities, key=probabilities.get) if probabilities else None)
    return Decision(verdict=question.band(score), score=score, threshold=question.block_at, engine="clef",
                    label=label, probabilities=probabilities or {"noul": answer.get("noul")}, cached=cached)


async def _ask_one(question: Question, key: str, items: dict[str, str], shared: dict[str, str], label: str) -> dict:
    """One item, one call, named and keyed as calibrated. Returns Clef's answer; raises ClefError."""
    state, asked = _call_of(question, key, items, shared)
    reply = await clef.ask(state, {question.clef_name: question.payload(asked)}, timeout=question.timeout_s,
                           label=label or question.name)
    return reply["answers"][question.clef_name]


async def _ask_clef(question: Question, keys: list[str], items: dict[str, str], shared: dict[str, str],
                    label: str) -> tuple[dict[str, Decision], list[str], str | None]:
    """
    Cached verdicts first; Clef for the rest, whose answers are cached. Returns (decisions, the
    keys Clef couldn't answer, why). A batched question is one set of calls: it all fails together.
    Items asked one per call fail one by one, so one slow chunk doesn't send every chunk to the fallback.
    """
    cache_keys = {key: cache_service.verdict_key(question.version, content_hash(question, key, items, shared))
                  for key in keys}
    cached = await cache_service.get_verdicts(list(cache_keys.values()))
    decisions, missing = {}, []
    for key, answer in zip(keys, cached):
        try:
            if answer is not None:
                decisions[key] = _from_clef(question, answer, cached=True)
                continue
        except (KeyError, TypeError, ValueError):
            pass                      # an entry in an old shape: ask again
        missing.append(key)
    if not missing:
        return decisions, [], None
    if question.batch:
        try:
            reply = await clef.ask({**shared, **items}, {key: question.payload(key) for key in missing},
                                   timeout=question.timeout_s, label=label or question.name)
        except clef.ClefError as e:
            return decisions, missing, f"clef {e.kind}"
        answers = {key: reply["answers"][key] for key in missing}
    else:
        results = await asyncio.gather(*(_ask_one(question, key, items, shared, label) for key in missing),
                                       return_exceptions=True)
        answers, failed, why = {}, [], None
        for key, result in zip(missing, results):
            if isinstance(result, clef.ClefError):
                failed.append(key)
                why = why or f"clef {result.kind}"
            elif isinstance(result, BaseException):
                raise result
            else:
                answers[key] = result
        if failed:
            decisions.update({key: _from_clef(question, a, cached=False) for key, a in answers.items()})
            await cache_service.set_verdicts({cache_keys[key]: a for key, a in answers.items()})
            return decisions, failed, why
    decisions.update({key: _from_clef(question, a, cached=False) for key, a in answers.items()})
    await cache_service.set_verdicts({cache_keys[key]: a for key, a in answers.items()})
    return decisions, [], None


async def _ask_classifier(question: Question, keys: list[str], items: dict[str, str], vectors: dict | None,
                          reason: str | None) -> dict[str, Decision]:
    clf = classifier.get(question.fallback) if question.fallback else None
    if clf is None:
        why = f"{reason}; " if reason else ""
        return {key: Decision.failed_open(f"{why}no fallback classifier for {question.name}", question.block_at)
                for key in keys}
    vectors = vectors or {}
    try:
        needed = [key for key in keys if key not in vectors]
        if needed:
            found = await asyncio.wait_for(classifier.vectors_for(clf, [items[k] for k in needed]), EMBED_TIMEOUT_SECONDS)
            vectors = {**vectors, **dict(zip(needed, found))}
    except Exception as e:
        why = f"{reason}; " if reason else ""
        kind = "timeout" if isinstance(e, asyncio.TimeoutError) else f"failed: {type(e).__name__}"
        return {key: Decision.failed_open(f"{why}embedding {kind}", clf.block_at) for key in keys}
    decisions = {}
    for key in keys:
        probabilities = clf.probabilities(vectors[key])
        score = clf.block_score(probabilities)
        decisions[key] = Decision(verdict=clf.band(score), score=score, threshold=clf.block_at, engine="embedding",
                                  label=max(probabilities, key=probabilities.get), probabilities=probabilities,
                                  reason=reason)
    return decisions


async def judge(question: Question, items: dict[str, str], shared: dict[str, str] | None = None,
                vectors: dict | None = None, label: str = "") -> dict[str, Decision]:
    """
    Asks `question` about each item (state key -> text), with `shared` state every question can
    read (the query, the articles). `vectors` (key -> the item's embedding) lets a caller that
    already has them (the index's chunk vectors) spare the fallback an embedding call.
    Returns {key: Decision}. Never raises for an engine failure: that's a fallback or a fail-open.
    """
    if not items:
        return {}
    shared = shared or {}
    keys = list(items)
    if settings.engine() != "clef":
        return await _ask_classifier(question, keys, items, vectors, "SAFETY_ENGINE=embedding")
    try:
        decisions, failed, reason = await _ask_clef(question, keys, items, shared, label)
    except clef.ClefError as e:          # not configured / over budget: raised before any call
        decisions, failed, reason = {}, keys, f"clef {e.kind}"
    if failed:
        decisions.update(await _ask_classifier(question, failed, items, vectors, reason))
    return decisions


async def judge_one(question: Question, text: str, key: str = "text", shared: dict[str, str] | None = None,
                    vector=None, label: str = "") -> Decision:
    decisions = await judge(question, {key: text}, shared, {key: vector} if vector is not None else None, label)
    return decisions[key]
