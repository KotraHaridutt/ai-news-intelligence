import time

import numpy as np

from services import checks, errors, observability, prompts
from services.safety import questions, settings
from services.safety.engine import Decision
from services.safety.guard import check

# --- The model's own conduct in an answer (feature 3, Section K #82-#85) ---
# The news an answer reports can be dark, hateful or one-sided; that's the news, and it's never moderated
# (rule 1). K looks at what the MODEL adds: its own wording, its own instructions showing up, and whether it
# presents the sides its articles give. Clef's three questions ride in J's call, about the whole answer
# (state key `answer`, ~+20 neurons): `tone` (#83), `leak` (#84), `balance` (#85).
#
#   toxicity  P(hateful) >= TONE_WITHHOLD_AT: withheld in enforce (decided 2026-11-02, #83: 502, the canary's
#             message); loaded / insulting wording: flagged only. No fallback (fails open).
#   leak      flag only, next to the canary (D). Fallback: an answer sentence very close by meaning to a
#             sentence of the system prompts (embedded once per process).
#   balance   flag only: Clef's level, plus free numbers (outlets in the chunks and named in the answer, the
#             biggest one's share; for contradictions, points that name two outlets).
# Gemini's own ratings (#82) are on the request's totals since F #46 (Gemini 3 sends them only on a block).

TONE, LEAK, BALANCE = (questions.get(n) for n in ("tone", "leak", "balance"))
CHECKS = {"toxicity": "tone", "leak": "leak", "balance": "balance"}      # guard -> question
LEAK_SIMILAR_AT = 0.80               # the fallback: cosine to a system-prompt sentence (K §6: leaks 0.82-1.0, others <= 0.765)

_system = {"sentences": None, "vectors": None}


def review_for() -> list[str]:
    """The K questions to ask on this request (their guards aren't off)."""
    return [q for name, q in CHECKS.items() if settings.mode(name) != "off"]


def system_sentences() -> list[str]:
    """Every instruction sentence of the four system prompts (E put all the rules there), placeholders out."""
    from services.safety import faithfulness
    out = []
    for name in ("report", "timeline", "contradictions", "fact_check"):
        system = prompts.get(name).system or ""
        for unit in faithfulness.units(system.replace("{canary}", "the marker")):
            unit = unit.lstrip("-•* ").strip()
            if len(unit.split()) >= 5 and unit not in out:
                out.append(unit)
    return out


async def _system_vectors() -> np.ndarray:
    if _system["vectors"] is None:
        from services import rag_service
        from services.safety.faithfulness import normalized
        _system["sentences"] = system_sentences()
        _system["vectors"] = normalized(await rag_service.get_vectors(_system["sentences"], purpose="safety"))
    return _system["vectors"]


async def leak_similarity(sentence_vectors) -> list[float]:
    """Each answer sentence's cosine to its closest system-prompt sentence (the leak fallback)."""
    if sentence_vectors is None or not len(sentence_vectors):
        return []
    return [round(float(x), 4) for x in (np.asarray(sentence_vectors) @ (await _system_vectors()).T).max(axis=1)]


def balance_numbers(view: str, answer: str, rows: list[dict]) -> dict:
    """Free numbers (#85): outlets among the chunks and among those the answer names, the biggest one's share."""
    outlets = [(r.get("source_name") or "").strip() for r in rows if (r.get("source_name") or "").strip()]
    named = {o: sum(1 for u in _sentences(answer) if checks._mentions(u, o)) for o in sorted(set(outlets))}
    named = {o: n for o, n in named.items() if n}
    out = {"chunk_outlets": len(set(outlets)),
           "chunk_top_share": round(max(outlets.count(o) for o in set(outlets)) / len(outlets), 3) if outlets else None,
           "named_outlets": len(named),
           "named_top_share": round(max(named.values()) / sum(named.values()), 3) if named else None}
    if view == "contradictions":
        result = checks.check_contradictions(answer, [{"source_name": o} for o in set(outlets)])
        out["points"] = result.get("points")
        out["points_one_sided"] = (result.get("single_outlet") or 0) + (result.get("no_outlet") or 0)
    return out


def _sentences(text: str) -> list[str]:
    from services.safety.faithfulness import units
    return units(text)


def level_name(answer: dict | None) -> str:
    """The likeliest level of a score answer, by name (Clef keys its probabilities by position, K §7)."""
    probs = (answer or {}).get("probabilities") or {}
    if not probs:
        return "?"
    top = max(probs, key=probs.get)
    names = list(BALANCE.options.values())
    return names[int(top)] if str(top).isdigit() and int(top) < len(names) else str(top)


def _p(answer: dict | None, option: str) -> float:
    return float(((answer or {}).get("probabilities") or {}).get(option, 0.0))


async def record(where: str, j, answer: str, rows: list[dict], view: str):
    """
    K's events for one answer, from J's Judgement `j` (Clef's review answers, the sentence vectors). Raises
    answer_withheld (502) when the model's own wording is hateful and toxicity is in enforce mode.
    """
    started = time.perf_counter()
    review = j.review or {}

    if settings.mode("toxicity") != "off":
        async def run_tone() -> Decision:
            tone = review.get("tone")
            if tone is None:
                return Decision.failed_open(j.reason or "no tone answer", threshold=questions.TONE_WITHHOLD_AT)
            hateful = _p(tone, "hateful")
            score = TONE.block_score(tone)
            verdict = "block" if hateful >= questions.TONE_WITHHOLD_AT else "flag" if score >= TONE.block_at else "allow"
            return Decision(verdict=verdict, score=hateful if verdict == "block" else score,
                            threshold=questions.TONE_WITHHOLD_AT if verdict == "block" else TONE.block_at, engine="clef",
                            label=tone.get("choice"), probabilities=tone.get("probabilities"))
        await check("toxicity", run_tone, where=where, text=answer,
                    refuse=lambda d: errors.answer_withheld(f"{where}: the answer's own wording was judged hateful "
                                                            f"(P {d.score:.2f})"))

    if settings.mode("leak") != "off":
        async def run_leak() -> Decision:
            leak = review.get("leak")
            if leak is not None:
                score = LEAK.block_score(leak)
                return Decision(verdict="flag" if score >= LEAK.block_at else "allow", score=score, threshold=LEAK.block_at,
                                engine="clef", label=leak.get("choice"), probabilities=leak.get("probabilities"))
            if j.sentence_vectors is not None:
                sims = await leak_similarity(j.sentence_vectors)
                best = max(sims) if sims else 0.0
                return Decision(verdict="flag" if best >= LEAK_SIMILAR_AT else "allow", score=best, threshold=LEAK_SIMILAR_AT,
                                engine="embedding", label=f"closest system sentence {best:.3f}", reason=j.reason)
            return Decision.failed_open(j.reason or "no leak answer", threshold=LEAK.block_at)
        await check("leak", run_leak, where=where, text=answer)

    numbers = balance_numbers(view, answer, rows)
    if settings.mode("balance") != "off":
        async def run_balance() -> Decision:
            bal = review.get("balance")
            one_sided_points = view == "contradictions" and numbers.get("points_one_sided", 0) > 0
            if bal is None:
                return Decision(verdict="flag" if one_sided_points else "allow", score=None, threshold=BALANCE.block_at,
                                engine="code", label=f"{numbers.get('points_one_sided', 0)} one-sided points" if one_sided_points else None,
                                reason=j.reason)
            score = BALANCE.block_score(bal)
            level = level_name(bal)
            parts = [level] + ([f"{numbers['points_one_sided']} one-sided points"] if one_sided_points else [])
            return Decision(verdict="flag" if score >= BALANCE.block_at or one_sided_points else "allow", score=score,
                            threshold=BALANCE.block_at, engine="clef", label="; ".join(parts), probabilities=bal.get("probabilities"))
        await check("balance", run_balance, where=where, text=answer)

    observability.annotate(conduct={"review": review, "balance_numbers": numbers, "ms": round((time.perf_counter() - started) * 1000)})
