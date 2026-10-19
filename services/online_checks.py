import os

from services import checks, observability

# --- Checks on live answers (feature 2, Section I) ---
# After every answer the app gives, the free code checks of services/checks.py run on it (in
# milliseconds, no LLM call), and every result goes to:
#   - the trace, as feedback scores on the request (check.score, check.no_bullets, ...), so
#     LangSmith can chart them over time (Section K) and route failures to a review queue (J)
#   - the request's totals (check_score, checks_failed), filterable on the root run
#   - one WARN line in the log when a check fails, so it's visible without LangSmith too
# A check that crashes, or a LangSmith that's down, is logged and skipped: it never fails the
# request. ONLINE_CHECKS=false turns all of it off.


def enabled() -> bool:
    return (os.getenv("ONLINE_CHECKS") or "true").strip().lower() not in ("0", "false", "no", "off")


def _record(results: dict, label: str, comments: dict | None = None):
    """Feedback, totals and a log line for one set of results ({name: 0..1})."""
    failed = sorted(name for name, score in results.items() if name != "check.score" and score < 1)
    observability.add_totals(checks_failed=failed, **({"check_score": results["check.score"]} if "check.score" in results else {}))
    for name, score in results.items():
        observability.feedback(name, score, comment=(comments or {}).get(name))
    if failed:
        print(f"[WARN] Checks on the {label}: failed {', '.join(failed)}")


def answer(view: str, text: str, documents, index_documents, k: int, cap: int) -> dict:
    """The checks for one answer (report / timeline / contradictions) and the retrieval behind it."""
    if not enabled():
        return {}
    try:
        result = checks.check_answer(view, text, [d.metadata for d in documents])
        retrieval = checks.check_retrieval(documents, k=k, cap=cap, index=index_documents)
        results = {"check.score": result["score"], **{f"check.{name}": float(ok) for name, ok in result["checks"].items()},
                   "check.story_cap": float(retrieval["checks"]["story_cap"])}
        details = {k: v for k, v in result.items() if k not in ("score", "ok", "checks") and v}
        _record(results, view, comments={"check.score": str(details)[:500] if details else None})
        observability.annotate(checks={**results, "details": details})
        return results
    except Exception as e:
        print(f"[WARN] Checks on the {view} couldn't run: {type(e).__name__}: {e}")
        return {}


def fact_check(answer_obj, outcomes: dict, sent_ids: list) -> dict:
    """
    The fact check's health (#66): readable JSON with every story answered once, the share of
    stories it hid as nothing new, and the share left unjudged. A prompt change that starts
    hiding everything (or answering nothing) shows up on the dashboard before readers notice.
    """
    if not enabled() or not sent_ids:
        return {}
    try:
        json_check = checks.check_fact_check(answer_obj, sent_ids)
        total = len(sent_ids)
        results = {"check.fact_check_json": json_check["score"],
                   "fact_check.usable": float(answer_obj is not None),
                   "fact_check.nothing_new_share": round(sum(1 for o in outcomes.values() if o == "nothing_new") / total, 4),
                   "fact_check.unjudged_share": round(sum(1 for o in outcomes.values() if o == "unjudged") / total, 4)}
        # The two shares are measurements, not pass/fail: only the first two count as failures
        failed = [n for n in ("check.fact_check_json", "fact_check.usable") if results[n] < 1]
        observability.add_totals(checks_failed=failed)
        for name, score in results.items():
            observability.feedback(name, score)
        if failed:
            print(f"[WARN] Checks on the fact check: failed {', '.join(failed)} {json_check.get('missing') or ''}")
        observability.annotate(checks=results)
        return results
    except Exception as e:
        print(f"[WARN] Checks on the fact check couldn't run: {type(e).__name__}: {e}")
        return {}
