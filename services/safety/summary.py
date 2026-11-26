import time

from services import observability

# --- One safety summary per request (feature 3, Section O #111, #112, #112a) ---
# Every SafetyEvent already goes on the span that was current when its check ran (events.record), often
# a step inside the request: the view's chain, the pipeline, the fact check. A LangSmith filter, chart
# or alert on whole requests (is_root) can't see a tag that sits on a step, and tags don't travel up.
# So the request also keeps its events here, and when its root span closes (an observability end hook,
# on success and on failure alike) this module writes, on the root run:
#
#   metadata safety_summary   every check that ran: its worst verdict, how often, engine, time, score;
#                             what was blocked, flagged, enforced; who decided; the time safety took;
#                             the request's Clef calls, neurons and AI Gateway log ids (#112, #112a)
#   metadata safety_verdict   the worst verdict of the request (allow / flag / block), to group by
#   tags                      safety:<check>:<verdict> and safety:<check>:enforced per check, and the
#                             request-level safety:block / safety:flag / safety:enforced /
#                             safety:fallback / safety:failed_open, clef_budget_warning / _fallback (#111)
#   feedback                  safety.* scores on the trace, for the charts and alerts (#111, FEEDBACK_KEYS)
#
# Every check reaches it through events.record: the guards' check(), Gemini's record_block (F), the ssrf
# refusals in the scraper and the origin and limits middleware's monitor-mode events (M, L: flushed when
# the root span opens). A middleware refusal in enforce mode has no trace yet: it's only logged.
#
# Nothing here holds checked text: the events carry scores and hashes (events.py) and the summary only
# counts them. A request with no safety event gets no summary, tag or feedback.

# Verdicts from best to worst: the request's verdict for a check is the worst one it got
_RANK = {"allow": 0, "flag": 1, "block": 2}
# Engines that decide by meaning. "code" (formats, counts, the canary) and "gemini" (Gemini's own
# filter) aren't a choice between Clef and the fallback, so they don't count in the fallback share.
MEANING_ENGINES = ("clef", "embedding", "none")
# The gateway log ids kept per request: enough to find a request's calls in Cloudflare's logs
MAX_LOG_IDS = 20

# Request-level tags
TAG_BLOCK = "safety:block"            # some check's verdict was block (monitor or enforce)
TAG_FLAG = "safety:flag"              # some check flagged (uncertain band, or failed open)
TAG_ENFORCED = "safety:enforced"      # a block was acted on: refused, removed, swapped, masked or noted
TAG_FALLBACK = "safety:fallback"      # the embedding classifier decided at least one check
TAG_FAILED_OPEN = "safety:failed_open"  # no engine could decide a check (#18)
BUDGET_TAGS = ("clef_budget_warning", "clef_budget_fallback")   # from clef._check_levels' totals (#93a)

# The feedback keys, each a score on the request's trace (sent only when it applies):
FEEDBACK_KEYS = {
    "safety.blocked": "1 when any check's verdict was block (in monitor mode too), else 0",
    "safety.flagged": "1 when any check flagged, else 0",
    "safety.enforced": "1 when a block was acted on, else 0",
    "safety.seconds": "wall time during which a safety check was running (overlaps counted once)",
    "safety.fallback_share": "share of the meaning checks the embedding classifier (or no engine) decided",
    "safety.clef_neurons": "Clef neurons this request spent",
    "safety.injection_found": "sentences, titles and outlets the injection scan would remove",
    "safety.injection_removed": "... and removed (enforce mode)",
    "safety.pii_found": "places personal data was found (query, comment, answer, texts)",
}
# The faithfulness check's own score (J #74, faithfulness.py) is also a safety.* key
FAITHFULNESS_KEY = "safety.faithfulness"


def _state() -> dict | None:
    """The request's safety part of observability.request_state(), made on first use; None outside a request."""
    state = observability.request_state()
    if state is None:
        return None
    return state.setdefault("safety", {"events": [], "intervals": [], "clef": {"calls": 0, "errors": 0, "neurons": 0.0, "log_ids": []}})


def collect(event, ended: float | None = None):
    """events.record: keeps the event (and when its check ran) for the request's summary."""
    state = _state()
    if state is None:
        return
    ended = time.perf_counter() if ended is None else ended
    state["events"].append(event)
    if event.ms:
        state["intervals"].append((ended - event.ms / 1000, ended))


def note_time(started: float, ended: float | None = None):
    """Safety work done outside a check()'s run(), e.g. the one Clef call that judges a whole answer
    before the faithfulness, topic and K checks read it (faithfulness.verify)."""
    state = _state()
    if state is not None:
        state["intervals"].append((started, time.perf_counter() if ended is None else ended))


def note_clef(neurons: float = 0.0, call: bool = False, failed: bool = False, log_id: str | None = None):
    """clef.py: a Clef call this request made (neurons are counted per response, failed ones too)."""
    state = _state()
    if state is None:
        return
    clef = state["clef"]
    clef["neurons"] = round(clef["neurons"] + neurons, 4)
    clef["calls"] += 1 if call else 0
    clef["errors"] += 1 if failed else 0
    if log_id and len(clef["log_ids"]) < MAX_LOG_IDS:
        clef["log_ids"].append(log_id)


def covered_seconds(intervals) -> float:
    """The wall time covered by at least one interval: checks that ran side by side count once."""
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(events, intervals=(), clef=None, totals=None) -> dict | None:
    """
    The request's summary from its events (SafetyEvent objects), the times safety ran, its Clef calls
    and its totals. None when no check ran. Pure: no trace needed (tests call it directly).
    """
    if not events:
        return None
    totals = totals or {}
    checks: dict[str, dict] = {}
    engines: dict[str, int] = {}
    for e in events:
        c = checks.get(e.check)
        if c is None:
            c = checks[e.check] = {"verdict": e.verdict, "runs": 0, "blocks": 0, "flags": 0, "mode": e.mode,
                                   "engines": [], "ms": 0, "score": None, "enforced": False, "where": []}
        c["runs"] += 1
        c["blocks"] += e.verdict == "block"
        c["flags"] += e.verdict == "flag"
        c["ms"] += round(e.ms)
        if _RANK[e.verdict] > _RANK[c["verdict"]]:
            c["verdict"] = e.verdict
        if e.score is not None and (c["score"] is None or e.score > c["score"]):
            c["score"] = round(e.score, 4)
        c["enforced"] = c["enforced"] or e.enforced
        if e.engine not in c["engines"]:
            c["engines"].append(e.engine)
        if e.where not in c["where"]:
            c["where"].append(e.where)
        engines[e.engine] = engines.get(e.engine, 0) + 1
        if e.cached:
            c["cached"] = c.get("cached", 0) + 1
        if e.label and e.verdict != "allow":
            c.setdefault("labels", [])
            if e.label not in c["labels"] and len(c["labels"]) < 3:
                c["labels"].append(e.label[:160])
        if e.reason and e.verdict != "allow" and "reason" not in c:
            c["reason"] = e.reason[:160]

    meaning = sum(engines.get(name, 0) for name in MEANING_ENGINES)
    out_of_clef = engines.get("embedding", 0) + engines.get("none", 0)
    worst = max((c["verdict"] for c in checks.values()), key=_RANK.get)
    summary = {
        "verdict": worst,
        "checks": checks,
        "blocked": sorted(name for name, c in checks.items() if c["verdict"] == "block"),
        "flagged": sorted(name for name, c in checks.items() if c["verdict"] == "flag"),
        "enforced": sorted(name for name, c in checks.items() if c["enforced"]),
        "events": len(events),
        "engines": engines,
        "fallback_share": round(out_of_clef / meaning, 4) if meaning else None,
        "failed_open": engines.get("none", 0),
        "seconds": round(covered_seconds(intervals), 3),
        "check_seconds": round(sum(e.ms for e in events) / 1000, 3),
    }
    if clef and (clef.get("calls") or clef.get("neurons") or clef.get("errors")):
        summary["clef"] = {"calls": clef["calls"], "errors": clef["errors"], "neurons": round(clef["neurons"], 2),
                           "log_ids": list(clef["log_ids"])}
    for key in ("injection_found", "injection_removed", "faithfulness_unsupported", "relevance_off_topic"):
        if key in totals:
            summary[key] = totals[key]
    budget = [tag for tag in BUDGET_TAGS if totals.get(tag)]
    if budget:
        summary["clef_budget"] = budget
    return summary


def root_tags(summary: dict) -> list[str]:
    """The tags the root run gets: per check, and for the request as a whole."""
    tags = []
    for name, c in summary["checks"].items():
        tags.append(f"safety:{name}:{c['verdict']}")
        if c["enforced"]:
            tags.append(f"safety:{name}:enforced")
    if summary["blocked"]:
        tags.append(TAG_BLOCK)
    if summary["flagged"]:
        tags.append(TAG_FLAG)
    if summary["enforced"]:
        tags.append(TAG_ENFORCED)
    if summary["engines"].get("embedding"):
        tags.append(TAG_FALLBACK)
    if summary["failed_open"]:
        tags.append(TAG_FAILED_OPEN)
    tags += summary.get("clef_budget", [])
    return sorted(set(tags))


def feedback_scores(summary: dict) -> dict[str, float]:
    """The safety.* feedback scores for the request (FEEDBACK_KEYS); a key that doesn't apply is left out."""
    scores = {
        "safety.blocked": 1 if summary["blocked"] else 0,
        "safety.flagged": 1 if summary["flagged"] else 0,
        "safety.enforced": 1 if summary["enforced"] else 0,
        "safety.seconds": summary["seconds"],
    }
    if summary["fallback_share"] is not None:
        scores["safety.fallback_share"] = summary["fallback_share"]
    if "clef" in summary:
        scores["safety.clef_neurons"] = summary["clef"]["neurons"]
    if "injection" in summary["checks"]:
        scores["safety.injection_found"] = summary.get("injection_found", 0)
        scores["safety.injection_removed"] = summary.get("injection_removed", 0)
    if "pii" in summary["checks"]:
        # one pii event per place checked (query, comment, answer, texts); those that found something block
        scores["safety.pii_found"] = summary["checks"]["pii"]["blocks"]
    return scores


def _comments(summary: dict) -> dict[str, str]:
    """The comments on safety.blocked / flagged / enforced: which checks, so a feedback list reads without
    opening the trace. Check names only, never text."""
    out = {}
    for key, field in (("safety.blocked", "blocked"), ("safety.flagged", "flagged"), ("safety.enforced", "enforced")):
        if summary[field]:
            out[key] = f"{field}: " + ", ".join(summary[field])
    return out


def finish(run, error: BaseException | None = None):
    """The observability end hook: the summary, tags and feedback on the request's root run."""
    state = _state()
    if state is None or not state["events"]:
        return
    summary = summarize(state["events"], state["intervals"], state["clef"], observability.current_totals())
    run.add_metadata({"safety_summary": summary, "safety_verdict": summary["verdict"]})
    # A check that ran on the root span itself already tagged it (events.record); add_tags doesn't dedupe
    have = set(run.tags or [])
    new = [tag for tag in root_tags(summary) if tag not in have]
    if new:
        run.add_tags(new)
    comments = _comments(summary)
    for key, score in feedback_scores(summary).items():
        observability.feedback(key, score, comment=comments.get(key))
    line = (f"[SAFETY] request {summary['verdict']}: {len(summary['checks'])} checks, {summary['events']} events, "
            f"{summary['seconds']:.2f}s")
    if comments:
        line += "; " + "; ".join(comments.values())
    if "clef" in summary:
        clef = summary["clef"]
        line += f"; Clef {clef['calls']} calls" + (f", {clef['errors']} failed" if clef["errors"] else "") + f", {clef['neurons']:.1f} neurons"
    if summary["fallback_share"]:
        line += f"; fallback decided {summary['fallback_share']:.0%}"
    print(line)


observability.add_request_end_hook(finish)
