import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from services import cache_service, observability
from services.safety import clef, pii, privacy

# --- Reranker (feature 4, Section B) ---
# A cross-encoder reads the query and each chunk together and gives one relevance score per chunk.
# FAISS embeds them separately, so it can rank a chunk that only shares the topic's words above
# one that is about it (Section A §3.5: tesla's index, where a third of the chunks are about other
# companies). Two engines, both called over HTTP (nothing installed):
#   cohere      Cohere Rerank (POST /v2/rerank) on the trial key: 10 calls a minute, 1,000 a month
#   cloudflare  @cf/baai/bge-reranker-base on Workers AI, through the same route, gateway and neuron
#               ledger as Clef (clef.post_run): about 6 neurons for a 60-chunk topic (A §3.2)
# RERANK_ENGINE picks the first; the other one is the fallback; when neither can run, rerank() returns
# None and retrieval keeps the FAISS order. It never raises: a reranker can't stop an answer.
# Only the ORDER is used by the app, never a score threshold: the engines' scales differ (A §3.4).

load_dotenv()

MODES = ("off", "on")
ENGINES = ("cohere", "cloudflare")
DEFAULT_MODE = "off"                 # on only after Section E's evals show a gain
DEFAULT_ENGINE = "cloudflare"        # free with no terms on its use; Section E decides (#29)
DEFAULT_COHERE_MODEL = "rerank-v4.0-pro"     # the most consistent in A §3.5; Section E decides

COHERE_URL = "https://api.cohere.com/v2/rerank"
CLOUDFLARE_MODEL = "@cf/baai/bge-reranker-base"
# Per call. Measured 1.1-1.7 s (Cohere) and 1.8-2.9 s (Cloudflare, with a cold connection), A §3.3.
TIMEOUT_SECONDS = 5.0
# bge-reranker-base reads about 512 tokens of query + chunk and cuts the rest silently (A §3.2: a
# 26,799-character chunk cost the neurons of ~512 tokens). At ~4 characters a token that's ~2,000
# characters. Today's chunks are at most 1,500 (rag_service.text_splitter), so this only matters if
# chunk_size grows: then Cloudflare isn't used, rather than ranking chunks by their first half.
CLOUDFLARE_MAX_CHARS = 2000
# Cohere takes up to 1,000 documents a call; a topic's index holds 7-60 chunks
COHERE_MAX_DOCUMENTS = 1000
# Like clef.py: after this many timeouts or busy answers in a row, the engine is skipped for a while
BACKOFF_AFTER_FAILURES = 3
BACKOFF_SECONDS = 30
KEEPALIVE_SECONDS = 300

# --- Free limits (Section C) ---
# Cohere's trial key: 10 rerank calls a minute and 1,000 calls a month (A §3.1, from its reply headers).
# The app stops a little below both, so a probe or an eval run on the same key in the same minute or
# month doesn't push the live app into Cohere's own 429: past either, Cloudflare ranks instead (no
# error, no wait). COHERE_MONTHLY_CALLS / COHERE_PER_MINUTE are the key's limits (another key may differ).
DEFAULT_COHERE_MONTHLY_CALLS = 1000
DEFAULT_COHERE_PER_MINUTE = 10
COHERE_MONTH_SHARE = 0.95          # the app stops at 950 of 1,000: the last 50 stay for probes and evals
COHERE_MINUTE_MARGIN = 2           # the app sends at most 8 of 10 in any 60 s
COHERE_WARN_SHARE = 0.80           # one [WARN] a day from 800 of 1,000
COHERE_PAUSE_SECONDS = 60          # after Cohere's own 429, or its header saying no calls are left this minute
# Cloudflare's reranker share of the day's 10,000 free neurons (shared with Clef, whose checks are the
# app's safety): measured ~2 neurons for 18 chunks and ~6 for 60 (A §3.2), so 300 is ~50-150 fresh topics.
# Past it the reranker stops using Cloudflare for the day; Clef keeps everything else.
DEFAULT_CLOUDFLARE_NEURONS_PER_DAY = 300


class RerankError(Exception):
    """
    A call that gave no usable scores. `kind`: not_configured, timeout, busy (5xx or a network error),
    rate_limited (Cohere's 429: the trial's per-minute or monthly limit), auth (401/403), billing (402),
    bad_request (400/413/422), bad_response, too_long (a chunk over Cloudflare's limit), over_budget
    (the day's neurons, clef.over_budget), backoff, over_limit (Section C: the app's share of Cohere's
    minute or month, or of Cloudflare's neurons). Every kind sends the call to the next engine.
    """
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


@dataclass
class Reranked:
    """One score per text, in the order the texts were given, all from one engine (scales differ)."""
    scores: list[float]
    engine: str
    model: str
    from_cache: int = 0                              # texts whose score was read from Redis
    failed: list[str] = field(default_factory=list)  # engines tried first that couldn't run, with why


# --- Settings ---
_settings = {"mode": DEFAULT_MODE, "engine": DEFAULT_ENGINE, "cohere_model": DEFAULT_COHERE_MODEL, "cohere_key": "",
             "cohere_monthly": DEFAULT_COHERE_MONTHLY_CALLS, "cohere_per_minute": DEFAULT_COHERE_PER_MINUTE,
             "cloudflare_neurons": DEFAULT_CLOUDFLARE_NEURONS_PER_DAY}
_warnings: list[str] = []


def _positive_int(environ, name: str, default: int) -> int:
    raw = (environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    _warnings.append(f"{name}={raw!r} isn't a whole number above 0; using {default}")
    return default


def configure(environ=None) -> dict:
    """Reads RERANK_MODE, RERANK_ENGINE, RERANK_COHERE_MODEL, COHERE_API_KEY and the limits
    (COHERE_MONTHLY_CALLS, COHERE_PER_MINUTE, RERANK_CLOUDFLARE_NEURONS). A value that isn't one of
    the choices warns once and the default applies. Returns the settings (without the key)."""
    environ = os.environ if environ is None else environ
    _warnings.clear()
    mode = (environ.get("RERANK_MODE") or "").strip().lower() or DEFAULT_MODE
    if mode not in MODES:
        _warnings.append(f"RERANK_MODE={mode!r} isn't one of {', '.join(MODES)}; using {DEFAULT_MODE}")
        mode = DEFAULT_MODE
    engine = (environ.get("RERANK_ENGINE") or "").strip().lower() or DEFAULT_ENGINE
    if engine not in ENGINES:
        _warnings.append(f"RERANK_ENGINE={engine!r} isn't one of {', '.join(ENGINES)}; using {DEFAULT_ENGINE}")
        engine = DEFAULT_ENGINE
    _settings.update(mode=mode, engine=engine,
                     cohere_model=(environ.get("RERANK_COHERE_MODEL") or "").strip() or DEFAULT_COHERE_MODEL,
                     cohere_key=(environ.get("COHERE_API_KEY") or "").strip(),
                     cohere_monthly=_positive_int(environ, "COHERE_MONTHLY_CALLS", DEFAULT_COHERE_MONTHLY_CALLS),
                     cohere_per_minute=_positive_int(environ, "COHERE_PER_MINUTE", DEFAULT_COHERE_PER_MINUTE),
                     cloudflare_neurons=_positive_int(environ, "RERANK_CLOUDFLARE_NEURONS", DEFAULT_CLOUDFLARE_NEURONS_PER_DAY))
    for line in _warnings:
        print(f"[WARN] Rerank: {line}.")
    return {k: v for k, v in _settings.items() if k != "cohere_key"}


def enabled() -> bool:
    return _settings["mode"] == "on"


def engine() -> str:
    return _settings["engine"]


def cohere_available() -> bool:
    return bool(_settings["cohere_key"])


def model_of(name: str) -> str:
    return _settings["cohere_model"] if name == "cohere" else CLOUDFLARE_MODEL


def summary() -> dict:
    """What decides the reranking, for rag_service's config version (feature 2 E): a change is a new version."""
    return {"mode": _settings["mode"], "engine": _settings["engine"], "cohere_model": _settings["cohere_model"],
            "cloudflare_model": CLOUDFLARE_MODEL}


def engines() -> list[str]:
    """The engines to try, in order: the chosen one, then the other, skipping one that isn't set up."""
    order = [_settings["engine"]] + [e for e in ENGINES if e != _settings["engine"]]
    ready = {"cohere": cohere_available(), "cloudflare": clef.available()}
    return [e for e in order if ready[e]]


def status() -> str:
    """For the startup line."""
    order = engines()
    if not order:
        return f"Rerank {_settings['mode']}: no engine set up (COHERE_API_KEY, or the Cloudflare settings Clef uses)"
    names = [f"{e} ({model_of(e)})" for e in order]
    line = f"Rerank {_settings['mode']}: {names[0]}" + (f", fallback {names[1]}" if len(names) > 1 else ", no fallback")
    usage = []
    if "cohere" in order:
        usage.append(f"Cohere {cohere_calls_this_month()}/{_settings['cohere_monthly']:,} calls this month "
                     f"(the app stops at {cohere_month_max():,})")
    if "cloudflare" in order:
        usage.append(f"Cloudflare {cloudflare_neurons_today():.0f}/{_settings['cloudflare_neurons']} rerank neurons today")
    return line + (f"; {', '.join(usage)}" if usage else "")


configure()


# --- Back-off, per engine ---
_trouble = {name: {"failures": 0, "skip_until": 0.0} for name in ENGINES}


def backing_off(name: str) -> bool:
    return time.monotonic() < _trouble[name]["skip_until"]


def _call_failed(name: str, kind: str):
    if kind not in ("timeout", "busy"):
        return
    trouble = _trouble[name]
    trouble["failures"] += 1
    if trouble["failures"] >= BACKOFF_AFTER_FAILURES and time.monotonic() >= trouble["skip_until"]:
        trouble["skip_until"] = time.monotonic() + BACKOFF_SECONDS
        print(f"[WARN] Rerank: {name} failed {trouble['failures']} times in a row ({kind}). "
              f"Skipping it for {BACKOFF_SECONDS}s.")


def _call_worked(name: str):
    _trouble[name]["failures"] = 0


# --- Free limits (Section C #13-#15) ---
def utc_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m")


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _next_month_start(month: str) -> str:
    year, number = (int(x) for x in month.split("-"))
    return f"{year + number // 12}-{number % 12 + 1:02d}-01"


def cohere_month_max() -> int:
    return max(1, int(_settings["cohere_monthly"] * COHERE_MONTH_SHARE))


def cohere_minute_max() -> int:
    return max(1, _settings["cohere_per_minute"] - COHERE_MINUTE_MARGIN)


# Month: counted in Redis (v1:usage:cohere:<YYYY-MM>) BEFORE each call, like GNews's searches (feature 3
# #92), so two requests can't both take the last one, and a restart, the evals and the probe all add to
# the same count. Without Redis the count is this process's own.
_cohere_month = {"month": None, "sent": 0}
# Minute: this process's calls in the last 60 s (one uvicorn worker on Render), plus a pause after
# Cohere's own 429 or its header saying none are left (that covers other users of the key too).
_cohere_minute: deque = deque()
_cohere_pause = {"until": 0.0}
# Cloudflare's reranker neurons today, in this process; startup reads Redis's count (load_usage)
_cloudflare_day = {"day": None, "neurons": 0.0}
# Each [WARN] once a day
_said: dict[str, str] = {}


def _say_once_a_day(key: str, line: str):
    day = _utc_day()
    if _said.get(key) != day:
        _said[key] = day
        print(line)


def _month_state() -> dict:
    month = utc_month()
    if _cohere_month["month"] != month:
        _cohere_month.update(month=month, sent=0)
    return _cohere_month


def cohere_calls_this_month() -> int:
    return _month_state()["sent"]


def cohere_calls_last_minute() -> int:
    cutoff = time.monotonic() - 60
    while _cohere_minute and _cohere_minute[0] < cutoff:
        _cohere_minute.popleft()
    return len(_cohere_minute)


def cloudflare_neurons_today() -> float:
    return _cloudflare_day["neurons"] if _cloudflare_day["day"] == _utc_day() else 0.0


def _pause_cohere(why: str):
    _cohere_pause["until"] = time.monotonic() + COHERE_PAUSE_SECONDS
    _say_once_a_day("cohere_pause", f"[WARN] Rerank: Cohere {why}. Not calling it for {COHERE_PAUSE_SECONDS}s "
                                    "(Cloudflare ranks meanwhile). Said once a day.")


def _month_used_up(state: dict):
    observability.add_totals(rerank_limited=1)
    _say_once_a_day("cohere_month", f"[WARN] Rerank: the app's share of Cohere's month is used ({state['sent']} of "
                                    f"{_settings['cohere_monthly']:,} calls). Cloudflare ranks until "
                                    f"{_next_month_start(state['month'])}.")


async def _reserve_cohere_call():
    """Counts one Cohere call before it's sent. Raises RerankError("over_limit") past the app's share of the
    minute or the month: the call isn't sent and the next engine ranks."""
    now = time.monotonic()
    if now < _cohere_pause["until"]:
        raise RerankError("over_limit", f"paused for {_cohere_pause['until'] - now:.0f}s more after Cohere's rate limit")
    minute = cohere_calls_last_minute()
    if minute >= cohere_minute_max():
        observability.add_totals(rerank_limited=1)
        raise RerankError("over_limit", f"{minute} Cohere calls in the last minute (the app sends at most {cohere_minute_max()})")
    state = _month_state()
    if state["sent"] >= cohere_month_max():
        _month_used_up(state)
        raise RerankError("over_limit", f"{state['sent']} Cohere calls this month (the app stops at {cohere_month_max()})")
    key = cache_service.cohere_month_key(state["month"])
    total = await cache_service.incr_counter(key, cache_service.COHERE_MONTH_TTL)
    state["sent"] = total if total is not None else state["sent"] + 1
    if state["sent"] > cohere_month_max():
        # Another worker or an eval run took the last one meanwhile: not sent, the reservation given back
        if total is not None:
            await cache_service.incr_counter(key, cache_service.COHERE_MONTH_TTL, -1)
        state["sent"] -= 1
        _month_used_up(state)
        raise RerankError("over_limit", f"{state['sent']} Cohere calls this month (the app stops at {cohere_month_max()})")
    _cohere_minute.append(time.monotonic())
    if state["sent"] >= int(_settings["cohere_monthly"] * COHERE_WARN_SHARE):
        _say_once_a_day("cohere_warn", f"[WARN] Rerank: {state['sent']} of {_settings['cohere_monthly']:,} Cohere calls "
                                       f"used this month; from {cohere_month_max():,} Cloudflare ranks until "
                                       f"{_next_month_start(state['month'])}.")


def _check_cloudflare_share():
    used = cloudflare_neurons_today()
    if used >= _settings["cloudflare_neurons"]:
        observability.add_totals(rerank_limited=1)
        _say_once_a_day("cloudflare_share", f"[WARN] Rerank: {used:.0f} of the reranker's {_settings['cloudflare_neurons']} "
                                            "Cloudflare neurons used today; it stops using Cloudflare until 00:00 UTC "
                                            "(Clef's checks keep the rest).")
        raise RerankError("over_limit", f"{used:.0f} rerank neurons today (share {_settings['cloudflare_neurons']})")


def _add_cloudflare_neurons(neurons: float):
    day = _utc_day()
    if _cloudflare_day["day"] != day:
        _cloudflare_day.update(day=day, neurons=0.0)
    _cloudflare_day["neurons"] += neurons


async def load_usage():
    """At startup: this month's Cohere calls and today's rerank neurons from Redis, so a restart remembers them."""
    state = _month_state()
    total = await cache_service.get_counter(cache_service.cohere_month_key(state["month"]))
    if total:
        state["sent"] = max(state["sent"], total)
    clef_day = await cache_service.get_usage("clef", _utc_day())
    if clef_day and clef_day.get("rerank_neurons_x100"):
        _cloudflare_day.update(day=_utc_day(),
                               neurons=max(cloudflare_neurons_today(), clef_day["rerank_neurons_x100"] / 100))


# --- The Cohere client: one per process, so the connection is reused ---
_client: httpx.AsyncClient | None = None
_transport = None             # tests put a fake server here (httpx.MockTransport)


def use_transport(transport):
    """Tests: send Cohere calls to `transport` instead of the network (None goes back to it)."""
    global _client, _transport
    _client, _transport = None, transport


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(transport=_transport, limits=httpx.Limits(keepalive_expiry=KEEPALIVE_SECONDS))
    return _client


def _cohere_error(response: httpx.Response) -> RerankError:
    code = response.status_code
    detail = f"HTTP {code}"
    try:
        message = response.json().get("message")
        if message:
            detail += ": " + str(message)[:200]
    except ValueError:
        pass
    if code == 429:
        return RerankError("rate_limited", detail)
    if code in (401, 403):
        return RerankError("auth", detail)
    if code == 402:
        return RerankError("billing", detail)
    if code >= 500:
        return RerankError("busy", detail)
    return RerankError("bad_request", detail)


def _scores_from(items, count: int, index_key: str, score_key: str) -> list[float]:
    """The reply's items mapped back to the texts' order. Every text must get exactly one score."""
    scores: list[float | None] = [None] * count
    try:
        for item in items:
            i = int(item[index_key])
            if not 0 <= i < count or scores[i] is not None:
                raise RerankError("bad_response", f"index {i} out of range or repeated")
            scores[i] = float(item[score_key])
    except (KeyError, TypeError, ValueError) as e:
        raise RerankError("bad_response", f"{type(e).__name__} in the scores") from e
    if any(s is None for s in scores):
        raise RerankError("bad_response", f"{sum(s is None for s in scores)} of {count} texts got no score")
    return scores


async def _cohere(query: str, texts: list[str], run) -> list[float]:
    if not cohere_available():
        raise RerankError("not_configured", "COHERE_API_KEY not set")
    if len(texts) > COHERE_MAX_DOCUMENTS:
        raise RerankError("bad_request", f"{len(texts)} documents; Cohere takes {COHERE_MAX_DOCUMENTS}")
    await _reserve_cohere_call()
    payload = {"model": _settings["cohere_model"], "query": query, "documents": texts, "top_n": len(texts)}
    headers = {"Authorization": f"Bearer {_settings['cohere_key']}", "Content-Type": "application/json",
               "Accept": "application/json"}
    try:
        # httpx's timeout is per phase (connect, read...); wait_for bounds the whole call
        response = await asyncio.wait_for(_http().post(COHERE_URL, json=payload, headers=headers, timeout=TIMEOUT_SECONDS),
                                          TIMEOUT_SECONDS)
    except (httpx.TimeoutException, asyncio.TimeoutError) as e:
        raise RerankError("timeout", type(e).__name__) from e
    except httpx.TransportError as e:
        raise RerankError("busy", type(e).__name__) from e
    # What's left of the trial's minute (Cohere's header), on the span for Section C's limiter
    remaining = response.headers.get("x-trial-endpoint-call-remaining")
    run.add_metadata({"cohere_month_calls": cohere_calls_this_month()})
    if remaining is not None:
        run.add_metadata({"cohere_minute_calls_left": remaining})
        if remaining.strip() == "0":
            _pause_cohere("says no calls are left this minute")
    if response.status_code != 200:
        error = _cohere_error(response)
        if error.kind == "rate_limited":
            _pause_cohere("answered 429 (rate limit)")
        raise error
    try:
        body = response.json()
    except ValueError as e:
        raise RerankError("bad_response", "the reply isn't JSON") from e
    run.add_metadata({"cohere_billed_units": (body.get("meta") or {}).get("billed_units")})
    return _scores_from(body.get("results"), len(texts), "index", "relevance_score")


async def _cloudflare(query: str, texts: list[str], run) -> list[float]:
    if not clef.available():
        raise RerankError("not_configured", clef.status())
    if clef.over_budget():
        raise RerankError("over_budget", f"{clef.neurons_today():.0f} neurons used today")
    _check_cloudflare_share()
    longest = max(len(t) for t in texts)
    if len(query) + longest > CLOUDFLARE_MAX_CHARS:
        raise RerankError("too_long", f"a chunk of {longest} characters; bge-reranker-base reads ~{CLOUDFLARE_MAX_CHARS}")
    payload = {"model": CLOUDFLARE_MODEL,
               "input": {"query": query, "contexts": [{"text": t} for t in texts], "top_k": len(texts)}}
    try:
        response, neurons, attempts = await clef.post_run(payload, TIMEOUT_SECONDS, run,
                                                          skip=lambda: backing_off("cloudflare"), usage="rerank")
    except clef.ClefError as e:
        # ClefError's text is "<kind>: <detail>"; the detail alone, so the kind isn't said twice
        detail = str(e)[len(e.kind) + 2:] if str(e).startswith(f"{e.kind}: ") else ""
        raise RerankError(e.kind, detail) from e
    _add_cloudflare_neurons(neurons)
    run.add_metadata({"neurons": neurons, "attempts": attempts, "cf_aig_log_id": response.headers.get("cf-aig-log-id"),
                      "rerank_neurons_today": round(cloudflare_neurons_today(), 2)})
    try:
        result = response.json()["result"]
        items = result["response"] if isinstance(result, dict) else result
    except (ValueError, KeyError, TypeError) as e:
        raise RerankError("bad_response", f"no result in the reply ({type(e).__name__})") from e
    return _scores_from(items, len(texts), "id", "score")


_CALLS = {"cohere": _cohere, "cloudflare": _cloudflare}


def outgoing_query(query: str) -> str:
    """The query sent to a reranker: one line, and without personal data while the PII guard is on
    (feature 3 #58: personal data never reaches Clef or the gateway's logs; nor Cohere)."""
    text = " ".join(str(query or "").split())
    if privacy.active():
        text = " ".join(pii.remove(text)[0].split())
    return text


async def _with(name: str, query: str, texts: list[str], run) -> tuple[list[float], int]:
    """Scores from one engine: the cached ones, then one call for the rest (each distinct text once).
    Returns (scores, how many came from the cache). Raises RerankError."""
    model = model_of(name)
    cached = await cache_service.get_rerank_scores(model, query, texts)
    missing = list(dict.fromkeys(t for t, s in zip(texts, cached) if s is None))
    if not missing:
        return cached, len(texts)
    if backing_off(name):
        raise RerankError("backoff", f"skipped for up to {BACKOFF_SECONDS}s after repeated failures")
    try:
        new = await _CALLS[name](query, missing, run)
    except RerankError as e:
        _call_failed(name, e.kind)
        raise
    _call_worked(name)
    scored = dict(zip(missing, new))
    await cache_service.set_rerank_scores(model, query, scored)
    observability.add_totals(rerank_calls=1)
    hits = sum(1 for s in cached if s is not None)
    return [s if s is not None else scored[t] for t, s in zip(texts, cached)], hits


async def rerank(query: str, texts: list[str], label: str = "") -> Reranked | None:
    """
    One relevance score per text, from the first engine that can give them all (engines()), or None
    when none can: the caller keeps its own order. Never raises. Doesn't check RERANK_MODE: the caller
    does (enabled()), so the evals can rerank with the mode off.
    """
    if not texts:
        return None
    started = time.perf_counter()
    failed: list[str] = []
    try:
        sent = outgoing_query(query)
        order = engines()
        async with observability.span("rerank", run_type="tool",
                                      inputs={"label": label, "documents": len(texts), "query_chars": len(sent)},
                                      metadata={"engines": order, "models": [model_of(e) for e in order]}) as run:
            result = None
            if sent:
                for name in order:
                    try:
                        scores, hits = await _with(name, sent, texts, run)
                    except RerankError as e:
                        failed.append(f"{name}: {e}")
                        continue
                    result = Reranked(scores=scores, engine=name, model=model_of(name), from_cache=hits, failed=failed)
                    break
            elif order:
                failed.append("query: nothing left to send after removing personal data")
            ms = round((time.perf_counter() - started) * 1000)
            run.add_outputs({"engine": result.engine if result else None, "from_cache": result.from_cache if result else 0,
                             "failed": failed, "ms": ms})
    except Exception as e:            # a bug or Redis trouble here must not stop the answer
        failed.append(f"error: {type(e).__name__}")
        result, ms = None, round((time.perf_counter() - started) * 1000)

    if failed:
        print(f"[WARN] Rerank{f' ({label})' if label else ''}: " + "; ".join(failed)
              + (f". Used {result.engine}." if result else ". Kept the search order."))
    if result is None:
        observability.add_totals(rerank_failed=1)
        return None
    observability.add_totals(rerank_engine=result.engine)
    print(f"[INFO] Rerank{f' ({label})' if label else ''}: {len(texts)} chunks by {result.engine} ({result.model}), "
          f"{result.from_cache} from cache, {ms} ms.")
    return result
