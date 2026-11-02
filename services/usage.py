import asyncio
import os
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from langchain_core.callbacks import AsyncCallbackHandler

from services import cache_service, errors, observability
from services.safety import gemini, limits

# --- Tokens, cost and the daily quota (feature 2, Section D) ---
# Every Gemini attempt is counted here: chat calls through a LangChain callback on both chat
# models (it works with tracing off too), embedding calls from rag_service.call_with_retry. Each
# one adds to the request's totals (observability.add_totals), to a per-model, per-day counter in
# Redis, and to one log line. A model whose free daily quota is used up is skipped until its
# reset (#34).

# --- Prices (#32): USD per 1M tokens; each row applies from its date ---
# Checked 2026-10-19 on ai.google.dev/gemini-api/docs/pricing (Section A1 §6). Output includes
# thinking tokens, which LangChain already counts in output_tokens. The free tier costs $0: these
# are what the same traffic would cost on the paid tier. LangSmith prices LLM runs with its own map
# (same 2026 prices, A2 §4); this table is for the log line and the request totals.
PRICES_CHECKED = "2026-10-19"
MODEL_PRICES = {
    "gemini-3.8-flash": [("2026-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    "gemini-3.6-flash": [("2026-01-01", 0.75, 3.75), ("2027-01-01", 1.50, 7.50)],
    # No longer on the pricing page; its GA price (A1 §6). Input only.
    "gemini-embedding-001": [("2025-07-14", 0.15, 0.0)],
}


def prices_for(model: str | None, on: date | None = None) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens for a model on a day, or None for a model not in the table.
    'models/gemini-3.6-flash-07-2026' matches 'gemini-3.6-flash' (longest name that fits)."""
    name = (model or "").removeprefix("models/")
    keys = [key for key in MODEL_PRICES if name == key or name.startswith(key + "-")]
    if not keys:
        return None
    day = (on or datetime.now(timezone.utc).date()).isoformat()
    current = [row for row in MODEL_PRICES[max(keys, key=len)] if row[0] <= day]
    return (current[-1][1], current[-1][2]) if current else None


def cost_usd(model, input_tokens: int, output_tokens: int, on: date | None = None) -> float | None:
    prices = prices_for(model, on)
    if prices is None:
        return None
    return round((input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000, 8)


# --- Embedding tokens (#35) ---
# The embeddings API returns no token count. Measured with countTokens on 56 chunks of the saved
# snapshots (2026-10-19): 67,666 characters = 13,660 tokens, 4.95 characters per token (median
# 4.95). Estimates are marked as such wherever they appear.
EMBED_CHARS_PER_TOKEN = 4.95


def estimate_tokens(characters: int) -> int:
    return round(characters / EMBED_CHARS_PER_TOKEN)


# --- The daily quota (#33, #34) ---
# The free tier allows GEMINI_DAILY_LIMIT chat requests per model per day (20, seen 2026-10-11),
# reset at midnight Pacific time.
QUOTA_TZ = ZoneInfo("America/Los_Angeles")
DAILY_LIMIT = int(os.getenv("GEMINI_DAILY_LIMIT", "20"))
WARN_AT = max(1, DAILY_LIMIT - 4)


def quota_day(now: datetime | None = None) -> str:
    """Today's date where the quota resets (Pacific), e.g. '2026-10-19'."""
    return (now or datetime.now(timezone.utc)).astimezone(QUOTA_TZ).date().isoformat()


def seconds_to_reset(now: datetime | None = None) -> int:
    """Seconds until the next midnight Pacific time."""
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(QUOTA_TZ)
    midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tzinfo=QUOTA_TZ)
    # Subtracted in UTC: two datetimes with the same tzinfo subtract by wall clock, which is an
    # hour off on the nights daylight saving starts or ends (the test caught it on 2026-11-01)
    return max(1, int((midnight.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds()))


def used_today(counts: dict | None) -> int | None:
    """Requests that count against the quota: every attempt except the ones refused for the daily limit."""
    if counts is None:
        return None
    return counts.get("attempts", 0) - counts.get("daily_limit", 0)


# Models whose daily quota is used up: model -> time.time() when Gemini said it resets. In this
# process only: after a restart the first call finds out again (one quick 429).
_exhausted_until: dict[str, float] = {}


def exhausted(model: str) -> int | None:
    """Seconds until the model's daily quota resets, if it's used up; else None."""
    until = _exhausted_until.get(model)
    if until is None:
        return None
    left = int(until - time.time())
    if left <= 0:
        _exhausted_until.pop(model, None)
        return None
    return left


def mark_exhausted(model: str, seconds: int | None):
    seconds = seconds or seconds_to_reset()
    _exhausted_until[model] = time.time() + seconds
    hours = seconds / 3600
    print(f"[WARN] {model}: today's free requests are used up. Skipping it for {hours:.1f} h, until its reset.")


# --- Counting in the background ---
# The Redis write and the log line that shows the day's count run after the call, as a task, so
# the request doesn't wait for Redis (Upstash: tens of ms per write).
_pending: set[asyncio.Task] = set()


def _background(coro):
    try:
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:          # no event loop (a sync call): count nothing rather than fail
        coro.close()
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def drain():
    """Waits for the pending counter writes (tests, shutdown)."""
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


async def _count(model: str, counts: dict):
    """Adds to the day's counter without a log line (an attempt being sent)."""
    try:
        await cache_service.add_usage(model, quota_day(), counts)
    except Exception as e:
        print(f"[WARN] Couldn't count Gemini usage: {type(e).__name__}: {e}")


async def _count_and_log(model: str, counts: dict, line: str, chat: bool):
    try:
        totals = await cache_service.add_usage(model, quota_day(), counts)
    except Exception as e:                    # never let counting fail anything
        totals = None
        print(f"[WARN] Couldn't count Gemini usage: {type(e).__name__}: {e}")
    used = used_today(totals)
    if not chat and used is not None and model == EMBEDDING_MODEL:
        _adopt_embedding_day(used)
    if chat:
        today = f"{used}/{DAILY_LIMIT}" if used is not None else "?"
    else:
        today = str(used) if used is not None else "?"
    print(f"{line}, calls today {today}")
    if chat and used is not None and used >= WARN_AT:
        print(f"[WARN] {model}: {used} of {DAILY_LIMIT} free requests used today "
              f"(resets at midnight Pacific, in {seconds_to_reset() / 3600:.1f} h).")


def _error_kind(error) -> str:
    """The counter field for a failed attempt."""
    if errors.is_daily_quota_error(error):
        return "daily_limit"
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    api = errors.api_error(error)
    return f"error_{api.code}" if api is not None else "error_other"


# --- Chat calls (#31) ---
VIEWS = ("report", "timeline", "contradictions", "update")


class UsageHandler(AsyncCallbackHandler):
    """
    Attached to each chat model (rag_service._chat_model), so it sees every attempt, retries
    included: on_chat_model_start when a request is sent, then on_llm_end or on_llm_error.
    `role` is "primary" or "fallback", for the log line.
    """

    _all: list = []          # every handler, so a cancelled run's entry can be forgotten (forget)

    def __init__(self, role: str):
        self.role = role
        self._started: dict = {}
        UsageHandler._all.append(self)

    async def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs):
        model = (metadata or {}).get("ls_model_name") or ((serialized or {}).get("kwargs") or {}).get("model") or "?"
        view = next((tag for tag in (tags or []) if tag in VIEWS), "chat")
        model = str(model).removeprefix("models/")
        self._started[run_id] = (time.perf_counter(), model, view)
        observability.llm_run_started(run_id)
        # Counted when it's sent, not when it ends: an attempt the 75 s deadline cuts off never
        # ends normally (no on_llm_end / on_llm_error), but Google has received it (Section G)
        observability.add_totals(chat_calls=1)
        _background(_count(model, {"attempts": 1}))

    async def on_llm_end(self, response, *, run_id, **kwargs):
        observability.llm_run_ended(run_id)
        started, model, view = self._started.pop(run_id, (time.perf_counter(), "?", "chat"))
        seconds = time.perf_counter() - started
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError):
            message = None
        usage = getattr(message, "usage_metadata", None) or {}
        metadata = getattr(message, "response_metadata", None) or {}
        answered = metadata.get("model_name") or model
        input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        thinking = (usage.get("output_token_details") or {}).get("reasoning", 0)
        cost = cost_usd(model, input_tokens, output_tokens)
        observability.add_totals(input_tokens=input_tokens, output_tokens=output_tokens,
                                 thinking_tokens=thinking, cost_usd=cost or 0, answered_by=answered)
        # Gemini's safety signals (feature 3 #46, A1 §3.2): finish_reason and the ratings on the message,
        # a blocked prompt's feedback in llm_output. Gemini 3 sends no ratings unless something is
        # blocked, so they're recorded only when there are some. The LLM run in LangSmith shows the
        # whole response_metadata already; these go on the request's totals (and the root's metadata).
        finish = metadata.get("finish_reason")
        block = gemini.block_of(message, getattr(response, "llm_output", None))
        ratings = gemini.ratings_text(metadata.get("safety_ratings"))
        if finish:
            observability.add_totals(finish_reason=finish)
        if ratings:
            observability.add_totals(gemini_ratings=ratings)
        counts = {"ok": 1, "input_tokens": input_tokens, "output_tokens": output_tokens, "thinking_tokens": thinking}
        # The visitor's daily share (feature 3 #89): answers, not attempts, so Google's busy errors
        # aren't held against them. Read from this request's context (limits.ClientLimits).
        _background(limits.count_chat(limits.current_client()))
        if block is not None:
            observability.add_totals(gemini_blocks=1, gemini_block=block.label())
            counts["blocked"] = 1
            # LangChain reads usage only from the candidates, so a blocked prompt (no candidates) has
            # no token counts at all, though Google counted its input (A1 §3.2)
            tokens = ("in ? (not reported for a blocked prompt)" if block.stage == "prompt" and not input_tokens
                      else f"in {input_tokens:,} · out {output_tokens:,} tokens")
            line = (f"[WARN] LLM {view}: {answered} BLOCKED by Gemini's filter ({block.label()}), {tokens}, {seconds:.1f} s, "
                    f"fallback {'yes' if self.role == 'fallback' else 'no'}")
        else:
            line = (f"[INFO] LLM {view}: {answered} in {input_tokens:,} · thinking {thinking:,} · out {output_tokens:,} tokens, "
                    f"{seconds:.1f} s, " + (f"~${cost:.4f}" if cost is not None else "price unknown")
                    + f", fallback {'yes' if self.role == 'fallback' else 'no'}"
                    + (f", finish {finish}" if finish and finish != "STOP" else ""))
        _background(_count_and_log(model, counts, line, chat=True))

    async def on_llm_error(self, error, *, run_id, **kwargs):
        observability.llm_run_ended(run_id)
        started, model, view = self._started.pop(run_id, (time.perf_counter(), "?", "chat"))
        kind = _error_kind(error)
        if kind == "daily_limit":
            mark_exhausted(model, errors.retry_after_seconds(error))
        observability.add_totals(chat_errors=1)
        line = f"[WARN] LLM {view}: {model} attempt failed ({kind}) after {time.perf_counter() - started:.1f} s"
        _background(_count_and_log(model, {kind: 1}, line, chat=True))


def forget(run_ids):
    """Attempts a deadline cancelled: no end callback will come, so their start entries are dropped."""
    for handler in UsageHandler._all:
        for run_id in run_ids:
            handler._started.pop(run_id, None)


# --- Embedding calls (#35) ---

def record_embedding(model: str, characters: int, texts: int, error=None):
    """One embedding attempt (rag_service.call_with_retry): estimated tokens and cost, counted per day."""
    _embedding_sent(texts)
    tokens = estimate_tokens(characters)
    if error is None:
        cost = cost_usd(model, tokens, 0) or 0
        observability.add_totals(embedding_calls=1, embedding_tokens_est=tokens, embedding_cost_est=cost)
        counts = {"attempts": 1, "ok": 1, "estimated_tokens": tokens}
        line = f"[INFO] Embedding: {model} {texts} text(s), ~{tokens:,} tokens (estimated), ~${cost:.5f}"
    else:
        kind = _error_kind(error)
        observability.add_totals(embedding_calls=1, embedding_errors=1)
        counts = {"attempts": 1, kind: 1}
        line = f"[WARN] Embedding: {model} attempt failed ({kind})"
    _background(_count_and_log(model, counts, line, chat=False))
    return tokens


# --- The embedding budget (feature 3, Section L #93) ---
# gemini-embedding-001's free tier (A1 §4.1): 100 TEXTS a minute (each text of a batch counts) and 100
# REQUESTS a day (a batch counts once), reset at midnight Pacific. Retrieval needs about 3 requests per
# fresh topic. Safety's embedding calls (the fallback classifier, J's comparison sample, the coverage and
# leak fallbacks) go through rag_service with purpose="safety" and may use only what retrieval leaves:
# a safety call is refused (EmbeddingBudget) when it would take the minute past SAFETY_EMBED_MINUTE_MAX
# texts, or when the day already has SAFETY_EMBED_DAY_MAX requests. The check then fails open (#18).
# Vectors already in the cache cost nothing and are never refused.
EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_TEXTS_PER_MINUTE = 100
EMBED_REQUESTS_PER_DAY = 100
SAFETY_EMBED_MINUTE_MAX = int(os.getenv("SAFETY_EMBED_MINUTE_MAX", "50"))    # half of the minute stays retrieval's
SAFETY_EMBED_DAY_MAX = int(os.getenv("SAFETY_EMBED_DAY_MAX", "80"))          # the day's last 20 requests too

# Texts sent in the last minute, in this process (one uvicorn worker on Render): (monotonic time, texts)
_embed_minute: deque = deque()
_embed_day = {"day": None, "requests": 0}


class EmbeddingBudget(Exception):
    """A safety embedding call that would eat into what retrieval needs. The check fails open."""


def _embedding_sent(texts: int):
    _embed_minute.append((time.monotonic(), texts))
    day = quota_day()
    if _embed_day["day"] != day:
        _embed_day.update(day=day, requests=0)
    _embed_day["requests"] += 1


def _adopt_embedding_day(used: int):
    """Redis's count for the day, when it's higher (a restart, scripts and evals on the same key)."""
    day = quota_day()
    if _embed_day["day"] != day:
        _embed_day.update(day=day, requests=0)
    _embed_day["requests"] = max(_embed_day["requests"], used)


def embedding_texts_last_minute() -> int:
    cutoff = time.monotonic() - 60
    while _embed_minute and _embed_minute[0][0] < cutoff:
        _embed_minute.popleft()
    return sum(n for _, n in _embed_minute)


def embedding_requests_today() -> int:
    return _embed_day["requests"] if _embed_day["day"] == quota_day() else 0


def safety_embedding_refusal(texts: int) -> str | None:
    """Why a safety call embedding `texts` new texts can't be sent now, or None when it can."""
    minute = embedding_texts_last_minute()
    if minute + texts > SAFETY_EMBED_MINUTE_MAX:
        return f"{minute} texts embedded in the last minute + {texts} would pass safety's {SAFETY_EMBED_MINUTE_MAX}"
    day = embedding_requests_today()
    if day >= SAFETY_EMBED_DAY_MAX:
        return f"{day} embedding requests today (safety stops at {SAFETY_EMBED_DAY_MAX} of {EMBED_REQUESTS_PER_DAY})"
    return None


# --- GNews searches (feature 3, Section L #92) ---
# The free plan allows 100 searches a day, reset at 00:00 UTC. Each search is counted in Redis
# (v1:usage:gnews:<UTC day>) BEFORE it's sent: the count is reserved first, so two builds can't both
# take the 100th. Past the limit the search isn't sent, and the user gets GNews's own daily-limit
# message at once (with Retry-After). A 403 from GNews (its own count, which also has other users of
# the key) marks the day used up in this process. Without Redis the count is this process's own.
GNEWS = "gnews"
GNEWS_DAILY_LIMIT = int(os.getenv("GNEWS_DAILY_LIMIT", "100"))
GNEWS_WARN_AT = max(1, GNEWS_DAILY_LIMIT * 8 // 10)
_gnews = {"day": None, "sent": 0, "used_up": False, "warned": False}


def utc_day(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()


def seconds_to_utc_midnight(now: datetime | None = None) -> int:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return max(1, int((midnight - now).total_seconds()))


def _gnews_today() -> dict:
    day = utc_day()
    if _gnews["day"] != day:
        _gnews.update(day=day, sent=0, used_up=False, warned=False)
    return _gnews


def gnews_sent_today() -> int:
    return _gnews_today()["sent"]


async def reserve_gnews_search():
    """Counts one GNews search before it's sent. Raises errors.news_daily_limit past the day's limit."""
    state = _gnews_today()
    if state["used_up"]:
        raise errors.news_daily_limit(seconds_to_utc_midnight(), "GNews answered 403 (daily limit) earlier today")
    totals = await cache_service.add_usage(GNEWS, state["day"], {"requests": 1})
    state["sent"] = totals.get("requests", 0) if totals else state["sent"] + 1
    if state["sent"] > GNEWS_DAILY_LIMIT:
        # Not sent: the reservation is given back, and the refusal counted
        if totals:
            await cache_service.add_usage(GNEWS, state["day"], {"requests": -1, "refused": 1})
        state["sent"] -= 1
        observability.add_totals(gnews_refused=1)
        raise errors.news_daily_limit(seconds_to_utc_midnight(),
                                      f"{state['sent']} GNews searches today (limit {GNEWS_DAILY_LIMIT}): not sent")
    observability.add_totals(gnews_calls=1)
    if state["sent"] >= GNEWS_WARN_AT and not state["warned"]:
        state["warned"] = True
        print(f"[WARN] GNews: {state['sent']} of {GNEWS_DAILY_LIMIT} searches used today "
              f"(resets at midnight UTC, in {seconds_to_utc_midnight() / 3600:.1f} h).")


def gnews_used_up():
    """GNews answered 403: its daily limit is reached. Later searches today are refused without a call."""
    _gnews_today()["used_up"] = True
    print(f"[WARN] GNews: today's searches are used up (403). Refusing searches for "
          f"{seconds_to_utc_midnight() / 3600:.1f} h, until midnight UTC.")


async def load_today():
    """At startup: today's embedding requests and GNews searches from Redis, so a restart remembers them."""
    used = used_today(await cache_service.get_usage(EMBEDDING_MODEL, quota_day()))
    if used:
        _adopt_embedding_day(used)
    gnews = await cache_service.get_usage(GNEWS, utc_day())
    if gnews:
        _gnews_today()["sent"] = max(0, gnews.get("requests", 0))
