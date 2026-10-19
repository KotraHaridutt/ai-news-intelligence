import asyncio
import os
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from langchain_core.callbacks import AsyncCallbackHandler

from services import cache_service, errors, observability

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


async def _count_and_log(model: str, counts: dict, line: str, chat: bool):
    try:
        totals = await cache_service.add_usage(model, quota_day(), counts)
    except Exception as e:                    # never let counting fail anything
        totals = None
        print(f"[WARN] Couldn't count Gemini usage: {type(e).__name__}: {e}")
    used = used_today(totals)
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

    def __init__(self, role: str):
        self.role = role
        self._started: dict = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs):
        model = (metadata or {}).get("ls_model_name") or ((serialized or {}).get("kwargs") or {}).get("model") or "?"
        view = next((tag for tag in (tags or []) if tag in VIEWS), "chat")
        self._started[run_id] = (time.perf_counter(), str(model).removeprefix("models/"), view)

    async def on_llm_end(self, response, *, run_id, **kwargs):
        started, model, view = self._started.pop(run_id, (time.perf_counter(), "?", "chat"))
        seconds = time.perf_counter() - started
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError):
            message = None
        usage = getattr(message, "usage_metadata", None) or {}
        answered = (getattr(message, "response_metadata", None) or {}).get("model_name") or model
        input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        thinking = (usage.get("output_token_details") or {}).get("reasoning", 0)
        cost = cost_usd(model, input_tokens, output_tokens)
        observability.add_totals(chat_calls=1, input_tokens=input_tokens, output_tokens=output_tokens,
                                 thinking_tokens=thinking, cost_usd=cost or 0, answered_by=answered)
        line = (f"[INFO] LLM {view}: {answered} in {input_tokens:,} · thinking {thinking:,} · out {output_tokens:,} tokens, "
                f"{seconds:.1f} s, " + (f"~${cost:.4f}" if cost is not None else "price unknown")
                + f", fallback {'yes' if self.role == 'fallback' else 'no'}")
        _background(_count_and_log(model, {"attempts": 1, "ok": 1, "input_tokens": input_tokens,
                                           "output_tokens": output_tokens, "thinking_tokens": thinking}, line, chat=True))

    async def on_llm_error(self, error, *, run_id, **kwargs):
        started, model, view = self._started.pop(run_id, (time.perf_counter(), "?", "chat"))
        kind = _error_kind(error)
        if kind == "daily_limit":
            mark_exhausted(model, errors.retry_after_seconds(error))
        observability.add_totals(chat_calls=1, chat_errors=1)
        line = f"[WARN] LLM {view}: {model} attempt failed ({kind}) after {time.perf_counter() - started:.1f} s"
        _background(_count_and_log(model, {"attempts": 1, kind: 1}, line, chat=True))


# --- Embedding calls (#35) ---

def record_embedding(model: str, characters: int, texts: int, error=None):
    """One embedding attempt (rag_service.call_with_retry): estimated tokens and cost, counted per day."""
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
