import contextvars
import hashlib
import ipaddress
import math
import os
import re
import time

from fastapi.responses import JSONResponse

from services import cache_service, errors, observability
from services.safety import events, settings

# --- Consumption limits per visitor (feature 3, Section L #87, #88, #89, #94) ---
# The endpoints are public. Without limits, one script could use the day's 40 chat calls and
# 100 GNews searches in a minute. Each visitor (one IP address, an IPv6 /64) gets:
#   per minute   view requests, fresh builds (a GNews search each), Follow checks, new follows, ratings
#   per day      a share of the chat calls (DAILY_CHAT_SHARE) and of new follows (FOLLOWS_PER_DAY)
# The per-minute and per-day checks run in an ASGI middleware, before the body is parsed and before
# the request's trace starts: a refused request costs one Redis round trip, no parsing, no LangSmith
# trace (a script hammering the app can't use up the trace quota either). Only the fresh-build limit
# runs inside the pipeline (check_build), because only the pipeline knows whether a build is fresh.
#
# SAFETY_RATE_LIMIT: off (nothing counted), monitor (counted, RateLimit headers sent, an over-limit
# request is logged and flagged on its trace but served), enforce (429 with Retry-After).
# The request body cap (#88) is always on: it isn't a judgement about a visitor, and every real body
# is far below it.
#
# Counters live in Redis (cache_service.limit_key): a fixed window per minute or day, one key per
# window, so a counter is never reset. Redis down -> every limit fails open (#95): the app already
# runs without Redis, and a limit must not take it down.


def _env_int(name: str, default: int, low: int = 1) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        print(f"[WARN] {name}={raw!r} isn't a whole number; using {default}.")
        return default
    return max(low, value)


# Requests per visitor per minute, by bucket. RATE_LIMIT_<BUCKET> overrides one.
MINUTE_LIMITS = {
    # Each view request is one chat call (report, timeline, contradictions): 3 views of 3 topics a minute
    "view": _env_int("RATE_LIMIT_VIEW", 10),
    # A fresh build is a GNews search (100 a day for everyone), a scrape and an embedding call
    "build": _env_int("RATE_LIMIT_BUILD", 4),
    "follow_check": _env_int("RATE_LIMIT_FOLLOW_CHECK", 6),
    "follow": _env_int("RATE_LIMIT_FOLLOW", 5),
    "feedback": _env_int("RATE_LIMIT_FEEDBACK", 10),
}
# Chat calls per visitor per day (Pacific, when Gemini's quota resets): a quarter of the 40 the two
# models allow together, so no visitor can use the whole day's limit (#89). Answers are counted, not
# attempts: a call Google failed (busy, timeout) isn't held against the visitor.
DAILY_CHAT_SHARE = _env_int("DAILY_CHAT_SHARE", 10)
# New followed topics per visitor per day (#94)
FOLLOWS_PER_DAY = _env_int("FOLLOWS_PER_DAY", 10)
# The largest body the app accepts (#88). The largest real body: a rating with a 1,000-character comment,
# or a Follow with a 1,000-character topic (C #21 reads up to 1,000), each at most ~6 KB even if every
# character is JSON-escaped. Bigger bodies get the specific "too long" messages; this only stops floods.
MAX_BODY_BYTES = _env_int("MAX_BODY_BYTES", 16_384, low=1024)
# Which X-Forwarded-For entry is the visitor, counted from the right. Render's proxy adds the address it
# saw to the END of the header and keeps whatever the client sent before it (reported on Render's
# forum, 2024), so the first entry can be anything a script writes there: limits keyed on it could be
# dodged with one header. 1 = the last entry, written by Render. 0 = the TCP peer (no proxy, local dev).
# scripts/check_client_ip.py checks on prod that a spoofed header doesn't get a new bucket.
CLIENT_IP_HOPS = _env_int("CLIENT_IP_HOPS", 1, low=0)

MINUTE = 60
_DAY_TTL = 2 * 24 * 60 * 60

# Which bucket and which daily share each route uses. Other routes (the page, static files, reading a
# followed topic's log, unfollowing) are Redis reads or nothing, and aren't limited.
_FOLLOW_CHECK = re.compile(r"/api/follow/[^/]+/check")
_ROUTES = {
    "/query": ("view", "chat"),
    "/api/timeline": ("view", "chat"),
    "/api/contradictions": ("view", "chat"),
    "/api/follow": ("follow", "follows"),
    "/api/feedback": ("feedback", None),
}
_DAILY = {"chat": lambda: DAILY_CHAT_SHARE, "follows": lambda: FOLLOWS_PER_DAY}
_BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def route_of(method: str, path: str) -> tuple[str, str | None] | None:
    """(minute bucket, daily share) for a request, or None when it isn't limited."""
    if method != "POST":
        return None
    path = path.rstrip("/") or "/"
    if path in _ROUTES:
        return _ROUTES[path]
    if _FOLLOW_CHECK.fullmatch(path):
        return ("follow_check", "chat")
    return None


# --- Who is asking ---
# The visitor's key: a hash of the address, so no IP is written to Redis, a trace or a log line. It is
# pseudonymous, not anonymous (IPv4 has few enough addresses to try them all).
_client: contextvars.ContextVar[str | None] = contextvars.ContextVar("limits_client", default=None)
# Per request: the RateLimit headers to send, and events waiting for the request's trace to open
_request: contextvars.ContextVar[dict | None] = contextvars.ContextVar("limits_request", default=None)


def _header_values(scope, name: bytes) -> list[str]:
    return [v.decode("latin-1") for k, v in scope.get("headers") or [] if k.lower() == name]


def client_address(scope, hops: int | None = None) -> str:
    """The visitor's address as a string: an IPv4 address or an IPv6 /64. A value that isn't an
    address is kept as it is (hashed later), so a garbled header can't make a visitor unlimited."""
    hops = CLIENT_IP_HOPS if hops is None else hops
    entries = [e.strip() for value in _header_values(scope, b"x-forwarded-for") for e in value.split(",") if e.strip()]
    if hops > 0 and entries:
        raw = entries[-hops] if len(entries) >= hops else entries[0]
    else:
        raw = str((scope.get("client") or ("unknown",))[0])
    host = raw.strip()
    if host.startswith("[") and "]" in host:              # "[2001:db8::1]:443"
        host = host[1:host.index("]")]
    elif host.count(":") == 1:                             # "203.0.113.7:51234"
        host = host.split(":", 1)[0]
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return raw[:100]
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        # One home or phone gets a whole /64: keying on the full address would give it 2^64 buckets
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def client_key(address: str) -> str:
    return hashlib.sha256(f"limits:{address}".encode("utf-8")).hexdigest()[:16]


def current_client() -> str | None:
    """This request's visitor key, or None outside a request (scripts, startup)."""
    return _client.get()


# --- Windows ---

def _minute_window(now: float) -> tuple[str, int]:
    """('m29849712', seconds left in this minute)."""
    return f"m{int(now // MINUTE)}", max(1, math.ceil(MINUTE - now % MINUTE))


def _day() -> tuple[str, int]:
    """(today's date where Gemini's quota resets, seconds until then). Follows use the same day."""
    from services import usage            # usage imports this module
    return usage.quota_day(), usage.seconds_to_reset()


# --- Events ---

def _event(where: str, count: int, limit: int, per: str, client: str, mode: str) -> events.SafetyEvent:
    return events.SafetyEvent(
        check="rate_limit", verdict="block", mode=mode, where=where, score=None, threshold=None, engine="code",
        ms=0.0, label=f"{count}/{limit} {per}", text_hash=client, enforced=mode == "enforce")


def _note(event: events.SafetyEvent):
    """An over-limit request in monitor mode is served: its event waits for the request's trace to open
    (flush_events, an observability request hook), so it lands on the root span. Outside a request: logged."""
    state = _request.get()
    if state is None:
        events.record(event)
    else:
        state["pending"].append(event)


def flush_events():
    """Called when a request's root span opens: records the events the middleware noted for it."""
    state = _request.get()
    if not state:
        return
    pending, state["pending"] = state["pending"], []
    for event in pending:
        events.record(event)


observability.add_request_hook(flush_events)

_last_fail_open = {"at": 0.0}


def _failed_open(what: str):
    """No Redis (or it's in its back-off): the limit isn't applied. One log line a minute at most."""
    now = time.monotonic()
    if now - _last_fail_open["at"] >= MINUTE:
        _last_fail_open["at"] = now
        why = "REDIS_URL isn't set" if cache_service.redis_client is None else "Redis is unavailable"
        print(f"[WARN] Limits: {why}, so {what} isn't limited (fails open).")


# --- The checks ---

async def check_route(bucket: str, daily: str | None, client: str, state: dict | None = None) -> errors.AppError | None:
    """
    Counts one request in its minute bucket and reads the visitor's daily share, in one Redis round
    trip. Returns the error to send in enforce mode, else None. Sets the RateLimit headers in `state`.
    """
    mode = settings.mode("rate_limit")
    if mode == "off":
        return None
    window, reset = _minute_window(time.time())
    limit = MINUTE_LIMITS[bucket]
    day, until_reset = _day()
    daily_key = cache_service.limit_key(daily, day, client) if daily else None
    result = await cache_service.count_and_read(cache_service.limit_key(bucket, window, client), 2 * MINUTE, daily_key)
    if result is None:
        _failed_open(f"the {bucket} rate")
        return None
    count, used = result
    if state is not None:
        state["headers"] = {"RateLimit-Limit": str(limit), "RateLimit-Remaining": str(max(0, limit - count)),
                            "RateLimit-Reset": str(reset)}
    if count > limit:
        event = _event(bucket, count, limit, "per minute", client, mode)
        if mode == "enforce":
            events.record(event)                  # no trace yet: the log line
            return errors.rate_limited(reset, f"{bucket}: {count} requests this minute (limit {limit}), client {client}")
        _note(event)
    if daily and used is not None:
        share = _DAILY[daily]()
        if used >= share:
            event = _event(f"{daily}_day", used, share, "today", client, mode)
            if mode == "enforce":
                events.record(event)
                log = f"{daily}: {used} today (limit {share}), client {client}"
                return (errors.daily_share_used(until_reset, log) if daily == "chat"
                        else errors.follow_limit(until_reset, log))
            _note(event)
    return None


async def check_build(label: str):
    """
    #87, the fresh-build bucket: called by the pipeline on a cache miss, before it takes a build slot
    (a GNews search, a scrape, an embedding call). Raises 429 in enforce mode. The error carries the
    visitor it was meant for, so a request that joined this build from another visitor builds again
    instead of getting someone else's 429 (pipeline_service.get_pipeline_data).
    """
    client = current_client()
    mode = settings.mode("rate_limit")
    if mode == "off" or client is None:
        return
    window, reset = _minute_window(time.time())
    limit = MINUTE_LIMITS["build"]
    count = await cache_service.incr_counter(cache_service.limit_key("build", window, client), 2 * MINUTE)
    if count is None:
        _failed_open("the fresh-build rate")
        return
    if count <= limit:
        return
    event = _event("build", count, limit, "per minute", client, mode)
    events.record(event)                          # the pipeline is inside the request's trace
    if mode == "enforce":
        error = errors.rate_limited(reset, f"fresh build of {label}: {count} this minute (limit {limit}), client {client}")
        error.client = client
        raise error


async def count_chat(client: str | None):
    """#89: one chat answer for this visitor today (usage.UsageHandler, in the background)."""
    if client is None or settings.mode("rate_limit") == "off":
        return
    day, _ = _day()
    await cache_service.incr_counter(cache_service.limit_key("chat", day, client), _DAY_TTL)


async def count_follow(client: str | None):
    """#94: one new followed topic for this visitor today (after the watch was saved)."""
    if client is None or settings.mode("rate_limit") == "off":
        return
    day, _ = _day()
    await cache_service.incr_counter(cache_service.limit_key("follows", day, client), _DAY_TTL)


def status() -> str:
    """For the startup line."""
    mode = settings.mode("rate_limit")
    if mode == "off":
        return "limits off"
    where = "Redis" if cache_service.redis_client is not None else "no Redis: per-visitor limits fail open"
    return (f"limits {mode} ({where}; per minute " + ", ".join(f"{k} {v}" for k, v in MINUTE_LIMITS.items())
            + f"; {DAILY_CHAT_SHARE} chat calls and {FOLLOWS_PER_DAY} follows per visitor a day; "
            f"bodies up to {MAX_BODY_BYTES:,} bytes; client = X-Forwarded-For entry {CLIENT_IP_HOPS} from the right)")


# --- The middleware ---

async def _send_error(error: errors.AppError, scope, receive, send, extra_headers: dict | None = None):
    print(f"[WARN] {scope.get('path')} -> {error.status}: {error.log}")
    headers = {**(extra_headers or {}), **(error.headers or {})}
    await JSONResponse(status_code=error.status, content={"detail": error.message}, headers=headers)(scope, receive, send)


async def _read_body(receive, limit: int) -> tuple[list[dict] | None, int]:
    """Reads the whole body (it's small) and returns its messages to replay, or None once it passes the limit."""
    messages, size = [], 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            messages.append(message)               # the client went away: let the app see it
            return messages, size
        size += len(message.get("body", b""))
        if size > limit:
            return None, size
        messages.append(message)
        if not message.get("more_body", False):
            return messages, size


class ClientLimits:
    """
    Pure ASGI middleware, inside SecurityHeaders (so every response it sends has those headers):
      1. who the visitor is (client_key), kept in a context variable for the pipeline and the chat counter
      2. body size (#88): Content-Length over the cap -> 413 at once; a body without one is read up to the cap
      3. the route's minute bucket and daily share (check_route) -> 429 in enforce mode
      4. the RateLimit headers on the response; a Follow that worked counts toward the day's follows
    The body is read here, not by FastAPI, because an error FastAPI hits while reading the body
    becomes its own 400 "There was an error parsing the body", not our message.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        client = client_key(client_address(scope))
        state = {"headers": {}, "pending": []}
        client_token, state_token = _client.set(client), _request.set(state)
        try:
            await self._handle(scope, receive, send, client, state)
        finally:
            for event in state["pending"]:         # a limited route without a trace: still logged
                events.record(event)
            _client.reset(client_token)
            _request.reset(state_token)

    async def _handle(self, scope, receive, send, client: str, state: dict):
        method, path = scope.get("method", "GET"), scope.get("path", "")
        if method in _BODY_METHODS:
            length = (_header_values(scope, b"content-length") or [""])[0].strip()
            if length.isdigit() and int(length) > MAX_BODY_BYTES:
                return await _send_error(errors.request_too_large(f"Content-Length {length} (limit {MAX_BODY_BYTES})"),
                                         scope, receive, send)
        route = route_of(method, path)
        if route is not None:
            error = await check_route(*route, client, state)
            if error is not None:
                return await _send_error(error, scope, receive, send, state["headers"])
        if method in _BODY_METHODS:
            messages, size = await _read_body(receive, MAX_BODY_BYTES)
            if messages is None:
                return await _send_error(errors.request_too_large(f"body over {MAX_BODY_BYTES} bytes, no Content-Length"),
                                         scope, receive, send)
            original = receive

            async def receive():
                return messages.pop(0) if messages else await original()

        status = {"code": None}
        extra = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in state["headers"].items()]

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                if extra:
                    message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)
        if route is not None and route[1] == "follows" and status["code"] == 200:
            await count_follow(client)
