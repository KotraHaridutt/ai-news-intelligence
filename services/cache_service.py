import os
import re
import json
import time
import random
import hashlib
import secrets
import numpy as np
from dotenv import load_dotenv
import redis.asyncio as redis

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

# --- Redis client ---
# Without REDIS_URL (e.g. local dev) the app still runs: every helper below
# checks for None and behaves as a miss, so each request runs the full pipeline.
if REDIS_URL:
    # from_url doesn't connect yet; the first command opens the connection.
    # Short timeouts so a slow or down Redis can't stall a request.
    # socket_timeout applies to each socket read, not the whole command,
    # so large MGET responses still work.
    redis_client = redis.from_url(
        REDIS_URL,
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )
    print("[INFO] Redis cache enabled.")
else:
    redis_client = None
    print("[WARN] REDIS_URL not set. Caching is disabled; every request runs the full pipeline.")

# --- TTLs (seconds) ---
ARTICLES_TTL = 2 * 60 * 60       # How fresh the news is
PAGE_TTL = 36 * 60 * 60          # Scraped page text; some pages (live blogs) change
EMB_TTL = 7 * 24 * 60 * 60       # Same text + same model always gives the same vector
TTL_JITTER = 0.10                # +/-10% so keys written together don't expire together
# An answer can be followed only while the build it came from is cached, so it lives as long
ANSWER_TTL = ARTICLES_TTL
# A URL purged as poisoned (feature 3 #55) stays out of every build this long: longer than the
# oldest article GNews returns (7 days, news_fetcher.MAX_LOOKBACK_HOURS), so it can't come back.
PURGED_TTL = 8 * 24 * 60 * 60
# A followed topic expires after this long without a check (feature-1 decision #14).
# Exact, not jittered: the page tells the user "30 days".
WATCH_TTL = 30 * 24 * 60 * 60

# Keys per MGET / pipeline. Keeps each request well under Upstash's size limit
# (~50 vectors x 12 KB = ~600 KB).
BATCH_SIZE = 50

# Bump when the stored format changes, so old data is never read back.
KEY_VERSION = "v1"

# After a Redis error, skip Redis for this long instead of paying the
# timeout on every cache call of every request while it's down.
BACKOFF_SECONDS = 30
_skip_redis_until = 0.0


# --- Keys ---

def normalize_query(query: str) -> str:
    """'  Tesla   Layoffs ' -> 'tesla layoffs'"""
    return " ".join(query.split()).lower()

def articles_key(normalized_query: str) -> str:
    return f"{KEY_VERSION}:articles:{normalized_query}"

def page_key(url: str) -> str:
    """A scraped page: its text and where and when it was read (feature 3 #50), as JSON."""
    return f"{KEY_VERSION}:scraped:{url}"

def legacy_page_key(url: str) -> str:
    """A page's text alone, as written before feature 3 G. Still read until they expire (PAGE_TTL)."""
    return f"{KEY_VERSION}:page:{url}"

def purged_key(url: str) -> str:
    """A URL scripts/purge_url.py removed as poisoned: no build uses it while this exists."""
    return f"{KEY_VERSION}:purged:{url}"

def emb_key(model: str, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{KEY_VERSION}:emb:{model}:{text_hash}"

def since_key(normalized_query: str, from_hour: str) -> str:
    """
    A check's article list (feature 1): one per topic and window start hour, so checks of the
    same topic in the same hour share one GNews request. Its own key family, not articles:,
    so no typed query can ever collide with it; from_hour has a fixed length and comes first.
    """
    return f"{KEY_VERSION}:since:{from_hour}:{normalized_query}"

def answer_key(answer_id: str) -> str:
    return f"{KEY_VERSION}:answer:{answer_id}"

def watch_key(watch_id: str) -> str:
    return f"{KEY_VERSION}:watch:{watch_id}"

def usage_key(model: str, day: str) -> str:
    """One model's Gemini usage on one day (feature 2 #33); day is the Pacific date, '2026-10-19'."""
    return f"{KEY_VERSION}:usage:{model}:{day}"

def verdict_key(question_version: str, content_hash: str) -> str:
    """A Clef verdict (feature 3 #15d): one question's version + a hash of the text it judged."""
    return f"{KEY_VERSION}:safety:{question_version}:{content_hash}"

def ttl_with_jitter(base_ttl: int) -> int:
    return int(base_ttl * random.uniform(1 - TTL_JITTER, 1 + TTL_JITTER))


# --- Ids for answers and followed topics ---
# Random and unguessable: whoever has a watch id can read and check that followed topic.
# token_urlsafe(16) gives 22 characters of A-Z a-z 0-9 - _.
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{22}")

def new_id() -> str:
    return secrets.token_urlsafe(16)

def is_valid_id(value) -> bool:
    """Checked before any Redis call, so a typed or cut-off id never reaches Redis."""
    return isinstance(value, str) and _ID_PATTERN.fullmatch(value) is not None


# --- Low-level batch helpers ---
# Cache errors never fail a request: a failed read is a miss, a failed write is skipped.

def _redis_available() -> bool:
    return redis_client is not None and time.monotonic() >= _skip_redis_until

def _on_redis_error(action: str, e: Exception):
    global _skip_redis_until
    _skip_redis_until = time.monotonic() + BACKOFF_SECONDS
    print(f"[WARN] Cache {action} failed ({type(e).__name__}: {e}). Skipping Redis for {BACKOFF_SECONDS}s.")

async def _mget(keys: list[str]) -> list[bytes | None]:
    if not _redis_available() or not keys:
        return [None] * len(keys)
    values = []
    try:
        for i in range(0, len(keys), BATCH_SIZE):
            values.extend(await redis_client.mget(keys[i:i + BATCH_SIZE]))
        return values
    except Exception as e:
        _on_redis_error("read", e)
        return [None] * len(keys)

async def _mset_with_ttl(items: list[tuple[str, bytes | str]], base_ttl: int) -> bool:
    """Returns whether everything was written. Callers that only cache can ignore it."""
    # Always SET ... EX: a plain SET has no TTL and would remove an existing one.
    if not items:
        return True
    if not _redis_available():
        return False
    try:
        for i in range(0, len(items), BATCH_SIZE):
            pipe = redis_client.pipeline(transaction=False)
            for key, value in items[i:i + BATCH_SIZE]:
                pipe.set(key, value, ex=ttl_with_jitter(base_ttl))
            await pipe.execute()
        return True
    except Exception as e:
        _on_redis_error("write", e)
        return False


# --- Article list (metadata only, no full_text) ---

async def get_articles(normalized_query: str, strict: bool = False) -> dict | None:
    """strict (Follow): a failed read raises WatchStoreUnavailable instead of reading as a miss."""
    if strict:
        raw = await _watch_call("article list read", lambda c: c.get(articles_key(normalized_query)))
    else:
        [raw] = await _mget([articles_key(normalized_query)])
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        print(f"[WARN] Corrupt cache entry for topic {_log_ref(normalized_query)}: {e}")
        return None

def _log_ref(normalized_query: str) -> str:
    """A topic in a log line, as a short hash: a query may hold personal data (feature 3 H #61)."""
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:12]

async def set_articles(normalized_query: str, data: dict):
    await _mset_with_ttl([(articles_key(normalized_query), json.dumps(data))], ARTICLES_TTL)

async def get_since_articles(normalized_query: str, from_hour: str) -> dict | None:
    [raw] = await _mget([since_key(normalized_query, from_hour)])
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        print(f"[WARN] Corrupt check cache entry for topic {_log_ref(normalized_query)} from {from_hour}: {e}")
        return None

async def set_since_articles(normalized_query: str, from_hour: str, data: dict):
    await _mset_with_ttl([(since_key(normalized_query, from_hour), json.dumps(data))], ARTICLES_TTL)


# --- Answers shown to the user (feature 1: what V1 said, for Follow) ---
# {"query" (normalized), "view", "text" (the answer's markdown), "built_at", "created_at"}.
# Still a cache: a failed write doesn't fail the answer, the answer just can't be followed.

async def set_answer(answer_id: str, record: dict) -> bool:
    """Returns whether the record was saved."""
    return await _mset_with_ttl([(answer_key(answer_id), json.dumps(record))], ANSWER_TTL)

async def get_answers(answer_ids: list[str], strict: bool = False) -> dict[str, dict]:
    """
    Returns {answer_id: record} for the ids that exist; malformed ids are skipped without a lookup.
    strict (Follow): a failed read raises WatchStoreUnavailable instead of reading as "no such
    answer", so a Redis outage isn't reported to the user as an expired report.
    """
    ids = [i for i in dict.fromkeys(answer_ids) if is_valid_id(i)]
    keys = [answer_key(i) for i in ids]
    if strict:
        values = await _watch_call("answers read", lambda c: c.mget(keys)) if keys else []
    else:
        values = await _mget(keys)
    records = {}
    for answer_id, raw in zip(ids, values):
        if raw is None:
            continue
        try:
            records[answer_id] = json.loads(raw)
        except ValueError as e:
            print(f"[WARN] Corrupt answer record {answer_id}: {e}")
    return records


# --- Followed topics (feature 1) ---
# Not a cache: a followed topic is the only copy of what the user has already been shown.
# A read or write that silently did nothing would show the same "new" stories again on the
# next check, so these raise WatchStoreUnavailable instead, and the endpoint answers 503.

class WatchStoreUnavailable(Exception):
    """Redis isn't configured, is in its back-off after an error, or the call failed."""

def _watch_store():
    if redis_client is None:
        raise WatchStoreUnavailable("REDIS_URL is not set")
    if not _redis_available():
        raise WatchStoreUnavailable(f"Redis is skipped after an error (back-off {BACKOFF_SECONDS}s)")
    return redis_client

def require_watch_store():
    """Raises WatchStoreUnavailable now, before any work is done for a followed topic."""
    _watch_store()

async def _watch_call(action: str, fn, *args, **kwargs):
    client = _watch_store()
    try:
        return await fn(client, *args, **kwargs)
    except Exception as e:
        _on_redis_error(f"watch {action}", e)
        raise WatchStoreUnavailable(f"watch {action} failed: {type(e).__name__}: {e}") from e

async def get_watch(watch_id: str) -> dict | None:
    """The followed topic's snapshot, or None if the id is malformed, unknown or expired."""
    if not is_valid_id(watch_id):
        return None
    raw = await _watch_call("read", lambda c: c.get(watch_key(watch_id)))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        print(f"[WARN] Corrupt watch {watch_id}: {e}")
        return None

async def set_watch(watch_id: str, snapshot: dict):
    """Saves the snapshot and restarts its 30-day expiry. Raises WatchStoreUnavailable on failure."""
    if not is_valid_id(watch_id):
        raise ValueError(f"invalid watch id: {watch_id!r}")
    await _watch_call("write", lambda c: c.set(watch_key(watch_id), json.dumps(snapshot), ex=WATCH_TTL))

async def delete_watch(watch_id: str):
    """Deleting a malformed, unknown or already expired id is not an error."""
    if not is_valid_id(watch_id):
        return
    await _watch_call("delete", lambda c: c.delete(watch_key(watch_id)))


# --- Scraped pages (feature 3, G #50) ---
# A record per URL: {"text", "final_url" (after redirects), "fetched_at" (ISO UTC), "page_date" (the
# page's own publish date, YYYY-MM-DD, or None)}. An answer sentence can then be traced to the exact
# page version it came from. Pages cached before G hold only the text (legacy_page_key): they're read
# as a record without provenance until they expire, so a deploy doesn't re-scrape every cached page.

def page_record(value) -> dict:
    """A record from what a scrape or a caller gives: a record, or a page's text alone."""
    if isinstance(value, dict):
        return {"text": value.get("text") or "", "final_url": value.get("final_url"),
                "fetched_at": value.get("fetched_at"), "page_date": value.get("page_date")}
    return {"text": value or "", "final_url": None, "fetched_at": None, "page_date": None}

async def get_page_records(urls: list[str]) -> dict[str, dict]:
    """Returns {url: record} for the URLs that are cached."""
    records = {}
    for url, raw in zip(urls, await _mget([page_key(url) for url in urls])):
        if raw is None:
            continue
        try:
            record = page_record(json.loads(raw))
        except (ValueError, AttributeError):
            continue
        if record["text"]:
            records[url] = record
    missing = [url for url in urls if url not in records]
    for url, raw in zip(missing, await _mget([legacy_page_key(url) for url in missing])):
        if raw is not None:
            records[url] = page_record(raw.decode("utf-8"))
    return records

async def set_page_records(pages: dict):
    """pages: {url: record or text}."""
    items = [(page_key(url), json.dumps(page_record(value), ensure_ascii=False)) for url, value in pages.items()]
    await _mset_with_ttl(items, PAGE_TTL)

async def get_pages(urls: list[str]) -> dict[str, str]:
    """Returns {url: full_text} for the URLs that are cached."""
    return {url: record["text"] for url, record in (await get_page_records(urls)).items()}

async def set_pages(pages: dict):
    """pages: {url: text or record}."""
    await set_page_records(pages)


# --- Purged URLs (feature 3, G #55) ---

async def get_purged(urls: list[str]) -> set[str]:
    """The URLs among `urls` that were purged as poisoned. A failed read is "none purged", like a cache miss."""
    return {url for url, raw in zip(urls, await _mget([purged_key(url) for url in urls])) if raw is not None}


# --- Embedding vectors (raw float32 bytes) ---

async def get_cached_vectors(model: str, texts: list[str]) -> list[np.ndarray | None]:
    """Returns one vector per text, in the same order, with None for misses."""
    values = await _mget([emb_key(model, text) for text in texts])
    vectors = []
    for raw in values:
        if raw is None or len(raw) % 4 != 0:
            vectors.append(None)
        else:
            vectors.append(np.frombuffer(raw, dtype=np.float32))
    return vectors

async def set_cached_vectors(model: str, texts: list[str], vectors):
    items = [
        (emb_key(model, text), np.asarray(vector, dtype=np.float32).tobytes())
        for text, vector in zip(texts, vectors)
    ]
    await _mset_with_ttl(items, EMB_TTL)


# --- Clef verdicts (feature 3, Section B #15d) ---
# The same text asked the same question gives the same answer (measured twice in Section A:
# identical), so a query or chunk that was already checked costs no neurons and no time. The key
# holds the question's version, so a change of wording never reads an old answer.
VERDICT_TTL = EMB_TTL

async def get_verdicts(keys: list[str]) -> list[dict | None]:
    verdicts = []
    for raw in await _mget(keys):
        try:
            verdicts.append(json.loads(raw) if raw is not None else None)
        except ValueError:
            verdicts.append(None)
    return verdicts

async def set_verdicts(items: dict[str, dict]):
    await _mset_with_ttl([(key, json.dumps(value)) for key, value in items.items()], VERDICT_TTL)


# --- Reranker scores (feature 4, Section B #8) ---
# One score per (engine and model, query, chunk text): a cross-encoder scores each pair on its own, so a
# cached chunk never needs the call again, and the three views of a topic share one call. Both engines
# returned identical scores for the same request (Section A §3.2), so caching changes nothing but the cost.
RERANK_TTL = EMB_TTL

def rerank_key(model: str, query: str, text: str) -> str:
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{KEY_VERSION}:rerank:{model}:{query_hash}:{text_hash}"

async def get_rerank_scores(model: str, query: str, texts: list[str]) -> list[float | None]:
    """One cached score per text, in order, None for a miss (and for all of them without Redis)."""
    scores = []
    for raw in await _mget([rerank_key(model, query, text) for text in texts]):
        try:
            scores.append(float(raw) if raw is not None else None)
        except (TypeError, ValueError):
            scores.append(None)
    return scores

async def set_rerank_scores(model: str, query: str, scores: dict[str, float]):
    """scores: text -> score."""
    await _mset_with_ttl([(rerank_key(model, query, text), repr(float(score))) for text, score in scores.items()], RERANK_TTL)


# --- Daily Gemini usage (feature 2, #33) ---
# A hash per model per day: attempts, answers, errors by kind, tokens. The day is Pacific time,
# when Google resets the free quota. Kept 2 days. Like the cache, a Redis failure never fails a
# request: the count is just missing for that call.
USAGE_TTL = 2 * 24 * 60 * 60

async def add_usage(model: str, day: str, counts: dict[str, int]) -> dict[str, int] | None:
    """Adds `counts` to the model's day and returns the day's totals, or None without Redis."""
    if not _redis_available() or not counts:
        return None
    key = usage_key(model, day)
    try:
        pipe = redis_client.pipeline(transaction=False)
        for field, amount in counts.items():
            pipe.hincrby(key, field, int(amount))
        pipe.expire(key, USAGE_TTL)
        pipe.hgetall(key)
        totals = (await pipe.execute())[-1]
        return {k.decode() if isinstance(k, bytes) else k: int(v) for k, v in totals.items()}
    except Exception as e:
        _on_redis_error("usage write", e)
        return None

async def get_usage(model: str, day: str) -> dict[str, int] | None:
    """The model's totals for the day ({} if none yet), or None without Redis."""
    if not _redis_available():
        return None
    try:
        totals = await redis_client.hgetall(usage_key(model, day))
        return {k.decode() if isinstance(k, bytes) else k: int(v) for k, v in totals.items()}
    except Exception as e:
        _on_redis_error("usage read", e)
        return None


# --- Per-visitor counters (feature 3, Section L #87, #89, #94) ---
# One integer per visitor and window: `v1:limit:<bucket>:<window>:<client>`, where client is a hash of
# the visitor's address (safety/limits.client_key: no IP is written to Redis) and window names the
# minute or the day, so a new window is a new key and no counter ever has to be reset. Like the
# cache, a Redis failure never fails a request: the count is None and the limit fails open (#95).

def limit_key(bucket: str, window: str, client: str) -> str:
    return f"{KEY_VERSION}:limit:{bucket}:{window}:{client}"

async def incr_counter(key: str, ttl: int, amount: int = 1) -> int | None:
    """Adds `amount` and returns the new value (one round trip: INCRBY + EXPIRE), or None without Redis."""
    if not _redis_available():
        return None
    try:
        pipe = redis_client.pipeline(transaction=False)
        pipe.incrby(key, int(amount))
        pipe.expire(key, int(ttl))
        return int((await pipe.execute())[0])
    except Exception as e:
        _on_redis_error("counter write", e)
        return None

async def get_counter(key: str) -> int | None:
    """The counter's value (0 if it doesn't exist), or None without Redis."""
    if not _redis_available():
        return None
    try:
        value = await redis_client.get(key)
        return int(value) if value is not None else 0
    except Exception as e:
        _on_redis_error("counter read", e)
        return None

async def count_and_read(key: str, ttl: int, read_key: str | None = None) -> tuple[int, int | None] | None:
    """Adds 1 to `key` and reads `read_key` (a day's share) in the same round trip: (new value, the other
    value or 0; None without read_key). None without Redis."""
    if not _redis_available():
        return None
    try:
        pipe = redis_client.pipeline(transaction=False)
        pipe.incr(key)
        pipe.expire(key, int(ttl))
        if read_key:
            pipe.get(read_key)
        results = await pipe.execute()
        other = (int(results[2]) if results[2] is not None else 0) if read_key else None
        return int(results[0]), other
    except Exception as e:
        _on_redis_error("counter write", e)
        return None
