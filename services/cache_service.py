import os
import json
import time
import random
import hashlib
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
    return f"{KEY_VERSION}:page:{url}"

def emb_key(model: str, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{KEY_VERSION}:emb:{model}:{text_hash}"

def ttl_with_jitter(base_ttl: int) -> int:
    return int(base_ttl * random.uniform(1 - TTL_JITTER, 1 + TTL_JITTER))


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

async def _mset_with_ttl(items: list[tuple[str, bytes | str]], base_ttl: int):
    # Always SET ... EX: a plain SET has no TTL and would remove an existing one.
    if not _redis_available() or not items:
        return
    try:
        for i in range(0, len(items), BATCH_SIZE):
            pipe = redis_client.pipeline(transaction=False)
            for key, value in items[i:i + BATCH_SIZE]:
                pipe.set(key, value, ex=ttl_with_jitter(base_ttl))
            await pipe.execute()
    except Exception as e:
        _on_redis_error("write", e)


# --- Article list (metadata only, no full_text) ---

async def get_articles(normalized_query: str) -> dict | None:
    [raw] = await _mget([articles_key(normalized_query)])
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as e:
        print(f"[WARN] Corrupt cache entry for '{normalized_query}': {e}")
        return None

async def set_articles(normalized_query: str, data: dict):
    await _mset_with_ttl([(articles_key(normalized_query), json.dumps(data))], ARTICLES_TTL)


# --- Page texts ---

async def get_pages(urls: list[str]) -> dict[str, str]:
    """Returns {url: full_text} for the URLs that are cached."""
    values = await _mget([page_key(url) for url in urls])
    return {url: raw.decode("utf-8") for url, raw in zip(urls, values) if raw is not None}

async def set_pages(pages: dict[str, str]):
    await _mset_with_ttl([(page_key(url), text) for url, text in pages.items()], PAGE_TTL)


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
