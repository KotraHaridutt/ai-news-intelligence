import os
from dotenv import load_dotenv
import redis.asyncio as redis

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")

# --- Redis client ---
# Without REDIS_URL (e.g. local dev) the app still runs: every cache helper
# checks for None and behaves as a miss, so each request runs the full pipeline.
if REDIS_URL:
    # from_url doesn't connect yet; the first command opens the connection.
    redis_client = redis.from_url(REDIS_URL)
    print("[INFO] Redis cache enabled.")
else:
    redis_client = None
    print("[WARN] REDIS_URL not set. Caching is disabled; every request runs the full pipeline.")
