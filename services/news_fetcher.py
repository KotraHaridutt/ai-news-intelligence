import asyncio
import contextlib
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
import ssl
import certifi
import httpx
from dotenv import load_dotenv
import trafilatura
from bs4 import BeautifulSoup

from services import errors

# Building a TLS context loads the CA bundle, which blocks for ~0.5 s. Build it once
# at import and share it, instead of once per AsyncClient (i.e. per request).
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Load your .env file to get the API key
load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

if not GNEWS_API_KEY:
    print("ERROR: GNEWS_API_KEY not found in .env file. Please add it.")

# GNews free plan: at most 10 articles per request (and a 12-hour delay).
# Asking for more doesn't return more, so ask for exactly what the plan gives.
GNEWS_MAX_ARTICLES = 10

# GNews free plan: at most 1 request per second, or it answers 429. Up to
# MAX_CONCURRENT_BUILDS builds can start together, so GNews requests go one at a time,
# each starting GNEWS_MIN_INTERVAL after the previous one *finished*. Spacing the start
# times wasn't enough (measured: sent 1.1 s apart, still 429), because each request opens
# a new connection and the handshake time varies, so requests can arrive closer together.
# A second or third user waits a second or two instead of getting an error.
GNEWS_MIN_INTERVAL = 1.1      # seconds from the end of one GNews request to the start of the next
_gnews_lock = asyncio.Lock()
_gnews_last_done = 0.0

# Only articles published in the last 48 hours: the app reports on the latest news.
# The free plan's 12-hour delay is not added on top.
NEWS_WINDOW_HOURS = 48

# Shared minimum article length: scraping, clustering and indexing all use this
# so an article is either usable everywhere or nowhere.
MIN_TEXT_LEN = 100

# Headers for scraping article pages. httpx's default User-Agent ("python-httpx/x.y")
# tells news sites the request comes from a script, and many answer 403 "Access Denied".
# A browser User-Agent gets past those simple rules (not JavaScript challenges or paywalls).
SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Only these pages are parsed; PDFs, images and feeds are skipped
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")

def has_enough_text(text) -> bool:
    return bool(text) and len(text) >= MIN_TEXT_LEN

@contextlib.asynccontextmanager
async def _gnews_turn():
    """One GNews request at a time, starting GNEWS_MIN_INTERVAL after the previous one ended."""
    global _gnews_last_done
    async with _gnews_lock:
        wait = _gnews_last_done + GNEWS_MIN_INTERVAL - time.monotonic()
        if wait > 0:
            print(f"[INFO] Waiting {wait:.1f}s for GNews (free plan: 1 request per second).")
            await asyncio.sleep(wait)
        try:
            yield
        finally:
            _gnews_last_done = time.monotonic()   # also after a failed request

def _gnews_time(dt: datetime) -> str:
    """The UTC format GNews expects, e.g. '2026-10-09T10:00:00Z'."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _window_start() -> str:
    """Start of the news window in the UTC format GNews expects, e.g. '2026-10-09T10:00:00Z'."""
    return _gnews_time(datetime.now(timezone.utc) - timedelta(hours=NEWS_WINDOW_HOURS))

# --- Window for "Check for updates" (feature 1) ---
# Measured on the free plan (feature-1 Section A): the newest article GNews returns is never
# less than 12 h old, so a fetch at time T sees articles published up to about T - 12 h.
GNEWS_DELAY_HOURS = 12
# How far back a check may search. Measured: the free plan returns articles 6-7 days old.
# Also how long a followed topic keeps its stories' vectors, so a story older than this
# couldn't be matched anyway. A longer gap is reported to the user instead of searched.
MAX_LOOKBACK_HOURS = 7 * 24

def _as_utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

def check_window(since, now=None) -> dict:
    """
    The search window for a check, given when the previous fetch ran (`since`: V1's build or
    the last check, a datetime or ISO string). The previous fetch saw articles up to about
    since - delay, so this one starts there:

      from       since - delay, rounded DOWN to the hour (so checks in the same hour share one
                 fetch; the extra minutes only re-find articles the seen list already holds).
                 If that is earlier than now - MAX_LOOKBACK_HOURS, the limit rounded UP instead
      to_approx  now - delay: the newest article this fetch can return (not sent to GNews)
      gap_not_covered  {"from", "to"}: the part since the previous fetch that this one can't
                 search (only when the limit applied), else None
    All values are GNews-format UTC strings; "from_hour" ('2026-10-11T13') names the window.
    """
    now = _as_utc(now or datetime.now(timezone.utc))
    previous_end = _as_utc(since) - timedelta(hours=GNEWS_DELAY_HOURS)
    earliest = now - timedelta(hours=MAX_LOOKBACK_HOURS)
    start = previous_end.replace(minute=0, second=0, microsecond=0)
    if start < earliest:
        # Up, not down, so the window never reaches past what the plan was measured to return.
        # Whatever lies between the previous fetch and here is reported, never skipped silently.
        start = earliest.replace(minute=0, second=0, microsecond=0)
        if start < earliest:
            start += timedelta(hours=1)
    gap = None
    if previous_end < start:
        gap = {"from": _gnews_time(previous_end), "to": _gnews_time(start)}
    return {
        "from": _gnews_time(start),
        "from_hour": start.strftime("%Y-%m-%dT%H"),
        "to_approx": _gnews_time(now - timedelta(hours=GNEWS_DELAY_HOURS)),
        "gap_not_covered": gap,
    }

async def fetch_news(query: str, window_from: str | None = None) -> list[dict]:
    """
    Fetches the article list for a query from GNews: the best matches
    published in the last NEWS_WINDOW_HOURS, best match first. With window_from (a check's
    window, see check_window), the best matches published since then instead.
    Returns [] only when GNews answered with no articles. Raises AppError when the
    request fails, so an outage isn't shown to the user as "no news found".
    """
    if not GNEWS_API_KEY:
        raise errors.news_not_configured("GNEWS_API_KEY is not set")

    api_url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": GNEWS_MAX_ARTICLES,
        # GNews's default is "publishedAt" (newest first). We can only re-rank the
        # 10 articles it returns, so ask for the 10 that match the query best.
        "sortby": "relevance",
        "from": window_from or _window_start(),
        "token": GNEWS_API_KEY,
    }

    try:
        async with _gnews_turn(), httpx.AsyncClient(verify=SSL_CONTEXT) as client:
            print(f"[INFO] Fetching real news from GNews for: {query}")
            response = await client.get(api_url, params=params, timeout=10.0)
    except httpx.TransportError as e:   # timeouts, DNS, refused connections
        raise errors.news_unreachable(f"GNews request failed: {type(e).__name__}: {e}") from e
    except httpx.HTTPError as e:        # other request errors (e.g. a broken response)
        raise errors.news_failed(f"GNews request failed: {type(e).__name__}: {e}") from e

    if response.status_code != 200:
        raise errors.news_http_error(response.status_code, response.text)

    try:
        data = response.json()
    except ValueError as e:
        raise errors.news_failed(f"GNews sent invalid JSON: {response.text[:200]!r}") from e

    articles = []
    for item in data.get("articles", []):
        articles.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("description"),
            "source_name": item.get("source", {}).get("name"),
            "published_at": item.get("publishedAt"),
        })
    return articles

def extract_text(html: str):
    """
    Extracts the article body from a page's HTML. CPU-bound, so callers run it in a thread.
    """
    # 1. Try Trafilatura (Clean, Markdown-style extraction)
    full_text = trafilatura.extract(html, include_comments=False, include_tables=False)

    # 2. Fallback to BeautifulSoup if Trafilatura fails
    if not full_text:
        soup = BeautifulSoup(html, 'html.parser')
        # Try to get paragraphs specifically to avoid script/nav noise
        paragraphs = soup.find_all('p')
        full_text = " ".join([p.get_text(strip=True) for p in paragraphs])
    return full_text

class ScrapeFailed(Exception):
    """A page was fetched but can't be used. The message is the failure reason shown in the logs."""

def _failure_reason(e: Exception) -> str:
    """
    Groups a scrape failure so the logs show which fix would pay off:
    HTTP errors point to blocking, too little text to paywalls or empty pages.
    ScrapeFailed reasons: "HTTP <code>", "not HTML (<type>)", "too little text".
    """
    if isinstance(e, ScrapeFailed):
        return str(e)
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, httpx.TransportError):
        return "connection error"
    return f"other: {type(e).__name__}"

async def fetch_one_article(client: httpx.AsyncClient, url: str) -> dict:
    """
    Asynchronously fetches and scrapes a single article using Trafilatura for clean text.
    Returns status "success" or "failed"; a failed page carries the reason it failed.
    """
    http_status = None
    try:
        response = await client.get(url, timeout=15.0, follow_redirects=True)
        http_status = response.status_code

        # Check before parsing: an error page ("Access Denied", "404 Not Found") can have
        # enough text to pass as an article, and would then be cached, embedded and indexed.
        # The status is the final one, after redirects.
        if http_status != 200:
            raise ScrapeFailed(f"HTTP {http_status}")
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type and content_type not in HTML_CONTENT_TYPES:
            raise ScrapeFailed(f"not HTML ({content_type})")

        # Parsing runs in a thread so the event loop keeps serving other requests
        full_text = await asyncio.to_thread(extract_text, response.text)

        if not has_enough_text(full_text):
            raise ScrapeFailed("too little text")

        return {
            "url": url,
            "full_text": full_text,
            "status": "success",
            "http_status": http_status,
            "reason": None,
        }

    except Exception as e:
        reason = _failure_reason(e)
        print(f"[WARN] Scrape failed ({reason}): {url}")
        return {
            "url": url,
            "full_text": None,
            "status": "failed",
            "http_status": http_status,
            "reason": reason,
        }

def _scrape_summary(results: list[dict]) -> str:
    """'Scrape: 7 tried, 5 succeeded (71%). Failed: timeout x1, too little text x1.'"""
    tried = len(results)
    succeeded = sum(res['status'] == 'success' for res in results)
    summary = f"Scrape: {tried} tried, {succeeded} succeeded ({succeeded / tried:.0%})."

    reasons = Counter(res['reason'] for res in results if res['status'] == 'failed')
    if reasons:
        summary += " Failed: " + ", ".join(f"{reason} x{n}" for reason, n in reasons.most_common()) + "."
    return summary

async def scrape_results(urls: list[str]) -> list[dict]:
    """
    Scrapes the given URLs in parallel and returns one result per URL, failures included.
    """
    if not urls:
        return []

    async with httpx.AsyncClient(verify=SSL_CONTEXT, headers=SCRAPE_HEADERS,
                                 limits=httpx.Limits(max_connections=20)) as client:
        print(f"[INFO] Starting parallel scrape for {len(urls)} articles...")
        results = await asyncio.gather(*(fetch_one_article(client, url) for url in urls))

    print(f"[INFO] {_scrape_summary(results)}")
    return results

async def scrape_pages(urls: list[str]) -> dict[str, str]:
    """
    Scrapes the given URLs in parallel.
    Returns {url: full_text} for the pages that yielded enough text; failed pages are left out.
    The pipeline calls this only for URLs whose text isn't already cached.
    """
    results = await scrape_results(urls)
    return {res['url']: res['full_text'] for res in results if res['status'] == 'success'}