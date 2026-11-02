import asyncio
import contextlib
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
import ssl
import certifi
import httpx
from dotenv import load_dotenv
import trafilatura
import lxml.etree
import lxml.html
from bs4 import BeautifulSoup

from services import errors, observability

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

def _found_summary(articles):
    """The GNews span's output: how many, from which outlets, and the titles (no page text)."""
    return {"found": len(articles), "outlets": sorted({a.get("source_name") or "?" for a in articles}),
            "titles": [(a.get("title") or "")[:100] for a in articles]}

@observability.traced("gnews.search", run_type="tool", inputs=lambda a: {"query": a.get("query"), "from": a.get("window_from")},
                      outputs=_found_summary)
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

# --- Hidden page text (feature 3, D #30) ---
# A page can hide text from readers and still have it extracted: an instruction to the AI in a
# display:none block or an aria-hidden element is invisible on the page and plain text to the
# model. Such elements are dropped before extraction. Decided by the HTML's structure (attributes
# and inline styles), never by the words: a reader can't see this text, so it isn't the news they read.
#
# Except a hidden ARTICLE: Indian Express sends its premium article to everyone inside a
# display:none block and shows it with a script (D1 §3). An element with at least
# KEEP_HIDDEN_PARAGRAPHS paragraphs and KEEP_HIDDEN_CHARS characters is kept: dropping it cut that
# article from 1,664 to 340 characters, and its text still goes through the chunk scan (#32).
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d]*[1-9])|"
                           r"opacity\s*:\s*0(?![.\d]*[1-9])", re.IGNORECASE)
KEEP_HIDDEN_PARAGRAPHS = 2
KEEP_HIDDEN_CHARS = 500


def _is_hidden(element) -> bool:
    return (element.tag == "template" or element.get("hidden") is not None or element.get("aria-hidden") == "true"
            or bool(_HIDDEN_STYLE.search(element.get("style") or "")))


def drop_hidden(doc) -> dict:
    """
    Removes hidden elements and comments from a parsed page (lxml), in place. Returns
    {"dropped": elements, "chars": text removed, "kept_articles": hidden article blocks kept}.
    """
    stats = {"dropped": 0, "chars": 0, "kept_articles": 0}
    for element in doc.xpath('//*[@hidden] | //*[@aria-hidden="true"] | //template | //*[@style]'):
        if not _is_hidden(element) or element.getparent() is None:
            continue                   # visible, the root, or already gone with a hidden parent
        text = element.text_content() or ""
        if len(element.findall(".//p")) >= KEEP_HIDDEN_PARAGRAPHS and len(text.strip()) >= KEEP_HIDDEN_CHARS:
            stats["kept_articles"] += 1
            continue
        stats["dropped"] += 1
        stats["chars"] += len(text)
        element.drop_tree()
    for comment in doc.xpath("//comment()"):
        if comment.getparent() is not None:
            stats["chars"] += len(comment.text or "")
            comment.drop_tree()
    return stats


def _parse(html: str):
    try:
        return lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return None


def _page_date(doc) -> str | None:
    """
    The page's own publish date (YYYY-MM-DD) from its metadata, for the date check (feature 3 #53).
    trafilatura asks htmldate for the original date (not the last update), never later than today.
    Measured on 72 pages (G §3): 35 ms median, 161 ms at most; the extracted text is the same with or
    without this step. Never raises: a page without a readable date gives None.
    """
    try:
        meta = trafilatura.extract_metadata(doc)
    except Exception:
        return None
    value = getattr(meta, "date", None) if meta is not None else None
    return value if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else None


def extract_page(html: str) -> tuple[str | None, str | None]:
    """(article text, the page's own publish date). CPU-bound, so callers run it in a thread."""
    doc = _parse(html)
    page_date = _page_date(doc) if doc is not None else None
    return _extract(html, doc), page_date


def extract_text(html: str):
    """The article body of a page's HTML (extract_page without the date)."""
    return _extract(html, _parse(html))


def _extract(html: str, doc):
    """
    Extracts the article body from a page's HTML. CPU-bound, so callers run it in a thread.
    Hidden elements are dropped first (D #30); the cleaned tree goes to trafilatura as it is,
    without being written back to HTML and parsed again.
    """
    if doc is not None:
        drop_hidden(doc)
    # 1. Try Trafilatura (Clean, Markdown-style extraction)
    full_text = trafilatura.extract(doc if doc is not None else html, include_comments=False, include_tables=False)

    # 2. Fallback: the paragraphs, from the cleaned tree (BeautifulSoup if lxml couldn't parse the page)
    if not full_text:
        if doc is not None:
            paragraphs = [" ".join(p.text_content().split()) for p in doc.iter("p")]
        else:
            # Try to get paragraphs specifically to avoid script/nav noise
            paragraphs = [p.get_text(strip=True) for p in BeautifulSoup(html, 'html.parser').find_all('p')]
        full_text = " ".join(p for p in paragraphs if p)
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

@observability.traced("scrape", run_type="tool", inputs=lambda a: {"url": a["url"]},
                      outputs=lambda r: {"status": r["status"], "http_status": r["http_status"], "reason": r["reason"],
                                         "chars": len(r["full_text"] or "")})
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
        full_text, page_date = await asyncio.to_thread(extract_page, response.text)

        if not has_enough_text(full_text):
            raise ScrapeFailed("too little text")

        return {
            "url": url,
            "full_text": full_text,
            "status": "success",
            "http_status": http_status,
            "reason": None,
            # Provenance (feature 3 #50, #51, #53): where the page ended up after redirects, when it
            # was read, and the date the page gives itself
            "final_url": str(response.url),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "page_date": page_date,
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

async def scrape_pages(urls: list[str]) -> dict[str, dict]:
    """
    Scrapes the given URLs in parallel.
    Returns {url: record} for the pages that yielded enough text; failed pages are left out. A record
    is what cache_service stores: {"text", "final_url", "fetched_at", "page_date"} (feature 3 #50).
    The pipeline calls this only for URLs whose text isn't already cached.
    """
    results = await scrape_results(urls)
    return {res['url']: {"text": res['full_text'], "final_url": res.get('final_url'), "fetched_at": res.get('fetched_at'),
                         "page_date": res.get('page_date')}
            for res in results if res['status'] == 'success'}