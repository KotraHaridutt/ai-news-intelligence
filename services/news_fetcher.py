import asyncio
import os
from collections import Counter
import ssl
import certifi
import httpx
from dotenv import load_dotenv
import trafilatura
from bs4 import BeautifulSoup

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

# Shared minimum article length: scraping, clustering and indexing all use this
# so an article is either usable everywhere or nowhere.
MIN_TEXT_LEN = 100

def has_enough_text(text) -> bool:
    return bool(text) and len(text) >= MIN_TEXT_LEN

async def fetch_news(query: str) -> list[dict]:
    """
    Fetches the article list for a query from GNews.
    """
    if not GNEWS_API_KEY:
        return []

    api_url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": GNEWS_MAX_ARTICLES,
        "token": GNEWS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(verify=SSL_CONTEXT) as client:
            print(f"[INFO] Fetching real news from GNews for: {query}")
            response = await client.get(api_url, params=params, timeout=10.0)
            response.raise_for_status()

            data = response.json()
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

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] GNews HTTP Error: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
        return []

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

        # Parsing runs in a thread so the event loop keeps serving other requests
        full_text = await asyncio.to_thread(extract_text, response.text)

        if not has_enough_text(full_text):
            # Name the status when it wasn't 200: an error page that is too short also lands here
            raise ScrapeFailed("too little text" if http_status == 200 else f"too little text (HTTP {http_status})")

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

    # A page that succeeded with a non-200 status is an error page used as an article
    accepted_errors = Counter(res['http_status'] for res in results
                              if res['status'] == 'success' and res['http_status'] != 200)
    if accepted_errors:
        summary += " Used as articles despite an error status: " + ", ".join(
            f"HTTP {code} x{n}" for code, n in accepted_errors.most_common()) + "."
    return summary

async def scrape_results(urls: list[str]) -> list[dict]:
    """
    Scrapes the given URLs in parallel and returns one result per URL, failures included.
    """
    if not urls:
        return []

    async with httpx.AsyncClient(verify=SSL_CONTEXT, limits=httpx.Limits(max_connections=20)) as client:
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