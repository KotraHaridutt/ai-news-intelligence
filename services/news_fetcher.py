import asyncio
import os
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

async def fetch_one_article(client: httpx.AsyncClient, url: str) -> dict:
    """
    Asynchronously fetches and scrapes a single article using Trafilatura for clean text.
    """
    try:
        response = await client.get(url, timeout=15.0, follow_redirects=True)

        # Parsing runs in a thread so the event loop keeps serving other requests
        full_text = await asyncio.to_thread(extract_text, response.text)

        if not has_enough_text(full_text):
            raise ValueError("Insufficient text extracted")
            
        return {
            "url": url,
            "full_text": full_text,
            "status": "success"
        }
        
    except Exception as e:
        print(f"[WARN] Error fetching/parsing {url}: {e}")
        return {
            "url": url,
            "full_text": None,
            "status": "failed"
        }

async def scrape_pages(urls: list[str]) -> dict[str, str]:
    """
    Scrapes the given URLs in parallel.
    Returns {url: full_text} for the pages that yielded enough text; failed pages are left out.
    The pipeline calls this only for URLs whose text isn't already cached.
    """
    if not urls:
        return {}

    async with httpx.AsyncClient(verify=SSL_CONTEXT, limits=httpx.Limits(max_connections=20)) as client:
        print(f"[INFO] Starting parallel scrape for {len(urls)} articles...")
        scraped_results = await asyncio.gather(*(fetch_one_article(client, url) for url in urls))
        print("[INFO] Parallel scrape complete.")

    return {res['url']: res['full_text'] for res in scraped_results if res['status'] == 'success'}