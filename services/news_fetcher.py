import asyncio
import os
import httpx
from dotenv import load_dotenv
import trafilatura
from bs4 import BeautifulSoup

# Load your .env file to get the API key
load_dotenv()
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY") or os.getenv("NEWS_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not GNEWS_API_KEY and not NEWS_API_KEY:
    print("ERROR: GNEWS_API_KEY/NEWS_API_KEY not found in .env file. Please add it.")

async def fetch_news(query: str) -> list[dict]:
    """
    Fetches news articles using the configured provider.
    """
    if not GNEWS_API_KEY and not NEWS_API_KEY:
        return []

    if GNEWS_API_KEY:
        api_url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": "en",
            "max": 30,
            "token": GNEWS_API_KEY,
        }
        provider_name = "GNews"
    else:
        api_url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": 30,
            "language": "en",
            "sortBy": "relevancy",
        }
        provider_name = "NewsAPI"

    try:
        async with httpx.AsyncClient() as client:
            print(f"[INFO] Fetching real news from {provider_name} for: {query}")
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
        print(f"[ERROR] {provider_name} HTTP Error: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
        return []

async def fetch_one_article(client: httpx.AsyncClient, url: str) -> dict:
    """
    Asynchronously fetches and scrapes a single article using Trafilatura for clean text.
    """
    try:
        response = await client.get(url, timeout=15.0, follow_redirects=True)
        
        # 1. Try Trafilatura (Clean, Markdown-style extraction)
        full_text = trafilatura.extract(response.text, include_comments=False, include_tables=False)
        
        # 2. Fallback to BeautifulSoup if Trafilatura fails
        if not full_text:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Try to get paragraphs specifically to avoid script/nav noise
            paragraphs = soup.find_all('p')
            full_text = " ".join([p.get_text(strip=True) for p in paragraphs])
            
        if not full_text or len(full_text) < 100:
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

async def fetch_all_articles(query: str) -> list[dict]:
    """
    Orchestrates the entire process:
    1. Fetches article list from NewsAPI.
    2. Scrapes full text for each article in parallel.
    3. Merges the data.
    """
    articles_from_api = await fetch_news(query)
    
    if not articles_from_api:
        print("[INFO] No articles found by NewsAPI.")
        return []

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=20)) as client:
        tasks = []
        for article in articles_from_api:
            tasks.append(fetch_one_article(client, article['url']))
            
        print(f"[INFO] Starting parallel scrape for {len(tasks)} articles...")
        scraped_results = await asyncio.gather(*tasks)
        print("[INFO] Parallel scrape complete.")
        
    # Merge the API data with the scraped full_text
    scraped_map = {res['url']: res['full_text'] for res in scraped_results if res['status'] == 'success'}
    
    final_articles = []
    for api_article in articles_from_api:
        url = api_article['url']
        if url in scraped_map:
            api_article['full_text'] = scraped_map[url]
            # Use real published_at, no hardcoded date
            final_articles.append(api_article)

    return final_articles