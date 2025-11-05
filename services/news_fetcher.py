import asyncio
import httpx  # The async-capable requests library
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load your .env file to get the API key
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not NEWS_API_KEY:
    print("ERROR: NEWS_API_KEY not found in .env file. Please add it.")


async def fetch_news(query: str) -> list[dict]:
    """
    Fetches real news articles from NewsAPI.org.
    This replaces the dummy function.
    """
    if not NEWS_API_KEY:
        return []

    # This is the URL for NewsAPI.org's "everything" endpoint
    api_url = "https://newsapi.org/v2/everything"
    
    params = {
        "q": query,           # Your search query
        "apiKey": NEWS_API_KEY, # Your secret API key
        "pageSize": 30,       # Get 30 articles
        "language": "en",     # Get English articles
        "sortBy": "relevancy" # Get most relevant first
    }

    try:
        # Use httpx.AsyncClient for our async app
        async with httpx.AsyncClient() as client:
            print(f"[INFO] Fetching real news from NewsAPI.org for: {query}")
            response = await client.get(api_url, params=params, timeout=10.0)
            
            # Raise an error if the request failed
            response.raise_for_status() 
            
            data = response.json()
            
            # --- IMPORTANT ---
            # We now return the list of article objects directly.
            # NewsAPI gives us the 'url', 'title', and 'description' (snippet)
            # so we don't need a separate list of dummy URLs.
            
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("description"), # 'description' is the snippet
                    "source_name": item.get("source", {}).get("name"),
                    "published_at": item.get("publishedAt")
                })
            
            return articles

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] NewsAPI HTTP Error: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch news: {e}")
        return []

# --- This is the function that runs in parallel ---
async def fetch_one_article(client: httpx.AsyncClient, url: str) -> dict:
    """
    Asynchronously fetches and scrapes a single article.
    """
    try:
        # 1. The async request. 'await' pauses this function
        #    until the request is complete, letting others run.
        response = await client.get(url, timeout=10.0)
        
        # 2. Parse the content
        # TODO: Replace this simple parser with your robust scraping logic
        soup = BeautifulSoup(response.text, 'html.parser')
        full_text = soup.body.get_text(separator=' ', strip=True)
        
        # Simulate extracting other data
        snippet = full_text[:150] + "..."
        published_date = "2025-11-05" # TODO: Extract real date
        
        return {
            "url": url,
            "full_text": full_text,
            "snippet": snippet,
            "published_date": published_date,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {
            "url": url,
            "full_text": None,
            "snippet": None,
            "published_date": None,
            "status": "failed"
        }

# --- This is the main function we will call from app.py ---
async def fetch_all_articles(query: str) -> list[dict]:
    """
    Orchestrates the entire process:
    1. Fetches article list from NewsAPI.
    2. Scrapes full text for each article in parallel.
    3. Merges the data.
    """
    
    # 1. Get the list of articles (with metadata) from NewsAPI
    articles_from_api = await fetch_news(query)
    
    if not articles_from_api:
        print("[INFO] No articles found by NewsAPI.")
        return []

    # 2. Create an async client to manage all connections
    async with httpx.AsyncClient() as client:
        
        # 3. Create a list of "tasks" - one for each URL.
        tasks = []
        for article in articles_from_api:
            tasks.append(fetch_one_article(client, article['url']))
            
        # 4. Run all tasks concurrently and gather the results.
        print(f"Starting parallel scrape for {len(tasks)} articles...")
        scraped_results = await asyncio.gather(*tasks)
        print("...Parallel scrape complete.")
        
    # 5. Merge the API data (title, source) with the scraped data (full_text)
    scraped_map = {res['url']: res for res in scraped_results if res['status'] == 'success'}
    
    final_articles = []
    for api_article in articles_from_api:
        url = api_article['url']
        
        if url in scraped_map:
            scraped_data = scraped_map[url]
            merged_article = api_article.copy() 
            merged_article['full_text'] = scraped_data['full_text']
            merged_article['snippet'] = scraped_data['snippet'] 
            final_articles.append(merged_article)

    return final_articles