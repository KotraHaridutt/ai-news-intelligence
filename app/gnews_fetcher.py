import requests
from .config import GNEWS_API_KEY

def fetch_news(query, max_results=3):
    """
    Fetch news articles from GNews API related to the query.
    """
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max={max_results}&token={GNEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "articles" not in data:
        return []

    articles = []
    for item in data["articles"]:
        articles.append({
            "title": item["title"],
            "url": item["url"],
            "source": item["source"]["name"],
            "published": item["publishedAt"]
        })
    return articles
