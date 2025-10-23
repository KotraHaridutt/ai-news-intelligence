from fastapi import FastAPI
from .gnews_fetcher import fetch_news
from .article_extractor import extract_text
from .gemini_summarizer import summarize_with_gemini
from .response_formatter import format_response

app = FastAPI(title="AI News Intelligence")

@app.get("/news")
def get_news(query: str):
    articles = fetch_news(query)
    if not articles:
        return {"error": "No news articles found."}

    combined_text = ""
    for a in articles:
        text = extract_text(a["url"])
        if text:
            combined_text += text + "\n\n"

    if not combined_text:
        return {"error": "Failed to extract text from all articles."}

    summary = summarize_with_gemini(query, combined_text)
    response = format_response(query, summary, articles)
    return response
