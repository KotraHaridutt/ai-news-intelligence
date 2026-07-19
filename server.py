from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
import markdown2

# --- Our Services ---
from services.news_fetcher import fetch_all_articles
from services.clustering_service import group_by_theme
from services.rag_service import build_vector_store, run_rag_query_with_store

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI News Intelligence API")

# Setup templates and static files
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# Ensure static directory exists before mounting
import os
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --- Simple In-Memory Cache ---
# Storing: { "query_string": {"articles": [...], "vector_store": <FAISS>, "related_links": [...], "timestamp": float} }
CACHE = {}
CACHE_TTL = 600 # 10 minutes cache duration

class QueryRequest(BaseModel):
    query: str

async def get_or_create_pipeline_data(query: str):
    """
    Fetches, clusters, and builds the FAISS index. Caches the result.
    """
    now = time.time()
    
    # Check Cache
    if query in CACHE:
        cached_data = CACHE[query]
        if now - cached_data["timestamp"] < CACHE_TTL:
            print(f"[INFO] Cache HIT for query: '{query}'")
            return cached_data
            
    print(f"[INFO] Cache MISS. Running full pipeline for: '{query}'")
    
    # 1. Fetch
    try:
        articles = await fetch_all_articles(query)
    except Exception as e:
        print(f"[ERROR] Async fetch failed: {e}")
        return None
        
    if not articles:
        return {"articles": [], "vector_store": None, "related_links": []}

    # 2. Cluster
    articles = group_by_theme(articles)
    
    # 3. Build "Related Links"
    unique_themes = {}
    for article in articles:
        theme = article.get('theme_id', -1)
        if theme != -1 and theme not in unique_themes:
            unique_themes[theme] = article
            
    top_themed_articles = list(unique_themes.values())[:5]
    related_links = [
        {
            "title": art.get("title"), 
            "url": art.get("url"), 
            "source": art.get("source_name")
        }
        for art in top_themed_articles
    ]
    
    # 4. Build Vector Store (Once per query)
    vector_store = build_vector_store(articles)
    
    # Save to Cache
    pipeline_data = {
        "articles": articles,
        "vector_store": vector_store,
        "related_links": related_links,
        "timestamp": now
    }
    CACHE[query] = pipeline_data
    
    return pipeline_data


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    start_time = time.time()
    query = req.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    data = await get_or_create_pipeline_data(query)
    
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to fetch articles")
    if not data["articles"] or data["vector_store"] is None:
        raise HTTPException(status_code=404, detail="No articles with sufficient content found")

    # Run RAG for main summary
    rag_data = run_rag_query_with_store(query, data["vector_store"], "report")

    final_response = {
        "summary_html": markdown2.markdown(rag_data.get("answer", "No answer generated.")),
        "cited_sources": rag_data.get("sources", []),
        "related_links": data["related_links"],
        "total_articles_found": len(data["articles"]),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }
    
    return final_response

@app.post("/api/timeline")
async def api_timeline(req: QueryRequest):
    start_time = time.time()
    query = req.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    data = await get_or_create_pipeline_data(query)
    
    if data is None or not data["articles"] or data["vector_store"] is None:
        raise HTTPException(status_code=404, detail="No articles found")

    # Run RAG for timeline
    rag_data = run_rag_query_with_store(query, data["vector_store"], "timeline")
    
    return {
        "query": query,
        "timeline_html": markdown2.markdown(rag_data.get("answer")),
        "cited_sources": rag_data.get("sources"),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

@app.post("/api/contradictions")
async def api_contradictions(req: QueryRequest):
    start_time = time.time()
    query = req.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    data = await get_or_create_pipeline_data(query)
    
    if data is None or not data["articles"] or data["vector_store"] is None:
        raise HTTPException(status_code=404, detail="No articles found")

    # Run RAG for contradictions
    rag_data = run_rag_query_with_store(query, data["vector_store"], "contradictions")
    
    return {
        "query": query,
        "analysis_html": markdown2.markdown(rag_data.get("answer")),
        "cited_sources": rag_data.get("sources"),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)