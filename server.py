from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
import markdown2

# --- Our Services ---
from services.pipeline_service import get_pipeline_data
from services.rag_service import run_rag_query_with_store

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI News Intelligence API")

# Setup templates and static files
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# Ensure static directory exists before mounting
import os
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Caching lives in services/pipeline_service.py (Redis). The process keeps nothing between requests.

class QueryRequest(BaseModel):
    query: str

def render_markdown(text) -> str:
    """
    Markdown -> HTML for the LLM's answer. safe_mode="escape" turns any raw HTML in the
    answer into plain text and replaces javascript: links with "#". The answer is built from
    scraped web pages, so a page could steer the model into echoing HTML/script (XSS).
    """
    return markdown2.markdown(text or "", safe_mode="escape")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    start_time = time.time()
    query = req.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    data = await get_pipeline_data(query)
    
    if data is None:
        raise HTTPException(status_code=500, detail="Failed to fetch articles")
    if not data["articles"] or data["vector_store"] is None:
        raise HTTPException(status_code=404, detail="No articles with sufficient content found")

    # Run RAG for main summary
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "report")

    final_response = {
        "summary_html": render_markdown(rag_data.get("answer", "No answer generated.")),
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

    data = await get_pipeline_data(query)
    
    if data is None or not data["articles"] or data["vector_store"] is None:
        raise HTTPException(status_code=404, detail="No articles found")

    # Run RAG for timeline
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "timeline")
    
    return {
        "query": query,
        "timeline_html": render_markdown(rag_data.get("answer")),
        "cited_sources": rag_data.get("sources"),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

@app.post("/api/contradictions")
async def api_contradictions(req: QueryRequest):
    start_time = time.time()
    query = req.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    data = await get_pipeline_data(query)
    
    if data is None or not data["articles"] or data["vector_store"] is None:
        raise HTTPException(status_code=404, detail="No articles found")

    # Run RAG for contradictions
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "contradictions")
    
    return {
        "query": query,
        "analysis_html": render_markdown(rag_data.get("answer")),
        "cited_sources": rag_data.get("sources"),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)