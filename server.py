import sys
# Log lines include article titles from news sites. Where the output encoding can't write a
# character (e.g. "₹" when output is redirected on Windows, cp1252), print() would raise and
# fail the build that logged it. Unwritable characters are escaped instead ("\u20b9").
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="backslashreplace")

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
import markdown2

# --- Our Services ---
# observability first: it decides whether traces are sent before any model is called
from services import observability
from services import cache_service, errors, follow_service
from services.errors import AppError
from services.pipeline_service import get_pipeline_data, build_time
from services.rag_service import run_rag_query_with_store

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown: send the traces still queued. Render's free plan stops an idle app, and
    # traces queued at that moment would be lost. Waits at most 5 s; never raises.
    await asyncio.to_thread(observability.shutdown)

app = FastAPI(title="AI News Intelligence API", lifespan=lifespan)

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

class FollowRequest(BaseModel):
    query: str
    # The answers on the page: {"report": id, "timeline": id, "contradictions": id}
    answer_ids: dict[str, str | None] = {}

# --- Errors ---
# Every failure reaches the page as {"detail": "<message for the user>"} with a real error
# status, never as 200 with error text. The page shows `detail` in its error box.

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    print(f"[WARN] {request.url.path} -> {exc.status}: {exc.log}")
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return await app_error_handler(request, errors.invalid_request(str(exc.errors())[:300]))

@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    # A bug. Starlette re-raises the exception after this response is sent,
    # so uvicorn still logs the full traceback.
    print(f"[ERROR] {request.url.path} -> 500: unexpected {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": errors.UNEXPECTED_MESSAGE})

def require_query(query: str) -> str:
    if not query or not query.strip():
        raise errors.empty_query()
    return query

def render_markdown(text) -> str:
    """
    Markdown -> HTML for the LLM's answer. safe_mode="escape" turns any raw HTML in the
    answer into plain text and replaces javascript: links with "#". The answer is built from
    scraped web pages, so a page could steer the model into echoing HTML/script (XSS).
    """
    return markdown2.markdown(text or "", safe_mode="escape")

async def remember_answer(query: str, view: str, text: str, built_at) -> str | None:
    """
    Saves an answer the user is shown, so that following the topic later can store what it
    said as "already told" (feature 1). The page gets the id and sends it back on Follow; ids,
    not the topic, so two people reading the same topic never get each other's answer.
    Returns None if it couldn't be saved (no Redis, or a failed write): the answer is still
    shown, it just can't be followed.
    """
    answer_id = cache_service.new_id()
    saved = await cache_service.set_answer(answer_id, {
        "query": cache_service.normalize_query(query),
        "view": view,
        "text": text,
        "built_at": built_at,
        "created_at": build_time(),
    })
    return answer_id if saved else None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    start_time = time.time()
    query = require_query(req.query)

    # Both raise AppError on failure; the handler above turns it into the error response
    data = await get_pipeline_data(query)
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "report")

    final_response = {
        "summary_html": render_markdown(rag_data["answer"]),
        "cited_sources": rag_data["sources"],
        "related_links": data["related_links"],
        "total_articles_found": len(data["articles"]),
        "answer_id": await remember_answer(query, "report", rag_data["answer"], data.get("built_at")),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }
    
    return final_response

@app.post("/api/timeline")
async def api_timeline(req: QueryRequest):
    start_time = time.time()
    query = require_query(req.query)

    data = await get_pipeline_data(query)
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "timeline")

    return {
        "query": query,
        "timeline_html": render_markdown(rag_data["answer"]),
        "cited_sources": rag_data["sources"],
        "answer_id": await remember_answer(query, "timeline", rag_data["answer"], data.get("built_at")),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

@app.post("/api/contradictions")
async def api_contradictions(req: QueryRequest):
    start_time = time.time()
    query = require_query(req.query)

    data = await get_pipeline_data(query)
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "contradictions")

    return {
        "query": query,
        "analysis_html": render_markdown(rag_data["answer"]),
        "cited_sources": rag_data["sources"],
        "answer_id": await remember_answer(query, "contradictions", rag_data["answer"], data.get("built_at")),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

# --- Following a topic (feature 1): services/follow_service.py ---

@app.post("/api/follow")
async def api_follow(req: FollowRequest):
    """Saves the report on the page (V1) as a followed topic. No GNews or chat call."""
    start_time = time.time()
    query = require_query(req.query)
    result = await follow_service.follow(query, req.answer_ids)
    return {**result, "time_taken": f"{time.time() - start_time:.2f}s"}

@app.post("/api/follow/{watch_id}/check")
async def api_follow_check(watch_id: str):
    """Check for updates: only what is new since the last look (or the stored result, within 30 min)."""
    start_time = time.time()
    result = await follow_service.check(watch_id, render_markdown)
    return {**result, "time_taken": f"{time.time() - start_time:.2f}s"}

@app.get("/api/follow/{watch_id}")
async def api_follow_log(watch_id: str):
    """The followed topic's story log and its last result. Redis only."""
    return await follow_service.story_log(watch_id)

@app.delete("/api/follow/{watch_id}")
async def api_unfollow(watch_id: str):
    return await follow_service.unfollow(watch_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)