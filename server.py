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

# --- Our Services ---
# observability first: it decides whether traces are sent before any model is called
from services import observability
from services import cache_service, errors, feedback_service, follow_service, safety
from services.errors import AppError
from services.pipeline_service import get_pipeline_data, build_time
from services.rag_service import run_rag_query_with_store
from services.safety import output, privacy
from services.safety.input_rails import check_query, checked_query

BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Safety (feature 3): loads the fallback classifiers, reads today's Clef neurons, one log line
    await safety.startup()
    yield
    # Shutdown: send the traces still queued. Render's free plan stops an idle app, and
    # traces queued at that moment would be lost. Waits at most 5 s; never raises.
    await asyncio.to_thread(observability.shutdown)

app = FastAPI(title="AI News Intelligence API", lifespan=lifespan)

# --- Security headers on every response (feature 3 #70) ---
# The answer is model output written from scraped pages, so the page assumes some text in it could be
# hostile. The CSP lets the page run only its own script and stylesheet (static/app.js, the Tailwind
# CSS built into static/tailwind.css), load nothing from other sites (an image link in an answer can't
# send data out, #68), talk only to this server, and not be framed. Inline scripts, inline styles and
# onclick="" attributes are refused by the browser, so injected HTML that got past the server still
# couldn't run. FastAPI's /docs and /redoc pages load Swagger from a CDN: they get the headers without
# the CSP (Section M #101 turns them off in prod).
CONTENT_SECURITY_POLICY = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
                           "font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",      # a JSON or text response is never run as a script or page
    "Referrer-Policy": "no-referrer",          # opening a source sends it nothing about this app
    "X-Frame-Options": "DENY",                 # frame-ancestors for older browsers
}
CSP_EXEMPT = ("/docs", "/redoc")


def security_headers(path: str) -> dict:
    headers = dict(SECURITY_HEADERS)
    if not path.startswith(CSP_EXEMPT):
        headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    return headers


class SecurityHeaders:
    """Pure ASGI middleware: adds the headers to every response, streamed ones and static files included.
    A 500 is sent by Starlette's outermost layer, past this one: its handler adds them itself."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        extra = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in security_headers(scope.get("path", "")).items()]

        async def with_headers(message):
            if message["type"] == "http.response.start":
                names = {name.lower() for name, _ in message.get("headers", [])}
                message["headers"] = list(message.get("headers", [])) + [h for h in extra if h[0] not in names]
            await send(message)
        await self.app(scope, receive, with_headers)


app.add_middleware(SecurityHeaders)

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

MAX_ANSWER_IDS = 3

class FeedbackRequest(BaseModel):
    # A reader's rating of one answer (feature 2, Section J): 1 for 👍, -1 for 👎
    answer_id: str
    score: int
    comment: str | None = None

# --- Errors ---
# Every failure reaches the page as {"detail": "<message for the user>"} with a real error
# status, never as 200 with error text. The page shows `detail` in its error box.
# With tracing on, a traced request's error also carries "ref": the end of its trace id, shown
# next to the message, so a reported error can be found in LangSmith (feature 2, #28).

def error_content(message: str, exc: Exception) -> dict:
    content = {"detail": message}
    ref = getattr(exc, "trace_ref", None)
    if ref:
        content["ref"] = ref
    return content

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    # The reason may quote the query (or what GNews said about it): personal data tagged (feature 3 #61)
    print(f"[WARN] {request.url.path} -> {exc.status}: {privacy.for_log(exc.log)}")
    return JSONResponse(status_code=exc.status, content=error_content(exc.message, exc), headers=exc.headers)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    log = str(exc.errors())[:300]
    # The generic message asks for a topic, which makes no sense next to a 👍 / 👎
    error = errors.rating_invalid(log) if request.url.path == "/api/feedback" else errors.invalid_request(log)
    return await app_error_handler(request, error)

@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    # A bug. Starlette re-raises the exception after this response is sent,
    # so uvicorn still logs the full traceback.
    print(f"[ERROR] {request.url.path} -> 500: unexpected {type(exc).__name__}: {privacy.for_log(exc)}")
    return JSONResponse(status_code=500, content=error_content(errors.UNEXPECTED_MESSAGE, exc),
                        headers=security_headers(request.url.path))

def render_markdown(text, source_urls=()) -> str:
    """
    Markdown -> HTML for model output, without the trace event (scripts and tests; the views use
    view_answer). safe_mode="escape" turns any raw HTML into plain text and javascript: links into "#";
    the nh3 allowlist keeps only the tags markdown makes; links to pages that aren't in source_urls
    and images are removed in enforce mode (SAFETY_OUTPUT, feature 3 I).
    """
    return output.render(text, output.allowed_urls(source_urls)).html


async def view_answer(view: str, rag_data: dict, field: str) -> dict:
    """The view's answer as HTML (field), and its notes when there are any: a cut answer (feature 3 #66),
    statements that don't match the sources (#77), the kind of news (#80)."""
    shown = await output.answer(view, rag_data["answer"], rag_data.get("source_urls") or [],
                                sources=rag_data.get("sources"), finish_reason=rag_data.get("finish_reason"))
    notes = [*shown["notes"], *(rag_data.get("notes") or [])]
    return {field: shown["html"], **({"answer_notes": notes} if notes else {})}

async def remember_answer(query: str, view: str, text: str, built_at) -> dict:
    """
    Saves an answer the user is shown, so that following the topic later can store what it
    said as "already told" (feature 1). The page gets the id and sends it back on Follow; ids,
    not the topic, so two people reading the same topic never get each other's answer.
    answer_id is None if it couldn't be saved (no Redis, or a failed write): the answer is still
    shown, it just can't be followed or rated. The record also keeps which model, prompt version
    and settings wrote it (feature 2 #40), from this request's totals, and the trace that wrote
    it (#68), so a rating can go on that trace. The page never sees the trace id, only can_rate.
    """
    totals = observability.current_totals() or {}
    trace = observability.sent_trace()
    answer_id = cache_service.new_id()
    saved = await cache_service.set_answer(answer_id, {
        "query": cache_service.normalize_query(query),
        "view": view,
        "text": text,
        "built_at": built_at,
        "created_at": build_time(),
        **{key: totals[key] for key in ("answered_by", "prompt_version", "config_version") if key in totals},
        **(trace or {}),
    })
    return {"answer_id": answer_id if saved else None, "can_rate": bool(saved and trace)}


def query_notice(checked) -> dict:
    """
    Feature 3 #58: when personal data was taken out of the topic (enforce mode), the page is told
    what (query_notice) and gets the topic as searched (query), so the tabs and Follow use that one.
    """
    if not checked.notice:
        return {}
    return {"query": checked.text, "query_notice": checked.notice}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/query")
@observability.request_span("POST /query", view="report")
async def query_endpoint(req: QueryRequest):
    start_time = time.time()
    # Cleaned and checked before any GNews, scrape or chat call (feature 3, Section C): from here on
    # `query` is the search form ("?" "!" removed, "-" a space, hidden characters out)
    checked = await checked_query(req.query)
    query = checked.text

    # Both raise AppError on failure; the handler above turns it into the error response
    data = await get_pipeline_data(query)
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "report")

    final_response = {
        **query_notice(checked),
        **await view_answer("report", rag_data, "summary_html"),
        "cited_sources": rag_data["sources"],
        "related_links": data["related_links"],
        "total_articles_found": len(data["articles"]),
        **await remember_answer(query, "report", rag_data["answer"], data.get("built_at")),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }
    
    return final_response

@app.post("/api/timeline")
@observability.request_span("POST /api/timeline", view="timeline")
async def api_timeline(req: QueryRequest):
    start_time = time.time()
    checked = await checked_query(req.query)
    query = checked.text

    data = await get_pipeline_data(query)
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "timeline")

    return {
        "query": query,
        **query_notice(checked),
        **await view_answer("timeline", rag_data, "timeline_html"),
        "cited_sources": rag_data["sources"],
        **await remember_answer(query, "timeline", rag_data["answer"], data.get("built_at")),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

@app.post("/api/contradictions")
@observability.request_span("POST /api/contradictions", view="contradictions")
async def api_contradictions(req: QueryRequest):
    start_time = time.time()
    checked = await checked_query(req.query)
    query = checked.text

    data = await get_pipeline_data(query)
    rag_data = await run_rag_query_with_store(query, data["vector_store"], "contradictions")

    return {
        "query": query,
        **query_notice(checked),
        **await view_answer("contradictions", rag_data, "analysis_html"),
        "cited_sources": rag_data["sources"],
        **await remember_answer(query, "contradictions", rag_data["answer"], data.get("built_at")),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }

# --- Following a topic (feature 1): services/follow_service.py ---
# Reading the story log and unfollowing are Redis-only and aren't traced.

@app.post("/api/follow")
@observability.request_span("POST /api/follow")
async def api_follow(req: FollowRequest):
    """Saves the report on the page (V1) as a followed topic. No GNews or chat call."""
    start_time = time.time()
    # The page sends at most one answer per view (feature 3 #21)
    if len(req.answer_ids) > MAX_ANSWER_IDS:
        raise errors.report_not_followable(f"{len(req.answer_ids)} answer ids (at most {MAX_ANSWER_IDS})")
    # The same cleaning as the views, so the topic's key matches the report's (the check is cached)
    query = await check_query(req.query)
    result = await follow_service.follow(query, req.answer_ids)
    return {**result, "time_taken": f"{time.time() - start_time:.2f}s"}

@app.post("/api/follow/{watch_id}/check")
@observability.request_span("POST /api/follow/{watch_id}/check")
async def api_follow_check(watch_id: str):
    """Check for updates: only what is new since the last look (or the stored result, within 30 min)."""
    start_time = time.time()
    result = await follow_service.check(watch_id)
    return {**result, "time_taken": f"{time.time() - start_time:.2f}s"}

@app.get("/api/follow/{watch_id}")
async def api_follow_log(watch_id: str):
    """The followed topic's story log and its last result. Redis only."""
    return await follow_service.story_log(watch_id)

@app.delete("/api/follow/{watch_id}")
async def api_unfollow(watch_id: str):
    return await follow_service.unfollow(watch_id)

# --- Rating an answer (feature 2, Section J): services/feedback_service.py ---

@app.post("/api/feedback")
@observability.request_span("POST /api/feedback")
async def api_feedback(req: FeedbackRequest):
    """👍 / 👎 (and an optional comment) on an answer the page is showing, saved on its trace."""
    return await feedback_service.rate(req.answer_id, req.score, req.comment)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)