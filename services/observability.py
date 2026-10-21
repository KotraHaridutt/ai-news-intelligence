import asyncio
import atexit
import functools
import hashlib
import os
import queue
import re
import subprocess
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import langsmith
from langsmith import utils as ls_utils
from langsmith.run_helpers import get_current_run_tree

# --- LLM observability (feature 2): LangSmith tracing, set up in one place ---
# Importing this module decides once whether traces are sent, where to, and with what
# attached to them. Every LangChain call (the chat models) and every @traceable function then
# uses the one client configured here, so the redaction below applies to all of them.
#
# Tracing is on only when LANGSMITH_TRACING is true AND LANGSMITH_API_KEY is set. Without a
# key, LangChain would still try to send and log a 401 for every batch, so it's switched off.
# Sending happens on LangSmith's background thread: a request never waits for it, and a
# LangSmith outage or a wrong key only logs a line (measured in feature 2 Section A).

load_dotenv()

APP_ENV = (os.getenv("APP_ENV") or "dev").strip().lower()
DEFAULT_PROJECT = f"ai-news-{APP_ENV}"
# Unsent traces are sent at shutdown, waiting at most this long (measured 0.4-0.8 s)
FLUSH_TIMEOUT_SECONDS = 5

_TRUE = {"1", "true", "yes", "on"}

# --- Redaction ---
# Masks secrets in run inputs, outputs, metadata and errors before they leave the process.
# Nothing leaked when measured (Section A), but the GNews key travels in the request URL
# (?token=...), and an httpx error keeps that URL on its request object. These rules catch it
# wherever it ends up, and the exact values below catch every key whatever its shape.
SECRET_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GNEWS_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
SECRET_PATTERNS = [
    # A key or token in a URL query
    (re.compile(r"(?i)([?&](?:token|key|api_key|apikey|access_token)=)[^&\s'\"<>]+"), r"\1[redacted]"),
    # Google API keys, LangSmith keys
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "[redacted google key]"),
    (re.compile(r"lsv2_(?:pt|sk)_[0-9A-Za-z]+_[0-9A-Za-z]+"), "[redacted langsmith key]"),
    # The password in a redis:// or rediss:// URL
    (re.compile(r"(rediss?://[^:/\s'\"@]*:)[^@\s'\"]+@"), r"\1[redacted]@"),
]
# Shorter values could match ordinary text
MIN_SECRET_LENGTH = 12


def _secret_values(environ) -> list[str]:
    values = [environ.get(name) or "" for name in SECRET_ENV_VARS]
    # The Redis password on its own: REDIS_URL is redis://user:password@host:port
    match = re.match(r"rediss?://[^:/]*:([^@]+)@", environ.get("REDIS_URL") or "")
    if match:
        values.append(match.group(1))
    # Longest first, so a value that contains another is replaced whole
    return sorted({v for v in values if len(v) >= MIN_SECRET_LENGTH}, key=len, reverse=True)


def make_redactor(environ=None):
    """
    Returns redact(text) -> text with every secret masked: the exact values of the keys in
    `environ` (default os.environ), then the patterns, which also catch keys we don't hold.
    """
    secrets = _secret_values(os.environ if environ is None else environ)

    def redact(text: str) -> str:
        for value in secrets:
            if value in text:
                text = text.replace(value, "[redacted]")
        for pattern, replacement in SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    return redact


# --- Size (Section C, #29) ---
# Measured in Section A: a traced answer stored the article text about five times (the chain's
# input, format_inputs, format_docs, the prompt, the LLM's messages), ~176 KB per trace. The LLM
# run keeps the full prompt and answer; in every other run, a string longer than TRIM_AT is cut
# to its first TRIM_TO characters. TRIM_AT is per string, and the copies are lists of ~1,500-
# character chunks, so it has to be below a chunk: 6,000 trimmed nothing (Section C measured
# 186 KB for one cold report, 4 chain runs holding all 10 chunks each).
TRIM_AT = 1000
TRIM_TO = 300
_MAX_DEPTH = 30
# The payload keys of an LLM run: chat model inputs, completion inputs, model outputs
_LLM_KEYS = ("messages", "prompts", "generations")


def _trimmed(text: str) -> str:
    return f"{text[:TRIM_TO]}... [trimmed: {len(text):,} characters; the full text is in the LLM run]"


def make_scrubber(environ=None):
    """
    Returns scrub(data) -> data, applied by the LangSmith client to every run's inputs,
    outputs, metadata and error before it's queued (it's the client's anonymizer, so it sees
    LangChain's runs too). Masks secrets in every string; trims long strings except in an LLM
    run's payload and in error text.
    """
    redact = make_redactor(environ)

    def walk(value, trim, depth):
        if isinstance(value, str):
            value = redact(value)
            return _trimmed(value) if trim and len(value) > TRIM_AT else value
        if depth >= _MAX_DEPTH:
            return value
        if isinstance(value, dict):
            return {k: walk(v, trim, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, trim, depth + 1) for v in value]
        return value

    def scrub(data):
        keep_whole = isinstance(data, dict) and (any(k in data for k in _LLM_KEYS) or set(data) == {"error"})
        return walk(data, not keep_whole, 0)
    return scrub


def short_hash(value: str) -> str:
    """
    A stable 12-character stand-in for an id that must not leave the server. A watch id works
    like a password for a followed topic (whoever has it can read and change it), so traces
    carry this instead: the same topic's checks still group together, and the hash can't be
    turned back into the id (feature 2 decision #14).
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _commit() -> str | None:
    """The deployed commit: Render sets RENDER_GIT_COMMIT; locally, ask git."""
    commit = os.getenv("RENDER_GIT_COMMIT")
    if commit:
        return commit[:7]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sampling_rate(environ) -> float | None:
    """LANGSMITH_TRACING_SAMPLING_RATE as a float, or None. LangSmith raises on a bad value
    when the client is created, which would stop the app from starting, so it's checked here."""
    raw = (environ.get("LANGSMITH_TRACING_SAMPLING_RATE") or "").strip()
    if not raw:
        return None
    try:
        rate = float(raw)
    except ValueError:
        rate = -1.0
    if not 0 <= rate <= 1:
        print(f"[WARN] LANGSMITH_TRACING_SAMPLING_RATE={raw!r} isn't a number from 0 to 1. Ignoring it (sending every trace).")
        environ.pop("LANGSMITH_TRACING_SAMPLING_RATE", None)
        return None
    return rate


# What configure_tracing decided, for the startup line and for tests
status = {"enabled": False, "project": None, "reason": "not configured yet"}
_client = None


def configure_tracing(environ=None) -> dict:
    """
    Decides whether traces are sent and sets up the one LangSmith client everything uses.
    Called once when this module is imported; tests call it again with their own environment.
    Writes LANGSMITH_TRACING / LANGSMITH_PROJECT back into the environment, because the
    LangChain tracer reads them on every call. Returns `status`.
    """
    global _client
    environ = os.environ if environ is None else environ
    wanted = (environ.get("LANGSMITH_TRACING") or environ.get("LANGCHAIN_TRACING_V2") or "").strip().lower() in _TRUE
    has_key = bool(environ.get("LANGSMITH_API_KEY") or environ.get("LANGCHAIN_API_KEY"))

    if not wanted or not has_key:
        if wanted:
            reason = "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is missing"
            print(f"[WARN] Tracing off: {reason}.")
        else:
            reason = "LANGSMITH_TRACING is not set to true"
            print(f"[INFO] Tracing off ({reason}).")
        environ["LANGSMITH_TRACING"] = "false"
        environ.pop("LANGCHAIN_TRACING_V2", None)
        # LangSmith caches environment reads; drop the cache so the values above are the ones read
        ls_utils.get_env_var.cache_clear()
        langsmith.configure(enabled=False)
        _client = None
        status.update(enabled=False, project=None, reason=reason)
        return status

    project = (environ.get("LANGSMITH_PROJECT") or environ.get("LANGCHAIN_PROJECT") or "").strip() or DEFAULT_PROJECT
    environ["LANGSMITH_PROJECT"] = project
    environ["LANGSMITH_TRACING"] = "true"
    # LangSmith's sending threads are non-daemon by default: at exit Python waits for them to
    # send everything left, with 60 s read timeouts and retries. A hung LangSmith kept a test
    # process alive for over 100 s (Section B). As daemon threads they never hold the exit;
    # shutdown() below sends what's queued, for at most FLUSH_TIMEOUT_SECONDS.
    environ.setdefault("LANGSMITH_USE_DAEMON", "true")
    rate = _sampling_rate(environ)
    ls_utils.get_env_var.cache_clear()
    commit = _commit()

    _client = langsmith.Client(anonymizer=make_scrubber(environ))
    metadata = {"app_env": APP_ENV}
    if commit:
        metadata["commit"] = commit
    langsmith.configure(client=_client, enabled=True, project_name=project, metadata=metadata)
    # Scripts (tests, evals) end without the app's shutdown hook: send their traces at exit too
    atexit.register(shutdown)

    status.update(enabled=True, project=project, reason=None, sampling_rate=1.0 if rate is None else rate, commit=commit)
    print(f"[INFO] Tracing on: project {project}, env {APP_ENV}, commit {commit or 'unknown'}, "
          f"sampling {status['sampling_rate']}.")
    return status


def flush(timeout: float = FLUSH_TIMEOUT_SECONDS):
    """Sends the traces and feedback still queued, waiting at most `timeout` seconds. Never raises."""
    if _client is None:
        return
    deadline = time.monotonic() + timeout
    _wait_for_feedback(timeout)
    try:
        _client.flush(timeout=max(0.1, deadline - time.monotonic()))
    except Exception as e:
        print(f"[WARN] Couldn't send the last traces to LangSmith: {type(e).__name__}: {e}")


_shut_down = False

def shutdown(timeout: float = FLUSH_TIMEOUT_SECONDS):
    """
    Sends what's queued, once, when the process ends: from the app's shutdown hook (Render's
    free plan stops an idle app, and traces queued then would be lost) and at exit for scripts.
    A second call does nothing, so the app's shutdown and exit don't wait twice.
    """
    global _shut_down
    if _shut_down:
        return
    _shut_down = True
    flush(timeout)


# --- Spans (Section C) ---
# The app's own steps, around what LangChain traces by itself. All of these do nothing that
# matters when tracing is off: LangSmith still builds the run objects but never sends them.

def span(name: str, run_type: str = "chain", inputs: dict | None = None, metadata: dict | None = None,
         tags: list[str] | None = None):
    """A traced block: `with span(...) as run:` or `async with`. An exception inside is recorded on it."""
    return langsmith.trace(name, run_type=run_type, inputs=inputs or {}, metadata=metadata, tags=tags)


def traced(name: str, run_type: str = "chain", inputs=None, outputs=None):
    """
    A traced function. `inputs(args: dict) -> dict` and `outputs(result) -> dict` choose what
    is recorded: counts and ids rather than whole articles or snapshots.
    """
    return langsmith.traceable(name=name, run_type=run_type, process_inputs=inputs, process_outputs=outputs)


def annotate(**metadata):
    """Adds metadata to the current span (no-op outside one)."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(metadata)


def add_tags(*tags: str):
    run = get_current_run_tree()
    if run is not None:
        run.add_tags(list(tags))


def ref_of(run) -> str | None:
    """
    The reference shown to the user with an error: the last 8 characters of the trace id
    (the first ones are a timestamp, shared by traces started close together). Every root
    run carries it as metadata `ref`, so a reported error is found by filtering on it.
    """
    if run is None or not status["enabled"]:
        return None
    return str(run.trace_id).replace("-", "")[-8:]


# --- Feedback scores (Section I) ---
# extend_trace_retention=False, because LangSmith otherwise upgrades every trace that gets feedback
# to its longer, paid retention, and these come on every answer. But the SDK batches feedback
# with the traces only when that flag is True: with False, create_feedback is a blocking POST
# /feedback with retries (found in Section I). So it's sent from one daemon thread of our own,
# never from the request, with a bounded queue (LangSmith down must not grow memory on 512 MB).
FEEDBACK_QUEUE_SIZE = 500
FEEDBACK_ATTEMPTS = 3
_feedback_queue: queue.Queue = queue.Queue(maxsize=FEEDBACK_QUEUE_SIZE)
_feedback_thread: threading.Thread | None = None
_feedback_lock = threading.Lock()
_project_ids: dict[str, str] = {}


def project_id(name: str | None) -> str | None:
    """The project's id, for session_id (feedback without it is deprecated by the SDK). Cached."""
    if not name:
        return None
    if name not in _project_ids:
        try:
            _project_ids[name] = str(_client.read_project(project_name=name).id)
        except Exception:
            return None            # not created yet (the first trace creates it): try again next time
    return _project_ids[name]


def _send_feedback():
    while True:
        item = _feedback_queue.get()
        try:
            if callable(item):            # other background work for LangSmith (close_open_llm_runs)
                item()
            else:
                _client.create_feedback(**item, session_id=project_id(item.pop("project", None)),
                                        extend_trace_retention=False, stop_after_attempt=FEEDBACK_ATTEMPTS)
        except Exception as e:
            what = f"feedback {item.get('key')}" if isinstance(item, dict) else "an update to LangSmith"
            print(f"[WARN] Couldn't send {what}: {type(e).__name__}: {e}")
        finally:
            _feedback_queue.task_done()


def _wait_for_feedback(timeout: float):
    """Waits until the queued feedback is sent, at most `timeout` seconds."""
    with _feedback_queue.all_tasks_done:
        _feedback_queue.all_tasks_done.wait_for(lambda: _feedback_queue.unfinished_tasks == 0, timeout)


# --- Chat runs cut off by a deadline (Section L) ---
# asyncio.wait_for cancels the answer step at its deadline while a chat attempt is still running.
# LangChain is then interrupted inside asyncio.gather and never calls on_llm_error, so the attempt's
# run is never closed: LangSmith showed it "pending" for good (live run, 2026-10-21). The usage
# callback registers each chat run here while it runs, and the deadline handlers close what's left.
_open_llm_runs: ContextVar[dict | None] = ContextVar("open_llm_runs", default=None)


def llm_run_started(run_id):
    runs = _open_llm_runs.get()
    if runs is not None:
        runs[run_id] = time.time()


def llm_run_ended(run_id):
    runs = _open_llm_runs.get()
    if runs is not None:
        runs.pop(run_id, None)


def close_open_llm_runs(reason: str) -> list:
    """Ends the request's chat runs that are still open, with `reason` as their error. Returns their ids."""
    runs = _open_llm_runs.get()
    if not runs:
        return []
    ids = list(runs)
    runs.clear()
    if _client is None or not status["enabled"]:
        return ids
    ended = datetime.now(timezone.utc)
    _start_feedback_thread()
    for run_id in ids:
        try:
            _feedback_queue.put_nowait(lambda run_id=run_id: _client.update_run(run_id, end_time=ended, error=reason))
        except queue.Full:
            print(f"[WARN] Couldn't close chat run {run_id}: {FEEDBACK_QUEUE_SIZE} updates waiting (is LangSmith down?)")
    return ids


def _start_feedback_thread():
    global _feedback_thread
    with _feedback_lock:
        if _feedback_thread is None:
            _feedback_thread = threading.Thread(target=_send_feedback, name="langsmith-feedback", daemon=True)
            _feedback_thread.start()


def feedback(key: str, score, comment: str | None = None):
    """
    A feedback score on the current request's trace. Queued and sent in the background, so the
    request never waits for it. No-op with tracing off or outside a traced request; never raises.
    """
    run = get_current_run_tree()
    if _client is None or run is None or not status["enabled"]:
        return
    _start_feedback_thread()
    try:
        _feedback_queue.put_nowait({"trace_id": run.trace_id, "key": key, "score": score, "comment": comment,
                                    "project": run.session_name})
    except queue.Full:
        print(f"[WARN] Couldn't record feedback {key}: {FEEDBACK_QUEUE_SIZE} scores waiting (is LangSmith down?)")


def client():
    """The LangSmith client while tracing is on, else None (Section J: saving a reader's rating)."""
    return _client if status["enabled"] else None


def sent_trace() -> dict | None:
    """
    {"trace_id", "project"} of the current request, if its trace is sent to LangSmith; else None
    (tracing off, outside a request, or sampled out). Kept with an answer so a reader's rating
    can go on its trace later (Section J). A sampled-out trace never reaches LangSmith, and the
    SDK silently drops feedback for it, so a rating there would look saved and be lost.
    """
    run = get_current_run_tree()
    if _client is None or run is None or not status["enabled"]:
        return None
    try:
        if not _client._should_sample(run.trace_id):
            return None
    except Exception:
        return None
    return {"trace_id": str(run.trace_id), "project": run.session_name, "trace_start": _root_start(run)}


def _root_start(run) -> str | None:
    """
    The root run's start time (ISO), which LangSmith needs with the id to find a run directly
    (annotation queues, Section J). The current run is usually a child, but its dotted_order
    starts with the root's: "20261020T125901346888Z<root id>.<...>".
    """
    try:
        stamp = run.dotted_order.split(".", 1)[0][:22]
        return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return None


# Where answers to look at go: every 👎 (added by the rating endpoint) and every answer that
# failed a code check (added by a LangSmith rule, scripts/setup_review_queue.py). Section J #71.
REVIEW_QUEUE = "ai-news-review"
_review_queue_id: str | None = None


def review_queue_id() -> str:
    """The review queue's id, created if it doesn't exist yet. Cached. Blocking; raises on failure."""
    global _review_queue_id
    if _review_queue_id is None:
        found = list(_client.list_annotation_queues(name=REVIEW_QUEUE))
        queue = found[0] if found else _client.create_annotation_queue(
            name=REVIEW_QUEUE, description="Answers to review: every 👎 from a reader and every answer that failed a code check.")
        _review_queue_id = str(queue.id)
    return _review_queue_id


def current_trace() -> dict | None:
    """The current trace's id and ref, to link a request that joins work another one started."""
    run = get_current_run_tree()
    ref = ref_of(run)
    return {"trace_id": str(run.trace_id), "ref": ref} if ref else None


def summarize(value, depth=0):
    """
    A small, safe picture of a response or result for a span's outputs: short strings as they
    are, long ones as their length, lists as their length, ids as hashes (an answer or watch
    id lets its holder follow or read a topic, so it never leaves the server).
    """
    if isinstance(value, dict):
        if depth >= 2:
            return f"{len(value)} keys"
        out = {}
        for key, item in value.items():
            if key.endswith("_id") and isinstance(item, str):
                out[key] = short_hash(item)
            elif isinstance(item, str) and len(item) > 200:
                out[f"{key}_chars"] = len(item)
            else:
                out[key] = summarize(item, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return len(value)
    return value


# --- Request totals (Section D, #36) ---
# One dict per request: chat calls, tokens, cost, embedding estimates, cache hits, the model that
# answered. Set by request_span; work started inside the request (the build task, the model
# callbacks) sees the same dict, because a task copies the context and the dict is shared. A
# request that joins someone else's build adds nothing for it: that work is in the other trace.
_totals: ContextVar[dict | None] = ContextVar("request_totals", default=None)

# Totals worth filtering or grouping by in LangSmith: copied onto the root as metadata too
_TOTALS_AS_METADATA = ("answered_by", "cost_usd", "chat_calls", "chat_errors", "skipped_model",
                       "prompt_version", "config_version", "check_score", "checks_failed")


def add_totals(**amounts):
    """
    Adds to the current request's totals: numbers are summed, anything else is set (e.g.
    answered_by). Does nothing outside a request (scripts, tests calling services directly).
    """
    totals = _totals.get()
    if totals is None:
        return
    for key, value in amounts.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[key] = round(totals.get(key, 0) + value, 8)
        else:
            totals[key] = value


def current_totals() -> dict | None:
    totals = _totals.get()
    return dict(totals) if totals is not None else None


def _record_totals(run, totals: dict):
    if not totals:
        return
    run.add_outputs({"usage": dict(totals)})
    run.add_metadata({key: totals[key] for key in _TOTALS_AS_METADATA if key in totals})


def _request_inputs(kwargs: dict, view: str | None) -> tuple[dict, dict]:
    inputs, metadata = {}, {}
    req = kwargs.get("req")
    if req is not None and getattr(req, "query", None) is not None:
        inputs["query"] = req.query
    if view:
        inputs["view"] = view
    answer_ids = getattr(req, "answer_ids", None)
    if answer_ids:
        inputs["answers"] = sorted(view for view, answer_id in answer_ids.items() if answer_id)
    if req is not None and getattr(req, "score", None) is not None:      # POST /api/feedback (Section J)
        inputs.update(score=req.score, answer=short_hash(str(req.answer_id)), comment_chars=len(req.comment or ""))
    watch_id = kwargs.get("watch_id")
    if watch_id:
        inputs["watch"] = short_hash(watch_id)
        # LangSmith groups root runs with the same thread_id into one thread
        metadata["thread_id"] = short_hash(watch_id)
    return inputs, metadata


def _record_failure(run, e: BaseException):
    """Status, the user's message and the log reason on the request's span (#27)."""
    if isinstance(e, asyncio.CancelledError):
        run.add_tags(["cancelled"])
        run.add_metadata({"status": "cancelled", "note": "the client left; shared work keeps running for others"})
        return
    status_code = getattr(e, "status", None)
    if status_code is None:
        run.add_tags(["error:500", "unexpected"])
        run.add_metadata({"status": 500, "error_type": type(e).__name__})
    else:
        run.add_tags([f"error:{status_code}"] + (["timeout"] if status_code == 504 else []))
        run.add_metadata({"status": status_code, "user_message": getattr(e, "message", ""), "log": getattr(e, "log", "")})
    ref = ref_of(run)
    if ref:
        try:
            e.trace_ref = ref        # server.py puts it in the error response (#28)
        except Exception:
            pass


def request_span(name: str, view: str | None = None):
    """
    The root span of one API request (#19), named after its route, e.g. "POST /query".
    Inputs: the query, the view, a followed topic's hash; outputs: a summary of the response
    and its status. A failure records its status and messages and gives the error a ref.
    """
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            inputs, metadata = _request_inputs(kwargs, view)
            totals = {}
            token = _totals.set(totals)
            open_runs_token = _open_llm_runs.set({})
            try:
                # LangSmith passes a span's tags and metadata down to the runs started inside it, so
                # the view tag marks every run of that answer ("all report runs"); requests themselves
                # are the root runs (no "request" tag: it would be on every run too)
                async with span(name, inputs=inputs, metadata=metadata, tags=[view] if view else None) as run:
                    ref = ref_of(run)
                    if ref:
                        run.add_metadata({"ref": ref})
                    started = time.perf_counter()
                    try:
                        result = await fn(*args, **kwargs)
                    except BaseException as e:
                        _record_failure(run, e)
                        _record_totals(run, totals)
                        raise
                    if isinstance(result, dict) and result.get("watch_id") and "thread_id" not in metadata:
                        run.add_metadata({"thread_id": short_hash(result["watch_id"])})
                    run.add_outputs({"status": 200, "seconds": round(time.perf_counter() - started, 2),
                                     "response": summarize(result)})
                    _record_totals(run, totals)
                    return result
            finally:
                _totals.reset(token)
                _open_llm_runs.reset(open_runs_token)
        return wrapper
    return decorate


configure_tracing()
