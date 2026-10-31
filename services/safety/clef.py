import asyncio
import os
import re
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from services import cache_service, observability

# --- Clef client (feature 3, Section B #15a) ---
# Clef (@cf/cloudflare/clef) is Cloudflare's decision model on Workers AI: it takes text in `state`
# and typed questions about it, and returns probabilities, never text. It's called through the
# AI Gateway, so every call is in the gateway's logs too (cf-aig-log-id on the response):
#   POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run
#   header cf-aig-gateway-id: {gateway}
# (The gateway.ai.cloudflare.com URL needs a separate gateway token: 401 in Section A.)
# It's billed in Workers AI neurons (header cf-ai-neurons): 10,000 a day free, reset at 00:00 UTC.
# Measured: neurons = 0.0218 x input tokens, no fixed cost per call (A2 §7.1).

load_dotenv()
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
GATEWAY_ID = os.getenv("CLOUDFLARE_AI_GATEWAY_ID", "").strip()

WORKERS_MODEL = "@cf/cloudflare/clef"
# Pinned: clef-omni is ~40% cheaper but missed instructions buried in a news paragraph (A, #7c);
# clef-flash doesn't exist ("Unsupported model 'clef-flash'. Use 'clef'."). The model each answer
# reports is recorded, because Cloudflare can update `clef` under the same name (N #106a).
MODEL = "clef"
MAX_QUESTIONS = 64             # per call (Clef's limit)
CONTEXT_TOKENS = 64_000        # Clef's window. A state that doesn't fit is cut SILENTLY
# State tokens per call: room is left for the questions and Clef's own template. Tokens are
# estimated high (3 characters a token; English is ~4), so a call is split before it could be cut.
STATE_TOKEN_BUDGET = 56_000
CHARS_PER_TOKEN = 3.0
# A reply this close to the window means the state may have been cut: treated as a failure
CUT_MARGIN_TOKENS = 512
RETRY_WAIT_SECONDS = 0.25      # before the one retry of a 429 / 5xx
MIN_RETRY_SECONDS = 0.5        # time that must be left before the deadline to retry at all
PARALLEL_CALLS = 6             # per process, as measured in Section A
# The first call in a process took 1.5-3.5 s, later ones 0.5-1.1 s (B1 §6). httpx closes an idle
# connection after 5 s by default, so with a few requests a minute nearly every check would start
# cold: connections are kept 5 minutes, and startup opens one (warm_up, free: no Clef call).
KEEPALIVE_SECONDS = 300

# Daily neurons (B #19; L #93a adds the warning alert and the gateway's spending limit). Counted
# per UTC day from every response's cf-ai-neurons, in Redis (v1:usage:clef:<day>) and in process.
DAILY_NEURONS = 10_000
FALLBACK_AT_NEURONS = 9_500    # from here on, checks use the embedding classifier (A2 §7.2)


class ClefError(Exception):
    """
    A call that gave no usable answer. `kind` says why: not_configured, timeout, busy (429 or 5xx
    after the retry), auth (401/403), billing (402), bad_request (400/413/422), too_large (one
    question's state can't fit the window), cut (the reply says the window was full),
    over_budget, bad_response. Every kind makes a check fall back to the embedding classifier.
    """
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


def available() -> bool:
    return bool(ACCOUNT_ID and API_TOKEN and GATEWAY_ID)


def status() -> str:
    """For the startup line."""
    if not available():
        missing = [name for name, value in (("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID), ("CLOUDFLARE_API_TOKEN", API_TOKEN),
                                            ("CLOUDFLARE_AI_GATEWAY_ID", GATEWAY_ID)) if not value]
        return f"Clef off ({', '.join(missing)} not set)"
    return f"Clef ready (model {MODEL}, gateway {GATEWAY_ID})"


# --- The HTTP client: one per process, so connections are reused ---
_client: httpx.AsyncClient | None = None
_transport = None             # tests put a fake server here (httpx.MockTransport)
_semaphore: asyncio.Semaphore | None = None


def use_transport(transport):
    """Tests: send calls to `transport` instead of Cloudflare (None goes back to the network)."""
    global _client, _transport, _semaphore
    _client, _transport, _semaphore = None, transport, None


def _http() -> httpx.AsyncClient:
    global _client, _semaphore
    if _client is None:
        _client = httpx.AsyncClient(transport=_transport, limits=httpx.Limits(keepalive_expiry=KEEPALIVE_SECONDS))
        _semaphore = asyncio.Semaphore(PARALLEL_CALLS)
    return _client


async def warm_up():
    """Opens the connection (TLS, the client's certificates) before the first check needs it.
    A HEAD without the token: no Clef call, no neurons; the 4xx it gets is expected. Never raises."""
    if not available():
        return
    try:
        await _http().head(_url(), timeout=5.0)
    except Exception as e:
        print(f"[WARN] Safety: couldn't reach Cloudflare at startup ({type(e).__name__}); the first check may fall back.")


def _url() -> str:
    return f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run"


# --- Neurons used today ---
_spent = {"day": None, "neurons": 0.0}
_pending: set[asyncio.Task] = set()


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def neurons_today() -> float:
    return _spent["neurons"] if _spent["day"] == utc_day() else 0.0


def over_budget() -> bool:
    return neurons_today() >= FALLBACK_AT_NEURONS


async def _count_neurons(day: str, neurons: float, failed: bool):
    """Adds to the day's count in Redis; the total Redis returns is kept if it's higher (another worker, a restart)."""
    counts = {"calls": 1, "neurons_x100": round(neurons * 100)}
    if failed:
        counts["errors"] = 1
    totals = await cache_service.add_usage("clef", day, counts)
    if totals and _spent["day"] == day:
        _spent["neurons"] = max(_spent["neurons"], totals.get("neurons_x100", 0) / 100)


def _add_neurons(neurons: float, failed: bool = False):
    day = utc_day()
    if _spent["day"] != day:
        _spent.update(day=day, neurons=0.0)
    _spent["neurons"] += neurons
    try:
        task = asyncio.get_running_loop().create_task(_count_neurons(day, neurons, failed))
    except RuntimeError:
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def load_today():
    """Reads today's neurons from Redis (startup), so a restart doesn't forget what's been spent."""
    day = utc_day()
    totals = await cache_service.get_usage("clef", day)
    if totals:
        _spent.update(day=day, neurons=totals.get("neurons_x100", 0) / 100)


async def drain():
    """Waits for the pending Redis counter writes (tests)."""
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


# --- Splitting questions over calls ---
_REFERENCED = re.compile(r"`([a-z][a-z0-9_]{0,31})`")


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _reads(question: dict) -> list[str]:
    return list(dict.fromkeys(_REFERENCED.findall(question.get("instructions", ""))))


def plan_calls(state: dict[str, str], questions: dict[str, dict]) -> list[tuple[dict, dict]]:
    """
    Splits the questions into calls of at most MAX_QUESTIONS, each carrying only the state keys
    its questions read, and at most STATE_TOKEN_BUDGET estimated tokens of them. A key shared by
    every question (the articles, the query) goes into every call. Raises ClefError("too_large")
    if one question's own state can't fit, and KeyError if a question reads a key `state` lacks.
    """
    calls, current_q, current_keys, current_tokens = [], {}, [], 0
    for name, question in questions.items():
        keys = _reads(question)
        for key in keys:
            if key not in state:
                raise KeyError(f"question {name!r} reads `{key}`, which isn't in state")
        own = sum(estimate_tokens(state[k]) for k in keys)
        if own > STATE_TOKEN_BUDGET:
            raise ClefError("too_large", f"question {name!r} reads ~{own} tokens; the window is {CONTEXT_TOKENS}")
        new_keys = [k for k in keys if k not in current_keys]
        added = sum(estimate_tokens(state[k]) for k in new_keys)
        if current_q and (len(current_q) >= MAX_QUESTIONS or current_tokens + added > STATE_TOKEN_BUDGET):
            calls.append((current_q, {k: state[k] for k in current_keys}))
            current_q, current_keys, current_tokens = {}, [], 0
            new_keys, added = keys, own
        current_q[name] = question
        current_keys.extend(new_keys)
        current_tokens += added
    if current_q:
        calls.append((current_q, {k: state[k] for k in current_keys}))
    return calls


# --- Calling ---

def _parse(name: str, answer) -> dict:
    """One typed answer: {"type": "noul", "noul": p} / {"type": "choice", "choice", "probabilities"} /
    {"type": "score", "score", "probabilities", "legend"}. Raises ClefError on anything else."""
    if not isinstance(answer, dict):
        raise ClefError("bad_response", f"no answer for {name!r}")
    kind = answer.get("type")
    try:
        if kind == "noul":
            return {"type": "noul", "noul": float(answer["noul"])}
        if kind == "choice":
            return {"type": "choice", "choice": answer.get("choice"),
                    "probabilities": {k: float(v) for k, v in answer["probabilities"].items()}}
        if kind == "score":
            return {"type": "score", "score": float(answer["score"]), "legend": answer.get("legend") or {},
                    "probabilities": {k: float(v) for k, v in answer["probabilities"].items()}}
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ClefError("bad_response", f"answer for {name!r}: {type(e).__name__}") from e
    raise ClefError("bad_response", f"answer for {name!r} has type {kind!r}")


def _error_of(response: httpx.Response) -> ClefError:
    code = response.status_code
    detail = f"HTTP {code}"
    try:
        messages = [e.get("message", "") for e in response.json().get("errors", [])]
        if messages:
            detail += ": " + "; ".join(messages)[:200]
    except ValueError:
        pass
    if code in (401, 403):
        return ClefError("auth", detail)
    if code == 402:
        return ClefError("billing", detail)
    if code == 429 or code >= 500:
        return ClefError("busy", detail)
    return ClefError("bad_request", detail)


async def _call(questions: dict, state: dict, deadline: float, label: str) -> dict:
    """One call, retried once on a 429 / 5xx if time is left. Returns the parsed reply."""
    payload = {"model": WORKERS_MODEL, "input": {"model": MODEL, "state": state, "questions": questions}}
    headers = {"Authorization": f"Bearer {API_TOKEN}", "cf-aig-gateway-id": GATEWAY_ID}
    estimated = sum(estimate_tokens(v) for v in state.values())
    client = _http()
    async with observability.span(
            "clef", run_type="tool",
            inputs={"label": label, "questions": list(questions), "state_chars": {k: len(v) for k, v in state.items()}},
            metadata={"ls_provider": "cloudflare", "ls_model_name": MODEL, "estimated_state_tokens": estimated}) as run:
        for attempt in (1, 2):
            try:
                async with _semaphore:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise ClefError("timeout", "no time left before the check's deadline")
                    # httpx's timeout is per phase (connect, read...); wait_for bounds the whole call
                    response = await asyncio.wait_for(
                        client.post(_url(), json=payload, headers=headers, timeout=left), left)
            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                raise ClefError("timeout", type(e).__name__) from e
            except httpx.TransportError as e:
                raise ClefError("busy", type(e).__name__) from e
            neurons = float(response.headers.get("cf-ai-neurons") or 0)
            if neurons:
                _add_neurons(neurons, failed=response.status_code != 200)
            if response.status_code == 200:
                break
            error = _error_of(response)
            run.add_metadata({f"attempt_{attempt}": str(error)})
            if error.kind != "busy" or attempt == 2 or deadline - time.monotonic() < MIN_RETRY_SECONDS:
                raise error
            await asyncio.sleep(RETRY_WAIT_SECONDS)

        try:
            result = response.json()["result"]
            raw_answers = result["answers"]
        except (ValueError, KeyError, TypeError) as e:
            raise ClefError("bad_response", f"no result in the reply ({type(e).__name__})") from e
        answers = {name: _parse(name, raw_answers.get(name)) for name in questions}
        input_tokens = int((result.get("usage") or {}).get("input_tokens") or 0)
        reply = {"answers": answers, "input_tokens": input_tokens, "neurons": neurons, "model": result.get("model"),
                 "log_id": response.headers.get("cf-aig-log-id")}
        run.add_metadata({"neurons": neurons, "input_tokens": input_tokens, "model_reported": result.get("model"),
                          "cf_aig_log_id": reply["log_id"], "attempts": attempt})
        run.add_outputs({"answers": {n: a.get("noul", a.get("probabilities")) for n, a in answers.items()}})
        # Clef doesn't say when it cut the state; a reply at the window's edge is the only sign
        if input_tokens >= CONTEXT_TOKENS - CUT_MARGIN_TOKENS:
            raise ClefError("cut", f"{input_tokens} input tokens: the state may have been cut")
        return reply


async def ask(state: dict[str, str], questions: dict[str, dict], timeout: float, label: str = "") -> dict:
    """
    Asks Clef `questions` (name -> Clef question) about `state` (key -> text), split over as many
    calls as needed (plan_calls), run in parallel, all within `timeout` seconds. Returns
    {"answers": {name: typed answer}, "input_tokens", "neurons", "calls", "models", "log_ids"}.
    Raises ClefError when any call gives no usable answer: a check is decided by one engine.
    """
    if not available():
        raise ClefError("not_configured", status())
    if over_budget():
        raise ClefError("over_budget", f"{neurons_today():.0f} neurons used today (fallback from {FALLBACK_AT_NEURONS})")
    if not questions:
        return {"answers": {}, "input_tokens": 0, "neurons": 0.0, "calls": 0, "models": [], "log_ids": []}
    deadline = time.monotonic() + timeout
    calls = plan_calls(state, questions)
    replies = await asyncio.gather(*(_call(q, s, deadline, label) for q, s in calls), return_exceptions=True)
    for reply in replies:
        if isinstance(reply, BaseException):
            raise reply
    answers = {}
    for reply in replies:
        answers.update(reply["answers"])
    return {"answers": answers, "input_tokens": sum(r["input_tokens"] for r in replies),
            "neurons": round(sum(r["neurons"] for r in replies), 2), "calls": len(replies),
            "models": sorted({r["model"] for r in replies if r["model"]}),
            "log_ids": [r["log_id"] for r in replies if r["log_id"]]}
