import asyncio
import os

import httpx

# --- Model versions (feature 3, Section N #106) ---
# Gemini's reply names only the model id it was asked for: response.model_version is "gemini-3.8-flash"
# (feature 2, A2 §4), which LangChain hands on as model_name and usage.py records as `answered_by`. So
# if Google moved the alias to a new build, nothing in a trace or a saved answer would show it.
#
# Google's model list does keep a version per model (GET /v1beta/models/{id}). Checked on 2026-11-26
# (N §5): gemini-3.8-flash reports version "3.0", gemini-3.6-flash "3.6-flash-07-2026", and
# gemini-embedding-001 "001". A versioned id can't be called for the two chat models
# (gemini-3.8-flash-001 and gemini-3.6-flash-07-2026 both answer 404 NOT_FOUND), so they stay on their
# ids; the embedding model's id is already its versioned one (001).
#
# So the app asks once at startup: one models.get per model, metadata only (not a generateContent
# call: it uses none of the 20 daily answers), at most TIMEOUT_SECONDS, in parallel with the reranker's
# warm-up. The versions go on the startup line, on every answer's trace (chat_model_version,
# fallback_model_version) and on every saved answer (model_version, for the model that answered).
# A version other than the one recorded below gets a [WARN] at startup: Google changed the model
# behind the name, so the answers may change, and the answer evals are worth running again.
# The check never stops the app: no key, a timeout or an error leaves the version unknown.

# What Google reported when it was last looked at. Update it (with the date) after checking a new
# version's answers.
RECORDED = {
    "gemini-3.8-flash": "3.0",
    "gemini-3.6-flash": "3.6-flash-07-2026",
    "gemini-embedding-001": "001",
}
RECORDED_ON = "2026-11-26"
API = "https://generativelanguage.googleapis.com/v1beta/models/"
TIMEOUT_SECONDS = 5.0

# model id -> the version Google reported at startup, or None when it couldn't be read
SEEN: dict[str, str | None] = {}
# model id -> why it couldn't be read
PROBLEMS: dict[str, str] = {}

_transport = None      # tests set an httpx.MockTransport


def use_transport(transport):
    global _transport
    _transport = transport


def _api_key() -> str:
    return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


async def _get(client: httpx.AsyncClient, model: str) -> str | None:
    try:
        response = await client.get(API + model)
    except httpx.HTTPError as e:
        PROBLEMS[model] = f"{type(e).__name__}"
        return None
    if response.status_code != 200:
        PROBLEMS[model] = f"HTTP {response.status_code}"
        return None
    try:
        version = response.json().get("version")
    except ValueError:
        version = None
    if not isinstance(version, str) or not version.strip():
        PROBLEMS[model] = "no version in the reply"
        return None
    return version.strip()


async def check(models) -> dict[str, str | None]:
    """Asks Google for each model's version (in parallel, TIMEOUT_SECONDS in all). Never raises."""
    SEEN.clear()
    PROBLEMS.clear()
    models = list(dict.fromkeys(models))
    if (os.getenv("MODEL_VERSION_CHECK") or "on").strip().lower() == "off":
        PROBLEMS.update({m: "check off (MODEL_VERSION_CHECK=off)" for m in models})
        SEEN.update({m: None for m in models})
        return dict(SEEN)
    key = _api_key()
    if not key:
        PROBLEMS.update({m: "no API key" for m in models})
        SEEN.update({m: None for m in models})
        return dict(SEEN)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers={"x-goog-api-key": key},
                                     transport=_transport) as client:
            versions = await asyncio.wait_for(asyncio.gather(*(_get(client, m) for m in models)), TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, httpx.HTTPError, OSError) as e:
        PROBLEMS.update({m: PROBLEMS.get(m) or type(e).__name__ for m in models})
        versions = [None] * len(models)
    SEEN.update(dict(zip(models, versions)))
    return dict(SEEN)


def version_of(model: str | None) -> str | None:
    """The version Google reported for `model` at startup (None if unknown or not checked)."""
    if not model:
        return None
    return SEEN.get(str(model).removeprefix("models/"))


def changed() -> dict[str, tuple[str, str]]:
    """{model: (recorded, seen)} for every model whose version isn't the recorded one."""
    return {m: (RECORDED[m], v) for m, v in SEEN.items() if v and m in RECORDED and RECORDED[m] != v}


def trace_metadata(chat_model: str, fallback_model: str) -> dict:
    """The versions on an answer's span (only those known)."""
    out = {}
    for key, model in (("chat_model_version", chat_model), ("fallback_model_version", fallback_model)):
        version = version_of(model)
        if version:
            out[key] = version
    return out


def status() -> str:
    """The startup line's text, and one [WARN] line per changed or unrecorded model."""
    parts = []
    for model, version in SEEN.items():
        if version is None:
            parts.append(f"{model} version unknown ({PROBLEMS.get(model, '?')})")
        elif model not in RECORDED:
            parts.append(f"{model} version {version} (not recorded)")
        elif RECORDED[model] != version:
            parts.append(f"{model} version {version} (CHANGED: {RECORDED[model]} on {RECORDED_ON})")
        else:
            parts.append(f"{model} version {version}")
    return "Models: " + ", ".join(parts) if parts else "Models: not checked"


def warnings() -> list[str]:
    return [f"[WARN] Google now reports version {seen!r} for {model} (recorded {recorded!r} on {RECORDED_ON}): the "
            f"model behind the name changed, so answers may change. Run the answer evals, then update "
            f"services/model_versions.RECORDED." for model, (recorded, seen) in changed().items()]


async def startup(models) -> None:
    """At app start: check the versions and print the line (and any warnings)."""
    await check(models)
    print(f"[INFO] {status()}.")
    for line in warnings():
        print(line)
