import atexit
import hashlib
import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
import langsmith
from langsmith import utils as ls_utils
from langsmith.anonymizer import create_anonymizer

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

    redact = make_redactor(environ)
    # max_depth: an LLM run nests its text ~6 levels deep (outputs.generations[0][0].message
    # .kwargs.content); the default of 10 is close enough to miss a deeper copy.
    _client = langsmith.Client(anonymizer=create_anonymizer(redact, max_depth=30))
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
    """Sends the traces still queued, waiting at most `timeout` seconds. Never raises."""
    if _client is None:
        return
    try:
        _client.flush(timeout=timeout)
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


configure_tracing()
