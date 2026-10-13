import asyncio
import httpx
from google.genai import errors as genai_errors

# --- User-facing errors ---
# Every place a request can fail raises AppError with its own HTTP status and a message
# written for the user. server.py turns it into {"detail": message}, which the page shows
# in its error box. All messages live here, so each failure point has exactly one.


class AppError(Exception):
    """
    A failure the user should see. `message` is shown in the UI; `log` is the
    technical reason for the server log (never sent to the user).
    """
    def __init__(self, status: int, message: str, log: str = ""):
        super().__init__(message)
        self.status = status
        self.message = message
        self.log = log or message


def _topic(topic: str) -> str:
    """The topic as the user typed it, shortened so a very long one can't flood the message."""
    topic = " ".join((topic or "").split())
    return topic if len(topic) <= 80 else topic[:77] + "..."


# --- Request ---

def empty_query():
    return AppError(400, "Please enter a topic first.")

def invalid_request(log=""):
    return AppError(422, "The request was invalid. Please enter a topic and try again.", log)

UNEXPECTED_MESSAGE = "Something went wrong on our side. Please try again."


# --- GNews ---
# Status codes from https://docs.gnews.io/error-handling

NEWS_NOT_CONFIGURED = "The news service isn't set up correctly on the server. Please contact the site owner."

def news_not_configured(log=""):
    return AppError(500, NEWS_NOT_CONFIGURED, log)

def news_unreachable(log=""):
    return AppError(503, "The news service didn't respond. Please try again in a minute.", log)

def news_failed(log=""):
    return AppError(502, "The news service is having problems right now. Please try again later.", log)

def news_http_error(status: int, body: str):
    """Maps a GNews error response to what the user can do about it."""
    log = f"GNews HTTP {status}: {body[:300]}"
    if status == 400:
        # 400 covers both a missing/malformed API key and an invalid search
        lowered = body.lower()
        if "api key" in lowered or "apikey" in lowered or "token" in lowered:
            return news_not_configured(log)
        return AppError(400, "The news service couldn't search for this topic. "
                             "Try simpler words, without quotes or special characters.", log)
    if status == 401:   # key is well-formed but matches no account
        return news_not_configured(log)
    if status == 403:   # daily quota used up, resets at 00:00 UTC
        return AppError(503, "Today's news search limit has been reached. "
                             "It resets at midnight UTC, so please try again later.", log)
    if status == 429:   # free plan: 1 request per second
        return AppError(503, "The news service is getting too many requests right now. "
                             "Please try again in a minute.", log)
    if status == 503:   # GNews maintenance
        return AppError(503, "The news service is down for maintenance. Please try again later.", log)
    return news_failed(log)

def no_recent_news(topic: str):
    return AppError(404, f'No news from the last 2 days was found for "{_topic(topic)}". '
                         "Try a broader or different topic.")

def no_readable_pages(topic: str, found: int):
    articles = "1 article" if found == 1 else f"{found} articles"
    return AppError(404, f'Found {articles} about "{_topic(topic)}", but none of their pages could be read '
                         "(blocked or paywalled). Try another topic.")


# --- Gemini (embeddings and chat) ---

AI_NOT_CONFIGURED = "The AI service isn't set up correctly on the server. Please contact the site owner."
BUSY_CODES = {429, 500, 502, 503}
CONFIG_CODES = {401, 403, 404}   # bad key, no permission, unknown model name

def gemini_failure_kind(e: Exception) -> str:
    """
    'config', 'busy', 'timeout' or 'other' for a failed Gemini call. LangChain wraps the
    SDK's errors, so the cause chain is searched (as rag_service._is_retryable does).
    """
    for _ in range(6):
        if e is None:
            break
        if isinstance(e, genai_errors.APIError):
            if e.code == 504:
                return "timeout"
            if e.code in BUSY_CODES:
                return "busy"
            if e.code in CONFIG_CODES or (e.code == 400 and "api key" in str(e).lower()):
                return "config"
            return "other"
        if isinstance(e, (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)):
            return "timeout"
        if isinstance(e, httpx.TransportError):
            return "busy"
        e = e.__cause__ or e.__context__
    return "other"

def ai_not_configured(log=""):
    return AppError(500, AI_NOT_CONFIGURED, log)

def analysis_failed(e: Exception):
    """Embedding the articles (building the index) failed."""
    kind, log = gemini_failure_kind(e), f"embedding failed: {type(e).__name__}: {e}"
    if kind == "config":
        return ai_not_configured(log)
    if kind == "busy":
        return AppError(503, "Couldn't analyze the articles right now because the AI service is busy. "
                             "Please try again in a minute.", log)
    if kind == "timeout":
        return AppError(504, "The AI service took too long to analyze the articles. Please try again.", log)
    return AppError(502, "The AI service couldn't analyze the articles. Please try again.", log)

# What the user asked for, by prompt type
_OUTPUT_NAMES = {"report": "report", "timeline": "timeline", "contradictions": "analysis of conflicting reports",
                 "update": "update"}

def report_failed(e: Exception, prompt_type: str):
    """The chat model (or the retrieval step before it) failed."""
    name = _OUTPUT_NAMES.get(prompt_type, "answer")
    kind, log = gemini_failure_kind(e), f"{prompt_type} generation failed: {type(e).__name__}: {e}"
    if kind == "config":
        return ai_not_configured(log)
    if kind == "busy":
        return AppError(503, f"The AI service is busy right now, so the {name} couldn't be written. "
                             "Please try again in a minute.", log)
    if kind == "timeout":
        return AppError(504, f"The AI took too long to write the {name}. Please try again.", log)
    return AppError(502, f"The AI couldn't write the {name}. Please try again.", log)

def empty_report(prompt_type: str):
    name = _OUTPUT_NAMES.get(prompt_type, "answer")
    return AppError(502, f"The AI returned an empty {name}. Please try again.", f"{prompt_type}: empty answer")


# --- Following a topic (feature 1) ---

def follow_unavailable(log=""):
    """Redis isn't configured, is unreachable, or is in its back-off: followed topics can't be read or saved."""
    return AppError(503, "Following topics is unavailable right now. You can still analyze topics as usual.", log)

def report_too_old(log=""):
    """V1's answer or article list expired, or the article list was rebuilt since the report."""
    return AppError(409, "This report is more than 2 hours old. Analyze the topic again, then follow it.", log)

def report_not_followable(log=""):
    """No report id, or an answer id that belongs to another topic or view."""
    return AppError(409, "This report can't be followed. Analyze the topic again, then follow it.", log)

def watch_not_found(log=""):
    return AppError(404, "This followed topic has expired after 30 days without a check, or the link is incomplete. "
                         "Follow it again to start a new history.", log)

def check_not_saved(log=""):
    """The check worked but its result couldn't be saved: showing it would show the same news again next time."""
    return AppError(503, "Couldn't save this check, so its results aren't shown. Please check again in a minute.", log)
