import os
from dataclasses import dataclass

from google.genai import types as genai_types
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from services import errors
from services.safety.events import SafetyEvent, record

# --- Gemini's own safety filter (feature 3, Section F) ---
#
# Thresholds (#45, decided 2026-10-31). Gemini 2.5 and later default to OFF for every category
# (Section A1 §3.4), and nothing in today's traffic was ever blocked by it: refusals come from the
# model itself. BLOCK_ONLY_HIGH (the first plan) would make Gemini stricter than today and could
# block war, crime or terror reporting that passes now. So the four categories are set to OFF
# explicitly: a change of Google's default can't change the app, and judging content is left to
# our own checks (C on the query, J and K on the answer). Civic integrity is deprecated (A1 §3.4)
# and isn't sent. The newer jailbreak category is only sent when GEMINI_JAILBREAK_THRESHOLD names
# a threshold, after it has been measured on the dark-news set (A #1): unset, it isn't sent at all.
CATEGORIES_OFF = (
    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
)
JAILBREAK_ENV = "GEMINI_JAILBREAK_THRESHOLD"
JAILBREAK_THRESHOLDS = ("OFF", "BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE")


def safety_settings(environ=None) -> dict:
    """{HarmCategory: HarmBlockThreshold} for every chat model. An unknown jailbreak value is
    ignored with a warning (the category isn't sent), so a typo can't stop the app from starting."""
    environ = os.environ if environ is None else environ
    settings = {category: genai_types.HarmBlockThreshold.OFF for category in CATEGORIES_OFF}
    raw = (environ.get(JAILBREAK_ENV) or "").strip().upper()
    if raw in JAILBREAK_THRESHOLDS:
        settings[genai_types.HarmCategory.HARM_CATEGORY_JAILBREAK] = genai_types.HarmBlockThreshold(raw)
    elif raw:
        print(f"[WARN] {JAILBREAK_ENV}={raw!r} isn't one of {', '.join(JAILBREAK_THRESHOLDS)}; "
              "the jailbreak category isn't sent.")
    return settings


def settings_summary(settings: dict) -> dict:
    """The settings as plain text, for the config version and the startup line:
    {"HARASSMENT": "OFF", ...}."""
    return {_name(category).removeprefix("HARM_CATEGORY_"): _name(threshold) for category, threshold in settings.items()}


def status(settings: dict) -> str:
    """The Gemini part of the startup line: 'Gemini filter OFF (4 categories), jailbreak not sent'."""
    summary = settings_summary(settings)
    jailbreak = summary.pop("JAILBREAK", None)
    thresholds = sorted(set(summary.values()))
    text = f"Gemini filter {'/'.join(thresholds)} ({len(summary)} categories)"
    return text + (f", jailbreak {jailbreak}" if jailbreak else ", jailbreak not sent")


# Read once at import: every chat model (rag_service._chat_model) gets the same settings
SETTINGS = safety_settings()


# --- What a block looks like (A1 §3.2-§3.4) ---
# A block raises nothing in langchain-google-genai: it logs a warning and returns an empty message.
#   prompt blocked  no candidates; promptFeedback.blockReason is set. LangChain puts prompt_feedback in
#                   llm_output only (the empty message carries none), so GeminiChat copies it onto the
#                   message, where the code after the call can see it.
#   answer blocked  a candidate with one of ANSWER_BLOCK_REASONS as finish_reason, with no text or with
#                   the text written so far. A cut answer is a blocked answer, never a full one.
#   HTTP 400        Google's newer shape: an error whose body code is safety / jailbreak /
#                   prohibited_content / blocklist, message "Input blocked: ...". GeminiChat turns it
#                   into the same empty message with prompt_feedback, so it's handled like a blocked prompt.
# MAX_TOKENS, LANGUAGE and OTHER aren't safety blocks (MAX_TOKENS is Section I #66).
ANSWER_BLOCK_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
                                  "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"})
HTTP_BLOCK_CODES = frozenset({"safety", "jailbreak", "prohibited_content", "blocklist"})
HTTP_BLOCK_PREFIX = "input blocked:"
NO_REASON = ("", "BLOCKED_REASON_UNSPECIFIED")


def _name(value) -> str:
    """An SDK enum (or plain text) as its name: BlockedReason.SAFETY -> 'SAFETY'."""
    return str(getattr(value, "value", value) or "")


@dataclass
class Block:
    stage: str              # "prompt" or "answer"
    reason: str             # SAFETY, PROHIBITED_CONTENT, JAILBREAK, ...
    ratings: str = ""       # the ratings that blocked it, e.g. "DANGEROUS_CONTENT: HIGH"
    partial_chars: int = 0  # text written before an answer was cut (shown to nobody)
    source: str = "response"  # "response" or "http_400"

    def label(self) -> str:
        label = f"{self.stage} {self.reason}"
        if self.ratings:
            label += f" ({self.ratings})"
        if self.partial_chars:
            label += f", cut after {self.partial_chars} chars"
        if self.source != "response":
            label += f", {self.source}"
        return label


def ratings_text(ratings) -> str:
    """The ratings Gemini marked as blocked, or else the ones above NEGLIGIBLE; '' when none."""
    out = []
    for rating in ratings or []:
        rating = rating if isinstance(rating, dict) else getattr(rating, "__dict__", {})
        probability = _name(rating.get("probability"))
        if rating.get("blocked") or probability not in ("", "NEGLIGIBLE", "HARM_PROBABILITY_UNSPECIFIED"):
            out.append(f"{_name(rating.get('category')).removeprefix('HARM_CATEGORY_')}: {probability or '?'}")
    return ", ".join(out)


def prompt_feedback_block(feedback) -> Block | None:
    if not feedback:
        return None
    feedback = feedback if isinstance(feedback, dict) else getattr(feedback, "__dict__", {})
    reason = _name(feedback.get("block_reason"))
    if reason in NO_REASON:
        return None
    return Block("prompt", reason, ratings_text(feedback.get("safety_ratings")), source=feedback.get("source") or "response")


def block_of(message, llm_output: dict | None = None) -> Block | None:
    """The block on one model reply, or None. Reads the message's response_metadata (and the
    LLMResult's llm_output, where LangChain keeps a blocked prompt's feedback)."""
    metadata = getattr(message, "response_metadata", None) or {}
    block = prompt_feedback_block(metadata.get("prompt_feedback")) or prompt_feedback_block((llm_output or {}).get("prompt_feedback"))
    if block is not None:
        return block
    finish = _name(metadata.get("finish_reason"))
    if finish in ANSWER_BLOCK_REASONS:
        text = getattr(message, "text", None)       # a property in langchain-core 1.x (a method before)
        if not isinstance(text, str):
            text = text() if callable(text) else str(getattr(message, "content", "") or "")
        return Block("answer", finish, ratings_text(metadata.get("safety_ratings")), partial_chars=len(text or ""))
    return None


def finish_reason(message) -> str | None:
    """Why the model stopped ('STOP', 'MAX_TOKENS', ...), from a reply; None for anything else (a stand-in's text)."""
    metadata = getattr(message, "response_metadata", None)
    reason = _name(metadata.get("finish_reason")) if isinstance(metadata, dict) else ""
    return reason or None


def http_block_reason(e) -> str | None:
    """The block code of Google's HTTP 400 shape ('jailbreak', 'safety', ...), or None for any other error."""
    api = errors.api_error(e)
    if api is None or api.code != 400:
        return None
    details = api.details if isinstance(api.details, dict) else {}
    body = details.get("error") if isinstance(details.get("error"), dict) else details
    code = str(body.get("code") or "").lower()
    message = str(api.message or body.get("message") or "").strip().lower()
    if code in HTTP_BLOCK_CODES:
        return code
    if message.startswith(HTTP_BLOCK_PREFIX):
        return "safety"
    return None


def _with_prompt_feedback(result: ChatResult) -> ChatResult:
    """Copies a blocked prompt's feedback from llm_output onto the (empty) message."""
    feedback = (result.llm_output or {}).get("prompt_feedback")
    if prompt_feedback_block(feedback) is None:
        return result
    for generation in result.generations:
        generation.message.response_metadata = {**(generation.message.response_metadata or {}), "prompt_feedback": feedback}
    return result


def _http_blocked_result(reason: str) -> ChatResult:
    feedback = {"block_reason": reason.upper(), "safety_ratings": [], "source": "http_400"}
    message = AIMessage("", response_metadata={"prompt_feedback": feedback})
    return ChatResult(generations=[ChatGeneration(message=message, generation_info={})], llm_output={"prompt_feedback": feedback})


class GeminiChat(ChatGoogleGenerativeAI):
    """
    ChatGoogleGenerativeAI where a block is a value on the reply, never an exception (#47): a
    blocked prompt's feedback is on the message, and the HTTP 400 "Input blocked" error comes back
    as the same empty, blocked message. So with_retry doesn't retry a block, with_fallbacks doesn't
    send it to the fallback model (it would spend a call and is usually blocked too), and the code
    after the call raises GeminiBlocked once (raise_if_blocked).
    Its runs keep the name ChatGoogleGenerativeAI in LangSmith, as before F (feature 2's traces,
    filters and monitoring descriptions use it).
    """

    name: str | None = "ChatGoogleGenerativeAI"

    def _generate(self, *args, **kwargs) -> ChatResult:
        try:
            return _with_prompt_feedback(super()._generate(*args, **kwargs))
        except Exception as e:
            reason = http_block_reason(e)
            if reason is None:
                raise
            return _http_blocked_result(reason)

    async def _agenerate(self, *args, **kwargs) -> ChatResult:
        try:
            return _with_prompt_feedback(await super()._agenerate(*args, **kwargs))
        except Exception as e:
            reason = http_block_reason(e)
            if reason is None:
                raise
            return _http_blocked_result(reason)


class GeminiBlocked(Exception):
    """Gemini blocked the prompt or the answer. Raised after the model (and its fallback) returned."""

    def __init__(self, block: Block, model: str | None = None):
        super().__init__(f"Gemini blocked the {block.label()}" + (f" on {model}" if model else ""))
        self.block = block
        self.model = model

    def app_error(self, prompt_type: str) -> errors.AppError:
        """The user's message: 422 prompt_blocked or 502 answer_blocked (approved in A3 §4.1)."""
        if self.block.stage == "prompt":
            return errors.prompt_blocked(str(self))
        return errors.answer_blocked(prompt_type, str(self))


def raise_if_blocked(reply):
    """After the chat model and its fallback: the reply, or GeminiBlocked. Anything that isn't a
    message (a test's stand-in returning text) passes through."""
    if isinstance(reply, AIMessage):
        block = block_of(reply)
        if block is not None:
            raise GeminiBlocked(block, (reply.response_metadata or {}).get("model_name"))
    return reply


def structured_raw_or_raise(result):
    """
    Inside each fact-check model (before its fallback). result is with_structured_output's
    include_raw dict {raw, parsed, parsing_error}. A blocked reply is returned as it is, so the
    fallback isn't called for it (#48); an unreadable one raises its parsing error, which sends it
    to the fallback as before.
    """
    if isinstance(result, dict) and "raw" in result:
        if block_of(result["raw"]) is None and result.get("parsing_error") is not None:
            raise result["parsing_error"]
    return result


def structured_answer(result):
    """After the fact check's model and fallback: the parsed answer, or GeminiBlocked. Anything
    else (a test's stand-in returning the answer itself) passes through."""
    if isinstance(result, dict) and "raw" in result:
        raise_if_blocked(result["raw"])
        return result.get("parsed")
    return result


def record_block(blocked: GeminiBlocked, where: str):
    """The block as a safety event on the current span (the view's): check "gemini". Gemini always
    acts, so it's an enforced block whatever the SAFETY_* modes say. Its ratings and reason are
    the label; the blocked text is never recorded."""
    label = blocked.block.label() + (f" on {blocked.model}" if blocked.model else "")
    record(SafetyEvent(check="gemini", verdict="block", mode="enforce", where=where, score=None, threshold=None,
                       engine="gemini", ms=0, label=label, enforced=True))
