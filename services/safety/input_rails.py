import asyncio
from dataclasses import dataclass, field

from services import errors, observability
from services.safety import privacy, questions, text
from services.safety.engine import Decision, judge_one
from services.safety.guard import check

# --- Input rails (feature 3, Section C) ---
# What the user sends is cleaned and checked before any GNews, scraping, embedding or chat call
# is spent. check_query runs for all three views and for Follow (it replaces server.require_query),
# so a refused query costs one Clef call (none when its verdict is cached) and nothing else.
#
#   1. size: a raw query over MAX_RAW_QUERY_CHARS isn't read at all (#21)
#   2. normalize + search form: hidden characters out, "?" "!" removed, "-" a space (#22)
#   3. personal data (H #58): taken out of what the checks read; in enforce mode, out of the search too
#   4. size again on what is searched: GNews's 200 characters (#21)
#   5. intent, by Clef or the fallback classifier: refused in enforce mode from 0.7 (#24, #25)
#   6. text shape: scrambled text is flagged, and blocked when the intent score is also >= 0.2 (#23)
#
# Real news about dark events, and news ABOUT AI attacks, must pass: 0 of 118 news queries are
# blocked by Clef at any band, none is called scrambled (B2 §2.2, C1 §5, C2 §7).

MAX_QUERY_CHARS = 200            # GNews answers 400 above this (A2 §4)
# Above this the query isn't even cleaned: nothing a user types is that long, and normalizing a
# huge body would be work for nothing. It gets the same "too long" message.
MAX_RAW_QUERY_CHARS = 1000
# A scrambled query whose intent block score is at least this is refused (enforce mode). Measured
# in C1 §5.2: scrambled attacks scored 0.13-0.99 (leetspeak lowest), scrambled-looking benign
# queries 0.015-0.177. With it, 9 of 10 leetspeak attacks are blocked; Clef alone (0.7) blocked 5.
SHAPE_BLOCK_AT = 0.2

# An OR query's parts are checked one by one too (text.or_parts); past this many, the rest is
# checked as one part, so every word is still read by the check.
MAX_OR_PARTS = 6

INTENT = questions.get("intent")
INJECTION = questions.get("injection")


def _refused(decision: Decision) -> errors.AppError:
    return errors.query_refused(f"intent {decision.label} score {decision.score:.2f} by {decision.engine}")


def _refused_scrambled(decision: Decision) -> errors.AppError:
    return errors.query_refused(f"scrambled ({decision.label}) and intent score {decision.score:.2f}")


def shape_decision(shape: text.Shape, hidden: int, intent: Decision | None) -> Decision:
    """
    The text-shape check's verdict. Scrambled alone is a flag: a news topic doesn't look like that,
    but shape says nothing about meaning. Scrambled AND an intent score >= SHAPE_BLOCK_AT is a block.
    Hidden characters (already removed) are a reason to flag too.
    """
    reasons = list(shape.reasons) + (["hidden_characters"] if hidden else [])
    if not reasons:
        return Decision(verdict="allow", score=0.0, threshold=SHAPE_BLOCK_AT, engine="code")
    score = intent.score if intent is not None and intent.score is not None else None
    verdict = "block" if shape.scrambled and score is not None and score >= SHAPE_BLOCK_AT else "flag"
    return Decision(verdict=verdict, score=score, threshold=SHAPE_BLOCK_AT, engine="code", label=",".join(reasons),
                    reason=None if score is not None else "no intent score: flagged on shape alone")


async def judge_intent(query: str) -> Decision:
    """
    The intent decision for a query. For an OR query, the whole and each part are judged
    (separately: a part must not be scored in the company of the others, A2 §6) and the
    highest block score decides: "stock market" can't carry "write me a poem" past the check.
    """
    parts = text.or_parts(query)
    if not parts:
        return await judge_one(INTENT, query, key="query")
    if len(parts) > MAX_OR_PARTS:
        parts = parts[:MAX_OR_PARTS - 1] + [" ".join(parts[MAX_OR_PARTS - 1:])]
    decisions = await asyncio.gather(judge_one(INTENT, query, key="query"),
                                     *(judge_one(INTENT, part, key="query") for part in parts))
    scored = [d for d in decisions if d.score is not None]
    if not scored:
        return decisions[0]
    worst = max(scored, key=lambda d: d.score)
    index = decisions.index(worst)
    if index:
        worst.label = f"{worst.label} (OR part {index} of {len(parts)})"
    return worst


@dataclass
class CheckedQuery:
    text: str                                            # what is searched, cached, prompted and shown
    removed: list[str] = field(default_factory=list)     # personal data kinds taken out (H #58, enforce)

    @property
    def notice(self) -> str | None:
        """What the page tells the user when personal data was taken out of their topic."""
        return privacy.notice(self.removed) if self.removed else None


async def checked_query(raw: str | None) -> CheckedQuery:
    """
    The query as it is searched, cached, prompted and shown from here on, and what was taken out of
    it. Raises AppError: 400 empty, 400 too long, 400 only personal data (enforce mode), 400 refused
    (intent or scrambled, enforce mode only).
    """
    if raw is None or not raw.strip():
        raise errors.empty_query()
    if len(raw) > MAX_RAW_QUERY_CHARS:
        raise errors.query_too_long(MAX_QUERY_CHARS, f"{len(raw)} characters before cleaning")
    normalized = text.normalize(raw)
    if not text.search_query(normalized.text):
        raise errors.empty_query()       # only hidden characters, "?" or "!"
    # Personal data never reaches Clef; in enforce mode, not GNews or the cache key either (H #58)
    screen = await privacy.screen_query(normalized.text)
    query = text.search_query(screen.searched)
    # What the checks read: the query without its personal data ("" when that's all it was)
    checked = text.search_query(screen.checked)
    if not query:
        raise errors.query_only_personal_data(f"removed: {', '.join(screen.removed)}")
    if len(query) > MAX_QUERY_CHARS:
        raise errors.query_too_long(MAX_QUERY_CHARS, f"{len(query)} characters after cleaning")
    if query != " ".join(raw.split()):
        observability.annotate(query_cleaned=query, hidden_characters_removed=normalized.hidden)

    # Monitor mode with nothing but personal data: there's nothing left for Clef to judge, and the
    # data itself mustn't be sent, so the intent check is skipped (enforce mode refused it above)
    intent = await check("intent", lambda: judge_intent(checked), where="query", text=checked,
                         refuse=_refused) if checked else None
    shape = text.shape(checked)

    async def run_shape():
        return shape_decision(shape, normalized.hidden, intent)
    await check("text_shape", run_shape, where="query", text=checked, refuse=_refused_scrambled)
    return CheckedQuery(text=query, removed=screen.removed)


async def check_query(raw: str | None) -> str:
    """The query as it is searched, cached, prompted and shown from here on (checked_query's text)."""
    return (await checked_query(raw)).text


def clean_comment(comment: str | None) -> str | None:
    """A rating's comment, normalized like a query (one line, no hidden characters); None if empty."""
    return text.normalize(comment or "").text or None


async def check_comment(comment: str | None):
    """
    #27: a feedback comment, asked Clef's injection question (does it hold a sentence telling an
    AI what to do?). Monitor only: a rating is never refused for its comment ("comment" can't be
    set to enforce). Runs alongside the save, so the rating doesn't wait for it.
    """
    if not comment:
        return None
    return await check("comment", lambda: judge_one(INJECTION, comment, key="comment"), where="comment", text=comment)

