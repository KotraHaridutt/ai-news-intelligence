import re
from dataclasses import dataclass, field

from services import observability
from services.safety import pii, questions, settings, text as T
from services.safety.engine import Decision, judge, judge_one
from services.safety.guard import check

# --- Personal data in what comes in and goes out (feature 3, Section H #58-#61, #64) ---
# The detectors are format checks with checksums (pii.py). This module decides where they run and
# what happens to what they find. SAFETY_PII sets the mode, like every guard:
#
#   off      nothing is looked at (the app as before H)
#   monitor  personal data is masked in the copies that only leave the app for other services or
#            are kept by it: the text Clef judges (and the AI Gateway logs), the comment saved to
#            LangSmith, traces, log lines. What the user searches for and reads is unchanged, and
#            every find is a "pii" event on the trace, so false alarms can be seen first (rule 3).
#   enforce  also what the user gets: personal data is removed from the query before GNews, the
#            cache key, the prompt and the saved answer (and the user is told), and an answer's
#            contact details that aren't in its sources are masked.
#
# Masking a copy in monitor mode is safe to do at once: no one is refused and nothing they see
# changes, and the checksums leave few false alarms (0 in 2.77M characters of saved articles and
# answers, H §4). Names in comments are judged by meaning (Clef), so they have their own setting,
# SAFETY_NAMES, and are only masked in enforce mode.
#
# Names are never touched in queries or answers: public figures are what news is about.

# An answer's personal data, masked when it isn't in the sources. IP addresses aren't: in news about
# cyber attacks they are the attacker's servers, which is the news.
ANSWER_KINDS = tuple(kind for kind in pii.KINDS if kind != "ip")
NAME_REMOVED = "[removed: names a private person]"
NAMES = questions.get("names")


def active() -> bool:
    return settings.mode("pii") != "off"


def for_log(text) -> str:
    """A log line's user text (a query, an error's reason) with personal data tagged."""
    text = "" if text is None else str(text)
    return pii.mask(text) if active() else text


def _for_traces(value: str) -> str:
    return pii.mask(value) if active() else value


# #61: every string a trace carries (inputs, outputs, metadata, errors, the LLM's prompt and answer)
# goes through observability's scrubber; this adds the detectors to it, after the secret masking.
observability.add_redactor(_for_traces)


def _decision(kinds: list[str]) -> Decision:
    """The pii event: block when something was found (removed or masked in enforce mode)."""
    return Decision(verdict="block" if kinds else "allow", score=1.0 if kinds else 0.0, threshold=1.0, engine="code",
                    label=",".join(kinds) or None)


async def _record(kinds: list[str], where: str, text: str | None) -> Decision | None:
    async def run():
        return _decision(kinds)
    return await check("pii", run, where=where, text=text)


# ---------------------------------------------------------------- queries (#58)

@dataclass
class QueryScreen:
    checked: str                  # what the checks read (Clef, the shape check): personal data removed
    searched: str                 # what is searched: `checked` in enforce mode, the original otherwise
    removed: list[str] = field(default_factory=list)   # kinds removed from the search (enforce only)


async def screen_query(normalized: str) -> QueryScreen:
    """
    #58, before the intent check: a query's personal data never reaches Clef (or the AI Gateway's
    logs); in enforce mode it doesn't reach GNews, the cache key, the prompt or the saved answer either.
    `normalized` is text.normalize()'s one-line text: the detectors read it before "-" becomes a
    space, so "ravi-kumar@mail.com" is still one address.
    """
    stripped, found = pii.remove(normalized)
    kinds = pii.kinds_of(found)
    decision = await _record(kinds, "query", stripped)
    if decision is None:
        return QueryScreen(checked=normalized, searched=normalized)
    if decision.enforced:
        return QueryScreen(checked=stripped, searched=stripped, removed=kinds)
    return QueryScreen(checked=stripped, searched=normalized)


def notice(removed: list[str]) -> str:
    """What the page shows above the answer when the query lost personal data."""
    return f"We removed {pii.describe(removed)} from your topic before searching, to keep it private."


# ---------------------------------------------------------------- feedback comments (#59, #64)

async def screen_comment(comment: str | None) -> str | None:
    """
    #59: a comment's personal data, tagged ("[email]") before Clef reads it and before it is saved
    to LangSmith, in monitor and enforce mode: the comment is only stored, never shown to anyone.
    """
    if not comment:
        return comment
    found = pii.find(comment)
    masked = pii.replace(comment, found, pii.tag) if found else comment
    decision = await _record(pii.kinds_of(found), "comment", masked)
    return comment if decision is None else masked


async def screen_names(comment: str | None) -> str | None:
    """
    #64: does the comment name a private person (a neighbour, a relative, the writer)? Clef's
    `names` question; no NER model. In enforce mode the sentences that do are replaced by
    NAME_REMOVED (each sentence of a flagged comment is asked the same question, like D's chunk ->
    sentence scan); a one-sentence comment, or one where no single sentence decides, loses all of
    its text. The rating itself is always saved. Clef down: fails open (#18), the comment is kept.
    """
    if not comment:
        return comment
    decision = await check("names", lambda: judge_one(NAMES, comment, key="comment"), where="comment", text=comment)
    if decision is None or not decision.enforced:
        return comment
    sentences = T.sentences(comment)
    if len(sentences) < 2:
        return NAME_REMOVED
    decisions = await judge(NAMES, {f"s{i}": sentence for i, sentence in enumerate(sentences)})
    flagged = [sentences[i] for i in range(len(sentences)) if decisions[f"s{i}"].verdict == "block"]
    if not flagged or any(d.score is None for d in decisions.values()):
        return NAME_REMOVED
    for sentence in flagged:
        comment = comment.replace(sentence, NAME_REMOVED, 1)
    observability.annotate(names_removed_sentences=len(flagged), names_sentences=len(sentences))
    return comment


# ---------------------------------------------------------------- answers (#60)
# Measured in H §4: every contact detail found in saved articles was published on purpose:
# suicide helplines, police tip lines, a newsroom's address. Masking those would cut the news
# (rule 1). A detail that IS in the sources the model read is the article's, and stays. One that
# isn't came from somewhere else: the user's query, or the model (made up, or steered by a page),
# and that is what a reader must never be sent to.

_SEPARATORS = re.compile(r"[\s().\-]")


def _in_sources(kind: str, value: str, sources: str, compact_sources: str) -> bool:
    if kind in ("phone", "card", "aadhaar"):
        digits = re.sub(r"\D", "", value)
        # A phone number's last 10 digits: "+91 98765 43210" in the answer, "098765-43210" in the article
        return (digits[-10:] if kind == "phone" else digits) in compact_sources
    if kind in ("pan", "iban"):
        return re.sub(r"\s", "", value).upper() in compact_sources.upper()
    return value.lower() in sources.lower()


def unsourced(text: str, sources: str, found: list[pii.Found] | None = None) -> list[pii.Found]:
    """The personal data in `text` (ANSWER_KINDS, or `found` when already looked up) that the sources don't contain."""
    found = pii.find(text, ANSWER_KINDS) if found is None else found
    if not found:
        return []
    compact = _SEPARATORS.sub("", sources)
    return [f for f in found if not _in_sources(f.kind, text[f.start:f.end], sources, compact)]


async def screen_answer(text: str, sources: str, where: str) -> str:
    """The answer to show and save: in enforce mode, contact details missing from `sources` masked."""
    if not active():
        return text
    found = pii.find(text, ANSWER_KINDS)
    missing = unsourced(text, sources, found)
    if len(found) > len(missing):                # the articles' own (helplines, tip lines): kept
        observability.annotate(pii_from_sources=len(found) - len(missing))
    decision = await _record(pii.kinds_of(missing), where, None)
    if decision is not None and decision.enforced:
        return pii.replace(text, missing, pii.removed_tag)
    return text


async def screen_texts(texts: dict[str, str], sources: str, where: str) -> dict[str, str]:
    """screen_answer for a check's story texts (feature 1's update): one event for all of them."""
    if not active() or not texts:
        return texts
    missing = {key: unsourced(text, sources) for key, text in texts.items()}
    kinds = pii.kinds_of([f for found in missing.values() for f in found])
    decision = await _record(kinds, where, None)
    if decision is not None and decision.enforced:
        return {key: pii.replace(text, missing[key], pii.removed_tag) for key, text in texts.items()}
    return texts
