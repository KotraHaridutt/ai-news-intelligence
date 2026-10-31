import secrets

from services import errors
from services.safety.engine import Decision
from services.safety.guard import check

# --- The canary (feature 3, Section D #36) ---
# Every answer's prompt carries a random marker made for that request, with "never write it". The
# model has no reason to repeat it, so if it shows up in the answer, the model leaked its
# instructions ("repeat everything above") or followed an injection that told it to. Nothing about
# meaning is guessed: the marker is ours, so an exact match is the right test.
# Enforce mode withholds the answer (decided 2026-10-31); monitor shows it without the marker, flagged.
# E moves the marker into Gemini's system instruction.

PREFIX = "mk-"
# A model may copy part of it, or change its case: the first MATCH_CHARS characters are enough
MATCH_CHARS = 10


def new() -> str:
    """A marker for one request: "mk-" and 12 random hex digits."""
    return PREFIX + secrets.token_hex(6)


def leaked(text: str, marker: str | None) -> bool:
    return bool(marker) and marker[:MATCH_CHARS].lower() in (text or "").lower()


def strip(text: str, marker: str | None) -> str:
    """The text without the marker (when there is one)."""
    return _without(text, marker) if marker and leaked(text, marker) else text


def _without(text: str, marker: str) -> str:
    """The text with every copy of the marker (any case, whole or its first MATCH_CHARS) taken out."""
    lowered, needle = text.lower(), marker[:MATCH_CHARS].lower()
    out, i = [], 0
    while (j := lowered.find(needle, i)) != -1:
        out.append(text[i:j])
        i = j + len(marker) if lowered[j:j + len(marker)] == marker.lower() else j + len(needle)
    out.append(text[i:])
    return "".join(out)


async def check_text(text: str, marker: str | None, where: str, prompt_type: str = "answer") -> str:
    """
    The text to show. Raises answer_withheld (502) in enforce mode when the marker leaked; in
    monitor mode the text comes back without the marker, and the leak is a block event on the trace.
    No marker (a prompt from the Hub without {canary}): nothing to check.
    """
    if not marker:
        return text
    found = leaked(text, marker)

    async def run():
        return Decision(verdict="block" if found else "allow", score=1.0 if found else 0.0, threshold=1.0, engine="code",
                        label="marker in the output" if found else None)
    await check("canary", run, where=where,
                refuse=lambda d: errors.answer_withheld(f"{prompt_type}: the request's canary marker appeared in the output"))
    return _without(text, marker) if found else text
