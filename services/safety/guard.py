import time
from typing import Awaitable, Callable

from services import observability
from services.errors import AppError
from services.safety import events, questions, settings
from services.safety.engine import Decision

# --- The one helper every guard calls (feature 3, Section B #13) ---
# safety.check(name, run) skips the check when it's off, times it, records its event, and raises
# the guard's AppError only in enforce mode. Guards then never repeat the mode logic.


async def check(name: str, run: Callable[[], Awaitable[Decision]], *, where: str,
                refuse: Callable[[Decision], AppError] | None = None, text: str | None = None) -> Decision | None:
    """
    Runs one check. `run()` returns its Decision; `refuse(decision)` builds the error a block
    raises in enforce mode (a guard that removes text instead passes None and reads
    decision.enforced). `text` is only hashed, for the event.
    Returns the Decision, or None when the check is off. A check that fails unexpectedly fails
    open with a flag (#18): a bug in a guard never fails the request.
    """
    mode = settings.mode(name)
    if mode == "off":
        return None
    started = time.perf_counter()
    try:
        decision = await run()
    except AppError:
        raise
    except Exception as e:
        decision = Decision.failed_open(f"check failed: {type(e).__name__}: {str(e)[:120]}")
    decision.enforced = decision.verdict == "block" and mode == "enforce"
    events.record(events.SafetyEvent(
        check=name, verdict=decision.verdict, mode=mode, where=where, score=decision.score,
        threshold=decision.threshold, engine=decision.engine, ms=(time.perf_counter() - started) * 1000, label=decision.label,
        text_hash=observability.short_hash(text) if text is not None else None,
        questions_version=questions.VERSION if decision.engine == "clef" else None,
        cached=decision.cached, reason=decision.reason, enforced=decision.enforced))
    if decision.enforced and refuse is not None:
        raise refuse(decision)
    return decision
