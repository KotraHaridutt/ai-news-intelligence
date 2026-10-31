from dataclasses import asdict, dataclass

from services import observability

# --- Safety events (feature 3, Section B #12) ---
# One shape for what every guard decided, so safety can be measured the same way everywhere.
# Each event goes on the current trace span (metadata "safety", a list, and a tag
# safety:<check>:<verdict>), into the request's totals, and into the log as one [SAFETY] line.
# An event never carries the text that was checked: only scores and a short hash of the text.

VERDICTS = ("allow", "flag", "block")


@dataclass
class SafetyEvent:
    check: str                # the guard's name, as in settings.CHECKS
    verdict: str              # allow / flag / block: what the check decided
    mode: str                 # monitor / enforce
    where: str                # what was checked: "query", "chunk", "answer", ...
    score: float | None       # the block score, 0-1 (None when no engine could run)
    threshold: float | None   # the score from which it blocks
    engine: str               # clef / embedding / code / none
    ms: float                 # time the check took
    label: str | None = None              # what it saw: the likeliest class ("off_topic"), shape reasons
    text_hash: str | None = None          # observability.short_hash of the checked text
    questions_version: str | None = None  # the Clef question set's version (questions.VERSION)
    cached: bool = False                  # the verdict came from the Redis cache (no Clef call)
    reason: str | None = None             # why the fallback decided, or why it failed open
    enforced: bool = False                # a block that was acted on (enforce mode only)

    def as_dict(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        if self.score is not None:
            data["score"] = round(self.score, 4)
        data["ms"] = round(self.ms)
        return data

    def log_line(self) -> str:
        score = "-" if self.score is None else f"{self.score:.3f}"
        line = (f"[SAFETY] {self.check} {self.verdict}{' (enforced)' if self.enforced else ''} "
                f"score={score} engine={self.engine} mode={self.mode} where={self.where} {round(self.ms)}ms")
        if self.label:
            line += f" label={self.label}"
        if self.cached:
            line += " cached"
        if self.text_hash:
            line += f" text={self.text_hash}"
        if self.reason:
            line += f" reason={self.reason}"
        return line


def record(event: SafetyEvent):
    """Puts the event on the current span (no-op outside one), in the request's totals and in the log."""
    run = observability.current_run()
    if run is not None:
        earlier = (run.metadata or {}).get("safety") or []
        run.add_metadata({"safety": [*earlier, event.as_dict()]})
        run.add_tags([f"safety:{event.check}:{event.verdict}"])
    observability.add_totals(**{f"safety_{event.verdict}s": 1, "safety_ms": round(event.ms)})
    print(event.log_line())
