"""
LLM safety (feature 3). The shared core every guard uses (Section B):

  settings     SAFETY_<CHECK>=off|monitor|enforce per guard, SAFETY_ENGINE=clef|embedding
  events       one SafetyEvent shape, on the trace and in the log
  guard        check(name, run): mode logic, timing, the event, the error in enforce mode
  questions    every Clef question, with its bands and version
  clef         the Clef client (Cloudflare Workers AI through the AI Gateway)
  classifier   the embedding classifier used when Clef can't run
  engine       judge(question, items): Clef, then the classifier, then fail open with a flag
  gemini       Gemini's own filter: explicit thresholds, blocks as values, GeminiBlocked (Section F)
  provenance   where each article came from: site, redirect, fetch time, text hash, date check,
               SAFETY_BLOCKED_DOMAINS (Section G)
  relevance    retrieved chunks that aren't about the query: Clef, swapped in enforce (Section G)
  pii          personal data by format and checksum: email, phone, card, Aadhaar, PAN, IBAN, IP (Section H)
  privacy      where pii runs: the query, comments (+ Clef's `names`), answers, traces, log lines (Section H)
  output       model output into the page: HTML allowlist (nh3), links only to sources, no images, length cap
               and timeline format flags (Section I)

The guards themselves (Sections C-L) call check() with a run() that usually calls judge().
"""
import asyncio

from services.safety import classifier, clef, gemini, settings
from services.safety.engine import Decision, judge, judge_one
from services.safety.guard import check
from services.safety.questions import QUESTIONS, get as question

__all__ = ["Decision", "check", "judge", "judge_one", "question", "QUESTIONS", "startup"]


async def startup():
    """At app start: loads the classifiers, reads today's Clef neurons, warms the Clef connection,
    prints the one safety line."""
    classifier.load_all()
    await asyncio.gather(clef.load_today(), clef.warm_up())
    if settings.engine() == "clef" and not clef.available():
        print(f"[WARN] Safety: {clef.status()}: every meaning-based check uses the embedding classifier.")
    line = (f"[INFO] Safety: engine {settings.engine()}, {clef.status()}, {classifier.status()}; {settings.summary()}; "
            f"{gemini.status(gemini.SETTINGS)}.")
    if clef.available() and clef.neurons_today():
        line = line[:-1] + f"; {clef.neurons_today():.0f} Clef neurons used today."
    print(line)
