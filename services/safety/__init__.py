"""
LLM safety (feature 3). The shared core every guard uses (Section B):

  settings     SAFETY_<CHECK>=off|monitor|enforce per guard, SAFETY_ENGINE=clef|embedding
  events       one SafetyEvent shape, on the trace and in the log
  guard        check(name, run): mode logic, timing, the event, the error in enforce mode
  questions    every Clef question, with its bands and version
  clef         the Clef client (Cloudflare Workers AI through the AI Gateway)
  classifier   the embedding classifier used when Clef can't run
  engine       judge(question, items): Clef, then the classifier, then fail open with a flag

The guards themselves (Sections C-L) call check() with a run() that usually calls judge().
"""
import asyncio

from services.safety import classifier, clef, settings
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
    line = f"[INFO] Safety: engine {settings.engine()}, {clef.status()}, {classifier.status()}; {settings.summary()}."
    if clef.available() and clef.neurons_today():
        line = line[:-1] + f"; {clef.neurons_today():.0f} Clef neurons used today."
    print(line)
