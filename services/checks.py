import re
from datetime import date, datetime

# --- Code checks on answers (feature 2, Section G #51) ---
# Free: plain code, no LLM call, milliseconds. Each check returns {"score": 0..1, "ok": bool, ...details}
# so the evaluation runner (evals/, Section H) and the live checks on every real answer
# (Section I) score the same thing the same way. They check what the prompts ask for in a form
# code can see; whether the content is TRUE is the LLM judges' job (evals/judges.py).
#
# `sources`: the chunks the answer was written from, each a dict with "source_name" (the outlet)
# and "published_at", as the app's retrieved documents carry them.

BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# "Based on the provided context..." and friends: the report prompt forbids them
BANNED_PHRASES = ("based on the provided context", "according to the provided context", "the provided context",
                  "the context provided", "based on the context", "provided articles")
# A timeline line: "2026-10-09: what happened (Outlet)", also with a partial date (2026-10 or 2026),
# a bullet or bold around the date, which the model sometimes adds
TIMELINE_LINE = re.compile(r"^\s*(?:[-*•]\s*)?\**(?P<date>\d{4}(?:-\d{2}(?:-\d{2})?)?)\**\s*:\s*(?P<event>.+?)\s*\((?P<outlet>[^()]+)\)\s*\.?\s*$")
CONSISTENT = re.compile(r"\b(largely|broadly|generally|mostly)\s+consistent\b|\bno (significant|direct|major) contradictions?\b", re.I)
REPORT_WORDS = (150, 1500)
# "According to X" often names a quoted firm or person ("according to Edwards Asset Management"),
# not an outlet. A name that isn't one of the sources fails the check only if it looks like a news
# outlet; the others are listed as details (found on gap 4's real report, Section G).
MEDIA_WORDS = re.compile(r"\b(news|times|post|journal|reuters|bloomberg|press|tribune|herald|guardian|wire|daily|telegraph|"
                         r"cnn|bbc|cnbc|gazette|insider|magazine|media|today|reporter|observer|express|standard|mail|"
                         r"chronicle|associated|agency|network|tv|radio)\b", re.I)


def _outlets(sources) -> list[str]:
    return sorted({(s.get("source_name") or "").strip() for s in sources or [] if (s.get("source_name") or "").strip()})


def _mentions(text: str, outlet: str) -> bool:
    """An outlet named in text: whole words, case-insensitive ("Reuters", "The Times of India")."""
    name = re.sub(r"^the\s+", "", outlet.strip(), flags=re.I)
    return bool(name) and re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text, re.I) is not None


def _newest_source(sources) -> date | None:
    days = []
    for s in sources or []:
        try:
            days.append(datetime.fromisoformat(str(s.get("published_at", "")).replace("Z", "+00:00")).date())
        except ValueError:
            continue
    return max(days) if days else None


def _lines(text: str) -> list[str]:
    return [line for line in (text or "").splitlines() if line.strip()]


def _result(checks: dict, **details) -> dict:
    """{"score": share of checks passed, "ok": all passed, "checks": {name: bool}, ...details}"""
    score = sum(checks.values()) / len(checks) if checks else 1.0
    return {"score": round(score, 4), "ok": all(checks.values()), "checks": checks, **details}


def check_report(answer: str, sources) -> dict:
    """The report prompt: prose, no bullets, no "based on the provided context", outlets named are real sources."""
    text = answer or ""
    words = len(text.split())
    bullets = [line for line in _lines(text) if BULLET.match(line)]
    banned = [p for p in BANNED_PHRASES if p in text.lower()]
    # "According to"/"according to" either case; the name itself must be capitalised
    named = [n.rstrip(".'’-") for n in
             re.findall(r"(?i:according to)\s+(?:(?i:the)\s+)?([A-Z][\w&.'’-]*(?:\s+[A-Z][\w&.'’-]*){0,4})", text)]
    outlets = _outlets(sources)
    unknown = sorted({n for n in named if not any(_mentions(n, o) or _mentions(o, n) for o in outlets)})
    unknown_outlets = [n for n in unknown if MEDIA_WORDS.search(n)]
    return _result({"no_bullets": not bullets, "no_banned_phrase": not banned, "named_outlets_in_sources": not unknown_outlets,
                    "length_ok": REPORT_WORDS[0] <= words <= REPORT_WORDS[1]},
                   words=words, bullets=len(bullets), banned=banned, unknown_outlets=unknown_outlets,
                   other_names=[n for n in unknown if n not in unknown_outlets])


def check_timeline(answer: str, sources, today: date | None = None) -> dict:
    """
    The timeline prompt: every line "YYYY-MM-DD: event (outlet)" (or a partial date), newest first,
    each outlet a real source, no event dated after the day the answer was written (`today`). An
    event after the newest source is allowed (an article can announce Monday's market opening on
    Friday) and only listed. Dates the prompt asks to resolve ("on Tuesday") can only be checked
    against the text by the judge.
    """
    lines = _lines(answer)
    parsed = [TIMELINE_LINE.match(line) for line in lines]
    events = [m for m in parsed if m]
    outlets = _outlets(sources)
    dates = [m.group("date") for m in events]
    # A partial date (2026-10, 2026) covers a range: a line is in order after the previous one if
    # some day of its range is on or before some day of the previous one's range
    high = lambda d: d + "-12-31"[len(d) - 4:] if len(d) < 10 else d
    low = lambda d: d + "-01-01"[len(d) - 4:] if len(d) < 10 else d
    in_order = sum(1 for a, b in zip(dates, dates[1:]) if high(a) >= low(b))
    today = today or date.today()
    newest = _newest_source(sources)
    full_dates = [date.fromisoformat(d) for d in dates if len(d) == 10]
    future = [d.isoformat() for d in full_dates if d > today]
    after_newest = [d.isoformat() for d in full_dates if newest and d > newest]
    unknown = sorted({m.group("outlet").strip() for m in events if not any(_mentions(m.group("outlet"), o) or _mentions(o, m.group("outlet")) for o in outlets)})
    checks = {"has_events": bool(events),
              "all_lines_formatted": bool(lines) and len(events) == len(lines),
              "newest_first": in_order == max(0, len(events) - 1),
              "no_future_date": not future,
              "outlets_in_sources": not unknown}
    return _result(checks, lines=len(lines), events=len(events), formatted_share=round(len(events) / len(lines), 4) if lines else 0.0,
                   out_of_order=max(0, len(events) - 1) - in_order, future_dates=future, after_newest_source=after_newest,
                   unknown_outlets=unknown)


def _points(text: str) -> list[str]:
    """The contradiction points: bullets or numbered items, else paragraphs."""
    lines = _lines(text)
    items, current = [], None
    if any(BULLET.match(line) for line in lines):
        for line in lines:
            if BULLET.match(line):
                current = BULLET.sub("", line)
                items.append(current)
            elif items:
                items[-1] += " " + line.strip()
        return items
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def check_contradictions(answer: str, sources) -> dict:
    """The contradictions prompt: each point names the outlets behind both sides (two or more), unless
    the answer says the reports are consistent. The same outlet twice is allowed with article titles,
    which code can't tell apart, so such points are counted as `single_outlet` for the judge to settle."""
    outlets = _outlets(sources)
    points = _points(answer)
    if not points:
        return _result({"has_content": False})
    if CONSISTENT.search(answer or "") and len(points) <= 2:
        return _result({"has_content": True, "points_name_two_outlets": True}, points=len(points), consistent=True)
    named = [sorted({o for o in outlets if _mentions(p, o)}) for p in points]
    two = sum(1 for n in named if len(n) >= 2)
    return _result({"has_content": True, "points_name_two_outlets": two == len(points)},
                   points=len(points), two_outlet_points=two, single_outlet=sum(1 for n in named if len(n) == 1),
                   no_outlet=sum(1 for n in named if not n), consistent=False)


def check_fact_check(answer, sent_ids) -> dict:
    """The fact check's JSON: readable, every story it was sent answered exactly once, no other ids.
    `answer`: the parsed FactCheckAnswer, or None if it couldn't be read."""
    if answer is None:
        return _result({"valid_json": False}, sent=len(sent_ids))
    given = [a.story_id for a in answer.new_stories] + [a.story_id for a in answer.updates]
    sent = set(sent_ids)
    duplicates = sorted({i for i in given if given.count(i) > 1})
    return _result({"valid_json": True, "all_answered": sent <= set(given), "answered_once": not duplicates,
                    "no_unknown_ids": set(given) <= sent},
                   missing=sorted(sent - set(given)), duplicates=duplicates, unknown=sorted(set(given) - sent))


def check_retrieval(documents, k: int = 10, cap: int = 3, index=None) -> dict:
    """
    What retrieval picked: at most `cap` chunks per story (the cap is soft: it may be exceeded when
    the stories can't fill k under it), and how many stories the chunks cover. `index`: every chunk
    of the index (documents or metadata dicts), when known: then going over the cap is allowed
    exactly when the index's stories can't give k chunks at `cap` each. Without it, only when the
    retrieved stories x cap < k (Section H: with k=15, OpenAI's index could give 14, Tesla's 9).
    """
    story = lambda m: ("theme", m.get("theme_id")) if m.get("theme_id", -1) != -1 else ("article", m.get("source"))
    meta = lambda doc: getattr(doc, "metadata", doc)
    per_story = {}
    for doc in documents:
        per_story[story(meta(doc))] = per_story.get(story(meta(doc)), 0) + 1
    over = {str(s): n for s, n in per_story.items() if n > cap}
    if index is not None:
        available = {}
        for doc in index:
            available[story(meta(doc))] = available.get(story(meta(doc)), 0) + 1
        fillable = sum(min(cap, n) for n in available.values())
    else:
        fillable = len(per_story) * cap
    soft_ok = not over or fillable < k              # over the cap only when the cap couldn't fill k
    return _result({"k_chunks": len(documents) == k or len(documents) < k and not over, "story_cap": soft_ok},
                   chunks=len(documents), stories=len(per_story), max_per_story=max(per_story.values()) if per_story else 0)


def check_injection(answer: str, signs) -> dict:
    """None of what a hijacked answer would contain (gap 4's planted article), case-insensitive."""
    found = [s for s in signs if s.lower() in (answer or "").lower()]
    return _result({"not_hijacked": not found}, found=found)


CHECKS = {"report": check_report, "timeline": check_timeline, "contradictions": check_contradictions}


def check_answer(view: str, answer: str, sources) -> dict:
    """The checks for one of the three views."""
    return CHECKS[view](answer, sources)
