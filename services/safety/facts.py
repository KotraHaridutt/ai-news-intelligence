import re
from dataclasses import dataclass, field
from datetime import date, datetime

from services import checks

# --- Figures, dates and named sources in an answer, against its own sources (feature 3, J #76, #78) ---
# Plain code, no model call. Clef catches a changed fact in prose; embeddings can't tell "40%" from
# "4%". So every figure, date and attributed name an answer sentence states must be found in the
# sources the answer was written from. What counts as found comes from THIS request's sources (their
# text, titles, outlets and publish dates), never from a list. The patterns below parse formats
# (numerals, calendar dates, the attribution forms the prompts prescribe), like the PII detectors
# parse an email address: they don't decide anything by which words are used.

MONTHS = {name: i for i, names in enumerate(
    (("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
     ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
     ("december", "dec")), 1) for name in names}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTH = r"(?P<month>" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d{2})"

# Dates, most specific first. Each match's span is taken out before figures are read.
DATE_PATTERNS = [
    re.compile(r"(?<!\d)" + _YEAR + r"-(?P<mnum>0[1-9]|1[0-2])-(?P<dnum>0[1-9]|[12]\d|3[01])(?!\d)"),
    re.compile(r"\b" + _MONTH + r"\s+" + _DAY + r"(?:,?\s+" + _YEAR + r")?\b", re.I),
    re.compile(r"\b" + _DAY + r"\s+(?:of\s+)?" + _MONTH + r"(?:,?\s+" + _YEAR + r")?\b", re.I),
    re.compile(r"(?<!\d)" + _YEAR + r"-(?P<mnum>0[1-9]|1[0-2])(?![\d-])"),
    re.compile(r"\b" + _MONTH + r"\s+" + _YEAR + r"\b", re.I),
]
TIME = re.compile(r"(?<![\d:.])(?P<h>[01]?\d|2[0-3])[:.](?P<m>[0-5]\d)(?!\d)(?:\s*(?P<ap>[ap])\.?m\.?)?", re.I)
TIME_AP = re.compile(r"(?<![\d:.])(?P<h>1[0-2]|0?[1-9])\s*(?P<ap>[ap])\.?m\.?(?![a-z])", re.I)
# A figure: digits with optional thousands separators and decimals, then an optional scale. A sign
# before it is fine ("-9.8%": the answer says "fell 9.8%"), and so is a word after a hyphen ("$99-per-month").
# Not part of a word or code: "Q3", "TV18", "COVID-19", "737-800", "10/20" keep their digits out.
FIGURE = re.compile(r"(?<![\w.])(?<!\w[-/])(?P<num>\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(?P<dec>\d+))?"
                    r"(?:\s*(?P<scale>trillion|billion|million|thousand|crore|lakh|bn|mn|tn|m|k)\b)?"
                    r"(?P<ord>st|nd|rd|th)?(?!\w)(?![-/]\d)", re.I)
SCALES = {"thousand": 1e3, "k": 1e3, "lakh": 1e5, "million": 1e6, "m": 1e6, "mn": 1e6, "crore": 1e7,
          "billion": 1e9, "bn": 1e9, "trillion": 1e12, "tn": 1e12}
# Numbers written as words in the SOURCES ("twelve people", "two hundred"), so an answer that writes
# 12 still finds them. Answers' own number words aren't checked ("one of the", "a third").
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
NUMBER_WORDS = re.compile(r"\b(?:(?:" + "|".join(list(_TENS) + list(_UNITS)) + r"|hundred|thousand|million|billion|and)"
                          r"(?:[\s-]+|\b))+", re.I)
# The attribution forms the prompts prescribe: "according to Reuters" (report, fact check),
# "<Outlet> reports [X], while <Outlet> says [Y]" (contradictions), "(Outlet)" ending a timeline line
_NAME = r"(?P<name>(?:[A-Z][\w&.'’-]*)(?:\s+(?:of|the|and|&|[A-Z][\w&.'’-]*)){0,5})"
ATTRIBUTIONS = [re.compile(r"(?i:according\s+to)\s+(?:(?i:the)\s+)?" + _NAME),
                re.compile(_NAME + r"\s+(?:reports|reported|says|said)\b(?!\s+by\b)"),
                re.compile(r"\b(?i:reported\s+by)\s+(?:(?i:the)\s+)?" + _NAME)]
_TRAILING = re.compile(r"(?:\s+(?:of|the|and|&))+$")


@dataclass(frozen=True)
class Figure:
    value: float
    precision: float          # half of it is the rounding allowed ("1.2 million" covers 1,150,000-1,250,000)
    text: str


@dataclass
class Sources:
    """What an answer's sources contain, read once per answer."""
    text: str                                   # titles + text, lowercased for name lookups
    figures: list[float] = field(default_factory=list)
    dates: set = field(default_factory=set)     # (year | None, month, day)
    months: set = field(default_factory=set)    # (year | None, month)
    years: set = field(default_factory=set)
    times: set = field(default_factory=set)     # (hour 0-23, minute)
    published: list[date] = field(default_factory=list)
    outlets: list[str] = field(default_factory=list)


def _date_parts(m) -> tuple[int | None, int, int | None]:
    g = m.groupdict()
    year = int(g["year"]) if g.get("year") else None
    month = int(g["mnum"]) if g.get("mnum") else MONTHS[g["month"].lower().rstrip(".")]
    day = g.get("dnum") or g.get("day")
    return year, month, int(day) if day else None


def find_dates(text: str) -> list[tuple[tuple, tuple[int, int]]]:
    """[((year, month, day | None), span)], no two overlapping; impossible days dropped."""
    found, taken = [], []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            year, month, day = _date_parts(m)
            if day is not None and not 1 <= day <= 31:
                continue
            found.append(((year, month, day), m.span()))
            taken.append(m.span())
    return sorted(found, key=lambda f: f[1][0])


def _blank(text: str, spans) -> str:
    out = list(text)
    for s, e in spans:
        out[s:e] = " " * (e - s)
    return "".join(out)


def find_times(text: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    found = []
    for m in TIME.finditer(text):
        h, mm = int(m["h"]), int(m["m"])
        if m["ap"] and h <= 12:
            h = h % 12 + (12 if m["ap"].lower() == "p" else 0)
        found.append(((h, mm), m.span()))
    for m in TIME_AP.finditer(text):
        if any(s <= m.start() < e for _, (s, e) in found):
            continue
        h = int(m["h"]) % 12 + (12 if m["ap"].lower() == "p" else 0)
        found.append(((h, 0), m.span()))
    return found


def find_figures(text: str) -> list[Figure]:
    """Figures in text whose dates and times are already blanked. Years (1900-2099 written alone) aren't
    figures: they're checked as dates."""
    out = []
    for m in FIGURE.finditer(text):
        digits = m["num"].replace(",", "")
        dec = m["dec"] or ""
        scale = SCALES.get((m["scale"] or "").lower(), 1.0)
        if not m["scale"] and not dec and "," not in m["num"] and len(digits) == 4 and 1900 <= int(digits) <= 2099:
            continue
        value = float(f"{digits}.{dec}" if dec else digits) * scale
        precision = (10 ** -len(dec) if dec else 1.0) * scale
        out.append(Figure(value, precision, m.group(0).strip()))
    return out


def _word_number(words: str) -> float | None:
    total, current, seen = 0, 0, False
    for word in re.split(r"[\s-]+", words.lower().strip()):
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in ("thousand", "million", "billion"):
            total += (current or 1) * {"thousand": 1e3, "million": 1e6, "billion": 1e9}[word]
            current = 0
        elif word == "and" or not word:
            continue
        else:
            return None
        seen = True
    return float(total + current) if seen else None


def _published(value) -> date | None:
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(str(value or "")[:10])
        except ValueError:
            return None


def read_sources(sources: list[dict]) -> Sources:
    """sources: [{"title", "source_name", "published_at", "text"}], one per chunk (or article) the answer
    was written from."""
    blob = "\n".join(f"{s.get('title') or ''}\n{s.get('text') or ''}" for s in sources)
    out = Sources(text=blob.lower(), outlets=sorted({(s.get("source_name") or "").strip() for s in sources if s.get("source_name")}))
    out.published = sorted({d for d in (_published(s.get("published_at")) for s in sources) if d})
    dates = find_dates(blob)
    for (year, month, day), _ in dates:
        if day is not None:
            out.dates.add((year, month, day))
        out.months.add((year, month))
        if year:
            out.years.add(year)
    times = find_times(blob)
    out.times = {t for t, _ in times}
    rest = _blank(blob, [span for _, span in dates] + [span for _, span in times])
    out.figures = [f.value for f in find_figures(rest)]
    out.figures += [v for v in (_word_number(m.group(0)) for m in NUMBER_WORDS.finditer(rest)) if v is not None]
    out.years |= {int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", blob)}
    for d in out.published:
        out.dates.add((d.year, d.month, d.day))
        out.months.add((d.year, d.month))
        out.years.add(d.year)
    return out


def _figure_found(figure: Figure, src: Sources) -> bool:
    slack = figure.precision / 2 + 1e-9 * max(1.0, abs(figure.value))
    return any(abs(v - figure.value) <= slack for v in src.figures)


def _date_found(year, month, day, src: Sources) -> bool:
    if day is None:
        return (year, month) in src.months or (None, month) in src.months and (year is None or year in src.years)
    if (year, month, day) in src.dates or (None, month, day) in src.dates and (year is None or year in src.years):
        return True
    if year is None:
        return any((m, d) == (month, day) for (_, m, d) in src.dates)
    try:
        when = date(year, month, day)
    except ValueError:
        return False
    # A day the articles name relative to when they were published ("on Tuesday", "yesterday"), as the
    # timeline prompt asks the model to resolve: within a day of a publish date, or the named weekday
    # within the week around one
    weekday = WEEKDAYS[when.weekday()]
    for published in src.published:
        if abs((when - published).days) <= 1:
            return True
        if abs((when - published).days) <= 7 and weekday in src.text:
            return True
    return False


def _name_found(name: str, src: Sources) -> bool:
    """An attributed name is one of the sources' outlets, or every word of it appears in their text (a quoted
    firm or person: "Father Velazquez" for "Father Filiberto Velazquez")."""
    name = _TRAILING.sub("", name.strip().rstrip(".,;:’'")).strip()
    if not name:
        return True
    if any(checks._mentions(name, o) or checks._mentions(o, name) for o in src.outlets):
        return True
    words = [w.lower() for w in re.findall(r"[\w&'’]+", name) if w.lower() not in ("of", "the", "and") and len(w) > 1]
    return all(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", src.text) for w in words)


def attributions(sentence: str) -> list[str]:
    names = [m.group("name") for pattern in ATTRIBUTIONS for m in pattern.finditer(sentence)]
    line = checks.TIMELINE_LINE.match(sentence)
    if line:
        names.append(line.group("outlet"))
    return names


@dataclass
class Problems:
    figures: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def __bool__(self):
        return bool(self.figures or self.dates or self.names)

    def labels(self) -> list[str]:
        return [*(["figure"] * len(self.figures)), *(["date"] * len(self.dates)), *(["source"] * len(self.names))]


def check_sentence(sentence: str, src: Sources) -> Problems:
    """The figures, dates, times and attributed names in one answer sentence that its sources don't contain."""
    text = re.sub(r"[*_`]", "", sentence)
    problems = Problems()
    dates = find_dates(text)
    for (year, month, day), span in dates:
        if not _date_found(year, month, day, src):
            problems.dates.append(text[span[0]:span[1]])
    times = find_times(text)
    for (h, m), span in times:
        if (h, m) not in src.times and not (m == 0 and ((h + 12) % 24, 0) in src.times):
            problems.figures.append(text[span[0]:span[1]])
    rest = _blank(text, [span for _, span in dates] + [span for _, span in times])
    for year in re.findall(r"(?<![\w.,\-])((?:19|20)\d{2})(?![\w,\-])", rest):
        if int(year) not in src.years:
            problems.dates.append(year)
    for figure in find_figures(rest):
        if not _figure_found(figure, src):
            problems.figures.append(figure.text)
    problems.names = [_clean_name(n) for n in attributions(text) if not _name_found(n, src)]
    return problems


def _clean_name(name: str) -> str:
    return _TRAILING.sub("", name.strip().rstrip(".,;:’'")).strip()
