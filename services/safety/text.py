import base64
import re
import unicodedata
from dataclasses import dataclass, field

# --- Text the user or a page sends, made safe to read (feature 3, Section C #22, #23) ---
# normalize() runs before anything else reads the text: the checks, the search, the cache key,
# the prompts. It removes characters a person can't see but a model reads, so a hidden
# instruction or a split-up word can't slip past a check that reads the visible text. D uses the
# same function on article text (#31).
#
# shape() describes what the text LOOKS like, never which words it uses (rule 2: no keyword
# lists): digits mixed into words (leetspeak, Clef's one weak spot in A #7c), long base64 or hex
# runs, letters spaced out one by one, words mixing alphabets. A news topic doesn't look like that.

# Unicode categories removed outright:
#   Cf  format characters: zero-width space/joiner (U+200B-200D), word joiner, BOM, soft hyphen,
#       bidi overrides and isolates (U+202A-202E, U+2066-2069), and the "tag" block (U+E0000-E007F)
#       that can spell out a hidden ASCII message
#   Co, Cs, Cn  private-use, surrogates, unassigned
_REMOVE = {"Cf", "Co", "Cs", "Cn"}
# Variation selectors are marks (Mn), but they carry no visible change on letters and can encode
# hidden bytes after a character ("emoji smuggling"). Removed too.
_VARIATION_SELECTORS = re.compile(f"[{chr(0xFE00)}-{chr(0xFE0F)}{chr(0xE0100)}-{chr(0xE01EF)}]")   # written as chr(): no invisible character in this file
# Zero-width joiner and non-joiner are removed too, but don't count as hidden: Devanagari and
# other scripts use them inside ordinary words, so a Hindi query isn't flagged for them.
_JOINERS = {chr(0x200C), chr(0x200D)}
_SPACES = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


@dataclass
class Normalized:
    text: str
    removed: int = 0               # invisible or format characters taken out
    hidden: int = 0                # of those, the ones no ordinary text needs (all but the joiners)
    changed: bool = False          # anything differs from the input (NFKC, removals, spaces)


def normalize(text: str, keep_newlines: bool = False) -> Normalized:
    """
    NFKC (full-width and compatibility forms become plain ones: "ｉｇｎｏｒｅ" -> "ignore"), then
    drops format, private-use and unassigned characters and variation selectors, turns control
    characters into spaces and collapses whitespace. keep_newlines keeps paragraph breaks
    (articles); otherwise everything becomes one line (queries, comments).
    """
    original = text or ""
    nfkc = unicodedata.normalize("NFKC", original)
    out, removed, joiners = [], 0, 0
    for ch in _VARIATION_SELECTORS.sub(lambda m: "\x00" * len(m.group()), nfkc):
        if ch == "\x00":
            removed += 1
            continue
        category = unicodedata.category(ch)
        if category in _REMOVE:
            removed += 1
            joiners += ch in _JOINERS
        elif category == "Cc":
            out.append("\n" if keep_newlines and ch == "\n" else " ")
        else:
            out.append(ch)
    cleaned = "".join(out)
    if keep_newlines:
        cleaned = _BLANK_LINES.sub("\n\n", _SPACES.sub(" ", cleaned))
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n")).strip()
    else:
        cleaned = " ".join(cleaned.split())
    return Normalized(text=cleaned, removed=removed, hidden=removed - joiners, changed=cleaned != original)


def strip_invisible(text: str) -> str:
    """
    Article text (#31): only the invisible characters go (the same set normalize removes), nothing
    else changes. No NFKC and no whitespace changes, so an ordinary article keeps its exact text,
    its chunks keep their hashes and their cached vectors and verdicts stay valid.
    """
    text = text or ""
    if text.isascii():
        return text
    text = _VARIATION_SELECTORS.sub("", text)
    return "".join(ch for ch in text if unicodedata.category(ch) not in _REMOVE)


# --- Sentences (D #33) ---
# Where a flagged chunk is split to find the sentence that speaks to the AI. A sentence ends at
# . ! ? or an ellipsis followed by a space and a capital, a digit, a quote or a bracket, or at a
# line break. A short abbreviation before the period ("U.S.", "Mr.", "Jan.") doesn't end one, and
# neither does a decimal (no space after its point). A wrong split only means a slightly shorter
# or longer piece is judged.
_CLOSERS = "\"'”’)]"
_OPENERS = "\"'“‘(["
_SENTENCE_END = re.compile(r"(?<=[.!?…])[" + re.escape(_CLOSERS) + r"]*\s+(?=[" + re.escape(_OPENERS) + r"]?[A-Z0-9])|\n+")
_ABBREVIATION = re.compile(r"(?:\b[A-Za-z]|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Inc|Corp|Ltd|Co|Gov|Sen|Rep|Gen|Lt|Col|"
                           r"Capt|Prof|No|vs|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec))\.$")


def sentences(text: str) -> list[str]:
    """The text's sentences, in order, stripped; each exactly as it appears in the text."""
    pieces, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        candidate = text[start:match.start()]
        if "\n" not in match.group() and _ABBREVIATION.search(candidate.rstrip(_CLOSERS)):
            continue
        pieces.append(candidate)
        start = match.end()
    pieces.append(text[start:])
    return [p.strip() for p in pieces if p.strip()]

# GNews answers 400 "syntax error" for "?", "!" and "-" outside quotes (A2 §4), so "what is happening
# with the stock market?" failed. Decided 2026-10-31: remove them and search, never refuse.
_SEARCH_DROP = re.compile(r"[?!¿¡]")
_SEARCH_SPACE = re.compile(r"-")


def search_query(text: str) -> str:
    """The query as it is searched, cached and checked: normalized, "?" and "!" removed, "-" a space."""
    cleaned = _SEARCH_SPACE.sub(" ", _SEARCH_DROP.sub("", normalize(text).text))
    return " ".join(cleaned.split())


# GNews's OR operator joins searches: '"stock market" OR "write me a love poem"' found 10 real
# articles for the first half and sent the whole query to the model (A1 §2.2). Clef scores the
# whole 0.395, in the flag band, because half of it is news (C2 §4). So the parts of an OR
# query are checked one by one as well. This reads the search syntax, as the "?" "!" "-" rule
# does; it doesn't look at the words. The query itself is searched unchanged.
_GROUPING = ' "()'


def or_parts(query: str) -> list[str]:
    """The searches an OR query joins, without the quotes and brackets that group them; [] if no OR.
    Only GNews's operator counts: an uppercase OR between terms, outside quotes."""
    parts, current, in_quote = [], [], False
    for token in query.split(" "):
        if token == "OR" and not in_quote:
            parts.append(" ".join(current))
            current = []
            continue
        current.append(token)
        if token.count('"') % 2:
            in_quote = not in_quote
    parts.append(" ".join(current))
    parts = [p.strip(_GROUPING) for p in parts]
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else []


# --- Shape (#23) ---
# Thresholds measured in C1 §5 on 239 real and written news queries (none scrambled) against
# leetspeak, base64, hex, spaced-out and mixed-alphabet versions of 254 non-news queries.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX_RUN = re.compile(r"\b[0-9a-fA-F]{24,}\b")
# Digits that stand in for letters in leetspeak: 0 o, 1 i/l, 3 e, 4 a, 5 s, 7 t. A token whose
# digits are all of these and that switches between letters and digits at least twice reads as
# letters written with digits ("pr3v10u5"), unlike a model or product name ("A320neo", "Y2K38",
# "RTX5090", "B2B"), which use other digits or switch once.
_LEET_DIGITS = set("013457")
# Leetspeak: at least 2 such words and 30% of the words, or 1 that is half of a short query
LEET_MIN_TOKENS = 2
LEET_MIN_SHARE = 0.3
LEET_ONE_TOKEN_SHARE = 0.5
SPACED_MIN_TOKENS = 6
SPACED_MIN_SHARE = 0.6


@dataclass
class Shape:
    leet_tokens: int = 0
    word_tokens: int = 0
    base64_runs: int = 0
    hex_runs: int = 0
    single_letters: int = 0
    mixed_scripts: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def scrambled(self) -> bool:
        return bool(self.reasons)

    @property
    def leet_share(self) -> float:
        return self.leet_tokens / self.word_tokens if self.word_tokens else 0.0


def _tokens(text: str) -> list[str]:
    """Runs of letters, digits and combining marks. Marks count as part of a word: Devanagari vowel
    signs are marks, and splitting at them made Hindi words look like letters spaced out (C1 §5.1)."""
    tokens, current = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in "LNM":
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _switches(token: str) -> int:
    kinds = ["d" if c.isdigit() else "a" for c in token if c.isalnum()]
    return sum(1 for a, b in zip(kinds, kinds[1:]) if a != b)


def _is_leet(token: str) -> bool:
    digits = [c for c in token if c.isdigit()]
    letters = [c for c in token if c.isalpha()]
    return (len(token) >= 4 and digits and letters and set(digits) <= _LEET_DIGITS and _switches(token) >= 2)


def _script(ch: str) -> str:
    """The alphabet a letter belongs to, from its Unicode name ("LATIN SMALL LETTER A" -> LATIN)."""
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "?"


def _looks_base64(run: str) -> bool:
    """A long run with upper, lower and digits that decodes: not an English word or a URL slug."""
    if not (re.search(r"[A-Z]", run) and re.search(r"[a-z]", run) and re.search(r"[0-9+/]", run)):
        return False
    body = run.rstrip("=")
    try:
        base64.b64decode(body + "=" * (-len(body) % 4), validate=True)
        return True
    except ValueError:
        return False


def shape(text: str) -> Shape:
    s = Shape()
    tokens = _tokens(text)
    words = [t for t in tokens if len(t) >= 3 and any(c.isalpha() for c in t)]
    s.word_tokens = len(words)
    s.leet_tokens = sum(1 for t in words if _is_leet(t))
    s.base64_runs = sum(1 for run in _BASE64_RUN.findall(text) if _looks_base64(run))
    s.hex_runs = len(_HEX_RUN.findall(text))
    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t) and not any(c.isdigit() for c in t)]
    s.single_letters = sum(1 for t in alpha_tokens if len(t) == 1)
    # Latin mixed with another alphabet inside one word: "раypal" with a Cyrillic "р" and "а"
    # (homoglyphs). Other mixes are normal writing (Japanese kanji + kana), so they don't count.
    s.mixed_scripts = sum(1 for t in alpha_tokens
                          if "LATIN" in (scripts := {_script(c) for c in t if c.isalpha()}) and len(scripts) > 1)
    if ((s.leet_tokens >= LEET_MIN_TOKENS and s.leet_share >= LEET_MIN_SHARE)
            or (s.leet_tokens >= 1 and s.leet_share >= LEET_ONE_TOKEN_SHARE)):
        s.reasons.append("leetspeak")
    if s.base64_runs:
        s.reasons.append("base64")
    if s.hex_runs:
        s.reasons.append("hex")
    if len(alpha_tokens) >= SPACED_MIN_TOKENS and s.single_letters / len(alpha_tokens) >= SPACED_MIN_SHARE:
        s.reasons.append("spaced_letters")
    if s.mixed_scripts:
        s.reasons.append("mixed_scripts")
    return s
