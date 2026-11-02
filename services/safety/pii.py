import ipaddress
import re
from dataclasses import dataclass

# --- Personal data by format (feature 3, Section H #57) ---
# Emails, phone numbers, card numbers, Aadhaar, PAN, IBAN and IP addresses have a fixed structure,
# and most carry a checksum: Luhn for cards, Verhoeff for Aadhaar, mod-97 for IBAN. A value is
# reported only when its structure AND its checksum hold, so a 12-digit price isn't taken for an
# Aadhaar number (9 in 10 random numbers fail Verhoeff) and a 16-digit article id isn't a card.
# This reads what the text LOOKS like, never which words it uses (rule 2): paraphrasing can't
# change what an email address looks like. The tables below (IBAN lengths, toll-free codes, card
# prefixes) are numbering-plan data, like the checksums, not lists of words.
#
# No dependency: everything here is the standard library, so it can run on every string a trace
# carries (observability's scrubber) without loading anything.

KINDS = ("email", "phone", "card", "aadhaar", "pan", "iban", "ip")
# What the user is told was removed ("We removed an email address ...")
NAMES = {"email": "an email address", "phone": "a phone number", "card": "a card number", "aadhaar": "an Aadhaar number",
         "pan": "a PAN", "iban": "an IBAN", "ip": "an IP address"}
# A kind's label in masked text: "[email]" in traces and logs, "[email removed]" in answers
LABELS = {"email": "email", "phone": "phone number", "card": "card number", "aadhaar": "Aadhaar number",
          "pan": "PAN", "iban": "IBAN", "ip": "IP address"}
# Where two matches overlap (a card number inside an IBAN, digits of an email's local part), the
# kind first in this order wins
_PRIORITY = {kind: i for i, kind in enumerate(("email", "iban", "card", "aadhaar", "phone", "pan", "ip"))}


@dataclass(frozen=True)
class Found:
    kind: str
    start: int
    end: int


# ---------------------------------------------------------------- checksums

def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


_VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
               [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
               [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
               [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
               [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
               [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(digits: str) -> bool:
    check = 0
    for i, ch in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[i % 8][int(ch)]]
    return check == 0


# IBAN length by country (the SWIFT IBAN registry). A country code missing here isn't an IBAN.
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22, "BH": 22, "BR": 29, "BY": 28,
    "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18,
    "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22,
    "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32, "LI": 21, "LT": 20,
    "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15,
    "PK": 24, "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SC": 31, "SE": 24, "SI": 19,
    "SK": 24, "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}


def iban_ok(compact: str) -> bool:
    if IBAN_LENGTHS.get(compact[:2]) != len(compact) or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


# ---------------------------------------------------------------- detectors
# Each yields (start, end). A match must not start or end inside a longer token: digits inside a
# URL slug ("/article-4111111111111111"), a decimal, an id. The end is checked by a look-ahead in
# the pattern (so the engine can still back off to a shorter match); the start is checked in code
# (_free_before), because a pattern that opens with a look-behind can't use the regex engine's
# skip-ahead on its first character: every pattern here starts with a plain character class or
# literal. That made scanning 180 KB of article text (one large trace) ~6x faster (H §6).

_WORD_EXTRA = "_"


def _free_before(text: str, start: int, also: str = "") -> bool:
    """The character before `start` is neither a letter or digit (or "_") nor one of `also`."""
    if start == 0:
        return True
    ch = text[start - 1]
    return not (ch.isalnum() or ch in _WORD_EXTRA or ch in also)


def _search(pattern, text, also: str = ""):
    """pattern's matches whose start is free (see _free_before); a refused start is retried one character on."""
    pos = 0
    while (m := pattern.search(text, pos)) is not None:
        if _free_before(text, m.start(), also):
            yield m
            pos = m.end()
        else:
            pos = m.start() + 1


# Emails: found from each "@" (rare in text), not by scanning every letter
_LOCAL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-")
_DOMAIN = re.compile(r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![\w-]|\.[A-Za-z0-9])")


def _emails(text):
    at = text.find("@")
    while at != -1:
        start = at
        while start > 0 and text[start - 1] in _LOCAL_CHARS and at - start < 64:
            start -= 1
        local = text[start:at]
        domain = _DOMAIN.match(text, at + 1)
        # "icon@2x.png" isn't an address: the domain before its last label needs 2 letters or more
        if (local and domain and local[0].isalnum() and local[-1] != "." and ".." not in local
                and _free_before(text, start, ".%+-")
                and sum(ch.isalpha() for ch in domain.group().rsplit(".", 1)[0]) >= 2):
            yield start, domain.end()
        at = text.find("@", at + 1)


# Toll-free numbers belong to organisations (helplines), and helplines are news worth keeping
_NANP_TOLL_FREE = {"800", "833", "844", "855", "866", "877", "888"}
_IN_TOLL_FREE = ("1800", "1860")
# A run of digits with this many zeros is a figure ("7000000000"), not a phone number
_MAX_ZEROS = 6
_CURRENCY = set("₹$€£¥")
# Characters that, right before a number, make it part of something else: a decimal, an amount, a URL
_JOINED = ".,+/=#-"


def _after_currency(text: str, start: int) -> bool:
    before = text[max(0, start - 2):start].rstrip()
    return bool(before) and before[-1] in _CURRENCY


def _nanp_ok(national: str) -> bool:
    return len(national) == 10 and national[0] in "23456789" and national[3] in "23456789" and national[:3] not in _NANP_TOLL_FREE


def _intl_ok(digits: str) -> bool:
    """A number written with "+": E.164 (8-15 digits), with India's and North America's plans checked."""
    if not 8 <= len(digits) <= 15 or digits[0] == "0" or digits[1:].count("0") >= _MAX_ZEROS:
        return False
    if digits.startswith("91"):
        national = digits[2:]
        return len(national) == 10 and national[0] != "0" and not national.startswith(_IN_TOLL_FREE)
    if digits.startswith("1"):
        return _nanp_ok(digits[1:])
    return True


def _prefix_start(text: str, start: int, prefixes) -> int:
    """Where a number really starts when one of `prefixes` ("0", "91", "1") is written before it,
    with or without one separator ("+91 " is the "+" rule's, and fails _free_before below)."""
    for sep in ("", " ", "-", "."):
        for prefix in prefixes:
            begin = start - len(sep) - len(prefix)
            if begin >= 0 and text[begin:start] == prefix + sep:
                return begin
    return start


# "+" and a run of digits and separators; the run is cut back group by group until it is a valid
# number, so "+91 98765 43210 (2026)" finds the phone number and leaves the year
_PLUS_RUN = re.compile(r"\+\d[\d ().-]{5,24}")
_DIGIT_GROUP = re.compile(r"\d+")
# An Indian mobile written without "+": 10 digits from 6-9, as 5-5 or 3-3-4 or whole, maybe after 0 or 91
_IN_MOBILE = re.compile(r"[6-9](?:\d{4}[ -]?\d{5}|\d{2}[ -]\d{3}[ -]\d{4})(?![\w]|[.,/-]\d)")
# A North American number written with separators: 212-555-1234, 212.555.1234, (212) 555-1234
_NANP_DASHED = re.compile(r"(\d{3})([-.])(\d{3})\2(\d{4})(?![\w]|[.,/-]\d)")
_NANP_BRACKETS = re.compile(r"\((\d{3})\) ?(\d{3})[-.](\d{4})(?![\w]|[.,/-]\d)")


def _phones(text):
    for m in _search(_PLUS_RUN, text, "+"):
        run = m.group()
        groups = [(g.start(), g.end()) for g in _DIGIT_GROUP.finditer(run)]
        for i in range(1, len(groups)):            # a separator longer than ") " ends the number
            if groups[i][0] - groups[i - 1][1] > 2:
                groups = groups[:i]
                break
        for n in range(len(groups), 0, -1):
            if _intl_ok("".join(run[s:e] for s, e in groups[:n])):
                yield m.start(), m.start() + groups[n - 1][1]
                break
    pos = 0
    while (m := _IN_MOBILE.search(text, pos)) is not None:
        start = _prefix_start(text, m.start(), ("91", "0"))
        if _free_before(text, start, _JOINED):
            digits = re.sub(r"\D", "", m.group())
            if digits.count("0") < _MAX_ZEROS and not _after_currency(text, start):
                yield start, m.end()
            pos = m.end()
        else:
            pos = m.start() + 1
    for pattern in (_NANP_DASHED, _NANP_BRACKETS):
        pos = 0
        while (m := pattern.search(text, pos)) is not None:
            start = _prefix_start(text, m.start(), ("1",))
            if not _free_before(text, start, _JOINED):
                pos = m.start() + 1
                continue
            national = "".join(g for g in m.groups() if g and g.isdigit())
            if _nanp_ok(national) and national.count("0") < _MAX_ZEROS and not _after_currency(text, start):
                yield start, m.end()
            pos = m.end()


_CARD = re.compile(r"\d(?:[ -]?\d){12,18}(?![\w]|[.,/-]\d)")
# Card numbers start with 2 (Mastercard 2-series), 3 (Amex, Diners, JCB), 4 (Visa), 5 (Mastercard,
# Maestro), 6 (Discover, RuPay, Maestro) or 8 (RuPay)
_CARD_FIRST = set("234568")


def _card_groups_ok(candidate: str) -> bool:
    """Unseparated, or grouped as cards are printed: 4-4-4-4(-3), 4-6-5 (Amex), 4-6-4 (Diners), one separator."""
    separators = set(re.findall(r"[ -]", candidate))
    if not separators:
        return True
    if len(separators) > 1:
        return False
    sizes = [len(g) for g in re.split(r"[ -]", candidate)]
    return sizes in ([4, 6, 5], [4, 6, 4]) or (all(s == 4 for s in sizes[:-1]) and 1 <= sizes[-1] <= 4)


def _cards(text):
    for m in _search(_CARD, text, _JOINED):
        digits = re.sub(r"\D", "", m.group())
        if (13 <= len(digits) <= 19 and digits[0] in _CARD_FIRST and _card_groups_ok(m.group()) and luhn_ok(digits)
                and not _after_currency(text, m.start())):
            yield m.start(), m.end()


_AADHAAR = re.compile(r"[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\w]|[.,/-]\d)")


def _timestamp_like(digits: str) -> bool:
    """YYYYMMDDhhmm: an unseparated 12-digit time stamp (in URLs and ids) isn't an Aadhaar number."""
    year, month, day, hour, minute = (int(digits[0:4]), int(digits[4:6]), int(digits[6:8]), int(digits[8:10]),
                                      int(digits[10:12]))
    return 1900 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31 and hour < 24 and minute < 60


def _aadhaar(text):
    for m in _search(_AADHAAR, text, _JOINED):
        digits = re.sub(r"\D", "", m.group())
        if not verhoeff_ok(digits) or (not m.group(1) and _timestamp_like(digits)) or _after_currency(text, m.start()):
            continue
        yield m.start(), m.end()


# PAN: 5 letters, 4 digits, 1 letter; the 4th letter is the holder's type (P person, C company, ...)
_PAN = re.compile(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z](?![\w])")


def _pans(text):
    for m in _search(_PAN, text):
        yield m.start(), m.end()


_IBAN = re.compile(r"[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}")


def _ibans(text):
    for m in _search(_IBAN, text):
        length = IBAN_LENGTHS.get(m.group()[:2])
        if not length:
            continue
        # The run may go on into the next word ("... 32 AND"): take exactly the country's length
        compact, end = [], m.start()
        for i, ch in enumerate(m.group()):
            if ch != " ":
                compact.append(ch)
                if len(compact) == length:
                    end = m.start() + i + 1
                    break
        if len(compact) != length or (end < len(text) and text[end].isalnum()):
            continue
        if iban_ok("".join(compact)):
            yield m.start(), end


_IPV4 = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\w]|\.\d)")
_HEX_OR_COLON = set("0123456789abcdefABCDEF:")


def _ipv6_spans(text):
    """Runs of hex digits and colons with 3 colons or more, found from the colons (like emails from "@")."""
    colon = text.find(":")
    while colon != -1:
        start = end = colon
        while start > 0 and text[start - 1] in _HEX_OR_COLON:
            start -= 1
        while end < len(text) and text[end] in _HEX_OR_COLON:
            end += 1
        if text.count(":", start, end) >= 3 and end - start <= 39 and _free_before(text, start, ":.") and \
                (end == len(text) or not (text[end].isalnum() or text[end] in "_:")):
            yield start, end
        colon = text.find(":", end)


def _ips(text):
    candidates = [(m.start(), m.end()) for m in _search(_IPV4, text, ".")]
    if text.count(":") >= 3:
        candidates += list(_ipv6_spans(text))
    for start, end in candidates:
        try:
            address = ipaddress.ip_address(text[start:end])
        except ValueError:
            continue
        # Loopback, "0.0.0.0" and multicast addresses don't point at anyone
        if not (address.is_loopback or address.is_unspecified or address.is_multicast):
            yield start, end


_DETECTORS = {"email": _emails, "phone": _phones, "card": _cards, "aadhaar": _aadhaar, "pan": _pans, "iban": _ibans,
              "ip": _ips}


# ---------------------------------------------------------------- the API

def find(text: str, kinds=KINDS) -> list[Found]:
    """Every piece of personal data of `kinds` in `text`, in order, without overlaps."""
    if not text or ("@" not in text and not any(ch.isdigit() for ch in text)):
        return []                    # every kind has a digit or an "@": most strings end here
    found = [Found(kind, s, e) for kind in kinds for s, e in _DETECTORS[kind](text)]
    kept: list[Found] = []
    for f in sorted(found, key=lambda f: (_PRIORITY[f.kind], f.start)):
        if all(f.end <= k.start or f.start >= k.end for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)


def replace(text: str, found: list[Found], replacement) -> str:
    """`text` with each found span replaced by replacement(kind)."""
    out, i = [], 0
    for f in found:
        out.append(text[i:f.start])
        out.append(replacement(f.kind))
        i = f.end
    out.append(text[i:])
    return "".join(out)


def tag(kind: str) -> str:
    """'[email]': what traces, logs, Clef and LangSmith see in place of the value."""
    return f"[{kind}]"


def removed_tag(kind: str) -> str:
    """'[phone number removed]': what a reader sees in an answer."""
    return f"[{LABELS[kind]} removed]"


def mask(text: str, kinds=KINDS) -> str:
    """`text` with personal data tagged ("[email]"); the text itself when there's none."""
    found = find(text, kinds)
    return replace(text, found, tag) if found else text


def remove(text: str, kinds=KINDS) -> tuple[str, list[Found]]:
    """`text` without its personal data (a query: nothing is left to be searched), spaces collapsed."""
    found = find(text, kinds)
    if not found:
        return text, []
    return " ".join(replace(text, found, lambda kind: " ").split()), found


def kinds_of(found: list[Found]) -> list[str]:
    """The kinds found, each once, in KINDS order."""
    present = {f.kind for f in found}
    return [kind for kind in KINDS if kind in present]


def describe(kinds) -> str:
    """'an email address', 'an email address and a phone number', 'a, b and c'."""
    names = [NAMES[kind] for kind in KINDS if kind in set(kinds)]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
