import hashlib
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit

from services.safety import settings

# --- Where each article came from (feature 3, Section G #50-#54) ---
# A page that plants false news, redirects to another site or carries a fake date reaches every user
# who asks about its topic (the article lists and pages are shared caches). These are the facts about
# a page that don't depend on its words: the site it was read from, when, which version of its text,
# and the date it gives itself. Decided by URLs, hosts and dates, never by what the text says.
#
#   #50 provenance   every article (and so every chunk) carries its URL, final URL after redirects,
#                    site, fetch time and a hash of the text it was read as
#   #51 redirect     a page that ends on another site than the one GNews listed is recorded, and its
#                    source line names the site the text really came from
#   #53 date         a GNews date in the future is replaced (the page's own date, or "unknown date");
#                    a page date far from GNews's is recorded, and GNews's date is kept
#   #54 blocklist    SAFETY_BLOCKED_DOMAINS (settings): sites the operator never wants used

# A GNews date this far ahead of now is impossible: the free plan's newest article is ~12 h old
FUTURE_SLACK = timedelta(hours=1)
# Measured on 72 pages (G §3): the page's own date and GNews's matched on 63, were 1 day apart on 8
# (time zones: the page's date is local, GNews's is UTC), and 13 days apart on 1 (a timeline page that
# kept being updated). Past 2 days the dates disagree about the story, not the time zone.
DATE_TOLERANCE_DAYS = 2


def site(url) -> str:
    """'https://www.Reuters.com/x' -> 'reuters.com'; '' for anything without a host."""
    try:
        host = urlsplit(str(url or "")).hostname or ""
    except ValueError:
        return ""
    return host.lower().rstrip(".").removeprefix("www.")


def same_site(a: str, b: str) -> bool:
    """One host is the other or a subdomain of it: 'edition.cnn.com' and 'cnn.com' are the same site,
    'usatoday.com' and 'tallahassee.com' are not (even in one newspaper group)."""
    return bool(a and b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def blocked_by(url) -> str | None:
    """The SAFETY_BLOCKED_DOMAINS entry `url`'s host falls under (itself or a subdomain), or None."""
    host = site(url)
    if not host:
        return None
    for domain in settings.blocked_domains():
        if host == domain or host.endswith("." + domain):
            return domain
    return None


def text_hash(text: str) -> str:
    """The page version: the first 16 hex digits of the sha256 of the text the app read."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def with_page(article: dict, record: dict) -> dict:
    """A copy of `article` with the page's text and its provenance (#50, #51). `record` is a cache
    record (cache_service.page_record); a page cached before G has no final URL or fetch time."""
    listed = article.get("url") or ""
    final = record.get("final_url") or listed
    listed_site, final_site = site(listed), site(final)
    return {**article, "full_text": record["text"], "final_url": final, "site": final_site, "listed_site": listed_site,
            "redirected": bool(listed_site and final_site) and not same_site(listed_site, final_site),
            "fetched_at": record.get("fetched_at"), "page_hash": text_hash(record["text"]),
            "page_date": record.get("page_date")}


def _gnews_time(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _day(value) -> date | None:
    try:
        return date.fromisoformat(value) if isinstance(value, str) and len(value) == 10 else None
    except ValueError:
        return None


def check_date(article: dict, now: datetime | None = None) -> str | None:
    """
    The date check (#53), in place. Returns the note it recorded on the article ("date_note"), or None.
      - GNews's date in the future: replaced by the page's own date when it has one (htmldate never
        gives a date after today), else removed (the source line then says "unknown date").
      - The page's date more than DATE_TOLERANCE_DAYS from GNews's: recorded, GNews's date kept. GNews
        saw the article when it came out; the page's metadata is whatever the page says now.
    An article that already carries a note (from the build that cached it) keeps it.
    """
    now = now or datetime.now(timezone.utc)
    gnews = _gnews_time(article.get("published_at"))
    page = _day(article.get("page_date"))
    note = None
    if gnews is not None and gnews > now + FUTURE_SLACK:
        usable = page is not None and page <= now.date()
        article["gnews_published_at"] = article.get("published_at")
        article["published_at"] = page.isoformat() if usable else None
        note = (f"GNews date {article['gnews_published_at']} is in the future; "
                + (f"using the page's date {page.isoformat()}" if usable else "no usable date"))
    elif gnews is not None and page is not None and abs((page - gnews.date()).days) > DATE_TOLERANCE_DAYS:
        days = (page - gnews.date()).days
        note = (f"page date {page.isoformat()} is {abs(days)} days {'after' if days > 0 else 'before'} "
                f"GNews's {gnews.date().isoformat()}; GNews's date kept")
    if note:
        article["date_note"] = note
    return note


def site_line(metadata: dict) -> str:
    """The Site field of a source line: the site the text was read from, and the one GNews listed
    when a redirect changed it (#51). Falls back to the URL's site for a chunk without provenance."""
    final = metadata.get("site") or site(metadata.get("final_url") or metadata.get("source") or metadata.get("url"))
    if not final:
        return "unknown site"
    listed = metadata.get("listed_site")
    if metadata.get("redirected") and listed:
        return f"{final} (listed as {listed})"
    return final


def reports_line(value) -> str:
    """The 'Reports on this story' field: a count, or 'unknown' (grouping failed, or an older cache entry)."""
    return str(value) if isinstance(value, int) and value > 0 else "unknown"
