from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlsplit

import markdown2
import nh3

from services import checks
from services.safety import settings
from services.safety.engine import Decision
from services.safety.guard import check

# --- What the model writes, on its way into the page (feature 3, Section I) ---
# Model output becomes HTML in the browser, and the answer is written from scraped pages, so a page
# can steer what it says. Three layers, outermost last:
#   1. markdown2 safe_mode="escape": raw HTML in the answer is shown as text, javascript: links become "#"
#   2. nh3 allowlist (#69, always on): only the tags markdown makes survive, with only href/title on links
#   3. the `output` guard (SAFETY_OUTPUT): links to pages that aren't this answer's sources (#67) and
#      images (#68) are recorded; enforce mode turns such a link into plain text and removes the image
# The page's Content-Security-Policy (server.py, #70) stops outside images loading in every mode.
# The guard also flags (never blocks) an answer Gemini cut at its length cap (#66) and a timeline whose
# format is broken (#72). Nothing here reads what the answer says: tags, URLs and formats only.

# Everything markdown2 makes from plain markdown (no extras are enabled): headings, paragraphs, lists,
# emphasis, code, quotes, rules, links and images. Anything else can't come from markdown.
TAGS = {"p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "strong", "em", "b", "i",
        "code", "pre", "blockquote", "a"}
IMAGE_TAGS = {"img"}
ATTRIBUTES = {"a": {"href", "title"}, "img": {"src", "alt", "title"}}
URL_SCHEMES = {"http", "https"}
LINK_REL = "noopener noreferrer nofollow"

CUT_NOTE = "This answer reached its length limit, so it may stop mid-sentence."

# The timeline's format checks (feature 2's checks.check_timeline) that make a safety flag (#72).
# outlets_in_sources is J's outlet check (#78); events after the newest source are allowed (an article
# can announce Monday's opening on Friday) and only listed in the label.
TIMELINE_FLAGS = ("has_events", "all_lines_formatted", "newest_first", "no_future_date")


def normalize_url(url) -> str | None:
    """'https://www.Example.com/a/?x=1#top' -> 'example.com/a?x=1'. Scheme, 'www.', a trailing slash and
    the fragment don't change which page it is. None for anything that isn't an http(s) URL."""
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in URL_SCHEMES or not parts.hostname:
        return None
    host = parts.hostname.lower().removeprefix("www.")
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    path = parts.path.rstrip("/")
    return f"{host}{port}{path}" + (f"?{parts.query}" if parts.query else "")


def allowed_urls(urls) -> frozenset[str]:
    """The pages this answer may link to: its sources (the URL GNews listed and the one read after
    redirects), normalised."""
    return frozenset(n for n in (normalize_url(u) for u in urls or ()) if n)


@dataclass
class Rendered:
    html: str
    outside_links: list[str] = field(default_factory=list)   # links to pages that aren't sources
    images: int = 0
    removed: bool = False                                      # enforce mode removed them


def render(text, allowed=frozenset(), mode: str | None = None) -> Rendered:
    """
    Markdown -> the HTML the page shows. `allowed`: allowed_urls() of this answer's sources.
    In enforce mode an outside link keeps its text but loses its href (an <a> without href is plain
    text, not a link) and images are removed; otherwise they're only counted. Never raises.
    """
    mode = mode or settings.mode("output")
    html = markdown2.markdown(text or "", safe_mode="escape")
    found = Rendered(html="")

    def screen(tag, attribute, value):
        if tag == "a" and attribute == "href":
            if normalize_url(value) not in allowed:
                found.outside_links.append(value)
                return None if mode == "enforce" else value
        elif tag == "img" and attribute == "src":
            found.images += 1
        return value

    # One pass sees everything (images included); in enforce mode it already drops outside hrefs
    cleaned = nh3.clean(html, tags=TAGS | IMAGE_TAGS, attributes=ATTRIBUTES, attribute_filter=screen,
                        url_schemes=URL_SCHEMES, link_rel=LINK_REL, set_tag_attribute_values={"a": {"target": "_blank"}})
    if mode == "enforce" and found.images:
        cleaned = nh3.clean(cleaned, tags=TAGS, attributes=ATTRIBUTES, url_schemes=URL_SCHEMES, link_rel=LINK_REL,
                            set_tag_attribute_values={"a": {"target": "_blank"}})
    found.html = cleaned
    found.removed = mode == "enforce" and bool(found.outside_links or found.images)
    return found


def timeline_problems(text: str, sources, today: date | None = None) -> list[str]:
    """The failed format checks of a timeline (#72): no dated lines, unformatted lines, wrong order,
    dates in the future."""
    result = checks.check_timeline(text, sources, today)
    return [name for name in TIMELINE_FLAGS if not result["checks"].get(name, True)]


def _label(rendered: list[Rendered], cut: bool, problems: list[str], extra: list[str]) -> str | None:
    parts = []
    links = sum(len(r.outside_links) for r in rendered)
    images = sum(r.images for r in rendered)
    if links:
        parts.append(f"outside_links={links}")
    if images:
        parts.append(f"images={images}")
    if cut:
        parts.append("cut")
    parts += [f"timeline:{p}" for p in problems] + extra
    return " ".join(parts) or None


async def record(where: str, rendered: list[Rendered], *, text: str | None = None, cut: bool = False,
                 problems: list[str] = (), extra: list[str] = ()) -> Decision | None:
    """
    One `output` event for one answer (or one Follow check's texts): block when it had outside links
    or images (enforced = enforce mode removed them), flag when it was cut, its timeline format is
    broken or `extra` lists something else (the fact check's rejected entries), allow otherwise.
    """
    found = any(r.outside_links or r.images for r in rendered)
    flagged = cut or bool(problems) or bool(extra)

    async def run():
        verdict = "block" if found else "flag" if flagged else "allow"
        return Decision(verdict=verdict, score=1.0 if verdict != "allow" else 0.0, threshold=1.0, engine="code",
                        label=_label(rendered, cut, list(problems), list(extra)))
    return await check("output", run, where=where, text=text)


async def answer(view: str, text: str, source_urls, *, sources=None, finish_reason: str | None = None) -> dict:
    """
    One view's answer for the page: {"html", "notes"}. notes are short lines shown under the answer
    (a cut answer says so). sources: the answer's sources (dates) for the timeline check.
    """
    mode = settings.mode("output")
    rendered = render(text, allowed_urls(source_urls), mode)
    if mode == "off":
        return {"html": rendered.html, "notes": []}
    cut = str(finish_reason or "").upper() == "MAX_TOKENS"
    problems = timeline_problems(text, sources) if view == "timeline" else []
    await record(view, [rendered], text=text, cut=cut, problems=problems)
    if cut:
        print(f"[WARN] {view}: Gemini stopped at the length cap (MAX_TOKENS); shown with a note.")
    return {"html": rendered.html, "notes": [CUT_NOTE] if cut else []}
