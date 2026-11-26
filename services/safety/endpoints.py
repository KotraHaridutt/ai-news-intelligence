from urllib.parse import urlsplit

from services import errors
from services.safety import events, limits, settings

# --- Endpoint checks (feature 3, Section M #100) ---
# Every endpoint that costs something or changes something is a POST or DELETE: the three views
# (a chat call each), Follow, a Follow check, unfollow and ratings. Another website can make a
# visitor's browser send such a request (cross-site request forgery): a hidden form, or
# navigator.sendBeacon, which sends a POST with no Content-Type at all. FastAPI reads a body with no
# Content-Type as JSON, so today such a request would be accepted.
#
#   1. JSON only, always on: a request with a body must say application/json (or a +json type), else
#      415. The page always sends application/json. A browser sends that type from another site only
#      after asking first (a CORS preflight), and the app answers no preflight: no CORS headers are
#      sent anywhere (the M tests keep it so), so the browser never sends the request.
#   2. Same origin, SAFETY_ORIGIN (monitor by default, 403 in enforce): which site sent the request,
#      as the browser reports it. Sec-Fetch-Site (all current browsers) must be same-origin, or none
#      (typed or bookmarked); without it, an Origin header must name this host. A request with
#      neither is a script or an old browser, not a page on another site, and is let through:
#      scripts are what Section L's limits are for.
# GET, HEAD and OPTIONS change nothing and aren't checked (reading a followed topic's log needs its
# unguessable watch id).

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
SAME_SITE_VALUES = frozenset({"same-origin", "none"})
_DEFAULT_PORTS = {"http": "80", "https": "443"}


def _header(scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1").strip()
    return None


def has_body(scope) -> bool:
    length = _header(scope, b"content-length")
    if length is not None:
        return not (length.isdigit() and int(length) == 0)
    return _header(scope, b"transfer-encoding") is not None


def is_json(content_type: str | None) -> bool:
    """'application/json; charset=utf-8' and 'application/merge-patch+json' -> True."""
    media = (content_type or "").split(";", 1)[0].strip().lower()
    return media == "application/json" or (media.startswith("application/") and media.endswith("+json"))


def _host_port(netloc: str, scheme: str | None = None) -> str:
    """'Example.com:443' -> 'example.com'; '[::1]:8000' -> '[::1]:8000'. A default port is dropped."""
    netloc = netloc.strip().lower()
    host, port = netloc, ""
    if netloc.startswith("["):
        end = netloc.find("]")
        host, rest = netloc[:end + 1], netloc[end + 1:]
        port = rest[1:] if rest.startswith(":") else ""
    elif netloc.count(":") == 1:
        host, port = netloc.split(":", 1)
    if port in ("80", "443") and (scheme is None or _DEFAULT_PORTS.get(scheme) == port):
        port = ""
    return f"{host}:{port}" if port else host


def cross_site_reason(scope) -> str | None:
    """Why the browser says this request came from another site, or None (same origin, or no browser
    signal). 'Sec-Fetch-Site cross-site', 'Origin https://evil.example', 'Origin null'."""
    fetch_site = _header(scope, b"sec-fetch-site")
    if fetch_site:
        return None if fetch_site.lower() in SAME_SITE_VALUES else f"Sec-Fetch-Site {fetch_site[:20]}"
    origin = _header(scope, b"origin")
    if not origin:
        return None
    if origin.lower() == "null":                 # a sandboxed frame or a local file
        return "Origin null"
    parts = urlsplit(origin)
    if parts.scheme not in _DEFAULT_PORTS or not parts.netloc:
        return f"Origin {origin[:80]}"
    sent_by = _host_port(parts.netloc, parts.scheme)
    ours = {_host_port(value) for value in (_header(scope, b"host"), _header(scope, b"x-forwarded-host")) if value}
    return None if sent_by in ours else f"Origin {origin[:80]}"


def _event(reason: str, path: str, mode: str) -> events.SafetyEvent:
    return events.SafetyEvent(check="origin", verdict="block", mode=mode, where=path[:60], score=None, threshold=None,
                              engine="code", ms=0.0, label=reason, text_hash=limits.current_client(),
                              enforced=mode == "enforce")


class RequestOrigin:
    """
    Pure ASGI middleware, inside ClientLimits (whose context holds the visitor and the events waiting
    for the request's trace): a body that isn't JSON -> 415; a request from another site -> flagged
    (monitor) or 403 (enforce). Both before FastAPI reads the body.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method", "GET") in SAFE_METHODS:
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if has_body(scope) and not is_json(_header(scope, b"content-type")):
            content_type = (_header(scope, b"content-type") or "none")[:60]
            return await limits.send_error(errors.not_json(f"Content-Type {content_type}"), scope, receive, send)
        mode = settings.mode("origin")
        if mode != "off":
            reason = cross_site_reason(scope)
            if reason:
                event = _event(reason, path, mode)
                if mode == "enforce":
                    events.record(event)           # no trace yet: the log line
                    return await limits.send_error(errors.cross_site_request(f"{reason}, client {limits.current_client()}"),
                                                   scope, receive, send)
                limits.note_event(event)           # served; the event goes on the request's trace
        await self.app(scope, receive, send)


def status() -> str:
    """For the startup line."""
    return f"same-origin check {settings.mode('origin')}, JSON bodies only"
