import asyncio
import ipaddress
import socket

import httpcore
import httpx

# --- SSRF guard for the scraper (feature 3, Section M #98) ---
# The scraper opens whatever URL GNews lists, and follows the page's redirects. A listed page (or a
# redirect from it) could point at the server's own network instead of the internet: 127.0.0.1, a
# private 10.x address, or the cloud metadata service at 169.254.169.254, which hands out credentials
# on many hosts. Fetching it would turn the app into a proxy into its own network.
#
# Two checks, both always on (no news page lives on a private address):
#   1. every request, each redirect hop included (an httpx request hook): http or https, a host,
#      no user:password@ part, and a host written as an IP must be a public address
#   2. every connection (a network backend under httpx): the host name is resolved here, EVERY
#      address it resolves to must be public, and the connection goes to the address that was
#      checked. httpx never resolves the name again, so a DNS answer that changes between the check
#      and the connection (DNS rebinding) can't slip a private address in. TLS still checks the
#      certificate against the host name.
# At most MAX_REDIRECTS redirects per page (httpx's default is 20).

MAX_REDIRECTS = 5
SCHEMES = ("http", "https")

# Cloud metadata endpoints, named in the reason. Each is also caught by the general rule (link-local,
# private or shared address space): the names only make the log line say what was attempted.
METADATA_ADDRESSES = {
    ipaddress.ip_address("169.254.169.254"): "cloud metadata",      # AWS, GCP, Azure, DigitalOcean, ...
    ipaddress.ip_address("fd00:ec2::254"): "cloud metadata",        # AWS over IPv6
    ipaddress.ip_address("100.100.100.200"): "cloud metadata",      # Alibaba Cloud
}
# IPv6 prefixes that carry an IPv4 address inside them: the IPv4 part decides
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class BlockedAddress(Exception):
    """A scrape (or one of its redirects) aimed somewhere the scraper must not go. `reason` is short,
    for the scrape's failure reason and the log line; it never holds the page's text."""

    def __init__(self, reason: str, url: str | None = None):
        super().__init__(reason if url is None else f"{reason}: {url}")
        self.reason = reason
        self.url = url


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address carries: mapped (::ffff:a.b.c.d), 6to4, Teredo or NAT64."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        return ip.teredo[1]                    # the client's address
    if ip in _NAT64:
        return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return None


def address_problem(value) -> str | None:
    """Why the scraper must not connect to this address, or None for a public one.
    'cloud metadata', 'loopback', 'private', 'link-local', 'shared (carrier NAT)', 'multicast',
    'unspecified', 'reserved', 'not public'."""
    ip = ipaddress.ip_address(value) if not isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address)) else value
    if ip in METADATA_ADDRESSES:
        return METADATA_ADDRESSES[ip]
    if ip.version == 6:
        inner = _embedded_ipv4(ip)
        if inner is not None:
            problem = address_problem(inner)
            return f"{problem} (inside IPv6)" if problem else None
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):
        return "shared (carrier NAT)"
    if ip.is_reserved:                         # before private: Python counts 240.0.0.0/4 as both
        return "reserved"
    if ip.is_private:
        return "private"
    if not ip.is_global:
        return "not public"
    return None


def _ip_literal(host: str):
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def check_url(url) -> None:
    """Raises BlockedAddress for a URL the scraper must not request: another scheme, no host, a
    user:password@ part, or a host written as a non-public IP. A host NAME is checked when it's
    resolved (GuardedBackend), because only then is its address known."""
    url = httpx.URL(str(url)) if not isinstance(url, httpx.URL) else url
    # For the log: scheme, host, port and path only (never a password from user:password@, nor the query)
    port = f":{url.port}" if url.port else ""
    shown = f"{url.scheme}://{url.host}{port}{url.path}"[:200] if url.host else f"{url.scheme}:{url.path}"[:200]
    if url.scheme not in SCHEMES:
        raise BlockedAddress(f"scheme {url.scheme or 'missing'!s} not allowed", shown)
    if not url.host:
        raise BlockedAddress("no host", shown)
    if url.userinfo:
        raise BlockedAddress("credentials in the URL", shown)
    ip = _ip_literal(url.host)
    if ip is not None:
        problem = address_problem(ip)
        if problem:
            raise BlockedAddress(f"{problem} address", shown)


async def check_request(request: httpx.Request) -> None:
    """httpx request hook: runs before the first request and before every redirect hop."""
    check_url(request.url)


async def resolve(host: str, port: int) -> list[str]:
    """
    The host's addresses, every one of them public, in the resolver's order. Raises BlockedAddress
    when ANY address isn't public (a name that resolves to both could otherwise be connected to the
    private one), httpcore.ConnectError when the name doesn't resolve (as a normal DNS failure).
    """
    ip = _ip_literal(host)
    if ip is not None:
        problem = address_problem(ip)
        if problem:
            raise BlockedAddress(f"{problem} address", host)
        return [str(ip)]
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        raise httpcore.ConnectError(f"DNS lookup failed for {host}: {e}") from e
    addresses = []
    for family, _, _, _, sockaddr in infos:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        address = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])    # drop an IPv6 zone ("%eth0")
        problem = address_problem(address)
        if problem:
            raise BlockedAddress(f"resolves to a {problem} address", host)
        if str(address) not in addresses:
            addresses.append(str(address))
    if not addresses:
        raise httpcore.ConnectError(f"DNS lookup for {host} returned no address")
    return addresses


class GuardedBackend(httpcore.AsyncNetworkBackend):
    """httpcore's network layer with the address check: connects only to addresses resolve() allowed.
    Every connection httpx opens goes through connect_tcp, redirects to another host included."""

    def __init__(self, inner: httpcore.AsyncNetworkBackend | None = None):
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        last_error = None
        for address in await resolve(host, port):
            try:
                return await self._inner.connect_tcp(address, port, timeout=timeout, local_address=local_address,
                                                     socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                last_error = e                 # the next address, as a browser would
        raise last_error

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise BlockedAddress("unix socket")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def guarded_transport(verify, limits: httpx.Limits) -> httpx.AsyncHTTPTransport:
    """
    httpx's own transport with its connection pool rebuilt on GuardedBackend. httpx 0.28 doesn't take a
    network backend, so the pool it made is swapped for one that has it, with the same settings. If a
    later httpx stops keeping its pool in `_pool`, this raises at once instead of scraping unguarded.
    """
    transport = httpx.AsyncHTTPTransport(verify=verify, limits=limits)
    pool = getattr(transport, "_pool", None)
    if not isinstance(pool, httpcore.AsyncConnectionPool):
        raise RuntimeError("httpx's transport no longer keeps an httpcore pool in _pool: the SSRF guard can't be installed")
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=pool._ssl_context,
        max_connections=limits.max_connections,
        max_keepalive_connections=limits.max_keepalive_connections,
        keepalive_expiry=limits.keepalive_expiry,
        http1=True,
        http2=False,
        network_backend=GuardedBackend(),
    )
    return transport


def is_guarded(transport) -> bool:
    """Whether a transport connects through GuardedBackend (the tests and the startup check)."""
    pool = getattr(transport, "_pool", None)
    return isinstance(getattr(pool, "_network_backend", None), GuardedBackend)
