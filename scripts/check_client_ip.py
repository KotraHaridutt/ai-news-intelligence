"""
Checks on a deployed app that the per-visitor limits can't be dodged with a made-up X-Forwarded-For
header (feature 3, Section L #87).

  python scripts/check_client_ip.py https://<your-app>.onrender.com

It sends 3 small POSTs to /api/feedback with an empty body (each is refused 422 before anything is
looked up: no Gemini, GNews, Clef or LangSmith call) and reads the RateLimit-Remaining header the
limits middleware adds:
  1. plain                                 -> remaining N
  2. with "X-Forwarded-For: <random IP>"   -> N-1 if the app ignored the made-up entry (safe),
                                              the full limit - 1 if it started a new bucket (spoofable)
  3. plain again                           -> N-2 (safe) or N-1 (spoofable)
It uses 3 of the feedback bucket's requests for this minute; nothing else is counted.
SAFETY_RATE_LIMIT must be monitor or enforce (off sends no RateLimit headers), and REDIS_URL set.
Exit code 0 = safe, 1 = spoofable or no headers, 2 = couldn't reach the app.
"""
import random
import sys
import time

import httpx


def remaining(client: httpx.Client, url: str, headers: dict | None = None) -> tuple[int | None, int | None, int | None]:
    r = client.post(url, json={}, headers=headers or {})
    get = lambda name: int(r.headers[name]) if name in r.headers else None
    return get("ratelimit-remaining"), get("ratelimit-limit"), get("ratelimit-reset")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    base = sys.argv[1].rstrip("/")
    url = f"{base}/api/feedback"
    fake = f"198.51.100.{random.randint(1, 254)}"      # TEST-NET-2: never a real visitor
    try:
        with httpx.Client(timeout=30) as client:
            first, limit, reset = remaining(client, url)
            if first is None:
                print("No RateLimit headers: SAFETY_RATE_LIMIT is off, or Redis isn't set up on this deploy.")
                sys.exit(1)
            if reset is not None and reset <= 3:              # don't straddle two minutes
                time.sleep(reset + 1)
                first, limit, reset = remaining(client, url)
            spoofed, _, _ = remaining(client, url, {"X-Forwarded-For": fake})
            last, _, _ = remaining(client, url)
    except httpx.HTTPError as e:
        print(f"Couldn't reach {url}: {type(e).__name__}: {e}")
        sys.exit(2)

    print(f"limit {limit}/minute; remaining: plain {first}, with X-Forwarded-For {fake} {spoofed}, plain again {last}")
    if spoofed == first - 1 and last == first - 2:
        print("SAFE: the made-up X-Forwarded-For entry was ignored; all three requests counted for the same visitor.")
        sys.exit(0)
    if spoofed == limit - 1 and last == first - 1:
        print("SPOOFABLE: the made-up entry got its own bucket. The entry the app reads isn't the one the proxy wrote.\n"
              "Set CLIENT_IP_HOPS on Render (1 = last X-Forwarded-For entry, 2 = the one before it) and run this again.")
        sys.exit(1)
    print("Unclear (another visitor, or a minute boundary, may have changed the counts). Run it again.")
    sys.exit(1)


if __name__ == "__main__":
    main()
