"""
Removes a page found poisoned from every cache that holds it, and keeps it out of every build for
8 days (feature 3, Section G #55). Without --yes it only shows what it would do.

  python scripts/purge_url.py <url> [<url> ...]          # what would be removed
  python scripts/purge_url.py <url> [<url> ...] --yes    # remove it

Per URL (the URL GNews listed, as the logs and the retriever span show it):
  page      the scraped record (v1:scraped:) and the text cached before G (v1:page:)
  vectors   the article's vector (grouping, Follow) and its chunks' vectors, found from the cached text
  lists     every topic's article list (v1:articles:) and every check's list (v1:since:) that names
            the URL: deleted whole, so the next request rebuilds the topic without it
  purged    v1:purged:<url> is set for 8 days, longer than GNews lists an article (7 days), so no
            build scrapes or uses the URL again
Only reported, never changed:
  watches   followed topics (v1:watch:) that name the URL. A watch is the only copy of what its reader
            was shown; its next check won't use the page, because builds skip purged URLs
Not touched:
  answers   hold no URLs and expire 2 hours after they're written
  verdicts  Clef's verdicts on the page's chunks: judgements, not page text; useless without the page

A whole site that should never be used goes in SAFETY_BLOCKED_DOMAINS instead (#54).
Reads REDIS_URL from .env, like the app.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services import cache_service  # noqa: E402

LIST_PATTERNS = (f"{cache_service.KEY_VERSION}:articles:*", f"{cache_service.KEY_VERSION}:since:*")
WATCH_PATTERN = f"{cache_service.KEY_VERSION}:watch:*"


def _names(entry: dict, url: str) -> bool:
    """Whether an article list (a topic's or a check's) names the URL anywhere."""
    for field in ("articles", "unreadable", "related_links"):
        if any(isinstance(item, dict) and item.get("url") == url for item in entry.get(field) or []):
            return True
    return False


def vector_keys(text: str) -> list[str]:
    """The embedding keys a page's text was cached under: the whole text (article vector) and each chunk,
    cut exactly as the index cuts it (rag_service.text_splitter, invisible characters removed)."""
    from services import rag_service as rag
    from services.safety import articles as safety_articles
    chunks = safety_articles.article_chunks({"full_text": text}, rag.text_splitter)
    return list(dict.fromkeys([cache_service.emb_key(rag.EMB_CACHE_MODEL, text)]
                              + [cache_service.emb_key(rag.EMB_CACHE_MODEL, chunk) for chunk in chunks]))


async def _scan(client, pattern: str) -> list[str]:
    return [key.decode() if isinstance(key, bytes) else key async for key in client.scan_iter(match=pattern, count=500)]


async def purge(urls: list[str], apply: bool, client=None) -> dict:
    """What was (or would be) removed per URL. `client`: a Redis client (default the app's)."""
    client = client or cache_service.redis_client
    if client is None:
        raise SystemExit("REDIS_URL isn't set: there is no cache to purge.")
    list_keys = [key for pattern in LIST_PATTERNS for key in await _scan(client, pattern)]
    watch_keys = await _scan(client, WATCH_PATTERN)
    lists, watches = {}, {}
    for key in list_keys:
        raw = await client.get(key)
        try:
            lists[key] = json.loads(raw) if raw is not None else None
        except ValueError:
            lists[key] = None
    for key in watch_keys:
        raw = await client.get(key)
        watches[key] = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")

    report = {}
    for url in urls:
        record_raw, legacy_raw = await client.mget([cache_service.page_key(url), cache_service.legacy_page_key(url)])
        text = None
        if record_raw is not None:
            try:
                text = cache_service.page_record(json.loads(record_raw))["text"]
            except (ValueError, AttributeError):
                text = None
        if text is None and legacy_raw is not None:
            text = legacy_raw.decode("utf-8") if isinstance(legacy_raw, bytes) else legacy_raw
        candidates = [cache_service.page_key(url), cache_service.legacy_page_key(url)]
        candidates += vector_keys(text) if text else []
        naming = [key for key, entry in lists.items() if isinstance(entry, dict) and _names(entry, url)]
        candidates += naming
        present = [key for key, exists in zip(candidates, await _exists_each(client, candidates)) if exists]
        found_in_watches = [key.rsplit(":", 1)[-1] for key, raw in watches.items() if url in raw]
        if apply:
            pipe = client.pipeline(transaction=False)
            for key in present:
                pipe.delete(key)
            pipe.set(cache_service.purged_key(url), "1", ex=cache_service.PURGED_TTL)
            await pipe.execute()
        report[url] = {
            "page": [k for k in present if k in (cache_service.page_key(url), cache_service.legacy_page_key(url))],
            "vectors": sum(1 for k in present if ":emb:" in k),
            "lists": [k for k in present if k in naming],
            "watches_naming_it": found_in_watches,
            "page_text_found": text is not None,
            "applied": apply,
        }
    return report


async def _exists_each(client, keys: list[str]) -> list[bool]:
    if not keys:
        return []
    pipe = client.pipeline(transaction=False)
    for key in keys:
        pipe.exists(key)
    return [bool(n) for n in await pipe.execute()]


def main():
    parser = argparse.ArgumentParser(description="Remove a poisoned page from every cache (feature 3 #55).")
    parser.add_argument("urls", nargs="+", help="the article URL(s), as GNews listed them")
    parser.add_argument("--yes", action="store_true", help="remove it (without this, only show what would be removed)")
    args = parser.parse_args()
    report = asyncio.run(purge(args.urls, args.yes))
    for url, r in report.items():
        verb = "Removed" if r["applied"] else "Would remove"
        print(f"\n{url}")
        print(f"  {verb}: page {len(r['page'])} key(s){'' if r['page_text_found'] else ' (no cached text: vectors not found)'}, "
              f"{r['vectors']} vector(s), {len(r['lists'])} article list(s)")
        for key in r["lists"]:
            print(f"    list {key}")
        print(f"  {'Kept out' if r['applied'] else 'Would keep out'} of every build for "
              f"{cache_service.PURGED_TTL // 86400} days ({cache_service.purged_key(url)})")
        if r["watches_naming_it"]:
            print(f"  Followed topics naming it (not changed): {', '.join(r['watches_naming_it'])}")
    if not any(r["applied"] for r in report.values()):
        print("\nNothing changed. Run again with --yes to remove.")


if __name__ == "__main__":
    main()
