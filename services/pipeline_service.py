import os
import asyncio
import time
from datetime import datetime, timezone

from services import cache_service, errors, observability
from services.news_fetcher import fetch_news, scrape_pages, check_window
from services.clustering_service import group_by_theme
from services.rag_service import build_vector_store, get_vectors, get_query_vector
from services.selection_service import (
    cosine_similarities, promote_unique, choose_articles, build_related_links, selection_report,
)

# Article fields stored in the articles: cache entry. No full_text: page text
# lives once per URL under page: keys and is re-attached on a cache hit.
# 'indexed' marks the articles the choose step picked for the FAISS index, so a hit
# rebuilds the same index as the miss did. 'unique' and 'similarity' record why.
ARTICLE_META_FIELDS = ("url", "title", "source_name", "published_at", "theme_id", "indexed", "unique", "similarity")
# What a check keeps of an article it found but couldn't read (blocked, paywalled, timed out)
UNREADABLE_FIELDS = ("url", "title", "source_name", "published_at")

# --- Single-flight ---
# One build per normalized query at a time in this process. Later identical
# requests await the same task instead of starting their own build.
# Keys: the normalized query for a report build, ("since", from_hour, query) for a check.
_inflight: dict[str | tuple, asyncio.Task] = {}
# Which trace started each running build: a request that joins it links to that trace (#24)
_started_by: dict[str | tuple, dict | None] = {}

# --- Build limit ---
# At most this many full builds (GNews + scrape + embed + index) run at once.
# Different queries beyond the limit wait in line, which bounds memory on a
# 512 MB instance and the burst of Gemini embedding calls. Cache hits don't wait.
MAX_CONCURRENT_BUILDS = max(1, int(os.getenv("MAX_CONCURRENT_BUILDS", "3")))
_build_slots = asyncio.Semaphore(MAX_CONCURRENT_BUILDS)


# --- Span summaries (feature 2, Section C) ---
# What a span records of a build: counts and names, never page text, vectors or the index.

def _summary(data) -> dict:
    if data is None:
        return {"result": None}
    out = {"articles": len(data.get("articles", []))}
    if "vector_store" in data:
        out["indexed"] = sum(1 for a in data["articles"] if a.get("indexed", True))
        out["chunks"] = data["vector_store"].index.ntotal if data["vector_store"] is not None else 0
        out["related_links"] = len(data.get("related_links", []))
    if "unreadable" in data:
        out.update(found=data.get("articles_found"), unreadable=len(data["unreadable"]), window=data.get("window"))
    out["built_at"] = data.get("built_at")
    return out

def _articles_in(args, name="articles"):
    return {"articles": len(args.get(name) or [])}


def build_time() -> str:
    """
    When a build ran, e.g. '2026-10-13T03:16:33.512+00:00'. It names the build: an answer
    records the built_at of the build it came from, so Follow can tell whether the article
    list it would save is still the one that answer was written from (feature 1).
    Milliseconds, so two builds of one topic can't share a value.
    """
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def get_pipeline_data(query: str, since=None):
    """
    Returns {"articles", "vector_store", "related_links", "built_at"} for a query, with a built index.
    built_at is None for a cache entry written before feature 1 (it had no built_at).
    Raises AppError (with the status and message the user sees) when no index can be
    built: GNews failed, no news in the window, no readable page, or Gemini failed.
    Every request waiting on the same build gets the same error.

    With `since` (when the previous fetch ran: V1's build or the last check), the build for a
    "Check for updates" instead: see _build_since for what it returns. No index, and an empty
    window or unreadable pages are results, not errors.
    """
    key = cache_service.normalize_query(query)
    # GNews gets the query as the user typed it (whitespace tidied), so search operators
    # like AND/OR/NOT reach it unchanged; only the cache key is lowercased.
    search_query = " ".join(query.split())

    if since is None:
        flight, label = key, f"'{key}'"
        start = lambda: _load_or_build(key, search_query)
    else:
        window = check_window(since)
        flight, label = ("since", window["from_hour"], key), f"'{key}' since {window['from']}"
        start = lambda: _load_or_build_since(key, search_query, window)

    async with observability.span("pipeline", inputs={"query": key, "since": since},
                                  metadata={"kind": "report" if since is None else "check"}) as run:
        task = _inflight.get(flight)
        if task is None:
            # Created inside this span, so the build's own spans are traced under this request
            task = asyncio.create_task(start())
            _inflight[flight] = task
            _started_by[flight] = observability.current_trace()
            task.add_done_callback(lambda t: _on_task_done(flight, t))
        else:
            print(f"[INFO] Joining in-flight build for {label}")
            # The build's spans are in the trace of the request that started it: link to it (#24)
            started = _started_by.get(flight) or {}
            run.add_tags(["joined_shared_build"])
            run.add_metadata({"joined": True, "build_trace_id": started.get("trace_id"), "build_ref": started.get("ref")})

        # shield: if this request is cancelled, the build keeps running for the others
        wait_start = time.time()
        try:
            data = await asyncio.shield(task)
        except asyncio.CancelledError:
            run.add_metadata({"note": "this request was cancelled; the shared build kept running for the others"})
            raise
        run.add_metadata({"waited_s": round(time.time() - wait_start, 2)})
        if since is not None:
            # Requests in the same hour share the fetch, but the part a long gap couldn't
            # search depends on each request's own `since`
            data = {**data, "window": {**data["window"], "gap_not_covered": window["gap_not_covered"]}}
        run.add_outputs(_summary(data))
        return data


def _on_task_done(key, task: asyncio.Task):
    if _inflight.get(key) is task:
        del _inflight[key]
        _started_by.pop(key, None)
    # Mark the exception as retrieved so an error with no waiters left isn't logged as "never retrieved"
    if not task.cancelled():
        task.exception()


@observability.traced("cache_or_build", outputs=_summary)
async def _load_or_build(key: str, search_query: str):
    start_time = time.time()

    cached = await cache_service.get_articles(key)
    if cached is not None and not cached.get("built_at"):
        # Written before builds had a built_at (feature 1). A report from it could never be
        # followed (Follow would say "more than 2 hours old" about a fresh report), so it is
        # rebuilt once. Only in the first 2 h after the deploy; page text and vectors are reused.
        print(f"[INFO] Cache entry for '{key}' has no built_at (written before feature 1). Rebuilding it.")
        cached = None
    if cached is not None:
        data = await _rebuild_from_cache(cached)
        if data is not None:
            observability.annotate(path="cache_hit")
            observability.add_totals(articles_cache="hit")
            print(f"[INFO] Cache HIT for '{key}': index rebuilt in {time.time() - start_time:.2f}s")
            return data
        observability.annotate(path="cache_unusable_full_build")
        observability.add_totals(articles_cache="unusable")
        print(f"[WARN] Cache entry for '{key}' couldn't be rebuilt. Running the full pipeline.")
    else:
        observability.annotate(path="cache_miss_full_build")
        observability.add_totals(articles_cache="miss")
        print(f"[INFO] Cache MISS for '{key}'. Running the full pipeline.")

    if _build_slots.locked():
        print(f"[INFO] All {MAX_CONCURRENT_BUILDS} build slots busy. '{key}' is waiting.")
    wait_start = time.time()
    async with _build_slots:
        waited = time.time() - wait_start
        observability.annotate(waited_for_build_slot_s=round(waited, 2))
        data = await _build(key, search_query)
    print(f"[INFO] Pipeline for '{key}' finished in {time.time() - start_time:.2f}s (waited {waited:.2f}s for a build slot)")
    return data


@observability.traced("pages", inputs=_articles_in, outputs=lambda with_text: {"with_text": len(with_text)})
async def _attach_pages(articles):
    """
    Reads page texts from the cache, scrapes only the URLs that aren't cached,
    caches the newly scraped ones and returns copies of the articles that have text.
    """
    urls = list(dict.fromkeys(article['url'] for article in articles))
    pages = await cache_service.get_pages(urls)

    to_scrape = [url for url in urls if url not in pages]
    scraped = {}
    if to_scrape:
        scraped = await scrape_pages(to_scrape)
        await cache_service.set_pages(scraped)
        pages.update(scraped)
    observability.annotate(urls=len(urls), from_cache=len(urls) - len(to_scrape), scraped=len(to_scrape), scrape_ok=len(scraped))
    observability.add_totals(pages_from_cache=len(urls) - len(to_scrape), pages_scraped=len(to_scrape))
    print(f"[INFO] Pages: {len(urls)} URLs, {len(urls) - len(to_scrape)} from cache, "
          f"{len(to_scrape)} to scrape, {len(scraped)} succeeded.")

    return [{**article, 'full_text': pages[article['url']]} for article in articles if article['url'] in pages]


@observability.traced("rebuild_from_cache", inputs=lambda a: {"articles": len(a["cached"].get("articles", []))},
                      outputs=_summary)
async def _rebuild_from_cache(cached):
    """
    Hit path: cached article list -> page texts -> chunk -> cached vectors -> FAISS.
    Returns None if none of the cached articles has text any more, so the caller falls
    back to a full build. Raises AppError if Gemini fails.
    """
    articles = cached.get("articles", [])
    indexed = [article for article in articles if article.get("indexed", True)]

    with_text = await _attach_pages(indexed)
    if len(with_text) < len(indexed):
        print(f"[WARN] {len(indexed) - len(with_text)} cached article(s) couldn't be re-scraped and are left out of the index.")

    # If Gemini fails this raises AppError and there is no fallback: a full build would
    # need Gemini too, and would spend a GNews request (free plan: 1/s, 100/day) on the way.
    vector_store = await build_vector_store(with_text)
    if vector_store is None:
        return None

    return {
        "articles": articles,
        "vector_store": vector_store,
        "related_links": cached.get("related_links", []),
        "built_at": cached.get("built_at"),
    }


@observability.traced("full_build", outputs=_summary)
async def _build(key: str, search_query: str):
    """
    Miss path: GNews -> page texts (cache, then scrape) -> article vectors -> group
    -> rank + choose -> related links -> chunk -> chunk vectors -> FAISS.
    Cached only if it fully succeeded. Raises AppError for every failure the user
    should see; raising also skips the cache write.
    """
    built_at = build_time()

    # 1. Fetch the article list (fetch_news raises AppError if GNews fails)
    found = await fetch_news(search_query)
    if not found:
        raise errors.no_recent_news(search_query)

    # GNews can list the same URL twice; keep the first
    found = _unique_by_url(found)

    # 2. Page texts: cached ones first, scrape only the rest
    articles = await _attach_pages(found)
    if not articles:
        raise errors.no_readable_pages(search_query, len(found))

    # 3. Group duplicates
    try:
        article_vectors = await get_vectors([article['full_text'] for article in articles])
    except Exception as e:
        print(f"[ERROR] Failed to embed articles for clustering: {e}")
        article_vectors = None

    # 3-4. Group, rank + choose (gap_1): which articles reach the index and the Related Links
    clustering_ok, ranking_ok, related_links = await _group_and_choose(key, f"'{key}'", articles, article_vectors)

    # 5. Index only the chosen articles (raises AppError if Gemini fails)
    vector_store = await build_vector_store([article for article in articles if article['indexed']])
    if vector_store is None:
        # Only when no chosen article yields a chunk, i.e. no usable text
        raise errors.no_readable_pages(search_query, len(found))

    # 6. Cache only a complete, successful build. Page and vector keys written
    #    above stay either way: that content is valid and the next build reuses it.
    if clustering_ok and ranking_ok:
        await cache_service.set_articles(key, {
            "articles": [{field: article.get(field) for field in ARTICLE_META_FIELDS} for article in articles],
            "related_links": related_links,
            "built_at": built_at,
        })
    else:
        print(f"[WARN] Build for '{key}' not cached (clustering ok: {clustering_ok}, ranking ok: {ranking_ok}).")

    return {
        "articles": articles,
        "vector_store": vector_store,
        "related_links": related_links,
        "built_at": built_at,
    }


def _unique_by_url(found):
    unique = {}
    for article in found:
        unique.setdefault(article['url'], article)
    return list(unique.values())


@observability.traced("group_and_choose", inputs=lambda a: {"articles": len(a["articles"]), "have_vectors": a["article_vectors"] is not None},
                      outputs=lambda r: {"clustering_ok": r[0], "ranking_ok": r[1], "related_links": len(r[2])})
async def _group_and_choose(key: str, label: str, articles, article_vectors):
    """
    Steps 3-4 of a build, shared by report builds and checks: group duplicates (DBSCAN),
    rank by query relevance, keep at most 2 articles per group (sets theme_id, unique,
    indexed and similarity on each article, in place). Returns (clustering_ok, ranking_ok,
    related_links). Without article vectors grouping fails, and every article is indexed.
    """
    # DBSCAN is CPU work: run it in a thread so the event loop stays free
    clustering_ok = await asyncio.to_thread(group_by_theme, articles, article_vectors)

    similarities, ranking_ok = None, False
    if clustering_ok:
        try:
            query_vector = await get_query_vector(key)
            similarities = cosine_similarities(article_vectors, query_vector)
            ranking_ok = True
        except Exception as e:
            print(f"[ERROR] Failed to embed the query for ranking: {e}. Ranking by GNews relevance order.")
        promote_unique(articles, similarities)
        choose_articles(articles, article_vectors, similarities)
        related_links = build_related_links(articles, similarities)
        print(f"[INFO] Selection for {label}: " + "\n".join(selection_report(articles, similarities)))
    else:
        # Grouping failed, so duplicates can't be told apart: index everything, no Related Links
        for article in articles:
            article['indexed'], article['unique'] = True, False
        related_links = []
    for i, article in enumerate(articles):
        article['similarity'] = round(similarities[i], 4) if similarities else None
    observability.annotate(groups=len({a.get('theme_id') for a in articles if a.get('theme_id', -1) != -1}),
                           indexed=sum(1 for a in articles if a.get('indexed')),
                           unique=sum(1 for a in articles if a.get('unique')))
    return clustering_ok, ranking_ok, related_links


# --- Builds for "Check for updates" (feature 1) ---
# A check needs the articles published since the previous fetch, their page text and their
# article vectors (to compare with the stories already seen), grouped and chosen like a
# report build. It needs no FAISS index: the fact check reads the chosen articles' first
# chunks directly, so building one would only spend embedding calls on chunks.

def _pick(article, fields):
    return {field: article.get(field) for field in fields}


async def _article_vectors(articles):
    """Article vectors, read from the 7-day cache first. A check can't compare without them."""
    if not articles:
        return []
    try:
        return await get_vectors([article['full_text'] for article in articles])
    except Exception as e:
        raise errors.analysis_failed(e) from e


@observability.traced("check_cache_or_build", inputs=lambda a: {"key": a["key"], "from": a["window"]["from"]}, outputs=_summary)
async def _load_or_build_since(key: str, search_query: str, window: dict):
    start_time = time.time()
    label = f"'{key}' since {window['from']}"

    cached = await cache_service.get_since_articles(key, window["from_hour"])
    observability.annotate(path="cache_hit" if cached is not None else "cache_miss_build")
    if cached is not None:
        data = await _since_from_cache(cached)
        print(f"[INFO] Check cache HIT for {label}: {len(data['articles'])} articles in {time.time() - start_time:.2f}s")
        return data
    print(f"[INFO] Check cache MISS for {label}. Fetching.")

    if _build_slots.locked():
        print(f"[INFO] All {MAX_CONCURRENT_BUILDS} build slots busy. Check for {label} is waiting.")
    async with _build_slots:
        data = await _build_since(key, search_query, window)
    print(f"[INFO] Check build for {label} finished in {time.time() - start_time:.2f}s: "
          f"{data['articles_found']} found, {len(data['articles'])} readable, {len(data['unreadable'])} unreadable")
    return data


@observability.traced("check_from_cache", inputs=lambda a: {"articles": len(a["cached"].get("articles", []))}, outputs=_summary)
async def _since_from_cache(cached):
    """Hit path: the cached article list -> page texts (scraping any that expired) -> article vectors."""
    articles = cached.get("articles", [])
    with_text = await _attach_pages(articles) if articles else []
    readable = {article['url'] for article in with_text}
    lost = [_pick(article, UNREADABLE_FIELDS) for article in articles if article['url'] not in readable]
    if lost:
        print(f"[WARN] {len(lost)} cached check article(s) couldn't be read again.")
    return {
        "articles": with_text,
        "article_vectors": await _article_vectors(with_text),
        "unreadable": cached.get("unreadable", []) + lost,
        "articles_found": cached.get("articles_found", len(articles)),
        "built_at": cached.get("built_at"),
        "window": cached.get("window", {}),
    }


@observability.traced("check_build", inputs=lambda a: {"key": a["key"], "from": a["window"]["from"]}, outputs=_summary)
async def _build_since(key: str, search_query: str, window: dict):
    """
    Miss path of a check: GNews from window['from'] -> page texts -> article vectors -> group
    -> rank + choose. Returns
      articles         readable articles with full_text, theme_id, unique, indexed, similarity
      article_vectors  one vector per article, same order
      unreadable       articles found but not readable (url, title, source_name, published_at)
      articles_found   how many GNews returned
      built_at, window ({from, from_hour, to_approx}; gap_not_covered is added per request)
    No articles in the window, or none readable, is a result (the check shows "nothing new" or
    the unreadable ones), not an error. Raises AppError if GNews or Gemini fails.
    Cached like a report build: only if it fully succeeded, and not when nothing was readable
    (a blocked page may be readable on the next try).
    """
    built_at = build_time()
    found = _unique_by_url(await fetch_news(search_query, window_from=window["from"]))
    result = {
        "articles": [], "article_vectors": [], "unreadable": [],
        "articles_found": len(found), "built_at": built_at,
        "window": {k: window[k] for k in ("from", "from_hour", "to_approx")},
    }

    def cache_entry():
        return {**{k: v for k, v in result.items() if k != "article_vectors"},
                "articles": [_pick(article, ARTICLE_META_FIELDS) for article in result["articles"]]}

    if not found:
        print(f"[INFO] No articles for '{key}' since {window['from']}.")
        await cache_service.set_since_articles(key, window["from_hour"], cache_entry())
        return result

    articles = await _attach_pages(found)
    readable = {article['url'] for article in articles}
    result["unreadable"] = [_pick(article, UNREADABLE_FIELDS) for article in found if article['url'] not in readable]
    if not articles:
        print(f"[WARN] Check for '{key}': {len(found)} found, none readable. Not cached.")
        return result

    vectors = await _article_vectors(articles)
    clustering_ok, ranking_ok, _ = await _group_and_choose(key, f"'{key}' since {window['from']}", articles, vectors)
    result.update(articles=articles, article_vectors=vectors)

    if clustering_ok and ranking_ok:
        await cache_service.set_since_articles(key, window["from_hour"], cache_entry())
    else:
        print(f"[WARN] Check build for '{key}' not cached (clustering ok: {clustering_ok}, ranking ok: {ranking_ok}).")
    return result
