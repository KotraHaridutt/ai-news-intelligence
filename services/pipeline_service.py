import os
import asyncio
import time

from services import cache_service
from services.news_fetcher import fetch_news, scrape_pages
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

# --- Single-flight ---
# One build per normalized query at a time in this process. Later identical
# requests await the same task instead of starting their own build.
_inflight: dict[str, asyncio.Task] = {}

# --- Build limit ---
# At most this many full builds (GNews + scrape + embed + index) run at once.
# Different queries beyond the limit wait in line, which bounds memory on a
# 512 MB instance and the burst of Gemini embedding calls. Cache hits don't wait.
MAX_CONCURRENT_BUILDS = max(1, int(os.getenv("MAX_CONCURRENT_BUILDS", "3")))
_build_slots = asyncio.Semaphore(MAX_CONCURRENT_BUILDS)


async def get_pipeline_data(query: str):
    """
    Returns {"articles", "vector_store", "related_links"} for a query,
    {"articles": [], ...} when nothing usable was found, or None if fetching failed.
    """
    key = cache_service.normalize_query(query)
    # GNews gets the query as the user typed it (whitespace tidied), so search operators
    # like AND/OR/NOT reach it unchanged; only the cache key is lowercased.
    search_query = " ".join(query.split())

    task = _inflight.get(key)
    if task is None:
        task = asyncio.create_task(_load_or_build(key, search_query))
        _inflight[key] = task
        task.add_done_callback(lambda t: _on_task_done(key, t))
    else:
        print(f"[INFO] Joining in-flight build for '{key}'")

    # shield: if this request is cancelled, the build keeps running for the others
    return await asyncio.shield(task)


def _on_task_done(key: str, task: asyncio.Task):
    if _inflight.get(key) is task:
        del _inflight[key]
    # Mark the exception as retrieved so an error with no waiters left isn't logged as "never retrieved"
    if not task.cancelled():
        task.exception()


async def _load_or_build(key: str, search_query: str):
    start_time = time.time()

    cached = await cache_service.get_articles(key)
    if cached is not None:
        data = await _rebuild_from_cache(cached)
        if data is not None:
            print(f"[INFO] Cache HIT for '{key}': index rebuilt in {time.time() - start_time:.2f}s")
            return data
        print(f"[WARN] Cache entry for '{key}' couldn't be rebuilt. Running the full pipeline.")
    else:
        print(f"[INFO] Cache MISS for '{key}'. Running the full pipeline.")

    if _build_slots.locked():
        print(f"[INFO] All {MAX_CONCURRENT_BUILDS} build slots busy. '{key}' is waiting.")
    wait_start = time.time()
    async with _build_slots:
        waited = time.time() - wait_start
        data = await _build(key, search_query)
    print(f"[INFO] Pipeline for '{key}' finished in {time.time() - start_time:.2f}s (waited {waited:.2f}s for a build slot)")
    return data


async def _attach_pages(articles):
    """
    Reads page texts from the cache, scrapes only the URLs that aren't cached,
    caches the newly scraped ones and returns copies of the articles that have text.
    """
    urls = list(dict.fromkeys(article['url'] for article in articles))
    pages = await cache_service.get_pages(urls)

    to_scrape = [url for url in urls if url not in pages]
    if to_scrape:
        scraped = await scrape_pages(to_scrape)
        await cache_service.set_pages(scraped)
        pages.update(scraped)
    print(f"[INFO] Pages: {len(urls)} URLs, {len(urls) - len(to_scrape)} from cache, {len(to_scrape)} scraped.")

    return [{**article, 'full_text': pages[article['url']]} for article in articles if article['url'] in pages]


async def _rebuild_from_cache(cached):
    """
    Hit path: cached article list -> page texts -> chunk -> cached vectors -> FAISS.
    Returns None if no index can be built, so the caller falls back to a full build.
    """
    articles = cached.get("articles", [])
    indexed = [article for article in articles if article.get("indexed", True)]

    with_text = await _attach_pages(indexed)
    if len(with_text) < len(indexed):
        print(f"[WARN] {len(indexed) - len(with_text)} cached article(s) couldn't be re-scraped and are left out of the index.")

    vector_store = await build_vector_store(with_text)
    if vector_store is None:
        return None

    return {
        "articles": articles,
        "vector_store": vector_store,
        "related_links": cached.get("related_links", []),
    }


async def _build(key: str, search_query: str):
    """
    Miss path: GNews -> page texts (cache, then scrape) -> article vectors -> group
    -> rank + choose -> related links -> chunk -> chunk vectors -> FAISS.
    Cached only if it fully succeeded.
    """
    # 1. Fetch the article list
    try:
        found = await fetch_news(search_query)
    except Exception as e:
        print(f"[ERROR] Async fetch failed: {e}")
        return None

    if not found:
        print(f"[INFO] No articles found for '{key}'.")
        return {"articles": [], "vector_store": None, "related_links": []}

    # GNews can list the same URL twice; keep the first
    unique = {}
    for article in found:
        unique.setdefault(article['url'], article)
    found = list(unique.values())

    # 2. Page texts: cached ones first, scrape only the rest
    articles = await _attach_pages(found)
    if not articles:
        print(f"[INFO] No articles with enough text for '{key}'. Not cached.")
        return {"articles": [], "vector_store": None, "related_links": []}

    # 3. Group duplicates
    try:
        article_vectors = await get_vectors([article['full_text'] for article in articles])
    except Exception as e:
        print(f"[ERROR] Failed to embed articles for clustering: {e}")
        article_vectors = None
    # DBSCAN is CPU work: run it in a thread so the event loop stays free
    clustering_ok = await asyncio.to_thread(group_by_theme, articles, article_vectors)

    # 4. Rank + choose (gap_1): which articles reach the index and the Related Links
    similarities, ranking_ok = None, False
    if clustering_ok:
        try:
            query_vector = await get_query_vector(key)
            similarities = cosine_similarities(article_vectors, query_vector)
            ranking_ok = True
        except Exception as e:
            print(f"[ERROR] Failed to embed the query for ranking: {e}. Ranking by GNews order.")
        promote_unique(articles, similarities)
        choose_articles(articles, article_vectors, similarities)
        related_links = build_related_links(articles, similarities)
        print(f"[INFO] Selection for '{key}': " + "\n".join(selection_report(articles, similarities)))
    else:
        # Grouping failed, so duplicates can't be told apart: index everything, no Related Links
        for article in articles:
            article['indexed'], article['unique'] = True, False
        related_links = []
    for i, article in enumerate(articles):
        article['similarity'] = round(similarities[i], 4) if similarities else None

    # 5. Index only the chosen articles
    vector_store = await build_vector_store([article for article in articles if article['indexed']])

    # 6. Cache only a complete, successful build. Page and vector keys written
    #    above stay either way: that content is valid and the next build reuses it.
    if vector_store is not None and clustering_ok and ranking_ok:
        await cache_service.set_articles(key, {
            "articles": [{field: article.get(field) for field in ARTICLE_META_FIELDS} for article in articles],
            "related_links": related_links,
        })
    else:
        print(f"[WARN] Build for '{key}' not cached (index built: {vector_store is not None}, "
              f"clustering ok: {clustering_ok}, ranking ok: {ranking_ok}).")

    return {
        "articles": articles,
        "vector_store": vector_store,
        "related_links": related_links,
    }
