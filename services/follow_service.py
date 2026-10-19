import asyncio
from datetime import datetime

from services import cache_service, errors, observability
from services import novelty_service as nov
from services.cache_service import WatchStoreUnavailable
from services.fact_check_service import run_fact_check
from services.pipeline_service import get_pipeline_data, build_time, _attach_pages
from services.rag_service import get_vectors

# --- feature 1: following a topic ---
# Follow saves what report V1 contained; Check shows only what is new since the last look.
# This module connects the pieces: answer records and watch storage (cache_service), the
# check build (pipeline_service), the what's-new decision (novelty_service) and the fact
# check (fact_check_service). server.py only adds the routes.

# A check this soon after the last one returns the stored result: no GNews or Gemini call
# (feature-1 decision #16). Both quotas are small (100 GNews requests and 20 chat calls a day).
REUSE_MINUTES = 30

# One check per followed topic at a time; a second request joins the first
_checks_inflight: dict[str, asyncio.Task] = {}
# Which trace started each running check: a request that joins it links to that trace (feature 2, #24)
_checks_started_by: dict[str, dict | None] = {}

RETRY_MESSAGE = "Couldn't check the new articles on stories you've seen. Check again to retry."


def _minutes_between(earlier: str, later: str) -> float:
    a = datetime.fromisoformat(earlier.replace("Z", "+00:00"))
    b = datetime.fromisoformat(later.replace("Z", "+00:00"))
    return (b - a).total_seconds() / 60


# --- Follow ---

async def follow(query: str, answer_ids: dict) -> dict:
    """
    Saves report V1 as a followed topic. answer_ids: {"report": id, "timeline": id?,
    "contradictions": id?}, the answers the page is showing. Their text becomes "already
    told" and the article list they were written from becomes the known stories and seen
    URLs. No GNews call and no chat call: everything comes from what the report already built.
    Raises AppError: 503 no Redis, 409 report too old / not followable.
    """
    search_query = " ".join(query.split())
    key = cache_service.normalize_query(query)
    try:
        cache_service.require_watch_store()
    except WatchStoreUnavailable as e:
        raise errors.follow_unavailable(str(e)) from e

    answer_ids = {view: aid for view, aid in (answer_ids or {}).items() if aid}
    report_id = answer_ids.get("report")
    if not cache_service.is_valid_id(report_id):
        raise errors.report_not_followable(f"no valid report answer id for '{key}': {report_id!r}")
    # Strict reads: if Redis fails here, the answer is 503 "unavailable", never a wrong
    # "your report is too old" (found in the live run: a failed read looked like an expired report)
    try:
        records = await cache_service.get_answers(list(answer_ids.values()), strict=True)
        entry = await cache_service.get_articles(key, strict=True)
    except WatchStoreUnavailable as e:
        raise errors.follow_unavailable(str(e)) from e

    report = records.get(report_id)
    if report is None:
        raise errors.report_too_old(f"report answer {report_id} for '{key}' expired or unknown")
    for view, aid in answer_ids.items():
        record = records.get(aid)
        if record is not None and (record.get("query") != key or record.get("view") != view):
            raise errors.report_not_followable(
                f"answer {aid} is the {record.get('view')} of '{record.get('query')}', not the {view} of '{key}'")

    # The article list must be the one the report was written from (Section B: built_at)
    if entry is None or not entry.get("built_at") or entry["built_at"] != report.get("built_at"):
        raise errors.report_too_old(
            f"article list for '{key}': {'missing' if entry is None else entry.get('built_at')}, report built_at {report.get('built_at')}")

    articles = entry.get("articles", [])
    with_text = await _attach_pages(articles) if articles else []
    try:
        vectors = await get_vectors([a["full_text"] for a in with_text]) if with_text else []
    except Exception as e:
        raise errors.analysis_failed(e) from e

    told = [{"at": r["created_at"], "kind": r["view"], "text": r["text"]}
            for r in (records.get(aid) for aid in answer_ids.values()) if r is not None and r.get("text")]
    now = build_time()
    with observability.span("novelty.baseline", inputs={"articles": len(with_text), "told": [t["kind"] for t in told]}) as run:
        snapshot = nov.baseline(search_query, with_text, vectors, told, entry["built_at"], now)
        run.add_outputs({"stories": len(snapshot["stories"]), "seen_urls": len(snapshot["seen_urls"])})
    # Articles whose page can't be read any more have no vector, so they can't be a story,
    # but they were in V1's build: seen all the same
    snapshot["seen_urls"] = nov._add_unique(snapshot["seen_urls"], [a["url"] for a in articles])[-nov.MAX_SEEN_URLS:]

    watch_id = cache_service.new_id()
    try:
        await cache_service.set_watch(watch_id, snapshot)
    except WatchStoreUnavailable as e:
        raise errors.follow_unavailable(str(e)) from e
    print(f"[INFO] FOLLOW '{key}' as {watch_id}: {len(snapshot['stories'])} stories, "
          f"{len(snapshot['seen_urls'])} seen URLs, told {', '.join(t['kind'] for t in snapshot['told'])}")
    return {"watch_id": watch_id, "query": search_query, "created_at": now, "stories": len(snapshot["stories"])}


# --- Reading and stopping ---

async def _load(watch_id: str) -> dict:
    try:
        snapshot = await cache_service.get_watch(watch_id)
    except WatchStoreUnavailable as e:
        raise errors.follow_unavailable(str(e)) from e
    if snapshot is None:
        raise errors.watch_not_found(f"watch {watch_id!r} unknown, expired or malformed")
    return snapshot

async def story_log(watch_id: str) -> dict:
    snapshot = await _load(watch_id)
    return {
        "watch_id": watch_id,
        "query": snapshot["query"],
        "created_at": snapshot["created_at"],
        "last_checked_at": snapshot["last_checked_at"],
        "checks": snapshot.get("checks", 0),
        "stories": nov.story_log(snapshot),
        "last_result": snapshot.get("last_result"),
    }

async def unfollow(watch_id: str) -> dict:
    try:
        await cache_service.delete_watch(watch_id)
    except WatchStoreUnavailable as e:
        raise errors.follow_unavailable(str(e)) from e
    return {"watch_id": watch_id, "deleted": True}


# --- Check ---

async def check(watch_id: str, render) -> dict:
    """
    "Check for updates". render: markdown -> safe HTML (server.render_markdown).
    One check per followed topic at a time; a second request gets the first one's result.
    """
    async with observability.span("check", inputs={"watch": observability.short_hash(watch_id)}) as run:
        task = _checks_inflight.get(watch_id)
        if task is None:
            # Created inside this span, so the check's own spans are traced under this request
            task = asyncio.create_task(_check(watch_id, render))
            _checks_inflight[watch_id] = task
            _checks_started_by[watch_id] = observability.current_trace()
            task.add_done_callback(lambda t: _on_check_done(watch_id, t))
        else:
            print(f"[INFO] Joining the check already running for {watch_id}")
            started = _checks_started_by.get(watch_id) or {}
            run.add_tags(["joined_running_check"])
            run.add_metadata({"joined": True, "check_trace_id": started.get("trace_id"), "check_ref": started.get("ref")})
        result = await asyncio.shield(task)
        run.add_outputs({key: len(result[key]) for key in ("new_stories", "updates", "hidden_repeats", "unreadable")}
                        | {"nothing_new": result["nothing_new"], "reused": result["reused"]})
        return result

def _on_check_done(watch_id, task):
    if _checks_inflight.get(watch_id) is task:
        del _checks_inflight[watch_id]
        _checks_started_by.pop(watch_id, None)
    if not task.cancelled():
        task.exception()


def _stories_to_judge(snapshot, articles, classified) -> list[dict]:
    titles = {s["id"]: s["title"] for s in snapshot["stories"]}
    stories = [{"id": n["key"], "kind": "new",
                "related_to": {"id": n["related_to"], "title": titles.get(n["related_to"])} if n["related_to"] else None,
                "articles": [articles[i] for i in n["judge"]]} for n in classified["new"]]
    stories += [{"id": u["story_id"], "kind": "update", "title": titles.get(u["story_id"]),
                 "articles": [articles[i] for i in u["judge"]]} for u in classified["updates"]]
    return stories


def _response(watch_id, snapshot, merged, data, articles, classified, result, ids, render, now) -> dict:
    """The check's answer for the page: what's new, what was hidden as a repeat, what couldn't be read."""
    outcomes, texts = result["outcomes"], result["texts"]
    stories = {s["id"]: s for s in merged["stories"]}
    links = lambda idxs: [nov.link(articles[i]) for i in nov._by_relevance(articles, idxs)]

    new_stories, updates, hidden = [], [], []
    for item in classified["new"]:
        outcome = outcomes.get(item["key"], nov.UNJUDGED)
        if outcome == nov.NOTHING_NEW:
            hidden += links(item["articles"])
            continue
        story = stories[ids[item["key"]]]
        related = item["related_to"]
        new_stories.append({
            "story_id": story["id"],
            "title": story["title"],
            "outlets": story["outlets"],
            "articles": len(item["articles"]),
            "summary_html": render(texts[item["key"]]) if outcome == nov.SHOWN else None,
            "related_to": {"id": related, "title": stories[related]["title"]} if related in stories else None,
            "links": links(item["articles"]),
        })
    retry = False
    for item in classified["updates"]:
        outcome = outcomes.get(item["story_id"], nov.UNJUDGED)
        if outcome == nov.NOTHING_NEW:
            hidden += links(item["articles"])
        elif outcome == nov.UNJUDGED:
            retry = True                      # not shown, not marked seen: the next check tries again
        else:
            story = stories[item["story_id"]]
            updates.append({
                "story_id": story["id"],
                "story_title": story["title"],
                "first_seen": story["first_seen"],
                "new_facts_html": render(texts[item["story_id"]]),
                "links": links(item["articles"]),
            })

    window = data["window"]
    return {
        "watch_id": watch_id,
        "query": snapshot["query"],
        "checked_at": now,
        "previous_check": snapshot["last_checked_at"],
        "window": {"from": window["from"], "to_approx": window["to_approx"], "gap_not_covered": window.get("gap_not_covered")},
        "articles_found": data["articles_found"],
        "new_stories": new_stories,
        "updates": updates,
        "hidden_repeats": hidden,
        "unreadable": [{k: a.get(k) for k in nov.LINK_FIELDS} for a in data["unreadable"]],
        "nothing_new": not new_stories and not updates,
        "message": RETRY_MESSAGE if retry else None,
        "reused": False,
    }


async def _check(watch_id: str, render) -> dict:
    snapshot = await _load(watch_id)
    now = build_time()

    last = snapshot.get("last_result")
    if last and _minutes_between(snapshot["last_checked_at"], now) < REUSE_MINUTES:
        observability.add_tags("reused_result")
        print(f"[INFO] CHECK {watch_id}: last check {snapshot['last_checked_at']} is under {REUSE_MINUTES} min old, its result reused")
        return {**last, "reused": True}

    # GNews and Gemini errors (AppError) go straight to the page; nothing is saved
    data = await get_pipeline_data(snapshot["query"], since=snapshot["last_fetch_at"])
    articles, vectors = data["articles"], data["article_vectors"]
    with observability.span("novelty.classify", inputs={"articles": len(articles), "known_stories": len(snapshot["stories"]),
                                                       "seen_urls": len(snapshot["seen_urls"])}) as run:
        classified = nov.classify(snapshot, articles, vectors)
        # Each article's decision and distance: the numbers the thresholds were tuned on
        run.add_outputs({"seen": len(classified["seen"]), "updates": len(classified["updates"]), "new": len(classified["new"]),
                         "related": sum(1 for n in classified["new"] if n["related_to"]),
                         "decisions": [{k: d[k] for k in ("outcome", "story_id", "distance")} for d in classified["decisions"]]})
    stories = _stories_to_judge(snapshot, articles, classified)
    known = [{"id": s["id"], "title": s["title"]} for s in snapshot["stories"]]
    result = await run_fact_check(snapshot["query"], stories, known, snapshot["told"])

    ids = {}
    with observability.span("novelty.merge", inputs={"stories_judged": len(stories)}) as run:
        merged = nov.merge(snapshot, articles, vectors, classified, result["outcomes"], result["texts"],
                           data["built_at"] or now, now, ids_out=ids)
        response = _response(watch_id, snapshot, merged, data, articles, classified, result, ids, render, now)
        run.add_outputs({"new_stories": len(response["new_stories"]), "updates": len(response["updates"]),
                         "hidden_repeats": len(response["hidden_repeats"]), "unreadable": len(response["unreadable"]),
                         "known_stories_after": len(merged["stories"])})
    merged["last_result"] = response

    # Saved before anything is shown: a result that isn't saved would be "new" again next time
    try:
        await cache_service.set_watch(watch_id, merged)
    except WatchStoreUnavailable as e:
        raise errors.check_not_saved(str(e)) from e

    decisions = ", ".join(f"{d['outcome']}{'' if d['distance'] is None else ' %.3f' % d['distance']}" for d in classified["decisions"])
    print(f"[INFO] CHECK {watch_id} '{snapshot['query']}' from {data['window']['from']}: {data['articles_found']} found, "
          f"{len(articles)} readable; {len(response['new_stories'])} new, {len(response['updates'])} updates, "
          f"{len(response['hidden_repeats'])} hidden repeats, {len(response['unreadable'])} unreadable [{decisions}]")
    return response
