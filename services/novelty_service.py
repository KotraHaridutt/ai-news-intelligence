import base64
import copy
from datetime import datetime
import numpy as np

# --- feature 1: what's new since the user last looked ---
# Pure functions (no I/O), called by follow_service: build a followed topic's snapshot from
# the V1 build, sort a check's articles into seen / update to a known story / new, and merge
# what the check showed back into the snapshot. Distances are cosine distances between
# whole-article vectors (gemini-embedding-001, RETRIEVAL_DOCUMENT), like DBSCAN's.

SNAPSHOT_VERSION = 1

# Measured across days (feature-1 Section A, 32 labelled articles a day apart):
#   same storyline 0.046-0.153, follow-up 0.107-0.165, different story 0.196+
# Below KNOWN_STORY_DISTANCE every article was the same storyline: an update to that story.
# Between it and RELATED_DISTANCE every article was a follow-up or the same story: shown as
# a new story "related to" the known one, and the fact check removes what was already told.
KNOWN_STORY_DISTANCE = 0.12
RELATED_DISTANCE = 0.18

# Outcomes of the fact check for one story (follow_service turns Section E's answer into these)
SHOWN = "shown"               # has text to show: a summary, or new facts
NOTHING_NEW = "nothing_new"   # judged: only repeats what was already told -> hidden
UNJUDGED = "unjudged"         # no usable answer for it -> a known story stays hidden and is retried

MAX_JUDGED_PER_STORY = 2      # articles per story sent to the fact check (like choose_articles' 2 per group)
MAX_LINKS_PER_STORY = 5
MAX_VECTORS_PER_STORY = 3     # the newest; distances are to the nearest one
MAX_SEEN_URLS = 300           # oldest dropped first; later windows rarely reach them again
MAX_TOLD_CHARS = 10_000       # what the fact check is given as "already told"
VECTOR_DAYS = 7               # = news_fetcher.MAX_LOOKBACK_HOURS: older stories can't come back

LINK_FIELDS = ("title", "url", "source_name", "published_at")


# --- Vectors: base64 float16 (measured in Section A: distances change by at most 0.000027) ---

def encode_vector(vector) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float16).tobytes()).decode("ascii")

def decode_vector(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float16).astype(np.float32)

def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


# --- Small helpers ---

def _days_between(earlier: str, later: str) -> float:
    a = datetime.fromisoformat(earlier.replace("Z", "+00:00"))
    b = datetime.fromisoformat(later.replace("Z", "+00:00"))
    return (b - a).total_seconds() / 86400

def _by_relevance(articles, idxs):
    """Most relevant first (the choose step's similarity); without one, the build's order."""
    return sorted(idxs, key=lambda i: (-(articles[i].get("similarity") or 0.0), i))

def _top_similarity(articles, idxs) -> float:
    return max((articles[i].get("similarity") or 0.0) for i in idxs)

def link(article) -> dict:
    return {field: article.get(field) for field in LINK_FIELDS}

def _add_unique(items, new_items, key=lambda x: x):
    seen = {key(x) for x in items}
    for item in new_items:
        if key(item) not in seen:
            items.append(item)
            seen.add(key(item))
    return items

def judged_articles(articles, idxs):
    """The articles of a story sent to the fact check: the chosen (indexed) ones first, then by relevance."""
    ranked = _by_relevance(articles, idxs)
    indexed = [i for i in ranked if articles[i].get("indexed")]
    return (indexed + [i for i in ranked if i not in indexed])[:MAX_JUDGED_PER_STORY]

def _new_story(snapshot, articles, vectors, idxs, now, related_to=None) -> dict:
    ranked = _by_relevance(articles, idxs)
    top = articles[ranked[0]]
    story = {
        "id": f"s{snapshot['next_story']}",
        "title": top.get("title") or "Untitled",
        "links": [link(articles[i]) for i in ranked[:MAX_LINKS_PER_STORY]],
        "outlets": _add_unique([], [articles[i].get("source_name") for i in ranked if articles[i].get("source_name")]),
        "first_seen": now,
        "last_seen": now,
        "articles_seen": len(idxs),
        # the chosen articles' vectors first: P6, distances were measured article to article
        "vectors": [encode_vector(vectors[i]) for i in judged_articles(articles, idxs)[:MAX_VECTORS_PER_STORY]],
    }
    if related_to:
        story["related_to"] = related_to
    snapshot["next_story"] += 1
    snapshot["stories"].append(story)
    return story

def _add_to_story(story, articles, vectors, idxs, now, add_vectors=True):
    ranked = _by_relevance(articles, idxs)
    story["last_seen"] = now
    story["articles_seen"] += len(idxs)
    # newest coverage first in the links, the cap keeps the most recent
    story["links"] = _add_unique([link(articles[i]) for i in ranked], story["links"], key=lambda l: l["url"])[:MAX_LINKS_PER_STORY]
    _add_unique(story["outlets"], [articles[i].get("source_name") for i in ranked if articles[i].get("source_name")])
    if add_vectors:
        fresh = [encode_vector(vectors[i]) for i in judged_articles(articles, idxs)]
        story["vectors"] = (story["vectors"] + fresh)[-MAX_VECTORS_PER_STORY:]


# --- The snapshot ---

def cap_told(told: list[dict]) -> list[dict]:
    """
    Keeps "already told" under MAX_TOLD_CHARS: drops the oldest texts first, V1's report last
    (it holds the most facts); if the report alone is too long, its beginning is kept.
    """
    told = [dict(t) for t in told]
    def total():
        return sum(len(t["text"]) for t in told)
    while total() > MAX_TOLD_CHARS:
        others = [t for t in told if t["kind"] != "report"]
        if others:
            told.remove(min(others, key=lambda t: t["at"]))
        else:
            report = told[0]
            report["text"] = report["text"][:MAX_TOLD_CHARS]
            told = [report]
    return told

def baseline(query: str, articles, article_vectors, told: list[dict], fetched_at: str, now: str) -> dict:
    """
    The snapshot of a newly followed topic, from V1's build (articles with theme_id, indexed,
    similarity, and their vectors in the same order) and V1's answer texts (`told`: {"at",
    "kind", "text"}). Every story V1 was built from is known; every article in the build is
    seen, also those over the unique cap that weren't shown (they're older than any later window).
    fetched_at: when V1's articles were fetched (its built_at): the next check searches from there.
    """
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "query": query,
        "created_at": now,
        "last_checked_at": now,
        "last_fetch_at": fetched_at,
        "checks": 0,
        "next_story": 1,
        "told": cap_told(sorted(told, key=lambda t: (t["kind"] != "report", t["at"]))),
        "seen_urls": [],
        "stories": [],
    }
    groups = {}
    for i, article in enumerate(articles):
        if article.get("theme_id", -1) != -1:
            groups.setdefault(article["theme_id"], []).append(i)
    # Most relevant story first, so story ids follow the order of importance
    for idxs in sorted(groups.values(), key=lambda g: -_top_similarity(articles, g)):
        _new_story(snapshot, articles, article_vectors, idxs, now)
    _add_unique(snapshot["seen_urls"], [a["url"] for a in articles])
    snapshot["seen_urls"] = snapshot["seen_urls"][-MAX_SEEN_URLS:]
    return snapshot


# --- Sorting a check's articles ---

def _known_vectors(snapshot):
    """(matrix of unit vectors, story id per row) for the stories that still have vectors."""
    rows, ids = [], []
    for story in snapshot["stories"]:
        for text in story.get("vectors", []):
            rows.append(_unit(decode_vector(text)))
            ids.append(story["id"])
    return (np.vstack(rows) if rows else None), ids

def classify(snapshot, articles, article_vectors) -> dict:
    """
    Sorts a check's readable articles (with theme_id from the check build, and their vectors):
      seen     URL already in the snapshot: never shown again
      updates  nearest known story closer than KNOWN_STORY_DISTANCE: a candidate update to it
      new      everything else; RELATED_DISTANCE or closer: "related to" that story
    Each article is decided by its OWN nearest distance (feature-1 C: one outlet's boilerplate
    put different stories in one DBSCAN group). New articles keep their check-build groups,
    split where the members' outcomes differ. Returns
      {"seen": [url], "updates": [{story_id, articles, judge, distance}],
       "new": [{key, articles, judge, related_to, distance}], "decisions": [one per article]}
    articles / judge are indices into `articles`; judge = the ones the fact check reads.
    """
    seen_urls = set(snapshot["seen_urls"])
    matrix, row_ids = _known_vectors(snapshot)
    result = {"seen": [], "updates": [], "new": [], "decisions": []}
    updates, new_groups = {}, {}

    for i, article in enumerate(articles):
        if article["url"] in seen_urls:
            result["seen"].append(article["url"])
            result["decisions"].append({"url": article["url"], "outcome": "seen", "story_id": None, "distance": None})
            continue
        story_id, distance = None, None
        if matrix is not None:
            distances = 1.0 - matrix @ _unit(article_vectors[i])
            row = int(np.argmin(distances))
            story_id, distance = row_ids[row], round(float(distances[row]), 4)
        if distance is not None and distance < KNOWN_STORY_DISTANCE:
            outcome = "update"
            updates.setdefault(story_id, []).append(i)
        else:
            related = story_id if distance is not None and distance < RELATED_DISTANCE else None
            outcome = "related" if related else "new"
            theme = article.get("theme_id", -1)
            group = (theme if theme != -1 else f"a{i}", related)     # split a group by outcome
            new_groups.setdefault(group, []).append(i)
        result["decisions"].append({"url": article["url"], "outcome": outcome, "story_id": story_id, "distance": distance})

    distance_of = {i: d["distance"] for i, d in enumerate(result["decisions"])}
    order = [s["id"] for s in snapshot["stories"]]
    for story_id in sorted(updates, key=order.index):
        idxs = updates[story_id]
        result["updates"].append({"story_id": story_id, "articles": idxs, "judge": judged_articles(articles, idxs),
                                  "distance": min(distance_of[i] for i in idxs)})
    groups = sorted(new_groups.items(), key=lambda kv: -_top_similarity(articles, kv[1]))
    for n, ((_, related), idxs) in enumerate(groups, start=1):
        known = [distance_of[i] for i in idxs if distance_of[i] is not None]
        result["new"].append({"key": f"new-{n}", "articles": idxs, "judge": judged_articles(articles, idxs),
                              "related_to": related, "distance": min(known) if known else None})
    return result


# --- Merging a check into the snapshot ---

def _told_text(snapshot, articles, classified, outcomes, texts) -> str:
    """What this check showed, as one text for the next check's "already told"."""
    titles = {s["id"]: s["title"] for s in snapshot["stories"]}
    parts = []
    for item in classified["new"]:
        if outcomes.get(item["key"]) == SHOWN and texts.get(item["key"]):
            title = articles[_by_relevance(articles, item["articles"])[0]].get("title") or "Untitled"
            parts.append(f"New story: {title}\n{texts[item['key']]}")
    for item in classified["updates"]:
        if outcomes.get(item["story_id"]) == SHOWN and texts.get(item["story_id"]):
            parts.append(f"Update on: {titles.get(item['story_id'], item['story_id'])}\n{texts[item['story_id']]}")
    return "\n\n".join(parts)

def merge(snapshot, articles, article_vectors, classified, outcomes: dict, texts: dict, fetched_at: str, now: str) -> dict:
    """
    The snapshot after a check. outcomes: {new key or story id: SHOWN | NOTHING_NEW | UNJUDGED};
    texts: {same key: the summary or new facts shown}. Returns a new dict; the input is unchanged.

      new story, shown or unjudged  -> a known story (s<n>); unjudged ones were shown without a
                                       summary, so they count as seen
      new story, nothing new        -> hidden: its articles join the story it's related to, so
                                       future coverage matches that story; unrelated ones are only
                                       marked seen (no story the user never saw is created)
      update, shown                 -> the story's last_seen, links, outlets, newest vectors
      update, nothing new           -> last_seen and counts only (a repeat adds no new angle)
      update, unjudged              -> nothing: its URLs stay unseen, so the next check retries
    Then: seen URLs, the shown text into "told", caps, vectors pruned after VECTOR_DAYS.
    """
    snap = copy.deepcopy(snapshot)
    by_id = {s["id"]: s for s in snap["stories"]}
    newly_seen = list(classified["seen"])

    for item in classified["new"]:
        outcome = outcomes.get(item["key"], UNJUDGED)
        idxs = item["articles"]
        if outcome in (SHOWN, UNJUDGED):
            _new_story(snap, articles, article_vectors, idxs, now, related_to=item["related_to"])
        elif item["related_to"] in by_id:
            _add_to_story(by_id[item["related_to"]], articles, article_vectors, idxs, now)
        newly_seen += [articles[i]["url"] for i in idxs]

    for item in classified["updates"]:
        outcome = outcomes.get(item["story_id"], UNJUDGED)
        story = by_id.get(item["story_id"])
        if outcome == UNJUDGED or story is None:
            continue
        _add_to_story(story, articles, article_vectors, item["articles"], now, add_vectors=(outcome == SHOWN))
        newly_seen += [articles[i]["url"] for i in item["articles"]]

    _add_unique(snap["seen_urls"], newly_seen)
    snap["seen_urls"] = snap["seen_urls"][-MAX_SEEN_URLS:]

    shown = _told_text(snapshot, articles, classified, outcomes, texts)
    if shown:
        snap["told"] = cap_told(snap["told"] + [{"at": now, "kind": "update", "text": shown}])

    for story in snap["stories"]:
        if story.get("vectors") and _days_between(story["last_seen"], now) > VECTOR_DAYS:
            story["vectors"] = []

    snap["last_fetch_at"] = fetched_at
    snap["last_checked_at"] = now
    snap["checks"] = snap.get("checks", 0) + 1
    return snap


# --- The story log ---

def story_log(snapshot) -> list[dict]:
    """Every story shown for this topic, most recently seen first, without vectors."""
    titles = {s["id"]: s["title"] for s in snapshot["stories"]}
    log = []
    for story in snapshot["stories"]:
        entry = {k: story[k] for k in ("id", "title", "outlets", "links", "first_seen", "last_seen", "articles_seen")}
        if story.get("related_to"):
            entry["related_to"] = {"id": story["related_to"], "title": titles.get(story["related_to"])}
        log.append(entry)
    return sorted(log, key=lambda s: (s["last_seen"], s["first_seen"]), reverse=True)
