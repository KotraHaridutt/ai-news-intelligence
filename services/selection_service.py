import numpy as np

# --- gap_1: decide which articles reach the index and the Related Links ---
# Pure functions (no I/O), called by pipeline_service between grouping and indexing.
# Decisions D1-D4 (2026-10-11); D1 and D2 were refined after measuring real data.

MAX_PER_GROUP = 2               # D1: at most 2 articles per group of duplicates
NEAR_IDENTICAL_DISTANCE = 0.02  # D1: don't keep a second copy of the same text
                                #     (measured: wire copies 0.000-0.009, distinct pieces 0.053+)
MAX_UNIQUE = 10                 # D4: cap on unique stories kept, most relevant first
RELATED_LINKS_TOTAL = 5
RELATED_GROUP_SLOTS = 3         # D3: 3 slots for the biggest groups, the rest for unique stories


def _unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm else v

def _outlet(article):
    return (article.get('source_name') or "").strip().lower()

def _rank_order(count, similarities):
    """
    Article indices from most to least relevant to the query.
    Without similarities, GNews's own order (its relevance ranking) is used.
    """
    if similarities is None:
        return list(range(count))
    return sorted(range(count), key=lambda i: -similarities[i])

def cosine_similarities(article_vectors, query_vector):
    """Cosine similarity of each article vector (RETRIEVAL_DOCUMENT) to the query vector (RETRIEVAL_QUERY)."""
    q = _unit(query_vector)
    return [float(_unit(v) @ q) for v in article_vectors]


def promote_unique(articles, similarities):
    """
    D2 (revised with real data): every article DBSCAN left ungrouped (-1) is a unique story.
    It gets its own theme id and unique=True. Nothing is dropped by similarity: on real
    queries, relevance scores of on-topic and off-topic articles overlap (0.58-0.71), so any
    threshold dropped relevant stories. D4 caps how many are kept, most relevant first;
    any beyond the cap stay -1 and are not indexed. Returns the number promoted.
    """
    for article in articles:
        article['unique'] = False

    next_theme = max((article['theme_id'] for article in articles), default=-1) + 1
    noise = [i for i in _rank_order(len(articles), similarities) if articles[i]['theme_id'] == -1]

    for i in noise[:MAX_UNIQUE]:
        articles[i]['theme_id'] = next_theme
        articles[i]['unique'] = True
        next_theme += 1
    return len(noise[:MAX_UNIQUE])


def choose_articles(articles, vectors, similarities):
    """
    D1 (refined): for each group of duplicates keep up to MAX_PER_GROUP articles, most
    relevant first, from different outlets when possible (so the contradictions view has
    two independent sources), and never a second article whose text is near-identical to
    one already kept. Unique stories are always kept. Sets article['indexed'];
    returns the number of indexed articles.
    """
    units = [_unit(v) for v in vectors] if vectors is not None else None

    def near_identical(i, j):
        return units is not None and 1.0 - float(units[i] @ units[j]) < NEAR_IDENTICAL_DISTANCE

    for article in articles:
        article['indexed'] = False

    groups = {}
    for i in _rank_order(len(articles), similarities):
        theme = articles[i]['theme_id']
        if theme != -1:
            groups.setdefault(theme, []).append(i)

    for members in groups.values():
        kept = []
        # First pass prefers a new outlet; the second fills from any outlet if still short
        for prefer_new_outlet in (True, False):
            for i in members:
                if len(kept) >= MAX_PER_GROUP:
                    break
                if i in kept:
                    continue
                if prefer_new_outlet and any(_outlet(articles[i]) == _outlet(articles[k]) for k in kept):
                    continue
                if any(near_identical(i, k) for k in kept):
                    continue
                kept.append(i)
        for i in kept:
            articles[i]['indexed'] = True

    return sum(1 for article in articles if article['indexed'])


def build_related_links(articles, similarities):
    """
    D3: up to 3 links for the biggest groups (most articles first, then most relevant),
    then the most relevant unique stories; if either kind is short, the other fills the
    remaining slots (5 in total). Each link is its group's most relevant indexed article.
    """
    order = _rank_order(len(articles), similarities)
    rank = {i: position for position, i in enumerate(order)}

    themes = {}
    for i in order:
        theme = articles[i]['theme_id']
        if theme != -1:
            themes.setdefault(theme, []).append(i)

    def representative(members):
        return next((i for i in members if articles[i].get('indexed')), members[0])

    groups = sorted((m for m in themes.values() if not articles[m[0]].get('unique')),
                    key=lambda m: (-len(m), rank[m[0]]))
    uniques = sorted((m for m in themes.values() if articles[m[0]].get('unique')),
                     key=lambda m: rank[m[0]])

    group_slots = min(RELATED_GROUP_SLOTS, len(groups))
    unique_slots = min(RELATED_LINKS_TOTAL - group_slots, len(uniques))
    group_slots = min(len(groups), RELATED_LINKS_TOTAL - unique_slots)   # groups fill unused unique slots

    links = []
    for members in groups[:group_slots] + uniques[:unique_slots]:
        art = articles[representative(members)]
        links.append({
            "title": art.get("title"),
            "url": art.get("url"),
            "source": art.get("source_name"),
            "articles_in_group": len(members),
        })
    return links


def selection_report(articles, similarities):
    """
    Summary line plus one line per article, logged on every build: the data for tuning
    DBSCAN_EPS and the relevance rules later.
    """
    sizes = {}
    for article in articles:
        if article['theme_id'] != -1 and not article.get('unique'):
            sizes[article['theme_id']] = sizes.get(article['theme_id'], 0) + 1
    unique = sum(1 for a in articles if a.get('unique'))
    dropped = sum(1 for a in articles if a['theme_id'] == -1)
    indexed = sum(1 for a in articles if a.get('indexed'))

    summary = (f"{len(articles)} articles: {len(sizes)} group(s) holding {sum(sizes.values())}, "
               f"{unique} unique, {dropped} over the unique cap; {indexed} indexed")
    if similarities:
        summary += (f"; query similarity {min(similarities):.3f}-{max(similarities):.3f} "
                    f"(median {float(np.median(similarities)):.3f})")

    lines = [summary]
    for i in _rank_order(len(articles), similarities):
        a = articles[i]
        kind = "unique" if a.get('unique') else ("dropped" if a['theme_id'] == -1 else f"group {a['theme_id']}")
        sim = f"{similarities[i]:.3f}" if similarities else "  -  "
        lines.append(f"  {sim} {kind:9} {'indexed' if a.get('indexed') else 'skipped':7} "
                     f"[{(a.get('source_name') or '')[:18]}] {(a.get('title') or '')[:70]}")
    return lines
