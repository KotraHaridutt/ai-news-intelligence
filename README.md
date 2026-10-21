# AI News Intelligence

Turn a topic's news coverage into one grounded briefing. Enter a topic, and the app fetches recent coverage, scrapes the full text, groups duplicate stories, keeps one or two articles per story, and uses retrieval-augmented generation (RAG) over what's left to produce an **intelligence report**, a **timeline**, and an analysis of **where sources disagree**, with links back to every source used. **Follow** a topic, and later checks show only what's new since you last looked: new stories, and new facts about the ones you've already read.

**Live demo:** https://ai-news-intelligence-xdtc.onrender.com
*(Free-tier hosting: the first request after a period of inactivity can take up to a minute while the server wakes up.)*

---

## Why

Following a news topic means reading the same story many times. A search for one topic returns dozens of articles, and many are the same wire report republished by different outlets. Getting the full picture means reading all of them, working out which ones add something new, piecing together the order of events, and noticing where sources contradict each other.

This project automates that work:

- **Groups duplicate coverage** so you see each distinct storyline once.
- **Answers only from fresh, fetched articles**, not from an LLM's training data.
- **Cites its sources**, so every output can be checked against the original articles.
- **Remembers what you've read**, so coming back to a topic shows what changed instead of the whole story again.

## Features

| Feature | Description |
|---|---|
| **Intelligence report** | A multi-paragraph synthesis of the key developments, themes and patterns across all sources, naming the outlet when a claim comes from one source or sources disagree |
| **Timeline** | Key events ordered by date, each with its outlet; relative dates ("on Tuesday") are resolved against the article's publish date |
| **Contradictions** | Conflicting claims and opposing viewpoints, attributed to the outlets (or quoted people) that make them |
| **Related links** | The biggest stories first (with how many articles covered each), then the most relevant unique stories, one link per story |
| **Source citations** | Every response lists the articles its answer was generated from, with their dates |
| **Follow a topic** | "Check for updates" shows only what the report you read (and every check since) didn't contain: new stories with short summaries, new facts on known stories, or "Nothing new". Repeats are hidden but listed, blocked pages are listed, and a story log keeps every story the topic has had |
| **Rate an answer** | 👍 / 👎 under each view and each check, with an optional comment after 👎. The rating goes onto the trace that wrote the answer, and every 👎 lands in a review queue |

In the page, the report, timeline and contradictions are three tabs over the same topic. Timeline and Contradictions load on their first click and are kept until the next search, so switching between views costs nothing. **Follow this topic** under the report adds it to a Following list kept in your browser (no account); a link moves a followed topic to another browser.

---

## How it works

```mermaid
flowchart TD
    A[User query] --> B{Article list for this query<br/>in Redis? 2 h TTL}
    B -- yes --> R[Rebuild the FAISS index from<br/>cached page text + vectors]
    R --> F
    B -- no --> Q[Wait for a build slot<br/>at most 3 builds at once]
    Q --> C[1. Discover articles<br/>GNews: best match, last 48 h]
    C --> D[2. Scrape full text concurrently<br/>only pages not already cached]
    D --> E[3. Group duplicates<br/>Gemini embeddings + DBSCAN]
    E --> CH[3b. Choose<br/>rank by query relevance<br/>max 2 per story, unique stories kept]
    CH --> E2[4. Chunk and index the chosen articles<br/>only uncached chunks are embedded]
    E2 --> CACHE[(Upstash Redis<br/>article list, page text, vectors)]
    E2 --> F[5. Retrieve 10 chunks per view<br/>report: similarity, timeline: date-expanded query,<br/>contradictions: MMR; max 3 per story]
    F --> G[6. Generate with Gemini Flash<br/>each chunk with title, outlet and date,<br/>in a block marked as untrusted data]
    G --> H[JSON response with citations]
```

1. **Discover:** ask the GNews API for the 10 English-language articles that **best match** the query (`sortby=relevance`) among those **published in the last 48 hours** (`from`), with title, URL, source and publish date. GNews's default order is newest first, which filled the 10 slots with articles that only mention the topic in passing; in a live comparison, titles naming the topic went from 3/10 to 10/10 ("Tesla") and from 7/10 to 9/10 ("artificial intelligence"). 10 is the free plan's per-request maximum, and free-plan articles arrive with a 12-hour delay. The free plan also allows 1 request per second, so GNews requests go one at a time, each 1.1 s after the previous one finished.
2. **Scrape:** fetch every article page concurrently with `asyncio.gather` on a shared `httpx.AsyncClient` (max 20 connections, 15 s timeout) that sends a browser `User-Agent`, `Accept` and `Accept-Language`, since many news sites answer a script's default User-Agent with 403. A page whose final status (after redirects) isn't 200, or that isn't HTML, is rejected before parsing, so an "Access Denied" page can never become a source. Trafilatura extracts the article body without navigation, ads or comments, with a BeautifulSoup paragraph fallback. Pages that fail or yield too little text are dropped, and each scrape logs its success rate with failures grouped by reason. Measured on the same 40 real pages: usable articles went from 25 (63%) to 34 (85%), bot blocks (403/400) from 11 to 2, and error pages used as articles from 6 to 0.
3. **Group duplicates:** embed each article with `gemini-embedding-001` and run DBSCAN (cosine distance, `eps=0.12`, `min_samples=2`). Articles carrying the same story (wire copies, pieces on the same event) share a `theme_id`; every other article is a unique story. `eps` was set from measured data: the same wire story in two outlets is 0.000-0.009 apart, pieces on the same event 0.05-0.10, and different stories on the same topic 0.14 or more.
   - **Choose:** embed the query and rank articles by similarity. Keep at most 2 articles per group, from different outlets when possible, and never a second copy of near-identical text. Keep every unique story (relevance scores of on-topic and off-topic articles overlap too much to drop any safely). Only the chosen articles are indexed, so duplicate coverage doesn't crowd the context.
4. **Index:** split the chosen articles with `RecursiveCharacterTextSplitter` (1500 characters, 300 overlap, paragraph-aware) and attach source URL, outlet name (`source_name`), title, publish date, theme ID and the chunk's position in its article (`chunk_index`) to each chunk. Embed the chunks into an in-memory FAISS index.
5. **Retrieve:** each view picks its own 10 chunks. The **report** takes the chunks nearest the query; the **timeline** searches with "what happened and when: dates, days, announcements" added to the query (chosen from 5 measured options: the most date words in the retrieved chunks); **contradictions** uses maximal marginal relevance (MMR, λ 0.5), so a chunk must also differ from those already picked and repeated wire copy gives way to other viewpoints. In every view a story gets at most 3 of the 10 slots, unless the other stories can't fill them. Measured on 8 topics, indexes hold 7-60 chunks, and without that cap one story (two long articles) took 14 of 15 slots ("artificial intelligence"); with it, that topic's contradictions context spans 8 outlets instead of 2. Each answer logs what was used: `RAG [contradictions, mmr]: Retrieved 10 of 22 chunks (k=10, max 3 per story) from 6 articles: "title" [0, 1] | ...`.
6. **Generate:** place the retrieved chunks in one prompt (LangChain's "stuff" strategy) with one of three templates. Each chunk starts with a `[Source: <title> | <outlet> | Published: 2026-10-09 14:30 UTC]` line, so the model can say which outlet reported what and turn "on Tuesday" into a date; in a live comparison the report went from naming 0 to all 5 outlets, and the timeline from 3 to 6 exact dates. The chunks sit in an `<articles>` block that the prompt marks as untrusted data whose instructions must never be followed, and scraped text can't close that block or fake a `[Source: ...]` line (both patterns are rewritten before the prompt is built). Each template instructs the model to use only the provided context and to say so when the context is insufficient. The whole answer step (retrieval and generation) has a 75 s deadline, so a slow model gets a clear "took too long" message before the page gives up.

### Following a topic

```mermaid
flowchart TD
    V1[Report V1 on the page] -- Follow --> S[(Snapshot in Redis<br/>V1's stories and their article vectors<br/>every V1 URL as seen<br/>V1's text as 'already told')]
    C[Check for updates] --> W[GNews: only articles published since the last fetch<br/>minus the 12 h delay, up to 7 days back]
    W --> P[Scrape, embed, group and choose<br/>as for a report, no index]
    P --> D{Each article: seen URL?<br/>distance to the nearest known story}
    S --> D
    D -- seen --> X[never shown again]
    D -- "< 0.12" --> K[update to a known story]
    D -- "0.12–0.18" --> RL[new story, related to a known one]
    D -- "≥ 0.18" --> N[new story]
    K & RL & N --> F[One Gemini call, JSON:<br/>per story, only the facts not already told,<br/>or null = nothing new, hidden]
    F --> M[Merge into the snapshot, save, then show]
```

1. **Follow** saves the report on the page as V1: the stories it was built from, every article in that build as *seen*, and the report's own text (plus any timeline or contradictions opened) as *already told*. No news search and no AI call.
2. **Check for updates** searches only what was published since the previous fetch. GNews's free plan returns articles at least **12 h** old (measured), so the window starts at the previous fetch minus 12 h; it reaches up to **7 days** back (measured), and a longer gap is reported instead of skipped.
3. **Code decides what's new.** An article whose URL was seen is never shown again. Every other article is compared with the stored vectors of the known stories. Measured on 32 hand-labelled articles a day apart: the same storyline was 0.046-0.153 apart, a follow-up 0.107-0.165, a different story 0.196 or more. So below **0.12** an article is an update to a known story, between 0.12 and **0.18** a new story *related* to one, and above that a new story. Each article is decided on its own: one outlet's page boilerplate once put different stories in one group.
4. **The LLM only judges facts.** One structured call per check, and only when there are unseen articles, asks: for each story, what in these articles was *not* already told? A known story gets only its new facts; a new story gets a summary without them; `null` hides it as a repeat. In a live run, an article re-reporting a V1 story a day later came back with only the facts V1 hadn't told; two other outlets' copies of one new event came back hidden or with only their extra facts.
5. **Saved before shown.** The result is stored before it is returned, so nothing shown can come back as "new". A check within 30 minutes returns the stored result; a check with nothing unseen costs no AI call.

### Caching

Steps 1–4 are cached in Redis ([Upstash](https://upstash.com/)), outside the app process, so cached work survives restarts and deploys:

| Key | Holds | TTL |
|---|---|---|
| `v1:articles:<query>` | Article list (metadata and theme IDs), related links and `built_at`, without page text | 2 h |
| `v1:page:<url>` | Scraped article text | 36 h |
| `v1:emb:<model>:<task type>:<sha256 of text>` | Embedding vector (`float32` bytes) | 7 days |
| `v1:since:<hour>:<query>` | A check's article list for a window (shared by checks of a topic in the same hour) | 2 h |
| `v1:answer:<answer_id>` | An answer shown on the page (query, view, markdown, `built_at`), so Follow can store what it said | 2 h |
| `v1:watch:<watch_id>` | A followed topic: its stories (float16 vectors), seen URLs, "already told" text, last result | 30 days after the last check |

- **A cache hit** rebuilds the FAISS index from cached page text and vectors, with no GNews call, scraping or embedding (about 1.5 s in testing, against about 28 s for a full build).
- **Page text and vectors are shared across queries**, keyed by URL and by text hash, so a new query that returns articles seen before only scrapes and embeds what's new. Identical wire copies are embedded once.
- **Queries are normalized** (`"  Tesla   Layoffs "` → `tesla layoffs`) and every TTL varies ±10% so entries don't expire together.
- **Failed builds are never cached** (Gemini errors, all-paywall results), so a temporary failure isn't remembered as "no articles". If only the answer step fails, the finished build stays cached and a retry repeats just the LLM call.
- **Identical concurrent requests share one build**, and at most 3 different builds run at once (`MAX_CONCURRENT_BUILDS`); cache hits don't wait.
- **The server stays responsive during a build:** Gemini calls are async, and parsing, DBSCAN and the FAISS build run in threads. Gemini embedding calls retry rate-limit (429) and server errors with exponential back-off.
- **Redis is optional for reports.** Without `REDIS_URL`, or while Redis is unreachable, every report request runs the full pipeline. Following topics needs Redis: without it, the follow endpoints answer 503 "Following topics is unavailable right now" and everything else works.
- **Followed topics are not a cache.** Their reads and writes report a Redis failure (503) instead of passing silently, because a check that was shown but not saved would show the same news again.

All three endpoints share the cache, so switching from the report to the timeline or contradictions view only costs one LLM call.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic |
| News source | GNews API |
| Scraping | httpx (async), asyncio, Trafilatura, BeautifulSoup |
| Embeddings | Google `gemini-embedding-001` |
| Clustering | scikit-learn DBSCAN, NumPy |
| RAG | LangChain (text splitter, retrieval and stuff-documents chains), FAISS (`faiss-cpu`) |
| Caching | Upstash Redis (`redis` asyncio client) |
| LLM | Google Gemini Flash (`gemini-3.8-flash`, falling back to `gemini-3.6-flash`), thinking level `low`, default temperature, 45 s timeout per attempt, 75 s deadline for the whole answer step; the follow check uses structured output (JSON schema, Pydantic) |
| Frontend | Jinja2 template, Tailwind CSS, vanilla JavaScript; followed topics in `localStorage` |
| Observability | LangSmith: traces and spans, tokens and cost, prompt versions, eval datasets and experiments, online checks, user feedback, dashboards; alerts to ntfy.sh |
| Deployment | Render (free tier) |

---

## API

The three views accept a JSON body `{"query": "<topic>"}`. Interactive documentation is available at `/docs`.

### `POST /query`: intelligence report

```json
{
  "summary_html": "<p>The European Union's AI Act ...</p>",
  "cited_sources": [
    {"title": "EU passes landmark AI law", "url": "https://...", "published_at": "2026-09-30T14:22:00Z"}
  ],
  "related_links": [
    {"title": "EU passes landmark AI law", "url": "https://...", "source": "Reuters"}
  ],
  "total_articles_found": 24,
  "answer_id": "Xy3kQ0…",
  "time_taken": "8.41s"
}
```

`answer_id` names this answer for **Follow** and for rating it (`null` if it couldn't be saved; the answer is still shown). `can_rate` says whether 👍 / 👎 can be saved for it (only when its trace was sent to LangSmith).

### `POST /api/timeline`: chronological events

Returns `query`, `timeline_html`, `cited_sources`, `answer_id` and `time_taken`. The page's Timeline tab.

### `POST /api/contradictions`: conflicting claims across sources

Returns `query`, `analysis_html`, `cited_sources`, `answer_id` and `time_taken`. The page's Contradictions tab.

### Following a topic

| Endpoint | Does | Calls GNews / Gemini |
|---|---|---|
| `POST /api/follow` `{"query", "answer_ids": {"report", "timeline"?, "contradictions"?}}` | Saves the report on the page as V1. Returns `watch_id`, `query`, `created_at`, `stories` | no / no |
| `POST /api/follow/{watch_id}/check` | What's new since the last check (the stored result within 30 min) | 1 request (0 if shared or reused) / 0 or 1 call |
| `GET /api/follow/{watch_id}` | The story log and the last result | no / no |
| `DELETE /api/follow/{watch_id}` | Stop following | no / no |

### `POST /api/feedback`: rate an answer

`{"answer_id": "<from the answer>", "score": 1 or -1, "comment": "optional, up to 1,000 characters"}` returns `{"saved": true, "score": 1}`. One rating per answer: rating again replaces it. Answers can be rated for 2 hours. A check's update has its own `answer_id`.

A check returns:

```json
{
  "watch_id": "m7Mo_3_ZiCUYl4Ce6-nQMA", "query": "Tesla", "checked_at": "…", "previous_check": "…",
  "window": {"from": "2026-10-11T16:00:00Z", "to_approx": "2026-10-12T16:42:10Z", "gap_not_covered": null},
  "articles_found": 10,
  "new_stories": [{"story_id": "s6", "title": "…", "outlets": ["Business Insider"], "articles": 1,
                   "summary_html": "<p>…</p>", "related_to": null, "links": [{"title": "…", "url": "…", "source_name": "…", "published_at": "…"}]}],
  "updates": [{"story_id": "s2", "story_title": "…", "first_seen": "…", "new_facts_html": "<p>…</p>", "links": []}],
  "hidden_repeats": [{"title": "…", "url": "…", "source_name": "…", "published_at": "…"}],
  "unreadable": [],
  "nothing_new": false, "message": null, "reused": false, "time_taken": "7.30s"
}
```

**Errors:** every failure returns a real error status and `{"detail": "<message for the user>"}`, never `200` with error text. The page shows `detail` as it is.

| Status | When | Example `detail` |
|---|---|---|
| `400` | Empty topic, or a search GNews can't parse (e.g. an unbalanced quote) | Please enter a topic first. |
| `404` | No news in the last 48 hours, or none of the found pages could be read | No news from the last 2 days was found for "Tesla". Try a broader or different topic. |
| `422` | Malformed request body | The request was invalid. Please enter a topic and try again. |
| `500` | A missing or invalid GNews/Gemini key, or a bug | The news service isn't set up correctly on the server. Please contact the site owner. |
| `502` | GNews or Gemini answered with an error, or the AI returned an empty answer | The AI couldn't write the report. Please try again. |
| `503` | Temporary: GNews daily quota, rate limit or maintenance; Gemini busy or rate-limited | Today's news search limit has been reached. It resets at midnight UTC, so please try again later. |
| `504` | Gemini took too long (embedding, or the 75 s answer deadline) | The AI took too long to write the report. Please try again. |
| `409` | Follow: the report's article list expired or was rebuilt since the report (or the answer belongs to another topic) | This report is more than 2 hours old. Analyze the topic again, then follow it. |
| `404` | A followed topic that expired (30 days without a check) or an incomplete link | This followed topic has expired after 30 days without a check, or the link is incomplete. Follow it again to start a new history. |
| `503` | Following without Redis; or a check that worked but couldn't be saved (it isn't shown) | Following topics is unavailable right now. You can still analyze topics as usual. |
| `422` / `404` / `409` / `503` | Rating: an unreadable rating or a comment over 1,000 characters; an answer older than 2 hours; an answer whose trace wasn't sent; Redis, tracing or LangSmith unavailable | Couldn't save your feedback right now. Please try again in a minute. |

With tracing on, every error also carries `"ref"`: the end of its trace id, shown on the page as "(Ref: 1a2b3c4d)" so a reported error can be found in LangSmith.

All messages live in [`services/errors.py`](services/errors.py). The page has its own messages for what the server can't answer: no connection, a proxy error while the server wakes up, and no answer after 120 s.

---

## Getting started

### Prerequisites

- Python 3.12
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- A [GNews](https://gnews.io/) API key

### Setup

```bash
git clone https://github.com/KotraHaridutt/ai-news-intelligence.git
cd ai-news-intelligence

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
GNEWS_API_KEY=your_gnews_api_key
REDIS_URL=rediss://default:your_password@your-db.upstash.io:6379
```

`REDIS_URL` is optional. Without it the app runs with caching disabled, so every request runs the full pipeline. Upstash shows the URL as `redis://...`; use `rediss://` (TLS), which Upstash requires.

Optional: `MAX_CONCURRENT_BUILDS` (default `3`) limits how many new queries are built at the same time, and `GEMINI_CHAT_MODEL` (default `gemini-3.8-flash`) and `GEMINI_FALLBACK_MODEL` (default `gemini-3.6-flash`) switch the answer models without a code change (Gemini 3+ models are required). The fallback is used only when the main model is overloaded or times out. Tracing to LangSmith is optional too: see [Observability](#observability) and `.env.example`.

### Run

```bash
uvicorn server:app --reload
```

Open http://localhost:8000 for the UI, or http://localhost:8000/docs to try every endpoint.

---

## Deployment

The repository includes a `render.yaml` blueprint and a `Procfile`. On Render, create a web service from the repository, set `GOOGLE_API_KEY`, `GNEWS_API_KEY` and `REDIS_URL` (an [Upstash](https://upstash.com/) Redis URL) as environment variables, and deploy. The start command is:

```bash
uvicorn server:app --host 0.0.0.0 --port $PORT
```

The free tier has 512 MB of RAM, so embeddings come from the Gemini API instead of a local Sentence-Transformers model. PyTorch alone would use most of the available memory.

For tracing in production, also set `LANGSMITH_API_KEY` (`render.yaml` asks for it and sets `LANGSMITH_TRACING=true`, `APP_ENV=prod`, project `ai-news-prod`). After the first traced request, run `scripts/setup_review_queue.py --project ai-news-prod` and `scripts/setup_monitoring.py --project ai-news-prod`, then create the three alerts it prints in the LangSmith UI.

---

## Observability

The app is traced end to end with [LangSmith](https://smith.langchain.com) (feature 2; how each part was built and measured is in `implemented/features/feature-2/docs/`, sections A-L).

| What | How |
|---|---|
| **Traces** | One trace per request, named after its route (`POST /query`, ...), with spans for each step: GNews search, scraping, grouping, indexing, retrieval, every chat attempt (retries and fallback visible), the follow check's novelty steps and fact check. A followed topic's checks share a thread. Errors carry their status, the user's message and a `ref`. |
| **Privacy** | API keys, tokens and Redis passwords are masked before anything leaves the process; long texts are trimmed; watch ids and answer ids leave the server only as hashes. Tracing runs on background threads and never fails or slows a request (measured in prod: +1 ms median). |
| **Tokens and cost** | Every chat attempt's input, output and thinking tokens and its cost; embeddings with estimated tokens. Per request: `cost_usd`, `chat_calls`, `answered_by` (primary or fallback). A Redis counter per model per Pacific day mirrors Gemini's free quota (20 a day), and a model out of quota is skipped until midnight Pacific. |
| **Versions** | Every prompt has a version (a hash of its text) and the settings a `config_version`; both are on every trace and saved answer. Prompts are mirrored to the LangSmith Prompt Hub (`scripts/push_prompts.py`). |
| **Evals** | Frozen eval data and 4 private datasets (retrieval, answers, fact check, prompt injection), free code checks, retrieval scores, per-sentence LLM judges with calibration against reviewed labels, and a resumable experiment runner inside the daily budget (`evals/run_evals.py`: run, compare, pairwise, thresholds). |
| **Online checks** | The free code checks run on every live answer and every fact check, as feedback on its trace; `evals/judge_yesterday.py` runs the LLM judge on a day's real answers by hand. |
| **User feedback** | 👍 / 👎 on the answer's trace (`user_rating`); 👎 and failed checks go to the `ai-news-review` annotation queue. |
| **Dashboards and alerts** | Three dashboards (usage and cost, speed and errors, quality: 27 charts) built by `scripts/setup_monitoring.py`, which also keeps LangSmith's price map equal to ours. Alerts (error rate, slow answers, unreadable fact check) go to a phone through ntfy.sh. |

Turn it on locally with `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true` in `.env` (project `ai-news-dev`). Without them the app runs exactly the same, untraced.

```bash
# evals (each counts against the daily Gemini budget; the runner stops before it runs out)
python evals/run_evals.py status
python evals/run_evals.py run retrieval                  # no chat calls
python evals/run_evals.py run answers --judge --set-baseline
python evals/run_evals.py compare <experiment-a> <experiment-b> --pairwise
python evals/judge_yesterday.py --dry-run                # what yesterday's real answers would cost to judge
```

**Free-plan limits that shape it:** Gemini allows 20 chat calls a day per model, so evals are resumable across days and LLM judging of live traffic is run by hand. LangSmith keeps traces 14 days on the free plan; worth-keeping examples go to datasets. LangSmith alerts can't email, so they go to ntfy.sh.

---

## Project structure

```
ai-news-intelligence/
├── server.py                    # FastAPI app and endpoints
├── services/
│   ├── pipeline_service.py      # Pipeline: cache hit/miss, single-flight, build limit; the check build for followed topics
│   ├── cache_service.py         # Redis keys, TTLs, batched reads/writes; answer records and followed topics
│   ├── news_fetcher.py          # Article discovery (GNews), the check window, concurrent scraping
│   ├── errors.py                # AppError: one status + user-facing message per failure point
│   ├── clustering_service.py    # DBSCAN theme clustering
│   ├── selection_service.py     # Rank by relevance, keep ≤ 2 articles per story, Related Links
│   ├── rag_service.py           # Chunking, cached embeddings, FAISS index, prompts, retrieval per view
│   ├── novelty_service.py       # Following: a topic's snapshot; seen / update / related / new; merge; story log
│   ├── fact_check_service.py    # Following: the one structured LLM call that keeps only facts not already told
│   ├── follow_service.py        # Following: Follow, Check, story log, unfollow
│   ├── observability.py         # LangSmith: setup, redaction, spans, request totals, feedback queue
│   ├── usage.py                 # Tokens, cost, the daily Gemini quota counter
│   ├── prompts.py               # The prompts, their versions, the Prompt Hub mirror
│   ├── checks.py                # Free code checks on answers, retrieval and fact checks
│   ├── online_checks.py         # Those checks on every live answer, as trace feedback
│   └── feedback_service.py      # 👍 / 👎 on an answer's trace, the review queue
├── evals/                       # Eval data, datasets, judges, calibration, the experiment runner, judge_yesterday
├── scripts/                     # Prompt Hub mirror, review queue, dashboards and alerts setup
├── templates/index.html         # Web UI: report / timeline / contradictions tabs, Following list, update view
├── static/style.css
├── render.yaml                  # Render deployment blueprint
├── Procfile
└── requirements.txt
```

---

## Design decisions

- **Query-time RAG instead of a persistent vector database.** Each query builds its own small index (7-60 chunks from the chosen articles, median 22.5, measured on 8 topics) from fresh articles, so the corpus is always current. An exact flat FAISS index is fast at this size, and an external vector database would add cost without benefit.
- **DBSCAN instead of K-Means.** The number of storylines per topic is unknown in advance, and DBSCAN leaves one-off stories unassigned instead of forcing them into a cluster. Its `eps` (0.12) comes from measuring 133 real article pairs, not from a default.
- **Deduplicate before indexing, not after retrieval.** Grouping and choosing happen before chunks are embedded, so skipped copies cost no embedding calls and can't take retrieval slots. In a real query this cut the index from 35 to 21 chunks and Related Links from 1 to 5 distinct stories.
- **Cap chunks per story, not only articles.** Keeping at most 2 articles per story still let one long article fill most of the context, so retrieval also limits each story to 3 of the 10 chunks an answer gets. The cap is soft: a topic with only one or two stories still gets 10 chunks.
- **Each view retrieves for its own question.** A report wants the most relevant chunks, a timeline wants chunks that date events, and a contradictions analysis wants different viewpoints, so the three use different searches over the same index instead of sharing one result.
- **Scraped text is data, never instructions.** Article text goes into the prompt inside a marked block, after a rule not to follow anything written inside it. The model has no tools or secrets and the answer is HTML-escaped, so the risk is a misleading answer; the block and the rewriting of fake delimiters and headers make that harder.
- **The "stuff" strategy instead of map-reduce.** Finding contradictions requires the model to see all sources at once. Map-reduce processes documents separately and would miss conflicts between them.
- **Cache the inputs, not the index.** Redis stores page text and embedding vectors, and the FAISS index is rebuilt from them on each request. Storing the FAISS object itself would need `pickle` (unsafe to load, tied to library versions), while rebuilding the index from 7-60 vectors takes milliseconds. Keying text and vectors by URL and content hash lets different queries share the expensive work.
- **Errors are raised, not returned.** Each failure point raises one error type carrying its HTTP status and a user-facing message, and one handler turns it into the response. Before, `None` and `[]` were passed up through three layers that each had to guess what they meant, and five different failures reached the user as the same "no articles" 404.
- **Paragraph-aware chunking.** News articles are plain prose, so splitting at paragraph boundaries keeps each claim together with its context without the extra cost of semantic chunking.
- **What's new is decided by code; the LLM only judges facts.** Whether an article was seen, and which story it belongs to, comes from stored URLs and embedding distances with thresholds measured across days, not from the model. The model is asked one narrower question (what in these articles wasn't already told?) and answers in JSON, so "nothing new" is a `null` the code acts on. A general assistant asked to "summarize what's new" doesn't know exactly what you read.
- **Follow by answer id, not by topic.** Every answer on the page has its own id, and Follow saves exactly those answers plus the article list they were built from (checked by its build time). Two people reading the same topic never get each other's report as "already told", and a list rebuilt after the report is refused instead of silently paired with it.
- **Decide each article on its own.** A day-later re-report of a known story measured 0.153 from it, and one outlet's page boilerplate grouped two different stories at 0.10-0.12; deciding per article (and splitting groups whose articles disagree) keeps a new story from being labelled as an old one.
- **Save before show.** A check's result is stored before it is returned; if storing fails, it isn't shown. Anything shown is therefore never shown again as new.

## Limitations and roadmap

- [x] Add timeline and contradictions views to the UI (tabs over the same topic)
- [x] Include source and date metadata in the LLM prompt so contradictions and timeline events are attributed to specific outlets
- [x] Give each view its own retrieval (similarity, date-expanded query, MMR) and cap chunks per story
- [x] Keep duplicate coverage from dominating the context (choose step: at most 2 articles per story before indexing)
- [x] Make embedding, indexing and LLM calls fully non-blocking (`ainvoke`, `asyncio.to_thread`)
- [x] Replace the in-process cache with a shared cache (Redis) that survives restarts
- [x] Follow a topic and see only what changed since the last look
- [x] Observability: traces, cost, versions, evals, online checks, user feedback, dashboards and alerts (LangSmith)
- [x] Automated faithfulness check: per-sentence LLM judges with calibration (the calibration run waits for a healthy Gemini quota day)
- [ ] Add inline per-claim citations

Limits of following, on the free plans:

- **"Latest" is at least 12 hours old:** the GNews free plan returns articles with a 12-hour delay, and a check shows the window it covered.
- **10 articles per check:** a busy topic can have more new articles than one search returns; the check shows the 10 best matches.
- **No account:** followed topics live in the browser; a link moves one to another browser, and a topic expires after 30 days without a check.
- **20 AI calls a day per model** on the Gemini free tier, shared by reports and checks (a check with nothing unseen uses none).

## License

[MIT](LICENSE)
