# AI News Intelligence

Turn a topic's news coverage into one grounded briefing. Enter a topic, and the app fetches recent coverage, scrapes the full text, groups duplicate stories, keeps one or two articles per story, and uses retrieval-augmented generation (RAG) over what's left to produce an **intelligence report**, a **timeline**, and an analysis of **where sources disagree**, with links back to every source used.

**Live demo:** https://ai-news-intelligence-xdtc.onrender.com
*(Free-tier hosting: the first request after a period of inactivity can take up to a minute while the server wakes up.)*

---

## Why

Following a news topic means reading the same story many times. A search for one topic returns dozens of articles, and many are the same wire report republished by different outlets. Getting the full picture means reading all of them, working out which ones add something new, piecing together the order of events, and noticing where sources contradict each other.

This project automates that work:

- **Groups duplicate coverage** so you see each distinct storyline once.
- **Answers only from fresh, fetched articles**, not from an LLM's training data.
- **Cites its sources**, so every output can be checked against the original articles.

## Features

| Feature | Description |
|---|---|
| **Intelligence report** | A multi-paragraph synthesis of the key developments, themes and patterns across all sources |
| **Timeline** | Key events extracted and ordered by date |
| **Contradictions** | Conflicting claims and opposing viewpoints between sources |
| **Related links** | The biggest stories first (with how many articles covered each), then the most relevant unique stories, one link per story |
| **Source citations** | Every response lists the articles its answer was generated from |

---

## How it works

```mermaid
flowchart TD
    A[User query] --> B{Article list for this query<br/>in Redis? 2 h TTL}
    B -- yes --> R[Rebuild the FAISS index from<br/>cached page text + vectors]
    R --> F
    B -- no --> Q[Wait for a build slot<br/>at most 3 builds at once]
    Q --> C[1. Discover articles<br/>GNews API]
    C --> D[2. Scrape full text concurrently<br/>only pages not already cached]
    D --> E[3. Group duplicates<br/>Gemini embeddings + DBSCAN]
    E --> CH[3b. Choose<br/>rank by query relevance<br/>max 2 per story, unique stories kept]
    CH --> E2[4. Chunk and index the chosen articles<br/>only uncached chunks are embedded]
    E2 --> CACHE[(Upstash Redis<br/>article list, page text, vectors)]
    E2 --> F[5. Retrieve top-15 chunks]
    F --> G[6. Generate with Gemini Flash<br/>report / timeline / contradictions]
    G --> H[JSON response with citations]
```

1. **Discover:** query the GNews API for up to 10 recent English-language articles (title, URL, source, publish date). 10 is the free plan's per-request maximum, and free-plan articles arrive with a 12-hour delay.
2. **Scrape:** fetch every article page concurrently with `asyncio.gather` on a shared `httpx.AsyncClient` (max 20 connections, 15 s timeout). Trafilatura extracts the article body without navigation, ads or comments, with a BeautifulSoup paragraph fallback. Pages that fail or yield too little text are dropped.
3. **Group duplicates:** embed each article with `gemini-embedding-001` and run DBSCAN (cosine distance, `eps=0.12`, `min_samples=2`). Articles carrying the same story (wire copies, pieces on the same event) share a `theme_id`; every other article is a unique story. `eps` was set from measured data: the same wire story in two outlets is 0.000-0.009 apart, pieces on the same event 0.05-0.10, and different stories on the same topic 0.14 or more.
   - **Choose:** embed the query and rank articles by similarity. Keep at most 2 articles per group, from different outlets when possible, and never a second copy of near-identical text. Keep every unique story (relevance scores of on-topic and off-topic articles overlap too much to drop any safely). Only the chosen articles are indexed, so duplicate coverage doesn't crowd the context.
4. **Index:** split the chosen articles with `RecursiveCharacterTextSplitter` (1500 characters, 300 overlap, paragraph-aware) and attach source URL, title, publish date and theme ID to each chunk. Embed the chunks into an in-memory FAISS index.
5. **Retrieve:** embed the query and fetch the 15 nearest chunks.
6. **Generate:** place the retrieved chunks in one prompt (LangChain's "stuff" strategy) with one of three templates. Each template instructs the model to use only the provided context and to say so when the context is insufficient.

### Caching

Steps 1–4 are cached in Redis ([Upstash](https://upstash.com/)), outside the app process, so cached work survives restarts and deploys:

| Key | Holds | TTL |
|---|---|---|
| `v1:articles:<query>` | Article list (metadata and theme IDs) and related links, without page text | 2 h |
| `v1:page:<url>` | Scraped article text | 36 h |
| `v1:emb:<model>:<task type>:<sha256 of text>` | Embedding vector (`float32` bytes) | 7 days |

- **A cache hit** rebuilds the FAISS index from cached page text and vectors, with no GNews call, scraping or embedding (about 1.5 s in testing, against about 28 s for a full build).
- **Page text and vectors are shared across queries**, keyed by URL and by text hash, so a new query that returns articles seen before only scrapes and embeds what's new. Identical wire copies are embedded once.
- **Queries are normalized** (`"  Tesla   Layoffs "` → `tesla layoffs`) and every TTL varies ±10% so entries don't expire together.
- **Failed builds are never cached** (Gemini errors, all-paywall results), so a temporary failure isn't remembered as "no articles".
- **Identical concurrent requests share one build**, and at most 3 different builds run at once (`MAX_CONCURRENT_BUILDS`); cache hits don't wait.
- **The server stays responsive during a build:** Gemini calls are async, and parsing, DBSCAN and the FAISS build run in threads. Gemini embedding calls retry rate-limit (429) and server errors with exponential back-off.
- **Redis is optional.** Without `REDIS_URL`, or while Redis is unreachable, every request runs the full pipeline.

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
| LLM | Google Gemini Flash (`gemini-3.8-flash`, falling back to `gemini-3.6-flash`), thinking level `low`, default temperature, 45 s timeout per attempt |
| Frontend | Jinja2 template, Tailwind CSS, vanilla JavaScript |
| Deployment | Render (free tier) |

---

## API

All endpoints accept a JSON body `{"query": "<topic>"}`. Interactive documentation is available at `/docs`.

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
  "time_taken": "8.41s"
}
```

### `POST /api/timeline`: chronological events

Returns `query`, `timeline_html`, `cited_sources` and `time_taken`.

### `POST /api/contradictions`: conflicting claims across sources

Returns `query`, `analysis_html`, `cited_sources` and `time_taken`.

**Errors:** `400` for an empty query, `404` when no articles with usable text are found, `500` when fetching fails.

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

Optional: `MAX_CONCURRENT_BUILDS` (default `3`) limits how many new queries are built at the same time, and `GEMINI_CHAT_MODEL` (default `gemini-3.8-flash`) and `GEMINI_FALLBACK_MODEL` (default `gemini-3.6-flash`) switch the answer models without a code change (Gemini 3+ models are required). The fallback is used only when the main model is overloaded or times out.

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

---

## Project structure

```
ai-news-intelligence/
├── server.py                    # FastAPI app and endpoints
├── services/
│   ├── pipeline_service.py      # Pipeline: cache hit/miss, single-flight, build limit
│   ├── cache_service.py         # Redis keys, TTLs, batched reads/writes
│   ├── news_fetcher.py          # Article discovery (GNews) and concurrent scraping
│   ├── clustering_service.py    # DBSCAN theme clustering
│   └── rag_service.py           # Chunking, cached embeddings, FAISS index, prompts, retrieval chain
├── templates/index.html         # Web UI
├── static/style.css
├── render.yaml                  # Render deployment blueprint
├── Procfile
└── requirements.txt
```

---

## Design decisions

- **Query-time RAG instead of a persistent vector database.** Each query builds its own small index (about 30-40 chunks from up to 10 articles) from fresh articles, so the corpus is always current. An exact flat FAISS index is fast at this size, and an external vector database would add cost without benefit.
- **DBSCAN instead of K-Means.** The number of storylines per topic is unknown in advance, and DBSCAN leaves one-off stories unassigned instead of forcing them into a cluster. Its `eps` (0.12) comes from measuring 133 real article pairs, not from a default.
- **Deduplicate before indexing, not after retrieval.** Grouping and choosing happen before chunks are embedded, so skipped copies cost no embedding calls and can't take retrieval slots. In a real query this cut the index from 35 to 21 chunks and Related Links from 1 to 5 distinct stories.
- **The "stuff" strategy instead of map-reduce.** Finding contradictions requires the model to see all sources at once. Map-reduce processes documents separately and would miss conflicts between them.
- **Cache the inputs, not the index.** Redis stores page text and embedding vectors, and the FAISS index is rebuilt from them on each request. Storing the FAISS object itself would need `pickle` (unsafe to load, tied to library versions), while rebuilding the index from 30-40 vectors takes milliseconds. Keying text and vectors by URL and content hash lets different queries share the expensive work.
- **Paragraph-aware chunking.** News articles are plain prose, so splitting at paragraph boundaries keeps each claim together with its context without the extra cost of semantic chunking.

## Limitations and roadmap

- [ ] Add timeline and contradictions views to the UI (currently available through the API and `/docs`)
- [ ] Include source and date metadata in the LLM prompt so contradictions and timeline events are attributed to specific outlets
- [x] Keep duplicate coverage from dominating the context (choose step: at most 2 articles per story before indexing)
- [x] Make embedding, indexing and LLM calls fully non-blocking (`ainvoke`, `asyncio.to_thread`)
- [x] Replace the in-process cache with a shared cache (Redis) that survives restarts
- [ ] Add inline per-claim citations and an automated faithfulness check

## License

[MIT](LICENSE)
