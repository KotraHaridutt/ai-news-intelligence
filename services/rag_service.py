import os
import asyncio
import random
import numpy as np
from dotenv import load_dotenv
from google.genai import errors as genai_errors

from services import cache_service, errors
from services.news_fetcher import has_enough_text

# --- LangChain Imports ---
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()
if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    print("[ERROR] GEMINI_API_KEY/GOOGLE_API_KEY not found. Please set it in your .env file.")

EMBEDDING_MODEL = "gemini-embedding-001"
# LangChain embeds documents as RETRIEVAL_DOCUMENT and queries as RETRIEVAL_QUERY,
# which give different vectors for the same text, so the task type is part of the cache key.
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
EMB_CACHE_MODEL = f"{EMBEDDING_MODEL}:{DOCUMENT_TASK_TYPE}"
# The query is embedded as RETRIEVAL_QUERY (the side of the pair that is compared against documents)
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
EMB_QUERY_CACHE_MODEL = f"{EMBEDDING_MODEL}:{QUERY_TASK_TYPE}"

# Chat model for the report / timeline / contradictions answers.
# gemini-3.8-flash is Google's current Flash model (2.5 Flash is two generations old).
# - No temperature: Google recommends the default (1.0) for Gemini 3+ models; lower values
#   can cause looping. LangChain sends no temperature for Gemini 3+ when it isn't set.
# - thinking_level="low": the task is writing from the retrieved context, not multi-step
#   reasoning, and the default ("high") adds reasoning tokens and latency.
# - Timeout + fallback: a new model can be overloaded (measured: 503 "high demand", one call
#   took 89 s, another 190 s). Each attempt is cut off after CHAT_TIMEOUT_SECONDS; the Google SDK
#   retries 429/5xx and timeouts (CHAT_ATTEMPTS attempts per model), and if the primary model
#   still fails, the request falls back to a stable older Flash model.
# GEMINI_CHAT_MODEL / GEMINI_FALLBACK_MODEL override the models without a code change
# (thinking_level needs Gemini 3+).
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
FALLBACK_CHAT_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
THINKING_LEVEL = "low"
CHAT_TIMEOUT_SECONDS = 45     # per attempt (normal answers take ~10-30 s)
CHAT_ATTEMPTS = 2             # per model, including the first try
# Deadline for the whole answer step (retrieval + every attempt on both models). Without it
# the worst case is ~2 models x 2 attempts x 45 s = 3 min, longer than the page waits (120 s),
# so the user would see "took too long" instead of the real reason. A fast failure such as a
# 429 still falls back within seconds; a model that hangs uses up the deadline.
CHAT_DEADLINE_SECONDS = 75

def _chat_model(model_name):
    return ChatGoogleGenerativeAI(
        model=model_name,
        thinking_level=THINKING_LEVEL,
        timeout=CHAT_TIMEOUT_SECONDS,
        max_retries=CHAT_ATTEMPTS,
    )

# Initialize Models (the one embeddings client used by clustering and indexing)
# The chat models retry through the Google SDK (see above).
# The embeddings client has no retries, so embedding calls go through call_with_retry below.
try:
    llm = _chat_model(CHAT_MODEL).with_fallbacks([_chat_model(FALLBACK_CHAT_MODEL)])
    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("[INFO] RAG models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load RAG models: {e}")
    llm = None
    embeddings_model = None

# PROMPTS
REPORT_PROMPT_TEMPLATE = """
You are a world-class intelligence analyst. Your task is to provide a comprehensive intelligence report on the user's query, based *only* on the provided context.

Your report should be detailed and well-structured, synthesizing all relevant information into a multi-paragraph, flowing narrative.
- Start with a high-level summary of the most critical findings.
- Then, elaborate on the key themes, developments, or viewpoints found in the context.
- Conclude with any underlying patterns or significant details.
- Do NOT use bullet points in the final output.

If the context is insufficient, state that a detailed report cannot be provided.
Do NOT include phrases like "Based on the provided context...". Just write the report.

CONTEXT:
{context}

QUERY:
{input}

COMPREHENSIVE REPORT:
"""

TIMELINE_PROMPT_TEMPLATE = """
You are a historian and intelligence analyst. Based *only* on the provided context, extract key events and dates to build a chronological timeline.

- List events in order, from latest to earliest.
- Format each event as: "YYYY-MM-DD: [Event description]"
- If a full date is not available, use "YYYY-MM" or "YYYY".
- Only include events mentioned in the context. Do not make up information.
- The output should be a clear, ordered list.

CONTEXT:
{context}

QUERY:
{input}

CHRONOLOGICAL TIMELINE:
"""

CONTRADICTIONS_PROMPT_TEMPLATE = """
You are a senior investigative analyst. Your task is to identify conflicting information, opposing viewpoints, or direct contradictions *within the provided context*.

- Clearly state the opposing points.
- Example: "Source A claims [X], while Source B suggests [Y]."
- If no significant contradictions are found, state that the information is largely consistent.
- Be objective and base your findings *only* on the provided context.

CONTEXT:
{context}

QUERY:
{input}

ANALYSIS OF CONFLICTING INFORMATION:
"""

PROMPT_TEMPLATES = {
    "report": REPORT_PROMPT_TEMPLATE,
    "timeline": TIMELINE_PROMPT_TEMPLATE,
    "contradictions": CONTRADICTIONS_PROMPT_TEMPLATE,
}

# Optimized chunking with better overlap for context preservation.
# Created once so every run splits the same text into the same chunks,
# which keeps the chunk hashes (and their cached vectors) stable.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

def chunk_articles(articles):
    """
    Splits each article with enough text into chunks, tagged with its metadata and
    chunk_index (its position in the article: 0, 1, 2...), so a retrieved chunk can be
    traced back to the part of the article it came from.
    """
    all_chunks = []
    for article in articles:
        text = article.get('full_text')
        if not has_enough_text(text):
            continue

        metadata = {
            "source": article.get('url', ''),
            "title": article.get('title', 'No Title'),
            "theme_id": article.get('theme_id', -1),
            "published_at": article.get('published_at', '')
        }

        for chunk_index, chunk_text in enumerate(text_splitter.split_text(text)):
            # A copy per chunk: the chunks of one article must not share one metadata dict
            all_chunks.append(Document(page_content=chunk_text, metadata={**metadata, "chunk_index": chunk_index}))
    return all_chunks

# --- Retry for Gemini embedding calls ---
RETRY_ATTEMPTS = 3            # retries after the first call
RETRY_BASE_DELAY = 2.0        # seconds: ~2, 4, 8 between attempts, plus up to 1 s jitter
RETRYABLE_CODES = {429, 500, 502, 503, 504}

def _is_retryable(e):
    # LangChain raises GoogleGenerativeAIError *from* the SDK's APIError, so check the cause chain
    for _ in range(5):
        if e is None:
            return False
        if isinstance(e, genai_errors.APIError) and e.code in RETRYABLE_CODES:
            return True
        e = e.__cause__ or e.__context__
    return False

async def call_with_retry(label, fn, *args, **kwargs):
    """
    Awaits fn(*args, **kwargs), retrying rate-limit (429) and server (5xx) errors
    with exponential backoff and jitter. Other errors are raised immediately.
    """
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == RETRY_ATTEMPTS or not _is_retryable(e):
                raise
            delay = RETRY_BASE_DELAY * 2 ** attempt + random.uniform(0, 1)
            print(f"[WARN] {label} failed with a retryable error ({type(e).__name__}). Retry {attempt + 1}/{RETRY_ATTEMPTS} in {delay:.1f}s.")
            await asyncio.sleep(delay)

class _QueryEmbeddingsWithRetry(Embeddings):
    """
    What FAISS uses to embed the query at search time: the shared embeddings
    client, with the same retry as document embedding.
    """
    def __init__(self, inner):
        self.inner = inner

    def embed_documents(self, texts):
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)

    async def aembed_query(self, text):
        return await call_with_retry("Query embedding", self.inner.aembed_query, text)

async def get_vectors(texts):
    """
    Returns one document vector per text, in order. Reads cached vectors first and
    embeds only the misses (each distinct text once), then caches the new ones.
    Raises if the embedding API fails.
    """
    if not embeddings_model:
        raise RuntimeError("Embeddings model not loaded.")

    vectors = await cache_service.get_cached_vectors(EMB_CACHE_MODEL, texts)
    cache_hits = sum(v is not None for v in vectors)

    # Wire copies often have identical text, so embed each distinct missing text once
    missing_texts = list(dict.fromkeys(t for t, v in zip(texts, vectors) if v is None))
    if missing_texts:
        # Async client call: the event loop keeps serving other requests while Gemini works
        new_vectors = await call_with_retry(
            "Embedding", embeddings_model.aembed_documents, missing_texts, task_type=DOCUMENT_TASK_TYPE
        )
        new_vectors = [np.asarray(v, dtype=np.float32) for v in new_vectors]
        await cache_service.set_cached_vectors(EMB_CACHE_MODEL, missing_texts, new_vectors)

        embedded = dict(zip(missing_texts, new_vectors))
        vectors = [v if v is not None else embedded[t] for t, v in zip(texts, vectors)]

    duplicates = len(texts) - cache_hits - len(missing_texts)
    print(f"[INFO] Vectors: {len(texts)} texts, {cache_hits} from cache, {len(missing_texts)} embedded, {duplicates} duplicates.")
    return vectors

async def get_query_vector(text):
    """
    Returns the query's RETRIEVAL_QUERY vector, used to rank articles by relevance.
    Cached under its own task type, so it can never be mixed up with a document vector.
    Raises if the embedding API fails.
    """
    if not embeddings_model:
        raise RuntimeError("Embeddings model not loaded.")

    [cached] = await cache_service.get_cached_vectors(EMB_QUERY_CACHE_MODEL, [text])
    if cached is not None:
        return cached

    vector = await call_with_retry("Query embedding", embeddings_model.aembed_query, text, task_type=QUERY_TASK_TYPE)
    vector = np.asarray(vector, dtype=np.float32)
    await cache_service.set_cached_vectors(EMB_QUERY_CACHE_MODEL, [text], [vector])
    return vector

async def build_vector_store(articles):
    """
    Builds and returns a FAISS vector store from cached or newly embedded chunk vectors.
    Returns None if no article has enough text to chunk. Raises AppError if Gemini
    isn't set up or the embedding call fails, so the user sees why.
    """
    if not embeddings_model:
        raise errors.ai_not_configured("embeddings model not loaded")

    # Splitting and the FAISS build are CPU work: run them in a thread so the event loop stays free
    all_chunks = await asyncio.to_thread(chunk_articles, articles)
    if not all_chunks:
        return None

    texts = [chunk.page_content for chunk in all_chunks]
    try:
        vectors = await get_vectors(texts)
    except Exception as e:
        raise errors.analysis_failed(e) from e

    # Vectors are precomputed, so FAISS only uses the embeddings client to embed the query at search time
    vector_store = await asyncio.to_thread(
        FAISS.from_embeddings,
        text_embeddings=list(zip(texts, vectors)),
        embedding=_QueryEmbeddingsWithRetry(embeddings_model),
        metadatas=[chunk.metadata for chunk in all_chunks],
    )
    print(f"[INFO] RAG: Built vector store from {len(all_chunks)} chunks.")
    return vector_store

def _retrieval_summary(documents, limit=60):
    """
    One line naming the retrieved chunks, grouped by article in retrieval order:
    '15 chunks from 6 articles: "Tesla renames FSD..." [0, 2, 1] | "Tesla China sales..." [0] | ...'
    """
    # Keyed by URL: two outlets can carry the same headline
    by_article = {}
    for doc in documents:
        url = doc.metadata.get('source', '')
        title = doc.metadata.get('title') or url or "?"
        by_article.setdefault(url, (title, []))[1].append(doc.metadata.get('chunk_index', '?'))
    parts = []
    for title, indexes in by_article.values():
        short = title if len(title) <= limit else title[:limit - 3] + "..."
        parts.append(f'"{short}" {indexes}')
    chunks = "1 chunk" if len(documents) == 1 else f"{len(documents)} chunks"
    articles = "1 article" if len(by_article) == 1 else f"{len(by_article)} articles"
    return f"{chunks} from {articles}: " + " | ".join(parts)

def _format_response(response):
    answer = response.get('answer') or ''
    source_documents = response.get('context', [])
    
    sources = []
    seen_urls = set()
    
    for doc in source_documents:
        url = doc.metadata.get('source')
        if url and url not in seen_urls:
            sources.append({
                "title": doc.metadata.get('title'),
                "url": url,
                "published_at": doc.metadata.get('published_at')
            })
            seen_urls.add(url)
    
    return {
        "answer": answer.strip(),
        "sources": sources
    }

async def run_rag_query_with_store(query, vector_store, prompt_type):
    """
    Executes a RAG query using a pre-built vector store.
    Async end to end: query embedding, FAISS search (in a thread) and the Gemini call.
    Raises AppError if the model isn't set up, the call fails or the answer is empty;
    an error is never returned as if it were the answer.
    """
    if not llm:
        raise errors.ai_not_configured("chat model not loaded")
    # Both of these are programming errors, not something the user can fix (-> 500)
    if vector_store is None:
        raise ValueError("run_rag_query_with_store called without a vector store")
    prompt_template = PROMPT_TEMPLATES.get(prompt_type)
    if prompt_template is None:
        raise ValueError(f"Unknown prompt type: {prompt_type!r}")

    # Increased 'k' to 15 to give the LLM more context without dumping everything
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
    print(f"[INFO] RAG: Invoking chain with query: '{query}' for type: '{prompt_type}'")
    try:
        response = await asyncio.wait_for(retrieval_chain.ainvoke({"input": query}), CHAT_DEADLINE_SECONDS)
    except Exception as e:   # a missed deadline is a TimeoutError -> "took too long" (504)
        raise errors.report_failed(e, prompt_type) from e

    print(f"[INFO] RAG: Retrieved {_retrieval_summary(response.get('context', []))}")
    result = _format_response(response)
    if not result["answer"]:
        raise errors.empty_report(prompt_type)
    return result