import os
import numpy as np
from dotenv import load_dotenv

from services import cache_service
from services.news_fetcher import has_enough_text

# --- LangChain Imports ---
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

# Initialize Models (the one embeddings client used by clustering and indexing)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
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

# Optimized chunking with better overlap for context preservation.
# Created once so every run splits the same text into the same chunks,
# which keeps the chunk hashes (and their cached vectors) stable.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

def chunk_articles(articles):
    """
    Splits each article with enough text into chunks, tagged with its metadata.
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

        for chunk_text in text_splitter.split_text(text):
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))
    return all_chunks

async def get_vectors(texts):
    """
    Returns one document vector per text, in order. Reads cached vectors first and
    embeds only the misses (each distinct text once), then caches the new ones.
    Raises if the embedding API fails.
    """
    if not embeddings_model:
        raise RuntimeError("Embeddings model not loaded.")

    vectors = await cache_service.get_cached_vectors(EMB_CACHE_MODEL, texts)

    # Wire copies often have identical text, so embed each distinct missing text once
    missing_texts = list(dict.fromkeys(t for t, v in zip(texts, vectors) if v is None))
    if missing_texts:
        new_vectors = embeddings_model.embed_documents(missing_texts, task_type=DOCUMENT_TASK_TYPE)
        new_vectors = [np.asarray(v, dtype=np.float32) for v in new_vectors]
        await cache_service.set_cached_vectors(EMB_CACHE_MODEL, missing_texts, new_vectors)

        embedded = dict(zip(missing_texts, new_vectors))
        vectors = [v if v is not None else embedded[t] for t, v in zip(texts, vectors)]

    print(f"[INFO] Vectors: {len(texts)} texts, {len(texts) - len(missing_texts)} from cache, {len(missing_texts)} embedded.")
    return vectors

async def build_vector_store(articles):
    """
    Builds and returns a FAISS vector store from cached or newly embedded chunk vectors.
    """
    if not embeddings_model:
        return None

    all_chunks = chunk_articles(articles)
    if not all_chunks:
        return None

    try:
        texts = [chunk.page_content for chunk in all_chunks]
        vectors = await get_vectors(texts)
        # Vectors are precomputed, so FAISS only uses embeddings_model to embed the query at search time
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, vectors)),
            embedding=embeddings_model,
            metadatas=[chunk.metadata for chunk in all_chunks],
        )
        print(f"[INFO] RAG: Built vector store from {len(all_chunks)} chunks.")
        return vector_store
    except Exception as e:
        print(f"[ERROR] RAG: Failed to create FAISS vector store: {e}")
        return None

def _format_response(response):
    answer = response.get('answer', 'No answer could be generated.')
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

def run_rag_query_with_store(query, vector_store, prompt_type):
    """
    Executes a RAG query using a pre-built vector store.
    """
    if not llm or not vector_store:
        return {"answer": "Error: RAG models or Vector Store missing.", "sources": []}
    
    if prompt_type == "report":
        prompt_template = REPORT_PROMPT_TEMPLATE
    elif prompt_type == "timeline":
        prompt_template = TIMELINE_PROMPT_TEMPLATE
    elif prompt_type == "contradictions":
        prompt_template = CONTRADICTIONS_PROMPT_TEMPLATE
    else:
        return {"answer": "Invalid prompt type.", "sources": []}

    # Increased 'k' to 15 to give the LLM more context without dumping everything
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
    print(f"[INFO] RAG: Invoking chain with query: '{query}' for type: '{prompt_type}'")
    try:
        response = retrieval_chain.invoke({"input": query})
        return _format_response(response)
    except Exception as e:
        print(f"[ERROR] RAG: Pipeline query failed: {e}")
        return {"answer": "Error: The AI query failed.", "sources": []}