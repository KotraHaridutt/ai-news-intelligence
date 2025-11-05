import os
import re
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Load API Key ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("[ERROR] GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- Initialize Models (Load them once) ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("[INFO] RAG models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load RAG models: {e}")
    llm = None
    embeddings_model = None


# --- START: NEW PROMPT TEMPLATES ---

# 1. The Main Report Prompt (Our old prompt, renamed)
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

# 2. The New Timeline Prompt
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

# 3. The New Contradictions Prompt
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
# --- END: NEW PROMPT TEMPLATES ---


# --- START: NEW REUSABLE FUNCTIONS ---

def _build_vector_store(articles):
    """
    Private function. Takes articles, chunks them, and builds a FAISS vector store.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_chunks = []
    
    for article in articles:
        text = article.get('full_text')
        if not text or len(text) < 100:
            continue
            
        metadata = {
            "source": article.get('url', ''),
            "title": article.get('title', 'No Title'),
            "theme_id": article.get('theme_id', -1),
            "snippet": article.get('snippet', '')
        }
        
        chunks = text_splitter.split_text(text)
        for chunk_text in chunks:
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not all_chunks:
        return None # Return None if no chunks were made

    try:
        vector_store = FAISS.from_documents(all_chunks, embeddings_model)
        print(f"[INFO] RAG: Built vector store from {len(all_chunks)} chunks.")
        return vector_store
    except Exception as e:
        print(f"[ERROR] RAG: Failed to create FAISS vector store: {e}")
        return None

def _create_retrieval_chain(vector_store, prompt_template_string):
    """
    Private function. Builds a retrieval chain from a vector store and prompt string.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 10}) # Use top 10 chunks
    
    prompt = PromptTemplate(template=prompt_template_string, input_variables=["context", "input"])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    return retrieval_chain

def _format_response(response):
    """
    Private function. Formats the RAG output and de-duplicates sources.
    """
    answer = response.get('answer', 'No answer could be generated.')
    source_documents = response.get('context', [])
    
    sources = []
    seen_urls = set()
    
    for doc in source_documents:
        url = doc.metadata.get('source')
        if url and url not in seen_urls:
            sources.append({
                "title": doc.metadata.get('title'),
                "url": url
            })
            seen_urls.add(url)
    
    return {
        "answer": answer.strip(),
        "sources": sources
    }

# --- END: NEW REUSABLE FUNCTIONS ---


# --- START: NEW PUBLIC FUNCTIONS (Called by app.py) ---

def _run_rag_query(query, articles, prompt_template):
    """
    Master function to run any RAG query.
    """
    if llm is None or embeddings_model is None:
        return {"answer": "Error: RAG models are not loaded.", "sources": []}
    
    # 1. Build Vector Store (Fast, in-memory)
    vector_store = _build_vector_store(articles)
    if vector_store is None:
        return {"answer": "No articles with enough content to build an answer.", "sources": []}
    
    # 2. Build Chain (Fast)
    retrieval_chain = _create_retrieval_chain(vector_store, prompt_template)
    
    # 3. Query (This is the main "thinking" part)
    print(f"[INFO] RAG: Invoking chain with query: '{query}'")
    try:
        response = retrieval_chain.invoke({"input": query})
    except Exception as e:
        print(f"[ERROR] RAG: Pipeline query failed: {e}")
        return {"answer": "Error: The AI query failed.", "sources": []}
    
    # 4. Format and Return
    return _format_response(response)


# Function for the main /query endpoint
def get_summary_report(query, articles):
    print("[INFO] RAG: Generating Summary Report...")
    return _run_rag_query(query, articles, REPORT_PROMPT_TEMPLATE)

# Function for the new /api/timeline endpoint
def get_timeline(query, articles):
    print("[INFO] RAG: Generating Timeline...")
    return _run_rag_query(query, articles, TIMELINE_PROMPT_TEMPLATE)

# Function for the new /api/contradictions endpoint
def get_contradictions(query, articles):
    print("[INFO] RAG: Finding Contradictions...")
    return _run_rag_query(query, articles, CONTRADICTIONS_PROMPT_TEMPLATE)

# --- END: NEW PUBLIC FUNCTIONS ---