import os
from dotenv import load_dotenv

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

# Initialize Models
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
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

# NEW CACHEABLE VECTOR STORE FUNCTION
def build_vector_store(articles):
    """
    Builds and returns a FAISS vector store. This should be called once per query and cached.
    """
    if not embeddings_model:
        return None
        
    # Optimized chunking with better overlap for context preservation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    all_chunks = []
    
    for article in articles:
        text = article.get('full_text')
        if not text or len(text) < 100:
            continue
            
        metadata = {
            "source": article.get('url', ''),
            "title": article.get('title', 'No Title'),
            "theme_id": article.get('theme_id', -1),
            "published_at": article.get('published_at', '')
        }
        
        chunks = text_splitter.split_text(text)
        for chunk_text in chunks:
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not all_chunks:
        return None

    try:
        vector_store = FAISS.from_documents(all_chunks, embeddings_model)
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