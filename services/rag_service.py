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
# Load environment variables from .env file (like GOOGLE_API_KEY)
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("[ERROR] GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- Initialize Models (Load them once) ---
try:
    # We use Gemini 1.5 Flash for speed and power
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # The new embedding model from Google
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("[INFO] RAG models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load RAG models: {e}")
    llm = None
    embeddings_model = None


# --- The "Meta-Prompt" ---
# This prompt template is the "brain" of our analyst
PROMPT_TEMPLATE = """
You are a world-class intelligence analyst. Your task is to answer the user's query based *only* on the provided context.

Synthesize a concise, coherent, multi-paragraph answer. Do not use bullet points.
Your answer should be a single, flowing narrative.

If the context is insufficient to answer the query, state that you cannot provide an answer based on the available information.

Do NOT include phrases like "Based on the provided context...". Just give the answer.

CONTEXT:
{context}

QUERY:
{input}

SYNTHESIZED ANSWER:
"""

def build_and_query(query, articles):
    """
    Builds an in-memory RAG pipeline from articles and queries it.
    """
    if llm is None or embeddings_model is None:
        return {"summary": "Error: RAG models are not loaded.", "sources": []}

    # --- 1. Load & Chunk ---
    print(f"[INFO] RAG: Loading {len(articles)} articles into RAG pipeline.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_chunks = []

    for article in articles:
        text = article.get('full_text')

        # We only process articles with text
        if not text or len(text) < 100:
            continue

        # --- This is the key: Add metadata to each chunk ---
        metadata = {
            "source": article.get('url', ''),
            "title": article.get('title', 'No Title'),
            "theme_id": article.get('theme_id', -1), # From our clustering!
            "snippet": article.get('snippet', '')
        }

        # Split the article text into chunks
        chunks = text_splitter.split_text(text)

        # Create a Document object for each chunk with its metadata
        for chunk_text in chunks:
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not all_chunks:
        return {"summary": "No articles with enough content to build an answer.", "sources": []}

    # --- 2. Embed & Store in FAISS (In-Memory Vector Store) ---
    print(f"[INFO] RAG: Creating vector store from {len(all_chunks)} chunks...")
    try:
        vector_store = FAISS.from_documents(all_chunks, embeddings_model)
    except Exception as e:
        print(f"[ERROR] RAG: Failed to create FAISS vector store: {e}")
        return {"summary": "Error: Failed to create vector store.", "sources": []}

    # --- 3. Define Retriever & Chain ---
    # Get the top 5 most relevant chunks for any query
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Create the prompt from our template
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "input"])

    # This chain combines the retrieved documents into the {context}
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # This chain runs the retriever, then the qa_chain
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)

    # --- 4. Query the Pipeline ---
    print(f"[INFO] RAG: Querying pipeline with: '{query}'")
    try:
        response = retrieval_chain.invoke({"input": query})
    except Exception as e:
        print(f"[ERROR] RAG: Pipeline query failed: {e}")
        return {"summary": "Error: The AI query failed.", "sources": []}

    # --- 5. Format and Return the Response ---

    # The main synthesized answer
    answer = response.get('answer', 'No answer could be generated.')

    # The list of source documents (chunks) the LLM used
    source_documents = response.get('context', [])

    # We need to de-duplicate the sources
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

    # Return the final object
    return {
        "summary": answer.strip(),
        "sources": sources
    }