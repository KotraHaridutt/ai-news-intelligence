from flask import Flask, render_template, request, jsonify
import asyncio
import markdown2
import time 

# --- Caching Imports ---
from flask_caching import Cache

# --- Our Services ---
from services.news_fetcher import fetch_all_articles 
from services.clustering_service import group_by_theme
# --- IMPORT ALL OUR NEW RAG FUNCTIONS ---
from services.rag_service import get_summary_report, get_timeline, get_contradictions

app = Flask(__name__)

# --- Configure Caching ---
app.config["CACHE_TYPE"] = "simple"
cache = Cache(app)


# --- START: NEW CACHED DATA FUNCTION ---
@cache.memoize(timeout=600) # Cache results for 10 minutes
def get_cached_article_data(query):
    """
    This is our *only* cached function.
    It does the expensive Fetch & Cluster steps.
    It returns the raw articles list and the related_links list.
    """
    print(f"\n[CACHE MISS] Processing new query: {query}\n")
    
    # --- Step 1: Fetch ---
    try:
        articles = asyncio.run(fetch_all_articles(query))
    except Exception as e:
        print(f"[ERROR] Async fetch failed: {e}")
        return None, None # Return None on error
    
    if not articles:
        return [], [] # Return empty lists if no articles found

    # --- Step 2: Cluster ---
    articles = group_by_theme(articles)

    # --- Step 3: Build "Related Links" ---
    unique_themes = {}
    for article in articles:
        theme = article.get('theme_id', -1)
        if theme != -1 and theme not in unique_themes:
            unique_themes[theme] = article
            
    top_themed_articles = list(unique_themes.values())[:5]
    
    related_links = [
        {
            "title": art.get("title"), 
            "url": art.get("url"), 
            "source": art.get("source_name")
        }
        for art in top_themed_articles
    ]
    
    # Return both items for the cache
    return articles, related_links

# --- END: NEW CACHED DATA FUNCTION ---


# --- START: ENDPOINT DEFINITIONS ---

@app.route("/")
def home():
    return render_template("index.html")

# 1. Main Query Endpoint (for our Frontend)
@app.route("/query", methods=["POST"])
def query():
    start_time = time.time()
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"[INFO] Main query received: {query}")
    
    # --- Get data from cache (or run the pipeline) ---
    articles, related_links = get_cached_article_data(query)
    
    if articles is None:
        return jsonify({"error": "Failed to fetch articles"}), 500
    if not articles:
        return jsonify({"error": "No articles found"}), 404

    # --- Run the RAG summary ---
    # This is fast, so we don't cache it
    rag_data = get_summary_report(query, articles) 

    # --- Assemble Final JSON Response ---
    final_response = {
        "summary_html": markdown2.markdown(rag_data.get("answer", "No answer generated.")),
        "cited_sources": rag_data.get("sources", []),
        "related_links": related_links,
        "total_articles_found": len(articles),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }
    
    return jsonify(final_response)


# 2. NEW: Timeline Endpoint
@app.route("/api/timeline", methods=["POST"])
def api_timeline():
    start_time = time.time()
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"[INFO] API/Timeline query received: {query}")
    
    # --- Get data from cache ---
    articles, _ = get_cached_article_data(query) # We don't need related_links here
    
    if articles is None:
        return jsonify({"error": "Failed to fetch articles"}), 500
    if not articles:
        return jsonify({"error": "No articles found"}), 404
        
    # --- Run the TIMELINE RAG query ---
    rag_data = get_timeline(query, articles)
    
    # Format and return
    response = {
        "query": query,
        "timeline_html": markdown2.markdown(rag_data.get("answer")),
        "cited_sources": rag_data.get("sources"),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }
    return jsonify(response)


# 3. NEW: Contradictions Endpoint
@app.route("/api/contradictions", methods=["POST"])
def api_contradictions():
    start_time = time.time()
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"[INFO] API/Contradictions query received: {query}")
    
    # --- Get data from cache ---
    articles, _ = get_cached_article_data(query)
    
    if articles is None:
        return jsonify({"error": "Failed to fetch articles"}), 500
    if not articles:
        return jsonify({"error": "No articles found"}), 404
        
    # --- Run the CONTRADICTIONS RAG query ---
    rag_data = get_contradictions(query, articles)
    
    # Format and return
    response = {
        "query": query,
        "analysis_html": markdown2.markdown(rag_data.get("answer")),
        "cited_sources": rag_data.get("sources"),
        "time_taken": f"{time.time() - start_time:.2f}s"
    }
    return jsonify(response)

# --- END: ENDPOINT DEFINITIONS ---

if __name__ == "__main__":
    app.run(debug=True)