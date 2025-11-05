from flask import Flask, render_template, request, jsonify
import asyncio
import markdown2
import time 

# --- Caching Imports ---
from flask_caching import Cache

# --- Our Services ---
from services.news_fetcher import fetch_all_articles 
from services.clustering_service import group_by_theme
from services.rag_service import build_and_query

app = Flask(__name__)

# --- Configure Caching ---
# We use a simple in-memory cache ("simple")
# You could change this to "redis", "memcached", etc. later
app.config["CACHE_TYPE"] = "simple"
cache = Cache(app)

# --- This is our new, cached function ---
@cache.memoize(timeout=600) # Cache results for 600 seconds (10 minutes)
def process_query(query):
    """
    This function contains our entire data pipeline.
    Flask-Caching will cache its return value based on the 'query' argument.
    """
    print(f"\n[CACHE MISS] Processing new query: {query}\n")
    
    # --- Step 1: Fetch ---
    try:
        articles = asyncio.run(fetch_all_articles(query))
    except Exception as e:
        print(f"[ERROR] Async fetch failed: {e}")
        # We return a dict so the frontend can handle the error
        return {"error": "Failed to fetch articles"}
    
    if not articles:
        return {"error": "No articles found."}

    # --- Step 2: Cluster ---
    articles = group_by_theme(articles)

    # --- Step 3: Build "Related Links" (THE NEW JSON) ---
    # This is the "Product Fix": We get the top 5 unique themes
    # to show as related links *before* we do the RAG.
    
    unique_themes = {}
    for article in articles:
        theme = article.get('theme_id', -1)
        if theme != -1 and theme not in unique_themes:
            unique_themes[theme] = article
            
    # Get the top 5 unique themed articles
    top_themed_articles = list(unique_themes.values())[:5]
    
    # Format them into a clean list for the frontend
    related_links = [
        {
            "title": art.get("title"), 
            "url": art.get("url"), 
            "source": art.get("source_name")
        }
        for art in top_themed_articles
    ]

    # --- Step 4: Build RAG and get Answer ---
    # We pass *all* articles to the RAG to get the best possible answer
    rag_data = build_and_query(query, articles) 

    # --- Step 5: Assemble Final JSON Response ---
    # This is the final structure our frontend will receive
    final_response = {
        # The 'summary' from RAG, converted to HTML
        "summary_html": markdown2.markdown(rag_data.get("summary", "No answer generated.")),
        
        # The 'sources' from RAG
        "cited_sources": rag_data.get("sources", []),
        
        # Our new "related links" list from clustering
        "related_links": related_links,
        
        # Metadata
        "total_articles_found": len(articles),
        "unique_themes_found": len(unique_themes)
    }
    
    return final_response

# --- This is our main Flask endpoint ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    """
    This endpoint is now just a thin wrapper.
    It gets the request, validates it, and calls our cached 'process_query' function.
    """
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"[INFO] Query received: {query}")
    start_time = time.time()
    
    # --- This call is now CACHED! ---
    # If 'process_query(query)' was run in the last 10 mins,
    # it returns the saved result instantly.
    # If not, it runs the function and saves the result.
    response_data = process_query(query)
    
    end_time = time.time()
    
    # Add time_taken to the response (this will be 0.0s on a cache hit)
    response_data['time_taken'] = f"{end_time - start_time:.2f}s"

    # Check if the cached result was an error
    if "error" in response_data:
        return jsonify(response_data), 500

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)