from flask import Flask, render_template, request, jsonify
import asyncio
import markdown2
import time 

from app.gemini_summarizer import summarize_with_gemini 

# --- Import our async fetcher ---
from services.news_fetcher import fetch_all_articles 
# --- IMPORT OUR NEW CLUSTERING SERVICE ---
from services.clustering_service import group_by_theme

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"[INFO] Query received: {query}")
    start_time = time.time()

    # --- Step 1: Fetch all articles in parallel ---
    print("[INFO] Starting async article fetch...")
    try:
        articles = asyncio.run(fetch_all_articles(query))
    except Exception as e:
        print(f"[ERROR] Async fetch failed: {e}")
        return jsonify({"error": "Failed to fetch articles"}), 500

    if not articles:
        return jsonify({"error": "No articles found."}), 400

    fetch_end_time = time.time()
    print(f"[INFO] Fetched {len(articles)} articles in {fetch_end_time - start_time:.2f}s")


    # --- START: REFACTORED SECTION ---

    # --- Step 2: Cluster articles by theme ---
    # This adds 'theme_id' (e.g., 0, 1, -1) to each article
    articles = group_by_theme(articles)

    # --- Step 3: Get the top 5 unique themes ---
    # This replaces our old "return first 3" test logic
    
    unique_themes = {} # We'll use a dict to store one article per theme
    
    for article in articles:
        theme = article.get('theme_id', -1)
        
        # We only care about grouped themes (not -1, which is "noise")
        # And we only want *one* article per theme
        if theme != -1 and theme not in unique_themes:
            unique_themes[theme] = article
            
    # Get the list of representative articles
    # We limit to 5 unique themes for now
    top_themed_articles = list(unique_themes.values())[:5]

    print(f"[INFO] Selected {len(top_themed_articles)} articles representing unique themes.")

    # --- Step 4: Return the unique articles (Test Mode) ---
    # We are still in test mode. We are *not* summarizing yet.
    # We are just returning the *raw data* for the top themed articles
    # to confirm our clustering logic works.
    
    total_time = time.time() - start_time
    
    return jsonify({
        "message": "Clustering successful! (Test mode)",
        "time_taken": f"{total_time:.2f}s",
        "total_articles_found": len(articles),
        "unique_themes_found": len(unique_themes),
        "articles_showing": len(top_themed_articles),
        "articles": top_themed_articles # This returns the raw data
    })

    # --- END: REFACTORED SECTION ---

    # --- NOTE: We will re-add the summarization loop in the NEXT phase. ---
    # The code below is still "disabled" by the 'return' statement above.
    
    print("[INFO] Starting summarization...")
    # ... (summarization loop will go here) ...

if __name__ == "__main__":
    app.run(debug=True)