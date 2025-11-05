from flask import Flask, render_template, request, jsonify
import asyncio
import markdown2  # <-- This is used now
import time 

# --- Our Services ---
from services.news_fetcher import fetch_all_articles 
from services.clustering_service import group_by_theme
from services.rag_service import build_and_query # <-- THE NEW BRAIN

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

    # --- Step 2: Cluster articles by theme ---
    # This adds the 'theme_id' metadata we need for the RAG service
    articles = group_by_theme(articles)

    # --- Step 3: Build RAG pipeline and get synthesized answer ---
    # This one function now does all the hard work
    print("[INFO] Building RAG pipeline and querying...")
    
    # We pass the full list of articles (with theme_ids)
    response_data = build_and_query(query, articles) 

    # --- Step 4: Return the final, intelligent answer ---
    total_time = time.time() - start_time
    print(f"[INFO] Full RAG pipeline complete in {total_time:.2f}s")
    
    # Add metadata to the response
    response_data['time_taken'] = f"{total_time:.2f}s"
    response_data['total_articles_found'] = len(articles)

    # The 'summary' text from RAG is pure text, not Markdown.
    # We must convert it to HTML for the browser.
    if 'summary' in response_data:
        response_data['summary'] = markdown2.markdown(response_data['summary'])

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)