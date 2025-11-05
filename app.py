from flask import Flask, render_template, request, jsonify
import asyncio  
import markdown2  
import time 

from app.gemini_summarizer import summarize_with_gemini 

# --- Importing new async service ---

from services.news_fetcher import fetch_all_articles 

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    """
    This endpoint is now a sync function that *calls*
    your new async service using asyncio.run().
    """
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print(f"[INFO] Query received: {query}")
    start_time = time.time()

    # --- START: REFACTORED SECTION ---

    # 1. This one line replaces BOTH 'fetch_news' AND the 'extract_text'
    #    calls from  old loop. It runs all fetches in parallel.
    print("[INFO] Starting async article fetch...")
    try:
        # Using asyncio.run() to "bridge" from sync Flask to your async function
        articles = asyncio.run(fetch_all_articles(query))
    except Exception as e:
        print(f"[ERROR] Async fetch failed: {e}")
        return jsonify({"error": "Failed to fetch articles"}), 500

    if not articles:
        return jsonify({"error": "No articles found."}), 400

    end_time = time.time()
    print(f"[INFO] Fetched {len(articles)} articles in {end_time - start_time:.2f}s")

    # 2.  "For now, just return the first 3 articles"
    #    This is a test to make sure the fetcher works.
    test_articles = articles[:3]

    return jsonify({
        "message": "Async fetch successful! (Test mode)",
        "time_taken": f"{end_time - start_time:.2f}s",
        "article_count": len(articles),
        "articles_showing": len(test_articles),
        "articles": test_articles # This returns the raw article data
    })

    # --- END: REFACTORED SECTION ---

    # --- NOTE: We will re-add the summarization loop in the NEXT step. ---
    
    
    print("[INFO] Starting summarization...")
    summaries_list = []
    
    for article in articles:
        try:
            print(f"[INFO] Summarizing: {article['url']}")
            
            
            text = article.get("text") # Or article.get("content")
            
            if not text:
                print(f"[WARN] No text found for: {article['url']}")
                continue 
                
            summary_markdown = summarize_with_gemini(query, text) 
            summary_html = markdown2.markdown(summary_markdown, safe_mode=True)
            
            summaries_list.append({
                "title": article["title"],
                "url": article["url"],
                "summary": summary_html
            })
        except Exception as e:
            print(f"[ERROR] Failed to process {article['url']}: {e}")

    if not summaries_list:
        return jsonify({"error": "Could not extract or summarize any articles."}), 400
    
    return jsonify({
        "summaries": summaries_list
    })

if __name__ == "__main__":
    app.run(debug=True)