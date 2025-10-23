from flask import Flask, render_template, request, jsonify
from app.gnews_fetcher import fetch_news
from app.gemini_summarizer import summarize_with_gemini
from app.article_extractor import extract_text
import markdown2  # Keep this import

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/fetch", methods=["POST"])
def fetch_and_summarize():
    data = request.get_json()
    query = data.get("query")

    print(f"[INFO] Query received: {query}")
    articles = fetch_news(query) 

    if not articles:
        return jsonify({"error": "No articles found or API limit reached."}), 400

    # --- START: MODIFIED SECTION ---
    
    summaries_list = [] # This will hold our new summary objects
    
    for article in articles:
        try:
            print(f"[INFO] Processing: {article['url']}")
            text = extract_text(article["url"])
            if not text:
                print(f"[WARN] No text extracted for: {article['url']}")
                continue # Skip if extraction fails
                
            # --- Call summarizer FOR EACH article ---
            # We pass the query and the single article's text
            summary_markdown = summarize_with_gemini(query, text) 
            summary_html = markdown2.markdown(summary_markdown, safe_mode=True)
            
            # Add the full summary object to our list
            summaries_list.append({
                "title": article["title"],
                "url": article["url"],
                "summary": summary_html
            })
        except Exception as e:
            # Log the error but don't crash the whole loop
            print(f"[ERROR] Failed to process {article['url']}: {e}")

    if not summaries_list:
        return jsonify({"error": "Could not extract or summarize any articles."}), 400
    
    # --- Return the new list ---
    # Instead of "summary" and "sources", we send one "summaries" list
    return jsonify({
        "summaries": summaries_list
    })

    # --- END: MODIFIED SECTION ---

if __name__ == "__main__":
    app.run(debug=True)