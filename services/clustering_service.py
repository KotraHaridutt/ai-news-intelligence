import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import time

# --- Initialize Model ---
# We load the model *once* when the app starts, not every time
# the function is called. This saves a lot of time.
# 'all-MiniLM-L6-v2' is a great, fast model for this.
try:
    print("[INFO] Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

def group_by_theme(articles):
    """
    Takes a list of article objects and adds a 'theme_id' to each.
    """
    if model is None:
        print("[ERROR] Model not loaded. Skipping clustering.")
        # Return articles without themes
        for i, article in enumerate(articles):
            article['theme_id'] = i # Assign unique IDs as a fallback
        return articles

    if not articles:
        return []

    print(f"[INFO] Clustering {len(articles)} articles...")
    start_time = time.time()

    # --- 1. Prepare Data ---
    # We need to separate articles with content from those without,
    # so we only cluster valid text.

    # We'll store (original_index, article_content)
    articles_to_cluster = [] 

    # Assign a default theme of -1 (noise/ungrouped) to all
    for i, article in enumerate(articles):
        article['theme_id'] = -1
        # Get text from 'content' key (from our news_fetcher)
        content = article.get('full_text') 

        # We only cluster articles with a decent amount of text
        if content and len(content) > 50:
            articles_to_cluster.append((i, content))

    if not articles_to_cluster:
        print("[WARN] No articles with sufficient content to cluster.")
        return articles # Return with default theme_id = -1

    # Unzip the list into separate lists for indices and texts
    original_indices, texts = zip(*articles_to_cluster)

    # --- 2. Create Embeddings ---
    # This converts our list of text snippets into a list of number vectors (embeddings)
    print("[INFO] Creating embeddings...")
    try:
        embeddings = model.encode(texts, show_progress_bar=True)
    except Exception as e:
        print(f"[ERROR] Failed to encode text: {e}")
        return articles

    # --- 3. Run Clustering (DBSCAN) ---
    # DBSCAN is great because we don't need to tell it *how many*
    # clusters to find. It finds them automatically.

    # 'eps' is the "distance" to consider for a cluster.
    # 'min_samples=2' means a theme needs at least 2 articles.
    print("[INFO] Running DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.25, min_samples=2, metric='cosine')

    # Fit the model to our embeddings
    dbscan.fit(embeddings)

    # Get the cluster labels (e.g., 0, 1, 2... -1 is noise)
    labels = dbscan.labels_

    # --- 4. Assign Theme IDs ---
    # Now we map the cluster labels back to our *original* articles list

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} unique themes.")

    for i, label in enumerate(labels):
        # The 'i' here is the index *within articles_to_cluster*
        # We get the *original* index from our 'original_indices' list
        original_article_index = original_indices[i]

        # Assign the cluster label as the theme_id (as an int)
        articles[original_article_index]['theme_id'] = int(label)

    end_time = time.time()
    print(f"[INFO] Clustering complete in {end_time - start_time:.2f}s")

    return articles