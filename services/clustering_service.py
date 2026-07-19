import numpy as np
from sklearn.cluster import DBSCAN
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# --- Initialize Google Embeddings ---
# This uses Google's API, eliminating the need for heavy local PyTorch models
try:
    print("[INFO] Initializing Google Generative AI Embeddings...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    print("[INFO] Embeddings model initialized.")
except Exception as e:
    print(f"[ERROR] Failed to initialize embeddings model: {e}")
    embeddings_model = None

def group_by_theme(articles):
    """
    Takes a list of article objects and adds a 'theme_id' to each.
    """
    if embeddings_model is None:
        print("[ERROR] Embeddings model not loaded. Skipping clustering.")
        for i, article in enumerate(articles):
            article['theme_id'] = i 
        return articles

    if not articles:
        return []

    print(f"[INFO] Clustering {len(articles)} articles...")
    start_time = time.time()

    articles_to_cluster = [] 

    for i, article in enumerate(articles):
        article['theme_id'] = -1
        content = article.get('full_text') 
        if content and len(content) > 100:
            articles_to_cluster.append((i, content))

    if not articles_to_cluster:
        print("[WARN] No articles with sufficient content to cluster.")
        return articles 

    original_indices, texts = zip(*articles_to_cluster)

    # --- 2. Create Embeddings via API ---
    print("[INFO] Creating embeddings via Google API...")
    try:
        # We process in small batches if necessary, but 30 articles is very small
        embeddings = embeddings_model.embed_documents(list(texts))
    except Exception as e:
        print(f"[ERROR] Failed to encode text via API: {e}")
        return articles

    # --- 3. Run Clustering (DBSCAN) ---
    print("[INFO] Running DBSCAN clustering...")
    # Cosine distance tuning for API embeddings might be slightly different
    # Adjust eps if clusters are too loose/strict
    dbscan = DBSCAN(eps=0.35, min_samples=2, metric='cosine')
    
    # DBSCAN expects numpy arrays
    dbscan.fit(np.array(embeddings))

    labels = dbscan.labels_

    # --- 4. Assign Theme IDs ---
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} unique themes.")

    for i, label in enumerate(labels):
        original_article_index = original_indices[i]
        articles[original_article_index]['theme_id'] = int(label)

    end_time = time.time()
    print(f"[INFO] Clustering complete in {end_time - start_time:.2f}s")

    return articles