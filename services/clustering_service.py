import numpy as np
from sklearn.cluster import DBSCAN
import time

# DBSCAN cosine-distance radius. Measured on 133 article pairs from 6 real queries
# (gemini-embedding-001, whole-article vectors):
#   same wire story in two outlets ........ 0.000-0.009
#   different pieces on the same event .... 0.053-0.103
#   different stories on the same topic ... 0.142 and up (median ~0.27)
# 0.35 chained every query into one group; 0.10-0.15 grouped exactly the duplicates.
# 0.12 sits between the furthest near-duplicate (0.103) and the closest different story (0.142).
DBSCAN_EPS = 0.12
DBSCAN_MIN_SAMPLES = 2

def group_by_theme(articles, vectors):
    """
    Adds a 'theme_id' to each article (in place) from its precomputed article vector.
    vectors[i] belongs to articles[i]; the caller keeps them for later steps.
    Returns True on success. On failure every article is left at -1 and it returns False,
    so the caller can tell "no themes found" apart from "grouping failed".
    """
    for article in articles:
        article['theme_id'] = -1

    if not articles:
        return True

    if vectors is None or len(vectors) != len(articles):
        print("[ERROR] Clustering skipped: article vectors missing or misaligned.")
        return False

    print(f"[INFO] Clustering {len(articles)} articles...")
    start_time = time.time()

    # --- Run Clustering (DBSCAN) ---
    try:
        dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='cosine')
        dbscan.fit(np.array(vectors))
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}")
        return False

    labels = dbscan.labels_

    # --- Assign Theme IDs ---
    # -1 now only means DBSCAN found no similar article (noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[INFO] Found {num_clusters} unique themes.")

    for article, label in zip(articles, labels):
        article['theme_id'] = int(label)

    end_time = time.time()
    print(f"[INFO] Clustering complete in {end_time - start_time:.2f}s")

    return True
