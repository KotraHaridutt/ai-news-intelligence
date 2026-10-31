"""
Trains a fallback classifier for the safety checks (feature 3, Section B #16).

    python scripts/train_safety_classifier.py intent
    python scripts/train_safety_classifier.py injection --data .local/safety_data/curated/chunks.jsonl

Reads labeled examples ({"text", "label"} per line; services/safety/data/<model>.jsonl unless
--data says otherwise), embeds them with the task type the app will score them with, trains a
logistic regression, picks the allow / uncertain / block thresholds from target false-alarm rates,
and writes services/safety/models/<model>.json: weights, classes, thresholds, the data's hash, the
held-out results and a checksum. Prints the confusion matrix.

How it's measured, so the numbers aren't flattered by the data the model saw:
1. 25% of the examples are held out (stratified, fixed seed).
2. On the other 75%, 5-fold out-of-fold scores pick C and the thresholds.
3. A model trained on that 75% is scored on the held-out 25%: those are the numbers reported.
4. The shipped model is trained on everything, with thresholds from out-of-fold scores on everything.
5. With a few hundred examples one held-out split is noisy (one news query is 5% of 19), so the
   shipped setup is also scored by 10 x 5-fold cross-validation at its bands: the steadier number.

Weighting: the fine examples (news queries) get half the total weight and the rest the other half.
"Balanced per class" gave the 9 system_probe examples as much weight as the 78 news queries, and
blocked 21% of held-out news (B2 §4).

Embeddings: the pipeline's own Redis cache first (the same keys rag_service uses, so the app reuses
these vectors), then a local copy (.local/safety_data/vectors.npz), then Gemini for the rest, in
batches of at most 90 texts, 61 s apart (the free tier allows 100 texts a minute and 100 requests
a day; a batch is one request). --max-requests stops before spending more than that.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Training needs no traces. Set, not removed: load_dotenv() in the services would refill a missing one
os.environ["LANGSMITH_TRACING"] = "false"

from services import cache_service, rag_service  # noqa: E402
from services.safety import classifier  # noqa: E402

DATA_DIR = ROOT / "services/safety/data"
LOCAL_VECTORS = ROOT / ".local/safety_data/vectors.npz"
SEED = 20261031
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
BATCH = 90
BATCH_WAIT_SECONDS = 61
WEIGHT_DIGITS = 7              # significant digits kept in the JSON; scores match to ~1e-6

# Per model: the task type of the vectors it scores, the classes that are fine, and the share of
# fine examples allowed above each threshold (the false-alarm targets the thresholds come from).
# max_blocked 0: block only above every fine example's out-of-fold score, and never below
# min_block_at. A news query wrongly refused costs more than a refusal missed: the request
# goes on to the other layers (E, F, I-K), and a flag is still recorded.
MODELS = {
    # The query check: scores the query's RETRIEVAL_QUERY vector, which the pipeline makes anyway
    "intent": {"task_type": rag_service.QUERY_TASK_TYPE, "allow": ["news_question"],
               "max_flagged": 0.05, "max_blocked": 0.0, "min_block_at": 0.5},
    # The chunk scan: scores the index's own RETRIEVAL_DOCUMENT vectors. Trained and measured in B, but
    # not shipped: on whole chunks it catches too few planted sentences (B2 §5). Run with --dry-run.
    "injection": {"task_type": rag_service.DOCUMENT_TASK_TYPE, "allow": ["normal"],
                  "max_flagged": 0.05, "max_blocked": 0.0, "min_block_at": 0.5},
}
CACHE_MODEL = {rag_service.QUERY_TASK_TYPE: rag_service.EMB_QUERY_CACHE_MODEL,
               rag_service.DOCUMENT_TASK_TYPE: rag_service.EMB_CACHE_MODEL}


def read_examples(path: Path) -> list[dict]:
    # split("\n"), not splitlines(): some prompts contain U+2028 / U+0085, which splitlines() breaks on
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return [{"text": r["text"], "label": r["label"]} for r in rows]


def data_hash(rows: list[dict]) -> str:
    raw = json.dumps([[r["text"], r["label"]] for r in rows], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --- Embedding ---

def _local_key(task: str, text: str) -> str:
    return hashlib.sha256(f"{task}|{text}".encode("utf-8")).hexdigest()


async def embed(texts: list[str], task: str, max_requests: int) -> tuple[np.ndarray, dict]:
    """One vector per text. Returns (vectors, {"redis", "local", "embedded", "requests"})."""
    local = dict(np.load(LOCAL_VECTORS)) if LOCAL_VECTORS.exists() else {}
    found = {}
    cached = await cache_service.get_cached_vectors(CACHE_MODEL[task], texts)
    for text, vector in zip(texts, cached):
        if vector is not None:
            found[text] = vector
    from_redis = len(found)
    for text in texts:
        if text not in found and _local_key(task, text) in local:
            found[text] = local[_local_key(task, text)]
    from_local = len(found) - from_redis
    missing = list(dict.fromkeys(t for t in texts if t not in found))
    batches = [missing[i:i + BATCH] for i in range(0, len(missing), BATCH)]
    if len(batches) > max_requests:
        sys.exit(f"{len(missing)} texts to embed need {len(batches)} requests; --max-requests is {max_requests}. "
                 "The free tier allows 100 a day, shared with the app.")
    for i, part in enumerate(batches):
        if i:
            print(f"[embed] waiting {BATCH_WAIT_SECONDS} s (100 texts a minute)...")
            await asyncio.sleep(BATCH_WAIT_SECONDS)
        print(f"[embed] {len(part)} texts as {task} (batch {i + 1} of {len(batches)})")
        vectors = await rag_service.call_with_retry("Safety training", rag_service.embeddings_model.aembed_documents,
                                                    part, task_type=task)
        vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        await cache_service.set_cached_vectors(CACHE_MODEL[task], part, vectors)
        for text, vector in zip(part, vectors):
            found[text] = vector
    # A local copy of every vector, including what only Redis had (Redis keeps vectors 7 days)
    new = {_local_key(task, t): found[t] for t in texts if _local_key(task, t) not in local}
    if new:
        LOCAL_VECTORS.parent.mkdir(parents=True, exist_ok=True)
        np.savez(LOCAL_VECTORS, **local, **new)
    X = np.stack([np.asarray(found[t], dtype=np.float64) for t in texts])
    return X, {"redis": from_redis, "local": from_local, "embedded": len(missing), "requests": len(batches)}


# --- Training ---

def normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


def weights_for(y, allow) -> np.ndarray:
    """The fine examples share half the total weight, the others the other half."""
    fine = np.isin(y, allow)
    return np.where(fine, 0.5 / max(1, fine.sum()), 0.5 / max(1, (~fine).sum())) * len(y)


def fit(X, y, C, allow):
    return LogisticRegression(C=C, max_iter=5000).fit(normalize(X), y, sample_weight=weights_for(y, allow))


def block_scores(model, X, allow) -> np.ndarray:
    p = model.predict_proba(normalize(X))
    fine = [list(model.classes_).index(c) for c in allow]
    return np.clip(1.0 - p[:, fine].sum(axis=1), 0.0, 1.0)


def out_of_fold(X, y, C, allow):
    """Each example's block score from a model that didn't see it (5 folds)."""
    scores = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for train, test in folds.split(X, y):
        scores[test] = block_scores(fit(X[train], y[train], C, allow), X[test], allow)
    return scores


def repeated_cv(X, y, C, allow, uncertain_at, block_at, repeats=10) -> dict:
    """Step 5: the rates at the bands, averaged over 10 x 5-fold cross-validation."""
    bad = ~np.isin(y, allow)
    flagged, blocked, caught, caught_block = [], [], [], []
    for train, test in RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED).split(X, y):
        s = block_scores(fit(X[train], y[train], C, allow), X[test], allow)
        b = bad[test]
        flagged.append(np.sum(s[~b] >= uncertain_at)); blocked.append(np.sum(s[~b] >= block_at))
        caught.append(np.sum(s[b] >= uncertain_at)); caught_block.append(np.sum(s[b] >= block_at))
    fine_n, bad_n = (~bad).sum() * repeats, bad.sum() * repeats
    return {"repeats": repeats, "fine_flagged_or_blocked": round(sum(flagged) / fine_n, 3),
            "fine_blocked": round(sum(blocked) / fine_n, 3),
            "bad_caught_flag_or_block": round(sum(caught) / bad_n, 3), "bad_blocked": round(sum(caught_block) / bad_n, 3)}


def log_loss_of(scores, bad):
    s = np.clip(scores, 1e-6, 1 - 1e-6)
    return float(-np.mean(np.where(bad, np.log(s), np.log(1 - s))))


def threshold_for(fine_scores: np.ndarray, max_share: float) -> float:
    """The lowest threshold with at most max_share of the fine examples at or above it."""
    for t in np.round(np.arange(0.05, 1.0, 0.01), 2):
        if np.mean(fine_scores >= t) <= max_share:
            return float(t)
    return 0.99


def pick(X, y, allow, cfg) -> dict:
    """C by out-of-fold log loss, then the thresholds from the false-alarm targets."""
    bad = ~np.isin(y, allow)
    best = None
    for C in C_GRID:
        scores = out_of_fold(X, y, C, allow)
        loss = log_loss_of(scores, bad)
        if best is None or loss < best[1]:
            best = (C, loss, scores)
    C, _, scores = best
    block_at = max(threshold_for(scores[~bad], cfg["max_blocked"]), cfg.get("min_block_at", 0.0))
    uncertain_at = min(threshold_for(scores[~bad], cfg["max_flagged"]), block_at)
    return {"C": C, "uncertain_at": uncertain_at, "block_at": block_at}


def results(scores, y, allow, uncertain_at, block_at) -> dict:
    bad = ~np.isin(y, allow)
    out = {"fine": int((~bad).sum()), "bad": int(bad.sum()),
           "fine_flagged_or_blocked": round(float(np.mean(scores[~bad] >= uncertain_at)), 3) if (~bad).any() else None,
           "fine_blocked": round(float(np.mean(scores[~bad] >= block_at)), 3) if (~bad).any() else None,
           "bad_caught_flag_or_block": round(float(np.mean(scores[bad] >= uncertain_at)), 3) if bad.any() else None,
           "bad_blocked": round(float(np.mean(scores[bad] >= block_at)), 3) if bad.any() else None}
    labels = sorted(set(y))
    out["per_class_blocked"] = {c: round(float(np.mean(scores[y == c] >= block_at)), 3) for c in labels}
    return out


def to_json(name, model, cfg, picked, rows, held_out, embedding_note) -> dict:
    classes = [str(c) for c in model.classes_]
    coef, intercept = model.coef_, model.intercept_
    if len(classes) == 2:
        # sklearn keeps one row for two classes (the second class's logit). Stored as two rows,
        # -z/2 and +z/2, so the app's softmax gives exactly the sigmoid sklearn uses.
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
        intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
    rounded = lambda a: [float(f"{v:.{WEIGHT_DIGITS}g}") for v in a]
    data = {
        "format": classifier.FORMAT, "name": name, "classes": classes, "allow": cfg["allow"],
        "bands": {"uncertain_at": picked["uncertain_at"], "block_at": picked["block_at"]},
        "embedding": {"model": rag_service.EMBEDDING_MODEL, "task_type": cfg["task_type"],
                      "dimensions": int(coef.shape[1])},
        "weights": [rounded(row) for row in coef], "bias": rounded(intercept),
        "training": {"trained_on": date.today().isoformat(), "examples": len(rows), "data_hash": data_hash(rows),
                     "per_class": {c: sum(1 for r in rows if r["label"] == c) for c in classes},
                     "C": picked["C"], "targets": {k: cfg[k] for k in ("max_flagged", "max_blocked", "min_block_at") if k in cfg},
                     "held_out": held_out, "embedding": embedding_note},
    }
    data["checksum"] = classifier.checksum_of(data)
    return data


def print_matrix(y_true, y_pred, labels):
    m = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(c) for c in labels) + 2
    print("label \\ predicted".ljust(width) + "".join(c[:10].rjust(12) for c in labels))
    for c, row in zip(labels, m):
        print(c.ljust(width) + "".join(str(v).rjust(12) for v in row))


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--data", type=Path, help="labeled examples (default services/safety/data/<model>.jsonl)")
    parser.add_argument("--max-requests", type=int, default=8, help="embedding requests this run may spend (default 8)")
    parser.add_argument("--dry-run", action="store_true", help="train and print, but don't write the model file")
    args = parser.parse_args()
    cfg = MODELS[args.model]
    path = args.data or DATA_DIR / f"{args.model}.jsonl"
    rows = read_examples(path)
    print(f"{len(rows)} examples from {path}")

    started = time.perf_counter()
    X, note = await embed([r["text"] for r in rows], cfg["task_type"], args.max_requests)
    await asyncio.sleep(0)       # let the usage counter's background writes run
    print(f"[embed] {note} in {time.perf_counter() - started:.0f} s")
    y = np.array([r["label"] for r in rows])
    allow = cfg["allow"]
    labels = sorted(set(y))

    # 1-3: held-out results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    picked = pick(X_train, y_train, allow, cfg)
    model = fit(X_train, y_train, picked["C"], allow)
    test_scores = block_scores(model, X_test, allow)
    held_out = {"examples": len(y_test), "C": picked["C"],
                "bands": {"uncertain_at": picked["uncertain_at"], "block_at": picked["block_at"]},
                "accuracy_most_likely_class": round(float(np.mean(model.predict(normalize(X_test)) == y_test)), 3),
                **results(test_scores, y_test, allow, picked["uncertain_at"], picked["block_at"])}
    print(f"\nHeld out ({len(y_test)} examples), model trained on the other {len(y_train)}:")
    print(json.dumps(held_out, indent=1))
    print_matrix(y_test, model.predict(normalize(X_test)), labels)

    # 4: the shipped model
    final = pick(X, y, allow, cfg)
    model = fit(X, y, final["C"], allow)
    cv = repeated_cv(X, y, final["C"], allow, final["uncertain_at"], final["block_at"])
    print(f"\nShipped setup, 10 x 5-fold cross-validation at its bands: {json.dumps(cv)}")
    data = to_json(args.model, model, cfg, final, rows, {**held_out, "cross_validated": cv}, note)
    # The app must score exactly as sklearn does
    app = classifier.from_dict(json.loads(json.dumps(data)), rag_service.EMBEDDING_MODEL, {cfg["task_type"]})
    diff = max(abs(app.block_score(app.probabilities(x)) - s) for x, s in zip(X, block_scores(model, X, allow)))
    print(f"\nShipped model: C={final['C']}, bands {final['uncertain_at']} / {final['block_at']}; "
          f"app vs sklearn largest difference {diff:.2e}")
    if diff > 1e-4:
        sys.exit("The app's scoring doesn't match sklearn's: not written.")
    if args.dry_run:
        return
    out = classifier.MODELS_DIR / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    from services import usage
    await usage.drain()


if __name__ == "__main__":
    asyncio.run(main())
