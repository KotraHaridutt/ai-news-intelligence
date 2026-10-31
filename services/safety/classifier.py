import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Fallback engine: embedding classifier (feature 3, Section B #15) ---
# When Clef can't run, a check is scored by a logistic regression over the text's Gemini
# embedding. It works on meaning, not words: texts that mean the same thing get close vectors.
#
# It scores the vectors the pipeline already makes (A1 §4): RETRIEVAL_QUERY for a query (the
# pipeline embeds and caches it anyway, to rank articles) and RETRIEVAL_DOCUMENT for a chunk (the
# index's own vectors). So the fallback normally costs no extra embedding request, out of the
# free tier's 100 a day.
#
# The weights are plain numbers in JSON (models/<name>.json, written by
# scripts/train_safety_classifier.py), never a pickle: loading a pickle can run code. The file
# carries a checksum and the embedding model, task type and size it was trained on. A file that
# doesn't match its checksum, or was trained on other vectors than the app makes, refuses to load
# instead of giving wrong scores.

MODELS_DIR = Path(__file__).resolve().parent / "models"
FORMAT = 1


class ClassifierError(Exception):
    """A classifier file that can't be used: missing, changed, or trained on other vectors."""


def checksum_of(data: dict) -> str:
    """sha256 of the file's content without its checksum field, in a fixed key order."""
    body = {k: v for k, v in data.items() if k != "checksum"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class Classifier:
    name: str
    classes: list[str]
    weights: np.ndarray        # classes x dimensions
    bias: np.ndarray           # classes
    allow: tuple               # the classes that are fine; block score = 1 - their summed probability
    uncertain_at: float
    block_at: float
    embedding_model: str
    task_type: str
    dimensions: int
    info: dict                 # training details (data hash, held-out results), for the startup line and docs

    def probabilities(self, vector) -> dict[str, float]:
        """Softmax over the classes. A two-class model is stored as two rows, so this is its sigmoid."""
        x = np.asarray(vector, dtype=np.float64)
        if x.shape != (self.dimensions,):
            raise ClassifierError(f"{self.name}: a vector of {x.shape} where {self.dimensions} numbers are expected")
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        logits = self.weights @ x + self.bias
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}

    def block_score(self, probabilities: dict[str, float]) -> float:
        return max(0.0, min(1.0, 1.0 - sum(probabilities.get(c, 0.0) for c in self.allow)))

    def band(self, score: float) -> str:
        if score >= self.block_at:
            return "block"
        if score >= self.uncertain_at:
            return "flag"
        return "allow"


def from_dict(data: dict, expect_model: str, expect_tasks: set[str]) -> Classifier:
    """Checks a classifier file's content and builds it. Raises ClassifierError."""
    name = data.get("name", "?")
    if data.get("format") != FORMAT:
        raise ClassifierError(f"{name}: format {data.get('format')!r}, this app reads {FORMAT}")
    if data.get("checksum") != checksum_of(data):
        raise ClassifierError(f"{name}: checksum doesn't match (the file was changed after training)")
    embedding = data.get("embedding") or {}
    if embedding.get("model") != expect_model:
        raise ClassifierError(f"{name}: trained on {embedding.get('model')!r} vectors, the app makes {expect_model!r}. Retrain it.")
    if embedding.get("task_type") not in expect_tasks:
        raise ClassifierError(f"{name}: trained on task type {embedding.get('task_type')!r}, the app makes {sorted(expect_tasks)}")
    weights = np.asarray(data["weights"], dtype=np.float64)
    bias = np.asarray(data["bias"], dtype=np.float64)
    classes = list(data["classes"])
    if weights.shape != (len(classes), embedding.get("dimensions")) or bias.shape != (len(classes),):
        raise ClassifierError(f"{name}: weights {weights.shape} don't match {len(classes)} classes x {embedding.get('dimensions')}")
    allow = tuple(data["allow"])
    if not set(allow) <= set(classes):
        raise ClassifierError(f"{name}: allow {allow} isn't among the classes {classes}")
    bands = data["bands"]
    return Classifier(name=name, classes=classes, weights=weights, bias=bias, allow=allow,
                      uncertain_at=float(bands["uncertain_at"]), block_at=float(bands["block_at"]),
                      embedding_model=embedding["model"], task_type=embedding["task_type"],
                      dimensions=int(embedding["dimensions"]), info=data.get("training") or {})


# --- Loading: once, on first use or at startup ---
_loaded: dict[str, Classifier] = {}
_failed: dict[str, str] = {}
_done = False


def load_all(directory: Path = MODELS_DIR) -> tuple[dict[str, Classifier], dict[str, str]]:
    """Loads every models/*.json. Returns (classifiers, {name: why it didn't load}). Never raises."""
    global _done
    # Imported here: rag_service imports the safety package (Section D), so a top-level import
    # would be circular. Its constants are what the app's vectors really are.
    from services import rag_service
    tasks = {rag_service.QUERY_TASK_TYPE, rag_service.DOCUMENT_TASK_TYPE}
    _loaded.clear()
    _failed.clear()
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            clf = from_dict(data, rag_service.EMBEDDING_MODEL, tasks)
            _loaded[clf.name] = clf
        except (ClassifierError, ValueError, KeyError, TypeError) as e:
            _failed[path.stem] = str(e)
            print(f"[WARN] Safety: the {path.stem} classifier isn't used: {e}")
    _done = True
    return dict(_loaded), dict(_failed)


def get(name: str) -> Classifier | None:
    if not _done:
        load_all()
    return _loaded.get(name)


def status() -> str:
    if not _done:
        load_all()
    if not _loaded:
        return "no fallback classifiers"
    return "fallback: " + ", ".join(sorted(_loaded))


async def vectors_for(clf: Classifier, texts: list[str]) -> list[np.ndarray]:
    """
    The texts' vectors, of the task type the classifier was trained on, through the pipeline's
    own embedding cache: a query or chunk the pipeline already embedded costs nothing, and one
    embedded here is reused by the pipeline. Raises if the embedding API fails.
    """
    from services import rag_service
    if clf.task_type == rag_service.QUERY_TASK_TYPE:
        return [await rag_service.get_query_vector(text) for text in texts]
    return await rag_service.get_vectors(texts)
