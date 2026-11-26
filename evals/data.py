"""
Feature 2 evals: reading the frozen data and its labels (.local/evals/, written by prepare_data.py).
Shared by review_labels.py, build_datasets.py and the experiment runner.

Labels come in two kinds, per topic: proposed (drafted by Claude, each with a reason) and reviewed
(saved from the review page by a person). A reviewed file replaces the proposed one completely.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / ".local" / "evals"
VIEWS = ("report", "timeline", "contradictions")

# Feature 3 N #109: every file read here, with the sha256 of exactly the text read. The experiment
# runner saves them with each run, so a run whose topics, cases or labels changed (a label reviewed
# again between two runs, say) shows it, instead of the metric moving for no visible reason.
_hashes: dict[str, str] = {}


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def name_of(path: Path) -> str:
    """A path as runs record it: relative to .local/evals (or to the repo), with forward slashes."""
    path = Path(path).resolve()
    for base in (EVALS.resolve(), ROOT.resolve()):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _read(path: Path):
    text = path.read_text(encoding="utf-8")
    _hashes[name_of(path)] = file_hash(text)
    return json.loads(text)


def files_read() -> dict[str, str]:
    """{file: sha256[:16]} of every data file read so far in this process, sorted."""
    return dict(sorted(_hashes.items()))


def forget_reads():
    _hashes.clear()


def topics() -> list[dict]:
    return [_read(p) for p in sorted((EVALS / "topics").glob("*.json"))]


def fact_check_cases() -> list[dict]:
    return [_read(p) for p in sorted((EVALS / "fact_check").glob("*.json"))]


def injection() -> dict:
    return _read(EVALS / "injection.json")


def label_path(kind: str, slug: str, status: str) -> Path:
    """kind: 'retrieval' or 'fact_check'; status: 'proposed' or 'reviewed'."""
    return EVALS / "labels" / status / f"{kind}_{slug}.json"


def labels(kind: str, slug: str) -> tuple[dict | None, str]:
    """(labels, status): the reviewed labels if a person saved them, else the proposed ones, else (None, 'missing')."""
    for status in ("reviewed", "proposed"):
        path = label_path(kind, slug, status)
        if path.exists():
            return _read(path), status
    return None, "missing"


def save_reviewed(kind: str, slug: str, data: dict) -> Path:
    path = label_path(kind, slug, "reviewed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def context(topic: dict, view: str) -> list[dict]:
    """The chunks the app sent the model for this view when the data was frozen, in order, with their source."""
    chunks = {c["id"]: c for c in topic["chunks"]}
    articles = {a["n"]: a for a in topic["articles"]}
    out = []
    for chunk_id in topic["retrieval"][view]:
        c = chunks[chunk_id]
        a = articles[c["article"]]
        out.append({"chunk_id": chunk_id, "title": a.get("title"), "source_name": a.get("source_name"),
                    "published_at": a.get("published_at"), "theme_id": a.get("theme_id"), "page_content": c["text"]})
    return out
