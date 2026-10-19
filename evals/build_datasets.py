"""
Feature 2, Section F (#45-#49): creates or updates the four LangSmith datasets from the frozen data
and labels in .local/evals/ (see prepare_data.py and review_labels.py).

  .local/venv/Scripts/python.exe evals/build_datasets.py             upload (only what changed)
  .local/venv/Scripts/python.exe evals/build_datasets.py --dry-run   show what would change

  ai-news-retrieval    topic x view: which chunks belong in an answer (grades 0/1/2, dated chunks)
  ai-news-answers      topic x view: the frozen context the model gets, and the grades of its chunks
  ai-news-fact-check   one "Check for updates" per topic: the stories, what was told, the expected outcome per story
  ai-news-injection    view: a context with a planted instruction, and what a hijacked answer would contain

Each example has a stable key (metadata "key"), so a rerun creates, updates or deletes only what
changed, and every example says whether its labels were "reviewed" by a person or only "proposed".
Each build tags the dataset version "build-<UTC time>", so an experiment can name the exact data it
ran on. The datasets are private and hold article text from news sites: they stay in LangSmith and
.local/, never in git (#50).
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["LANGSMITH_TRACING"] = "false"

from evals import data

NOTE = "Private: holds article text scraped from news sites. Never make public or commit to git."
DATASETS = {
    "ai-news-retrieval": "Which chunks belong in an answer, per topic and view (grades 0/1/2, dated chunks). " + NOTE,
    "ai-news-answers": "The frozen context the model gets per topic and view, with the grades of its chunks. " + NOTE,
    "ai-news-fact-check": "One 'Check for updates' per topic: stories to judge, what was told, expected outcomes. " + NOTE,
    "ai-news-injection": "A context with a planted instruction; what a hijacked answer would contain. " + NOTE,
}


def retrieval_examples() -> list[dict]:
    out = []
    for topic in data.topics():
        labels, status = data.labels("retrieval", topic["slug"])
        grades = {cid: v["grade"] for cid, v in (labels or {}).get("chunks", {}).items()}
        dated = sorted(cid for cid, v in (labels or {}).get("chunks", {}).items() if v.get("timeline"))
        for view in data.VIEWS:
            out.append({
                "inputs": {"topic": topic["slug"], "query": topic["query"], "view": view},
                "outputs": {"grades": grades, "relevant": sorted(c for c, g in grades.items() if g == 2),
                            "useful": sorted(c for c, g in grades.items() if g >= 1), "dated": dated},
                "metadata": {"key": f"{topic['slug']}/{view}", "topic": topic["slug"], "view": view, "labels": status,
                             "chunks": len(topic["chunks"]), "frozen_at": topic["frozen_at"],
                             "app_retrieval_at_freeze": topic["retrieval"][view]},
            })
    return out


def answer_examples() -> list[dict]:
    out = []
    for topic in data.topics():
        labels, status = data.labels("retrieval", topic["slug"])
        chunk_labels = (labels or {}).get("chunks", {})
        for view in data.VIEWS:
            ctx = data.context(topic, view)
            out.append({
                "inputs": {"query": topic["query"], "view": view, "context": ctx},
                "outputs": {"context_grades": {c["chunk_id"]: chunk_labels.get(c["chunk_id"], {}).get("grade") for c in ctx},
                            "dated_chunks": [c["chunk_id"] for c in ctx if chunk_labels.get(c["chunk_id"], {}).get("timeline")],
                            "outlets": sorted({c["source_name"] for c in ctx if c["source_name"]})},
                "metadata": {"key": f"{topic['slug']}/{view}", "topic": topic["slug"], "view": view, "labels": status,
                             "prompt_version_at_freeze": topic["prompt_versions"][view],
                             "config_version_at_freeze": topic["config_version"], "frozen_at": topic["frozen_at"]},
            })
    return out


def _everything_told(case) -> list[dict]:
    """
    "Already told" holding the very excerpts the fact check reads for every story (the first
    CHUNKS_PER_ARTICLE chunks of each article): whatever the stories say, the reader was told it.
    """
    from services.fact_check_service import CHUNKS_PER_ARTICLE
    from services.rag_service import text_splitter
    text = "\n\n".join(chunk for story in case["stories"] for article in story["articles"]
                        for chunk in text_splitter.split_text(article.get("full_text") or "")[:CHUNKS_PER_ARTICLE])
    return [{"at": case["check_window_from"], "kind": "report", "text": text}]


def fact_check_examples() -> list[dict]:
    """
    Per topic, the real check (labels: reviewed or proposed) and a negative made from it by
    construction: the same stories after the reader was told all of them, so every story must
    come back "nothing new". The real cases are mostly new stories (day 2 reported other events
    than day 1), so without these the dataset couldn't show a model that never hides anything.
    """
    out = []
    for case in data.fact_check_cases():
        out.append({
            "inputs": {"query": case["query"], "stories": case["stories"], "known_stories": case["known_stories"],
                       "told": _everything_told(case)},
            "outputs": {"expected": {s["id"]: "nothing_new" for s in case["stories"]}, "new_facts": {s["id"]: [] for s in case["stories"]}},
            "metadata": {"key": f"{case['slug']}/all-told", "topic": case["slug"], "labels": "by construction",
                         "stories": len(case["stories"]), "prompt_version_at_freeze": case["prompt_version"],
                         "config_version_at_freeze": case["config_version"], "frozen_at": case["frozen_at"]},
        })
        labels, status = data.labels("fact_check", case["slug"])
        stories = (labels or {}).get("stories", {})
        out.append({
            "inputs": {"query": case["query"], "stories": case["stories"], "known_stories": case["known_stories"],
                       "told": case["told"]},
            "outputs": {"expected": {sid: v["expected"] for sid, v in stories.items()},
                        "new_facts": {sid: v.get("new_facts", []) for sid, v in stories.items()}},
            "metadata": {"key": f"{case['slug']}/check", "topic": case["slug"], "labels": status, "stories": len(case["stories"]),
                         "prompt_version_at_freeze": case["prompt_version"], "config_version_at_freeze": case["config_version"],
                         "v1_built": case["v1_built"], "check_window_from": case["check_window_from"], "frozen_at": case["frozen_at"]},
        })
    return out


def injection_examples() -> list[dict]:
    inj = data.injection()
    return [{
        "inputs": {"query": inj["query"], "view": view, "context": data.context(inj, view)},
        "outputs": {"must_not_contain": inj["signs"], "injected_chunks": inj["injected_chunks"]},
        "metadata": {"key": f"{inj['slug']}/{view}", "view": view, "labels": "by construction", "frozen_at": inj["frozen_at"]},
    } for view in data.VIEWS]


BUILDERS = {"ai-news-retrieval": retrieval_examples, "ai-news-answers": answer_examples,
            "ai-news-fact-check": fact_check_examples, "ai-news-injection": injection_examples}


def sync(client, name: str, examples: list[dict], dry_run: bool) -> dict:
    """Creates, updates or deletes examples by their key. Returns the counts."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
        existing = {e.metadata.get("key"): e for e in client.list_examples(dataset_id=dataset.id)}
    else:
        dataset, existing = None, {}
    wanted = {e["metadata"]["key"]: e for e in examples}
    # LangSmith adds "dataset_split" to every example's metadata: not ours, so not compared
    ours = lambda metadata: {k: v for k, v in (metadata or {}).items() if k != "dataset_split"}
    create = [e for k, e in wanted.items() if k not in existing]
    update = [(existing[k], e) for k, e in wanted.items() if k in existing
              and (existing[k].inputs != e["inputs"] or existing[k].outputs != e["outputs"] or ours(existing[k].metadata) != e["metadata"])]
    delete = [e for k, e in existing.items() if k not in wanted]
    counts = {"create": len(create), "update": len(update), "delete": len(delete), "unchanged": len(wanted) - len(create) - len(update)}
    if dry_run:
        return counts
    if dataset is None:
        dataset = client.create_dataset(name, description=DATASETS[name])
    if create:
        client.create_examples(dataset_id=dataset.id, examples=create)
    if update:
        client.update_examples(dataset_id=dataset.id, updates=[{"id": old.id, **new} for old, new in update])
    if delete:
        client.delete_examples([e.id for e in delete])
    if create or update or delete:
        tag = "build-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        client.update_dataset_tag(dataset_id=dataset.id, as_of=datetime.now(timezone.utc), tag=tag)
        counts["tag"] = tag
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create or update the evaluation datasets in LangSmith.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set (put it in .env).")
    from langsmith import Client
    client = Client()
    results = {}
    for name, build in BUILDERS.items():
        examples = build()
        statuses = sorted({e["metadata"]["labels"] for e in examples})
        results[name] = sync(client, name, examples, args.dry_run)
        print(f"  {name:20} {len(examples):3} examples, labels {'/'.join(statuses)}: {results[name]}")
    if args.dry_run:
        print("Dry run: nothing changed.")
    return results


if __name__ == "__main__":
    main()
