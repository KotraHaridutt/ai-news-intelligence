"""
Feature 4, Section E: which reranker the app should use, from the retrieval experiments' cached results.

  .local/venv/Scripts/python.exe evals/rerank_report.py table <baseline> <experiment> [<experiment> ...]
  .local/venv/Scripts/python.exe evals/rerank_report.py planted --rerank cohere --rerank-model rerank-v4.0-pro

table    reads each experiment's summary.json and outputs (no calls) and prints:
         1. the mean of every metric (the reranker alone: ndcg_full, mrr_full; what the answer gets: @k)
         2. per view: the metrics that view depends on (timeline: dated chunks; contradictions: stories)
         3. per topic: where each experiment beats or loses to the baseline on ndcg@k (>= 0.02 either way),
            so one topic can't decide alone (#26)
         4. time: the first view of each topic makes the reranker call (the other views read the cache)
         5. the gates from evals/thresholds.json against the baseline
planted  the planted-instruction topic (.local/evals/injection.json): where its injected chunks rank in the
         search's order and in the reranker's, and whether they reach the answer's 10 (#25). One reranker
         call per engine (cached).
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_PROJECT", "ai-news-evals")

from evals import data, run_evals  # noqa: E402

RUNS = run_evals.RUNS
ALONE = ("ndcg_full", "mrr_full")
ANSWER = ("ndcg@k", "recall@k", "precision@k", "useful_precision@k", "mrr", "dated_recall@k", "stories@k", "off_topic@k")
PER_VIEW = {"report": ("ndcg@k", "precision@k", "mrr"), "timeline": ("ndcg@k", "precision@k", "dated_recall@k"),
            "contradictions": ("ndcg@k", "precision@k", "stories@k")}
WIN = 0.02


def _summary(name: str) -> dict:
    path = RUNS / name / "summary.json"
    if not path.exists():
        sys.exit(f"{name}: no summary.json (run it to the end first)")
    return json.loads(path.read_text(encoding="utf-8"))


def _label(name: str) -> str:
    return json.loads((RUNS / name / "manifest.json").read_text(encoding="utf-8"))["label"]


def _mean(rows, metric, keep=lambda key: True):
    values = [r["metrics"][metric] for r in rows if metric in r["metrics"] and keep(r["key"])]
    return statistics.mean(values) if values else None


def _fmt(value, base=None):
    if value is None:
        return f"{'-':>15}"
    delta = f" ({value - base:+.3f})" if base is not None else ""
    return f"{value:6.3f}{delta:>9}"


def table(baseline: str, names: list[str]):
    runs = [baseline] + names
    sums = {n: _summary(n) for n in runs}
    labels = {n: _label(n) for n in runs}
    width = max(len(labels[n]) for n in runs) + 2
    base_rows = sums[baseline]["rows"]

    print("\n== 1. Mean over 15 examples (delta vs the baseline)")
    for group, metrics in (("the reranker alone (whole index, before the cap)", ALONE), ("what the answer gets (after the cap)", ANSWER)):
        print(f"\n   {group}")
        print(" " * (width + 3) + "".join(f"{m:>15}" for m in metrics))
        for n in runs:
            row = "".join(_fmt(_mean(sums[n]["rows"], m), None if n == baseline else _mean(base_rows, m)) for m in metrics)
            print(f"   {labels[n]:<{width}}{row}")

    print("\n== 2. Per view")
    for view, metrics in PER_VIEW.items():
        keep = lambda key, v=view: key.endswith("/" + v)
        print(f"\n   {view}")
        print(" " * (width + 3) + "".join(f"{m:>15}" for m in metrics))
        for n in runs:
            row = "".join(_fmt(_mean(sums[n]["rows"], m, keep), None if n == baseline else _mean(base_rows, m, keep)) for m in metrics)
            print(f"   {labels[n]:<{width}}{row}")

    print(f"\n== 3. Per topic: mean ndcg@k over the 3 views vs the baseline (win/loss = a change of {WIN} or more)")
    topics = sorted({r["key"].split("/")[0] for r in base_rows})
    print(" " * (width + 3) + "".join(f"{t[:14]:>15}" for t in topics) + f"{'wins':>7}{'losses':>8}")
    for n in runs:
        cells, wins, losses = [], 0, 0
        for t in topics:
            keep = lambda key, t=t: key.startswith(t + "/")
            value, base = _mean(sums[n]["rows"], "ndcg@k", keep), _mean(base_rows, "ndcg@k", keep)
            if n != baseline:
                wins += value - base >= WIN
                losses += base - value >= WIN
            cells.append(_fmt(value, None if n == baseline else base))
        print(f"   {labels[n]:<{width}}" + "".join(cells) + (f"{wins:>7}{losses:>8}" if n != baseline else ""))

    print("\n== 4. Time per fresh topic (the first view calls the reranker; the others read its cache)")
    for n in runs:
        outs = [json.loads(p.read_text(encoding="utf-8")) for p in (RUNS / n / "outputs").glob("*.json")]
        ms = [o.get("ms") for o in outs if o.get("ms") is not None]
        if not ms:
            print(f"   {labels[n]:<{width}} (no times recorded)")
            continue
        slow = sorted(ms, reverse=True)[:5]          # one first view per topic: the 5 slowest
        print(f"   {labels[n]:<{width}} first views {statistics.mean(slow) / 1000:.2f} s mean, {max(ms) / 1000:.2f} s max; "
              f"cached views {statistics.median(sorted(ms)[:10]) / 1000:.2f} s median")

    print("\n== 5. Gates (evals/thresholds.json) against the baseline")
    base = sums[baseline]["summary"]
    for n in names:
        broken = run_evals.gate("retrieval", sums[n]["summary"], base)
        print(f"   {labels[n]:<{width}} " + ("passes" if not broken else "BROKEN: " + "; ".join(broken)))


async def planted(rerank, rerank_model, rerank_method=None, rerank_query=None):
    from services import rag_service as rag, rerank_service as rr
    variant = run_evals.Variant(rerank=rerank, rerank_model=rerank_model, rerank_method=rerank_method, rerank_query=rerank_query)
    variant.apply()
    topic, store = await run_evals._store("injection")
    injected = set(topic["injected_chunks"])
    ids = [c["id"] for c in topic["chunks"]]
    searched = [d.metadata["chunk_id"] for d in await store.asimilarity_search(topic["query"], k=len(ids))]
    label = " ".join(x for x in (rerank_model or rerank or "search only", rerank_method, rerank_query and f"query={rerank_query}") if x)
    print(f"\n== planted instructions ({len(injected)} injected chunks of {len(ids)}), query {topic['query']!r}: {label}")
    print(f"   search's order: injected at ranks {sorted(searched.index(c) + 1 for c in injected)}")
    if rerank:
        result = await rr.rerank(topic["query"], [c["text"] for c in topic["chunks"]], label="eval:planted")
        if result is None or result.engine != rerank:
            sys.exit(f"not reranked by {rerank}")
        order = [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-result.scores[i], i))]
        print(f"   reranker's order: injected at ranks {sorted(order.index(c) + 1 for c in injected)}")
    for view in data.VIEWS:
        docs = await rag._retrieve(store, topic["query"], view)
        got = [d.metadata["chunk_id"] for d in docs]
        inside = sorted(got.index(c) + 1 for c in injected if c in got)
        print(f"   {view:15} injected chunks in the answer's 10: {len(inside)}" + (f" (at {inside})" if inside else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Feature 4 E: compare the rerankers' retrieval experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("baseline")
    t.add_argument("experiments", nargs="+")
    p = sub.add_parser("planted")
    p.add_argument("--rerank", choices=["cohere", "cloudflare"])
    p.add_argument("--rerank-model")
    p.add_argument("--rerank-method")
    p.add_argument("--rerank-query", choices=["plain", "search"])
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.cmd == "table":
        table(args.baseline, args.experiments)
    else:
        asyncio.run(planted(args.rerank, args.rerank_model, args.rerank_method, args.rerank_query))


if __name__ == "__main__":
    main()
