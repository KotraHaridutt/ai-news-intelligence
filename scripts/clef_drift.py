"""
Has Clef changed? A monthly rerun of a fixed calibration sample (feature 3, Section N #106a).

    python scripts/clef_drift.py                    # rerun the sample, compare with the baseline
    python scripts/clef_drift.py --record           # rerun and save it as the new baseline
    python scripts/clef_drift.py --chunks           # also the injection question on labeled article chunks

Cloudflare can update `clef` under the same name. Every call records the `model` Clef reports (#15a),
but a new build can keep the name, and the thresholds in services/safety/questions.py were calibrated on
the old one (#7b). So once a month this asks Clef the same questions about the same texts and compares
each block score with the baseline:

- intent: 40 queries from services/safety/data/intent.jsonl (8 per class, fixed seed). Its baseline,
  services/safety/data/clef_drift.json, is committed: the texts are already public in the repo, and the
  baseline stores only their hashes and scores.
- injection (--chunks): 6 planted and 6 normal article chunks from .local/safety_data/curated/chunks.jsonl.
  Real articles aren't ours to publish (services/safety/data/SOURCES.md), so this part and its baseline
  (.local/safety_data/clef_drift_chunks.json) stay on this machine.

Clef isn't deterministic: run twice on the same day (N §6), the same texts moved by 0.004 (intent) and
0.009 (chunks) on average, and by up to 0.086 and 0.101 for one text; a jailbreak query near the 0.3
line went from allow to flag (0.24 -> 0.33) that way. So the limits sit above that noise. Drift, which
exits 1: a mean change of more than MEAN_DELTA_MAX, one text moving more than ITEM_DELTA_MAX, or a band
(allow / flag / block at the question's thresholds) changing on a move larger than NOISE. A band change
on a smaller move is listed as within the noise. Then: rerun the question's calibration (#7b, B2 §2 /
A2 §5.2), adjust the thresholds if needed, and --record again.
A question whose wording changed since the baseline can't be compared: the script says so (re-record).

Cost: about 7-15 neurons a text (the wording plus the text), roughly 400 for intent and 150 for the
chunks, out of 10,000 a day. It stops before starting if that would pass the warning level.
"""
import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from services.safety import clef, engine, questions  # noqa: E402

INTENT_DATA = ROOT / "services" / "safety" / "data" / "intent.jsonl"
INTENT_BASELINE = ROOT / "services" / "safety" / "data" / "clef_drift.json"
CHUNK_DATA = ROOT / ".local" / "safety_data" / "curated" / "chunks.jsonl"
CHUNK_BASELINE = ROOT / ".local" / "safety_data" / "clef_drift_chunks.json"
PER_INTENT_CLASS = 8
PER_CHUNK_LABEL = 6
SEED = 106
# Same-day noise (N §6): mean 0.004 / 0.009, largest 0.086 / 0.101
MEAN_DELTA_MAX = 0.03
ITEM_DELTA_MAX = 0.2
NOISE = 0.12
NEURONS_PER_TEXT = 15          # a high estimate, for the budget check before starting


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    # split("\n"), not splitlines(): some rows contain U+2028 / U+0085 (as in train_safety_classifier.py)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sample(rows: list[dict], per_label: int, seed: int = SEED) -> list[dict]:
    """A fixed sample: `per_label` rows of each label (or all of a smaller label), in a seeded order."""
    rng = random.Random(seed)
    picked = []
    for label in sorted({r["label"] for r in rows}):
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: text_id(r["text"]))
        picked += rng.sample(group, min(per_label, len(group)))
    return picked


def compare(baseline: list[dict], now: dict[str, float], question) -> dict:
    """{"changed_bands": [...], "noise_bands": [...], "mean_delta", "max_delta", "missing": [...], "drift": bool}.
    A band change counts only when the text moved more than NOISE; smaller ones go in noise_bands."""
    deltas, changed, noise, missing = [], [], [], []
    for item in baseline:
        score = now.get(item["id"])
        if score is None:
            missing.append(item["id"])
            continue
        delta = abs(score - item["score"])
        deltas.append(delta)
        before, after = question.band(item["score"]), question.band(score)
        if before != after:
            entry = {"id": item["id"], "label": item["label"], "before": round(item["score"], 4),
                     "after": round(score, 4), "band": f"{before} -> {after}"}
            (changed if delta > NOISE else noise).append(entry)
    mean = sum(deltas) / len(deltas) if deltas else 0.0
    biggest = max(deltas, default=0.0)
    return {"compared": len(deltas), "changed_bands": changed, "noise_bands": noise, "mean_delta": round(mean, 4),
            "max_delta": round(biggest, 4), "missing": missing,
            "drift": bool(changed) or mean > MEAN_DELTA_MAX or biggest > ITEM_DELTA_MAX}


async def score_all(question, texts: dict[str, str]) -> tuple[dict[str, float], set[str], float]:
    """{id: block score} asked one text per call, exactly as the app asks (engine._ask_one), the models
    Clef reported, and the neurons spent."""
    before = clef.neurons_today()
    models = set()

    async def one(key: str, text: str):
        state, asked = engine._call_of(question, "item", {"item": text}, {})
        reply = await clef.ask(state, {question.clef_name: question.payload(asked)}, timeout=max(question.timeout_s, 10.0),
                               label=f"clef_drift:{question.name}")
        models.update(reply["models"])
        return key, question.block_score(reply["answers"][question.clef_name])

    results = await asyncio.gather(*(one(k, t) for k, t in texts.items()), return_exceptions=True)
    scores = {}
    for result in results:
        if isinstance(result, BaseException):
            print(f"[drift] a call failed: {type(result).__name__}: {str(result)[:150]}")
            continue
        scores[result[0]] = result[1]
    await clef.drain()
    return scores, models, round(clef.neurons_today() - before, 1)


async def run_part(name: str, question, rows: list[dict], per_label: int, baseline_path: Path, record: bool) -> bool | None:
    """One question's part. Returns True when it drifted, False when not, None when it couldn't compare."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    by_id = {text_id(r["text"]): r for r in rows}
    if record or baseline is None:
        chosen = {text_id(r["text"]): r for r in sample(rows, per_label)}
    else:
        chosen = {item["id"]: by_id[item["id"]] for item in baseline["items"] if item["id"] in by_id}
    scores, models, neurons = await score_all(question, {k: r["text"] for k, r in chosen.items()})
    print(f"[drift] {name}: {len(scores)}/{len(chosen)} texts scored, {neurons} neurons, Clef reported {sorted(models) or '?'}")
    if record or baseline is None:
        if len(scores) < len(chosen):
            print(f"[drift] {name}: not recorded, {len(chosen) - len(scores)} calls failed")
            return None
        data = {"question": question.name, "question_version": question.version, "bands": [question.uncertain_at, question.block_at],
                "model_reported": sorted(models), "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": SEED, "per_label": per_label,
                "items": [{"id": k, "label": chosen[k]["label"], "score": round(scores[k], 6)} for k in chosen]}
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"[drift] {name}: baseline {'recorded' if record else 'created (there was none)'} in "
              f"{baseline_path.relative_to(ROOT)}")
        return False
    if baseline.get("question_version") != question.version:
        print(f"[drift] {name}: the question's wording changed since the baseline ({baseline.get('question_version')} -> "
              f"{question.version}): scores can't be compared. Recalibrate, then --record.")
        return None
    result = compare(baseline["items"], scores, question)
    if set(baseline.get("model_reported") or []) != models and models:
        print(f"[drift] {name}: Clef now reports {sorted(models)}, the baseline {baseline.get('model_reported')}")
    print(f"[drift] {name}: {result['compared']} compared with {baseline['recorded_at'][:10]}: mean change "
          f"{result['mean_delta']}, largest {result['max_delta']}, {len(result['changed_bands'])} band changes beyond the noise")
    for item in result["changed_bands"]:
        print(f"[drift]   {item['id']} ({item['label']}): {item['before']} -> {item['after']} ({item['band']})")
    for item in result["noise_bands"]:
        print(f"[drift]   near a threshold, within the noise ({NOISE}): {item['id']} ({item['label']}) "
              f"{item['before']} -> {item['after']} ({item['band']})")
    if result["missing"]:
        print(f"[drift] {name}: {len(result['missing'])} baseline texts weren't scored (removed from the data, or a call failed)")
    print(f"[drift] {name}: " + ("DRIFT: rerun the calibration (#7b) before trusting the thresholds" if result["drift"]
                                  else f"no drift (limits: mean {MEAN_DELTA_MAX}, one text {ITEM_DELTA_MAX}, a band change beyond {NOISE})"))
    return result["drift"]


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="save this run as the new baseline")
    parser.add_argument("--chunks", action="store_true", help="also the injection question on local article chunks")
    args = parser.parse_args(argv)
    if not clef.available():
        print(f"[drift] {clef.status()}")
        return 2
    parts = [("intent", questions.get("intent"), read_jsonl(INTENT_DATA), PER_INTENT_CLASS, INTENT_BASELINE)]
    if args.chunks:
        if not CHUNK_DATA.exists():
            print(f"[drift] no {CHUNK_DATA.relative_to(ROOT)} on this machine: the chunk part is skipped")
        else:
            parts.append(("injection", questions.get("injection"), read_jsonl(CHUNK_DATA), PER_CHUNK_LABEL, CHUNK_BASELINE))
    await clef.load_today()
    texts = sum(len({r["label"] for r in rows}) * per for _, _, rows, per, _ in parts)
    if clef.neurons_today() + texts * NEURONS_PER_TEXT > clef.WARN_AT_NEURONS:
        print(f"[drift] {clef.neurons_today():.0f} neurons used today; ~{texts * NEURONS_PER_TEXT} more would pass the warning "
              f"level ({clef.WARN_AT_NEURONS}). Run it after 00:00 UTC.")
        return 2
    started = time.perf_counter()
    results = [await run_part(*part, args.record) for part in parts]
    print(f"[drift] done in {time.perf_counter() - started:.0f} s; {clef.neurons_today():.0f} neurons used today")
    if any(r is True for r in results):
        return 1
    return 2 if any(r is None for r in results) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
