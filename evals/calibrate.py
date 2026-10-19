"""
Feature 2, Section G (#56): is the judge right often enough to be trusted?

  .local/venv/Scripts/python.exe evals/calibrate.py prepare   build the calibration set (5 chat calls: new answers from frozen contexts)
  .local/venv/Scripts/python.exe evals/calibrate.py judge     run the judges on it (9 chat calls), save their verdicts
  .local/venv/Scripts/python.exe evals/calibrate.py score     compare the verdicts with the labels (no calls; rerun after reviewing)

The set: 7 answers (2 real answers from gap 4, 5 new ones from Section F's frozen contexts: a report,
a timeline, two contradictions, and one on the injection context) and the two real fact checks
feature 1 and Section A2 ran on Tesla. Each answer is split into units (judges.units); labels say, per unit, what
the judge should answer. A judge field is trusted when it agrees with REVIEWED labels on at least 80%
of at least 10 comparable units; on proposed labels the result is only provisional.

Writes to .local/evals/calibration/: items.json, judged/<id>.json, agreement.json.
"""
import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_PROJECT", "ai-news-evals")     # eval traffic isn't app traffic

from evals import data

CAL = data.EVALS / "calibration"
TESTS = ROOT / ".local" / "tests"
THRESHOLD, MIN_UNITS = 0.80, 10

# The new answers: (topic, view); "injection" uses the injection context
NEW = [("tesla", "report"), ("climate_change", "timeline"), ("openai", "contradictions"),
       ("artificial_intelligence", "contradictions"), ("injection", "report")]
# Fields compared, and the units where each is comparable (by the label)
ANSWER_FIELDS = {"factual": lambda lab: True,
                 "supported": lambda lab: lab.get("factual") is True,
                 "outlet_correct": lambda lab: lab.get("outlet_correct") is not None,
                 "disagreement_real": lambda lab: lab.get("disagreement") is True}
FACT_FIELDS = {"factual": lambda lab: True,
               "already_told": lambda lab: lab.get("factual") is True,
               "supported": lambda lab: lab.get("factual") is True}


def _context_from_prompt(prompt: str) -> list[dict]:
    """The excerpts of a saved gap-4 prompt, each after a [Source: ...] header: between <articles> tags in
    today's prompts, between CONTEXT: and QUERY: in gap 4 Section B's (before the tags were added)."""
    if "<articles>" in prompt:
        block = prompt.split("<articles>", 1)[1].rsplit("</articles>", 1)[0]
    else:
        block = prompt.split("CONTEXT:", 1)[1].rsplit("QUERY:", 1)[0]
    out = []
    for m in re.finditer(r"\[Source: (?P<title>[^|\]]*)\|(?P<outlet>[^|\]]*)\| Published: (?P<pub>[^\]]*)\]\n(?P<text>.*?)(?=\n\n\[Source: |\Z)", block, re.S):
        pub = m.group("pub").strip().replace(" UTC", "")
        out.append({"title": m.group("title").strip(), "source_name": m.group("outlet").strip(),
                    "published_at": pub.replace(" ", "T") + ("Z" if len(pub) > 10 else ""), "page_content": m.group("text").strip()})
    return out


def _save_items(items):
    CAL.mkdir(parents=True, exist_ok=True)
    (CAL / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")


async def prepare():
    """
    Builds the set step by step and saves after every new answer, so a run that stops (Gemini
    overloaded: 504s were seen on 2026-10-19) keeps what it made; the next run only makes the rest.
    """
    from evals import judges, targets
    done = {}
    if (CAL / "items.json").exists():
        done = {i["id"]: i for i in json.loads((CAL / "items.json").read_text(encoding="utf-8"))}
    items = []
    saved = json.loads((TESTS / "answers_after-B.json").read_text(encoding="utf-8"))
    for view in ("report", "timeline"):
        t = saved["types"][view]
        items.append({"id": f"gap4-stock_market-{view}", "kind": "answer", "query": saved["topic"], "view": view,
                      "context": _context_from_prompt(t["prompt"]), "answer": t["answer"], "source": "gap 4 after-B (2026-10-11)"})
    topics = {t["slug"]: t for t in data.topics()}
    missing = []
    for slug, view in NEW:
        item_id = f"new-{slug}-{view}"
        if item_id in done:
            items.append(done[item_id])
            continue
        topic = data.injection() if slug == "injection" else topics[slug]
        context = data.context(topic, view)
        try:
            result = await targets.generate_answer(topic["query"], view, context)
        except Exception as e:
            print(f"[G] {slug}/{view}: not generated ({type(e).__name__}); run prepare again later")
            missing.append(item_id)
            continue
        items.append({"id": item_id, "kind": "answer", "query": topic["query"], "view": view, "context": context,
                      "answer": result["answer"], "source": f"generated {datetime.now(timezone.utc).date()} (prompt {result['prompt_version']})",
                      **({"signs": topic["signs"]} if slug == "injection" else {})})
        print(f"[G] {slug}/{view}: {len(result['answer'])} characters")
        _save_items(_with_units(items, judges))
    case = next(c for c in data.fact_check_cases() if c["slug"] == "tesla")
    for n, run in enumerate(case.get("earlier_model_answers", []), 1):
        items.append({"id": f"tesla-fact-check-run{n}", "kind": "fact_check", "query": case["query"],
                      "stories": case["stories"], "texts": run["texts"], "told": "\n\n".join(t["text"] for t in case["told"])})
    _save_items(_with_units(items, judges))
    print(f"[G] calibration set: {len(items)} items, {sum(len(i['units']) for i in items)} units"
          + (f"; still missing: {', '.join(missing)}" if missing else ""))


def _with_units(items, judges):
    for item in items:
        item["units"] = (judges.units(item["answer"]) if item["kind"] == "answer"
                         else [{"story_id": sid, "text": u} for sid, text in item["texts"].items() for u in judges.units(text)])
    return items


async def judge():
    from evals import judges
    items = json.loads((CAL / "items.json").read_text(encoding="utf-8"))
    (CAL / "judged").mkdir(parents=True, exist_ok=True)
    for item in items:
        path = CAL / "judged" / f"{item['id']}.json"
        if path.exists():
            print(f"[G] {item['id']}: already judged")
            continue
        if item["kind"] == "answer":
            result = await judges.judge_answer(item["query"], item["view"], item["answer"], item["context"])
        else:
            result = await judges.judge_fact_check(item["query"], item["stories"], item["texts"], item["told"])
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[G] {item['id']}: {result['scores']}")


def score() -> dict:
    items = json.loads((CAL / "items.json").read_text(encoding="utf-8"))
    tallies, per_item = {}, {}
    statuses = set()
    for item in items:
        judged_path = CAL / "judged" / f"{item['id']}.json"
        labels, status = data.labels("judge", item["id"])
        if not judged_path.exists() or labels is None:
            continue
        statuses.add(status)
        verdict = json.loads(judged_path.read_text(encoding="utf-8"))["verdict"]
        fields = ANSWER_FIELDS if item["kind"] == "answer" else FACT_FIELDS
        per_item[item["id"]] = {}
        for unit_id, lab in labels["units"].items():
            got = verdict.get(str(unit_id)) or verdict.get(int(unit_id)) if isinstance(verdict, dict) else None
            for field, comparable in fields.items():
                if not comparable(lab):
                    continue
                key = f"{item['kind']}.{field}"
                t = tallies.setdefault(key, {"agree": 0, "n": 0, "disagreements": []})
                t["n"] += 1
                if got is not None and got.get(field) == lab.get(field):
                    t["agree"] += 1
                else:
                    t["disagreements"].append(f"{item['id']}#{unit_id}")
    reviewed = statuses == {"reviewed"}
    report = {}
    for key, t in sorted(tallies.items()):
        agreement = round(t["agree"] / t["n"], 4) if t["n"] else None
        verdict = ("trusted" if reviewed and t["n"] >= MIN_UNITS and agreement >= THRESHOLD else
                   "not trusted" if reviewed and t["n"] >= MIN_UNITS else
                   "too few units" if t["n"] < MIN_UNITS else
                   f"provisional: {'passes' if agreement >= THRESHOLD else 'fails'} on proposed labels")
        report[key] = {"agreement": agreement, "units": t["n"], "status": verdict, "disagreements": t["disagreements"][:20]}
    out = {"scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "labels": sorted(statuses),
           "threshold": THRESHOLD, "min_units": MIN_UNITS, "fields": report}
    (CAL / "agreement.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key, r in report.items():
        print(f"  {key:28} agreement {r['agreement']}  on {r['units']:3} units  -> {r['status']}")
    return out


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Calibrate the LLM judges against labels.")
    parser.add_argument("step", choices=["prepare", "judge", "score"])
    step = parser.parse_args().step
    if step == "score":
        score()
    else:
        asyncio.run(prepare() if step == "prepare" else judge())
