"""
Feature 2, Section I (#65): the LLM judge on real answers, run by hand (decision #65: it can't spend
quota without you knowing).

  .local/venv/Scripts/python.exe evals/judge_yesterday.py --dry-run          what would be judged, no calls
  .local/venv/Scripts/python.exe evals/judge_yesterday.py                    yesterday (UTC) in ai-news-prod
  .local/venv/Scripts/python.exe evals/judge_yesterday.py --project ai-news-dev --day 2026-10-19 --max 5

For each answer trace of that day (POST /query, /api/timeline, /api/contradictions that succeeded),
the answer and the excerpts are read back from the trace's LLM run (the exact prompt the model got),
the answer judge runs (1 call each, inside the budget guard of Section H), and its scores go back on
that trace as feedback: judge.faithfulness, judge.citation_correctness, judge.relevance,
judge.contradictions_real, each with the judge version. Traces already judged are skipped, so a
second run the same day continues where the budget stopped the first. Fact checks are covered by
the code health checks on every check (#66), not judged here.
"""
import argparse
import asyncio
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_PROJECT", "ai-news-evals")     # the judge's own calls are eval traffic

ANSWER_ROUTES = {"POST /query": "report", "POST /api/timeline": "timeline", "POST /api/contradictions": "contradictions"}
# runs.query returns only the fields asked for (every other field is None): found in Section I
ROOT_FIELDS = ["id", "name", "start_time", "trace_id", "error", "inputs", "feedback_stats"]
LLM_FIELDS = ["id", "start_time", "error", "inputs", "outputs"]
JUDGE_KEYS = ("faithfulness", "citation_correctness", "relevance", "contradictions_real")


def prompt_of(llm_run) -> str | None:
    """The prompt text of a traced LangChain chat-model run (messages[0][0], in its serialized forms)."""
    inputs = getattr(llm_run, "inputs", None) or {}
    try:
        message = inputs["messages"][0][0]
    except (KeyError, IndexError, TypeError):
        return inputs.get("prompts", [None])[0] if isinstance(inputs.get("prompts"), list) else None
    if isinstance(message, dict):
        return (message.get("kwargs") or {}).get("content") or message.get("content")
    return getattr(message, "content", None)


def answer_of(llm_run) -> str | None:
    outputs = getattr(llm_run, "outputs", None) or {}
    try:
        generation = outputs["generations"][0][0] if isinstance(outputs["generations"][0], list) else outputs["generations"][0]
    except (KeyError, IndexError, TypeError):
        return None
    message = generation.get("message") or {}
    content = (message.get("kwargs") or {}).get("content") if isinstance(message, dict) else None
    if isinstance(content, list):          # Gemini 3: a list of parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or generation.get("text")


def already_judged(root) -> bool:
    stats = getattr(root, "feedback_stats", None) or {}
    return any(key.startswith("judge.") for key in stats)


async def candidates(client, project: str, day: date) -> list:
    """The day's successful answer requests (root runs), newest first."""
    project_id = str(client.read_project(project_name=project).id)
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    page = await client.runs.query(project_ids=[project_id], is_root=True, min_start_time=start,
                                   max_start_time=start + timedelta(days=1), page_size=100, selects=ROOT_FIELDS)
    roots = []
    while True:
        roots += [r for r in page.items if r.name in ANSWER_ROUTES and not r.error]
        if not page.has_next_page():
            break
        page = await page.get_next_page()
    return roots


async def llm_run_of(client, project: str, trace_id) -> object | None:
    """The trace's chat-model run that produced the answer (the last successful one, after any retry)."""
    project_id = str(client.read_project(project_name=project).id)
    page = await client.runs.query(project_ids=[project_id], trace_id=str(trace_id), run_type="llm", page_size=20, selects=LLM_FIELDS)
    runs = [r for r in page.items if not r.error]
    return sorted(runs, key=lambda r: r.start_time)[-1] if runs else None


async def main(project: str, day: date, max_items: int, reserve: int, dry_run: bool):
    from langsmith import Client
    from evals import calibrate, judges, run_evals
    client = Client()
    project_id = str(client.read_project(project_name=project).id)
    roots = await candidates(client, project, day)
    todo = [r for r in roots if not already_judged(r)][:max_items]
    print(f"[I] {project} on {day}: {len(roots)} answers, {len(roots) - len([r for r in roots if not already_judged(r)])} already judged, "
          f"{len(todo)} to judge now (max {max_items})")
    judged = 0
    for root in todo:
        view = ANSWER_ROUTES[root.name]
        llm = await llm_run_of(client, project, root.trace_id)
        prompt, answer = (prompt_of(llm), answer_of(llm)) if llm else (None, None)
        if not prompt or not answer:
            print(f"[I]   {root.trace_id}: no prompt/answer in the trace (sampled out or failed), skipped")
            continue
        context = calibrate._context_from_prompt(prompt)
        query = (root.inputs or {}).get("query", "")
        if dry_run:
            print(f"[I]   would judge {view} '{query}' ({root.trace_id}): {len(judges.units(answer))} units, {len(context)} excerpts")
            continue
        left = await run_evals.calls_left(reserve)
        if left is not None and left <= 0:
            print("[I]   today's budget is used up: run again tomorrow to continue")
            break
        result = await judges.judge_answer(query, view, answer, context)
        for key in JUDGE_KEYS:
            if result["scores"].get(key) is not None:
                client.create_feedback(trace_id=root.trace_id, key=f"judge.{key}", score=result["scores"][key], session_id=project_id,
                                       comment=f"judge {result['judge_model']} {result['judge_version']}", extend_trace_retention=False)
        judged += 1
        print(f"[I]   judged {view} '{query}': {result['scores']}")
    client.flush()
    print(f"[I] {judged} judged")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Judge yesterday's real answers and attach the scores to their traces.")
    parser.add_argument("--project", default="ai-news-prod")
    parser.add_argument("--day", type=date.fromisoformat, default=(datetime.now(timezone.utc) - timedelta(days=1)).date())
    parser.add_argument("--max", type=int, default=10, help="at most this many answers (1 call each)")
    parser.add_argument("--reserve", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    asyncio.run(main(a.project, a.day, a.max, a.reserve, a.dry_run))
