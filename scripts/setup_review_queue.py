"""
Sets up the review queue in LangSmith (feature 2, #71). Run it once per project; running it again
changes nothing:

  .local/venv/Scripts/python.exe scripts/setup_review_queue.py --project ai-news-dev --dry-run
  .local/venv/Scripts/python.exe scripts/setup_review_queue.py --project ai-news-prod

1. The annotation queue "ai-news-review" (private to the workspace), if it doesn't exist.
2. A rule on the project: every request whose answer failed a code check (feedback check.score
   below 1, from Section I) is added to the queue. LangSmith applies rules on its side, after
   the trace and its feedback have arrived, so nothing in the app waits for it.

The other way into the queue is a reader's 👎: the rating endpoint adds that trace itself
(services/feedback_service.py), because a 👎 comes long after LangSmith has processed the trace.
In the queue, a reviewed run can be added to a dataset from the LangSmith page ("Add to dataset").

Traces in the queue keep the normal retention (14 days on the free plan): review them within that,
or add them to a dataset, which keeps a copy. Needs LANGSMITH_API_KEY (from .env).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUEUE = "ai-news-review"
# Root runs (requests) with a failed code check. The syntax was checked against the API (Section J)
CHECK_FILTER = 'and(eq(is_root, true), and(eq(feedback_key, "check.score"), lt(feedback_score, 1)))'


def rule_name(project: str) -> str:
    return f"{QUEUE}: failed code checks ({project})"


def find_rule(client, name: str) -> dict | None:
    response = client.request_with_retries("GET", "/runs/rules", params={"limit": 100})
    return next((r for r in response.json() if r.get("display_name") == name), None)


def main(project: str, dry_run: bool):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from langsmith import Client
    client = Client()
    try:
        project_id = str(client.read_project(project_name=project).id)
    except Exception as e:
        print(f"[J] project {project} not found ({type(e).__name__}): it's created by its first trace. Run this after that.")
        sys.exit(1)

    queues = list(client.list_annotation_queues(name=QUEUE))
    if queues:
        queue_id = str(queues[0].id)
        print(f"[J] queue {QUEUE}: exists ({queue_id})")
    elif dry_run:
        queue_id = None
        print(f"[J] queue {QUEUE}: would be created")
    else:
        queue_id = str(client.create_annotation_queue(
            name=QUEUE, description="Answers to review: every 👎 from a reader and every answer that failed a code check.").id)
        print(f"[J] queue {QUEUE}: created ({queue_id})")

    name = rule_name(project)
    existing = find_rule(client, name)
    if existing:
        print(f"[J] rule '{name}': exists ({existing.get('id')}), filter {existing.get('filter')}")
        return
    if dry_run:
        print(f"[J] rule '{name}': would be created on {project} with filter {CHECK_FILTER}")
        return
    rule = {"display_name": name, "session_id": project_id, "sampling_rate": 1.0, "is_enabled": True,
            "filter": CHECK_FILTER, "add_to_annotation_queue_id": queue_id,
            # Normal retention, as for every automated score (Section I): no paid upgrade per queued trace
            "extend_annotation_queue_trace_retention": False}
    response = client.request_with_retries("POST", "/runs/rules", json=rule)
    print(f"[J] rule '{name}': created ({response.json().get('id')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the ai-news-review queue and its failed-check rule for a project.")
    parser.add_argument("--project", required=True, help="e.g. ai-news-dev or ai-news-prod")
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    main(a.project, a.dry_run)
