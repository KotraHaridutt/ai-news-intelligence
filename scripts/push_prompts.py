"""
Mirrors the app's prompts (services/prompts.py) to the LangSmith Prompt Hub (feature 2, #41).
Run it by hand after changing a prompt:

  .local/venv/Scripts/python.exe scripts/push_prompts.py              push the prompts whose text changed
  .local/venv/Scripts/python.exe scripts/push_prompts.py --dry-run    only show what would be pushed
  .local/venv/Scripts/python.exe scripts/push_prompts.py --tag prod   also point the "prod" tag at the current versions,
                                              which is what PROMPT_SOURCE=hub:prod loads

Each prompt is "ai-news-<name>" (private). Each commit is tagged "v-<version>" and described with
the version and the git commit, so a Hub commit leads back to the code. A prompt whose text is
already the Hub's latest is skipped. Needs LANGSMITH_API_KEY (from .env).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PROMPT_SOURCE"] = "code"   # always push the code's texts (and don't pull the Hub's on import)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "services/prompts.py"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def hub_text(client, hub_name: str) -> str | None:
    """The text of the Hub's latest commit of a prompt, or None if it doesn't exist yet."""
    try:
        return client.pull_prompt(hub_name, skip_cache=True).template
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            return None
        raise


def plan(client, prompts) -> list[dict]:
    """What to do for each prompt: create (not on the Hub), update (text differs), unchanged."""
    actions = []
    for prompt in prompts:
        current = hub_text(client, prompt.hub_name)
        action = "create" if current is None else ("unchanged" if current == prompt.template else "update")
        actions.append({"prompt": prompt, "action": action})
    return actions


def latest_commit(client, hub_name: str):
    """The Hub's newest commit of a prompt: its id (a UUID, what tags point at) and its hash."""
    return next(iter(client.list_prompt_commits(hub_name, limit=1)))


def set_tag(client, hub_name: str, tag: str, commit_id: str):
    """Points a commit tag at a commit (by its id, not its hash): created if new, moved if it exists."""
    path = f"/repos/-/{hub_name}/tags"     # "-": the key's own workspace
    try:
        client.request_with_retries("POST", path, json={"tag_name": tag, "commit_id": commit_id})
    except Exception as e:
        if "409" not in str(e) and "conflict" not in type(e).__name__.lower() and "already exists" not in str(e).lower():
            raise
        client.request_with_retries("PATCH", f"{path}/{tag}", json={"commit_id": commit_id})


def push(client, actions, tag: str | None, commit: str) -> list[dict]:
    from langchain_core.prompts import PromptTemplate
    from services.prompts import variables_of
    results = []
    for item in actions:
        prompt = item["prompt"]
        if item["action"] != "unchanged":
            template = PromptTemplate(template=prompt.template, input_variables=sorted(variables_of(prompt.template)))
            client.push_prompt(prompt.hub_name, object=template, is_public=False,
                               description=f"AI News Intelligence: the {prompt.name} prompt (services/prompts.py)",
                               commit_tags=[f"v-{prompt.version}"],
                               commit_description=f"version {prompt.version} from git {commit}")
        latest = latest_commit(client, prompt.hub_name)
        if tag:
            set_tag(client, prompt.hub_name, tag, str(latest.id))
        results.append({"name": prompt.hub_name, "version": prompt.version, "action": item["action"], "commit": latest.commit_hash[:8]})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mirror the app's prompts to the LangSmith Prompt Hub.")
    parser.add_argument("--dry-run", action="store_true", help="only show what would be pushed")
    parser.add_argument("--tag", help="also point this tag (e.g. prod) at the current versions")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set (put it in .env).")
    from langsmith import Client
    from services import prompts

    client = Client()
    actions = plan(client, list(prompts.CODE_PROMPTS.values()))
    for item in actions:
        print(f"  {item['prompt'].hub_name:22} {item['prompt'].version}  {item['action']}")
    if args.dry_run:
        print("Dry run: nothing pushed.")
        return actions
    results = push(client, actions, args.tag, _git_commit())
    for r in results:
        print(f"  {r['name']:22} {r['version']}  {r['action']:9}  commit {r['commit']}" + (f"  tag {args.tag}" if args.tag else ""))
    return results


if __name__ == "__main__":
    main()
