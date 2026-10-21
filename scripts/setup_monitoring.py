"""
Feature 2, Section K: the LangSmith dashboards, alerts and model prices for one project.
Everything is defined here, so it can be rebuilt exactly; running it again replaces what it made.

  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --project ai-news-dev --dry-run    what it would do
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --project ai-news-dev --preview    every series' filter and grouping
                                                                                               run on the last 14 days, nothing created
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --project ai-news-dev              prices and dashboards, then
                                                                                               the alert settings to enter in the UI
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --markdown                         the doc's chart and alert tables

1. Prices (#73): services/usage.MODEL_PRICES mirrored into the workspace's model price map, so
   LangSmith's own cost numbers match ours, including the dated price change (2027-01-01).
2. Dashboards (#73-#75, #77): three custom dashboards, "ai-news · Usage and cost", "· Speed and
   errors", "· Quality", for this project. The charts are below; the doc lists them all.
3. Alerts (#76): three alert rules, sent to ntfy.sh (NTFY_ALERT_TOPIC in .env), which turns
   LangSmith's alert JSON into a phone notification. They are made in the LangSmith UI: the API
   answered 200 to every create but kept none (2026-10-20), so this prints the exact settings.

How LangSmith's chart API behaves (found in Section K): saved charts must use its "V2" definitions
(metric_definition, filter_definition), but its public API returns data only for "V1" series
(the preview). So each series is defined once below and rendered both ways: --preview runs the V1
form to prove every filter and grouping is accepted and finds the right runs; the V2 form is saved.
Needs LANGSMITH_API_KEY and NTFY_ALERT_TOPIC (from .env).
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_TRACING", "false")       # this script's own calls aren't traced
PREFIX = "ai-news"

# --- What the app records (the names the filters below rely on) ---
# Requests are root runs named after their route; failures carry tags error:<status> and
# "timeout" (Section C); chat attempts are llm runs with metadata ls_model_name, a retried call
# has a run tagged retry:attempt:2 (D); embedding attempts are embedding runs with estimated
# usage_metadata (K); the article-list cache records metadata path on cache_or_build (C); root
# runs carry answered_by and prompt_version (D, E); feedback is on the request's trace (its root
# run): check.*, fact_check.* (I), user_rating (J), judge.* (I's judge_yesterday).
ANSWER_ROUTES = ["POST /query", "POST /api/timeline", "POST /api/contradictions"]
STEPS = ["gnews.search", "scrape", "embed.documents", "embed.query", "retrieve", "fact_check"]
STATUS_CODES = [400, 404, 409, 422, 500, 502, 503, 504]
ROOT_FILTER = "eq(is_root, true)"
LLM = 'eq(run_type, "llm")'
EMBEDDING = 'eq(run_type, "embedding")'
ERRORED = 'eq(status, "error")'
RETRIED = 'has(tags, "retry:attempt:2")'          # LangSmith refuses or() across tags


def any_of(*filters: str) -> str:
    return filters[0] if len(filters) == 1 else f"or({', '.join(filters)})"


def name_in(names) -> str:
    return any_of(*[f'eq(name, "{n}")' for n in names])


# A series, once: {"name", "kind", "filter", "group", ...}. kind: count | sum (field) |
# percentile (p) | error_rate | feedback_avg (key) | feedback_count (key).
def series(name, kind, filter=None, group=None, **extra):
    return {"name": name, "kind": kind, "filter": filter, "group": group, **extra}


def count(name, filter=None, group=None):
    return series(name, "count", filter, group)


def total(name, field, filter=None, group=None):
    return series(name, "sum", filter, group, field=field)


def latency(name, p, filter=None, group=None):
    return series(name, "percentile", filter, group, p=p)


def feedback_avg(name, key, group=None):
    return series(name, "feedback_avg", ROOT_FILTER, group, key=key)


def by(attribute, path=None):
    return (attribute, path)


def chart(title, chart_type, series_list, description):
    return {"title": title, "chart_type": chart_type, "description": description, "series": series_list}


# Classic metric names for a summed field (the V1 preview has no "thinking" token total)
V1_SUM = {"prompt_tokens": "prompt_tokens", "completion_tokens": "completion_tokens", "total_cost": "total_cost",
          "completion_token_details.reasoning": "completion_tokens"}


def to_v1(s: dict) -> dict:
    """The preview form. Only the filter and grouping are being checked, so a metric the classic
    API lacks maps to its nearest one (p95 -> p99, thinking tokens -> output tokens)."""
    kind = s["kind"]
    if kind == "count":
        out = {"metric": "run_count"}
    elif kind == "sum":
        out = {"metric": V1_SUM[s["field"]]}
    elif kind == "percentile":
        out = {"metric": "latency_p50" if s["p"] <= 0.5 else "latency_p99"}
    elif kind == "error_rate":
        out = {"metric": "error_rate"}
    elif kind == "feedback_avg":
        out = {"metric": "feedback_score_avg", "feedback_key": s["key"]}
    else:                                              # feedback_count
        out = {"metric": "feedback", "feedback_key": s["key"]}
    out.update(name=s["name"], filters={"filter": s["filter"] or ROOT_FILTER})
    if s["group"]:
        attribute, path = s["group"]
        out["group_by"] = {"attribute": attribute, **({"path": path} if path else {}), "max_groups": 8}
    return out


def to_v2(s: dict, project_id: str) -> dict:
    """The saved form."""
    kind = s["kind"]
    if kind == "count":
        metric = {"type": "count"}
    elif kind == "sum":
        metric = {"type": "sum", "field": s["field"]}
    elif kind == "percentile":
        metric = {"type": "percentile", "field": "latency_seconds", "params": {"p": s["p"]}}
    elif kind == "error_rate":
        metric = {"type": "ratio", "numerator": {"type": "count", "filter": ERRORED}, "denominator": {"type": "count"}}
    elif kind == "feedback_avg":
        metric = {"type": "avg", "field": "feedback_score", "params": {"feedback_key": s["key"]}}
    else:
        metric = {"type": "count", "entity": "feedback", "params": {"feedback_key": s["key"]}}
    out = {"name": s["name"], "metric_definition": metric,
           "filter_definition": {"source_type": "tracing_project", "project_ids": [project_id], "run_filter": s["filter"]}}
    if s["group"]:
        attribute, path = s["group"]
        out["group_by_definitions"] = [{"attribute": attribute, **({"path": path} if path else {})}]
    return out


def dashboards(project: str) -> list[dict]:
    """The three dashboards: [{title, description, charts}]."""
    usage = {
        "title": f"{PREFIX} · Usage and cost · {project}",
        "description": "Gemini calls, tokens and cost (#73). Costs are LangSmith's, from the price map this script keeps equal to services/usage.py.",
        "charts": [
            chart("Chat calls (period)", "kpi", [count("chat calls", LLM)],
                  "Every attempt counts, as it does against Gemini's free quota (20 a day per model)."),
            chart("Cost in USD (period)", "kpi", [total("cost", "total_cost", any_of(LLM, EMBEDDING))],
                  "Chat and embeddings together; embeddings are priced from estimated tokens."),
            chart("Chat calls per day, by model (limit 20 a day per model)", "bar",
                  [count("calls", LLM, by("metadata", "ls_model_name"))],
                  "Attempts per model per day. The free tier allows 20 per model per day (Pacific time); read the bars against 20."),
            chart("Chat tokens per day", "line",
                  [total("input", "prompt_tokens", LLM), total("output", "completion_tokens", LLM),
                   total("thinking", "completion_token_details.reasoning", LLM)],
                  "Input, output and thinking tokens of all chat attempts (output includes thinking)."),
            chart("Cost per day, by model (USD)", "bar",
                  [total("cost", "total_cost", any_of(LLM, EMBEDDING), by("metadata", "ls_model_name"))],
                  "Chat models and the embedding model."),
            chart("Embedding tokens per day (estimated)", "line",
                  [total("tokens", "prompt_tokens", EMBEDDING), count("calls", EMBEDDING)],
                  "The embeddings API returns no count: 4.95 characters per token, measured in Section D."),
            chart("Article list: cache hit or full build", "bar",
                  [count("requests", 'eq(name, "cache_or_build")', by("metadata", "path"))],
                  "cache_hit costs no GNews call and no embedding; a full build costs both."),
        ],
    }
    steps = any_of(name_in(STEPS), LLM)
    speed = {
        "title": f"{PREFIX} · Speed and errors · {project}",
        "description": "Latency per endpoint and step, errors by status, timeouts, fallbacks, retries (#74).",
        "charts": [
            chart("p95 latency by endpoint (s)", "line", [latency("p95", 0.95, ROOT_FILTER, by("name"))],
                  "Whole requests. The page gives up after 120 s; the answer step after 75 s."),
            chart("p50 latency by endpoint (s)", "line", [latency("p50", 0.5, ROOT_FILTER, by("name"))], "Whole requests: the typical one."),
            chart("Latency per step, p50 and p95 (s)", "bar",
                  [latency("p50", 0.5, steps, by("name")), latency("p95", 0.95, steps, by("name"))],
                  "GNews search, scraping, embedding, retrieval, each chat attempt (ChatGoogleGenerativeAI) and the fact check."),
            chart("Error rate by endpoint", "line", [series("error rate", "error_rate", ROOT_FILTER, by("name"))],
                  "Share of requests that failed (any status)."),
            chart("Failed requests by status code", "bar",
                  [count(str(code), f'and({ROOT_FILTER}, has(tags, "error:{code}"))') for code in STATUS_CODES],
                  "Each failure point has its own status (errors.py): 503 busy or unavailable, 504 timeout, 502 bad answer, 404 nothing found, ..."),
            chart("Timeouts", "bar", [count("timed-out requests", f'and({ROOT_FILTER}, has(tags, "timeout"))'),
                                       count("timed-out answer steps", 'and(eq(is_root, false), has(tags, "timeout"))')],
                  "A timed-out chat attempt goes to the fallback model; a request times out when the 75 s deadline passes."),
            chart("Who answered: primary or fallback model", "bar",
                  [count("answers", f"and({ROOT_FILTER}, {name_in(ANSWER_ROUTES)})", by("metadata", "answered_by"))],
                  "The fallback's share is the fallback rate."),
            chart("Retries", "bar", [count("chat retries", f"and({LLM}, {RETRIED})"),
                                      count("embedding retries", f"and({EMBEDDING}, {RETRIED})")],
                  "Calls that needed a retry after a busy answer (429/5xx)."),
            chart("Requests per day by endpoint", "bar", [count("requests", ROOT_FILTER, by("name"))],
                  "Traffic, to read the rates above against."),
        ],
    }
    quality = {
        "title": f"{PREFIX} · Quality · {project}",
        "description": "Code checks, reader ratings, fact-check health and judge scores (#75). Scores are feedback on the request's trace.",
        "charts": [
            chart("Code-check score by endpoint", "line", [feedback_avg("check.score", "check.score", by("name"))],
                  "Share of the free code checks passed (Section I); 1 = all passed."),
            chart("Code-check score by prompt version", "line",
                  [feedback_avg("check.score", "check.score", by("metadata", "prompt_version"))],
                  "A new prompt version that scores lower shows here first."),
            chart("Report checks: pass rate", "bar",
                  [feedback_avg(k, f"check.{k}") for k in ("no_bullets", "no_banned_phrase", "named_outlets_in_sources", "length_ok")],
                  "Per check, the share of reports that passed."),
            chart("Timeline checks: pass rate", "bar",
                  [feedback_avg(k, f"check.{k}") for k in ("has_events", "all_lines_formatted", "newest_first", "no_future_date", "outlets_in_sources")],
                  "Per check, the share of timelines that passed."),
            chart("Contradictions and retrieval checks: pass rate", "bar",
                  [feedback_avg(k, f"check.{k}") for k in ("has_content", "points_name_two_outlets", "story_cap")],
                  "story_cap: at most 3 chunks per story when the index had others to use."),
            chart("Reader ratings: share of 👍", "line", [feedback_avg("share of 👍", "user_rating", by("name"))],
                  "user_rating is 1 for 👍 and 0 for 👎 (Section J), so the average is the share of 👍."),
            chart("Reader ratings: how many", "bar", [series("ratings", "feedback_count", ROOT_FILTER, key="user_rating")],
                  "Few ratings make the share above noisy."),
            chart("Fact-check health", "line",
                  [feedback_avg("JSON valid", "check.fact_check_json"), feedback_avg("readable", "fact_check.usable"),
                   feedback_avg("hidden as nothing new", "fact_check.nothing_new_share"),
                   feedback_avg("unjudged", "fact_check.unjudged_share")],
                  "On every check (#66). A jump in 'hidden as nothing new' after a prompt change means it hides too much."),
            chart("Unreadable fact-check answers", "bar", [count("unreadable", 'has(tags, "unreadable_answer")')],
                  "The model's JSON couldn't be read: every story stays unjudged. Also an alert."),
            chart("LLM judge scores (judge_yesterday.py)", "line",
                  [feedback_avg(k, f"judge.{k}") for k in ("faithfulness", "citation_correctness", "relevance")],
                  "Written by hand-run judging (Section I). Trust a field only once calibration (Section G) says it's trusted."),
            chart("Judge faithfulness by prompt version", "line",
                  [feedback_avg("faithfulness", "judge.faithfulness", by("metadata", "prompt_version"))],
                  "Compare prompt versions on real answers."),
        ],
    }
    return [usage, speed, quality]


# --- Alerts (#76) ---
# Plan: error rate over 20% in an hour, p95 latency over 60 s, any unreadable answer. Made in the UI,
# which offers only 5- or 15-minute windows and averages latency (no percentile): all three use 15 min.
# Each was checked with the UI's Test button (2026-10-20), and the unreadable one with a real alert.

def alert_rules() -> list[dict]:
    return [
        {"name": f"{PREFIX}: error rate over 20%", "metric": "Errors", "aggregation": "Percentage", "condition": "greater than 20",
         "window_minutes": 15, "filter": "Is Trace = true (whole requests)"},
        {"name": f"{PREFIX}: answers slower than 60 s", "metric": "Latency", "aggregation": "Average", "condition": "greater than 60 s",
         "window_minutes": 15, "filter": "Is Trace = true (whole requests)"},
        # NOT "Is Trace": the tag is on the fact_check step inside the request, so the two filters together
        # can never match (a real unreadable run fired nothing until it was removed, 2026-10-20)
        {"name": f"{PREFIX}: unreadable fact-check answer", "metric": "Run Count", "aggregation": "Count", "condition": "at least 1",
         "window_minutes": 15, "filter": "Tag = unreadable_answer only (no Is Trace)"},
    ]


def ntfy_url(topic: str) -> str:
    """LangSmith posts its alert JSON (alert_rule_name, triggered_metric_value, ...) and appends its own
    keys; it substitutes nothing. ntfy's templates (tpl=yes) read those keys into the notification.
    The title is the rule's name alone: the names already start with "ai-news:"."""
    params = {"tpl": "yes", "t": "{{.alert_rule_name}}",
              "m": "{{.alert_rule_attribute}} = {{.triggered_metric_value}} (threshold {{.triggered_threshold}}) in {{.project_name}}\n{{.runs_url}}",
              "p": "high"}
    return f"https://ntfy.sh/{topic}?" + urllib.parse.urlencode(params)


def alert_settings(project: str, topic: str | None) -> str:
    """What to enter in the LangSmith UI: <project> -> Alerts (top right) -> + Alert, once per rule."""
    if not topic:
        return "[K] alerts: NTFY_ALERT_TOPIC isn't set in .env, so there's no webhook URL to give"
    lines = [f"[K] alerts: make these 3 in the UI: {project} -> Alerts icon (top right) -> + Alert. For each:",
             "      Channel: Webhook",
             f"      URL:     {ntfy_url(topic)}",
             '      Headers: {"Content-Type": "application/json"}',
             "      Body:    {}",
             "      Press Test before saving: the phone gets a notification titled with the rule's name."]
    for r in alert_rules():
        lines.append(f"    - {r['name']}: {r['metric']} / {r['aggregation']}, {r['condition']}, {r['window_minutes']} min, filter: {r['filter']}")
    return "\n".join(lines)


# --- Prices (#73) ---

def price_rows() -> list[dict]:
    from services.usage import MODEL_PRICES
    per_token = lambda usd_per_million: f"{usd_per_million / 1e6:.10f}".rstrip("0").rstrip(".") or "0"
    return [{"name": f"{PREFIX} {model} from {start}", "match_pattern": f"^(models/)?{re.escape(model)}$",
             "start_time": f"{start}T00:00:00Z", "prompt_cost": per_token(input_price), "completion_cost": per_token(output_price)}
            for model, periods in MODEL_PRICES.items() for start, input_price, output_price in periods]


# --- Talking to LangSmith ---

class Api:
    def __init__(self, client):
        self.c = client

    def call(self, method, path, **kw):
        # api_url has no /api/v1; most paths work without it, the /api/v1/platform ones don't
        base = self.c.api_url.removesuffix("/api/v1") if path.startswith("/api/v1/") else self.c.api_url
        r = self.c.session.request(method, base + path, headers=self.c._headers, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else None


def preview(api, project_id: str, days: int = 14) -> int:
    """Runs every series' V1 form on the last `days` days: a filter or grouping LangSmith doesn't accept fails here."""
    now = datetime.now(timezone.utc)
    bucket = {"start_time": (now - timedelta(days=days)).isoformat(), "end_time": now.isoformat(), "stride": {"days": 1}}  # max 7 days
    boards = dashboards("preview")
    # Filters used only inside a V2 metric get a chart of their own here
    boards.append({"title": "x · V2-only filters", "charts": [chart("error-rate numerator", "bar", [count("errored", f"and({ROOT_FILTER}, {ERRORED})")], "")]})
    bad = 0
    for board in boards:
        print(f"[K] {board['title'].split(' · ')[1]}")
        for ch in board["charts"]:
            v1 = [{**to_v1(s), "id": f"s{i}"} for i, s in enumerate(ch["series"])]
            try:
                data = api.call("POST", "/charts/preview", json={"bucket_info": bucket, "chart": {"series": v1, "common_filters": {"session": [project_id]}}})
            except RuntimeError as e:
                bad += 1
                print(f"[K]   FAIL {ch['title']}: {e}")
                continue
            values = {}
            for point in data.get("data", []):
                key = v1[int(point["series_id"][1:])]["name"] + (f" [{point['group']}]" if point.get("group") else "")
                v = point.get("value")
                if isinstance(v, dict):          # feedback: {key: {"n", "avg", ...}}: the count of scores
                    v = sum((stats or {}).get("n") or 0 for stats in v.values())
                    key += " (scores)"
                if isinstance(v, (int, float)):
                    values[key] = values.get(key, 0) + v
            shown = ", ".join(f"{k}: {round(v, 4)}" for k, v in list(values.items())[:6]) or "no data yet"
            print(f"[K]   ok   {ch['title']}: {shown}")
    return bad


def setup_prices(api, dry_run: bool):
    existing, offset = {}, 0
    while True:
        page = api.call("GET", "/model-price-map", params={"limit": 100, "offset": offset})
        items = page if isinstance(page, list) else page.get("items", [])
        existing.update({i["name"]: i for i in items if i.get("tenant_id") and i["name"].startswith(PREFIX + " ")})
        if len(items) < 100:
            break
        offset += 100
    for row in price_rows():
        old = existing.get(row["name"])
        same = old and float(old["prompt_cost"]) == float(row["prompt_cost"]) and float(old["completion_cost"]) == float(row["completion_cost"])
        if same:
            print(f"[K] price {row['name']}: up to date")
        elif dry_run:
            print(f"[K] price {row['name']}: would {'update' if old else 'add'} ({row['prompt_cost']} / {row['completion_cost']} per token)")
        elif old:
            api.call("PUT", f"/model-price-map/{old['id']}", json=row)
            print(f"[K] price {row['name']}: updated")
        else:
            api.call("POST", "/model-price-map", json=row)
            print(f"[K] price {row['name']}: added")


def setup_dashboards(api, project: str, project_id: str, dry_run: bool) -> list[str]:
    sections = api.call("GET", "/charts/section", params={"limit": 100}) or []
    sections = sections if isinstance(sections, list) else sections.get("items", [])
    ids = []
    for index, board in enumerate(dashboards(project)):
        old = [s for s in sections if s.get("title") == board["title"]]
        if dry_run:
            print(f"[K] dashboard '{board['title']}': would {'replace' if old else 'create'} ({len(board['charts'])} charts)")
            continue
        for s in old:
            api.call("DELETE", f"/charts/section/{s['id']}")
        section = api.call("POST", "/charts/section", json={"title": board["title"], "description": board["description"], "index": index})
        for i, ch in enumerate(board["charts"]):
            api.call("POST", "/charts/create", json={"title": ch["title"], "description": ch["description"], "chart_type": ch["chart_type"],
                                                     "index": i, "section_id": section["id"],
                                                     "series": [to_v2(s, project_id) for s in ch["series"]]})
        ids.append(section["id"])
        print(f"[K] dashboard '{board['title']}': {'replaced' if old else 'created'} ({len(board['charts'])} charts, {section['id']})")
    return ids


def describe(s: dict) -> str:
    kind = s["kind"]
    return {"count": "runs", "sum": f"sum of {s.get('field')}", "percentile": f"p{round(s.get('p', 0) * 100)} latency",
            "error_rate": "errored / all", "feedback_avg": f"avg feedback {s.get('key')}",
            "feedback_count": f"count of feedback {s.get('key')}"}[kind]


def markdown() -> str:
    """The doc's tables (#77), made from the definitions above, so the doc can't drift from what's built."""
    cell = lambda text: (text or "").replace("|", "\\|")
    lines = []
    for board in dashboards("<project>"):
        lines += [f"#### {board['title']}", "", board["description"], "",
                  "| # | Chart | Type | Series: metric | Filter | Grouped by |", "|---|---|---|---|---|---|"]
        for n, ch in enumerate(board["charts"], 1):
            for i, s in enumerate(ch["series"]):
                group = "" if not s["group"] else (f"metadata.{s['group'][1]}" if s["group"][1] else s["group"][0])
                lines.append(f"| {n if i == 0 else ''} | {cell(ch['title']) if i == 0 else ''} | {ch['chart_type'] if i == 0 else ''} | "
                             f"{cell(s['name'])}: {describe(s)} | `{cell(s['filter'] or ROOT_FILTER)}` | {group} |")
        lines.append("")
    lines += ["#### Alerts (made in the UI)", "", "| Alert | Metric / aggregation | Fires when | Window | Filter |", "|---|---|---|---|---|"]
    for r in alert_rules():
        lines.append(f"| {r['name']} | {r['metric']} / {r['aggregation']} | {r['condition']} | {r['window_minutes']} min | {cell(r['filter'])} |")
    return "\n".join(lines)


def main(project: str, dry_run: bool, do_preview: bool):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from langsmith import Client
    client = Client()
    api = Api(client)
    try:
        project_id = str(client.read_project(project_name=project).id)
    except Exception as e:
        sys.exit(f"[K] project {project} not found ({type(e).__name__}): it's created by its first trace. Run this after that.")
    if do_preview:
        sys.exit(1 if preview(api, project_id) else 0)
    setup_prices(api, dry_run)
    setup_dashboards(api, project, project_id, dry_run)
    print(alert_settings(project, os.getenv("NTFY_ALERT_TOPIC")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangSmith prices, dashboards and alerts for a project (feature 2, Section K).")
    parser.add_argument("--project", help="e.g. ai-news-dev or ai-news-prod (required except with --markdown)")
    parser.add_argument("--markdown", action="store_true", help="print the doc's chart and alert tables; no API calls")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", action="store_true", help="run every series' filter and grouping on recent data; create nothing")
    a = parser.parse_args()
    if a.markdown:
        print(markdown())
    elif not a.project:
        parser.error("--project is required")
    else:
        main(a.project, a.dry_run, a.preview)
