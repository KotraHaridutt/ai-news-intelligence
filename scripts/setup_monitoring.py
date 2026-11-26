"""
Feature 2, Section K: the LangSmith dashboards, alerts and model prices for one project.
Everything is defined here, so it can be rebuilt exactly; running it again replaces what it made.

  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --project ai-news-dev --dry-run    what it would do
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --project ai-news-dev --preview    every series' filter and grouping
                                                                                               run on the last 14 days, nothing created
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --project ai-news-dev              prices and dashboards, then
                                                                                               the alert settings to enter in the UI
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --markdown                         the doc's chart and alert tables
  .local/venv/Scripts/python.exe scripts/setup_monitoring.py --markdown safety                  ... of the safety dashboard (feature 3 O)

1. Prices (#73): services/usage.MODEL_PRICES mirrored into the workspace's model price map, so
   LangSmith's own cost numbers match ours, including the dated price change (2027-01-01).
2. Dashboards (#73-#75, #77): three custom dashboards, "ai-news · Usage and cost", "· Speed and
   errors", "· Quality", for this project. The charts are below; the doc lists them all.
3. Alerts (#76): three alert rules, sent to ntfy.sh (NTFY_ALERT_TOPIC in .env), which turns
   LangSmith's alert JSON into a phone notification. They are made in the LangSmith UI: the API
   answered 200 to every create but kept none (2026-10-20), so this prints the exact settings.
4. Feature 3, Section O (#113, #114): a fourth dashboard, "ai-news · Safety", and five safety alerts
   (printed with the others). They read what services/safety/summary.py writes on each request's root
   run: tags safety:<check>:<verdict>, safety:<check>:enforced, safety:block / flag / enforced /
   fallback / failed_open, clef_budget_warning / _fallback, and the safety.* feedback keys.

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
# run): check.*, fact_check.* (I), user_rating (J), judge.* (I's judge_yesterday). Feature 4 F: root runs
# carry metadata rerank_engine (cohere / cloudflare, absent when not reranked); the "rerank" step is tagged
# rerank_failed (no engine could rank: the search's order was used) and rerank_limited (a free limit sent
# the ranking to the other engine).
ANSWER_ROUTES = ["POST /query", "POST /api/timeline", "POST /api/contradictions"]
STEPS = ["gnews.search", "scrape", "embed.documents", "embed.query", "rerank", "retrieve", "fact_check"]
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
# percentile (p) | error_rate | feedback_avg (key) | feedback_count (key) | feedback_sum (key, feature 3 O).
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
    else:                                              # feedback_count, feedback_sum (V1 has no sum: the count checks the filter)
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
    elif kind == "feedback_sum":
        # Feature 3 O: like avg, summed (Clef neurons per day). The API can't check it (V2 returns no data
        # there): whether it draws is checked in the UI, as feature 2 K did for every chart
        metric = {"type": "sum", "field": "feedback_score", "params": {"feedback_key": s["key"]}}
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
            chart("Reranker: who ordered the chunks", "bar",
                  [count("answers", f"and({ROOT_FILTER}, {name_in(ANSWER_ROUTES)})", by("metadata", "rerank_engine"))],
                  "Feature 4: cohere, cloudflare (the fallback, or past Cohere's free limit), or none (reranking off or failed)."),
            chart("Reranker: failures and free limits", "bar",
                  [count("no engine could rank", 'and(eq(name, "rerank"), has(tags, "rerank_failed"))'),
                   count("a free limit moved it", 'and(eq(name, "rerank"), has(tags, "rerank_limited"))')],
                  "Failed: the search's order was used. Limited: Cohere's minute or month (or Cloudflare's neuron share) sent it to the other engine. Also alerts."),
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
        # Feature 4 F #31. Tags on the "rerank" step inside the request: no Is Trace, as above
        {"name": f"{PREFIX}: reranker failing", "metric": "Run Count", "aggregation": "Count", "condition": "at least 3",
         "window_minutes": 15, "filter": "Tag = rerank_failed only (no Is Trace)"},
        # Fires while Cohere's month (950 of 1,000) or minute is used up, so expect it every window then:
        # the sign to look at the month counter (startup line / v1:usage:cohere:<month>)
        {"name": f"{PREFIX}: reranker past a free limit", "metric": "Run Count", "aggregation": "Count", "condition": "at least 1",
         "window_minutes": 15, "filter": "Tag = rerank_limited only (no Is Trace)"},
    ]


# --- Safety (feature 3, Section O #113, #114) ---
# What services/safety/summary.py puts on each request's root run. A check's tag is also on the step
# where it ran, so every series is filtered to root runs: one count per request.
# The checks by where they sit: services/safety/settings.CHECKS, plus the two that only record events and
# always act: "gemini" (Gemini's own filter, F) and "ssrf" (a scrape refused for a private address, M #98).
QUERY_CHECKS = ["origin", "intent", "text_shape", "pii", "names", "comment", "rate_limit"]
ARTICLE_CHECKS = ["ssrf", "injection", "relevance", "coverage"]
ANSWER_CHECKS = ["canary", "output", "faithfulness", "sensitive_topic", "toxicity", "leak", "balance", "gemini"]
EVENT_ONLY_CHECKS = ["gemini", "ssrf"]
# In enforce mode, a block of these refuses the request or withholds the answer (origin isn't here: in enforce
# its middleware refuses before a trace starts, so the series would always be 0; its refusals are in the log) ...
REFUSING_CHECKS = ["intent", "text_shape", "rate_limit", "coverage", "canary", "toxicity", "gemini"]
# ... and of these it changes what is used or shown: removes, masks, swaps, drops a page, or adds a note
CHANGING_CHECKS = ["pii", "names", "ssrf", "injection", "relevance", "output", "faithfulness", "sensitive_topic"]
# Checks whose events can be "flag": the meaning checks (uncertain band, or failed open) and I's format flags
FLAGGING_CHECKS = ["intent", "text_shape", "comment", "names", "injection", "relevance", "coverage",
                   "output", "faithfulness", "sensitive_topic", "toxicity", "leak", "balance"]
CLEF = 'eq(name, "clef")'                           # each Clef call is a span named clef (B #15a)
FAITHFULNESS = "safety.faithfulness"                # J #74's score, sent by faithfulness.verify


def tagged(tag: str) -> str:
    return f'and({ROOT_FILTER}, has(tags, "{tag}"))'


def per_check(checks, verdict: str) -> list[dict]:
    """One count per check: requests whose root run has safety:<check>:<verdict>."""
    return [count(check, tagged(f"safety:{check}:{verdict}")) for check in checks]


def safety_dashboard(project: str) -> dict:
    """The fourth dashboard: what the safety checks decided, what they cost, and who decided."""
    return {
        "title": f"{PREFIX} · Safety · {project}",
        "description": "Feature 3: each safety check's verdicts per day, what was enforced, injection and personal data finds, "
                       "Gemini's blocks, answer support, the time and Clef neurons safety costs, and how often the fallback decided (#113).",
        "charts": [
            chart("Requests safety acted on (period)", "kpi", [count("requests", tagged("safety:enforced"))],
                  "Requests a check refused or changed (enforce mode). In monitor mode this stays 0: read the blocks below."),
            chart("Requests with a block, a flag, or an action", "bar",
                  [count("a block", tagged("safety:block")), count("a flag", tagged("safety:flag")),
                   count("acted on", tagged("safety:enforced"))],
                  "Block: some check's verdict was block (in monitor mode it would have acted). Flag: uncertain, or no engine could decide."),
            chart("Blocks per day: query checks", "bar", per_check(QUERY_CHECKS, "block"),
                  "Requests sent by another website (origin, monitor mode: in enforce they're refused before a trace), intent and text shape "
                  "of the query, personal data in the query or a comment (pii), names in comments, per-visitor limits."),
            chart("Blocks per day: article checks", "bar", per_check(ARTICLE_CHECKS, "block"),
                  "Scrapes refused for a private or internal address (ssrf, always on), article text written to the AI (injection), "
                  "off-topic chunks (relevance), every chunk off-topic (coverage)."),
            chart("Blocks per day: answer checks", "bar", per_check(ANSWER_CHECKS, "block"),
                  "The canary in an answer, outside links or images, unsupported sentences, loaded wording, leaks, one-sidedness, Gemini's own filter."),
            chart("Flags per day: query and article checks", "bar",
                  per_check([c for c in FLAGGING_CHECKS if c in QUERY_CHECKS + ARTICLE_CHECKS], "flag"),
                  "Uncertain verdicts and checks that failed open: where to look for false alarms (#115)."),
            chart("Flags per day: answer checks", "bar",
                  per_check([c for c in FLAGGING_CHECKS if c in ANSWER_CHECKS], "flag"),
                  "Uncertain verdicts, sensitive topics, cut or malformed answers (output), checks that failed open."),
            chart("Enforced: requests refused or answers withheld", "bar", per_check(REFUSING_CHECKS, "enforced"),
                  "The user got an error message instead of an answer. Cross-site requests (origin) and the middleware's 429s are refused "
                  "before a trace starts: they're only in the log."),
            chart("Enforced: text removed, masked, swapped or noted", "bar", per_check(CHANGING_CHECKS, "enforced"),
                  "The user got an answer, changed: a page not fetched (ssrf), injected text removed, personal data masked, chunks swapped, "
                  "links dropped, a note added."),
            chart("Injection scan per request", "line",
                  [feedback_avg("would remove", "safety.injection_found"), feedback_avg("removed", "safety.injection_removed")],
                  "Sentences, titles and outlets the scan found written to the AI, and those removed (enforce). Averages over requests that scanned."),
            chart("Personal data found per request", "line", [feedback_avg("places", "safety.pii_found", by("name"))],
                  "How many of the places checked (query, comment, answer, texts) held an email, phone, card, Aadhaar, PAN, IBAN or IP."),
            chart("Rate limits per day", "bar",
                  [count("429 on a traced request", tagged("error:429")), count("over a limit (monitor)", tagged("safety:rate_limit:block"))],
                  "In enforce mode the limits middleware refuses before a trace starts, so those 429s are only in the log; "
                  "fresh-build 429s (inside the pipeline) and monitor-mode overs are here."),
            chart("Gemini's own blocks per day", "bar", [count("blocked", tagged("safety:gemini:block"), by("name"))],
                  "Gemini's filter (F) refused the prompt (422) or the answer (502). Its reason and ratings are the event's label."),
            chart("Answer support over time", "line", [feedback_avg("supported share", FAITHFULNESS, by("name"))],
                  "Share of an answer's sentences Clef matched to its sources (J). A drop after a prompt change is worth a look."),
            chart("Time safety added per request (s)", "line", [feedback_avg("seconds", "safety.seconds", by("name"))],
                  "Wall time during which a safety check ran (checks side by side counted once)."),
            chart("Reader ratings where safety acted or flagged: share of 👍", "line",
                  [series("acted on", "feedback_avg", tagged("safety:enforced"), key="user_rating"),
                   series("flagged", "feedback_avg", tagged("safety:flag"), key="user_rating"),
                   series("a block (monitor)", "feedback_avg", tagged("safety:block"), key="user_rating"),
                   feedback_avg("all answers", "user_rating")],
                  "A 👎 rate on changed answers well above 'all answers' means a check is hurting answers."),
            chart("Clef neurons per day (free: 10,000)", "bar",
                  [series("neurons", "feedback_sum", ROOT_FILTER, key="safety.clef_neurons")],
                  "Neurons the safety checks spent, summed over requests. The reranker's Cloudflare share counts against the same 10,000 "
                  "(the startup line and Redis hold the exact day total); warning at 7,000, fallback at 9,500."),
            chart("Clef neurons per request", "line", [feedback_avg("neurons", "safety.clef_neurons", by("name"))],
                  "A fresh report measured ~505 (K); a cached view far less. A jump means more text is being judged."),
            chart("Clef latency per call, p50 and p95 (s)", "line", [latency("p50", 0.5, CLEF), latency("p95", 0.95, CLEF)],
                  "Each Clef call's span: the slot wait, the call and its retry. Query checks time out at 2 s, chunks at 3 s, others 5 s."),
            chart("Clef calls and errors", "bar", [count("calls", CLEF), count("failed", f"and({CLEF}, {ERRORED})")],
                  "A failed call (timeout, 429/5xx after a retry, an unreadable or cut reply) sends its check to the fallback."),
            chart("Who decided: fallback, fail-open, budget", "bar",
                  [count("fallback decided a check", tagged("safety:fallback")), count("no engine (failed open)", tagged("safety:failed_open")),
                   count("crossed the 9,500 switch", tagged("clef_budget_fallback"))],
                  "Requests where the embedding classifier decided at least one check, where no engine could (#18), and the one that crossed the neuron switch."),
            chart("Share of meaning checks the fallback decided", "line", [feedback_avg("fallback share", "safety.fallback_share")],
                  "0 when Clef decided everything. An alert fires when it's most checks for an hour."),
        ],
    }


def all_dashboards(project: str) -> list[dict]:
    """Feature 2's three dashboards and feature 3's safety dashboard: what setup and --preview build."""
    return dashboards(project) + [safety_dashboard(project)]


def safety_alert_rules() -> list[dict]:
    """Feature 3 O #114. The tags are on the root run (summary.py), so Is Trace and the tag together match."""
    return [
        {"name": f"{PREFIX}: canary leaked", "metric": "Run Count", "aggregation": "Count", "condition": "at least 1",
         "window_minutes": 15, "filter": "Is Trace = true, Tag = safety:canary:block"},
        {"name": f"{PREFIX}: safety blocks spiking", "metric": "Run Count", "aggregation": "Count", "condition": "at least 10",
         "window_minutes": 15, "filter": "Is Trace = true, Tag = safety:block"},
        # Alerts can't group by query: the runs link lists the requests, and their queries show the topic
        {"name": f"{PREFIX}: injection found again and again", "metric": "Run Count", "aggregation": "Count", "condition": "at least 3",
         "window_minutes": 60, "filter": "Is Trace = true, Tag = safety:injection:block"},
        # Set once a day, on the request that crossed the 7,000 warning level (clef._check_levels)
        {"name": f"{PREFIX}: Clef past its neuron warning", "metric": "Run Count", "aggregation": "Count", "condition": "at least 1",
         "window_minutes": 15, "filter": "Is Trace = true, Tag = clef_budget_warning"},
        {"name": f"{PREFIX}: fallback deciding most checks", "metric": "Feedback Score (key safety.fallback_share)",
         "aggregation": "Average", "condition": "greater than 0.5", "window_minutes": 60, "filter": "Is Trace = true"},
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
    lines.append("    Safety (feature 3 O), with the same channel, URL, headers and body:")
    for r in safety_alert_rules():
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
    boards = all_dashboards("preview")
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
    for index, board in enumerate(all_dashboards(project)):
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
            "feedback_count": f"count of feedback {s.get('key')}", "feedback_sum": f"sum of feedback {s.get('key')}"}[kind]


def markdown(part: str = "feature2") -> str:
    """The doc's tables (#77), made from the definitions above, so the doc can't drift from what's built.
    part "feature2": feature 2 K's three dashboards and alerts; "safety": feature 3 O's dashboard and alerts."""
    cell = lambda text: (text or "").replace("|", "\\|")
    lines = []
    if part == "feature2":
        boards, rules = dashboards("<project>"), alert_rules()
    else:
        boards, rules = [safety_dashboard("<project>")], safety_alert_rules()
    for board in boards:
        lines += [f"#### {board['title']}", "", board["description"], "",
                  "| # | Chart | Type | Series: metric | Filter | Grouped by |", "|---|---|---|---|---|---|"]
        for n, ch in enumerate(board["charts"], 1):
            for i, s in enumerate(ch["series"]):
                group = "" if not s["group"] else (f"metadata.{s['group'][1]}" if s["group"][1] else s["group"][0])
                lines.append(f"| {n if i == 0 else ''} | {cell(ch['title']) if i == 0 else ''} | {ch['chart_type'] if i == 0 else ''} | "
                             f"{cell(s['name'])}: {describe(s)} | `{cell(s['filter'] or ROOT_FILTER)}` | {group} |")
        lines.append("")
    lines += ["#### Alerts (made in the UI)", "", "| Alert | Metric / aggregation | Fires when | Window | Filter |", "|---|---|---|---|---|"]
    for r in rules:
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
    parser.add_argument("--markdown", nargs="?", const="feature2", choices=["feature2", "safety"],
                        help="print the doc's chart and alert tables (feature 2's, or 'safety': feature 3 O's); no API calls")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", action="store_true", help="run every series' filter and grouping on recent data; create nothing")
    a = parser.parse_args()
    if a.markdown:
        print(markdown(a.markdown))
    elif not a.project:
        parser.error("--project is required")
    else:
        main(a.project, a.dry_run, a.preview)
