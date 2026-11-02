import asyncio
import json
import re
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from services import errors, observability, online_checks, prompts, usage
from services import rag_service as rag
from services.novelty_service import SHOWN, NOTHING_NEW, UNJUDGED
from services.safety import articles as safety_articles, canary, gemini, privacy, provenance

# --- feature 1: the fact check of a "Check for updates" ---
# One LLM call per check (only when the check found unseen articles). novelty_service has
# already decided, by measured distances, which articles are unseen and which story each
# belongs to. This asks the model one thing per story: what in these articles was NOT
# already told to the reader? A new story gets a summary without the facts already told;
# a known story gets only its new facts. null = nothing new, and the story is hidden.
# The answer is JSON (a response schema), so the code acts on null instead of parsing prose.

# Chunks per article (feature-1 decision #12, measured in Section A): the first 6 hold the
# news (the lead); one live blog had 45. With 2 articles per story, at most 12 per story.
CHUNKS_PER_ARTICLE = 6
# What the code accepts from the model's JSON (feature 3 #71). A text is a story's 2-5 sentence summary
# or its list of new facts (a few hundred to ~1,000 characters); one four times longer than that isn't
# what was asked for (a page that got the model to copy it out, or a runaway reply), and its story is
# left unjudged. One entry per story sent is expected; past twice that, the rest isn't read.
MAX_TEXT_CHARS = 4000
MAX_ENTRIES_PER_STORY = 2
# The fact check's settings: the answers' (chunking, thinking, timeouts) plus its own chunk count and caps
FACT_CHECK_CONFIG_VERSION = prompts.version_of(json.dumps({**rag.ANSWER_CONFIG, "chunks_per_article": CHUNKS_PER_ARTICLE,
                                                           "max_output_tokens": rag.FACT_CHECK_MAX_OUTPUT_TOKENS,
                                                           "max_text_chars": MAX_TEXT_CHARS,
                                                           "max_entries_per_story": MAX_ENTRIES_PER_STORY},
                                                          sort_keys=True))


class NewStoryAnswer(BaseModel):
    story_id: str = Field(description="The id of a new story from STORIES TO JUDGE, e.g. new-1")
    summary: str | None = Field(description="What the story's excerpts report, without facts already told; null if nothing is left")

class UpdateAnswer(BaseModel):
    story_id: str = Field(description="The id of a known story from STORIES TO JUDGE, e.g. s3")
    new_facts: str | None = Field(description="Only the facts about this story that were not already told; null if there are none")

class FactCheckAnswer(BaseModel):
    new_stories: list[NewStoryAnswer] = Field(description="One entry per new story id")
    updates: list[UpdateAnswer] = Field(description="One entry per known story id")


def _structured(model_name, role="primary"):
    # Structured output per model, then the fallback: the fallback wrapper itself has no
    # with_structured_output (plan P7). Same thinking level, timeout and attempts as the views;
    # the retries wrap the structured model, because a retry wrapper has no with_structured_output
    # either. Only busy answers are retried: an answer that can't be parsed is not.
    # include_raw (feature 3 #48): the reply comes back with its parse result, so a reply Gemini's
    # filter blocked is returned as it is and never reaches the fallback (it used to fail the parser
    # and spend a second call, A1 §3.3). An unreadable reply still raises and goes to the fallback.
    # Output cap (feature 3 #66): a reply cut at the cap is unreadable JSON, handled as one already
    structured = rag._chat_model(model_name, role, rag.FACT_CHECK_MAX_OUTPUT_TOKENS).with_structured_output(FactCheckAnswer, method="json_schema", include_raw=True)
    return rag._retrying(structured | RunnableLambda(gemini.structured_raw_or_raise, name="parse_or_blocked"))

try:
    fact_check_llm = _structured(rag.CHAT_MODEL).with_fallbacks([_structured(rag.FALLBACK_CHAT_MODEL, "fallback")])
    # Used instead while the primary's daily quota is used up (feature 2 #34)
    fact_check_fallback_only = _structured(rag.FALLBACK_CHAT_MODEL, "fallback")
except Exception as e:
    print(f"[ERROR] Failed to load the fact-check model: {e}")
    fact_check_llm = None
    fact_check_fallback_only = None


# The prompt is in services/prompts.py (FACT_CHECK), with its version (feature 2 E): a system part
# (the rules) and a user part (the blocks), feature 3 E. An experiment may put a one-part text here.
UPDATE_PROMPT = prompts.get("fact_check")

# Scraped text and earlier answers go inside the two blocks. rag._as_data already rewrites
# <articles> tags and "[Source:" headers; <already_told> tags are rewritten the same way, so
# no text can close either block early.
_TOLD_TAG = re.compile(r"<\s*(/?)\s*already_told\s*>", re.IGNORECASE)

def _as_data(text) -> str:
    return _TOLD_TAG.sub(lambda m: f"[{m.group(1)}already_told]", rag._as_data(str(text or "")))

def _line(text) -> str:
    return _as_data(rag._header_text(text))


# --- Building the prompt ---

def _story_line(story) -> str:
    if story["kind"] == "update":
        return f'- {story["id"]}: known story "{_line(story["title"])}". Report only its new facts.'
    related = story.get("related_to")
    if related:
        return f'- {story["id"]}: new story, related to known story {related["id"]} "{_line(related["title"])}".'
    return f'- {story["id"]}: new story.'

def _told_label(entry) -> str:
    try:
        when = datetime.fromisoformat(entry["at"].replace("Z", "+00:00")).strftime("%d %b").lstrip("0")
    except (KeyError, ValueError):
        when = "earlier"
    return f'{entry.get("kind", "text").capitalize()}, {when}'

def _already_told(known_stories, told) -> str:
    lines = ["Stories the reader knows:"]
    lines += [f'- {s["id"]}: {_line(s["title"])}' for s in known_stories] or ["- (none)"]
    for entry in told:                       # oldest first, as the snapshot keeps them
        lines.append(f"\n[{_told_label(entry)}]\n{_as_data(entry['text'])}")
    return "\n".join(lines)

def _excerpts(stories):
    """The excerpts of every article to judge, story by story: (header + text, chunk count)."""
    blocks = []
    for story in stories:
        for article in story["articles"]:
            # The same chunks the injection scan judged: invisible characters removed first (#31)
            for chunk in safety_articles.article_chunks(article, rag.text_splitter, CHUNKS_PER_ARTICLE):
                header = (f"[Source: {_line(article.get('title')) or 'untitled'} | "
                          f"{_line(article.get('source_name')) or 'unknown outlet'} | "
                          f"Published: {_as_data(rag._published_text(article.get('published_at')))} | "
                          f"Site: {_line(provenance.site_line(article))} | "      # feature 3 #51
                          f"Story: {story['id']}]")
                blocks.append(f"{header}\n{_as_data(chunk)}")
    return blocks

def build_messages(query, stories, known_stories, told, marker=None) -> tuple[list, int]:
    """
    The fact check's messages and the number of excerpts in them: the system part as a system
    message (Gemini's system_instruction, feature 3 #39), the user part as the user's message.
    A one-part prompt (an experiment's older text) is one user message, as before.
    marker: the request's canary (feature 3 #36); a fresh one when not given.
    """
    blocks = _excerpts(stories)
    prompt = rag.as_prompt("fact_check", UPDATE_PROMPT)
    values = dict(query=rag.as_query(_line(query)), stories="\n".join(_story_line(s) for s in stories),
                  already_told=_already_told(known_stories, told), context="\n\n".join(blocks), canary=marker or canary.new())
    messages = [HumanMessage(prompt.template.format(**values))]
    if prompt.system:
        messages.insert(0, SystemMessage(prompt.system.format(**values)))
    return messages, len(blocks)

def build_prompt(query, stories, known_stories, told, marker=None) -> tuple[str, int]:
    """The whole prompt as one text (system part, then user part) and the number of excerpts:
    for the evals, which store its size and show it."""
    messages, excerpts = build_messages(query, stories, known_stories, told, marker)
    return "\n\n".join(m.content for m in messages), excerpts


async def _cleaned_stories(stories) -> list:
    """The stories with their articles cleaned of text written for the AI (D #37), when the
    injection check (enforce mode) removed any; the same stories otherwise."""
    articles = [a for s in stories for a in s["articles"]]
    cleaned = await safety_articles.clean_articles(articles, rag.text_splitter, "fact_check", max_chunks=CHUNKS_PER_ARTICLE)
    if cleaned is None:
        return stories
    out, i = [], 0
    for story in stories:
        out.append({**story, "articles": cleaned[i:i + len(story["articles"])]})
        i += len(story["articles"])
    return out


# --- Reading the answer ---

def _usable(text) -> bool:
    return isinstance(text, str) and text.strip() != ""

def read_answer(answer, stories) -> dict:
    """
    The model's answer as novelty_service outcomes, per story id that was sent:
      text            -> SHOWN, with the text
      null or blank   -> NOTHING_NEW (hidden)
      missing         -> UNJUDGED (a new story is shown without a summary; a known story
                         stays hidden and its articles are retried on the next check)
    Ids the code didn't send are ignored. answer None (unusable) -> everything UNJUDGED.
    Feature 3 #71: a text over MAX_TEXT_CHARS leaves its story UNJUDGED, and past MAX_ENTRIES_PER_STORY
    entries per story sent the rest isn't read. "rejected" names both for the output event.
    """
    sent = {s["id"]: s["kind"] for s in stories}
    outcomes = {story_id: UNJUDGED for story_id in sent}
    texts, decided, too_long, rejected = {}, set(), [], []
    if answer is not None:
        given = [(a.story_id, a.summary, "new") for a in answer.new_stories] + \
                [(a.story_id, a.new_facts, "update") for a in answer.updates]
        limit = MAX_ENTRIES_PER_STORY * len(sent)
        if len(given) > limit:
            rejected.append(f"entries_over_cap={len(given) - limit}")
            given = given[:limit]
        for story_id, text, kind in given:
            if sent.get(story_id) != kind or story_id in decided:
                continue                      # not sent, sent as the other kind, or answered twice
            decided.add(story_id)
            if _usable(text) and len(text.strip()) > MAX_TEXT_CHARS:
                too_long.append(story_id)     # stays UNJUDGED
            elif _usable(text):
                outcomes[story_id], texts[story_id] = SHOWN, text.strip()
            else:
                outcomes[story_id] = NOTHING_NEW
    if too_long:
        rejected.append(f"too_long={len(too_long)}")
        print(f"[WARN] UPDATE: {len(too_long)} text(s) over {MAX_TEXT_CHARS:,} characters left unjudged: {', '.join(too_long)}")
    return {"outcomes": outcomes, "texts": texts, "usable": answer is not None, "rejected": rejected}


async def run_fact_check(query, stories, known_stories, told) -> dict:
    """
    One LLM call for a check. stories: [{id, kind: "new" | "update", title (known story),
    related_to: {id, title} | None, articles: [article dicts with full_text]}], in display
    order; known_stories: [{id, title}]; told: the snapshot's "already told" texts.
    Returns {"outcomes", "texts", "usable", "rejected", "blocked"} (see read_answer). An answer that isn't valid
    JSON for the schema is unusable, not an error: the check still shows its new stories. The same
    when Gemini's filter blocked it: "blocked" is then "prompt" or "answer" (else None), feature 3 #48.
    Raises AppError (503 busy / 504 too long / 502 other / 500 not set up) if the call fails.
    """
    if not stories:
        return {"outcomes": {}, "texts": {}, "usable": True, "blocked": None, "rejected": []}
    if fact_check_llm is None:
        raise errors.ai_not_configured("fact-check model not loaded")

    stories = await _cleaned_stories(stories)
    blocked = None       # "prompt" / "answer" when Gemini's filter blocked the check (#48)
    marker = canary.new() if "canary" in rag.as_prompt("fact_check", UPDATE_PROMPT).variables else None
    messages, excerpts = build_messages(query, stories, known_stories, told, marker)
    prompt_chars = sum(len(m.content) for m in messages)
    articles = sum(len(s["articles"]) for s in stories)
    new_count = sum(1 for s in stories if s["kind"] == "new")
    versions = {**rag.prompt_metadata("fact_check", UPDATE_PROMPT), "config_version": FACT_CHECK_CONFIG_VERSION}
    observability.add_totals(prompt_version=versions["prompt_version"], config_version=FACT_CHECK_CONFIG_VERSION)
    async with observability.span("fact_check", inputs={"query": query, "stories": [s["id"] for s in stories]},
                                  metadata={**versions, "new": new_count, "known": len(stories) - new_count, "articles": articles,
                                            "chunks": excerpts, "prompt_chars": prompt_chars, "chunks_per_article": CHUNKS_PER_ARTICLE,
                                            "chat_model": rag.CHAT_MODEL, "fallback_model": rag.FALLBACK_CHAT_MODEL},
                                  tags=["answer"]) as run:
        model = rag.pick_chat_model(fact_check_llm, fact_check_fallback_only)
        try:
            answer = await asyncio.wait_for(model.ainvoke(messages, config={"run_name": "generate", "tags": ["update"]}),
                                            rag.CHAT_DEADLINE_SECONDS)
            # The parsed answer, or GeminiBlocked when the filter blocked the prompt or the reply
            answer = gemini.structured_answer(answer)
            if not isinstance(answer, FactCheckAnswer):
                raise OutputParserException(f"expected FactCheckAnswer, got {type(answer).__name__}")
        except gemini.GeminiBlocked as e:
            # Feature 3 #48: like an unreadable answer, every story is unjudged (new stories are listed
            # without a summary, known ones are tried again next check), with its own message
            print(f"[WARN] UPDATE: Gemini's filter blocked the fact check ({e.block.label()}). Every story unjudged.")
            run.add_tags(["gemini_blocked"])
            gemini.record_block(e, where="update")
            answer, blocked = None, e.block.stage
        except OutputParserException as e:
            print(f"[WARN] UPDATE: the fact check's answer couldn't be read ({str(e)[:200]}). Every story unjudged.")
            # The whole reply is in the LLM run below this span; the start of it is kept here (#27)
            raw = getattr(e, "llm_output", None) or ""
            run.add_tags(["unreadable_answer"])
            run.add_metadata({"parse_error": str(e)[:500], "raw_reply_chars": len(raw), "raw_reply_start": raw[:1000]})
            answer = None
        except Exception as e:                    # a missed deadline is a TimeoutError -> 504
            if isinstance(e, TimeoutError):
                run.add_tags(["timeout"])
                # The attempt the deadline cut off would stay "pending" in LangSmith (Section L)
                usage.forget(observability.close_open_llm_runs(rag.DEADLINE_CANCELLED))
            raise errors.report_failed(e, "update") from e

        result = {**read_answer(answer, stories), "blocked": blocked}
        # The canary in any story's text: the whole check is withheld in enforce mode (D #36)
        if marker and result["texts"]:
            await canary.check_text("\n".join(result["texts"].values()), marker, where="update", prompt_type="update")
            result["texts"] = {k: canary.strip(v, marker) for k, v in result["texts"].items()}
        # Contact details the prompt's articles don't contain: masked in enforce mode (H #60)
        result["texts"] = await privacy.screen_texts(result["texts"], "\n".join(str(m.content) for m in messages),
                                                     where="update")
        counts = {kind: sum(1 for story_id, o in result["outcomes"].items() if o == kind) for kind in (SHOWN, NOTHING_NEW, UNJUDGED)}
        run.add_outputs({"usable": result["usable"], "blocked": blocked, **counts})
        # The fact check's health on this check (Section I #66): feedback on the trace, never an error
        online_checks.fact_check(answer, result["outcomes"], [s["id"] for s in stories])
    print(f"[INFO] UPDATE: judged {articles} articles in {len(stories)} stories ({new_count} new, {len(stories) - new_count} known), "
          f"{excerpts} chunks, {prompt_chars:,} characters; result: {counts[SHOWN]} shown, "
          f"{counts[NOTHING_NEW]} nothing new, {counts[UNJUDGED]} unjudged")
    return result
