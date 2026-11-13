"""
Feature 2 evals: the app's answer step on frozen data (used by calibrate.py and the experiment runner).

generate_answer runs exactly what run_rag_query_with_store runs after retrieval (the view's prompt,
the [Source: ...] headers, the chat model with its retries and fallback, the 75 s deadline), on the
context frozen in Section F, so an answer differs from another only by what an experiment changed.
"""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from services import prompts, rag_service as rag
from services.safety import gemini


def documents(context: list[dict]) -> list[Document]:
    return [Document(page_content=c["page_content"],
                     metadata={"title": c.get("title"), "source_name": c.get("source_name"), "published_at": c.get("published_at"),
                               "theme_id": c.get("theme_id", -1), "chunk_id": c.get("chunk_id"), "source": c.get("chunk_id")})
            for c in context]


async def generate_answer(query: str, view: str, context: list[dict], template=None) -> dict:
    """One chat call. template: a prompt to try instead of the app's (an experiment: a Prompt, or one
    older text); default the app's. Sent as the app sends it (system + user parts, feature 3 E)."""
    template = rag.as_prompt(view, template or rag.ANSWER_PROMPTS[view])
    prompt, marker = rag.answer_prompt(template)        # the canary too, when the prompt has one
    # As the app calls it (feature 5 E1): the view's output cap (#66) and the block check, so an answer cut
    # off at its cap shows here too. finish_reason is read on the way, as in run_rag_query_with_store.
    finish = {}

    def note_finish(message):
        finish["reason"] = gemini.finish_reason(message)
        return message
    model = rag.with_cap(rag.pick_chat_model(rag.llm, rag.llm_fallback_only), view)
    chain = create_stuff_documents_chain(model | rag.BLOCK_CHECK | RunnableLambda(note_finish, name="finish_reason"), prompt,
                                         document_prompt=rag.DOCUMENT_PROMPT)
    measured = Measured()
    started = time.perf_counter()
    answer = await asyncio.wait_for(
        chain.ainvoke({"input": rag.as_query(query), "context": rag._with_source_headers(documents(context)),
                       **({"canary": marker} if marker else {})},
                      config={"run_name": "generate", "tags": [view, "eval"], "callbacks": [measured],
                              "metadata": {**rag.prompt_metadata(view, template), "config_version": rag.CONFIG_VERSION}}),
        rag.CHAT_DEADLINE_SECONDS)
    # Feature 5 #26: the answer's time (retries and fallback included) and the tokens of the attempt that answered
    return {"answer": answer.strip(), "prompt_version": template.version, "config_version": rag.CONFIG_VERSION,
            "seconds": round(time.perf_counter() - started, 2), "finish_reason": finish.get("reason"), **measured.tokens}


class Measured(AsyncCallbackHandler):
    """The input and output tokens of the last chat attempt that ended (the one whose answer is used)."""

    def __init__(self):
        self.tokens = {}

    async def on_llm_end(self, response, **kwargs):
        try:
            usage = response.generations[0][0].message.usage_metadata or {}
        except (AttributeError, IndexError):
            return
        if usage:
            self.tokens = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}
