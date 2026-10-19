"""
Feature 2 evals: the app's answer step on frozen data (used by calibrate.py and the experiment runner).

generate_answer runs exactly what run_rag_query_with_store runs after retrieval (the view's prompt,
the [Source: ...] headers, the chat model with its retries and fallback, the 75 s deadline), on the
context frozen in Section F, so an answer differs from another only by what an experiment changed.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from services import prompts, rag_service as rag


def documents(context: list[dict]) -> list[Document]:
    return [Document(page_content=c["page_content"],
                     metadata={"title": c.get("title"), "source_name": c.get("source_name"), "published_at": c.get("published_at"),
                               "theme_id": c.get("theme_id", -1), "chunk_id": c.get("chunk_id"), "source": c.get("chunk_id")})
            for c in context]


async def generate_answer(query: str, view: str, context: list[dict], template: str | None = None) -> dict:
    """One chat call. template: a prompt to try instead of the app's (an experiment); default the app's."""
    template = template or rag.PROMPT_TEMPLATES[view]
    prompt = PromptTemplate(template=template, input_variables=["context", "input"])
    chain = create_stuff_documents_chain(rag.pick_chat_model(rag.llm, rag.llm_fallback_only), prompt, document_prompt=rag.DOCUMENT_PROMPT)
    answer = await asyncio.wait_for(
        chain.ainvoke({"input": query, "context": rag._with_source_headers(documents(context))},
                      config={"run_name": "generate", "tags": [view, "eval"],
                              "metadata": {**rag.prompt_metadata(view, template), "config_version": rag.CONFIG_VERSION}}),
        rag.CHAT_DEADLINE_SECONDS)
    return {"answer": answer.strip(), "prompt_version": prompts.version_of(template), "config_version": rag.CONFIG_VERSION}
