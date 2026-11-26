from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

# --- Excessive agency (feature 3, Section M #96) ---
# The model has no tools: it writes text, and the code decides everything that happens with it. The
# M doc's agency map lists what the model's output can and can't affect. This keeps it that way: it
# walks every chat model the app calls (through the retry, fallback, structured-output and cap
# wrappers) and reports any tool or function bound to one. The startup line says the result, and a
# test fails if anything is found, so a tool can't be added without someone deciding to.

# Arguments that give a chat model tools or functions to call (LangChain's bind_tools, Gemini's
# tool_config, OpenAI-style function calling)
TOOL_ARGUMENTS = ("tools", "functions", "tool_choice", "tool_config", "function_call", "functions_call")
# Where LangChain's wrappers keep the runnables they wrap
_CHILDREN = ("bound", "runnable", "fallbacks", "steps", "steps__", "mapper", "first", "middle", "last", "default", "branches")


def _children(runnable) -> list:
    found = []
    for name in _CHILDREN:
        value = getattr(runnable, name, None)
        if value is None:
            continue
        if isinstance(value, dict):
            value = list(value.values())
        elif not isinstance(value, (list, tuple)):
            value = [value]
        for item in value:
            if isinstance(item, tuple):          # RunnableBranch: (condition, runnable)
                found.extend(item)
            else:
                found.append(item)
    return [item for item in found if isinstance(item, Runnable)]


def inspect(runnable) -> dict:
    """{"models": chat models reached, "tools": ["<where>: <argument>", ...]} for one runnable tree."""
    models, tools, seen = 0, [], set()

    def walk(node, path):
        if id(node) in seen:
            return
        seen.add(id(node))
        name = type(node).__name__
        kwargs = getattr(node, "kwargs", None)
        if isinstance(kwargs, dict):
            tools.extend(f"{path}/{name}: {arg}" for arg in TOOL_ARGUMENTS if kwargs.get(arg))
        if isinstance(node, BaseChatModel):
            nonlocal models
            models += 1
            tools.extend(f"{path}/{name}: {arg}" for arg in TOOL_ARGUMENTS if getattr(node, arg, None))
        for child in _children(node):
            walk(child, f"{path}/{name}")

    walk(runnable, "")
    return {"models": models, "tools": tools}


def chat_models() -> dict:
    """Every chat model the app answers with, by name (None when it couldn't load, e.g. no key)."""
    from services import fact_check_service, rag_service      # they import the safety package
    return {"rag.llm": rag_service.llm, "rag.llm_fallback_only": rag_service.llm_fallback_only,
            "fact_check.llm": fact_check_service.fact_check_llm,
            "fact_check.fallback_only": fact_check_service.fact_check_fallback_only}


def check() -> dict:
    """{"chains": loaded chains, "models": chat models reached, "tools": every tool binding found}."""
    chains = {name: chain for name, chain in chat_models().items() if chain is not None}
    models, tools = 0, []
    for name, chain in chains.items():
        result = inspect(chain)
        models += result["models"]
        tools.extend(f"{name}{where}" for where in result["tools"])
    return {"chains": len(chains), "models": models, "tools": tools}


def status() -> str:
    """For the startup line: 'no tools on 4 chat chains (8 chat models)'. A tool found prints an error."""
    result = check()
    if result["tools"]:
        print(f"[ERROR] Agency: a tool is bound to a chat model, which the agency map (feature 3 M #96) doesn't allow: "
              f"{'; '.join(result['tools'])}")
        return f"{len(result['tools'])} TOOL BINDING(S) on {result['chains']} chat chains"
    return f"no tools on {result['chains']} chat chains ({result['models']} chat models)"
