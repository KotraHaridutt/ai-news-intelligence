def format_response(query, summary, articles):
    """
    Format the final output including Gemini summary and sources.
    """
    sources_list = "\n".join(
        [f"{i+1}. {a['source']} — {a['title']} → {a['url']}"
         for i, a in enumerate(articles)]
    )

    return {
        "query": query,
        "summary": summary,
        "sources": articles,
        "readable_sources": f"\n📚 Sources:\n{sources_list}"
    }
