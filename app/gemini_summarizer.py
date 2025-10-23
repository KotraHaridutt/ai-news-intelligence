import google.generativeai as genai
from .config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def summarize_with_gemini(query, article_text): # <-- Renamed combined_text
    """
    Summarize a single article's text using Gemini API.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    # --- THIS IS THE NEW PROMPT ---
    # It's now focused on summarizing one article
    prompt = f"""
    Provide a detail and insightful summary in depth of the following article, focusing on all
    its key points and relevance to the topic: '{query}'.

    - Do NOT use a persona (like "As an analyst...").
    - Do NOT use Markdown headings.
    - Write it as a clean, professional paragraph or two.
    - You may use **bold** for emphasis.

    Article Text:
    {article_text[:15000]}
    """

    response = model.generate_content(prompt)
    return response.text