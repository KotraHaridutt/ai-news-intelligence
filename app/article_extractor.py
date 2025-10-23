from newspaper import Article
import trafilatura

def extract_text(url):
    """
    Extract article text with Newspaper3k first, then Trafilatura fallback.
    """
    try:
        art = Article(url)
        art.download()
        art.parse()
        if len(art.text) > 300:
            return art.text
    except:
        pass

    try:
        downloaded = trafilatura.fetch_url(url)
        extracted = trafilatura.extract(downloaded)
        if extracted and len(extracted) > 300:
            return extracted
    except:
        pass

    return ""
