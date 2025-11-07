"""Web search helpers to fetch current context from the web.

Provides a small wrapper around Bing Web Search (Azure/Bing) and SerpAPI
for optional context enrichment. Returns a list of dicts: {title, snippet, url}.

Environment variables supported (see src.config.Settings):
- BING_API_KEY, BING_ENDPOINT
- SERPAPI_KEY

Note: This is a lightweight helper for demo purposes. For production use,
handle rate limits, retries, caching, and parsing more robustly.
"""
from typing import List, Dict
import os
import requests
from src.config import settings


def _bing_search(query: str, k: int = 5) -> List[Dict]:
    headers = {"Ocp-Apim-Subscription-Key": settings.BING_API_KEY}
    params = {"q": query, "count": k}
    resp = requests.get(settings.BING_ENDPOINT, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = []
    webPages = data.get("webPages", {}).get("value", [])
    for it in webPages[:k]:
        results.append({"title": it.get("name"), "snippet": it.get("snippet"), "url": it.get("url")})
    return results


def _serpapi_search(query: str, k: int = 5) -> List[Dict]:
    # SerpAPI simple JSON interface
    key = settings.SERPAPI_KEY
    url = "https://serpapi.com/search.json"
    params = {"q": query, "api_key": key, "num": k}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for it in data.get("organic_results", [])[:k]:
        results.append({"title": it.get("title"), "snippet": it.get("snippet"), "url": it.get("link")})
    return results


def search_web(query: str, k: int = 5) -> List[Dict]:
    """Search the web using available provider. Returns list of results.

    Provider selection order:
    1. Bing (if BING_API_KEY present)
    2. SerpAPI (if SERPAPI_KEY present)
    3. Empty list if none configured
    """
    # Prefer Google Custom Search when configured
    try:
        if getattr(settings, "GOOGLE_API_KEY", None) and getattr(settings, "GOOGLE_CX", None):
            return _google_search(query, k=k)
    except Exception:
        pass

    # Next prefer Bing
    try:
        if settings.BING_API_KEY:
            return _bing_search(query, k=k)
    except Exception:
        pass

    # Fallback to SerpAPI
    try:
        if settings.SERPAPI_KEY:
            return _serpapi_search(query, k=k)
    except Exception:
        pass

    return []


def _google_search(query: str, k: int = 5) -> List[Dict]:
    """Call Google Custom Search JSON API and return list of {title,snippet,url}."""
    key = settings.GOOGLE_API_KEY
    cx = settings.GOOGLE_CX
    if not key or not cx:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": key, "cx": cx, "num": k}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for it in data.get("items", [])[:k]:
        results.append({"title": it.get("title"), "snippet": it.get("snippet"), "url": it.get("link")})
    return results
