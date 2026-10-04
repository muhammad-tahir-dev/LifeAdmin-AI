"""Tavily web research helpers."""

import logging

logger = logging.getLogger("lifeadmin.research.web")


def create_tavily_client(api_key):
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def _normalize(response):
    return [
        {
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
        }
        for item in (response or {}).get("results", [])
    ]


def web_research(tavily_client, query, max_results=5, include_domains=None):
    """Run one Tavily search. Raises on API errors (see the safe_* wrappers)."""
    kwargs = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
    }
    if include_domains:
        kwargs["include_domains"] = list(include_domains)
    return _normalize(tavily_client.search(**kwargs))


def safe_web_research(tavily_client, query, max_results=5, include_domains=None):
    try:
        return web_research(tavily_client, query, max_results, include_domains)
    except Exception as exc:
        logger.warning("Web research failed: %s", type(exc).__name__)
        return []


def official_web_research(tavily_client, query, domains=None, max_results=8):
    """Search biased towards official sources.

    If `domains` is given, results are restricted to those domains. Otherwise a
    neutral "official website" hint is appended (no topic is hardcoded, so
    passport, CNIC, internship or visa questions all work).
    """
    if domains:
        return safe_web_research(
            tavily_client, query, max_results=max_results, include_domains=domains
        )
    return safe_web_research(
        tavily_client,
        f"{query} official government website",
        max_results=max_results,
    )
