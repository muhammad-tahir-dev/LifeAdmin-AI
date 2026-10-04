"""Source classification, scoring, cleaning and prioritisation.

Trust decisions are made on the URL's *hostname* only. Substring matching on
the whole URL is spoofable (e.g. passport.gov.pk.phish.xyz), so it is not used.
"""

from urllib.parse import urlparse

from . import config


def get_hostname(url):
    """Return the lower-cased hostname of a URL ('' if it cannot be parsed)."""
    if not url:
        return ""
    text = url.strip()
    if "//" not in text:
        text = "//" + text
    try:
        host = urlparse(text).hostname or ""
    except ValueError:
        return ""
    return host.lower().rstrip(".")


def _matches(host, domains):
    return any(host == d or host.endswith("." + d) for d in domains)


def is_official(url):
    return _matches(get_hostname(url), config.OFFICIAL_DOMAINS)


def get_source_type(url):
    host = get_hostname(url)
    if not host:
        return "other"
    if _matches(host, config.OFFICIAL_DOMAINS):
        return "official"
    if _matches(host, config.COMMUNITY_DOMAINS):
        return "community"
    return "other"


def score_source(result):
    source_type = get_source_type(result.get("url", ""))
    return {
        **result,
        "source_type": source_type,
        "confidence": config.SOURCE_CONFIDENCE[source_type],
    }


def filter_official_sources(results):
    return [r for r in results if get_source_type(r.get("url", "")) == "official"]


def _dedup_key(url):
    parsed = urlparse(url.strip())
    return (
        (parsed.hostname or "").lower(),
        parsed.path.rstrip("/"),
        parsed.query,
    )


def clean_web_sources(web_results):
    """Drop entries without a URL and remove duplicate URLs."""
    unique_sources = []
    seen = set()

    for result in web_results or []:
        url = (result.get("url") or "").strip()
        if not url:
            continue
        key = _dedup_key(url)
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(
            {
                "title": (result.get("title") or "").strip(),
                "url": url,
                "content": result.get("content") or "",
            }
        )

    return unique_sources


def prioritize_sources(web_results):
    """Official sources first, then other, then community (stable order)."""
    rank = {"official": 0, "other": 1, "community": 2}
    return sorted(
        web_results,
        key=lambda r: rank[get_source_type(r.get("url", ""))],
    )
