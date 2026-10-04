"""The research pipeline: one function Member 2/3 can call.

    result = run_research("passport renewal documents", user_id="u1")

Flow:  knowledge base -> (web research if needed) -> source scoring ->
       evidence extraction -> conflict detection -> trust-aware answer.

At most 3 LLM calls per request (extract, conflicts, answer). Every failure mode
degrades to a structured result instead of an exception.
"""

import logging
from datetime import date, datetime, timezone

from . import config
from .conflict_detection import detect_conflicts, extract_source_evidence, rank_evidence
from .llm import LLMUnavailable, groq_call
from .rag import search_documents
from .research_api import (
    add_confidence_notice,
    build_research_result,
    calculate_overall_confidence,
    generate_trust_aware_answer,
    make_api_response,
)
from .source_scoring import (
    clean_web_sources,
    get_source_type,
    prioritize_sources,
    score_source,
)
from .web_research import official_web_research, safe_web_research

logger = logging.getLogger("lifeadmin.research.pipeline")

NO_EVIDENCE_ANSWER = (
    "I could not find reliable information for this question. "
    "Please check the relevant official government or institution website."
)
AI_UNAVAILABLE_ANSWER = (
    "The AI summarisation service is temporarily unavailable. "
    "The sources found are listed below; please check them directly."
)


def _today():
    return datetime.now(timezone.utc).date().isoformat()


def _kb_evidence(hits, start_number):
    """Turn knowledge-base hits into evidence items (no LLM needed)."""
    evidence = []
    for offset, hit in enumerate(hits):
        meta = hit["metadata"]
        source_type = meta.get("source_type", "user_document")
        evidence.append(
            {
                "source_number": start_number + offset,
                "title": meta.get("title") or "Knowledge base",
                "url": meta.get("source_url", ""),
                "evidence": hit["text"][: config.MAX_SOURCE_CHARS],
                "source_type": source_type,
                "confidence": config.SOURCE_CONFIDENCE.get(
                    source_type, config.SOURCE_CONFIDENCE["other"]
                ),
                "retrieved_at": meta.get("date_fetched"),
            }
        )
    return evidence


def _kb_is_fresh_official(hits):
    """True if an official, recently-fetched chunk already answers the query."""
    today = date.today()
    for hit in hits:
        meta = hit["metadata"]
        if meta.get("source_type") != "official":
            continue
        try:
            fetched = date.fromisoformat(str(meta.get("date_fetched", ""))[:10])
        except ValueError:
            continue
        if (today - fetched).days <= config.KB_MAX_AGE_DAYS:
            return True
    return False


def _gather_web_sources(tavily_client, query, official_domains):
    results = official_web_research(tavily_client, query, domains=official_domains)
    cleaned = clean_web_sources(results)

    if not any(get_source_type(r["url"]) == "official" for r in cleaned):
        # Restricted pass found nothing official: open search, rely on scoring.
        cleaned = clean_web_sources(cleaned + safe_web_research(tavily_client, query))

    stamp = _today()
    scored = []
    for source in prioritize_sources(cleaned):
        scored.append({**score_source(source), "retrieved_at": stamp})
    return scored[: config.MAX_WEB_SOURCES]


def _source_label(has_kb, has_web):
    if has_kb and has_web:
        return "KNOWLEDGE_BASE+WEB"
    if has_kb:
        return "KNOWLEDGE_BASE"
    if has_web:
        return "WEB"
    return "NONE"


def run_research(
    query,
    collection=None,
    tavily_client=None,
    llm=groq_call,
    user_id=None,
    force_web=False,
    official_domains=None,
):
    """Run the full research pipeline and return a structured result dict.

    collection / tavily_client may be None (that source is simply skipped).
    official_domains: restrict the first Tavily pass; [] disables the restricted
    pass, None uses config.DEFAULT_OFFICIAL_SEARCH_DOMAINS.
    """
    if official_domains is None:
        official_domains = config.DEFAULT_OFFICIAL_SEARCH_DOMAINS

    # 1. Knowledge base -------------------------------------------------
    kb_hits = []
    if collection is not None:
        try:
            kb_hits = search_documents(collection, query, owner=user_id)
        except Exception as exc:
            logger.warning("Knowledge-base search failed: %s", type(exc).__name__)
    kb_evidence = _kb_evidence(kb_hits, start_number=1)

    # 2. Web research (skipped when fresh official KB data already answers) --
    web_sources = []
    need_web = force_web or not _kb_is_fresh_official(kb_hits)
    if need_web and tavily_client is not None:
        web_sources = _gather_web_sources(tavily_client, query, official_domains)

    # 3. Evidence extraction from web sources ----------------------------
    web_evidence = []
    try:
        web_evidence = extract_source_evidence(
            query, web_sources, llm, start_number=len(kb_evidence) + 1
        )
    except LLMUnavailable as exc:
        logger.warning("Evidence extraction unavailable: %s", exc)
        if not kb_evidence:
            return make_api_response(
                build_research_result(
                    query,
                    AI_UNAVAILABLE_ANSWER,
                    _source_label(False, bool(web_sources)),
                    "low",
                    sources=web_sources,
                    status="AI_UNAVAILABLE",
                )
            )

    evidence = rank_evidence(kb_evidence + web_evidence, web_sources)
    source_label = _source_label(bool(kb_evidence), bool(web_evidence))

    if not evidence:
        return make_api_response(
            build_research_result(
                query, NO_EVIDENCE_ANSWER, "NONE", "low",
                sources=web_sources, status="NO_EVIDENCE",
            )
        )

    # 4. Conflict detection ----------------------------------------------
    conflicts, conflicts_checked = [], True
    try:
        conflicts = detect_conflicts(query, evidence, llm)
    except LLMUnavailable as exc:
        logger.warning("Conflict check unavailable: %s", exc)
        conflicts_checked = False

    confidence = calculate_overall_confidence(evidence, conflicts, conflicts_checked)

    # 5. Final answer ------------------------------------------------------
    status = "SUCCESS"
    try:
        answer = generate_trust_aware_answer(query, evidence, llm, conflicts)
    except LLMUnavailable as exc:
        logger.warning("Answer generation unavailable: %s", exc)
        status = "PARTIAL"
        answer = AI_UNAVAILABLE_ANSWER

    answer = add_confidence_notice(answer, confidence, conflicts, conflicts_checked)

    return make_api_response(
        build_research_result(
            query, answer, source_label, confidence,
            sources=evidence, conflicts=conflicts, status=status,
        )
    )
