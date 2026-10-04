"""Lazily-built shared resources (vector store + Tavily client).

main.py (FastAPI) and the CrewAI tool both call get_runtime(), so the
embedding model and Chroma client are created once per process.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("lifeadmin.research.runtime")


@dataclass
class Runtime:
    collection: Optional[Any] = None
    tavily_client: Optional[Any] = None
    kb_error: Optional[str] = None
    web_error: Optional[str] = None


_runtime = None
_lock = threading.Lock()


def get_runtime():
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = _build_runtime()
        return _runtime


def _build_runtime():
    runtime = Runtime()

    try:
        from .rag import create_chroma_collection

        _, runtime.collection = create_chroma_collection()
    except Exception as exc:
        runtime.kb_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Knowledge base unavailable: %s", runtime.kb_error)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        runtime.web_error = "TAVILY_API_KEY is not set"
    else:
        try:
            from .web_research import create_tavily_client

            runtime.tavily_client = create_tavily_client(api_key)
        except Exception as exc:
            runtime.web_error = f"{type(exc).__name__}: {exc}"
    if runtime.web_error:
        logger.warning("Web research unavailable: %s", runtime.web_error)

    return runtime
