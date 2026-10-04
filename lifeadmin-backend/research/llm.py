"""Thin Groq wrapper used by every LLM step of the research pipeline."""

import json
import logging
import os
import time

from . import config

logger = logging.getLogger("lifeadmin.research.llm")


class LLMUnavailable(Exception):
    """Raised when the LLM cannot give a usable answer (rate limit, outage,
    missing key, or malformed output). The pipeline degrades gracefully."""


_client = None
_RETRYABLE = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
}


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set")
        try:
            from groq import Groq
        except ImportError as exc:
            raise LLMUnavailable("the 'groq' package is not installed") from exc
        _client = Groq(api_key=api_key)
    return _client


def _build_kwargs(messages, temperature, max_tokens, json_mode):
    kwargs = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens or config.GROQ_MAX_COMPLETION_TOKENS,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if config.GROQ_MODEL.startswith("openai/gpt-oss") and config.GROQ_REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    return kwargs


def groq_call(
    prompt,
    *,
    system=None,
    json_mode=False,
    temperature=0.0,
    max_tokens=None,
    retries=2,
):
    """Send one prompt to Groq and return the text reply.

    Raises LLMUnavailable instead of returning magic strings, so callers can
    never mistake an error message for real content.
    """
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = _build_kwargs(messages, temperature, max_tokens, json_mode)

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if not text:
                reason = getattr(choice, "finish_reason", None)
                raise LLMUnavailable(
                    f"empty response from model (finish_reason={reason}); "
                    "raise GROQ_MAX_COMPLETION_TOKENS or lower GROQ_REASONING_EFFORT"
                )
            return text
        except LLMUnavailable as exc:
            logger.error("%s", exc)
            raise
        except TypeError as exc:
            # Older groq SDKs reject newer parameters such as reasoning_effort.
            if "reasoning_effort" in kwargs and "reasoning_effort" in str(exc):
                logger.warning("SDK does not support reasoning_effort; retrying without it "
                               "(upgrade with: pip install -U groq)")
                kwargs.pop("reasoning_effort")
                continue
            raise LLMUnavailable(f"Groq call failed: {exc}") from exc
        except Exception as exc:  # SDK error classes vary between versions
            name = type(exc).__name__
            if name in _RETRYABLE and attempt < retries:
                delay = 1.5 * (2**attempt)
                logger.warning("Groq %s, retrying in %.1fs", name, delay)
                time.sleep(delay)
                continue
            detail = str(exc)[:300]
            logger.error("Groq call failed (model=%s): %s: %s", config.GROQ_MODEL, name, detail)
            raise LLMUnavailable(f"Groq call failed: {name}: {detail}") from exc
    raise LLMUnavailable("Groq call failed")  # pragma: no cover


def parse_json_object(text):
    """Parse a JSON object out of an LLM reply (tolerates ``` fences / chatter)."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        value = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMUnavailable("model did not return JSON")
        try:
            value = json.loads(cleaned[start : end + 1])
        except ValueError as exc:
            raise LLMUnavailable("model returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LLMUnavailable("model returned JSON that is not an object")
    return value
