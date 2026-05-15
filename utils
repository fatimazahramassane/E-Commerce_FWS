"""
utils.py — Shared utilities for all agent nodes.

Centralises:
  - _parse_json_safe : strip markdown fences then parse JSON
  - with_retry       : exponential-backoff wrapper for LLM calls
  - normalize_order_id : clean raw LLM order-id output to a plain string or None
"""

import re
import json
import time
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_safe(text: str) -> dict:
    """
    Strip Markdown code fences (```json … ```) that LLMs sometimes add,
    then parse the remaining text as JSON.

    Raises json.JSONDecodeError if the text is not valid JSON after stripping.
    """
    clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(clean)


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def with_retry(fn: Callable[[], T], *, retries: int = 3, base_delay: float = 1.0) -> T:
    """
    Call *fn* up to *retries* times with exponential back-off.

    Intended for wrapping llm.invoke() calls so that transient Groq 429 / 503
    errors do not fail the entire workflow immediately.

    Args:
        fn:         Zero-argument callable that performs the LLM call.
        retries:    Maximum number of attempts (default 3).
        base_delay: Initial wait in seconds before the first retry (doubles
                    each attempt, so: 1 s, 2 s, 4 s, …).

    Returns:
        The return value of *fn* on the first successful call.

    Raises:
        The last exception raised by *fn* if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"[with_retry] attempt {attempt}/{retries} failed: {exc}. "
                    f"Retrying in {delay:.1f}s…"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[with_retry] all {retries} attempts failed. Last error: {exc}"
                )
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Order-ID normalisation
# ---------------------------------------------------------------------------

def normalize_order_id(value) -> str | None:
    """
    Coerce the raw LLM output for order_id into a plain numeric string or None.

    Handles the following LLM quirks:
      - Returned as integer:       12345    → "12345"
      - Returned as float string: "12345.0" → "12345"
      - Returned with hash:       "#12345"  → "12345"
      - Returned as null/None/nan → None
    """
    if value is None:
        return None
    s = str(value).strip().lstrip("#")
    if s.lower() in ("none", "", "nan", "null"):
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return s  # return as-is for non-numeric identifiers