"""Retries and time budgets for the outside APIs (Mistral, Groq).

429s, 5xx errors and dropped connections are retried, honouring Retry-After. Anything else, like a bad key,
fails straight away with the provider's own error text.

`with deadline(60):` shares one budget between every HTTP call inside it: each attempt's timeout shrinks to
what's left, and no retry waits past the end. Without it a single Groq call could take ten minutes (5 attempts
of 120 s plus backoff), and a question makes several calls.
"""
import contextvars
import logging
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx

log = logging.getLogger(__name__)

MAX_RETRY_WAIT_SECONDS = 60

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("hyrag_deadline", default=None)


class DeadlineExceeded(TimeoutError):
    """The request's time budget ran out before this call could finish."""


class QuotaExhausted(RuntimeError):
    """A per-DAY limit (tokens or requests per day) is used up. Retrying in seconds cannot help: Groq's own
    retry hint was 7-8 minutes when the judge model's 200,000 tokens/day ran out (measured 2026-09-28)."""


_ACCOUNT_IDS = re.compile(r"\b(org|proj|user|acct)_[A-Za-z0-9]{6,}")


def provider_text(resp: httpx.Response, limit: int = 300) -> str:
    """A provider's error body, shortened and with account identifiers removed: these messages end up in logs
    and in the `reason` of API responses. (Groq's 429 text names the organization id; it was reaching clients.)"""
    return _ACCOUNT_IDS.sub(lambda m: f"{m.group(1)}_<redacted>", resp.text[:limit])


def _is_daily_limit(resp: httpx.Response) -> bool:
    text = resp.text.lower()
    return resp.status_code == 429 and ("per day" in text or "(tpd)" in text or "(rpd)" in text)


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """Budget every HTTP call inside the block to finish within `seconds` in total (nested budgets: the
    earlier end wins)."""
    end = time.monotonic() + seconds
    outer = _DEADLINE.get()
    token = _DEADLINE.set(end if outer is None else min(end, outer))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining() -> float | None:
    """Seconds left in the current budget, or None when there is no budget."""
    end = _DEADLINE.get()
    return None if end is None else end - time.monotonic()


def retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    """Honour the server's Retry-After (seconds) when it sends one. Otherwise a 429 backs off 5s, 10s, 20s, 40s
    and anything else 1s, 2s, 4s, 8s. Mistral rate-limits per minute (60 requests) and sends no Retry-After, so
    the short schedule gave up after 15 s, inside the same minute: the Docker seed failed that way."""
    header = resp.headers.get("Retry-After") if resp is not None else None
    try:
        return min(float(header), MAX_RETRY_WAIT_SECONDS)
    except (TypeError, ValueError):
        return float(5 * 2**attempt if resp is not None and resp.status_code == 429 else 2**attempt)


def post_json(client: httpx.Client, url: str, payload: dict, *, what: str, attempts: int = 5,
              on_attempt: Callable[[], None] | None = None) -> dict:
    """POST `payload` and return the JSON reply, retrying temporary failures within the current deadline.
    `on_attempt` runs before every try (callers count requests with it)."""
    for attempt in range(attempts):
        left = remaining()
        if left is not None and left <= 0:
            raise DeadlineExceeded(f"{what}: time budget used up before attempt {attempt + 1}")
        kwargs = {}
        if left is not None:  # never wait longer on one attempt than the whole request has left
            kwargs["timeout"] = min(left, client.timeout.read or left)
        try:
            if on_attempt:
                on_attempt()
            resp = client.post(url, json=payload, **kwargs)
        except httpx.TransportError as e:  # timeout, dropped connection, DNS hiccup
            resp, reason = None, type(e).__name__
        else:
            reason = f"HTTP {resp.status_code}"
        if resp is not None and _is_daily_limit(resp):
            raise QuotaExhausted(f"{what}: daily limit reached: {provider_text(resp)}")
        if resp is None or resp.status_code == 429 or resp.status_code >= 500:
            if attempt < attempts - 1:
                wait = retry_wait(resp, attempt)
                left = remaining()
                if left is not None and wait >= left:
                    raise DeadlineExceeded(f"{what} failed ({reason}); retrying in {wait:.0f}s would pass the "
                                           f"deadline ({max(left, 0):.1f}s left)")
                log.warning("%s failed (%s); retry %d/%d in %.1fs", what, reason, attempt + 1, attempts - 1, wait)
                time.sleep(wait)
            continue
        if resp.is_error:
            raise httpx.HTTPStatusError(f"{what} failed: HTTP {resp.status_code}: {provider_text(resp, 500)}",
                                        request=resp.request, response=resp)
        return resp.json()
    raise RuntimeError(f"{what} failed after {attempts} attempts")
