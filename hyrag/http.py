"""One retry policy for every outside API (Mistral embeddings, Groq chat), and one time budget per request.

429 (rate limited) and 5xx (provider-side trouble) and dropped connections are temporary: wait and retry,
honouring the server's Retry-After when it sends one. Anything else (bad key, bad request) won't fix itself:
fail at once, with the provider's error text in the message (it says WHAT was wrong).

Deadline: `with deadline(60):` gives every HTTP call made inside it (in this thread) a shared budget. Each
attempt's timeout is cut to what is left, and no retry wait sleeps past the end. Without it, one Groq call
could take 10.2 minutes (5 attempts x 120 s + backoff, measured in docs/phase3-audit.md) and a question makes
several calls in a row.
"""
import contextvars
import logging
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
    """Honour the server's Retry-After (seconds) when it sends one; otherwise back off 1s, 2s, 4s..."""
    header = resp.headers.get("Retry-After") if resp is not None else None
    try:
        return min(float(header), MAX_RETRY_WAIT_SECONDS)
    except (TypeError, ValueError):
        return float(2**attempt)


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
            raise QuotaExhausted(f"{what}: daily limit reached: {resp.text[:300]}")
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
            raise httpx.HTTPStatusError(f"{what} failed: HTTP {resp.status_code}: {resp.text[:500]}",
                                        request=resp.request, response=resp)
        return resp.json()
    raise RuntimeError(f"{what} failed after {attempts} attempts")
