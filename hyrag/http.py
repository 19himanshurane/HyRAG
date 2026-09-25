"""One retry policy for every outside API (Mistral embeddings, Groq chat).

429 (rate limited) and 5xx (provider-side trouble) and dropped connections are temporary: wait and retry,
honouring the server's Retry-After when it sends one. Anything else (bad key, bad request) won't fix itself:
fail at once, with the provider's error text in the message (it says WHAT was wrong).
"""
import logging
import time
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

MAX_RETRY_WAIT_SECONDS = 60


def retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    """Honour the server's Retry-After (seconds) when it sends one; otherwise back off 1s, 2s, 4s..."""
    header = resp.headers.get("Retry-After") if resp is not None else None
    try:
        return min(float(header), MAX_RETRY_WAIT_SECONDS)
    except (TypeError, ValueError):
        return float(2**attempt)


def post_json(client: httpx.Client, url: str, payload: dict, *, what: str, attempts: int = 5,
              on_attempt: Callable[[], None] | None = None) -> dict:
    """POST `payload` and return the JSON reply, retrying temporary failures. `on_attempt` runs before
    every try (callers count requests with it)."""
    for attempt in range(attempts):
        try:
            if on_attempt:
                on_attempt()
            resp = client.post(url, json=payload)
        except httpx.TransportError as e:  # timeout, dropped connection, DNS hiccup
            resp, reason = None, type(e).__name__
        else:
            reason = f"HTTP {resp.status_code}"
        if resp is None or resp.status_code == 429 or resp.status_code >= 500:
            if attempt < attempts - 1:
                wait = retry_wait(resp, attempt)
                log.warning("%s failed (%s); retry %d/%d in %.1fs", what, reason, attempt + 1, attempts - 1, wait)
                time.sleep(wait)
            continue
        if resp.is_error:
            raise httpx.HTTPStatusError(f"{what} failed: HTTP {resp.status_code}: {resp.text[:500]}",
                                        request=resp.request, response=resp)
        return resp.json()
    raise RuntimeError(f"{what} failed after {attempts} attempts")
