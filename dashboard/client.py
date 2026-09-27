"""A thin client for the HyRAG HTTP API. The dashboard uses ONLY this: it never imports the `hyrag` package, so it
runs in its own small container, and any other frontend (React, a CLI) could replace it without touching HyRAG.
"""
import time
from dataclasses import dataclass

import httpx

# The server's own per-question budget is 60 s; wait a little longer than that so the server, not the client,
# decides when a question has taken too long (and says so in a structured response).
ASK_TIMEOUT_SECONDS = 75.0
SIDE_TIMEOUT_SECONDS = 10.0


class ApiError(Exception):
    """The API could not be used: unreachable, rejected the request, or answered with an unexpected body."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class AskResult:
    body: dict              # the API's structured answer (hyrag.answer.Response as JSON)
    http_status: int
    seconds: float          # wall-clock time of the HTTP call, as the dashboard saw it


class HyragClient:
    def __init__(self, base_url: str, api_key: str = "", transport: httpx.BaseTransport | None = None):
        headers = {"X-API-Key": api_key} if api_key else {}
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, headers=headers, transport=transport)

    def close(self) -> None:
        self.http.close()

    def _get(self, path: str, **params) -> dict:
        try:
            r = self.http.get(path, params=params, timeout=SIDE_TIMEOUT_SECONDS)
        except httpx.HTTPError as e:
            raise ApiError(f"The HyRAG API at {self.base_url} is unreachable ({type(e).__name__}).") from e
        if r.status_code == 401:
            raise ApiError("The API rejected the key (401): check HYRAG_API_KEY.", 401)
        if r.status_code not in (200, 503):
            raise ApiError(f"Unexpected response from {path}: HTTP {r.status_code}", r.status_code)
        return r.json()

    def ready(self) -> dict:
        return self._get("/ready")

    def documents(self, limit: int = 1000) -> dict:
        return self._get("/v1/documents", limit=limit)

    def ask(self, question: str, mode: str = "hybrid") -> AskResult:
        t0 = time.perf_counter()
        try:
            r = self.http.post("/v1/ask", json={"question": question, "mode": mode}, timeout=ASK_TIMEOUT_SECONDS)
        except httpx.TimeoutException as e:
            raise ApiError(f"No answer within {ASK_TIMEOUT_SECONDS:.0f} s.") from e
        except httpx.HTTPError as e:
            raise ApiError(f"The HyRAG API at {self.base_url} is unreachable ({type(e).__name__}).") from e
        seconds = time.perf_counter() - t0
        if r.status_code == 422:
            detail = r.json().get("detail")
            message = detail if isinstance(detail, str) else "The question is empty or too long (2,000 characters max)."
            raise ApiError(message, 422)
        if r.status_code == 401:
            raise ApiError("The API rejected the key (401): check HYRAG_API_KEY.", 401)
        try:
            body = r.json()
        except ValueError as e:
            raise ApiError(f"Unexpected response from /v1/ask: HTTP {r.status_code}", r.status_code) from e
        if "status" not in body:  # 5xx from a proxy or crash, not HyRAG's structured answer
            raise ApiError(f"Unexpected response from /v1/ask: HTTP {r.status_code}", r.status_code)
        return AskResult(body, r.status_code, seconds)
