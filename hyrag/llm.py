"""A small Groq chat client, with the same retry policy as the embedder.

The default model, openai/gpt-oss-120b, was the strongest one on this Groq plan. It reasons before answering;
the reasoning comes back in a separate field, and reasoning_effort="low" keeps it short.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from hyrag.http import post_json

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
JSON_REGENERATIONS = 2  # extra tries when the model's output fails the requested JSON schema

log = logging.getLogger(__name__)


def _is_json_generation_failure(e: httpx.HTTPStatusError) -> bool:
    body = e.response.text if e.response is not None else ""
    return e.response is not None and e.response.status_code == 400 and (
        "json_validate_failed" in body or "Failed to generate" in body)


@dataclass
class ChatResult:
    text: str
    model: str
    finish_reason: str          # "stop" = finished; "length" = cut off at max_completion_tokens
    prompt_tokens: int
    completion_tokens: int      # includes reasoning tokens
    reasoning_tokens: int
    seconds: float


class GroqChat:
    def __init__(self, model: str = "openai/gpt-oss-120b", reasoning_effort: str | None = "low",
                 temperature: float = 0.0, seed: int = 7, max_completion_tokens: int = 1024,
                 timeout: float = 30.0, attempts: int = 5):
        load_dotenv()  # read .env when a client is created, not as a side effect of importing this module
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file in the project root.")
        self.model, self.reasoning_effort = model, reasoning_effort
        # temperature 0 + a fixed seed: as repeatable as the provider allows (not guaranteed identical)
        self.temperature, self.seed = temperature, seed
        self.max_completion_tokens = max_completion_tokens  # budget for reasoning + answer together
        # 30 s per attempt (was 120 s: one call could take 10.2 min). The attempt count stays 5: rate limits
        # (429) are the common failure, and cutting to 3 turned bursts into unchecked answers (measured). What
        # bounds a question is its time budget (hyrag.http.deadline, set by hyrag.answer.ask), not the count.
        self.client = httpx.Client(headers={"Authorization": f"Bearer {self.api_key}"}, timeout=timeout)
        self.attempts = attempts
        self.requests_made = 0
        self._count_lock = threading.Lock()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "GroqChat":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def complete(self, messages: list[dict], json_schema: dict | None = None) -> ChatResult:
        """`json_schema`: force the reply to match this JSON Schema (strict structured output), e.g. a verdict."""
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "seed": self.seed,
                   "max_completion_tokens": self.max_completion_tokens}
        if self.reasoning_effort is not None:  # only reasoning models (gpt-oss) take this; None to omit it
            payload["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "reply", "strict": True, "schema": json_schema}}
        t0 = time.perf_counter()
        for regenerate in range(JSON_REGENERATIONS + 1):
            try:
                data = post_json(self.client, GROQ_CHAT_URL, payload, what="Groq chat request",
                                 attempts=self.attempts, on_attempt=self._count)
                break
            except httpx.HTTPStatusError as e:
                # A 400 normally means OUR request is wrong: fail at once. The exception: the model's output
                # failed the JSON schema ("json_validate_failed"). That is a bad sample, not a bad request; it
                # hit 1 of 8 batched judge calls once, and the identical request then passed 6 times out of 6.
                if not (json_schema is not None and _is_json_generation_failure(e)) or regenerate == JSON_REGENERATIONS:
                    raise
                log.warning("Groq returned invalid JSON for the schema; regenerating (%d/%d)",
                            regenerate + 1, JSON_REGENERATIONS)
        choice, usage = data["choices"][0], data.get("usage", {})
        return ChatResult(
            text=choice["message"].get("content") or "",
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            seconds=time.perf_counter() - t0,
        )

    def _count(self) -> None:
        with self._count_lock:
            self.requests_made += 1
