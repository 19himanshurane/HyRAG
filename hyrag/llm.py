"""A small Groq chat client (OpenAI-compatible API) using the same retry policy as the embedder.

Default model: openai/gpt-oss-120b (open weights, Apache-2.0), the strongest chat model on this account's
Groq plan (listed 2026-09-25). It is a reasoning model: it "thinks" before answering. The thinking comes back
in a separate `reasoning` field and never reaches the answer text; reasoning_effort="low" keeps it short
(12 reasoning tokens and 0.7 s on a small grounded question, measured).
"""
import os
import threading
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from hyrag.http import post_json

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


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
    def __init__(self, model: str = "openai/gpt-oss-120b", reasoning_effort: str = "low",
                 temperature: float = 0.0, seed: int = 7, max_completion_tokens: int = 1024):
        load_dotenv()  # read .env when a client is created, not as a side effect of importing this module
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file in the project root.")
        self.model, self.reasoning_effort = model, reasoning_effort
        # temperature 0 + a fixed seed: as repeatable as the provider allows (not guaranteed identical)
        self.temperature, self.seed = temperature, seed
        self.max_completion_tokens = max_completion_tokens  # budget for reasoning + answer together
        self.client = httpx.Client(headers={"Authorization": f"Bearer {self.api_key}"}, timeout=120)
        self.requests_made = 0
        self._count_lock = threading.Lock()

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "GroqChat":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def complete(self, messages: list[dict]) -> ChatResult:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature, "seed": self.seed,
                   "reasoning_effort": self.reasoning_effort, "max_completion_tokens": self.max_completion_tokens}
        t0 = time.perf_counter()
        data = post_json(self.client, GROQ_CHAT_URL, payload, what="Groq chat request", on_attempt=self._count)
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
