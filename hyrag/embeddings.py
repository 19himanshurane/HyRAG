import os
import time

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root into environment variables

MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"


class MistralEmbedder:
    def __init__(self, model: str = "mistral-embed", batch_size: int = 32):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set. Add it to the .env file in the project root.")
        self.model = model
        self.batch_size = batch_size
        # One client reuses the same connection for every request instead of reconnecting each time.
        self.client = httpx.Client(headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60)

    def _request(self, batch: list[str], attempts: int = 5) -> list[list[float]]:
        for attempt in range(attempts):
            try:
                resp = self.client.post(MISTRAL_EMBEDDINGS_URL, json={"model": self.model, "input": batch})
            except httpx.TransportError:  # timeout, dropped connection, DNS hiccup
                resp = None
            # 429 = rate limited, 5xx = Mistral-side problem: both are temporary, so wait 1s, 2s, 4s... and retry.
            if resp is None or resp.status_code == 429 or resp.status_code >= 500:
                if attempt < attempts - 1:
                    time.sleep(2**attempt)
                continue
            resp.raise_for_status()  # anything else (bad key, bad input) won't fix itself: fail now
            items = sorted(resp.json()["data"], key=lambda d: d["index"])
            return [item["embedding"] for item in items]
        raise RuntimeError(f"Mistral embeddings failed after {attempts} attempts")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one vector per text, as rows of a (len(texts), dimensions) array, each scaled to length 1."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._request(texts[i : i + self.batch_size]))
        arr = np.array(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
