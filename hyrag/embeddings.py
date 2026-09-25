import hashlib
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv

from hyrag.http import post_json

log = logging.getLogger(__name__)

MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"
KNOWN_DIMENSIONS = {"mistral-embed": 1024}


class EmbeddingCache:
    """On-disk (model, text) -> vector store, so re-running the pipeline only pays for text it hasn't
    embedded before. SQLite ships with Python: no server, one file, safe to delete to start over."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # A web server calls the embedder from a pool of worker threads. SQLite connections refuse other
        # threads by default, so allow them and let one lock serialise every read and write.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found = {}
        with self.lock:
            for i in range(0, len(keys), 500):  # SQLite limits how many "?" one query may have
                part = keys[i : i + 500]
                rows = self.db.execute(
                    f"SELECT key, vector FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)
                found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                [(k, v.astype(np.float32).tobytes()) for k, v in items])
            self.db.commit()

    def close(self) -> None:
        with self.lock:
            self.db.close()


class MistralEmbedder:
    def __init__(self, model: str = "mistral-embed", batch_size: int = 32,
                 cache_path: Path | None = Path("data/embeddings.sqlite")):
        load_dotenv()  # read .env when an embedder is created, not as a side effect of importing this module
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set. Add it to the .env file in the project root.")
        self.model = model
        self.batch_size = batch_size
        self.dim = KNOWN_DIMENSIONS.get(model, 0)  # corrected from the first real response
        # One client reuses the same connection for every request instead of reconnecting each time.
        self.client = httpx.Client(headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60)
        self.cache = EmbeddingCache(cache_path) if cache_path else None
        self.requests_made = 0
        self._count_lock = threading.Lock()  # `+= 1` is not atomic: concurrent searches would lose counts

    def close(self) -> None:
        self.client.close()
        if self.cache:
            self.cache.close()

    def __enter__(self) -> "MistralEmbedder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _request(self, batch: list[str], attempts: int = 5) -> list[list[float]]:
        data = post_json(self.client, MISTRAL_EMBEDDINGS_URL, {"model": self.model, "input": batch},
                         what="embedding request", attempts=attempts, on_attempt=self._count_request)
        return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]

    def _count_request(self) -> None:
        with self._count_lock:
            self.requests_made += 1

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one vector per text, as rows of a (len(texts), dimensions) array, each scaled to length 1.
        Texts already in the cache (or repeated within this call) are not sent to the API again."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        keys = [EmbeddingCache.key(self.model, t) for t in texts]
        vectors: dict[str, np.ndarray] = self.cache.get_many(list(set(keys))) if self.cache else {}
        todo = list({k: t for k, t in zip(keys, texts) if k not in vectors}.items())  # unique, in order
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i : i + self.batch_size]
            arr = np.array(self._request([t for _, t in batch]), dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            self.dim = arr.shape[1]
            fresh = list(zip((k for k, _ in batch), arr))
            vectors.update(fresh)
            if self.cache:
                self.cache.put_many(fresh)
        return np.stack([vectors[k] for k in keys])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
