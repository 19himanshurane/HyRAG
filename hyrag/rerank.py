"""Cross-encoder reranking: score each (question, chunk) pair by reading them TOGETHER.

The searches before this step never see the question and the chunk side by side. Meaning search compares two
vectors made separately (a "bi-encoder"); BM25 counts shared words. A cross-encoder feeds "question [SEP]
chunk" through one transformer, so every question word can attend to every chunk word. That is how it can
tell "ERR_TUNNEL_4012" from "ERR_TUNNEL_4013" where a single vector blurs them. The price: one model pass per
pair, so it only runs on the ~20 fused candidates, never on the whole corpus. Measured on this project's
laptop CPU (6 threads): 31 ms for one pair, ~410-550 ms for 20; about 13 s to import PyTorch and load the model.

Model: cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0 model card; 22M parameters, 88 MB, CPU is enough).
Trained on MS MARCO (real Bing questions + answer passages). MS MARCO's own terms are non-commercial: the
licence on the weights says Apache-2.0, but check with legal before shipping it in a commercial product.

The model is fetched ONCE, explicitly (`python -m hyrag.rerank`, or a Docker build step), and loaded from
the local cache at runtime. Letting the load reach the Hub costs 250 s when the Hub is unreachable, even
with the model cached (measured, docs/phase2-audit.md round 2).
"""
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Pinned: a new upload of the model would otherwise change every score (and every eval number) silently.
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
RETRY_LOAD_AFTER_SECONDS = 60  # after a failed load, fail fast instead of re-trying it on every request


class RerankerUnavailable(RuntimeError):
    """The model couldn't be loaded (not downloaded, corrupt, out of memory)."""


def is_downloaded(model: str = MODEL, revision: str = REVISION) -> bool:
    """True if the pinned model's weights are in the local cache (no network call)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model, "model.safetensors", revision=revision), str)


def download(model: str = MODEL, revision: str = REVISION) -> str:
    """Fetch the pinned model into the local Hugging Face cache (the one step that uses the network)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)


class CrossEncoderScorer:
    """Relevance of each text to a query: the model's raw score (a logit, measured roughly -11..+9 on our
    corpus). Higher = more relevant; it is NOT a probability, and 0 is not a calibrated cut-off.

    Loaded on first use: importing this module doesn't import PyTorch, and an app that never reranks never
    pays the model load (~13 s here, mostly importing PyTorch: a server should warm it up at startup)."""

    def __init__(self, model: str = MODEL, revision: str | None = REVISION, device: str = "cpu",
                 batch_size: int = 32, local_files_only: bool = True):
        self.model_name, self.revision, self.device = model, revision, device
        self.batch_size = batch_size
        self.local_files_only = local_files_only  # never touch the network at request time (see download())
        self._model = None
        self._failure: tuple[float, Exception] | None = None  # (when, why) the last load failed
        self._load_lock = threading.Lock()
        # One inference at a time. PyTorch already spreads one batch over the CPU cores, so parallel calls
        # gain little (8 concurrent requests: 3.21 s locked vs 2.85 s unlocked, same scores), and Hugging Face
        # fast tokenizers can fail with "Already borrowed" when shared across threads. The lock waits on CPU
        # work only, never on network I/O.
        self._predict_lock = threading.Lock()

    @property
    def model(self):
        with self._load_lock:  # two first requests at once must not load the model twice
            if self._model is None:
                if self._failure and time.monotonic() - self._failure[0] < RETRY_LOAD_AFTER_SECONDS:
                    raise RerankerUnavailable(f"reranker failed to load recently: {self._failure[1]}")
                try:
                    self._model = self._load()
                except Exception as e:
                    self._failure = (time.monotonic(), e)
                    raise RerankerUnavailable(
                        f"could not load {self.model_name}@{(self.revision or 'latest')[:8]} "
                        f"(local_files_only={self.local_files_only}): {type(e).__name__}. "
                        f"Download it once with: python -m hyrag.rerank") from e
                self._failure = None
                log.info("loaded reranker %s@%s (max %d tokens per pair)",
                         self.model_name, (self.revision or "latest")[:8], self._model.max_seq_length)
            return self._model

    def _load(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, revision=self.revision, device=self.device,
                            local_files_only=self.local_files_only)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty(0, dtype=np.float32)
        model = self.model
        with self._predict_lock:
            scores = model.predict([(query, t) for t in texts], batch_size=self.batch_size,
                                   show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(scores, dtype=np.float32)


if __name__ == "__main__":
    print(f"downloading {MODEL}@{REVISION[:8]} ...")
    print(f"cached at {download()}")
