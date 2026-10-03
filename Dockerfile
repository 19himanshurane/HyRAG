# HyRAG API image (also runs the seed job). Build: docker compose build
# Python matches the one the lock file was tested with.
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN useradd --create-home --uid 10001 hyrag

# Dependencies first (cached until the requirements change). PyTorch comes from its CPU-only index: the
# PyPI wheel for Linux bundles CUDA (several GB) that this CPU-only reranker never uses. Everything else is
# the runtime part of requirements.txt (the dev section is cut), pinned to the tested versions of the lock.
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
        "torch==$(sed -n 's/^torch==//p' requirements-lock.txt)" \
 && sed '/^# dev/,$d' requirements.txt > /tmp/runtime.txt \
 && pip install -r /tmp/runtime.txt -c requirements-lock.txt \
 && python -c "import torch; assert not torch.cuda.is_available() and '+cpu' in torch.__version__, torch.__version__"

# The pinned reranker model goes into the image: the API loads it offline and never downloads at startup.
COPY hyrag ./hyrag
RUN python -m hyrag.rerank && chown -R hyrag /opt/hf

# The documents the seed job indexes. Data (chunk table, parsed docs, embedding cache) lives in a volume.
COPY corpus ./corpus
COPY sample_docs ./sample_docs
RUN mkdir -p /app/data && chown hyrag /app/data

USER hyrag
EXPOSE 8000
# One worker: the answer cache and the ingest lock are in memory (see docs/deployment.md).
CMD ["uvicorn", "hyrag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
