# HyRAG dashboard image: Streamlit + an HTTP client, nothing else (no hyrag package, no PyTorch).
# Build context is the repo root (it needs the lock file): docker compose build dashboard
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN useradd --create-home --uid 10001 dashboard

COPY dashboard/requirements.txt requirements-lock.txt ./
RUN pip install -r requirements.txt -c requirements-lock.txt

COPY dashboard/app.py dashboard/client.py dashboard/safety.py ./

USER dashboard
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
