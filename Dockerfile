# Multi-agent research pipeline (FastAPI + LangGraph) container image.
#
# Build:  docker build -t research-pipeline .
# Run:    docker run -p 8000:8000 -e MOCK_MODE=true research-pipeline
# Prefer `docker compose up` — it wires up the redis + qdrant services too.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so code-only changes do not invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (filtered by .dockerignore).
COPY . .

# Run as an unprivileged user: the app binds 8000 (>1024) and only needs a
# writable HOME for the runtime fastembed model download, so root is unneeded.
RUN useradd --create-home appuser
USER appuser
ENV HOME=/home/appuser

EXPOSE 8000

# The slim image ships without curl, so probe /health with the stdlib instead.
# urlopen raises (and python exits non-zero) on connection errors and HTTP 4xx/5xx.
# The generous start period covers the first-boot fastembed model download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
