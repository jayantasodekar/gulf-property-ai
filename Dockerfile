# syntax=docker/dockerfile:1
#
# Multi-stage build.
#
# Stage 1 compiles the React SPA. Stage 2 is the runtime: it never sees Node,
# never sees the scraper's browser-impersonation stack, and builds the SQLite +
# vector index at BUILD time so the container starts serving immediately rather
# than embedding ~2,800 documents on first request.

# ---------- stage 1: build the UI ----------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build


# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed

# Hugging Face Spaces runs containers as UID 1000.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /opt/models \
    && chown -R appuser:appuser /opt/models

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- expensive layer, deliberately isolated ------------------------------
# Only the three files the index build actually needs are copied here, so
# editing agent.py or retrieval.py does NOT invalidate the ~10 minute embedding
# step below. The rest of app/ is copied afterwards.
COPY app/__init__.py app/config.py app/index.py ./app/
COPY data/corpus.jsonl.gz ./data/corpus.jsonl.gz

# Builds the SQLite + FTS5 + vector index AND warms the embedding-model cache
# into the image, so cold start is fast and a missing/corrupt corpus fails the
# build loudly rather than the first request.
RUN python -m app.index --build && python -m app.index --verify

# --- cheap layers -------------------------------------------------------
COPY app/ ./app/
COPY --from=web /web/dist ./web/dist

RUN chown -R appuser:appuser /app/data /opt/models

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=8).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
