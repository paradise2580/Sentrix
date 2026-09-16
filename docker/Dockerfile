# SENTRIX API — production container image.
# Multi-stage build keeps the final image lean: dependencies compile in the
# builder stage, only the installed packages + app code ship in the runtime
# stage.

# ---------- Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- Runtime ----------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code — src/, api/, config/ are what the API needs to run.
# frontend/, notebooks/, tests/ are intentionally excluded from this image
# (see .dockerignore) since the API container doesn't need them.
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/

# Model artifacts (MLflow registry, ChromaDB index, evaluation results,
# fitted preprocessor) — several endpoints (/chat, /metrics, /health) read
# these from disk directly, so they ship inside the image rather than
# being fetched at runtime. ~59MB, well within reason for a model service.
COPY artifacts/ ./artifacts/

# Runtime directories the app writes to
RUN mkdir -p logs data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
