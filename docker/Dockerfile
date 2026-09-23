# SENTRIX API: full image (training dependencies included), multi-stage build.
# The deployed image is docker/Dockerfile.api.

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

# Application code (frontend/, notebooks/, tests/ excluded via .dockerignore).
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/

# Model artifacts read from disk by /chat, /metrics and /health.
COPY artifacts/ ./artifacts/

# Runtime directories the app writes to
RUN mkdir -p logs data/processed

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
