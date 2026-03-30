# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install deps in a separate layer so they get cached between code changes
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to compile some wheels (numpy, scipy, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix we can copy cleanly to the runtime stage
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

COPY . .


# Non-root user — Railway best practice
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Railway injects PORT at runtime; default to 8000 locally
ENV PORT=8000

EXPOSE $PORT

# Uvicorn with Railway's injected PORT
CMD ["sh", "-c", "ls -la && ls -la data/ && python train.py --model rf --data data/ai4i2020.csv --save rf && uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]