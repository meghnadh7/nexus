# ─── Stage 1: builder ─────────────────────────────────────────────────────────
# Build context is ./services/ (set in docker-compose.yml) so both
# orchestrator/ and rag-indexer/ are accessible without path traversal.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir build==1.2.1

# Copy sources
COPY rag-indexer/ /rag-indexer/
COPY orchestrator/pyproject.toml .
COPY orchestrator/src/ src/

# Build orchestrator wheel first, then install rag-indexer + orchestrator
# in ONE pip call so the solver sees all constraints together (prevents
# httpx 0.28+ being pulled in before the <0.28 constraint is visible).
RUN python -m build --wheel --outdir /dist && \
    pip install --no-cache-dir /rag-indexer /dist/*.whl --target /install

# ─── Stage 2: production ──────────────────────────────────────────────────────
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/lib

# Install system dependencies for asyncpg / psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /app/lib

# Create non-root user
RUN groupadd --gid 1001 nexus && \
    useradd --uid 1001 --gid nexus --shell /bin/bash --create-home nexus

WORKDIR /home/nexus
USER nexus

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
