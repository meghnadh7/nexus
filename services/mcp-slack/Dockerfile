FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir build==1.2.1

COPY pyproject.toml .
COPY src/ src/

RUN python -m build --wheel --outdir /dist && \
    pip install --no-cache-dir /dist/*.whl --target /install

FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/lib

COPY --from=builder /install /app/lib

RUN groupadd --gid 1001 nexus && \
    useradd --uid 1001 --gid nexus --shell /bin/bash --create-home nexus

WORKDIR /home/nexus
USER nexus

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://localhost:8002/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "mcp_slack.server:app", "--host", "0.0.0.0", "--port", "8002"]
