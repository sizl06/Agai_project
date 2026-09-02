# FastAPI backend.
#
# Multi-stage: wheels are built once in the builder and copied into a slim runtime,
# so the final image carries no compiler toolchain.
#
# The image deliberately does NOT bake in data or a trained model. Those are mounted
# at runtime (see docker-compose.yml), because a model is an artifact with its own
# lifecycle - baking one in means rebuilding the image to retrain, and makes it
# impossible to tell which model a running container is serving.

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# The backend installs requirements-api.txt, not requirements.txt - the latter is the
# slim dashboard set (streamlit/pandas/plotly/requests) that Streamlit Cloud consumes.
COPY requirements-api.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-api.txt


FROM python:3.12-slim

# PYTHONUNBUFFERED so logs reach docker logs immediately rather than sitting in a buffer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-api.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

COPY app/ ./app/
COPY pipelines/ ./pipelines/
COPY scripts/ ./scripts/

# Run as a non-root user.
RUN useradd --create-home --uid 1000 hrai \
    && mkdir -p /app/data /app/models /app/logs /app/reports \
    && chown -R hrai:hrai /app
USER hrai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
