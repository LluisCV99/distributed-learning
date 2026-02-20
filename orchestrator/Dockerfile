# Orchestrator Dockerfile (CPU-only, multi-stage build)
# Target size: ~2-2.5 GB

# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /datasets /models

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY . .

EXPOSE 11999

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "11999"]
