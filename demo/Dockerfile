# Demo Dockerfile (CPU-only, multi-stage build)
# Target size: ~1.5-2 GB

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

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY app.py .

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV MODEL_PATH=/models/latest.pt

EXPOSE 7860

CMD ["python", "app.py"]
