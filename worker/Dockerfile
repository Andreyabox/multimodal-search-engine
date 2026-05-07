FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --no-cache \
    qdrant-client \
    redis \
    httpx \
    numpy \
    pillow \
    opencv-python-headless

COPY worker /app/worker
COPY store /app/store
COPY data /app/data
COPY encoders/__init__.py /app/encoders/__init__.py
COPY encoders/client.py /app/encoders/client.py

CMD ["python", "-m", "worker.consumer"]
