FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache

WORKDIR /app

RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    pillow \
    transformers \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    fastembed

RUN mkdir -p /opt/hf-cache

COPY encoders /app/encoders

EXPOSE 8001

CMD ["uvicorn", "encoders.embedder_service:app", "--host", "0.0.0.0", "--port", "8001"]
