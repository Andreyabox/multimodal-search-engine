FROM python:3.11-slim

# Устанавливаем uv через копирование бинарника из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN uv pip install --system --no-cache \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    qdrant-client \
    redis \
    httpx \
    numpy

COPY api /app/api
COPY store /app/store
COPY data /app/data
COPY encoders/__init__.py /app/encoders/__init__.py
COPY encoders/client.py /app/encoders/client.py

EXPOSE 8000

CMD ["uvicorn", "api.search_api:app", "--host", "0.0.0.0", "--port", "8000"]
