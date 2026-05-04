FROM python:3.11-slim

# Устанавливаем uv через копирование бинарника из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Указываем uv не создавать виртуальное окружение внутри контейнера
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    pillow \
    transformers \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    qdrant-client 

COPY api /app/api
COPY encoders /app/encoders
COPY store /app/store
COPY data /app/data

EXPOSE 8000

CMD ["uvicorn", "api.search_api:app", "--host", "0.0.0.0", "--port", "8000"]