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
    transformers

COPY encoders /app/encoders

CMD ["python", "-m", "encoders.embedder"]
