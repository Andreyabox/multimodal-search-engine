FROM python:3.11-slim

# Устанавливаем uv через копирование бинарника из официального образа
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Указываем uv не создавать виртуальное окружение внутри контейнера
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN uv pip install --system --no-cache streamlit

COPY ui /app/ui

EXPOSE 8501

CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
