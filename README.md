# Система мультимодального анализа и поиска изображений по пользовательским запросам.

## Что делает?
- Создает мультимодальные эмбеддинги для изображений;
- Осуществляет поиск изображений по текстовому запросу с использованием векторной базы данных Qdrant;

## Архитектура:
1. `encoders` - отдельный FastAPI-сервис, грузит CLIP один раз и отдаёт эмбеддинги по HTTP;
2. `api` и `worker` - тонкие сервисы без torch/transformers, ходят в `encoders` через HTTP-клиент;
3. `store` - Qdrant, хранение векторов и поиск по ним;
4. `redis` - очередь задач индексации и хранение их статусов;
5. `ui` + `nginx` - веб-интерфейс и обратный прокси.

## Подготовка датасета (MultiVENT 2.0)

Проект использует [hltcoe/MultiVENT2.0](https://huggingface.co/datasets/hltcoe/MultiVENT2.0) —
gated‑датасет с Hugging Face. В git он **не хранится** (1.93 TB полностью), его нужно
скачать локально один раз перед запуском.

1. Зарегистрируйтесь на huggingface.co и **примите условия** на странице датасета
   ([ссылка](https://huggingface.co/datasets/hltcoe/MultiVENT2.0)) — без этого даже с
   токеном HF вернёт 403.
2. Получите read‑only токен: https://huggingface.co/settings/tokens.
3. Скопируйте `.env.example` в `.env` и подставьте `HF_TOKEN=hf_...`.
4. Установите зависимости и скачайте подвыборку или полный датасет:

   ```bash
   uv sync
   # подвыборка для разработки/демо (~600 МБ)
   uv run python scripts/download_dataset.py --subset 000724
   # после скачивания распоковать архив
   uv run python scripts/download_dataset.py --subset 000724 --extract
   # если нужно удалить архивы
   uv run python scripts/download_dataset.py --subset 000724 --extract --delete-archives
   # скачать полный датасет (~1.93 TB, долго)
   uv run python scripts/download_dataset.py --full
   ```

   Файлы появятся в `data/multi_vent_2/`. Этот путь монтируется в контейнеры
   `api` и `worker` через `./data:/app/data:ro`.

## Как запустить проект при помощи docker-compose.yml:
1. Убедитесь, что выполнили шаги выше (скачали хотя бы одну подвыборку датасета и
   создали `.env` с `HF_TOKEN`).
2. `docker compose up --build`
3. При повторных запусках `docker compose down && docker compose up`

## Куда смотреть после запуска:
1. Зайти на: http://localhost
2. Нажать на кнопку "Найти"
3. Если все прошло успешно, то появятся результаты поиска
4. Qdrant: http://localhost:6333/dashboard#/collections
