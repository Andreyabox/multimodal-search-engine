import os
import uuid

import uvicorn
import redis

from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from encoders.embedder import CLIPEmbedder
from store.qdrant_search_client import QdrantSearchClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "web_harvested_images"
QUEUE_NAME = "indexing_queue"
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
    return _redis


embedder: CLIPEmbedder | None = None
search_client = QdrantSearchClient()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")


@app.post("/index")
async def start_indexing() -> dict[str, str]:
    """
    Создаёт задачу на индексацию и отправляет её в очередь Redis.
    Возвращает task_id мгновенно (не дожидаясь результата).
    """
    r = _get_redis()
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    r.hset(
        f"task:{task_id}",
        mapping={
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        },
    )

    r.lpush(QUEUE_NAME, task_id)

    return {"task_id": task_id, "status": "pending"}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """Возвращает текущий статус задачи по её task_id."""
    r = _get_redis()
    data = r.hgetall(f"task:{task_id}")
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **data}


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """Возвращает список всех задач и их статусов."""
    r = _get_redis()
    keys = r.keys("task:*")
    tasks = []
    for key in keys:
        task_id = key.removeprefix("task:")
        data = r.hgetall(key)
        tasks.append({"task_id": task_id, **data})
    # Сортируем по времени создания (newest first)
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return {"tasks": tasks}


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    """Search images by text query. Index must be built first via POST /index."""
    global embedder

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="'query' must not be empty.")

    try:
        try:
            collection_info = search_client.client.get_collection(COLLECTION_NAME)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Index is not ready yet. Submit POST /index first and wait for completion.",
            )

        if embedder is None:
            embedder = CLIPEmbedder()

        query_vector = embedder.embed_texts([query])[0].tolist()
        points = search_client.search_text(
            COLLECTION_NAME,
            query_vector=query_vector,
            limit=request.top_k,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    results = []
    for point in points:
        payload = point.payload or {}
        results.append(
            {
                "image_url": payload.get("image_url"),
                "caption": payload.get("caption"),
                "score": float(point.score),
            }
        )

    return {"query": query, "results": results}


@app.get("/")
async def get_home() -> dict[str, str]:
    return {"message": "Welcome to the Multimodal Search Engine API!"}
