import os
import re
import uuid
import uvicorn
import redis

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from encoders.client import CLIPEmbedderClient
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
VIDEOS_COLLECTION_NAME = "multi_vent_videos"
QUEUE_NAME = "indexing_queue"
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

VIDEOS_BASE_DIR = (Path(__file__).resolve().parents[1] / "data" / "multi_vent_2" / "train").resolve()
_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9_\-]+$")

_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
    return _redis


embedder: CLIPEmbedderClient | None = None
search_client = QdrantSearchClient()


def _require_embedder() -> CLIPEmbedderClient:
    global embedder
    if embedder is None:
        embedder = CLIPEmbedderClient()
    return embedder


def _enqueue_task(task_type: str) -> dict[str, str]:
    r = _get_redis()
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    r.hset(
        f"task:{task_id}",
        mapping={
            "status": "pending",
            "task_type": task_type,
            "created_at": now,
            "updated_at": now,
        },
    )
    r.lpush(QUEUE_NAME, task_id)
    return {"task_id": task_id, "status": "pending", "task_type": task_type}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")


class VideoSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    mode: str = Field(
        default="text",
        pattern="^(text|image|sparse|hybrid)$",
        description=(
            "Which Qdrant vector to query: 'text'/'image' (CLIP dense), "
            "'sparse' (BM25), or 'hybrid' (CLIP text + BM25 RRF fusion)."
        ),
    )


@app.post("/index")
async def start_indexing() -> dict[str, str]:
    """
    Создаёт задачу на индексацию изображений и отправляет её в очередь Redis.
    Возвращает task_id мгновенно (не дожидаясь результата).
    """
    return _enqueue_task("images")


@app.post("/index/video")
async def start_video_indexing() -> dict[str, str]:
    """Создаёт задачу на индексацию видео из data/multi_vent_2."""
    return _enqueue_task("videos")


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
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="'query' must not be empty.")

    try:
        try:
            search_client.client.get_collection(COLLECTION_NAME)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Index is not ready yet. Submit POST /index first and wait for completion.",
            )

        query_vector = _require_embedder().embed_texts([query])[0].tolist()
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


@app.post("/search/video")
async def search_video(request: VideoSearchRequest) -> dict[str, Any]:
    """Search videos by text query. Index must be built first via POST /index/video."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="'query' must not be empty.")

    try:
        try:
            search_client.client.get_collection(VIDEOS_COLLECTION_NAME)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Video index is not ready yet. Submit POST /index/video first "
                    "and wait for completion."
                ),
            )

        embedder_client = _require_embedder()

        if request.mode == "sparse":
            sparse_query = embedder_client.embed_query_sparse(query)
            points = search_client.search(
                VIDEOS_COLLECTION_NAME,
                query_vector=sparse_query,
                using="bm25",
                limit=request.top_k,
            )
        elif request.mode == "hybrid":
            dense_vector = embedder_client.embed_texts([query])[0].tolist()
            sparse_query = embedder_client.embed_query_sparse(query)
            points = search_client.search_hybrid(
                VIDEOS_COLLECTION_NAME,
                dense_text_vector=dense_vector,
                sparse_query=sparse_query,
                limit=request.top_k,
            )
        else:
            query_vector = embedder_client.embed_texts([query])[0].tolist()
            points = search_client.search(
                VIDEOS_COLLECTION_NAME,
                query_vector=query_vector,
                using=request.mode,
                limit=request.top_k,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video search failed: {exc}") from exc

    results = []
    for point in points:
        payload = point.payload or {}
        results.append(
            {
                "video_id": payload.get("video_id"),
                "shard": payload.get("shard"),
                "title": payload.get("title"),
                "caption": payload.get("caption"),
                "youtube_url": payload.get("youtube_url"),
                "thumbnail_url": payload.get("thumbnail_url"),
                "video_url": payload.get("video_url"),
                "category": payload.get("category"),
                "duration": payload.get("duration"),
                "score": float(point.score),
            }
        )

    return {"query": query, "mode": request.mode, "results": results}


@app.get("/videos/{shard}/{video_id}.mp4")
async def stream_video(shard: str, video_id: str) -> FileResponse:
    """Стримит локальный .mp4 из data/multi_vent_2/train/<shard>/<video_id>.mp4 с поддержкой Range."""
    if not _SAFE_PATH_PART.match(shard) or not _SAFE_PATH_PART.match(video_id):
        raise HTTPException(status_code=400, detail="Invalid shard or video id.")

    candidate = (VIDEOS_BASE_DIR / shard / f"{video_id}.mp4").resolve()
    try:
        candidate.relative_to(VIDEOS_BASE_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed.")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")

    return FileResponse(
        path=str(candidate),
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
    )


@app.get("/")
async def get_home() -> dict[str, str]:
    return {"message": "Welcome to the Multimodal Search Engine API!"}
