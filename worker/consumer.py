import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import redis

from encoders.embedder import CLIPEmbedder
from store.qdrant_search_client import QdrantSearchClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
QUEUE_NAME = "indexing_queue"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_DIR / "web_harvested_dataset" / "train.csv"
COLLECTION_NAME = "web_harvested_images"
BATCH_SIZE = 32

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Received signal %s — shutting down after current task…", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def _redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _update_task_status(
    r: redis.Redis,
    task_id: str,
    status: str,
    *,
    error: str | None = None,
    progress: str | None = None,
):
    now = datetime.now(timezone.utc).isoformat()
    mapping: dict[str, str] = {
        "status": status,
        "updated_at": now,
    }
    if error is not None:
        mapping["error"] = error
    if progress is not None:
        mapping["progress"] = progress
    r.hset(f"task:{task_id}", mapping=mapping)

def _load_dataset() -> list[dict[str, str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file was not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            image_url = (row.get("image_url") or "").strip()
            caption = (row.get("caption") or "").strip()
            if image_url and caption:
                rows.append({"image_url": image_url, "caption": caption})
    return rows


def _run_indexing(r: redis.Redis, task_id: str) -> None:
    """Load CLIP model, embed dataset, upsert into Qdrant."""
    _update_task_status(r, task_id, "running", progress="Loading CLIP model…")
    logger.info("[%s] Loading CLIP model…", task_id)
    embedder = CLIPEmbedder()

    _update_task_status(r, task_id, "running", progress="Loading dataset…")
    logger.info("[%s] Loading dataset…", task_id)
    rows = _load_dataset()
    if not rows:
        raise RuntimeError("Dataset is empty and cannot be indexed.")

    _update_task_status(
        r, task_id, "running", progress=f"Embedding {len(rows)} captions…"
    )
    logger.info("[%s] Embedding %d captions…", task_id, len(rows))
    captions = [row["caption"] for row in rows]
    vectors = embedder.embed_texts(captions).tolist()
    vector_size = len(vectors[0])

    _update_task_status(r, task_id, "running", progress="Upserting into Qdrant…")
    logger.info("[%s] Upserting into Qdrant…", task_id)
    search_client = QdrantSearchClient()
    search_client.create_collection(COLLECTION_NAME, vector_size=vector_size)
    search_client.upsert_text_points(COLLECTION_NAME, vectors=vectors, payloads=rows)

    _update_task_status(r, task_id, "completed", progress="Done")
    logger.info("[%s] Indexing completed!", task_id)


def main() -> None:
    logger.info(
        "Worker starting — redis=%s:%s queue=%s",
        REDIS_HOST, REDIS_PORT, QUEUE_NAME,
    )
    r = _redis_client()

    while not _shutdown:
        try:
            r.ping()
            break
        except redis.ConnectionError:
            logger.warning("Redis not reachable yet, retrying in 2 s…")
            time.sleep(2)

    logger.info("Connected to Redis. Waiting for tasks…")

    while not _shutdown:
        result = r.brpop(QUEUE_NAME, timeout=1)
        if result is None:
            continue

        _, task_id = result
        logger.info("Picked up task %s", task_id)

        try:
            _run_indexing(r, task_id)
        except Exception as exc:
            logger.exception("Task %s failed: %s", task_id, exc)
            _update_task_status(r, task_id, "failed", error=str(exc))

    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    main()
