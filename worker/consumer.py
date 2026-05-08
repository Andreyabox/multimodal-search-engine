import csv
import json
import logging
import os
import signal
import sys
import time
import uuid

import cv2
import numpy as np
import redis

from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from encoders.client import CLIPEmbedderClient
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

VIDEOS_DATASET_DIR = DATA_DIR / "multi_vent_2" / "train"
VIDEOS_COLLECTION_NAME = "multi_vent_videos"
VIDEO_NAMESPACE = uuid.UUID("4e7e2a1f-5b9b-4c0e-9a3a-8a1d2c5f7e10")
FRAMES_PER_VIDEO = 4
FRAME_RESIZE = 224

BATCH_SIZE = 32
SERVICE_RETRY_DELAY_SECONDS = 2

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Received signal %s — shutting down after current task…", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def _redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _wait_for_qdrant() -> None:
    while not _shutdown:
        try:
            QdrantSearchClient().client.get_collections()
            return
        except Exception as exc:
            logger.warning(
                "Qdrant not reachable yet, retrying in %s s: %s",
                SERVICE_RETRY_DELAY_SECONDS,
                exc,
            )
            time.sleep(SERVICE_RETRY_DELAY_SECONDS)


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
    _update_task_status(r, task_id, "running", progress="Connecting to embedder…")
    logger.info("[%s] Connecting to embedder service…", task_id)
    embedder = CLIPEmbedderClient()

    _update_task_status(r, task_id, "running", progress="Loading dataset…")
    logger.info("[%s] Loading dataset…", task_id)
    rows = _load_dataset()
    if not rows:
        raise RuntimeError("Dataset is empty and cannot be indexed.")

    _update_task_status(
        r, task_id, "running", progress=f"Embedding {len(rows)} captions…"
    )
    logger.info("[%s] Embedding %d captions…", task_id, len(rows))
    vectors = []
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]
        captions = [row["caption"] for row in batch]
        vectors.extend(embedder.embed_texts(captions).tolist())
        _update_task_status(
            r,
            task_id,
            "running",
            progress=f"Embedding captions {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)}...",
        )
    vector_size = len(vectors[0])

    _update_task_status(r, task_id, "running", progress="Upserting into Qdrant…")
    logger.info("[%s] Upserting into Qdrant…", task_id)
    search_client = QdrantSearchClient()
    search_client.create_collection(COLLECTION_NAME, vector_size=vector_size)
    search_client.upsert_text_points(COLLECTION_NAME, vectors=vectors, payloads=rows)

    _update_task_status(r, task_id, "completed", progress="Done")
    logger.info("[%s] Indexing completed!", task_id)


def _enumerate_videos() -> list[dict[str, str | Path]]:
    """Walk MultiVENT 2.0 dataset and pair every JSON metadata with its sibling .mp4."""
    if not VIDEOS_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Video dataset directory was not found: {VIDEOS_DATASET_DIR}"
        )

    items: list[dict[str, str | Path]] = []
    for json_path in sorted(VIDEOS_DATASET_DIR.rglob("*.json")):
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue
        shard = json_path.parent.name
        items.append(
            {
                "json_path": json_path,
                "mp4_path": mp4_path,
                "shard": shard,
                "video_id": json_path.stem,
            }
        )
    return items


def _read_video_metadata(json_path: Path) -> dict[str, str | int | None]:
    """Extract searchable + display fields from MultiVENT JSON sidecar."""
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    yt_info = (data.get("yt_meta_dict") or {}).get("info") or {}
    return {
        "caption": (data.get("caption") or "").strip(),
        "title": (yt_info.get("title") or "").strip(),
        "youtube_url": (yt_info.get("webpage_url") or data.get("url") or "").strip(),
        "thumbnail_url": (yt_info.get("thumbnail") or "").strip(),
        "category": (data.get("category") or "").strip(),
        "duration": yt_info.get("duration"),
        "lang": (data.get("lang") or "").strip(),
    }


def _extract_frames(mp4_path: Path, n: int = FRAMES_PER_VIDEO) -> list[Image.Image]:
    """Pick `n` evenly-spaced frames from an mp4 and return as PIL images (RGB, square)."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        cap.release()
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            ok, bgr = cap.read()
            if not ok:
                return []
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return [Image.fromarray(rgb).resize((FRAME_RESIZE, FRAME_RESIZE))]

        indices = [max(0, min(total - 1, int(total * (i + 0.5) / n))) for i in range(n)]
        frames: list[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).resize((FRAME_RESIZE, FRAME_RESIZE)))
        return frames
    finally:
        cap.release()


def _video_point_id(shard: str, video_id: str) -> str:
    return str(uuid.uuid5(VIDEO_NAMESPACE, f"{shard}/{video_id}"))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _run_video_indexing(r: redis.Redis, task_id: str) -> None:
    """Index MultiVENT 2.0 videos with hybrid (caption text + mean-pooled frame image) vectors."""
    _update_task_status(r, task_id, "running", progress="Connecting to embedder…")
    logger.info("[%s] Connecting to embedder service…", task_id)
    embedder = CLIPEmbedderClient()

    _update_task_status(r, task_id, "running", progress="Scanning video dataset…")
    logger.info("[%s] Scanning %s…", task_id, VIDEOS_DATASET_DIR)
    items = _enumerate_videos()
    if not items:
        raise RuntimeError(
            f"No (json, mp4) pairs were found under {VIDEOS_DATASET_DIR}."
        )

    total = len(items)
    logger.info("[%s] Found %d videos", task_id, total)

    ids: list[str] = []
    text_vectors: list[list[float]] = []
    image_vectors: list[list[float]] = []
    sparse_vectors: list[dict] = []
    payloads: list[dict] = []

    for processed, item in enumerate(items, start=1):
        if _shutdown:
            raise RuntimeError("Shutdown requested before indexing finished.")

        shard = str(item["shard"])
        video_id = str(item["video_id"])
        json_path = item["json_path"]  # type: ignore[assignment]
        mp4_path = item["mp4_path"]  # type: ignore[assignment]

        try:
            meta = _read_video_metadata(json_path)
        except Exception as exc:
            logger.warning("[%s] Skipping %s: bad JSON (%s)", task_id, json_path, exc)
            continue

        text_for_embed = ". ".join(part for part in (meta["caption"], meta["title"]) if part)
        if not text_for_embed:
            text_for_embed = video_id

        try:
            frames = _extract_frames(mp4_path)
        except Exception as exc:
            logger.warning("[%s] Frame extraction failed for %s: %s", task_id, mp4_path, exc)
            frames = []
        if not frames:
            logger.warning("[%s] No frames decoded for %s, skipping", task_id, mp4_path)
            continue

        try:
            text_vec = embedder.embed_texts([text_for_embed])[0]
            frame_vecs = embedder.embed_images(frames)
            sparse_vec = embedder.embed_texts_sparse([text_for_embed])[0]
        except Exception as exc:
            logger.warning("[%s] Embedding failed for %s: %s", task_id, video_id, exc)
            continue

        image_vec = _normalize(frame_vecs.mean(axis=0))

        ids.append(_video_point_id(shard, video_id))
        text_vectors.append(text_vec.astype(np.float32).tolist())
        image_vectors.append(image_vec.astype(np.float32).tolist())
        sparse_vectors.append(sparse_vec)
        payloads.append(
            {
                "video_id": video_id,
                "shard": shard,
                "title": meta["title"],
                "caption": meta["caption"],
                "youtube_url": meta["youtube_url"],
                "thumbnail_url": meta["thumbnail_url"],
                "category": meta["category"],
                "duration": meta["duration"],
                "lang": meta["lang"],
                "video_url": f"/videos/{shard}/{video_id}.mp4",
            }
        )

        if processed % 5 == 0 or processed == total:
            _update_task_status(
                r,
                task_id,
                "running",
                progress=f"Embedded {processed}/{total} videos…",
            )
            logger.info("[%s] Progress: %d/%d", task_id, processed, total)

    if not ids:
        raise RuntimeError("No videos produced embeddings; nothing to upsert.")

    vector_size = len(text_vectors[0])

    _update_task_status(r, task_id, "running", progress="Upserting into Qdrant…")
    logger.info("[%s] Upserting %d points into Qdrant…", task_id, len(ids))
    search_client = QdrantSearchClient()
    search_client.create_collection(
        VIDEOS_COLLECTION_NAME,
        vector_size=vector_size,
        with_bm25=True,
    )
    search_client.upsert_video_points(
        VIDEOS_COLLECTION_NAME,
        ids=ids,
        text_vectors=text_vectors,
        image_vectors=image_vectors,
        payloads=payloads,
        sparse_vectors=sparse_vectors,
    )

    _update_task_status(r, task_id, "completed", progress=f"Indexed {len(ids)} videos")
    logger.info("[%s] Video indexing completed!", task_id)


def _dispatch_task(r: redis.Redis, task_id: str) -> None:
    task_type = (r.hget(f"task:{task_id}", "task_type") or "images").strip()
    if task_type == "videos":
        _run_video_indexing(r, task_id)
    else:
        _run_indexing(r, task_id)


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
            logger.warning(
                "Redis not reachable yet, retrying in %s s...",
                SERVICE_RETRY_DELAY_SECONDS,
            )
            time.sleep(SERVICE_RETRY_DELAY_SECONDS)

    logger.info("Connected to Redis. Waiting for Qdrant...")
    _wait_for_qdrant()

    logger.info("Connected to Redis and Qdrant. Waiting for tasks...")

    while not _shutdown:
        result = r.brpop(QUEUE_NAME, timeout=1)
        if result is None:
            continue

        _, task_id = result
        logger.info("Picked up task %s", task_id)

        try:
            _dispatch_task(r, task_id)
        except Exception as exc:
            logger.exception("Task %s failed: %s", task_id, exc)
            _update_task_status(r, task_id, "failed", error=str(exc))

    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    main()
