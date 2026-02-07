import csv
import uvicorn

from pathlib import Path
from threading import Lock
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from encoders.embedder import CLIPEmbedder
from store.qdrant_search_client import QdrantSearchClient

app = FastAPI()
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_DIR / "web_harvested_dataset" / "train.csv"
COLLECTION_NAME = "web_harvested_images"

embedder: CLIPEmbedder | None = None
search_client = QdrantSearchClient()
index_lock = Lock()
index_ready = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")


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


def _ensure_index() -> None:
    global embedder
    global index_ready
    if index_ready:
        return

    with index_lock:
        if index_ready:
            return
        if embedder is None:
            embedder = CLIPEmbedder()

        rows = _load_dataset()
        if not rows:
            raise RuntimeError("Dataset is empty and cannot be indexed.")

        captions = [row["caption"] for row in rows]
        vectors = embedder.embed_texts(captions).tolist()
        vector_size = len(vectors[0])

        search_client.create_collection(COLLECTION_NAME, vector_size=vector_size)
        search_client.upsert_text_points(
            COLLECTION_NAME,
            vectors=vectors,
            payloads=rows,
        )
        index_ready = True


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    """Search images by text query."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="'query' must not be empty.")

    try:
        _ensure_index()
        if embedder is None:
            raise RuntimeError("Embedder is not initialized.")
        query_vector = embedder.embed_texts([query])[0].tolist()
        points = search_client.search_text(
            COLLECTION_NAME,
            query_vector=query_vector,
            limit=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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

# if __name__ == "__main__":
#     uvicorn.run(app=app, reload=True, port=8000)
