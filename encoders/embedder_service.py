"""HTTP service exposing CLIP text/image embeddings.

The model is loaded once during FastAPI's lifespan startup and shared between
all incoming requests. Once /healthz returns 200 the model is guaranteed to be
ready, which lets dependent services rely on Compose healthchecks for ordering.
"""

from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from encoders.embedder import CLIPEmbedder

_embedder: CLIPEmbedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder
    _embedder = CLIPEmbedder()
    yield
    _embedder = None


app = FastAPI(title="CLIP Embedder", lifespan=lifespan)


class TextEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


def _require_embedder() -> CLIPEmbedder:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedder is not ready yet.")
    return _embedder


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    return {"status": "ok"}


@app.post("/embed/text", response_model=EmbedResponse)
def embed_text(request: TextEmbedRequest) -> EmbedResponse:
    embedder = _require_embedder()
    vectors = embedder.embed_texts(request.texts).tolist()
    return EmbedResponse(vectors=vectors)


@app.post("/embed/images", response_model=EmbedResponse)
async def embed_images_endpoint(
    files: list[UploadFile] = File(...),
) -> EmbedResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    embedder = _require_embedder()

    images = []
    for upload in files:
        try:
            data = await upload.read()
            images.append(Image.open(BytesIO(data)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode image '{upload.filename}': {exc}",
            ) from exc

    vectors = embedder.embed_images(images).tolist()
    return EmbedResponse(vectors=vectors)
