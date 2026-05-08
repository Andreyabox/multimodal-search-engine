from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field
from encoders.embedder import CLIPEmbedder
from encoders.sparse_embedder import BM25Embedder

_embedder: CLIPEmbedder | None = None
_sparse_embedder: BM25Embedder | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _sparse_embedder
    _embedder = CLIPEmbedder()
    _sparse_embedder = BM25Embedder()
    yield
    _embedder = None
    _sparse_embedder = None


app = FastAPI(title="CLIP Embedder", lifespan=lifespan)


class TextEmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]


class SparseQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SparseVectorPayload(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbedResponse(BaseModel):
    vectors: list[SparseVectorPayload]


class SparseQueryResponse(BaseModel):
    vector: SparseVectorPayload


def _require_embedder() -> CLIPEmbedder:
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedder is not ready yet.")
    return _embedder


def _require_sparse_embedder() -> BM25Embedder:
    if _sparse_embedder is None:
        raise HTTPException(status_code=503, detail="Sparse embedder is not ready yet.")
    return _sparse_embedder


@app.get("/healthz")
def healthz() -> dict[str, str]:
    if _embedder is None or _sparse_embedder is None:
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


@app.post("/embed/text/sparse", response_model=SparseEmbedResponse)
def embed_text_sparse(request: TextEmbedRequest) -> SparseEmbedResponse:
    """BM25 sparse embeddings tuned for indexing (document-time tokenisation)."""
    sparse = _require_sparse_embedder()
    vectors = [SparseVectorPayload(**vec) for vec in sparse.embed_documents(request.texts)]
    return SparseEmbedResponse(vectors=vectors)


@app.post("/embed/text/sparse-query", response_model=SparseQueryResponse)
def embed_text_sparse_query(request: SparseQueryRequest) -> SparseQueryResponse:
    """BM25 sparse embedding tuned for querying (no IDF weighting client-side)."""
    sparse = _require_sparse_embedder()
    vector = SparseVectorPayload(**sparse.embed_query(request.text))
    return SparseQueryResponse(vector=vector)
