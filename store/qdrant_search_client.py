import os

from qdrant_client import QdrantClient, models


SPARSE_VECTOR_NAME = "bm25"


def _to_sparse_vector(sparse) -> models.SparseVector:
    """Accept dict({indices, values}) or already-built SparseVector."""
    if isinstance(sparse, models.SparseVector):
        return sparse
    return models.SparseVector(indices=sparse["indices"], values=sparse["values"])


class QdrantSearchClient:
    def __init__(self, host: str | None = None, port: int | None = None):
        # self.client = QdrantClient(":memory:")
        if host is None:
            host = os.environ.get("QDRANT_HOST", "localhost")
        if port is None:
            port_env = os.environ.get("QDRANT_PORT", "6333")
            try:
                port = int(port_env)
            except ValueError:
                port = 6333
        self.client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        with_bm25: bool = False,
    ):
        sparse_vectors_config = None
        if with_bm25:
            sparse_vectors_config = {
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            }
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "image": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config=sparse_vectors_config,
        )

    def upsert_text_points(self, collection_name: str, vectors, payloads):
        points = [
            models.PointStruct(
                id=idx,
                vector={"text": vector, "image": vector},
                payload=payload,
            )
            for idx, (vector, payload) in enumerate(zip(vectors, payloads), start=1)
        ]
        self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def upsert_video_points(
        self,
        collection_name: str,
        ids,
        text_vectors,
        image_vectors,
        payloads,
        sparse_vectors=None,
    ):
        if sparse_vectors is None:
            sparse_vectors = [None] * len(ids)
        points = []
        for point_id, text_vector, image_vector, payload, sparse in zip(
            ids, text_vectors, image_vectors, payloads, sparse_vectors
        ):
            vector: dict = {"text": text_vector, "image": image_vector}
            if sparse is not None:
                vector[SPARSE_VECTOR_NAME] = _to_sparse_vector(sparse)
            points.append(
                models.PointStruct(id=point_id, vector=vector, payload=payload)
            )
        self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def search_text(self, collection_name: str, query_vector, limit: int = 5):
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="text",
            limit=limit,
            with_payload=True,
        )
        return response.points

    def search(
        self,
        collection_name: str,
        query_vector,
        using: str = "text",
        limit: int = 5,
    ):
        """Generic vector search supporting named dense ('text'/'image') or sparse ('bm25') vectors."""
        if using not in {"text", "image", SPARSE_VECTOR_NAME}:
            raise ValueError(f"Unsupported vector name: {using!r}")
        if using == SPARSE_VECTOR_NAME:
            query_vector = _to_sparse_vector(query_vector)
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=using,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def search_hybrid(
        self,
        collection_name: str,
        dense_text_vector,
        sparse_query,
        limit: int = 5,
        prefetch_limit: int | None = None,
    ):
        """RRF fusion of CLIP `text` dense vector and `bm25` sparse vector."""
        if prefetch_limit is None:
            prefetch_limit = max(limit * 4, 20)
        sparse_vec = _to_sparse_vector(sparse_query)
        response = self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_text_vector,
                    using="text",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )
        return response.points
