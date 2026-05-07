import os

from qdrant_client import QdrantClient, models


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

    def create_collection(self, collection_name: str, vector_size: int):
        self.client.recreate_collection(
            collection_name=collection_name,
            # vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            vectors_config={
                "image": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
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
    ):
        points = [
            models.PointStruct(
                id=point_id,
                vector={"text": text_vector, "image": image_vector},
                payload=payload,
            )
            for point_id, text_vector, image_vector, payload in zip(
                ids, text_vectors, image_vectors, payloads
            )
        ]
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
        """Generic vector search supporting either named vector ('text' or 'image')."""
        if using not in {"text", "image"}:
            raise ValueError(f"Unsupported vector name: {using!r}")
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=using,
            limit=limit,
            with_payload=True,
        )
        return response.points
