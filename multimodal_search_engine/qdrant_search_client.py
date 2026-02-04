from qdrant_client import QdrantClient, models


class QdrantSearchClient:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(":memory:")
        # self.client = QdrantClient(host=host, port=port)

    def create_collection(self, collection_name: str, vector_size: int):
        self.client.recreate_collection(
            collection_name=collection_name,
            # vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            vectors_config={
                "image": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                "text": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
        )