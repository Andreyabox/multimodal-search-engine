"""BM25 sparse embedder built on top of FastEmbed.

The model (`Qdrant/bm25`) is unsupervised and tokenizer-only, so it loads in
under a second and produces (indices, values) pairs ready for Qdrant's
`SparseVector`. We expose two entry points to mirror FastEmbed's split between
document-time and query-time tokenisation.
"""

from typing import Iterable, TypedDict

from fastembed import SparseTextEmbedding


class SparseVector(TypedDict):
    indices: list[int]
    values: list[float]


class BM25Embedder:
    DEFAULT_MODEL = "Qdrant/bm25"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Iterable[str]) -> list[SparseVector]:
        return [
            {
                "indices": embedding.indices.tolist(),
                "values": embedding.values.tolist(),
            }
            for embedding in self._model.embed(list(texts))
        ]

    def embed_query(self, text: str) -> SparseVector:
        embedding = next(iter(self._model.query_embed([text])))
        return {
            "indices": embedding.indices.tolist(),
            "values": embedding.values.tolist(),
        }
