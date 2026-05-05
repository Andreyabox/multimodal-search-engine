"""HTTP client for the encoders service.

Drop-in replacement for ``CLIPEmbedder`` so callers in ``api`` and ``worker``
can switch with a single import change. Keeping a numpy-array return type
preserves the existing ``.tolist()`` / indexing call sites.
"""

import os
import httpx
import numpy as np

from io import BytesIO
from typing import Iterable


class CLIPEmbedderClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        if base_url is None:
            host = os.environ.get("EMBEDDER_HOST", "encoders")
            port = os.environ.get("EMBEDDER_PORT", "8001")
            base_url = f"http://{host}:{port}"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        payload = {"texts": list(texts)}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/embed/text", json=payload)
            response.raise_for_status()
        return np.asarray(response.json()["vectors"], dtype=np.float32)

    def embed_images(self, images) -> np.ndarray:
        files = []
        for index, image in enumerate(images):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            files.append(
                ("files", (f"image_{index}.png", buffer.getvalue(), "image/png"))
            )

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/embed/images", files=files)
            response.raise_for_status()
        return np.asarray(response.json()["vectors"], dtype=np.float32)
