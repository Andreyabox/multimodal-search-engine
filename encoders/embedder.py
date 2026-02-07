import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def _extract_tensor(features):
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "text_embeds") and features.text_embeds is not None:
            return features.text_embeds
        if hasattr(features, "image_embeds") and features.image_embeds is not None:
            return features.image_embeds
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
            return features[0]
        raise TypeError(f"Unsupported CLIP output type: {type(features)!r}")

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        feats = self._extract_tensor(self.model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy()

    @torch.no_grad()
    def embed_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        feats = self._extract_tensor(self.model.get_image_features(**inputs))
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy()
