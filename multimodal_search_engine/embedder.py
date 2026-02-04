import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy()

    @torch.no_grad()
    def embed_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy()