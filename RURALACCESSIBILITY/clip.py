import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIP(nn.Module):
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=device):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
    
    def get_text_embeddings(self, text):
        text_inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**text_inputs)
        return text_embeddings
    
    def get_image_embeddings(self, images):
        images = self.processor(text = None,
                                images = images,
                                return_tensors="pt"
                                )["pixel_values"].to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(images)
        return image_embeddings
    
    def compute_cosine_sim(self, text, images, query_embeddings, image_embeddings, query_type="text"):
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

        cosine_sim = torch.matmul(query_embeddings, image_embeddings.T)
        max_sim_indices = torch.argmax(cosine_sim, dim=-1)
        for i, idx in enumerate(max_sim_indices):
            if query_type == "text":
                print(f"Text feature '{text[i]}' is most similar to image: {images[idx]}")
            else:
                print(f"The most similar mask to query: {images[idx]}")
        return cosine_sim