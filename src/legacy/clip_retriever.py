from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import clip
except ImportError:
    clip = None  # type: ignore[assignment]


class CLIPRetriever:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        if clip is None:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, query: str) -> np.ndarray:
        tokens = clip.tokenize([query]).to(self.device)
        vec = self.model.encode_text(tokens)
        vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def score_frames(
        self, rgbs: List[np.ndarray], text_vec: np.ndarray, batch_size: int = 32
    ) -> List[float]:
        text_tensor = torch.from_numpy(text_vec).float().to(self.device)
        scores: List[float] = []
        for i in range(0, len(rgbs), batch_size):
            batch_rgbs = rgbs[i : i + batch_size]
            pil_images = [Image.fromarray(rgb) for rgb in batch_rgbs]
            image_tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
            image_vecs = self.model.encode_image(image_tensors).float()
            image_vecs = image_vecs / image_vecs.norm(dim=-1, keepdim=True)
            batch_scores = (image_vecs @ text_tensor.T).squeeze(-1)
            scores.extend(batch_scores.cpu().tolist())
        return scores
