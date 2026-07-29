from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

from .base import BaseDetector
from ..detection import Detection

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
except Exception:
    AutoProcessor = None  # type: ignore[assignment]
    AutoModelForZeroShotObjectDetection = None  # type: ignore[assignment]


@dataclass
class GroundingDINOConfig:
    model_id: str = "IDEA-Research/grounding-dino-base"
    box_threshold: float = 0.25
    text_threshold: float = 0.25


class GroundingDINODetector(BaseDetector):
    def __init__(self, cfg: GroundingDINOConfig, device: Optional[str] = None):
        if AutoProcessor is None or AutoModelForZeroShotObjectDetection is None:
            raise ImportError("transformers>=4.38 with Grounding DINO support is required")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.model_id, trust_remote_code=True).to(self.device).eval()
        self.box_threshold = cfg.box_threshold
        self.text_threshold = cfg.text_threshold

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, query: str) -> List[Detection]:
        return self.detect_batch([rgb], query)[0]

    @torch.no_grad()
    def detect_batch(self, rgbs: List[np.ndarray], query: str) -> List[List[Detection]]:
        if not rgbs:
            return []
        pil_images = [Image.fromarray(rgb) for rgb in rgbs]
        target_sizes = torch.tensor([im.size[::-1] for im in pil_images], device=self.device)

        inputs = self.processor(
            images=pil_images,
            text=query,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )

        all_dets: List[List[Detection]] = []
        for res in results:
            dets: List[Detection] = []
            for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                x1, y1, x2, y2 = [float(v) for v in box.cpu().numpy().tolist()]
                dets.append(Detection(bbox=[x1, y1, x2, y2], score=float(score.cpu().item()), label=str(label)))
            all_dets.append(dets)
        return all_dets