from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

from .detection import Detection

try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
except Exception:
    Owlv2Processor = None  # type: ignore[assignment]
    Owlv2ForObjectDetection = None  # type: ignore[assignment]


@dataclass
class OWLConfig:
    model_id: str = "google/owlv2-base-patch16-ensemble"
    box_threshold: float = 0.25
    text_threshold: float = 0.25


class OwlV2Localizer:
    def __init__(self, cfg: OWLConfig, device: Optional[str] = None):
        if Owlv2Processor is None or Owlv2ForObjectDetection is None:
            raise ImportError("transformers with OWLv2 support is required")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(cfg.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(cfg.model_id).to(self.device).eval()
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
            text=[query],
            images=pil_images,
            return_tensors="pt",
            padding=True,
            max_length=77,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.box_threshold,
            target_sizes=target_sizes,
        )

        all_dets: List[List[Detection]] = []
        for res in results:
            dets: List[Detection] = []
            for box, score, label_id in zip(res["boxes"], res["scores"], res["labels"]):
                x1, y1, x2, y2 = [float(v) for v in box.cpu().numpy().tolist()]
                dets.append(Detection(bbox=[x1, y1, x2, y2], score=float(score.cpu().item()), label=query))
            all_dets.append(dets)
        return all_dets
