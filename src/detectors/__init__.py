from __future__ import annotations

from typing import Optional

from .base import BaseDetector
from .owlv2 import OwlV2Detector, OwlV2Config
from .grounding_dino import GroundingDINODetector, GroundingDINOConfig


DETECTOR_REGISTRY = {
    "owlv2": (OwlV2Detector, OwlV2Config),
    "grounding_dino": (GroundingDINODetector, GroundingDINOConfig),
}


def create_detector(
    name: str,
    box_threshold: float = 0.25,
    device: Optional[str] = None,
) -> BaseDetector:
    name_lower = name.lower()
    if name_lower not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector: {name!r}. Available: {list(DETECTOR_REGISTRY.keys())}"
        )
    detector_cls, config_cls = DETECTOR_REGISTRY[name_lower]
    cfg = config_cls(box_threshold=box_threshold)
    return detector_cls(cfg, device=device)


def create_detector_with_text_threshold(
    name: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.05,
    device: Optional[str] = None,
) -> BaseDetector:
    name_lower = name.lower()
    if name_lower not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {name!r}. Available: {list(DETECTOR_REGISTRY.keys())}")
    detector_cls, config_cls = DETECTOR_REGISTRY[name_lower]
    if name_lower == "grounding_dino":
        cfg = config_cls(box_threshold=box_threshold, text_threshold=text_threshold)
    else:
        cfg = config_cls(box_threshold=box_threshold)
    return detector_cls(cfg, device=device)