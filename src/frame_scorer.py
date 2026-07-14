from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np

from .detection import Detection


def laplacian_sharpness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def bbox_area_ratio(bbox: List[float], image_hw: tuple[int, int]) -> float:
    h, w = image_hw
    image_area = max(1.0, float(h * w))
    x1, y1, x2, y2 = bbox
    area = max(0.0, (x2 - x1) * (y2 - y1))
    return area / image_area


def score_detection(det: Detection, rgb: np.ndarray, w: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
    w1, w2, w3 = w
    area_ratio = bbox_area_ratio(det.score and det.bbox, rgb.shape[:2])
    sharpness = laplacian_sharpness(rgb)
    # normalize sharpness softly
    sharpness_norm = math.log1p(sharpness) / 10.0
    return w1 * det.score + w2 * min(1.0, area_ratio * 5.0) + w3 * min(1.0, sharpness_norm)
