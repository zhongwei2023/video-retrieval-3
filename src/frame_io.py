from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_frame_rgb(path: str | Path, rgb: np.ndarray) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)
    return str(p)


def save_image_rgb(path: str | Path, rgb: np.ndarray) -> str:
    return save_frame_rgb(path, rgb)
