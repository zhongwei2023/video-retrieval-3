from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import json
import numpy as np

from .frame_io import save_frame_rgb


def crop_with_padding(rgb: np.ndarray, bbox: List[float], padding_ratio: float = 0.08) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * padding_ratio
    py = bh * padding_ratio
    x1c = max(0, int(round(x1 - px)))
    y1c = max(0, int(round(y1 - py)))
    x2c = min(w, int(round(x2 + px)))
    y2c = min(h, int(round(y2 + py)))
    if x2c <= x1c or y2c <= y1c:
        return rgb
    return rgb[y1c:y2c, x1c:x2c].copy()


def save_outputs(
    out_dir: Path,
    best_rgb: np.ndarray,
    crop_rgb: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_frame_rgb(out_dir / "best_frame.jpg", best_rgb)
    crop_path = save_frame_rgb(out_dir / "target_crop.jpg", crop_rgb)
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"best_frame": best_path, "target_crop": crop_path, "meta_json": str(meta_path)}
