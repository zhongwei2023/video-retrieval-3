from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FrameSample:
    index: int
    timestamp_sec: float
    rgb: np.ndarray


def _auto_fps(duration_sec: float) -> float:
    if duration_sec <= 0:
        return 2.0
    if duration_sec <= 20 * 60:
        return 4.0
    if duration_sec <= 60 * 60:
        return 2.0
    return 1.5


def _resize_to_max_side(rgb: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if not max_side or max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return rgb
    scale = max_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_frames(
    video_path: str,
    fps: Optional[float] = None,
    sampling_mode: str = "high_density",
    max_side: Optional[int] = None,
) -> Tuple[List[FrameSample], dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

    if fps is not None:
        target_fps = float(fps)
    elif sampling_mode == "all_frames":
        target_fps = native_fps if native_fps > 0 else 25.0
    else:
        target_fps = _auto_fps(duration_sec)

    if target_fps <= 0:
        target_fps = 2.0
    frame_interval = max(1, int(round(native_fps / target_fps))) if native_fps > 0 else 1

    samples: List[FrameSample] = []
    frame_index = 0
    saved_count = 0

    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if frame_index % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _resize_to_max_side(rgb, max_side)
            timestamp_sec = frame_index / native_fps if native_fps > 0 else float(saved_count)
            samples.append(FrameSample(index=frame_index, timestamp_sec=timestamp_sec, rgb=rgb))
            saved_count += 1
        frame_index += 1

    cap.release()

    meta = {
        "video_path": video_path,
        "native_fps": native_fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "requested_fps": fps,
        "sampling_mode": sampling_mode,
        "target_fps": target_fps,
        "frame_interval": frame_interval,
        "sampled_frames": len(samples),
    }
    return samples, meta
