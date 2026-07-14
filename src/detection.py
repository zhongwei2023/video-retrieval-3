from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    bbox: List[float]
    score: float
    label: str
