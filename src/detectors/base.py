from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..detection import Detection


class BaseDetector(ABC):
    """Abstract interface for open-vocabulary object detectors."""

    @abstractmethod
    def detect(self, rgb: np.ndarray, query: str) -> List[Detection]:
        """Detect objects matching query in a single image."""
        ...

    @abstractmethod
    def detect_batch(self, rgbs: List[np.ndarray], query: str) -> List[List[Detection]]:
        """Detect objects matching query in a batch of images."""
        ...
