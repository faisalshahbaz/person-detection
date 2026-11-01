import threading
import time
from typing import Any

import cv2
import numpy as np


class HogPersonDetector:
    """OpenCV default people detector (HOG + linear SVM)."""

    def __init__(
        self,
        hit_threshold: float,
        win_stride: int,
        padding: int,
        scale: float,
        min_weight: float,
    ) -> None:
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.min_weight = min_weight
        self._hog: cv2.HOGDescriptor | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._hog is None:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self._hog = hog

    def predict(self, img: np.ndarray) -> tuple[list[list[float]], list[float], float]:
        self.load()
        assert self._hog is not None
        start = time.perf_counter()
        rects, weights = self._hog.detectMultiScale(
            img,
            hitThreshold=self.hit_threshold,
            winStride=(self.win_stride, self.win_stride),
            padding=(self.padding, self.padding),
            scale=self.scale,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        boxes: list[list[float]] = []
        scores: list[float] = []
        if len(rects) > 0:
            weights_list = weights.flatten().tolist() if hasattr(weights, "flatten") else list(weights)
            for idx, (x, y, w, h) in enumerate(rects):
                score = float(weights_list[idx]) if idx < len(weights_list) else 1.0
                if score < self.min_weight:
                    continue
                boxes.append([float(x), float(y), float(w), float(h)])
                scores.append(score)
        return boxes, scores, duration_ms
