import asyncio
import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import (
    CONF_THRESHOLD,
    DETECTOR,
    HOG_HIT_THRESHOLD,
    HOG_MIN_WEIGHT,
    HOG_PADDING,
    HOG_SCALE,
    HOG_WIN_STRIDE,
    IOU_THRESHOLD,
    ML_QUEUE_MAX,
    ML_WORKERS,
    ONNX_INPUT_SIZE,
    ONNX_MODEL_PATH,
    PERSON_CLASS_ID,
    YOLO_MODEL,
)
from app.hog_detector import HogPersonDetector
from app.onnx_yolo_detector import OnnxYoloPersonDetector


@dataclass
class Metrics:
    total_requests: int = 0
    rejected_queue_full: int = 0
    inference_errors: int = 0
    total_inference_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_success(self, duration_ms: float) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_inference_ms += duration_ms

    def record_error(self) -> None:
        with self.lock:
            self.inference_errors += 1

    def record_rejected(self) -> None:
        with self.lock:
            self.rejected_queue_full += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            avg = (
                self.total_inference_ms / self.total_requests
                if self.total_requests
                else 0.0
            )
            return {
                "detector": DETECTOR,
                "total_requests": self.total_requests,
                "rejected_queue_full": self.rejected_queue_full,
                "inference_errors": self.inference_errors,
                "avg_inference_ms": round(avg, 2),
            }


class InferenceEngine:
    def __init__(self) -> None:
        self._ultralytics = None
        self._ultralytics_lock = threading.Lock()
        self._onnx = OnnxYoloPersonDetector(
            ONNX_MODEL_PATH,
            ONNX_INPUT_SIZE,
            CONF_THRESHOLD,
            IOU_THRESHOLD,
            PERSON_CLASS_ID,
        )
        self._hog = HogPersonDetector(
            HOG_HIT_THRESHOLD,
            HOG_WIN_STRIDE,
            HOG_PADDING,
            HOG_SCALE,
            HOG_MIN_WEIGHT,
        )
        self._executor = ThreadPoolExecutor(max_workers=ML_WORKERS)
        self._semaphore = asyncio.Semaphore(ML_WORKERS)
        self._queue_depth = 0
        self._queue_lock = asyncio.Lock()
        self.metrics = Metrics()

    def load_model(self) -> None:
        if DETECTOR == "ultralytics":
            from ultralytics import YOLO

            with self._ultralytics_lock:
                if self._ultralytics is None:
                    self._ultralytics = YOLO(YOLO_MODEL)
        elif DETECTOR == "hog":
            self._hog.load()
        else:
            self._onnx.load()

    def _decode_image(self, image_b64: str) -> np.ndarray:
        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img

    def _pack_result(
        self,
        img: np.ndarray,
        boxes: list[list[float]],
        scores: list[float],
        duration_ms: float,
        detector: str,
    ) -> dict[str, Any]:
        h, w = img.shape[:2]
        self.metrics.record_success(duration_ms)
        return {
            "boxes": boxes,
            "scores": scores,
            "person_count": len(boxes),
            "frame_width": w,
            "frame_height": h,
            "inference_ms": round(duration_ms, 2),
            "detector": detector,
        }

    def _predict_onnx(self, img: np.ndarray) -> dict[str, Any]:
        boxes, scores, duration_ms = self._onnx.predict(img)
        return self._pack_result(img, boxes, scores, duration_ms, "yolo-onnx")

    def _predict_hog(self, img: np.ndarray) -> dict[str, Any]:
        boxes, scores, duration_ms = self._hog.predict(img)
        return self._pack_result(img, boxes, scores, duration_ms, "hog")

    def _predict_ultralytics(self, img: np.ndarray) -> dict[str, Any]:
        assert self._ultralytics is not None
        start = time.perf_counter()
        results = self._ultralytics.predict(
            source=img,
            classes=[PERSON_CLASS_ID],
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        boxes: list[list[float]] = []
        scores: list[float] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(box.conf[0]))
        return self._pack_result(img, boxes, scores, duration_ms, "ultralytics")

    def _predict_sync(self, image_b64: str) -> dict[str, Any]:
        self.load_model()
        img = self._decode_image(image_b64)
        if DETECTOR == "ultralytics":
            return self._predict_ultralytics(img)
        if DETECTOR == "hog":
            return self._predict_hog(img)
        return self._predict_onnx(img)

    async def predict(self, image_b64: str) -> dict[str, Any]:
        async with self._queue_lock:
            if self._queue_depth >= ML_QUEUE_MAX:
                self.metrics.record_rejected()
                raise RuntimeError("Inference queue full")
            self._queue_depth += 1
        try:
            async with self._semaphore:
                loop = asyncio.get_running_loop()
                try:
                    return await loop.run_in_executor(
                        self._executor, self._predict_sync, image_b64
                    )
                except Exception:
                    self.metrics.record_error()
                    raise
        finally:
            async with self._queue_lock:
                self._queue_depth -= 1


engine = InferenceEngine()
