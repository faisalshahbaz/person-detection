import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OnnxYoloPersonDetector:
    """YOLOv8n person detection via ONNX Runtime (no PyTorch at runtime)."""

    def __init__(
        self,
        model_path: str,
        input_size: int,
        conf_threshold: float,
        iou_threshold: float,
        person_class_id: int = 0,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_class_id = person_class_id
        self._session: ort.InferenceSession | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._session is None:
                path = Path(self.model_path)
                if not path.is_file():
                    raise FileNotFoundError(f"ONNX model not found: {path}")
                self._session = ort.InferenceSession(
                    str(path),
                    providers=["CPUExecutionProvider"],
                )

    @staticmethod
    def _letterbox(
        img: np.ndarray, new_size: int
    ) -> tuple[np.ndarray, float, float, float]:
        h, w = img.shape[:2]
        scale = min(new_size / h, new_size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_w = (new_size - nw) / 2
        pad_h = (new_size - nh) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return out, scale, pad_w, pad_h

    def predict(self, img: np.ndarray) -> tuple[list[list[float]], list[float], float]:
        self.load()
        assert self._session is not None
        start = time.perf_counter()
        h0, w0 = img.shape[:2]
        lb, scale, pad_w, pad_h = self._letterbox(img, self.input_size)
        blob = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})
        preds = np.squeeze(outputs[0]).T  # (N, 4 + num_classes)

        boxes_xywh: list[list[float]] = []
        scores: list[float] = []
        if preds.size > 0:
            class_scores = preds[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[np.arange(len(preds)), class_ids]
            mask = (class_ids == self.person_class_id) & (
                confidences >= self.conf_threshold
            )
            filtered = preds[mask]
            conf_filtered = confidences[mask]
            if len(filtered) > 0:
                cx = filtered[:, 0]
                cy = filtered[:, 1]
                bw = filtered[:, 2]
                bh = filtered[:, 3]
                x1 = (cx - bw / 2 - pad_w) / scale
                y1 = (cy - bh / 2 - pad_h) / scale
                x2 = (cx + bw / 2 - pad_w) / scale
                y2 = (cy + bh / 2 - pad_h) / scale
                x1 = np.clip(x1, 0, w0)
                y1 = np.clip(y1, 0, h0)
                x2 = np.clip(x2, 0, w0)
                y2 = np.clip(y2, 0, h0)
                nms_inputs = []
                for i in range(len(filtered)):
                    nms_inputs.append(
                        [int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])]
                    )
                indices = cv2.dnn.NMSBoxes(
                    nms_inputs,
                    conf_filtered.tolist(),
                    self.conf_threshold,
                    self.iou_threshold,
                )
                if len(indices) > 0:
                    for idx in np.array(indices).flatten():
                        bx, by, bw_i, bh_i = nms_inputs[int(idx)]
                        boxes_xywh.append(
                            [float(bx), float(by), float(bw_i), float(bh_i)]
                        )
                        scores.append(float(conf_filtered[int(idx)]))

        duration_ms = (time.perf_counter() - start) * 1000
        return boxes_xywh, scores, duration_ms
