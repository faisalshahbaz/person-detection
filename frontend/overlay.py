import base64

import cv2
import numpy as np


def draw_detections(
    frame: np.ndarray,
    boxes: list[list[float]],
    scores: list[float],
    person_count: int,
) -> np.ndarray:
    out = frame.copy()
    for box, score in zip(boxes, scores):
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"person {score:.2f}",
            (x, max(y - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.putText(
        out,
        f"Count: {person_count}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 200, 255),
        2,
    )
    return out


def frame_to_b64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")
