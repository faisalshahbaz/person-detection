import asyncio
import base64
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2

from app.ml_client import MLClient


@dataclass
class VideoJob:
    job_id: str
    user_id: str
    video_path: str
    status: str = "pending"
    subscribers: set[Any] = field(default_factory=set)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, VideoJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, user_id: str, video_path: str) -> VideoJob:
        job = VideoJob(job_id=str(uuid.uuid4()), user_id=user_id, video_path=video_path)
        async with self._lock:
            self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: str, user_id: str) -> VideoJob:
        async with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError("Job not found")
        if job.user_id != user_id:
            raise PermissionError("Forbidden")
        return job

    async def subscribe(self, job_id: str, user_id: str, ws: Any) -> VideoJob:
        job = await self.get(job_id, user_id)
        job.subscribers.add(ws)
        return job

    async def unsubscribe(self, job_id: str, ws: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
        if job:
            job.subscribers.discard(ws)

    async def broadcast(self, job: VideoJob, message: dict[str, Any]) -> None:
        text = json.dumps(message)
        dead: list[Any] = []
        for ws in list(job.subscribers):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            job.subscribers.discard(ws)

    async def _wait_for_subscriber(self, job: VideoJob, timeout_sec: float = 60.0) -> bool:
        steps = int(timeout_sec * 10)
        for _ in range(steps):
            if job.subscribers:
                return True
            await asyncio.sleep(0.1)
        return False

    async def process_job(self, job: VideoJob, ml: MLClient, sample_every: int = 2) -> None:
        job.status = "processing"
        if not await self._wait_for_subscriber(job):
            job.status = "failed"
            await self.broadcast(job, {"type": "error", "detail": "No viewer connected"})
            return
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            job.status = "failed"
            await self.broadcast(job, {"type": "error", "detail": "Cannot open video"})
            return
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % sample_every == 0:
                    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    try:
                        result = await ml.predict(b64)
                    except Exception as exc:
                        await self.broadcast(
                            job, {"type": "error", "detail": str(exc), "frame_index": idx}
                        )
                        break
                    annotated = frame.copy()
                    for box, score in zip(
                        result.get("boxes", []), result.get("scores", [])
                    ):
                        x, y, w, h = [int(v) for v in box]
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            annotated,
                            f"person {score:.2f}",
                            (x, max(y - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                    count = result.get("person_count", 0)
                    cv2.putText(
                        annotated,
                        f"Count: {count}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 200, 255),
                        2,
                    )
                    _, ann_buf = cv2.imencode(
                        ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    await self.broadcast(
                        job,
                        {
                            "type": "frame",
                            "frame_index": idx,
                            "person_count": count,
                            "image_b64": base64.b64encode(ann_buf.tobytes()).decode("ascii"),
                        },
                    )
                    await asyncio.sleep(0.03)
                idx += 1
            job.status = "completed"
            await self.broadcast(job, {"type": "done", "frames_processed": idx})
        finally:
            cap.release()
