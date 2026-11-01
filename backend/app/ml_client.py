import asyncio
import time
from typing import Any

import httpx

from app.config import Settings


class CircuitBreaker:
    def __init__(self, fail_threshold: int, open_seconds: int) -> None:
        self.fail_threshold = fail_threshold
        self.open_seconds = open_seconds
        self.failures = 0
        self.open_until = 0.0
        self._lock = asyncio.Lock()

    async def before_call(self) -> None:
        async with self._lock:
            if time.time() < self.open_until:
                raise RuntimeError("ML service circuit open")

    async def record_success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.open_until = 0.0

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.fail_threshold:
                self.open_until = time.time() + self.open_seconds


class MLClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_inflight_infer)
        self._breaker = CircuitBreaker(
            settings.ml_circuit_fail_threshold, settings.ml_circuit_open_sec
        )
        self._client = httpx.AsyncClient(
            base_url=settings.ml_service_url,
            timeout=settings.ml_request_timeout_sec,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def predict(self, image_b64: str) -> dict[str, Any]:
        await self._breaker.before_call()
        async with self._semaphore:
            try:
                resp = await self._client.post(
                    "/predict",
                    json={"image_b64": image_b64},
                    headers={"X-Internal-Token": self.settings.ml_internal_token},
                )
                if resp.status_code == 503:
                    raise RuntimeError("ML queue full")
                resp.raise_for_status()
                await self._breaker.record_success()
                return resp.json()
            except Exception:
                await self._breaker.record_failure()
                raise
