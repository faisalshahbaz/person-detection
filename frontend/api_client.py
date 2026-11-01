import os
from typing import Any

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


class BackendClient:
    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self._client = httpx.Client(base_url=BACKEND_URL, timeout=120.0)

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def login(self, username: str, password: str) -> str:
        resp = self._client.post(
            "/auth/login", json={"username": username, "password": password}
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def create_session(self, source: str) -> str:
        resp = self._client.post(
            "/sessions",
            json={"source": source},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()["session_id"]

    def stop_session(self, session_id: str) -> None:
        self._client.delete(f"/sessions/{session_id}", headers=self._headers())

    def infer(self, session_id: str, image_b64: str) -> dict[str, Any]:
        resp = self._client.post(
            "/infer",
            json={"session_id": session_id, "image_b64": image_b64},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def upload(self, filename: str, data: bytes) -> dict[str, str]:
        resp = self._client.post(
            "/upload",
            files={"file": (filename, data, "application/octet-stream")},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()
