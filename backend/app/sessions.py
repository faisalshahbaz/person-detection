import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import redis.asyncio as redis
from fastapi import HTTPException, status

from app.config import Settings

SESSION_PREFIX = "session:"
USER_SESSIONS_PREFIX = "user_sessions:"
GLOBAL_SESSIONS_KEY = "global_sessions"


@dataclass
class SessionInfo:
    session_id: str
    user_id: str
    source: str
    created_at: str


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    @property
    def r(self) -> redis.Redis:
        if self._redis is None:
            raise RuntimeError("SessionStore not connected")
        return self._redis

    async def create_session(self, user_id: str, source: str) -> SessionInfo:
        global_count = int(await self.r.get(GLOBAL_SESSIONS_KEY) or 0)
        if global_count >= self.settings.max_global_sessions:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="System at capacity. Try again shortly.",
            )
        user_key = f"{USER_SESSIONS_PREFIX}{user_id}"
        if await self.r.scard(user_key) >= self.settings.max_user_sessions:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="User session limit reached.",
            )
        session_id = str(uuid.uuid4())
        info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            source=source,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        pipe = self.r.pipeline()
        pipe.set(f"{SESSION_PREFIX}{session_id}", json.dumps(info.__dict__), ex=14400)
        pipe.sadd(user_key, session_id)
        pipe.expire(user_key, 14400)
        pipe.incr(GLOBAL_SESSIONS_KEY)
        await pipe.execute()
        return info

    async def get_session(self, session_id: str, user_id: str) -> SessionInfo:
        raw = await self.r.get(f"{SESSION_PREFIX}{session_id}")
        if not raw:
            raise HTTPException(status_code=404, detail="Session not found")
        data = json.loads(raw)
        if data.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Session forbidden")
        return SessionInfo(**data)

    async def stop_session(self, session_id: str, user_id: str) -> None:
        try:
            await self.get_session(session_id, user_id)
        except HTTPException:
            return
        pipe = self.r.pipeline()
        pipe.delete(f"{SESSION_PREFIX}{session_id}")
        pipe.srem(f"{USER_SESSIONS_PREFIX}{user_id}", session_id)
        pipe.decr(GLOBAL_SESSIONS_KEY)
        await pipe.execute()
        if int(await self.r.get(GLOBAL_SESSIONS_KEY) or 0) < 0:
            await self.r.set(GLOBAL_SESSIONS_KEY, 0)
