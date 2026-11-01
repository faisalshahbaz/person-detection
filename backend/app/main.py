import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from app.auth import create_access_token, decode_token, get_current_user, verify_demo_user
from app.config import Settings, get_settings
from app.ml_client import MLClient
from app.sessions import SessionStore
from app.storage import UploadStorage
from app.video_jobs import JobManager

session_store = SessionStore(get_settings())
ml_client: MLClient | None = None
storage: UploadStorage | None = None
job_manager = JobManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ml_client, storage
    settings = get_settings()
    await session_store.connect()
    ml_client = MLClient(settings)
    storage = UploadStorage(settings)
    yield
    await session_store.close()
    if ml_client:
        await ml_client.close()


app = FastAPI(title="Person Detection Backend", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SessionCreateRequest(BaseModel):
    source: str = Field(..., pattern="^(webcam|upload)$")


class SessionResponse(BaseModel):
    session_id: str
    source: str


class InferRequest(BaseModel):
    session_id: str
    image_b64: str = Field(..., min_length=1)


class InferResponse(BaseModel):
    boxes: list[list[float]]
    scores: list[float]
    person_count: int
    frame_width: int
    frame_height: int
    inference_ms: float | None = None


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    settings: Settings = Depends(get_settings),
) -> LoginResponse:
    if not verify_demo_user(body.username, body.password, settings):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return LoginResponse(access_token=create_access_token(body.username, settings))


@app.post("/sessions", response_model=SessionResponse)
async def create_session(
    body: SessionCreateRequest,
    user_id: str = Depends(get_current_user),
) -> SessionResponse:
    info = await session_store.create_session(user_id, body.source)
    return SessionResponse(session_id=info.session_id, source=info.source)


@app.delete("/sessions/{session_id}")
async def stop_session(
    session_id: str,
    user_id: str = Depends(get_current_user),
) -> dict[str, str]:
    await session_store.stop_session(session_id, user_id)
    return {"status": "stopped"}


@app.post("/infer", response_model=InferResponse)
async def infer(
    body: InferRequest,
    user_id: str = Depends(get_current_user),
) -> InferResponse:
    if ml_client is None:
        raise HTTPException(status_code=503, detail="ML client unavailable")
    await session_store.get_session(body.session_id, user_id)
    try:
        result = await ml_client.predict(body.image_b64)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="ML inference failed") from exc
    return InferResponse(**result)


@app.post("/upload")
async def upload_video(
    file: Annotated[UploadFile, File(...)],
    user_id: str = Depends(get_current_user),
) -> dict[str, str]:
    if storage is None or ml_client is None:
        raise HTTPException(status_code=503, detail="Storage unavailable")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 200MB)")
    path = storage.save_local(file.filename, data)
    job = await job_manager.create(user_id, path)
    asyncio.create_task(job_manager.process_job(job, ml_client))
    return {"job_id": job.job_id, "status": job.status}


@app.websocket("/jobs/{job_id}/stream")
async def job_stream(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    settings = get_settings()
    try:
        user_id = str(decode_token(token, settings)["sub"])
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        job = await job_manager.subscribe(job_id, user_id, websocket)
    except (KeyError, PermissionError):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.send_json({"type": "connected", "job_id": job.job_id})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await job_manager.unsubscribe(job_id, websocket)
