from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import ML_INTERNAL_TOKEN
from app.inference import engine


class PredictRequest(BaseModel):
    image_b64: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    boxes: list[list[float]]
    scores: list[float]
    person_count: int
    frame_width: int
    frame_height: int
    inference_ms: float


def verify_internal_token(
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    if not x_internal_token or x_internal_token != ML_INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid internal token")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    engine.load_model()
    yield


app = FastAPI(title="Person Detection ML Service", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    from app.config import DETECTOR

    return {"status": "ok", "detector": DETECTOR}


@app.get("/metrics")
async def metrics(_: None = Depends(verify_internal_token)) -> dict:
    return engine.metrics.snapshot()


@app.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    _: None = Depends(verify_internal_token),
) -> PredictResponse:
    try:
        result = await engine.predict(body.image_b64)
    except RuntimeError as exc:
        if "queue full" in str(exc).lower():
            raise HTTPException(status_code=503, detail="Inference queue full") from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    result.pop("detector", None)
    return PredictResponse(**result)
