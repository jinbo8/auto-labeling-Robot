from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ALP SAM3 Service", version="0.1.0-stub")


class PredictIn(BaseModel):
    image_id: str
    text: str
    multimask: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": "stub"}


@app.post("/v1/sam3/image/predict")
def predict(body: PredictIn) -> dict:
    t0 = time.perf_counter()
    # Deterministic stub boxes from text hash so UI can demo multi-instance
    seed = sum(ord(c) for c in body.text) % 50
    boxes = [
        [80.0 + seed, 70.0, 200.0 + seed, 180.0],
        [240.0, 140.0 + seed / 2, 340.0, 240.0 + seed / 2],
    ]
    masks = [{"type": "bbox", "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]} for b in boxes]
    return {
        "masks": masks,
        "boxes": boxes,
        "scores": [0.91, 0.77],
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "text": body.text,
        "note": "stub — load github/sam3 weights for production",
    }
