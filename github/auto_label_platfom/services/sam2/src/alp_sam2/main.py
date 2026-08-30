from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="ALP SAM2 Service", version="0.1.0-stub")


class PredictIn(BaseModel):
    image_id: str
    points: list[dict[str, Any]] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
    multimask: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": "stub"}


@app.post("/v1/sam2/image/predict")
def predict(body: PredictIn) -> dict:
    t0 = time.perf_counter()
    boxes = list(body.boxes)
    if not boxes and body.points:
        x = float(body.points[0].get("x", 160))
        y = float(body.points[0].get("y", 120))
        boxes = [[x - 40, y - 40, x + 40, y + 40]]
    if not boxes:
        boxes = [[120.0, 100.0, 220.0, 200.0]]
    masks = []
    for b in boxes:
        x0, y0, x1, y1 = b
        masks.append({"type": "bbox", "bbox": [x0, y0, x1 - x0, y1 - y0]})
    return {
        "masks": masks,
        "boxes": boxes,
        "scores": [0.88] * len(masks),
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "note": "stub — load github/sam2 weights for production",
    }
