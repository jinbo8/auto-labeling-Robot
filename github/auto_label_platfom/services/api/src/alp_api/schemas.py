from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class LoginIn(BaseModel):
    email: str
    password: str


class ProjectCreate(BaseModel):
    name: str
    robot_type: str | None = None
    fps: float = 30.0
    camera_keys: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class ProjectOut(BaseModel):
    id: str
    name: str
    robot_type: str | None
    fps: float
    camera_keys: list[str]
    ontology_version_id: str | None
    settings: dict[str, Any]

    model_config = {"from_attributes": True}


class OntologyDocument(BaseModel):
    version: int = 1
    labels: list[dict[str, Any]] = Field(default_factory=list)
    attributes: list[dict[str, Any]] = Field(default_factory=list)


class ImportCreate(BaseModel):
    source_uri: str
    format: Literal["lerobot_v3"] = "lerobot_v3"


class ImportOut(BaseModel):
    id: str
    source_uri: str
    format: str
    status: str
    meta_snapshot: dict[str, Any]
    qa_report_id: str | None

    model_config = {"from_attributes": True}


class EpisodeOut(BaseModel):
    episode_index: int
    length: int
    duration_s: float
    task_index: int | None
    task_text: str | None
    video_refs: dict[str, Any]

    model_config = {"from_attributes": True}


class EpisodeListOut(BaseModel):
    total: int
    items: list[EpisodeOut]


class JobSplitIn(BaseModel):
    episode_from: int = 0
    episode_to: int | None = None
    assignee_id: str | None = None
    skip_qa_failed: bool = True
    camera_filter: list[str] | None = None


class JobOut(BaseModel):
    id: str
    project_id: str
    episode_index: int
    status: str
    assignee_id: str | None
    camera_filter: list[str] | None

    model_config = {"from_attributes": True}


class Geometry(BaseModel):
    type: Literal["bbox", "polygon", "mask"]
    bbox: list[float] | None = None
    polygon: list[list[float]] | None = None
    rle: str | None = None


class AnnotationIn(BaseModel):
    id: str | None = None
    frame_index: int
    camera_key: str
    track_id: str | None = None
    label: str
    geometry: Geometry
    attributes: dict[str, Any] = Field(default_factory=dict)
    source: str = "human"


class AnnotationOut(AnnotationIn):
    id: str


class AnnotationsPut(BaseModel):
    items: list[AnnotationIn]


class PredictionOut(BaseModel):
    id: str
    frame_index: int
    camera_key: str
    track_id: str | None
    label: str
    geometry: dict[str, Any]
    attributes: dict[str, Any]
    source: str
    score: float
    model_name: str
    model_version: str
    prompt: dict[str, Any]

    model_config = {"from_attributes": True}


class AcceptPredictionsIn(BaseModel):
    prediction_ids: list[str] | None = None
    min_score: float | None = None


class ReviewIn(BaseModel):
    decision: Literal["accept", "reject"]
    issues: list[dict[str, Any]] = Field(default_factory=list)


class QaTriggerIn(BaseModel):
    align_only: bool = False
    sample_fps: float = 2.0


class QaReportOut(BaseModel):
    id: str
    status: str
    summary: dict[str, Any]
    artifact_uri: str | None

    model_config = {"from_attributes": True}


class PrelabelIn(BaseModel):
    strategy: Literal["sam3_text_keyframes", "sam3_then_sam2_track", "interactive_only"] = (
        "sam3_text_keyframes"
    )
    episode_indices: list[int] | None = None
    min_score: float = 0.5
    frame_stride: int = 15


class ExportCreate(BaseModel):
    formats: list[Literal["coco", "yolo", "lerobot_sidecar"]] = Field(
        default_factory=lambda: ["coco", "lerobot_sidecar"]
    )
    only_accepted: bool = True
    include_cameras: list[str] | None = None


class ExportOut(BaseModel):
    id: str
    formats: list[str]
    status: str
    artifact_uri: str | None

    model_config = {"from_attributes": True}


class Sam2PredictIn(BaseModel):
    image_id: str
    points: list[dict[str, Any]] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
    multimask: bool = False


class Sam3PredictIn(BaseModel):
    image_id: str
    text: str
    multimask: bool = True


class ModelPredictOut(BaseModel):
    masks: list[dict[str, Any]] = Field(default_factory=list)
    boxes: list[list[float]] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    latency_ms: float = 0.0
