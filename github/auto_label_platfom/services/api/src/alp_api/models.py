from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), default="default")
    plan: Mapped[str] = mapped_column(String(32), default="community")
    users: Mapped[list[User]] = relationship(back_populates="tenant")
    projects: Mapped[list[Project]] = relationship(back_populates="tenant")


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="annotator")
    status: Mapped[str] = mapped_column(String(32), default="active")
    tenant: Mapped[Tenant] = relationship(back_populates="users")


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"))
    name: Mapped[str] = mapped_column(String(255))
    robot_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fps: Mapped[float] = mapped_column(Float, default=30.0)
    camera_keys: Mapped[list] = mapped_column(JSON, default=list)
    ontology_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    tenant: Mapped[Tenant] = relationship(back_populates="projects")
    ontology_versions: Mapped[list[OntologyVersion]] = relationship(back_populates="project")
    imports: Mapped[list[DatasetImport]] = relationship(back_populates="project")
    episodes: Mapped[list[Episode]] = relationship(back_populates="project")
    jobs: Mapped[list[EpisodeJob]] = relationship(back_populates="project")


class OntologyVersion(Base):
    __tablename__ = "ontology_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    document: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    project: Mapped[Project] = relationship(back_populates="ontology_versions")


class DatasetImport(Base):
    __tablename__ = "dataset_imports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    source_uri: Mapped[str] = mapped_column(Text)
    format: Mapped[str] = mapped_column(String(32), default="lerobot_v3")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    meta_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    qa_report_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    project: Mapped[Project] = relationship(back_populates="imports")


class Episode(Base):
    __tablename__ = "episodes"
    __table_args__ = (UniqueConstraint("project_id", "episode_index"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    episode_index: Mapped[int] = mapped_column(Integer)
    length: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    task_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    task_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_refs: Mapped[dict] = mapped_column(JSON, default=dict)
    project: Mapped[Project] = relationship(back_populates="episodes")


class EpisodeJob(Base):
    __tablename__ = "episode_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    episode_index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="created")
    assignee_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    camera_filter: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    project: Mapped[Project] = relationship(back_populates="jobs")
    annotations: Mapped[list[Annotation]] = relationship(back_populates="job")
    predictions: Mapped[list[Prediction]] = relationship(back_populates="job")
    reviews: Mapped[list[Review]] = relationship(back_populates="job")


class Annotation(Base):
    __tablename__ = "annotations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("episode_jobs.id"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    camera_key: Mapped[str] = mapped_column(String(255))
    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label: Mapped[str] = mapped_column(String(128))
    geometry: Mapped[dict] = mapped_column(JSON, default=dict)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(32), default="human")
    job: Mapped[EpisodeJob] = relationship(back_populates="annotations")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("episode_jobs.id"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    camera_key: Mapped[str] = mapped_column(String(255))
    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    label: Mapped[str] = mapped_column(String(128))
    geometry: Mapped[dict] = mapped_column(JSON, default=dict)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(32), default="sam3")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    model_name: Mapped[str] = mapped_column(String(64), default="sam3")
    model_version: Mapped[str] = mapped_column(String(64), default="stub")
    prompt: Mapped[dict] = mapped_column(JSON, default=dict)
    job: Mapped[EpisodeJob] = relationship(back_populates="predictions")


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("episode_jobs.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(String(36))
    decision: Mapped[str] = mapped_column(String(16))
    issues: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    job: Mapped[EpisodeJob] = relationship(back_populates="reviews")


class QaReport(Base):
    __tablename__ = "qa_reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExportJob(Base):
    __tablename__ = "export_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    formats: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
