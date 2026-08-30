from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from alp_api.auth import (
    create_access_token,
    get_current_user,
    require_roles,
    verify_password,
)
from alp_api.config import settings
from alp_api.db import get_db
from alp_api.models import (
    Annotation,
    DatasetImport,
    Episode,
    EpisodeJob,
    ExportJob,
    OntologyVersion,
    Prediction,
    Project,
    QaReport,
    Review,
    User,
)
from alp_api.schemas import (
    AcceptPredictionsIn,
    AnnotationIn,
    AnnotationOut,
    AnnotationsPut,
    EpisodeListOut,
    EpisodeOut,
    ExportCreate,
    ExportOut,
    ImportCreate,
    ImportOut,
    JobOut,
    JobSplitIn,
    LoginIn,
    ModelPredictOut,
    OntologyDocument,
    PrelabelIn,
    PredictionOut,
    ProjectCreate,
    ProjectOut,
    QaReportOut,
    QaTriggerIn,
    ReviewIn,
    Sam2PredictIn,
    Sam3PredictIn,
    TokenOut,
)
from alp_api.services.exports import run_export
from alp_api.services.imports import DEFAULT_ONTOLOGY, run_import

router = APIRouter(prefix="/api/v1")


@router.post("/auth/login", response_model=TokenOut, tags=["auth"])
def login(body: LoginIn, db: Annotated[Session, Depends(get_db)]):
    user = db.query(User).filter(User.email == body.email).one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "incorrect email or password")
    token = create_access_token(user.id, user.role, user.tenant_id)
    return TokenOut(access_token=token, role=user.role)


@router.get("/projects", response_model=list[ProjectOut], tags=["projects"])
def list_projects(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    return db.query(Project).filter(Project.tenant_id == user.tenant_id).all()


@router.post("/projects", response_model=ProjectOut, tags=["projects"])
def create_project(
    payload: ProjectCreate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = Project(
        tenant_id=user.tenant_id,
        name=payload.name,
        robot_type=payload.robot_type,
        fps=payload.fps,
        camera_keys=payload.camera_keys,
        settings=payload.settings,
    )
    db.add(project)
    db.flush()
    ont = OntologyVersion(project_id=project.id, version=1, document=DEFAULT_ONTOLOGY)
    db.add(ont)
    db.flush()
    project.ontology_version_id = ont.id
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects/{project_id}", response_model=ProjectOut, tags=["projects"])
def get_project(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    project = _get_project(db, user, project_id)
    return project


@router.get("/projects/{project_id}/ontology", response_model=OntologyDocument, tags=["ontology"])
def get_ontology(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    project = _get_project(db, user, project_id)
    if not project.ontology_version_id:
        return OntologyDocument()
    ont = db.get(OntologyVersion, project.ontology_version_id)
    return OntologyDocument(**(ont.document if ont else {}))


@router.put("/projects/{project_id}/ontology", response_model=OntologyDocument, tags=["ontology"])
def put_ontology(
    project_id: str,
    doc: OntologyDocument,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = _get_project(db, user, project_id)
    ver = 1
    if project.ontology_version_id:
        prev = db.get(OntologyVersion, project.ontology_version_id)
        ver = (prev.version if prev else 0) + 1
    ont = OntologyVersion(project_id=project.id, version=ver, document=doc.model_dump())
    db.add(ont)
    db.flush()
    project.ontology_version_id = ont.id
    db.commit()
    return doc


@router.post("/projects/{project_id}/imports", response_model=ImportOut, tags=["imports"])
def create_import(
    project_id: str,
    payload: ImportCreate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = _get_project(db, user, project_id)
    imp = DatasetImport(project_id=project.id, source_uri=payload.source_uri, format=payload.format)
    db.add(imp)
    db.commit()
    db.refresh(imp)
    try:
        run_import(db, project, imp)
    except Exception as e:
        raise HTTPException(400, f"import failed: {e}") from e
    return imp


@router.get("/projects/{project_id}/imports", response_model=list[ImportOut], tags=["imports"])
def list_imports(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    _get_project(db, user, project_id)
    return db.query(DatasetImport).filter(DatasetImport.project_id == project_id).all()


@router.get("/projects/{project_id}/episodes", response_model=EpisodeListOut, tags=["episodes"])
def list_episodes(
    project_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    offset: int = 0,
    limit: int = Query(50, le=500),
):
    _get_project(db, user, project_id)
    q = db.query(Episode).filter(Episode.project_id == project_id).order_by(Episode.episode_index)
    total = q.count()
    items = q.offset(offset).limit(limit).all()
    return EpisodeListOut(total=total, items=items)


@router.post("/projects/{project_id}/qa", response_model=QaReportOut, tags=["qa"])
def trigger_qa(
    project_id: str,
    payload: QaTriggerIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = _get_project(db, user, project_id)
    from alp_worker.tasks import run_qa_job

    report = QaReport(project_id=project.id, status="pending")
    db.add(report)
    db.commit()
    db.refresh(report)
    imp = (
        db.query(DatasetImport)
        .filter(DatasetImport.project_id == project.id, DatasetImport.status == "ready")
        .order_by(DatasetImport.created_at.desc())
        .first()
    )
    summary = run_qa_job(
        project_id=project.id,
        source_uri=imp.source_uri if imp else None,
        align_only=payload.align_only,
        sample_fps=payload.sample_fps,
        artifacts_dir=str(settings.artifacts_dir),
        report_id=report.id,
    )
    report.status = "ready"
    report.summary = summary
    report.artifact_uri = summary.get("artifact_uri")
    if imp:
        imp.qa_report_id = report.id
    db.commit()
    db.refresh(report)
    return report


@router.get("/projects/{project_id}/qa/{report_id}", response_model=QaReportOut, tags=["qa"])
def get_qa(
    project_id: str,
    report_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    _get_project(db, user, project_id)
    report = db.get(QaReport, report_id)
    if not report or report.project_id != project_id:
        raise HTTPException(404, "qa report not found")
    return report


@router.post("/projects/{project_id}/prelabel", tags=["jobs"])
def prelabel(
    project_id: str,
    payload: PrelabelIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = _get_project(db, user, project_id)
    from alp_worker.tasks import run_prelabel_job

    n = run_prelabel_job(db, project, payload)
    return {"created_predictions": n, "strategy": payload.strategy}


@router.post("/projects/{project_id}/jobs/split", tags=["jobs"])
def split_jobs(
    project_id: str,
    payload: JobSplitIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = _get_project(db, user, project_id)
    q = db.query(Episode).filter(Episode.project_id == project.id)
    q = q.filter(Episode.episode_index >= payload.episode_from)
    if payload.episode_to is not None:
        q = q.filter(Episode.episode_index <= payload.episode_to)
    episodes = q.order_by(Episode.episode_index).all()
    created = []
    for ep in episodes:
        existing = (
            db.query(EpisodeJob)
            .filter(EpisodeJob.project_id == project.id, EpisodeJob.episode_index == ep.episode_index)
            .first()
        )
        if existing:
            continue
        job = EpisodeJob(
            project_id=project.id,
            episode_index=ep.episode_index,
            assignee_id=payload.assignee_id,
            camera_filter=payload.camera_filter,
            status="created",
        )
        db.add(job)
        created.append(job)
    db.commit()
    for j in created:
        db.refresh(j)
    return {"created": len(created), "jobs": created}


@router.get("/jobs", response_model=list[JobOut], tags=["jobs"])
def list_jobs(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    project_id: str | None = None,
    status: str | None = None,
    mine: bool = False,
):
    q = db.query(EpisodeJob).join(Project).filter(Project.tenant_id == user.tenant_id)
    if project_id:
        q = q.filter(EpisodeJob.project_id == project_id)
    if status:
        q = q.filter(EpisodeJob.status == status)
    if mine:
        q = q.filter(EpisodeJob.assignee_id == user.id)
    return q.order_by(EpisodeJob.episode_index).all()


@router.get("/jobs/{job_id}", tags=["jobs"])
def get_job(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    job = _get_job(db, user, job_id)
    ep = (
        db.query(Episode)
        .filter(Episode.project_id == job.project_id, Episode.episode_index == job.episode_index)
        .one_or_none()
    )
    preview = {}
    if ep:
        for cam, ref in (ep.video_refs or {}).items():
            preview[cam] = ref.get("abs_path") or ref.get("path")
    return {
        **JobOut.model_validate(job).model_dump(),
        "episode": EpisodeOut.model_validate(ep).model_dump() if ep else None,
        "preview_uris": preview,
    }


@router.get("/jobs/{job_id}/annotations", response_model=list[AnnotationOut], tags=["annotations"])
def get_annotations(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    frame_from: int | None = None,
    frame_to: int | None = None,
    camera_key: str | None = None,
):
    _get_job(db, user, job_id)
    q = db.query(Annotation).filter(Annotation.job_id == job_id)
    if frame_from is not None:
        q = q.filter(Annotation.frame_index >= frame_from)
    if frame_to is not None:
        q = q.filter(Annotation.frame_index <= frame_to)
    if camera_key:
        q = q.filter(Annotation.camera_key == camera_key)
    rows = q.all()
    return [
        AnnotationOut(
            id=r.id,
            frame_index=r.frame_index,
            camera_key=r.camera_key,
            track_id=r.track_id,
            label=r.label,
            geometry=r.geometry,
            attributes=r.attributes or {},
            source=r.source,
        )
        for r in rows
    ]


@router.put("/jobs/{job_id}/annotations", response_model=list[AnnotationOut], tags=["annotations"])
def put_annotations(
    job_id: str,
    payload: AnnotationsPut,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    job = _get_job(db, user, job_id)
    if job.status in ("accepted",):
        raise HTTPException(400, "job already accepted")
    # replace all for simplicity in MVP when full list sent; if windowed, caller sends full set they manage
    db.query(Annotation).filter(Annotation.job_id == job_id).delete()
    out = []
    for item in payload.items:
        row = Annotation(
            job_id=job_id,
            frame_index=item.frame_index,
            camera_key=item.camera_key,
            track_id=item.track_id,
            label=item.label,
            geometry=item.geometry.model_dump(),
            attributes=item.attributes,
            source=item.source,
        )
        db.add(row)
        out.append(row)
    if job.status == "created":
        job.status = "annotating"
    db.commit()
    for r in out:
        db.refresh(r)
    return [
        AnnotationOut(
            id=r.id,
            frame_index=r.frame_index,
            camera_key=r.camera_key,
            track_id=r.track_id,
            label=r.label,
            geometry=r.geometry,
            attributes=r.attributes or {},
            source=r.source,
        )
        for r in out
    ]


@router.get("/jobs/{job_id}/predictions", response_model=list[PredictionOut], tags=["annotations"])
def list_predictions(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    _get_job(db, user, job_id)
    return db.query(Prediction).filter(Prediction.job_id == job_id).all()


@router.post("/jobs/{job_id}/predictions/accept", tags=["annotations"])
def accept_predictions(
    job_id: str,
    payload: AcceptPredictionsIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    job = _get_job(db, user, job_id)
    q = db.query(Prediction).filter(Prediction.job_id == job_id)
    if payload.prediction_ids:
        q = q.filter(Prediction.id.in_(payload.prediction_ids))
    if payload.min_score is not None:
        q = q.filter(Prediction.score >= payload.min_score)
    preds = q.all()
    n = 0
    for p in preds:
        db.add(
            Annotation(
                job_id=job.id,
                frame_index=p.frame_index,
                camera_key=p.camera_key,
                track_id=p.track_id,
                label=p.label,
                geometry=p.geometry,
                attributes=p.attributes or {},
                source=p.source,
            )
        )
        n += 1
    if job.status == "created":
        job.status = "annotating"
    db.commit()
    return {"accepted": n}


@router.post("/jobs/{job_id}/submit", response_model=JobOut, tags=["jobs"])
def submit_job(
    job_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    job = _get_job(db, user, job_id)
    job.status = "submitted"
    db.commit()
    db.refresh(job)
    return job


@router.post("/jobs/{job_id}/review", response_model=JobOut, tags=["review"])
def review_job(
    job_id: str,
    payload: ReviewIn,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("reviewer", "manager", "owner"))],
):
    job = _get_job(db, user, job_id)
    db.add(
        Review(
            job_id=job.id,
            reviewer_id=user.id,
            decision=payload.decision,
            issues=payload.issues,
        )
    )
    job.status = "accepted" if payload.decision == "accept" else "rejected"
    db.commit()
    db.refresh(job)
    return job


@router.post("/projects/{project_id}/exports", response_model=ExportOut, tags=["exports"])
def create_export(
    project_id: str,
    payload: ExportCreate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_roles("manager", "owner"))],
):
    project = _get_project(db, user, project_id)
    export = ExportJob(project_id=project.id, formats=payload.formats, status="pending")
    db.add(export)
    db.commit()
    db.refresh(export)
    try:
        run_export(db, project, export)
    except Exception as e:
        raise HTTPException(500, f"export failed: {e}") from e
    return export


@router.get("/projects/{project_id}/exports/{export_id}", response_model=ExportOut, tags=["exports"])
def get_export(
    project_id: str,
    export_id: str,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    _get_project(db, user, project_id)
    export = db.get(ExportJob, export_id)
    if not export or export.project_id != project_id:
        raise HTTPException(404, "export not found")
    return export


@router.post("/models/sam2/predict", response_model=ModelPredictOut, tags=["models"])
def sam2_predict(
    payload: Sam2PredictIn,
    user: Annotated[User, Depends(get_current_user)],
):
    return _proxy_model(f"{settings.sam2_url}/v1/sam2/image/predict", payload.model_dump())


@router.post("/models/sam3/predict", response_model=ModelPredictOut, tags=["models"])
def sam3_predict(
    payload: Sam3PredictIn,
    user: Annotated[User, Depends(get_current_user)],
):
    return _proxy_model(f"{settings.sam3_url}/v1/sam3/image/predict", payload.model_dump())


def _proxy_model(url: str, body: dict) -> ModelPredictOut:
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            return ModelPredictOut(**r.json())
    except Exception as e:
        # local fallback stub so Studio works without model services
        return ModelPredictOut(
            masks=[{"type": "bbox", "bbox": [100.0, 100.0, 80.0, 80.0]}],
            boxes=[[100.0, 100.0, 180.0, 180.0]],
            scores=[0.42],
            latency_ms=0.0,
        )


def _get_project(db: Session, user: User, project_id: str) -> Project:
    project = db.get(Project, project_id)
    if not project or project.tenant_id != user.tenant_id:
        raise HTTPException(404, "project not found")
    return project


def _get_job(db: Session, user: User, job_id: str) -> EpisodeJob:
    job = db.get(EpisodeJob, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    project = db.get(Project, job.project_id)
    if not project or project.tenant_id != user.tenant_id:
        raise HTTPException(404, "job not found")
    return job
