from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from lerobot_index import annotations_to_coco, build_export_zip

from alp_api.config import settings
from alp_api.models import Annotation, Episode, EpisodeJob, ExportJob, Project


def run_export(db: Session, project: Project, export: ExportJob) -> ExportJob:
    export.status = "running"
    db.commit()
    try:
        q = db.query(EpisodeJob).filter(EpisodeJob.project_id == project.id)
        # ExportCreate.only_accepted handled by caller via formats settings stored? we store in formats only
        # Use project.settings or assume only accepted if "accepted" jobs preferred — filter accepted
        jobs = q.filter(EpisodeJob.status == "accepted").all()
        if not jobs:
            jobs = q.all()

        items: list[dict] = []
        cats: list[str] = []
        ont = None
        if project.ontology_version_id:
            from alp_api.models import OntologyVersion

            ont = db.get(OntologyVersion, project.ontology_version_id)
            if ont:
                cats = [x.get("name") for x in (ont.document or {}).get("labels") or [] if x.get("name")]

        sidecar: dict[int, dict] = {}
        for job in jobs:
            ep = (
                db.query(Episode)
                .filter(Episode.project_id == project.id, Episode.episode_index == job.episode_index)
                .one_or_none()
            )
            anns = db.query(Annotation).filter(Annotation.job_id == job.id).all()
            ep_items = []
            for a in anns:
                row = {
                    "episode_index": job.episode_index,
                    "frame_index": a.frame_index,
                    "camera_key": a.camera_key,
                    "track_id": a.track_id,
                    "label": a.label,
                    "geometry": a.geometry,
                    "source": a.source,
                }
                items.append(row)
                ep_items.append(row)
                if a.label not in cats:
                    cats.append(a.label)
            sidecar[job.episode_index] = {
                "episode_index": job.episode_index,
                "task": ep.task_text if ep else None,
                "fps": project.fps,
                "items": ep_items,
            }

        coco = None
        formats = export.formats or []
        if "coco" in formats:
            coco = annotations_to_coco(items, categories=cats or ["object"])

        out = settings.artifacts_dir / "exports" / f"{export.id}.zip"
        manifest = {
            "project_id": project.id,
            "project_name": project.name,
            "formats": formats,
            "n_jobs": len(jobs),
            "n_annotations": len(items),
        }
        # always write sidecar when requested or as default companion
        if "lerobot_sidecar" not in formats and coco is None:
            formats = list(formats) + ["lerobot_sidecar"]
        build_export_zip(out, manifest=manifest, sidecar_by_episode=sidecar, coco=coco)
        export.artifact_uri = str(out)
        export.status = "ready"
        db.commit()
        db.refresh(export)
        return export
    except Exception:
        export.status = "failed"
        db.commit()
        raise
