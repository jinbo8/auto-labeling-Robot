from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from lerobot_index import index_lerobot_v3

from alp_api.models import DatasetImport, Episode, Project


DEFAULT_ONTOLOGY = {
    "version": 1,
    "labels": [
        {
            "name": "cube",
            "color": "#E74C3C",
            "tools": ["bbox", "polygon", "mask"],
            "prompt_text": "cube",
        },
        {
            "name": "box",
            "color": "#3498DB",
            "tools": ["bbox", "mask"],
            "prompt_text": "box",
        },
        {
            "name": "gripper",
            "color": "#2ECC71",
            "tools": ["bbox", "mask"],
            "prompt_text": "gripper",
        },
    ],
    "attributes": [{"name": "occluded", "type": "boolean", "applies_to": ["*"]}],
}


def run_import(db: Session, project: Project, imp: DatasetImport) -> DatasetImport:
    imp.status = "indexing"
    db.commit()
    try:
        root = Path(imp.source_uri).expanduser().resolve()
        idx = index_lerobot_v3(root)
        # clear old episodes for re-import of same project
        db.query(Episode).filter(Episode.project_id == project.id).delete()
        for ep in idx.episodes:
            db.add(
                Episode(
                    project_id=project.id,
                    episode_index=ep.episode_index,
                    length=ep.length,
                    duration_s=ep.duration_s,
                    task_index=ep.task_index,
                    task_text=ep.task_text,
                    video_refs=ep.video_refs,
                )
            )
        project.robot_type = idx.robot_type or project.robot_type
        project.fps = idx.fps
        project.camera_keys = idx.camera_keys
        imp.meta_snapshot = idx.meta_snapshot
        imp.status = "ready"
        imp.error = None
        db.commit()
        db.refresh(imp)
        return imp
    except Exception as e:
        imp.status = "failed"
        imp.error = str(e)
        db.commit()
        db.refresh(imp)
        raise
