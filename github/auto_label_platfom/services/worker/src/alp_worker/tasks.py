from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lerobot_index import index_lerobot_v3


def run_qa_job(
    *,
    project_id: str,
    source_uri: str | None,
    align_only: bool,
    sample_fps: float,
    artifacts_dir: str,
    report_id: str,
) -> dict[str, Any]:
    """Lightweight QA: index consistency checks. Full GPU QA can wrap lerobot/script later."""
    out_dir = Path(artifacts_dir) / "qa" / report_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "project_id": project_id,
        "align_only": align_only,
        "sample_fps": sample_fps,
        "checks": [],
        "ok": True,
    }
    if not source_uri:
        summary["ok"] = False
        summary["checks"].append({"name": "source", "ok": False, "detail": "no import"})
    else:
        try:
            idx = index_lerobot_v3(source_uri)
            n_ep = len(idx.episodes)
            missing_videos = []
            for ep in idx.episodes[:20]:
                for cam, ref in ep.video_refs.items():
                    p = Path(ref.get("abs_path") or "")
                    if not p.is_file():
                        missing_videos.append(f"ep{ep.episode_index}:{cam}")
            summary["checks"].extend(
                [
                    {
                        "name": "index",
                        "ok": True,
                        "detail": f"episodes={n_ep} frames={idx.total_frames} cameras={idx.camera_keys}",
                    },
                    {
                        "name": "meta_vs_index",
                        "ok": n_ep == idx.total_episodes,
                        "detail": f"indexed={n_ep} meta_total={idx.total_episodes}",
                    },
                    {
                        "name": "videos_sample",
                        "ok": len(missing_videos) == 0,
                        "detail": "ok" if not missing_videos else f"missing={missing_videos[:5]}",
                    },
                ]
            )
            summary["meta_snapshot"] = idx.meta_snapshot
            summary["ok"] = all(c["ok"] for c in summary["checks"])
        except Exception as e:
            summary["ok"] = False
            summary["checks"].append({"name": "index", "ok": False, "detail": str(e)})

    artifact = out_dir / "summary.json"
    artifact.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["artifact_uri"] = str(artifact)
    return summary


def run_prelabel_job(db: Any, project: Any, payload: Any) -> int:
    """Create stub SAM3 predictions on keyframes for existing jobs."""
    from alp_api.models import Episode, EpisodeJob, OntologyVersion, Prediction

    ont = None
    if project.ontology_version_id:
        ont = db.get(OntologyVersion, project.ontology_version_id)
    labels = []
    if ont:
        for lab in (ont.document or {}).get("labels") or []:
            labels.append(
                {
                    "name": lab.get("name"),
                    "prompt": lab.get("prompt_text") or lab.get("name"),
                }
            )
    if not labels:
        labels = [{"name": "cube", "prompt": "cube"}]

    q = db.query(EpisodeJob).filter(EpisodeJob.project_id == project.id)
    if payload.episode_indices:
        q = q.filter(EpisodeJob.episode_index.in_(payload.episode_indices))
    jobs = q.all()
    n = 0
    stride = max(1, int(payload.frame_stride or 15))
    for job in jobs:
        ep = (
            db.query(Episode)
            .filter(Episode.project_id == project.id, Episode.episode_index == job.episode_index)
            .one_or_none()
        )
        if not ep:
            continue
        cameras = job.camera_filter or project.camera_keys or list((ep.video_refs or {}).keys())
        for frame in range(0, max(ep.length, 1), stride):
            for cam in cameras:
                for lab in labels:
                    db.add(
                        Prediction(
                            job_id=job.id,
                            frame_index=frame,
                            camera_key=cam,
                            track_id=f"{lab['name']}-{frame}",
                            label=lab["name"],
                            geometry={
                                "type": "bbox",
                                "bbox": [80.0 + frame % 50, 90.0, 120.0, 100.0],
                            },
                            source="sam3",
                            score=0.75,
                            model_name="sam3",
                            model_version="stub",
                            prompt={"type": "text", "text": lab["prompt"]},
                        )
                    )
                    n += 1
        if job.status == "created":
            job.status = "prelabeled"
    db.commit()
    return n
