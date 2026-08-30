from __future__ import annotations

import argparse
import os
from pathlib import Path

from alp_api.auth import hash_password
from alp_api.db import SessionLocal, init_db
from alp_api.models import EpisodeJob, Project, Tenant, User
from alp_api.services.imports import DEFAULT_ONTOLOGY, run_import
from alp_api.models import DatasetImport, OntologyVersion


def cmd_init(_: argparse.Namespace) -> None:
    init_db()
    db = SessionLocal()
    try:
        tenant = db.query(Tenant).first()
        if not tenant:
            tenant = Tenant(name="default", plan="community")
            db.add(tenant)
            db.flush()
        seeds = [
            ("manager@local", "manager123", "manager"),
            ("annotator@local", "annotator123", "annotator"),
            ("reviewer@local", "reviewer123", "reviewer"),
            ("owner@local", "owner123", "owner"),
        ]
        for email, password, role in seeds:
            if db.query(User).filter(User.email == email).first():
                continue
            db.add(
                User(
                    tenant_id=tenant.id,
                    email=email,
                    hashed_password=hash_password(password),
                    role=role,
                )
            )
        db.commit()
        print("DB initialized and seed users ready.")
    finally:
        db.close()


def cmd_seed_demo(_: argparse.Namespace) -> None:
    init_db()
    root = os.environ.get("ALP_DATA_ROOT")
    if not root:
        # repo-relative default
        here = Path(__file__).resolve()
        # alp_api/cli.py -> repo root = parents[6]
        candidate = here.parents[6] / "lerobot" / "datasets" / "svla_so100_pickplace"
        if not candidate.is_dir():
            candidate = (here.parents[4] / ".." / ".." / "lerobot" / "datasets" / "svla_so100_pickplace").resolve()
        root = str(candidate)
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise SystemExit(f"dataset not found: {root_path}")

    db = SessionLocal()
    try:
        tenant = db.query(Tenant).first()
        if not tenant:
            raise SystemExit("run init first")
        manager = db.query(User).filter(User.email == "manager@local").one()
        annotator = db.query(User).filter(User.email == "annotator@local").one()
        project = db.query(Project).filter(Project.name == "svla_so100_pickplace").first()
        if not project:
            project = Project(
                tenant_id=tenant.id,
                name="svla_so100_pickplace",
                settings={},
            )
            db.add(project)
            db.flush()
            ont = OntologyVersion(project_id=project.id, version=1, document=DEFAULT_ONTOLOGY)
            db.add(ont)
            db.flush()
            project.ontology_version_id = ont.id
            db.commit()
            db.refresh(project)

        imp = DatasetImport(project_id=project.id, source_uri=str(root_path), format="lerobot_v3")
        db.add(imp)
        db.commit()
        db.refresh(imp)
        run_import(db, project, imp)
        print(f"imported {root_path} -> project {project.id}")

        # split first 5 jobs assigned to annotator
        from alp_api.models import Episode

        eps = (
            db.query(Episode)
            .filter(Episode.project_id == project.id, Episode.episode_index < 5)
            .order_by(Episode.episode_index)
            .all()
        )
        n = 0
        for ep in eps:
            exists = (
                db.query(EpisodeJob)
                .filter(EpisodeJob.project_id == project.id, EpisodeJob.episode_index == ep.episode_index)
                .first()
            )
            if exists:
                continue
            db.add(
                EpisodeJob(
                    project_id=project.id,
                    episode_index=ep.episode_index,
                    assignee_id=annotator.id,
                    status="created",
                )
            )
            n += 1
        db.commit()
        print(f"created {n} jobs for annotator {annotator.email}; manager={manager.email}")
        print(f"PROJECT_ID={project.id}")
    finally:
        db.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="alp-api")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(func=cmd_init)
    sub.add_parser("seed-demo").set_defaults(func=cmd_seed_demo)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
