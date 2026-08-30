from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
LEROBOT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASETS_DIR = LEROBOT_DIR / "datasets"
DEFAULT_RUN_DIR = LEROBOT_DIR / "run"
DEFAULT_REPORTS_DIR = DEFAULT_RUN_DIR
RUN_STAMP_FMT = "%Y%m%d_%H%M%S"


def make_staging_dir(run_root: Path) -> Path:
    """Create a hidden working directory under ``run_root``; renamed when the run ends."""
    run_root.mkdir(parents=True, exist_ok=True)
    staging = run_root / f".running_{os.getpid()}"
    if staging.exists():
        staging = run_root / f".running_{os.getpid()}_{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def unique_run_dir(run_root: Path, stamp: str | None = None) -> Path:
    """Return ``run_root/<end-time>``, appending ``_2``, ``_3``, ... on collision."""
    stamp = stamp or datetime.now().strftime(RUN_STAMP_FMT)
    dest = run_root / stamp
    if not dest.exists():
        return dest
    i = 2
    while True:
        cand = run_root / f"{stamp}_{i}"
        if not cand.exists():
            return cand
        i += 1


def finalize_run_dir(staging: Path, run_root: Path) -> Path:
    """Rename the staging folder to the local end-time stamp and return the final path."""
    dest = unique_run_dir(run_root)
    if staging.exists():
        staging.rename(dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def resolve_run_dir_for_viz(path: Path) -> Path:
    """If ``path`` is the run root, pick the latest timestamped subfolder."""
    if not path.is_dir():
        return path
    if any(path.glob("*/alignment_episodes.parquet")):
        return path
    children = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if children:
        return sorted(children, key=lambda p: p.name)[-1]
    return path


def discover_datasets(datasets_dir: Path) -> list[Path]:
    """Find LeRobot v3 roots (directories that contain meta/info.json)."""
    roots: list[Path] = []
    if not datasets_dir.is_dir():
        return roots
    for info in sorted(datasets_dir.glob("*/meta/info.json")):
        roots.append(info.parents[1])
    return roots


def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def video_keys(info: dict) -> list[str]:
    keys = []
    for name, spec in (info.get("features") or {}).items():
        if spec.get("dtype") == "video":
            keys.append(name)
    return keys


def format_video_path(root: Path, template: str, camera: str, chunk: int, file_idx: int) -> Path:
    return root / template.format(video_key=camera, chunk_index=chunk, file_index=file_idx)
