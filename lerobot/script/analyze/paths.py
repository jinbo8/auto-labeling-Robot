from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
LEROBOT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASETS_DIR = LEROBOT_DIR / "datasets"
DEFAULT_REPORTS_DIR = SCRIPT_DIR / "reports"


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
