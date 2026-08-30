from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class EpisodeInfo:
    episode_index: int
    length: int
    duration_s: float
    task_index: int | None
    task_text: str | None
    video_refs: dict[str, Any] = field(default_factory=dict)
    data_from_index: int | None = None
    data_to_index: int | None = None


@dataclass
class DatasetIndex:
    root: str
    codebase_version: str | None
    robot_type: str | None
    fps: float
    total_episodes: int
    total_frames: int
    camera_keys: list[str]
    tasks: list[dict[str, Any]]
    episodes: list[EpisodeInfo]
    meta_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _video_keys(features: dict) -> list[str]:
    keys = []
    for name, spec in (features or {}).items():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            keys.append(name)
    return keys


def _load_tasks(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        return []
    df = pd.read_parquet(path)
    out: list[dict[str, Any]] = []
    # v3 often uses task text as index
    if "task_index" in df.columns:
        for text, row in df.iterrows():
            out.append({"task_index": int(row["task_index"]), "task_text": str(text)})
    elif "task" in df.columns:
        for _, row in df.iterrows():
            out.append(
                {
                    "task_index": int(row.get("task_index", 0)),
                    "task_text": str(row["task"]),
                }
            )
    else:
        for i, (idx, _) in enumerate(df.iterrows()):
            out.append({"task_index": i, "task_text": str(idx)})
    return out


def _task_text_from_cell(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        return str(val[0]) if val else None
    if hasattr(val, "tolist"):
        try:
            lst = list(val.tolist())
            return str(lst[0]) if lst else None
        except Exception:
            return str(val)
    return str(val)


def index_lerobot_v3(root: str | Path) -> DatasetIndex:
    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing meta/info.json under {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version") or "")
    if not version.startswith("v3"):
        raise ValueError(f"expected LeRobot v3, got codebase_version={version!r}")

    fps = float(info.get("fps") or 30)
    cameras = _video_keys(info.get("features") or {})
    tasks = _load_tasks(root)
    task_by_text = {t["task_text"]: t["task_index"] for t in tasks}

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"no episodes parquet under {root / 'meta' / 'episodes'}")
    ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)

    episodes: list[EpisodeInfo] = []
    for _, row in ep_df.iterrows():
        ei = int(row["episode_index"])
        length = int(row["length"]) if "length" in row and pd.notna(row["length"]) else 0
        task_text = _task_text_from_cell(row["tasks"]) if "tasks" in row else None
        task_index = task_by_text.get(task_text) if task_text else None
        video_refs: dict[str, Any] = {}
        for cam in cameras:
            ccol, fcol = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
            from_c, to_c = f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"
            if ccol in row and fcol in row and pd.notna(row[ccol]):
                chunk, file_idx = int(row[ccol]), int(row[fcol])
                rel = f"videos/{cam}/chunk-{chunk:03d}/file-{file_idx:03d}.mp4"
                video_refs[cam] = {
                    "chunk_index": chunk,
                    "file_index": file_idx,
                    "path": rel,
                    "from_timestamp": float(row[from_c]) if from_c in row and pd.notna(row[from_c]) else None,
                    "to_timestamp": float(row[to_c]) if to_c in row and pd.notna(row[to_c]) else None,
                    "abs_path": str(root / rel),
                }
        episodes.append(
            EpisodeInfo(
                episode_index=ei,
                length=length,
                duration_s=round(length / fps, 4) if fps else 0.0,
                task_index=task_index,
                task_text=task_text,
                video_refs=video_refs,
                data_from_index=int(row["dataset_from_index"])
                if "dataset_from_index" in row and pd.notna(row["dataset_from_index"])
                else None,
                data_to_index=int(row["dataset_to_index"])
                if "dataset_to_index" in row and pd.notna(row["dataset_to_index"])
                else None,
            )
        )
    episodes.sort(key=lambda e: e.episode_index)

    snapshot = {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": fps,
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "splits": info.get("splits"),
        "cameras": cameras,
    }
    return DatasetIndex(
        root=str(root),
        codebase_version=info.get("codebase_version"),
        robot_type=info.get("robot_type"),
        fps=fps,
        total_episodes=int(info.get("total_episodes") or len(episodes)),
        total_frames=int(info.get("total_frames") or sum(e.length for e in episodes)),
        camera_keys=cameras,
        tasks=tasks,
        episodes=episodes,
        meta_snapshot=snapshot,
    )
