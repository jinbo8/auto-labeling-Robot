#!/usr/bin/env python3
"""下载一份小型 LeRobot v3.0 机器人 VLA 数据到 lerobot/datasets，供本地分析。

默认数据集: lerobot/svla_so100_pickplace
  - SmolVLA 论文使用的 SO-100 抓放示范
  - codebase_version = v3.0（chunked parquet + 按相机分片的 mp4）
  - 50 episodes / 19631 frames / 1 条语言任务 / 双相机（top + wrist）
  - 体积约 0.5–1 GB，适合看格式，不必下 LIBERO（几十 GB）

用法（先激活 conda 环境 autolabel）::

    python lerobot/script/1.download_lerobot_vla_v3.py
    python lerobot/script/1.download_lerobot_vla_v3.py --inspect-only
    python lerobot/script/1.download_lerobot_vla_v3.py --endpoint https://hf-mirror.com

加载已下载数据::

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        "lerobot/svla_so100_pickplace",
        root="lerobot/datasets/svla_so100_pickplace",
    )
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LEROBOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = LEROBOT_DIR / "datasets"
DEFAULT_REPO_ID = "lerobot/svla_so100_pickplace"
DEFAULT_ROOT = DATASETS_DIR / "svla_so100_pickplace"
# 避免写入 ~/.cache/huggingface（可能无权限）；Hub 缓存放在 lerobot 下，不混进数据集目录
os.environ.setdefault("HF_HOME", str(LEROBOT_DIR / ".hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(LEROBOT_DIR / ".hf_cache" / "hub"))


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{n} B"


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _print_tree(root: Path, max_depth: int = 4) -> None:
    print(f"\n目录树 ({root}):")
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        if depth > max_depth:
            dirnames.clear()
            continue
        indent = "  " * depth
        name = "." if rel == Path(".") else rel.name
        print(f"{indent}{name}/")
        if depth == max_depth:
            continue
        for fn in sorted(filenames)[:12]:
            fp = Path(dirpath) / fn
            print(f"{indent}  {fn}  ({_human_bytes(fp.stat().st_size)})")
        if len(filenames) > 12:
            print(f"{indent}  ... ({len(filenames) - 12} more files)")


def download(repo_id: str, root: Path, download_videos: bool) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root.mkdir(parents=True, exist_ok=True)
    print(f"下载 {repo_id}")
    print(f"目标  {root}")
    print(f"视频  {'是' if download_videos else '否（仅 meta/data）'}")
    if os.environ.get("HF_ENDPOINT"):
        print(f"镜像  HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print()

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=download_videos,
    )
    inspect(ds, root)


def inspect_existing(repo_id: str, root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"未找到 {root / 'meta' / 'info.json'}，请先不加 --inspect-only 下载。")
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    inspect(ds, root)


def inspect(ds, root: Path) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))

    print("=" * 60)
    print("LeRobot v3 格式检查")
    print("=" * 60)
    print(f"repo_id           : {ds.repo_id}")
    print(f"codebase_version  : {info.get('codebase_version')}")
    print(f"robot_type        : {info.get('robot_type')}")
    print(f"fps               : {info.get('fps')}")
    print(f"episodes          : {ds.num_episodes}  (meta.total={info.get('total_episodes')})")
    print(f"frames            : {ds.num_frames}")
    print(f"tasks             : {info.get('total_tasks')}")
    print(f"data_path         : {info.get('data_path')}")
    print(f"video_path        : {info.get('video_path')}")
    print(f"本地体积          : {_human_bytes(_dir_size(root))}")

    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        print(f"\n[警告] 期望 v3.x，实际是 {version}")
    else:
        print("\n[通过] codebase_version 为 v3.x")

    print("\nfeatures:")
    for key, spec in ds.features.items():
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        print(f"  - {key:32s}  dtype={dtype}  shape={shape}")

    tasks = getattr(ds.meta, "tasks", None)
    print("\n语言任务 (VLA 的 Language 部分):")
    if tasks is None:
        print("  (无 tasks)")
    else:
        try:
            print(tasks.to_string() if hasattr(tasks, "to_string") else str(tasks))
        except Exception:
            print(tasks)

    print("\n取第 0 帧看一条样本:")
    sample = ds[0]
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:32s}  type={type(v).__name__}  shape={tuple(v.shape)}  dtype={getattr(v, 'dtype', '')}")
        else:
            preview = repr(v)
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  {k:32s}  {preview}")

    _print_tree(root)
    print("\n分析加载示例:")
    print(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print(f"  ds = LeRobotDataset({ds.repo_id!r}, root={str(root)!r})")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载小型 LeRobot v3.0 VLA 数据集到 lerobot/datasets")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face 数据集 id")
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="本地落盘目录（直接是 v3 的 meta/data/videos）",
    )
    p.add_argument("--no-videos", action="store_true", help="不下视频，只看 parquet/meta")
    p.add_argument("--inspect-only", action="store_true", help="已下载则只打印分析，不再拉 Hub")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", ""),
        help="Hugging Face 端点。国内可设 https://hf-mirror.com；默认同环境变量 HF_ENDPOINT",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
    try:
        if args.inspect_only:
            inspect_existing(args.repo_id, root)
        else:
            download(args.repo_id, root, download_videos=not args.no_videos)
    except Exception as e:
        print(f"[失败] {e}", file=sys.stderr)
        if os.environ.get("HF_ENDPOINT"):
            print("当前使用了 HF_ENDPOINT。镜像若不完整，可改为官方源:", file=sys.stderr)
            print("  python lerobot/script/1.download_lerobot_vla_v3.py --endpoint https://huggingface.co", file=sys.stderr)
        else:
            print("若 Hub 超时: python lerobot/script/1.download_lerobot_vla_v3.py --endpoint https://hf-mirror.com", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
