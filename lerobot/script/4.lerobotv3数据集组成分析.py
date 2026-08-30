#!/usr/bin/env python3
"""简单梳理 LeRobot v3 数据集组成（目录、meta、episode、特征、视频）。

默认目标: lerobot/datasets/svla_so100_pickplace

用法::

    conda activate /home/jin/6t/learn/env/autolabel
    python lerobot/script/4.lerobotv3数据集组成分析.py
    python lerobot/script/4.lerobotv3数据集组成分析.py --root lerobot/datasets/xxx
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LEROBOT_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = LEROBOT_DIR / "datasets" / "svla_so100_pickplace"


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024
    return f"{n}B"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _count_files(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.rglob(pattern) if _.is_file())


def _load_info(root: Path) -> dict:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 meta/info.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _load_tasks(root: Path) -> pd.DataFrame:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_data_sample(root: Path, info: dict) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        return pd.DataFrame(), []
    # 只读第一个文件拿 schema；行数用 meta / episodes 更稳
    return pd.read_parquet(files[0]), files


def _feature_table(features: dict) -> list[tuple[str, str, str, str]]:
    rows = []
    for name, spec in (features or {}).items():
        dtype = str(spec.get("dtype", ""))
        shape = spec.get("shape")
        shape_s = "x".join(str(x) for x in shape) if isinstance(shape, list) else str(shape)
        vinfo = spec.get("info") or {}
        names = spec.get("names")
        if dtype == "video" and vinfo:
            extra = (
                f"{vinfo.get('video.width')}x{vinfo.get('video.height')} "
                f"{vinfo.get('video.codec')} @{vinfo.get('video.fps')}fps"
            )
        elif isinstance(names, list):
            extra = ", ".join(str(n) for n in names)
        else:
            extra = ""
        rows.append((name, dtype, shape_s, extra))
    return rows


def _print_section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def analyze(root: Path) -> int:
    root = root.expanduser().resolve()
    if not root.is_dir():
        print(f"[失败] 目录不存在: {root}", file=sys.stderr)
        return 1

    info = _load_info(root)
    episodes = _load_episodes(root)
    tasks = _load_tasks(root)
    data_df, data_files = _load_data_sample(root, info)

    _print_section("1. 基本信息 (meta/info.json)")
    print(f"路径:              {root}")
    print(f"codebase_version:  {info.get('codebase_version')}")
    print(f"robot_type:        {info.get('robot_type')}")
    print(f"fps:               {info.get('fps')}")
    print(f"total_episodes:    {info.get('total_episodes')}")
    print(f"total_frames:      {info.get('total_frames')}")
    print(f"total_tasks:       {info.get('total_tasks')}")
    print(f"chunks_size:       {info.get('chunks_size')}")
    print(f"splits:            {info.get('splits')}")
    print(f"data_path 模板:    {info.get('data_path')}")
    print(f"video_path 模板:   {info.get('video_path')}")

    _print_section("2. 目录与体积")
    parts = [
        ("meta/", root / "meta"),
        ("data/", root / "data"),
        ("videos/", root / "videos"),
    ]
    total = _dir_size(root)
    print(f"{'子目录':<12} {'体积':>10}  说明")
    for name, p in parts:
        sz = _dir_size(p)
        if name == "data/":
            note = f"{_count_files(p, '*.parquet')} parquet"
        elif name == "videos/":
            note = f"{_count_files(p, '*.mp4')} mp4"
        else:
            note = "info / episodes / tasks / stats"
        print(f"{name:<12} {_human_bytes(sz):>10}  {note}")
    print(f"{'合计':<12} {_human_bytes(total):>10}")

    _print_section("3. 特征 (features)")
    print(f"{'名称':<32} {'dtype':<10} {'shape':<12} 细节")
    for name, dtype, shape_s, extra in _feature_table(info.get("features") or {}):
        print(f"{name:<32} {dtype:<10} {shape_s:<12} {extra}")

    _print_section("4. 任务 (meta/tasks.parquet)")
    if tasks.empty:
        print("(无 tasks.parquet)")
    else:
        # v3: 任务文本常在 index，task_index 在列里
        if "task_index" in tasks.columns and tasks.index.name is None and not isinstance(tasks.index, pd.RangeIndex):
            for text, row in tasks.iterrows():
                print(f"  [{int(row['task_index'])}] {text}")
        elif "task" in tasks.columns:
            for _, row in tasks.iterrows():
                idx = row.get("task_index", "?")
                print(f"  [{idx}] {row['task']}")
        else:
            print(tasks.to_string())

    _print_section("5. Episodes (meta/episodes/*.parquet)")
    if episodes.empty:
        print("(无 episodes parquet)")
    else:
        print(f"行数: {len(episodes)}  (info.total_episodes={info.get('total_episodes')})")
        if "length" in episodes.columns:
            lengths = episodes["length"].astype(int)
            print(
                f"每集帧数 length: min={lengths.min()}  mean={lengths.mean():.1f}  "
                f"median={lengths.median():.1f}  max={lengths.max()}  sum={int(lengths.sum())}"
            )
            fps = float(info.get("fps") or 30)
            dur = lengths / fps
            print(
                f"每集时长(~length/fps): min={dur.min():.2f}s  mean={dur.mean():.2f}s  "
                f"max={dur.max():.2f}s  合计={dur.sum()/60:.2f} min"
            )
            # 简易直方图
            bins = [0, 300, 350, 400, 450, 500, 600, 10**9]
            labels = ["<300", "300-349", "350-399", "400-449", "450-499", "500-599", ">=600"]
            cats = pd.cut(lengths, bins=bins, labels=labels, right=False)
            print("帧数分布:")
            for lab, cnt in cats.value_counts().sort_index().items():
                if cnt:
                    print(f"  {lab}: {cnt}")
        if "tasks" in episodes.columns:
            task_texts: list[str] = []
            for val in episodes["tasks"]:
                if isinstance(val, (list, tuple)) or getattr(val, "ndim", 0) == 1:
                    task_texts.extend(str(x) for x in list(val))
                else:
                    task_texts.append(str(val))
            print("episode 任务统计:")
            for text, cnt in Counter(task_texts).most_common():
                print(f"  x{cnt}: {text}")
        # 视频文件归属
        video_keys = [
            k for k, v in (info.get("features") or {}).items() if v.get("dtype") == "video"
        ]
        for cam in video_keys:
            ccol, fcol = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
            if ccol in episodes.columns and fcol in episodes.columns:
                pairs = list(zip(episodes[ccol].tolist(), episodes[fcol].tolist()))
                uniq = sorted(set(pairs))
                print(f"相机 {cam}: 落在 {len(uniq)} 个视频文件 {uniq[:5]}{'...' if len(uniq) > 5 else ''}")

    _print_section("6. 轨迹数据 (data/*.parquet)")
    print(f"parquet 文件数: {len(data_files)}")
    for f in data_files[:8]:
        print(f"  {f.relative_to(root)}  {_human_bytes(f.stat().st_size)}")
    if len(data_files) > 8:
        print(f"  ... 共 {len(data_files)} 个")
    if not data_df.empty:
        print(f"首个文件行数: {len(data_df)}  列: {list(data_df.columns)}")
        for col in data_df.columns:
            s = data_df[col]
            sample = s.iloc[0]
            if hasattr(sample, "shape"):
                print(f"  {col}: dtype={s.dtype}  element_shape={getattr(sample, 'shape', None)}")
            else:
                print(f"  {col}: dtype={s.dtype}  sample={sample}")

    _print_section("7. 视频 (videos/)")
    videos_root = root / "videos"
    if not videos_root.is_dir():
        print("(无 videos/)")
    else:
        for cam_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
            mp4s = sorted(cam_dir.rglob("*.mp4"))
            sz = sum(p.stat().st_size for p in mp4s)
            print(f"{cam_dir.name}: {len(mp4s)} 文件  {_human_bytes(sz)}")
            for p in mp4s[:5]:
                print(f"  {p.relative_to(root)}  {_human_bytes(p.stat().st_size)}")
            if len(mp4s) > 5:
                print(f"  ... 共 {len(mp4s)} 个")

    _print_section("组成小结")
    n_ep = info.get("total_episodes")
    n_fr = info.get("total_frames")
    cams = [k for k, v in (info.get("features") or {}).items() if v.get("dtype") == "video"]
    print(
        f"{root.name}: {info.get('codebase_version')} / {info.get('robot_type')} / "
        f"{n_ep} episodes / {n_fr} frames / {info.get('fps')} fps / "
        f"{len(cams)} 相机 ({', '.join(cams)}) / 体积 {_human_bytes(total)}"
    )
    print("目录角色: meta=索引与统计  data=关节状态与动作轨迹  videos=按相机分片的 mp4")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeRobot v3 数据集组成速览")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="数据集根目录")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return analyze(args.root)
    except Exception as e:
        print(f"[失败] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
