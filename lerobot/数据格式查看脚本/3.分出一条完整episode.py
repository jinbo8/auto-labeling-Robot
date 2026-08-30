"""把一条完整 episode（一次演示 / 一段视频）的全部对应信息导出到 1episode/。

v3 里多个 episode 拼在同一个 mp4 / parquet 里。本脚本按 episodes 元数据
的时间戳与帧区间切开，方便单独查看。

用法（在仓库根目录）:
  /home/jin/6t/learn/env/autolabel/bin/python \\
      lerobot/数据格式查看脚本/3.分出一条完整episode.py
  /home/jin/6t/learn/env/autolabel/bin/python \\
      lerobot/数据格式查看脚本/3.分出一条完整episode.py --episode 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "lerobot/datasets/svla_so100_pickplace"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "1episode"

VIDEO_KEYS = ("observation.images.top", "observation.images.wrist")


def _to_builtin(val):
    """numpy / pandas 值转成 json / 文本可写的 Python 类型。

    图像 stats 常是 object 数组套 object 数组，tolist() 仍会留下 ndarray，必须递归。
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        if val.ndim == 0:
            return _to_builtin(val.item())
        return [_to_builtin(x) for x in val]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if np.isnan(val):
            return None
        return float(val)
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (list, tuple)):
        return [_to_builtin(x) for x in val]
    if hasattr(val, "item") and np.ndim(val) == 0:
        return _to_builtin(val.item())
    return val


def _fmt(val) -> str:
    v = _to_builtin(val)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _run_ffmpeg(args: list[str]) -> None:
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "ffmpeg 失败:\n"
            + " ".join(args)
            + "\n"
            + (r.stderr or r.stdout or "")
        )


def load_info() -> dict:
    return json.loads((DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))


def load_episode(episode_index: int) -> pd.Series:
    ep_path = DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet"
    df = pd.read_parquet(ep_path)
    hit = df.loc[df.episode_index == episode_index]
    if hit.empty:
        raise SystemExit(
            f"找不到 episode_index={episode_index}，范围是 "
            f"{int(df.episode_index.min())} .. {int(df.episode_index.max())}"
        )
    return hit.iloc[0]


def load_task_text(task_index: int) -> str:
    tasks = pd.read_parquet(DATASET_ROOT / "meta/tasks.parquet")
    if "task_index" in tasks.columns:
        hit = tasks.loc[tasks.task_index == task_index]
        if not hit.empty:
            return str(hit.index[0])
    if len(tasks):
        return str(tasks.index[0])
    return ""


def load_frames(ep: pd.Series) -> pd.DataFrame:
    chunk = int(ep["data/chunk_index"])
    file_idx = int(ep["data/file_index"])
    data_path = DATASET_ROOT / f"data/chunk-{chunk:03d}/file-{file_idx:03d}.parquet"
    frames = pd.read_parquet(data_path)
    lo = int(ep["dataset_from_index"])
    hi = int(ep["dataset_to_index"])
    sub = frames[(frames["index"] >= lo) & (frames["index"] < hi)].copy()
    sub = sub.sort_values("index").reset_index(drop=True)
    return sub, data_path


def cut_video(src: Path, dst: Path, from_ts: float, to_ts: float) -> None:
    """从拼接 mp4 里按秒切开这一段。重编码成 h264，方便直接播放。"""
    duration = max(0.0, float(to_ts) - float(from_ts))
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{from_ts:.6f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.6f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )


def extract_stills(video: Path, out_dir: Path, prefix: str, duration_s: float) -> list[Path]:
    """从切好的片段里抽首 / 中 / 末三帧，方便不播视频也能看画面。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    last_t = max(0.0, duration_s - 1.0 / 30.0)
    specs = [
        ("first", 0.0),
        ("mid", duration_s / 2.0),
        ("last", last_t),
    ]
    written = []
    for name, t in specs:
        dest = out_dir / f"{prefix}_{name}.jpg"
        _run_ffmpeg(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{t:.6f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(dest),
            ]
        )
        written.append(dest)
    return written


def write_episode_meta(path: Path, ep: pd.Series) -> dict:
    meta = {col: _to_builtin(ep[col]) for col in ep.index}
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def write_frames_txt(path: Path, frames: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"n_frames: {len(frames)}\n")
        f.write(f"columns: {frames.columns.tolist()}\n\n")
        for i, row in frames.iterrows():
            f.write(f"--- iloc={i} index={row.get('index', i)} ---\n")
            for col, val in row.items():
                f.write(f"  {col}: {_fmt(val)}\n")
            f.write("\n")


def write_readme(
    path: Path,
    *,
    episode_index: int,
    info: dict,
    ep: pd.Series,
    frames: pd.DataFrame,
    task_text: str,
    video_files: dict[str, dict],
) -> None:
    fps = float(info.get("fps") or 30)
    lo = int(ep["dataset_from_index"])
    hi = int(ep["dataset_to_index"])
    length = int(ep["length"])
    lines = [
        "一条完整 episode 导出（LeRobot Dataset v3）",
        "",
        "原始数据集把多条轨迹拼在同一个 mp4 / parquet 里。",
        "本目录是按 episode 切出来的完整一段：视频 + 每一帧关节数据 + 元信息。",
        "",
        f"dataset: {DATASET_ROOT}",
        f"episode_index: {episode_index}",
        f"task: {task_text}",
        f"task_index: {_to_builtin(frames['task_index'].iloc[0]) if len(frames) else None}",
        f"fps: {fps}",
        f"length: {length} 帧  ≈ {length / fps:.3f} s",
        f"全局帧 index: [{lo}, {hi})  （左闭右开）",
        f"本段 frame_index: 0 .. {length - 1}",
        "",
        "文件说明:",
        "  README.txt            本说明",
        "  info.json             数据集全局 meta/info.json 的拷贝",
        "  episode_meta.json     该 episode 在 meta/episodes 里的全部字段（含 stats/*）",
        "  task.txt              任务文本（来自 meta/tasks.parquet）",
        "  frames.parquet        该 episode 的全部帧（一行一帧）",
        "  frames.txt            同上，纯文本便于通读",
        "  videos/               从拼接 mp4 按时间戳切开的 top / wrist 片段",
        "  sample_frames/        每个相机的首 / 中 / 末帧 JPG",
        "  sources.json          原始文件路径与裁剪时间戳",
        "",
        "视频时间（在原始拼接 mp4 上的秒）:",
    ]
    for cam, rec in video_files.items():
        lines.append(
            f"  {cam}: [{rec['from_timestamp']:.6f}, {rec['to_timestamp']:.6f}] s  "
            f"src={rec['src']}"
        )
    lines += [
        "",
        "对齐关系:",
        "  第 k 帧 (frame_index=k) 的画面 ≈ 切开后视频的 t = k / fps 秒",
        "  parquet 的 timestamp 是相对本 episode 起点的秒，通常等于 frame_index / fps",
        "  action / observation.state 是 6 维关节角（度），顺序见 info.json features.names",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_episode(episode_index: int) -> Path:
    info = load_info()
    ep = load_episode(episode_index)
    frames, data_path = load_frames(ep)
    if len(frames) == 0:
        raise SystemExit(f"episode {episode_index} 在 data parquet 里没有帧")

    task_index = int(frames["task_index"].iloc[0])
    task_text = load_task_text(task_index)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    video_dir = OUT_DIR / "videos"
    still_dir = OUT_DIR / "sample_frames"
    video_dir.mkdir()
    still_dir.mkdir()

    template = info.get("video_path") or (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    video_files: dict[str, dict] = {}
    for cam in VIDEO_KEYS:
        chunk = int(ep[f"videos/{cam}/chunk_index"])
        file_idx = int(ep[f"videos/{cam}/file_index"])
        from_ts = float(ep[f"videos/{cam}/from_timestamp"])
        to_ts = float(ep[f"videos/{cam}/to_timestamp"])
        src = DATASET_ROOT / template.format(
            video_key=cam, chunk_index=chunk, file_index=file_idx
        )
        if not src.is_file():
            raise SystemExit(f"视频不存在: {src}")
        short = cam.replace("observation.images.", "")
        dst = video_dir / f"{short}.mp4"
        print(f"裁剪 {cam}: {from_ts:.3f}s .. {to_ts:.3f}s -> {dst.name}")
        cut_video(src, dst, from_ts, to_ts)
        extract_stills(dst, still_dir, short, to_ts - from_ts)
        video_files[cam] = {
            "src": str(src),
            "dst": str(dst),
            "chunk_index": chunk,
            "file_index": file_idx,
            "from_timestamp": from_ts,
            "to_timestamp": to_ts,
            "duration_s": to_ts - from_ts,
        }

    (OUT_DIR / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_episode_meta(OUT_DIR / "episode_meta.json", ep)
    (OUT_DIR / "task.txt").write_text(task_text + "\n", encoding="utf-8")
    frames.to_parquet(OUT_DIR / "frames.parquet", index=False)
    write_frames_txt(OUT_DIR / "frames.txt", frames)

    sources = {
        "dataset_root": str(DATASET_ROOT),
        "episode_index": episode_index,
        "data_parquet": str(data_path),
        "episodes_parquet": str(DATASET_ROOT / "meta/episodes/chunk-000/file-000.parquet"),
        "tasks_parquet": str(DATASET_ROOT / "meta/tasks.parquet"),
        "dataset_from_index": int(ep["dataset_from_index"]),
        "dataset_to_index": int(ep["dataset_to_index"]),
        "n_frames": len(frames),
        "task": task_text,
        "videos": video_files,
    }
    (OUT_DIR / "sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(
        OUT_DIR / "README.txt",
        episode_index=episode_index,
        info=info,
        ep=ep,
        frames=frames,
        task_text=task_text,
        video_files=video_files,
    )

    print(f"已导出到: {OUT_DIR}")
    print(f"  episode_index={episode_index}  帧数={len(frames)}  任务={task_text!r}")
    for cam, rec in video_files.items():
        print(f"  {cam}: {rec['duration_s']:.3f}s")
    return OUT_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出一条完整 episode 到 1episode/")
    p.add_argument(
        "--episode",
        type=int,
        default=0,
        help="episode_index，默认 0（第一条完整轨迹）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_episode(args.episode)


if __name__ == "__main__":
    main()
