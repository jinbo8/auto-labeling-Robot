"""CPU pass: fps / timestamp / parquet-vs-video frame alignment.

Does not decode pixels. Safe to run on thousands of hours (metadata + container probe).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .paths import format_video_path, load_info, video_keys


@dataclass
class Probe:
    path: str
    width: int | None
    height: int | None
    codec: str | None
    fps: float | None
    duration_s: float | None
    nb_frames: int | None
    error: str | None = None


def probe_video(path: Path) -> Probe:
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        fps = None
        if stream.average_rate:
            fps = float(stream.average_rate)
        elif stream.base_rate:
            fps = float(stream.base_rate)
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        nb = int(stream.frames) if stream.frames else None
        if (not nb) and duration and fps:
            nb = int(round(duration * fps))
        codec = stream.codec_context.name if stream.codec_context else None
        container.close()
        return Probe(
            path=str(path),
            width=stream.codec_context.width if stream.codec_context else None,
            height=stream.codec_context.height if stream.codec_context else None,
            codec=codec,
            fps=fps,
            duration_s=duration,
            nb_frames=nb,
        )
    except Exception as e:
        return Probe(
            path=str(path),
            width=None,
            height=None,
            codec=None,
            fps=None,
            duration_s=None,
            nb_frames=None,
            error=str(e),
        )


def _load_episodes(root: Path) -> pd.DataFrame:
    ep_dir = root / "meta" / "episodes"
    files = sorted(ep_dir.glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet under {ep_dir}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _timestamp_fps_by_episode(root: Path) -> pd.DataFrame:
    """Per-episode timestamp stats from data/*.parquet (columnar, chunk-friendly)."""
    data_dir = root / "data"
    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(columns=["episode_index", "timestamp", "frame_index"])
    frame = table.to_pandas()
    rows = []
    for ep, g in frame.groupby("episode_index", sort=True):
        ts = np.sort(g["timestamp"].to_numpy(dtype=np.float64))
        n = len(ts)
        dts = np.diff(ts) if n > 1 else np.array([], dtype=np.float64)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        fps = float(1.0 / np.median(dts)) if len(dts) else None
        rows.append(
            {
                "episode_index": int(ep),
                "parquet_frames": n,
                "ts_min": float(ts[0]) if n else None,
                "ts_max": float(ts[-1]) if n else None,
                "dt_median": float(np.median(dts)) if len(dts) else None,
                "dt_std": float(np.std(dts)) if len(dts) else None,
                "dt_min": float(dts.min()) if len(dts) else None,
                "dt_max": float(dts.max()) if len(dts) else None,
                "timestamp_fps": fps,
                "gap_count": int((dts > (np.median(dts) * 1.8)).sum()) if len(dts) else 0,
            }
        )
    return pd.DataFrame(rows)


def analyze_alignment(root: Path, fps_atol: float = 0.15, frame_atol: int = 2) -> dict:
    info = load_info(root)
    meta_fps = float(info.get("fps") or 0)
    cams = video_keys(info)
    template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    episodes = _load_episodes(root)
    ts_stats = _timestamp_fps_by_episode(root)
    merged = episodes.merge(ts_stats, on="episode_index", how="left")

    probes: dict[str, Probe] = {}
    episode_rows: list[dict] = []
    file_rows: list[dict] = []

    video_files: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for _, ep in merged.iterrows():
        ep_i = int(ep["episode_index"])
        length = int(ep["length"]) if pd.notna(ep.get("length")) else int(ep.get("parquet_frames") or 0)
        parquet_frames = int(ep["parquet_frames"]) if pd.notna(ep.get("parquet_frames")) else length
        ts_fps = ep.get("timestamp_fps")
        ts_fps = float(ts_fps) if pd.notna(ts_fps) else None

        cam_spans = []
        for cam in cams:
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            from_col = f"videos/{cam}/from_timestamp"
            to_col = f"videos/{cam}/to_timestamp"
            chunk = int(ep[chunk_col]) if chunk_col in ep and pd.notna(ep[chunk_col]) else 0
            file_idx = int(ep[file_col]) if file_col in ep and pd.notna(ep[file_col]) else 0
            from_ts = float(ep[from_col]) if from_col in ep and pd.notna(ep[from_col]) else None
            to_ts = float(ep[to_col]) if to_col in ep and pd.notna(ep[to_col]) else None
            vpath = format_video_path(root, template, cam, chunk, file_idx)
            key = str(vpath)
            if key not in probes:
                probes[key] = probe_video(vpath) if vpath.is_file() else Probe(
                    path=str(vpath), width=None, height=None, codec=None, fps=None,
                    duration_s=None, nb_frames=None, error="missing file",
                )
            span = None if from_ts is None or to_ts is None else max(0.0, to_ts - from_ts)
            expected_frames = int(round(span * meta_fps)) + 1 if span is not None and meta_fps else None
            video_files[(cam, chunk, file_idx)].append(ep_i)
            cam_spans.append(span)
            frame_delta = None if expected_frames is None else parquet_frames - expected_frames
            episode_rows.append(
                {
                    "episode_index": ep_i,
                    "camera": cam,
                    "length_meta": length,
                    "parquet_frames": parquet_frames,
                    "from_timestamp": from_ts,
                    "to_timestamp": to_ts,
                    "span_s": span,
                    "expected_frames_from_span": expected_frames,
                    "frame_delta_parquet_vs_span": frame_delta,
                    "timestamp_fps": ts_fps,
                    "meta_fps": meta_fps,
                    "container_fps": probes[key].fps,
                    "aligned_len": abs(parquet_frames - length) <= frame_atol,
                    "aligned_span": frame_delta is not None and abs(frame_delta) <= frame_atol,
                    "aligned_fps": ts_fps is not None and abs(ts_fps - meta_fps) <= fps_atol,
                    "video_path": key,
                    "video_error": probes[key].error,
                }
            )

        # cross-camera span mismatch
        spans = [s for s in cam_spans if s is not None]
        cam_mismatch = (max(spans) - min(spans)) if len(spans) >= 2 else 0.0
        for row in episode_rows[-len(cams) :]:
            row["cross_camera_span_mismatch_s"] = cam_mismatch

    for (cam, chunk, file_idx), ep_ids in video_files.items():
        vpath = format_video_path(root, template, cam, chunk, file_idx)
        pr = probes[str(vpath)]
        file_rows.append(
            {
                "camera": cam,
                "chunk_index": chunk,
                "file_index": file_idx,
                "n_episodes": len(ep_ids),
                **asdict(pr),
                "meta_fps": meta_fps,
                "fps_ok": pr.fps is not None and abs(pr.fps - meta_fps) <= fps_atol,
            }
        )

    ep_df = pd.DataFrame(episode_rows)
    n = len(ep_df)
    summary = {
        "dataset": root.name,
        "root": str(root),
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "meta_fps": meta_fps,
        "cameras": cams,
        "n_episodes": int(info.get("total_episodes") or merged["episode_index"].nunique()),
        "n_frames_meta": int(info.get("total_frames") or 0),
        "n_align_rows": n,
        "pct_len_ok": float(ep_df["aligned_len"].mean() * 100) if n else None,
        "pct_span_ok": float(ep_df["aligned_span"].mean() * 100) if n else None,
        "pct_fps_ok": float(ep_df["aligned_fps"].mean() * 100) if n else None,
        "max_cross_camera_mismatch_s": float(ep_df["cross_camera_span_mismatch_s"].max()) if n else None,
        "n_video_errors": int(ep_df["video_error"].notna().sum()) if n else 0,
    }
    return {
        "summary": summary,
        "episodes": ep_df,
        "videos": pd.DataFrame(file_rows),
        "probes": probes,
        "info": info,
    }
