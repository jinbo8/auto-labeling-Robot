"""Orchestrate alignment + quality. Multi-GPU via torch.multiprocessing spawn."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp

from .align import analyze_alignment
from .paths import DEFAULT_DATASETS_DIR, DEFAULT_REPORTS_DIR, discover_datasets
from .quality import QualityConfig, analyze_video_quality


@dataclass
class RunConfig:
    datasets_dir: Path = DEFAULT_DATASETS_DIR
    out_dir: Path = DEFAULT_REPORTS_DIR
    sample_fps: float = 2.0
    batch_size: int = 32
    align_only: bool = False
    gpus: str = "auto"
    fps_atol: float = 0.15
    frame_atol: int = 2


def _parse_gpus(spec: str) -> list[int]:
    if spec == "cpu" or not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    if spec in ("auto", "all", ""):
        return list(range(n))
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return [i for i in ids if 0 <= i < n]


def _jobs_from_alignment(root: Path, align: dict) -> list[dict]:
    info = align["info"]
    template = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    ep = align["episodes"]
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for _, row in ep.iterrows():
        grouped[(row["camera"], row["video_path"])].append(
            (int(row["episode_index"]), float(row["from_timestamp"]), float(row["to_timestamp"]))
        )
    jobs = []
    for (cam, vpath), spans in grouped.items():
        jobs.append(
            {
                "dataset": root.name,
                "root": str(root),
                "camera": cam,
                "video_path": vpath,
                "spans": spans,
                "container_fps": align["probes"].get(vpath).fps if vpath in align["probes"] else None,
            }
        )
    return jobs


def _quality_worker(rank: int, gpu_ids: list[int], jobs: list[dict], cfg: QualityConfig, shard_path: str) -> None:
    device = torch.device(f"cuda:{gpu_ids[rank]}" if gpu_ids else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mine = jobs[rank :: max(len(gpu_ids), 1)]
    file_rows = []
    ep_frames = []
    for i, job in enumerate(mine):
        print(f"[gpu{device}] {i+1}/{len(mine)} {job['camera']} {Path(job['video_path']).name}", flush=True)
        summary, ep_df = analyze_video_quality(
            job["video_path"],
            device,
            cfg,
            episode_spans=job["spans"],
            container_fps=job["container_fps"],
        )
        summary["dataset"] = job["dataset"]
        summary["camera"] = job["camera"]
        file_rows.append(summary)
        if not ep_df.empty:
            ep_df = ep_df.copy()
            ep_df["dataset"] = job["dataset"]
            ep_df["camera"] = job["camera"]
            ep_df["video_path"] = job["video_path"]
            ep_frames.append(ep_df)
    payload = {
        "files": file_rows,
        "episodes": pd.concat(ep_frames, ignore_index=True).to_dict(orient="records") if ep_frames else [],
    }
    Path(shard_path).write_text(json.dumps(payload), encoding="utf-8")


def _run_quality(jobs: list[dict], gpu_ids: list[int], qcfg: QualityConfig, shard_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    nproc = max(len(gpu_ids), 1)
    if nproc == 1:
        shard = shard_dir / "gpu0.json"
        _quality_worker(0, gpu_ids, jobs, qcfg, str(shard))
        shards = [shard]
    else:
        ctx = mp.get_context("spawn")
        procs = []
        shards = []
        for rank in range(nproc):
            shard = shard_dir / f"gpu{rank}.json"
            shards.append(shard)
            p = ctx.Process(target=_quality_worker, args=(rank, gpu_ids, jobs, qcfg, str(shard)))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"quality worker exited {p.exitcode}")

    files, eps = [], []
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        files.extend(payload["files"])
        eps.extend(payload["episodes"])
    return pd.DataFrame(files), pd.DataFrame(eps)


def analyze_one(root: Path, cfg: RunConfig, gpu_ids: list[int]) -> dict:
    print(f"== align {root.name} ==", flush=True)
    align = analyze_alignment(root, fps_atol=cfg.fps_atol, frame_atol=cfg.frame_atol)
    out = cfg.out_dir / root.name
    out.mkdir(parents=True, exist_ok=True)
    align["episodes"].to_parquet(out / "alignment_episodes.parquet", index=False)
    align["videos"].to_parquet(out / "alignment_videos.parquet", index=False)
    (out / "alignment_summary.json").write_text(
        json.dumps(align["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    quality_files = pd.DataFrame()
    quality_eps = pd.DataFrame()
    if not cfg.align_only:
        jobs = _jobs_from_alignment(root, align)
        print(f"== quality {root.name}: {len(jobs)} video files, gpus={gpu_ids or 'cpu'} ==", flush=True)
        qcfg = QualityConfig(sample_fps=cfg.sample_fps, batch_size=cfg.batch_size)
        quality_files, quality_eps = _run_quality(jobs, gpu_ids, qcfg, out / "shards")
        if not quality_files.empty:
            quality_files.to_parquet(out / "quality_videos.parquet", index=False)
        if not quality_eps.empty:
            quality_eps.to_parquet(out / "quality_episodes.parquet", index=False)

    from .visualize import visualize_dataset

    print(f"== visualize {root.name} ==", flush=True)
    visualize_dataset(out)

    report = {
        "alignment": align["summary"],
        "quality_videos": quality_files.to_dict(orient="records") if not quality_files.empty else [],
        "n_quality_episodes": 0 if quality_eps.empty else int(len(quality_eps)),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(cfg: RunConfig) -> list[dict]:
    roots = discover_datasets(cfg.datasets_dir)
    if not roots:
        raise FileNotFoundError(f"no LeRobot datasets under {cfg.datasets_dir}")
    gpu_ids = _parse_gpus(cfg.gpus)
    reports = []
    for root in roots:
        reports.append(analyze_one(root, cfg, gpu_ids))
    (cfg.out_dir / "summary.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    return reports


def print_report(reports: list[dict]) -> None:
    print("\n" + "=" * 64)
    print("LeRobot v3 分析汇总")
    print("=" * 64)
    for r in reports:
        a = r["alignment"]
        print(f"\n[{a['dataset']}] version={a.get('codebase_version')} fps={a.get('meta_fps')} cameras={a.get('cameras')}")
        print(f"  episodes={a.get('n_episodes')} frames={a.get('n_frames_meta')}")
        print(f"  对齐  length={a.get('pct_len_ok'):.1f}%  span={a.get('pct_span_ok'):.1f}%  fps={a.get('pct_fps_ok'):.1f}%")
        print(f"  跨相机时间差 max={a.get('max_cross_camera_mismatch_s')}s  video_errors={a.get('n_video_errors')}")
        qv = r.get("quality_videos") or []
        if qv:
            blur = [x.get("blur_mean") for x in qv if x.get("blur_mean") is not None]
            frozen = [x.get("frozen_pct") for x in qv if x.get("frozen_pct") is not None]
            print(f"  质量  files={len(qv)}  blur_mean={sum(blur)/len(blur):.2f}  frozen%={sum(frozen)/len(frozen):.2f}")
    print("=" * 64)
