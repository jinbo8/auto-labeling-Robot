"""Orchestrate alignment + quality. Multi-GPU via torch.multiprocessing spawn."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp

from .align import analyze_alignment
from .paths import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_REPORTS_DIR,
    discover_datasets,
    finalize_run_dir,
    make_staging_dir,
)
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


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _progress_bar(pct: float, width: int = 22) -> str:
    pct = min(100.0, max(0.0, pct))
    filled = int(round(width * pct / 100.0))
    return "#" * filled + "-" * (width - filled)


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
    gpu_id = gpu_ids[rank] if gpu_ids else None
    device = torch.device(f"cuda:{gpu_id}" if gpu_ids else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mine = jobs[rank :: max(len(gpu_ids), 1)]
    file_rows = []
    ep_frames = []
    timings: list[dict] = []
    clip_timings: list[dict] = []
    worker_t0 = time.perf_counter()
    nproc = max(len(gpu_ids), 1)
    dataset_clips = sum(len(j["spans"]) for j in jobs)
    worker_clips = sum(len(j["spans"]) for j in mine)
    clip_done = 0

    def on_clip_done(rec: dict, *, camera: str, video_name: str, video_path: str) -> None:
        nonlocal clip_done
        clip_done += 1
        elapsed_wall = time.perf_counter() - worker_t0
        avg = elapsed_wall / clip_done
        worker_total_est = avg * worker_clips if worker_clips else elapsed_wall
        remain = max(0.0, worker_total_est - elapsed_wall)
        # 多卡并行时，全集墙钟 ≈ 本卡剩余工作（负载均分）
        dataset_est = worker_total_est
        pct = 100.0 * clip_done / worker_clips if worker_clips else 100.0
        dataset_pct = min(100.0, 100.0 * clip_done / dataset_clips * nproc) if dataset_clips else pct
        row = {
            "gpu": str(device),
            "gpu_id": gpu_id,
            "rank": rank,
            "episode_index": rec["episode_index"],
            "camera": camera,
            "video_name": video_name,
            "video_path": video_path,
            "elapsed_s": rec["elapsed_s"],
            "n_sampled": rec.get("n_sampled"),
            "clip_done": clip_done,
            "clip_total_gpu": worker_clips,
            "clip_total_dataset": dataset_clips,
            "progress_pct": round(pct, 3),
            "dataset_progress_pct": round(dataset_pct, 3),
        }
        clip_timings.append(row)
        print(
            f"[{device}] [{_progress_bar(dataset_pct)}] {dataset_pct:5.1f}%  "
            f"片段 ep={rec['episode_index']} {camera.rsplit('.', 1)[-1]}  "
            f"{rec['elapsed_s']:.3f}s  "
            f"已用 {_fmt_hms(elapsed_wall)}  "
            f"预估全集 {_fmt_hms(dataset_est)}  "
            f"剩余 {_fmt_hms(remain)}  "
            f"({clip_done}/{worker_clips} 本卡)",
            flush=True,
        )

    for i, job in enumerate(mine):
        vpath = Path(job["video_path"])
        try:
            video_name = str(vpath.relative_to(Path(job["root"]) / "videos"))
        except ValueError:
            video_name = f"{job['camera']}/{vpath.name}"
        print(f"[{device}] 视频 {i+1}/{len(mine)} {video_name}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        summary, ep_df = analyze_video_quality(
            job["video_path"],
            device,
            cfg,
            episode_spans=job["spans"],
            container_fps=job["container_fps"],
            on_clip_done=lambda rec, cam=job["camera"], vn=video_name, vp=job["video_path"]: on_clip_done(
                rec, camera=cam, video_name=vn, video_path=vp
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"[{device}] 视频完成 {video_name}  {elapsed:.3f}s", flush=True)
        summary["dataset"] = job["dataset"]
        summary["camera"] = job["camera"]
        summary["elapsed_s"] = elapsed
        file_rows.append(summary)
        timings.append(
            {
                "gpu": str(device),
                "gpu_id": gpu_id,
                "rank": rank,
                "video_name": video_name,
                "video_path": job["video_path"],
                "camera": job["camera"],
                "dataset": job["dataset"],
                "elapsed_s": round(elapsed, 6),
                "n_decoded": summary.get("n_decoded"),
                "n_sampled": summary.get("n_sampled"),
                "n_clips": len(job["spans"]),
                "error": summary.get("error"),
            }
        )
        if not ep_df.empty:
            ep_df = ep_df.copy()
            ep_df["dataset"] = job["dataset"]
            ep_df["camera"] = job["camera"]
            ep_df["video_path"] = job["video_path"]
            ep_frames.append(ep_df)
    worker_elapsed = time.perf_counter() - worker_t0
    payload = {
        "gpu": str(device),
        "gpu_id": gpu_id,
        "rank": rank,
        "worker_elapsed_s": round(worker_elapsed, 6),
        "n_videos": len(timings),
        "n_clips": len(clip_timings),
        "timings": timings,
        "clip_timings": clip_timings,
        "files": file_rows,
        "episodes": pd.concat(ep_frames, ignore_index=True).to_dict(orient="records") if ep_frames else [],
    }
    Path(shard_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _collect_timings(shard_dir: Path, gpu_ids: list[int]) -> dict:
    shards = sorted(shard_dir.glob("gpu*.json"))
    videos: list[dict] = []
    clips: list[dict] = []
    workers: list[dict] = []
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        videos.extend(payload.get("timings") or [])
        clips.extend(payload.get("clip_timings") or [])
        workers.append(
            {
                "gpu": payload.get("gpu"),
                "gpu_id": payload.get("gpu_id"),
                "rank": payload.get("rank"),
                "n_videos": payload.get("n_videos"),
                "n_clips": payload.get("n_clips"),
                "worker_elapsed_s": payload.get("worker_elapsed_s"),
            }
        )
    by_gpu: dict[str, dict] = {}
    for row in videos:
        key = str(row.get("gpu"))
        bucket = by_gpu.setdefault(key, {"gpu": key, "n_videos": 0, "total_s": 0.0, "videos": []})
        bucket["n_videos"] += 1
        bucket["total_s"] += float(row.get("elapsed_s") or 0)
        bucket["videos"].append({"video_name": row.get("video_name"), "elapsed_s": row.get("elapsed_s")})
    for bucket in by_gpu.values():
        bucket["total_s"] = round(bucket["total_s"], 6)
    return {
        "gpus": gpu_ids,
        "n_videos": len(videos),
        "n_clips": len(clips),
        "videos": videos,
        "clips": clips,
        "workers": workers,
        "by_gpu": by_gpu,
    }


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
    timings: dict | None = None
    if not cfg.align_only:
        jobs = _jobs_from_alignment(root, align)
        print(f"== quality {root.name}: {len(jobs)} video files, gpus={gpu_ids or 'cpu'} ==", flush=True)
        qcfg = QualityConfig(sample_fps=cfg.sample_fps, batch_size=cfg.batch_size)
        quality_files, quality_eps = _run_quality(jobs, gpu_ids, qcfg, out / "shards")
        if not quality_files.empty:
            quality_files.to_parquet(out / "quality_videos.parquet", index=False)
        if not quality_eps.empty:
            quality_eps.to_parquet(out / "quality_episodes.parquet", index=False)
        timings = _collect_timings(out / "shards", gpu_ids)
        (out / "gpu_video_timings.json").write_text(
            json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out / "clip_timings.json").write_text(
            json.dumps(
                {
                    "n_clips": timings.get("n_clips") if timings else 0,
                    "clips": (timings or {}).get("clips") or [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"== timings -> {out / 'gpu_video_timings.json'} , {out / 'clip_timings.json'} ==", flush=True)

    from .visualize import visualize_dataset

    print(f"== visualize {root.name} ==", flush=True)
    visualize_dataset(out)

    report = {
        "alignment": align["summary"],
        "quality_videos": quality_files.to_dict(orient="records") if not quality_files.empty else [],
        "n_quality_episodes": 0 if quality_eps.empty else int(len(quality_eps)),
        "gpu_video_timings": timings,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run(cfg: RunConfig) -> list[dict]:
    roots = discover_datasets(cfg.datasets_dir)
    if not roots:
        raise FileNotFoundError(f"no LeRobot datasets under {cfg.datasets_dir}")
    run_root = cfg.out_dir
    staging = make_staging_dir(run_root)
    cfg.out_dir = staging
    try:
        gpu_ids = _parse_gpus(cfg.gpus)
        reports = []
        for root in roots:
            reports.append(analyze_one(root, cfg, gpu_ids))
        (cfg.out_dir / "summary.json").write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return reports
    finally:
        cfg.out_dir = finalize_run_dir(staging, run_root)


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
            times = [x.get("elapsed_s") for x in qv if x.get("elapsed_s") is not None]
            if times:
                print(f"  耗时  videos={len(times)}  total={sum(times):.2f}s  max={max(times):.2f}s")
    print("=" * 64)
