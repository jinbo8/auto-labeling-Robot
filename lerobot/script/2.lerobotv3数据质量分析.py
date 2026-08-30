#!/usr/bin/env python3
"""分析 lerobot/datasets 下的 LeRobot v3 数据。

覆盖：
  - 视频 / parquet / 时间戳 帧数对齐
  - 标称 fps vs 容器 fps vs 时间戳推算 fps
  - 图像质量（模糊、过暗、过曝、卡死帧），GPU 批量卷积
  - 每个 GPU / 每个视频 / 每个片段(episode) 的分析耗时

按视频文件分片，可水平扩展到数千小时；单机多卡用 --gpus。

用法::

    conda activate /home/jin/6t/learn/env/autolabel
    python lerobot/script/2.lerobotv3数据质量分析.py
    python lerobot/script/2.lerobotv3数据质量分析.py --align-only
    python lerobot/script/2.lerobotv3数据质量分析.py --gpus 0,1 --sample-fps 2
    python lerobot/script/2.lerobotv3数据质量分析.py --viz-only

结果默认写到 lerobot/run/<结束时间>/ 。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze.paths import DEFAULT_DATASETS_DIR, DEFAULT_REPORTS_DIR, resolve_run_dir_for_viz
from analyze.run import RunConfig, print_report, run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeRobot v3 对齐 / 帧率 / 图像质量分析")
    p.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help="数据集根目录（每个子目录一份 v3，含 meta/info.json）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="结果根目录（默认 lerobot/run）；每次运行在其下新建「结束时间」文件夹",
    )
    p.add_argument("--sample-fps", type=float, default=2.0, help="质量分析采样帧率，小时级数据用 0.5–2")
    p.add_argument("--batch-size", type=int, default=32, help="GPU 质量 batch")
    p.add_argument("--gpus", default="auto", help="auto | cpu | 0,1,2")
    p.add_argument("--align-only", action="store_true", help="只做对齐/帧率，不解码像素")
    p.add_argument("--fps-atol", type=float, default=0.15, help="fps 对齐容差")
    p.add_argument("--frame-atol", type=int, default=2, help="帧数对齐容差")
    p.add_argument(
        "--viz-only",
        action="store_true",
        help="不重跑分析，只根据 --out 下已有结果出图（根目录时取最近一次运行）",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(
        datasets_dir=args.datasets_dir.expanduser().resolve(),
        out_dir=args.out.expanduser().resolve(),
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        align_only=args.align_only,
        gpus=args.gpus,
        fps_atol=args.fps_atol,
        frame_atol=args.frame_atol,
    )
    print(f"datasets: {cfg.datasets_dir}")
    print(f"out:      {cfg.out_dir}")
    try:
        if args.viz_only:
            from analyze.visualize import visualize_reports_root

            viz_dir = resolve_run_dir_for_viz(cfg.out_dir)
            visualize_reports_root(viz_dir)
            print(f"\n报告目录: {viz_dir}")
            return 0
        reports = run(cfg)
    except Exception as e:
        print(f"[失败] {e}", file=sys.stderr)
        print(f"报告目录: {cfg.out_dir}", file=sys.stderr)
        return 1
    print_report(reports)
    print(f"\n报告目录: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
