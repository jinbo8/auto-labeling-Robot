"""Save analysis charts under reports/<dataset>/plots/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.unicode_minus": False,
        }
    )
    for name in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ):
        try:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
        except Exception:
            continue


def _cap(df: pd.DataFrame, n: int = 8000) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=0)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def visualize_dataset(out_dir: Path) -> list[Path]:
    """Build PNG + HTML from parquet already written to ``out_dir``."""
    _setup_style()
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    summary_path = out_dir / "alignment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    align_path = out_dir / "alignment_episodes.parquet"
    q_path = out_dir / "quality_episodes.parquet"
    align = pd.read_parquet(align_path) if align_path.is_file() else pd.DataFrame()
    quality = pd.read_parquet(q_path) if q_path.is_file() else pd.DataFrame()

    if summary:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        labels = ["length", "span", "fps"]
        vals = [summary.get("pct_len_ok") or 0, summary.get("pct_span_ok") or 0, summary.get("pct_fps_ok") or 0]
        colors = ["#2ca02c" if v >= 99 else "#ff7f0e" if v >= 90 else "#d62728" for v in vals]
        ax.bar(labels, vals, color=colors)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% OK")
        ax.set_title(f"{summary.get('dataset', out_dir.name)}  alignment")
        for i, v in enumerate(vals):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
        _save(fig, plots / "01_alignment_overview.png")
        saved.append(plots / "01_alignment_overview.png")

    if not align.empty and "timestamp_fps" in align.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        meta = float(align["meta_fps"].dropna().iloc[0]) if "meta_fps" in align.columns else None
        for cam, g in align.groupby("camera"):
            ax.hist(g["timestamp_fps"].dropna(), bins=40, alpha=0.55, label=str(cam).rsplit(".", 1)[-1])
        if meta:
            ax.axvline(meta, color="k", ls="--", lw=1.2, label=f"meta fps={meta:g}")
        ax.set_xlabel("timestamp fps")
        ax.set_ylabel("episodes")
        ax.set_title("FPS: timestamps vs meta")
        ax.legend(fontsize=8)
        _save(fig, plots / "02_fps_histogram.png")
        saved.append(plots / "02_fps_histogram.png")

    if not align.empty and "frame_delta_parquet_vs_span" in align.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        delta = align["frame_delta_parquet_vs_span"].dropna()
        ax.hist(delta, bins=min(50, max(10, delta.nunique())), color="#1f77b4", alpha=0.85)
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("parquet_frames - expected_from_span")
        ax.set_ylabel("count")
        ax.set_title("Frame-count alignment")
        _save(fig, plots / "03_frame_delta.png")
        saved.append(plots / "03_frame_delta.png")

    if not align.empty and "parquet_frames" in align.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        one = align.drop_duplicates("episode_index") if "episode_index" in align.columns else align
        ax.hist(one["parquet_frames"].dropna(), bins=30, color="#17becf")
        ax.set_xlabel("frames / episode")
        ax.set_ylabel("episodes")
        ax.set_title("Episode length")
        _save(fig, plots / "04_episode_length.png")
        saved.append(plots / "04_episode_length.png")

    if not align.empty and "cross_camera_span_mismatch_s" in align.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        one = align.drop_duplicates("episode_index")
        ax.hist(one["cross_camera_span_mismatch_s"].dropna(), bins=20, color="#9467bd")
        ax.set_xlabel("max camera span mismatch (s)")
        ax.set_ylabel("episodes")
        ax.set_title("Cross-camera time alignment")
        _save(fig, plots / "05_cross_camera.png")
        saved.append(plots / "05_cross_camera.png")

    if not quality.empty:
        q = _cap(quality)
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))
        metrics = [
            ("blur_mean", "blur (laplacian var)"),
            ("brightness_mean", "brightness"),
            ("contrast_mean", "contrast"),
            ("frozen_pct", "frozen %"),
        ]
        for ax, (col, title) in zip(axes.ravel(), metrics, strict=False):
            if col not in q.columns:
                ax.set_visible(False)
                continue
            for cam, g in q.groupby("camera"):
                ax.scatter(
                    g["episode_index"],
                    g[col],
                    s=10,
                    alpha=0.55,
                    label=str(cam).rsplit(".", 1)[-1],
                )
            ax.set_xlabel("episode")
            ax.set_ylabel(title)
            ax.set_title(title)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
        fig.suptitle("Per-episode image quality")
        _save(fig, plots / "06_quality_per_episode.png")
        saved.append(plots / "06_quality_per_episode.png")

        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
        for ax, col in zip(axes, ("blur_mean", "brightness_mean", "contrast_mean"), strict=False):
            if col not in quality.columns:
                continue
            data, names = [], []
            for cam, g in quality.groupby("camera"):
                data.append(g[col].dropna().to_numpy())
                names.append(str(cam).rsplit(".", 1)[-1])
            ax.boxplot(data, tick_labels=names, showfliers=False)
            ax.set_title(col.replace("_mean", ""))
        fig.suptitle("Quality by camera")
        _save(fig, plots / "07_quality_box_by_camera.png")
        saved.append(plots / "07_quality_box_by_camera.png")

        if {"dark_pct", "sat_pct", "frozen_pct", "blurry_pct"} <= set(quality.columns):
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            flags = ["dark_pct", "sat_pct", "frozen_pct", "blurry_pct"]
            cams = list(quality["camera"].unique())
            x = np.arange(len(flags))
            width = 0.8 / max(len(cams), 1)
            for i, cam in enumerate(cams):
                g = quality[quality["camera"] == cam]
                means = [float(g[c].mean()) for c in flags]
                ax.bar(x + i * width, means, width, label=str(cam).rsplit(".", 1)[-1])
            ax.set_xticks(x + width * (len(cams) - 1) / 2)
            ax.set_xticklabels(["dark", "saturated", "frozen", "blurry"])
            ax.set_ylabel("% of sampled frames (mean over episodes)")
            ax.set_title("Quality flags")
            ax.legend(fontsize=8)
            _save(fig, plots / "08_quality_flags.png")
            saved.append(plots / "08_quality_flags.png")

    html = _write_html(out_dir, plots, summary, saved)
    saved.append(html)
    return saved


def _write_html(out_dir: Path, plots: Path, summary: dict, images: list[Path]) -> Path:
    rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in (summary or {}).items()
    )
    imgs = ""
    for p in images:
        if p.suffix.lower() != ".png":
            continue
        rel = p.relative_to(out_dir).as_posix()
        imgs += f'<figure><h3>{p.stem}</h3><img src="{rel}" alt="{p.stem}"/></figure>\n'
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8"/>
  <title>LeRobot v3 report — {out_dir.name}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1100px; margin: 24px auto; color: #222; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    figure {{ margin: 18px 0; }}
  </style>
</head>
<body>
  <h1>LeRobot v3 数据质量报告</h1>
  <p>dataset: <b>{out_dir.name}</b></p>
  <table>{rows}</table>
  {imgs}
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def visualize_reports_root(reports_dir: Path) -> list[Path]:
    """Visualize every dataset subfolder that has alignment parquet."""
    written: list[Path] = []
    if not reports_dir.is_dir():
        return written
    for parquet in sorted(reports_dir.glob("*/alignment_episodes.parquet")):
        saved = visualize_dataset(parquet.parent)
        written.extend(saved)
        print(f"plots -> {parquet.parent / 'plots'}  ({len(saved)} files)", flush=True)
    return written
