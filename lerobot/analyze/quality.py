"""GPU image-quality pass over sampled video frames.

Decode is sequential (AV1 / H264). Metrics run in batches on one CUDA device.
For thousands of hours: keep --sample-fps low (0.5–2) and shard videos across GPUs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
).view(1, 1, 3, 3)


@dataclass
class QualityConfig:
    sample_fps: float = 2.0
    batch_size: int = 32
    dark_thresh: float = 15.0
    sat_thresh: float = 250.0
    frozen_l1: float = 1.5
    blur_thresh: float = 15.0


def _metrics_batch(frames_u8: torch.Tensor) -> dict[str, torch.Tensor]:
    """frames: [B, H, W, 3] uint8 on CUDA or CPU."""
    x = frames_u8.permute(0, 3, 1, 2).float()  # BCHW
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
    kernel = LAPLACIAN.to(device=gray.device, dtype=gray.dtype)
    lap = F.conv2d(gray, kernel, padding=1)
    blur = lap.flatten(1).var(dim=1)
    mean = gray.mean(dim=(1, 2, 3))
    std = gray.std(dim=(1, 2, 3))
    dark = (gray < 15.0).float().mean(dim=(1, 2, 3))
    sat = (gray > 250.0).float().mean(dim=(1, 2, 3))
    return {"blur": blur, "brightness": mean, "contrast": std, "dark": dark, "sat": sat, "gray": gray}


def analyze_video_quality(
    video_path: str,
    device: torch.device,
    cfg: QualityConfig,
    episode_spans: list[tuple[int, float, float]] | None = None,
    container_fps: float | None = None,
    on_clip_done: Callable[[dict], None] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Return (file_summary, per-episode quality dataframe).

    ``on_clip_done`` is called after each episode span in this file is finished.
    """
    episode_spans = episode_spans or []
    acc: dict[int, dict[str, list]] = {}
    clip_elapsed: dict[int, float] = {}
    for ep_i, _, _ in episode_spans:
        acc[ep_i] = {k: [] for k in ("blur", "brightness", "contrast", "dark", "sat", "l1")}

    file_vals: dict[str, list] = {k: [] for k in ("blur", "brightness", "contrast", "dark", "sat", "l1")}
    n_decoded = 0
    n_sampled = 0
    prev_gray = None
    error = None
    width = height = None
    fps = container_fps
    stride = 1
    last_ep: int | None = None
    clip_t0 = time.perf_counter()
    clip_n_sampled = 0

    def finish_clip(ep_i: int) -> None:
        nonlocal clip_t0, clip_n_sampled
        elapsed = time.perf_counter() - clip_t0
        clip_elapsed[ep_i] = clip_elapsed.get(ep_i, 0.0) + elapsed
        rec = {
            "episode_index": ep_i,
            "elapsed_s": round(clip_elapsed[ep_i], 6),
            "n_sampled": clip_n_sampled,
        }
        if on_clip_done is not None:
            on_clip_done(rec)
        clip_t0 = time.perf_counter()
        clip_n_sampled = 0

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        if fps is None and stream.average_rate:
            fps = float(stream.average_rate)
        stride = 1
        if fps and cfg.sample_fps > 0:
            stride = max(1, int(round(fps / cfg.sample_fps)))
        batch_cpu: list[np.ndarray] = []
        batch_t: list[float] = []

        def flush() -> None:
            nonlocal prev_gray, last_ep, clip_n_sampled
            if not batch_cpu:
                return
            arr = np.stack(batch_cpu, axis=0)
            t = torch.from_numpy(arr).to(device, non_blocking=True)
            m = _metrics_batch(t)
            if prev_gray is None:
                l1 = torch.zeros(t.shape[0], device=device)
                l1[1:] = (m["gray"][1:] - m["gray"][:-1]).abs().mean(dim=(1, 2, 3))
            else:
                g = m["gray"]
                first = (g[0] - prev_gray).abs().mean()
                rest = (g[1:] - g[:-1]).abs().mean(dim=(1, 2, 3)) if g.shape[0] > 1 else None
                l1 = torch.cat([first.view(1), rest], dim=0) if rest is not None else first.view(1)
            prev_gray = m["gray"][-1].detach()
            cpu = {k: m[k].detach().float().cpu().numpy() for k in ("blur", "brightness", "contrast", "dark", "sat")}
            l1n = l1.detach().float().cpu().numpy()
            for i, ts in enumerate(batch_t):
                rec = {k: float(cpu[k][i]) for k in cpu}
                rec["l1"] = float(l1n[i])
                for k, v in rec.items():
                    file_vals[k].append(v)
                matched = None
                for ep_i, a, b in episode_spans:
                    if a <= ts < b or (ts >= a and abs(ts - b) < 1e-3):
                        matched = ep_i
                        for k, v in rec.items():
                            acc[ep_i][k].append(v)
                        break
                if matched is not None:
                    if last_ep is not None and matched != last_ep:
                        finish_clip(last_ep)
                    last_ep = matched
                    clip_n_sampled += 1
            batch_cpu.clear()
            batch_t.clear()

        for frame in container.decode(video=0):
            n_decoded += 1
            if (n_decoded - 1) % stride != 0:
                continue
            n_sampled += 1
            rgb = frame.to_ndarray(format="rgb24")
            if width is None:
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
            ts = float(frame.time) if frame.time is not None else (n_decoded - 1) / (fps or 30.0)
            batch_cpu.append(rgb)
            batch_t.append(ts)
            if len(batch_cpu) >= cfg.batch_size:
                flush()
        flush()
        if last_ep is not None:
            finish_clip(last_ep)
        container.close()
    except Exception as e:
        error = str(e)

    def _mean(xs: list[float]) -> float | None:
        return float(np.mean(xs)) if xs else None

    def _pct(xs: list[float], pred) -> float | None:
        if not xs:
            return None
        a = np.asarray(xs)
        return float(np.mean(pred(a)) * 100.0)

    file_summary = {
        "video_path": video_path,
        "n_decoded": n_decoded,
        "n_sampled": n_sampled,
        "stride": stride if fps else None,
        "width": width,
        "height": height,
        "error": error,
        "blur_mean": _mean(file_vals["blur"]),
        "brightness_mean": _mean(file_vals["brightness"]),
        "contrast_mean": _mean(file_vals["contrast"]),
        "dark_pct": _pct(file_vals["dark"], lambda a: a > 0.5),
        "sat_pct": _pct(file_vals["sat"], lambda a: a > 0.3),
        "frozen_pct": _pct(file_vals["l1"], lambda a: a < cfg.frozen_l1),
        "blurry_pct": _pct(file_vals["blur"], lambda a: a < cfg.blur_thresh),
    }

    ep_rows = []
    for ep_i, a, b in episode_spans:
        bkt = acc[ep_i]
        ep_rows.append(
            {
                "episode_index": ep_i,
                "camera_from_ts": a,
                "camera_to_ts": b,
                "n_sampled": len(bkt["blur"]),
                "blur_mean": _mean(bkt["blur"]),
                "brightness_mean": _mean(bkt["brightness"]),
                "contrast_mean": _mean(bkt["contrast"]),
                "dark_pct": _pct(bkt["dark"], lambda a: a > 0.5),
                "sat_pct": _pct(bkt["sat"], lambda a: a > 0.3),
                "frozen_pct": _pct(bkt["l1"], lambda a: a < cfg.frozen_l1),
                "blurry_pct": _pct(bkt["blur"], lambda a: a < cfg.blur_thresh),
                "elapsed_s": clip_elapsed.get(ep_i),
            }
        )
    return file_summary, pd.DataFrame(ep_rows)
