#!/usr/bin/env python3
"""下载 Hugging Face 数据集 BAAI/ShareRobot 到指定目录。

体积约 351 GB。官方源超时或偏慢时自动改用 hf-mirror。
Clash 的 7891 是 SOCKS，不能当 HTTP 代理。镜像直连 403 视为可达（反爬）。
真正的 socks5:// 留给官方源。中断后重跑会续传，并周期性打印总体进度。

用法（先激活 conda 环境 autolabel）::

    python lerobot/script/5.download_sharerobot.py
    python lerobot/script/5.download_sharerobot.py --endpoint mirror
    python lerobot/script/5.download_sharerobot.py --proxy http://127.0.0.1:7890
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

# 必须在 import huggingface_hub 之前：xet 走独立连接，会绕过代理并触发 Event 报错
os.environ["HF_HUB_DISABLE_XET"] = "1"

DEFAULT_REPO_ID = "BAAI/ShareRobot"
DEFAULT_LOCAL_DIR = Path("/home/jin/6t/item/hf")
OFFICIAL = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"
SLOW_SECONDS = 3.0
PROBE_TIMEOUT = 20.0
PROXY_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "SOCKS_PROXY",
    "SOCKS5_PROXY",
    "socks_proxy",
    "socks5_proxy",
)


def _human_bytes(n: int | float) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{n} B"


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


_ORIG_GETADDRINFO = socket.getaddrinfo


def force_ipv4() -> None:
    """探测时避免先走 IPv6 卡住；下载前会 restore，以免 httpx 握手失败。"""

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _ORIG_GETADDRINFO(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_only  # type: ignore[assignment]


def restore_getaddrinfo() -> None:
    socket.getaddrinfo = _ORIG_GETADDRINFO


def _host_port(url: str) -> tuple[str, int] | None:
    raw = url.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, int(port)


def _is_socks(url: str) -> bool:
    return url.lower().startswith(("socks://", "socks4://", "socks5://", "socks5h://"))


def _is_http_proxy(url: str) -> bool:
    return url.lower().startswith(("http://", "https://")) and not _is_socks(url)


def normalize_socks5(url: str) -> str:
    hp = _host_port(url)
    if not hp:
        return url.rstrip("/")
    return f"socks5://{hp[0]}:{hp[1]}"


def proxy_candidates(explicit: str | None) -> list[str | None]:
    """直连优先；保留真正的 socks5://；不要把 7891 伪装成 HTTP（会 SSL EOF）。"""
    ordered: list[str | None] = []
    seen: set[str] = set()

    def add(item: str | None) -> None:
        key = "direct" if item is None else item
        if key in seen:
            return
        seen.add(key)
        ordered.append(item)

    add(None)
    if explicit:
        val = explicit.rstrip("/")
        add(normalize_socks5(val) if _is_socks(val) else val)

    for key in PROXY_KEYS:
        val = os.environ.get(key) or ""
        if _is_socks(val):
            add(normalize_socks5(val))
        elif _is_http_proxy(val):
            hp = _host_port(val)
            if hp:
                add(f"http://{hp[0]}:{hp[1]}")

    add("http://127.0.0.1:7890")
    return ordered


def strip_socks_from_env() -> list[str]:
    removed = []
    for key in PROXY_KEYS:
        val = os.environ.get(key)
        if val and _is_socks(val):
            removed.append(f"{key}={val}")
            del os.environ[key]
    return removed


def apply_proxy(proxy: str | None) -> None:
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    if not proxy:
        return
    if _is_socks(proxy):
        socks_url = normalize_socks5(proxy)
        for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = socks_url
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = proxy


def configure_hf_http(proxy: str | None) -> None:
    """让 huggingface_hub 的 httpx 显式走探测成功的代理，并关闭 HTTP/2。"""
    import httpx
    from huggingface_hub.utils._http import hf_request_event_hook, set_client_factory

    proxy_url = None
    if proxy:
        proxy_url = normalize_socks5(proxy) if _is_socks(proxy) else proxy

    def factory() -> httpx.Client:
        kw: dict = {
            "event_hooks": {"request": [hf_request_event_hook]},
            "follow_redirects": True,
            "timeout": httpx.Timeout(120.0, connect=30.0),
            "trust_env": False,
        }
        if proxy_url:
            kw["proxy"] = proxy_url
        try:
            return httpx.Client(http2=False, **kw)
        except TypeError:
            return httpx.Client(**kw)

    set_client_factory(factory)
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    if not proxy:
        return
    if _is_socks(proxy):
        socks_url = normalize_socks5(proxy)
        # httpx + socksio 认 socks5://，不认 socks://
        for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = socks_url
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = proxy


def _opener_for_proxy(proxy: str | None):
    if proxy and _is_socks(proxy):
        import socks
        from sockshandler import SocksiPyHandler

        hp = _host_port(proxy)
        if not hp:
            raise RuntimeError(f"无法解析 SOCKS 代理: {proxy}")
        return urllib.request.build_opener(SocksiPyHandler(socks.SOCKS5, hp[0], hp[1]))
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _probe(endpoint: str, repo_id: str, timeout: float, proxy: str | None) -> tuple[float | None, str]:
    # 用 Hub API，避免 /resolve/README 被镜像 403 反爬误判成失败
    url = f"{endpoint}/api/datasets/{repo_id}"
    headers = {"User-Agent": "huggingface_hub/1.17.0; hf_hub; python/3.12"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    t0 = time.perf_counter()
    try:
        opener = _opener_for_proxy(proxy)
        with opener.open(req, timeout=timeout) as resp:
            resp.read(256)
            elapsed = time.perf_counter() - t0
            return elapsed, f"HTTP {getattr(resp, 'status', 200)} / {elapsed:.2f}s"
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        # 403：镜像反爬，主机已通，huggingface_hub 仍可下载
        if e.code in (401, 403, 429):
            return elapsed, f"HTTP {e.code}（主机可达） / {elapsed:.2f}s"
        return None, f"HTTP {e.code}"
    except Exception as e:
        msg = re.sub(r"\s+", " ", str(e))
        if len(msg) > 120:
            msg = msg[:117] + "..."
        return None, f"{type(e).__name__}: {msg}"


def pick_route(
    repo_id: str,
    mode: str,
    slow_seconds: float,
    proxies: list[str | None],
) -> tuple[str, str | None]:
    endpoints = [OFFICIAL, MIRROR] if mode == "auto" else (
        [OFFICIAL] if mode == "official" else [MIRROR]
    )
    print("探测下载路径（直连 / SOCKS5 / HTTP × 官方 / 镜像，并行）...")
    jobs = [(ep, px) for ep in endpoints for px in proxies]
    results: list[tuple[float, str, str | None]] = []

    def _job(ep: str, px: str | None) -> tuple[str, str | None, float | None, str]:
        t, msg = _probe(ep, repo_id, timeout=PROBE_TIMEOUT, proxy=px)
        return ep, px, t, msg

    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as pool:
        futs = [pool.submit(_job, ep, px) for ep, px in jobs]
        for fut in as_completed(futs):
            endpoint, proxy, t, msg = fut.result()
            print(f"  {endpoint}  via {proxy or '直连'}: {msg}", flush=True)
            if t is not None:
                results.append((t, endpoint, proxy))

    if not results:
        raise RuntimeError(
            "官方源和镜像都连不上。镜像直连若是 403 应已视为成功；"
            "请检查网络，或加 --proxy socks5://127.0.0.1:7891。"
        )

    mirror_ok = [r for r in results if r[1] == MIRROR]
    official_ok = [r for r in results if r[1] == OFFICIAL]
    mirror_direct = [r for r in mirror_ok if r[2] is None]
    if mode == "auto" and mirror_direct:
        best = min(mirror_direct, key=lambda x: x[0])
        print(f"镜像直连可达，使用 {MIRROR}（不走代理）")
        return best[1], best[2]
    if mode == "auto" and mirror_ok:
        best = min(mirror_ok, key=lambda x: x[0])
        print(f"选用镜像 {MIRROR}  via {best[2] or '直连'}  ({best[0]:.2f}s)")
        return best[1], best[2]
    if official_ok:
        best = min(official_ok, key=lambda x: x[0])
        if mode == "auto" and best[0] > slow_seconds:
            print("官方源偏慢且镜像不可用，仍使用官方源")
        print(f"选用 {best[1]}  via {best[2] or '直连'}  ({best[0]:.2f}s)")
        return best[1], best[2]
    best = min(results, key=lambda x: x[0])
    print(f"选用 {best[1]}  via {best[2] or '直连'}  ({best[0]:.2f}s)")
    return best[1], best[2]


def repo_stats(repo_id: str, endpoint: str) -> tuple[int, int]:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint)
    n_files = 0
    total = 0
    for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        size = getattr(item, "size", None)
        if size is None:
            continue
        n_files += 1
        total += int(size)
    return n_files, total


class SizeMonitor(threading.Thread):
    """每几秒打印一次目录体积，作为总体进度（tqdm 是按文件的）。"""

    def __init__(self, path: Path, total_bytes: int | None, interval: float = 5.0):
        super().__init__(daemon=True)
        self.path = path
        self.total_bytes = total_bytes
        self.interval = interval
        self._stop = threading.Event()
        self.size0 = _dir_size(path)
        self.t0 = time.perf_counter()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self._print()

    def _print(self) -> None:
        now = _dir_size(self.path)
        elapsed = max(time.perf_counter() - self.t0, 1e-6)
        gained = max(0, now - self.size0)
        speed = gained / elapsed
        pct = ""
        if self.total_bytes and self.total_bytes > 0:
            pct = f"  {100.0 * now / self.total_bytes:.1f}%"
            rest = max(0, self.total_bytes - now)
            eta = rest / speed if speed > 1 else float("inf")
            eta_s = "未知" if eta == float("inf") else time.strftime("%H:%M:%S", time.gmtime(min(eta, 99 * 3600)))
            extra = f"  剩余约 {_human_bytes(rest)}  ETA {eta_s}"
        else:
            extra = ""
        print(
            f"[进度] 目录 {_human_bytes(now)}{pct}  "
            f"本次 +{_human_bytes(gained)}  { _human_bytes(speed)}/s{extra}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        self._print()


def download(
    repo_id: str,
    local_dir: Path,
    endpoint: str,
    proxy: str | None,
    max_workers: int,
) -> None:
    os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    restore_getaddrinfo()
    apply_proxy(proxy)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
    except ImportError:
        print("需要 huggingface_hub: pip install -U huggingface_hub", file=sys.stderr)
        raise SystemExit(1)

    configure_hf_http(proxy)
    local_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"仓库     {repo_id}")
    print(f"目标     {local_dir}")
    print(f"端点     {os.environ['HF_ENDPOINT']}")
    print(f"代理     {proxy or '直连'}")
    print(f"已有体积 {_human_bytes(_dir_size(local_dir))}（续传会跳过已下完的文件）")
    print(f"并发     {max_workers}")

    n_files, total = 0, 0
    try:
        n_files, total = repo_stats(repo_id, os.environ["HF_ENDPOINT"])
        print(f"远端     {n_files} 个文件, 约 {_human_bytes(total)}")
    except Exception as e:
        print(f"远端文件列表暂不可用（不影响下载）: {e}")
        total = 0
    print()

    monitor = SizeMonitor(local_dir, total_bytes=total or None, interval=5.0)
    monitor.start()
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            endpoint=os.environ["HF_ENDPOINT"],
            max_workers=max_workers,
            tqdm_class=hf_tqdm,
        )
    finally:
        monitor.stop()
        monitor.join(timeout=2)

    print()
    print(f"完成。目录体积: {_human_bytes(_dir_size(local_dir))}")
    print(f"路径: {local_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载 BAAI/ShareRobot 到指定文件夹")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    p.add_argument(
        "--endpoint",
        choices=("auto", "official", "mirror"),
        default="auto",
        help="auto=测速后选择；官方慢或失败则用 hf-mirror",
    )
    p.add_argument(
        "--proxy",
        default="",
        help="代理 URL。HTTP 如 http://127.0.0.1:7890；SOCKS 如 socks5://127.0.0.1:7891",
    )
    p.add_argument(
        "--slow-seconds",
        type=float,
        default=SLOW_SECONDS,
        help="官方源探测超过该秒数则切镜像，默认 3",
    )
    p.add_argument("--max-workers", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    force_ipv4()
    proxies = proxy_candidates(args.proxy or None)
    removed = strip_socks_from_env()
    if removed:
        print("探测阶段暂时清掉 socks://，避免 urllib 报 Unknown scheme:")
        for item in removed:
            print(f"  {item}")
        print("若镜像直连可用则不走代理；官方源会用 socks5://")

    try:
        endpoint, proxy = pick_route(
            args.repo_id,
            args.endpoint,
            args.slow_seconds,
            proxies,
        )
        download(
            args.repo_id,
            args.local_dir.expanduser().resolve(),
            endpoint,
            proxy,
            args.max_workers,
        )
    except KeyboardInterrupt:
        print("\n已中断。再次运行同一命令会续传。", file=sys.stderr)
        return 130
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[失败] {e}", file=sys.stderr)
        print(
            "可手动指定: python lerobot/script/5.download_sharerobot.py "
            "--endpoint official --proxy http://127.0.0.1:7891",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
