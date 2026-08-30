#!/usr/bin/env python3
"""一键清空文件夹内所有内容（保留文件夹本身）。

默认目标: lerobot/run

用法::

    python lerobot/script/3.clear_reports.py
    python lerobot/script/3.clear_reports.py --yes
    python lerobot/script/3.clear_reports.py --path /path/to/dir --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_TARGET = Path(__file__).resolve().parent.parent / "run"


def clear_dir(target: Path) -> int:
    if not target.exists():
        print(f"目录不存在，无需清理: {target}")
        return 0
    if not target.is_dir():
        print(f"[失败] 不是目录: {target}", file=sys.stderr)
        return 1

    n = 0
    for child in target.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
        n += 1
        print(f"  已删 {child.name}")
    print(f"已清空 {n} 项: {target}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="清空文件夹内全部内容，默认 lerobot/run")
    p.add_argument("--path", type=Path, default=DEFAULT_TARGET, help="要清空的目录")
    p.add_argument("--yes", action="store_true", help="非默认目录时必须加此参数确认")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target = args.path.expanduser().resolve()
    default = DEFAULT_TARGET.resolve()

    if target in (Path("/"), Path.home()):
        print(f"[失败] 拒绝清空: {target}", file=sys.stderr)
        return 1

    if target != default and not args.yes:
        print(f"[失败] 非默认目录，请加 --yes 确认: {target}", file=sys.stderr)
        return 1

    print(f"清空: {target}")
    return clear_dir(target)


if __name__ == "__main__":
    raise SystemExit(main())
