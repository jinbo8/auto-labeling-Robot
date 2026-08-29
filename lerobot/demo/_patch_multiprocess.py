"""Python 3.12.0 + multiprocess 0.70.x：进程退出时 ResourceTracker 会误报。

CPython 较新版本的 resource_tracker 会调用 RLock._recursion_count()，
但 3.12.0 的 `_thread.RLock` 没有这个方法。在 import lerobot 之前先打补丁。
"""

from __future__ import annotations

import os


def apply() -> None:
    try:
        from multiprocess.resource_tracker import ResourceTracker
    except ImportError:
        return

    def _stop_locked(
        self,
        close=os.close,
        waitpid=os.waitpid,
        waitstatus_to_exitcode=os.waitstatus_to_exitcode,
    ):
        rec = getattr(self._lock, "_recursion_count", None)
        if rec is not None and rec() > 1:
            return self._reentrant_call_error()
        if self._fd is None or self._pid is None:
            return
        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None

    def _safe_del(self):
        try:
            self._stop(use_blocking_lock=False)
        except Exception:
            pass

    ResourceTracker._stop_locked = _stop_locked
    ResourceTracker.__del__ = _safe_del


apply()
