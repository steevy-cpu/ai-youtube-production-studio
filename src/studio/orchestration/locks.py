"""Named locks stored in SQLite and kept alive by a heartbeat thread.

Used for "only one worker" and "only one run per project". Liveness is judged by the
heartbeat age, not by process ids, so it behaves the same on macOS, Linux and Windows and a
crashed process simply stops refreshing its lock.
"""
from __future__ import annotations

import threading
from pathlib import Path

from studio.memory import MemoryStore
from studio.utils import new_id

HEARTBEAT_S = 5.0
STALE_AFTER_S = 30.0


class LockBusy(Exception):
    """Another live process holds the lock."""


class HeldLock:
    def __init__(self, db_path: Path | str, name: str, interval: float = HEARTBEAT_S,
                 stale_after: float = STALE_AFTER_S):
        self.db_path, self.name, self.owner = db_path, name, new_id("lock")
        self.interval, self.stale_after = interval, stale_after
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def acquire(self) -> "HeldLock":
        store = MemoryStore(self.db_path)
        try:
            if not store.acquire_lock(self.name, self.owner, self.stale_after):
                raise LockBusy(self.name)
        finally:
            store.close()
        self._thread = threading.Thread(target=self._beat, name=f"heartbeat:{self.name}", daemon=True)
        self._thread.start()
        return self

    def _beat(self) -> None:
        store = MemoryStore(self.db_path)          # SQLite connections belong to one thread
        try:
            while not self._stop.wait(self.interval):
                try:
                    store.touch_lock(self.name, self.owner)
                except Exception:                  # a busy database must never kill the heartbeat
                    pass
        finally:
            store.close()

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
        store = MemoryStore(self.db_path)
        try:
            store.release_lock(self.name, self.owner)
        finally:
            store.close()

    def __enter__(self) -> "HeldLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
