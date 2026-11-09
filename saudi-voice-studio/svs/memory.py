"""Allocator hygiene for a long-running server (Linux/glibc).

Why: generation runs in worker threads. glibc gives every thread its own malloc arena; the large, short-lived tensors of a synthesis
are freed into that arena but not returned to the OS, so a server process keeps growing (measured: 3.97 -> 4.86 -> 5.57 GB RSS
over the first three jobs of a real session) even though nothing is leaked in Python. Two standard fixes, both applied here:
  * cap the number of arenas (mallopt M_ARENA_MAX) BEFORE the worker threads exist, and
  * return freed heap pages to the OS after every job / synthesis (malloc_trim).
No-ops on non-glibc platforms. Never restarts anything, never touches model state.
"""
from __future__ import annotations

import gc
import os
import sys

_LIBC = None


def _libc():
    global _LIBC
    if _LIBC is None:
        try:
            import ctypes
            _LIBC = ctypes.CDLL("libc.so.6") if sys.platform.startswith("linux") else False
        except Exception:
            _LIBC = False
    return _LIBC or None


def tune() -> bool:
    """Limit glibc malloc arenas (default: 2). Call once, early. Returns True if applied."""
    lc = _libc()
    if lc is None:
        return False
    try:
        n = int(os.environ.get("SVS_MALLOC_ARENAS", "2"))
        return lc.mallopt(-8, n) == 1             # M_ARENA_MAX
    except Exception:
        return False


def release() -> None:
    """Collect garbage and give freed heap pages back to the OS (all arenas)."""
    gc.collect()
    lc = _libc()
    if lc is not None:
        try:
            lc.malloc_trim(0)
        except Exception:
            pass
