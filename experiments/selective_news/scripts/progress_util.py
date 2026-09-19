#!/usr/bin/env python3
"""Print progress as done/total (pct%)."""
from __future__ import annotations


def fmt_progress(done: int, total: int, prefix: str = "") -> str:
    pct = (100.0 * done / total) if total else 100.0
    bar_n = 20
    filled = int(bar_n * done / total) if total else bar_n
    bar = "#" * filled + "-" * (bar_n - filled)
    head = f"{prefix} " if prefix else ""
    return f"{head}[{bar}] {done}/{total} ({pct:.1f}%)"


def print_progress(done: int, total: int, prefix: str = "", extra: str = "") -> None:
    line = fmt_progress(done, total, prefix=prefix)
    if extra:
        line = f"{line}  {extra}"
    print(line, flush=True)
