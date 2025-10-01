from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import csv
import json
import os
import re
import numpy as np


PathLike = Union[str, os.PathLike]


@dataclass
class StreamOptions:
    frames_per_chunk: int = 240
    print_fps_period: float = 1.0
    trace_buffer_sec: float = 600.0
    max_frames: int = 0  # 0 = unlimited


def load_circles(path: PathLike) -> np.ndarray:
    """Load ROI circles as an array of shape (K, 3) [xc, yc, r] floats.

    Supports CSV (with or without header, lines starting with '#' ignored)
    and JSON (list of [xc, yc, r]).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ROI file not found: {p}")

    ext = p.suffix.lower()
    if ext in {".json", ".jsn"}:
        data = json.loads(p.read_text())
        arr = np.asarray(data, dtype=float)
    else:
        # CSV or text: read rows of 3 floats, skipping comments/blank lines
        rows = []
        with p.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                joined = "".join(row).strip()
                if not joined or joined.startswith("#"):
                    continue
                try:
                    vals = [float(x) for x in row[:3]]
                except ValueError:
                    # Maybe there is a header; skip lines that cannot parse
                    continue
                if len(vals) != 3:
                    continue
                rows.append(vals)
        if not rows:
            # Fallback: whitespace-delimited load
            arr = np.loadtxt(str(p), dtype=float)
        else:
            arr = np.asarray(rows, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("ROI circles must be an Nx3 array of [xc, yc, r]")
    return arr


def parse_source(src: str) -> Union[int, str]:
    """Parse source: integer device index or file path string."""
    s = src.strip()
    # Integer device index (supports leading +/-, but typically non-negative)
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            pass
    return s


def parse_format(fmt: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Parse a format string like '1280x720@60' → (W, H, FPS).

    Returns (width, height, fps) with None for unspecified parts.
    Accepts partials like '1280x720' or '@60'.
    """
    if not fmt:
        return None, None, None
    fmt = fmt.strip()
    width = height = None
    fps: Optional[float] = None

    # Split on '@' for FPS
    if "@" in fmt:
        dims, fps_str = fmt.split("@", 1)
        try:
            fps = float(fps_str)
        except ValueError:
            fps = None
    else:
        dims = fmt

    dims = dims.strip()
    if "x" in dims:
        wh = dims.split("x")
        if len(wh) == 2:
            try:
                width = int(wh[0]) if wh[0] else None
                height = int(wh[1]) if wh[1] else None
            except ValueError:
                width = height = None

    return width, height, fps

