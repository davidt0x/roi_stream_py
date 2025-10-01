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


def _parse_resolution_from_header(path: Path) -> Optional[Tuple[int, int]]:
    """Parse an optional resolution hint from comment/header lines in a CSV.

    Recognizes patterns like:
      - "# resolution: 1280x720"
      - "# resolution=1280x720"
      - "# width=1280 height=720"
      - "# 1280x720"
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(20):  # scan first few lines
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if not s.startswith("#"):
                    # stop at first non-comment line
                    break
                # search for WxH
                m = re.search(r"(\d+)\s*[xX,]\s*(\d+)", s)
                if m:
                    w = int(m.group(1))
                    h = int(m.group(2))
                    if w > 0 and h > 0:
                        return (w, h)
                # search for width/height keywords
                m2 = re.search(r"width\s*[=:]\s*(\d+).*height\s*[=:]\s*(\d+)", s, flags=re.IGNORECASE)
                if m2:
                    w = int(m2.group(1))
                    h = int(m2.group(2))
                    if w > 0 and h > 0:
                        return (w, h)
                # else continue
    except Exception:
        return None
    return None


def load_circles_with_meta(path: PathLike) -> tuple[np.ndarray, Optional[Tuple[int, int]]]:
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
        src_res: Optional[Tuple[int, int]] = None
    else:
        # CSV or text: read rows of 3 floats, skipping comments/blank lines
        src_res = _parse_resolution_from_header(p)
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
    return arr, src_res


def load_circles(path: PathLike) -> np.ndarray:
    arr, _ = load_circles_with_meta(path)
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
