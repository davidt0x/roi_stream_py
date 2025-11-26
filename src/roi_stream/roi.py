from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import cv2
import numpy as np

_LUT_GRAY_TO_16 = (np.arange(256, dtype=np.uint16) * 257).reshape(-1, 1)
_LUT_B = np.round(np.arange(256, dtype=np.float32) * 0.1140209043 * 257.0).astype(np.uint16).reshape(-1, 1)
_LUT_G = np.round(np.arange(256, dtype=np.float32) * 0.5870430745 * 257.0).astype(np.uint16).reshape(-1, 1)
_LUT_R = np.round(np.arange(256, dtype=np.float32) * 0.2989360213 * 257.0).astype(np.uint16).reshape(-1, 1)

def to_uint16_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an image (uint8/uint16/float, gray or BGR) to uint16 grayscale [0..65535].

    - If color, assumes BGR channel order (OpenCV convention).
    - Float inputs: if max <= 1 → scaled by 65535; if <= 255 → scaled by 257; else clipped to 0..65535.
    """
    if frame is None:
        raise ValueError("frame is None")

    arr = np.asarray(frame)
    if arr.ndim == 3:
        # Color (assume BGR)
        if arr.dtype == np.uint8:
            b = cv2.LUT(arr[..., 0], _LUT_B)
            g = cv2.LUT(arr[..., 1], _LUT_G)
            r = cv2.LUT(arr[..., 2], _LUT_R)
            total = b.astype(np.uint32, copy=False)
            total += g.astype(np.uint32, copy=False)
            total += r.astype(np.uint32, copy=False)
            return total.astype(np.uint16, copy=False)
        if arr.dtype == np.uint16:
            return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Fallback for float or other integer types
        # Use float32 to limit conversion cost
        bgr = arr.astype(np.float32, copy=False)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return to_uint16_gray(gray)

    # Single-channel
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return cv2.LUT(arr, _LUT_GRAY_TO_16)

    # Float / other integer types
    y = arr.astype(np.float32, copy=False)
    mx = float(np.max(np.abs(y))) if y.size else 0.0
    if mx <= 1.0:
        out = np.clip(y, 0.0, 1.0) * 65535.0
    elif mx <= 255.0:
        out = np.clip(y, 0.0, 255.0) * 257.0
    else:
        out = np.clip(y, 0.0, 65535.0)
    np.rint(out, out=out)
    return out.astype(np.uint16, copy=False)


@dataclass(frozen=True)
class ROIShape:
    """Immutable ROI shape described by an ellipse (circle is rx==ry)."""

    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float = 0.0

    def as_array(self) -> np.ndarray:
        arr = np.array(
            [float(self.cx), float(self.cy), float(self.rx), float(self.ry), float(self.angle_deg)],
            dtype=np.float64,
        )
        return arr

    @staticmethod
    def from_sequence(vals: Sequence[float]) -> "ROIShape":
        if len(vals) == 3:
            cx, cy, r = vals
            return ROIShape(cx=float(cx), cy=float(cy), rx=float(r), ry=float(r), angle_deg=0.0)
        if len(vals) == 5:
            cx, cy, rx, ry, ang = vals
            return ROIShape(cx=float(cx), cy=float(cy), rx=float(rx), ry=float(ry), angle_deg=float(ang))
        raise ValueError("ROI sequences must have 3 (circle) or 5 (ellipse) values")


def roi_table_from_iter(shapes: Iterable[ROIShape]) -> np.ndarray:
    rows = [shape.as_array() for shape in shapes]
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    return np.vstack(rows)


def normalize_roi_table(
    data: Iterable[ROIShape] | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return an array shaped (K,5) -> [cx, cy, rx, ry, angle_deg]."""
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("ROI array must be 2D")
        if arr.shape[1] == 5:
            return arr.astype(np.float64, copy=False)
        if arr.shape[1] == 3:
            out = np.zeros((arr.shape[0], 5), dtype=np.float64)
            out[:, 0:2] = arr[:, 0:2]
            out[:, 2] = arr[:, 2]
            out[:, 3] = arr[:, 2]
            # angle stays zero
            return out
        raise ValueError("ROI array must have 3 (circle) or 5 (ellipse) columns")
    # Iterable of shapes
    try:
        shapes = list(data)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError("Unsupported ROI data type") from None
    if not shapes:
        return np.zeros((0, 5), dtype=np.float64)
    if isinstance(shapes[0], ROIShape):
        return roi_table_from_iter(shapes)  # type: ignore[arg-type]
    # Otherwise convert sequences to ROIShape first
    converted = [ROIShape.from_sequence(s) for s in shapes]  # type: ignore[arg-type]
    return roi_table_from_iter(converted)


@dataclass
class ROISet:
    height: int
    width: int
    table: np.ndarray  # (K,5) floats [cx, cy, rx, ry, angle_deg]

    def __post_init__(self) -> None:
        self.table = normalize_roi_table(self.table)
        if self.table.ndim != 2 or self.table.shape[1] != 5:
            raise ValueError("table must be a (K,5) array")
        self.table = self.table.astype(np.float64, copy=False)
        self.K = int(self.table.shape[0])
        self._build_indices()

    def _build_indices(self) -> None:
        H, W = int(self.height), int(self.width)
        yy = np.arange(H, dtype=np.float64)[:, None]
        xx = np.arange(W, dtype=np.float64)[None, :]
        self._indices: List[np.ndarray] = []
        self.npix: np.ndarray = np.zeros(self.K, dtype=np.uint32)
        for k in range(self.K):
            cx, cy, rx, ry, angle_deg = self.table[k]
            rx = max(0.0, float(rx))
            ry = max(0.0, float(ry))
            if rx == 0.0 or ry == 0.0:
                self._indices.append(np.empty(0, dtype=np.int64))
                self.npix[k] = 0
                continue
            # Rotate coordinates around center by -angle
            angle_rad = float(angle_deg) * np.pi / 180.0
            cos_a = float(np.cos(angle_rad))
            sin_a = float(np.sin(angle_rad))
            dx = xx - float(cx)
            dy = yy - float(cy)
            xr = cos_a * dx + sin_a * dy
            yr = -sin_a * dx + cos_a * dy
            mask = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
            idx = np.flatnonzero(mask)
            self._indices.append(idx)
            self.npix[k] = np.uint32(idx.size)

    def compute_means(self, frame16: np.ndarray) -> np.ndarray:
        """Compute per-ROI mean intensity for a uint16 frame.

        Returns float32 array of shape (K,).
        """
        if frame16.dtype != np.uint16:
            raise ValueError("compute_means expects a uint16 frame")
        if frame16.ndim != 2:
            raise ValueError("compute_means expects a single-channel frame")
        if frame16.shape != (self.height, self.width):
            raise ValueError("frame shape does not match ROI precompute")

        means = np.empty(self.K, dtype=np.float32)
        # Use float64 accumulations to avoid overflow/precision issues
        f = frame16.ravel()
        for k in range(self.K):
            idx = self._indices[k]
            if idx.size == 0:
                means[k] = np.nan
                continue
            s = float(f[idx].mean(dtype=np.float64))
            means[k] = np.float32(s)
        return means


class CirclesROI(ROISet):
    """Compatibility wrapper that accepts an Nx3 circle array."""

    def __init__(self, height: int, width: int, circles: np.ndarray) -> None:
        super().__init__(height=height, width=width, table=circles)
