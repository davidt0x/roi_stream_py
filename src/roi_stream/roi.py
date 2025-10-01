from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def to_uint16_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an image (uint8/uint16/float, gray or BGR) to uint16 grayscale [0..65535].

    - If color, assumes BGR channel order (OpenCV convention).
    - Float inputs: if max <= 1 → scaled by 65535; if <= 255 → scaled by 257; else clipped to 0..65535.
    """
    if frame is None:
        raise ValueError("frame is None")

    arr = frame
    if arr.ndim == 3:
        # Color (assume BGR)
        b = arr[..., 0].astype(np.float64, copy=False)
        g = arr[..., 1].astype(np.float64, copy=False)
        r = arr[..., 2].astype(np.float64, copy=False)

        # Luma approximation: Y ≈ 0.299 R + 0.587 G + 0.114 B
        y = 0.2989360213 * r + 0.5870430745 * g + 0.1140209043 * b

        if arr.dtype == np.uint8:
            out = np.round(y * 257.0)
        elif arr.dtype == np.uint16:
            out = np.clip(np.round(y), 0.0, 65535.0)
        else:
            mx = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
            if mx <= 1.0:
                out = np.round(np.clip(y, 0.0, 1.0) * 65535.0)
            elif mx <= 255.0:
                out = np.round(np.clip(y, 0.0, 255.0) * 257.0)
            else:
                out = np.clip(np.round(y), 0.0, 65535.0)
        return out.astype(np.uint16, copy=False)

    # Single-channel
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return (arr.astype(np.uint16) * 257)

    # Float / other integer types
    y = arr.astype(np.float64, copy=False)
    mx = float(np.nanmax(np.abs(y))) if y.size else 0.0
    if mx <= 1.0:
        out = np.round(np.clip(y, 0.0, 1.0) * 65535.0)
    elif mx <= 255.0:
        out = np.round(np.clip(y, 0.0, 255.0) * 257.0)
    else:
        out = np.clip(np.round(y), 0.0, 65535.0)
    return out.astype(np.uint16, copy=False)


@dataclass
class CirclesROI:
    height: int
    width: int
    circles: np.ndarray  # (K,3) floats [xc, yc, r]

    def __post_init__(self) -> None:
        if self.circles.ndim != 2 or self.circles.shape[1] != 3:
            raise ValueError("circles must be (K,3) array")
        self.K = int(self.circles.shape[0])
        self._build_indices()

    def _build_indices(self) -> None:
        H, W = int(self.height), int(self.width)
        circles = self.circles.astype(np.float64, copy=False)
        # Precompute per-ROI boolean masks and flattened indices
        yy = np.arange(H, dtype=np.float64)[:, None]
        xx = np.arange(W, dtype=np.float64)[None, :]
        self._indices: List[np.ndarray] = []
        self.npix: np.ndarray = np.zeros(self.K, dtype=np.uint32)
        for k in range(self.K):
            xc, yc, r = circles[k]
            # Clip radius and centers to reasonable bounds
            r = max(0.0, float(r))
            mask = (xx - float(xc)) ** 2 + (yy - float(yc)) ** 2 <= r ** 2
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

