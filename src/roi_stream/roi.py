from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
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
