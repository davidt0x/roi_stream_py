from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import threading
import numpy as np


class TraceRing:
    """Thread-safe ring buffer for timeseries with K series.

    Stores a shared time vector and K value series in NumPy circular buffers.
    """

    def __init__(self, k: int, maxlen: int) -> None:
        self._lock = threading.Lock()
        self._k = int(max(0, k))
        self._maxlen = int(max(1, maxlen))
        self._t = np.zeros(self._maxlen, dtype=np.float64)
        self._y = np.zeros((self._k, self._maxlen), dtype=np.float64)
        self._size = 0
        self._head = 0
        # Monotonic count of total samples appended since start
        self._total_count: int = 0

    @property
    def k(self) -> int:
        return self._k

    @property
    def maxlen(self) -> int:
        return self._maxlen

    def append(self, t: float, y: np.ndarray) -> None:
        """Append a new sample (t, y[k])."""
        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()
        if y.size != self._k:
            raise ValueError("TraceRing.append: y size mismatch with k")
        with self._lock:
            idx = self._head
            self._t[idx] = float(t)
            if self._k > 0:
                self._y[:, idx] = y.astype(np.float64, copy=False)
            self._head = (idx + 1) % self._maxlen
            if self._size < self._maxlen:
                self._size += 1
            self._total_count += 1

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of t and y-series for the UI thread.

        Returns
        -------
        times : np.ndarray
            1D array of timestamps, ascending.
        values : np.ndarray
            2D array shaped (k, N) with matching samples.
        """
        with self._lock:
            size = self._size
            if size == 0:
                if self._k > 0:
                    return np.empty(0, dtype=np.float64), np.empty((self._k, 0), dtype=np.float64)
                return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)
            indices = self._ordered_indices_unlocked(size)
            t_vals = self._t[indices].astype(np.float64, copy=True)
            if self._k > 0:
                y_vals = self._y[:, indices].astype(np.float64, copy=True)
            else:
                y_vals = np.empty((0, t_vals.size), dtype=np.float64)
            return t_vals, y_vals

    def last_time(self) -> float:
        with self._lock:
            if self._size == 0:
                return 0.0
            idx = (self._head - 1) % self._maxlen
            return float(self._t[idx])

    def total_count(self) -> int:
        """Return the total number of samples ever appended (monotonic)."""
        with self._lock:
            return int(self._total_count)

    def snapshot_window(self, start_time: float, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return only the samples with t >= start_time, capped at max_points from the end.

        This avoids copying the entire ring when only a sliding window is needed.
        """
        with self._lock:
            size = self._size
            if size == 0:
                return np.empty(0, dtype=np.float64), np.empty((self._k, 0), dtype=np.float64)
            indices = self._ordered_indices_unlocked(size)
            t_vals = self._t[indices]
            start_idx = int(np.searchsorted(t_vals, float(start_time), side="left"))
            t_window = t_vals[start_idx:]
            if t_window.size == 0:
                return np.empty(0, dtype=np.float64), np.empty((self._k, 0), dtype=np.float64)
            if max_points is not None and max_points > 0 and t_window.size > max_points:
                t_window = t_window[-max_points:]
                idx_window = indices[start_idx:][-max_points:]
            else:
                idx_window = indices[start_idx:]
            t_out = np.asarray(t_window, dtype=np.float64)
            if self._k > 0:
                y_window = self._y[:, idx_window % self._maxlen]
                y_out = np.ascontiguousarray(y_window, dtype=np.float64)
            else:
                y_out = np.empty((0, t_out.size), dtype=np.float64)
            return t_out, y_out

    def _ordered_indices_unlocked(self, size: int) -> np.ndarray:
        """Return indices (ascending time) for the current buffer size. Caller holds lock."""
        if size <= 0:
            return np.empty(0, dtype=np.int64)
        start = (self._head - size) % self._maxlen
        if start + size <= self._maxlen:
            return np.arange(start, start + size, dtype=np.int64)
        first_len = self._maxlen - start
        first = np.arange(start, self._maxlen, dtype=np.int64)
        second = np.arange(0, size - first_len, dtype=np.int64)
        return np.concatenate((first, second))


@dataclass
class SharedState:
    """Shared state between the streaming worker and the GUI."""

    traces: TraceRing
    resolution: Optional[Tuple[int, int]] = None  # (W, H)
    circles: Optional[np.ndarray] = None  # (K, 3)

    # Latest frame (uint16 grayscale). Access guarded by _frame_lock.
    _frame_lock: threading.Lock = field(default_factory=threading.Lock)
    _frame16: Optional[np.ndarray] = None
    _preview8: Optional[np.ndarray] = None

    def update_frame(self, frame16: np.ndarray, resolution: Tuple[int, int], *, preview8: Optional[np.ndarray] = None) -> None:
        if frame16 is None:
            return
        with self._frame_lock:
            if self._frame16 is None or self._frame16.shape != frame16.shape:
                self._frame16 = frame16.copy()
            else:
                np.copyto(self._frame16, frame16)
            self.resolution = (int(resolution[0]), int(resolution[1]))
            if preview8 is not None:
                if self._preview8 is None or self._preview8.shape != preview8.shape or self._preview8.dtype != preview8.dtype:
                    self._preview8 = preview8.copy()
                else:
                    np.copyto(self._preview8, preview8)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._frame16 is None:
                return None
            return self._frame16.copy()

    def get_preview_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._preview8 is None:
                return None
            return self._preview8.copy()

    def copy_preview_frame(self, dst: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Copy the latest preview frame into dst (if provided) or return a copy."""
        with self._frame_lock:
            if self._preview8 is None:
                return None
            if (
                dst is not None
                and dst.shape == self._preview8.shape
                and dst.dtype == self._preview8.dtype
            ):
                np.copyto(dst, self._preview8)
                return dst
            return self._preview8.copy()
