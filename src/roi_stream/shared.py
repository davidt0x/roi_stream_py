from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque
import threading
import numpy as np


class TraceRing:
    """Thread-safe ring buffer for timeseries with K series.

    Stores a shared time vector and K value series as deques with a fixed maxlen.
    """

    def __init__(self, k: int, maxlen: int) -> None:
        self._lock = threading.Lock()
        self._k = int(max(0, k))
        self._maxlen = int(max(1, maxlen))
        self._t: deque[float] = deque(maxlen=self._maxlen)
        self._y: List[deque[float]] = [deque(maxlen=self._maxlen) for _ in range(self._k)]

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
            self._t.append(float(t))
            for i in range(self._k):
                self._y[i].append(float(y[i]))

    def snapshot(self) -> tuple[list[float], list[list[float]]]:
        """Return shallow copies of t and y-series for UI thread."""
        with self._lock:
            t = list(self._t)
            y = [list(series) for series in self._y]
        return t, y


@dataclass
class SharedState:
    """Shared state between the streaming worker and the GUI."""

    traces: TraceRing
    resolution: Optional[Tuple[int, int]] = None  # (W, H)
    circles: Optional[np.ndarray] = None  # (K, 3)

    # Latest frame (uint16 grayscale). Access guarded by _frame_lock.
    _frame_lock: threading.Lock = threading.Lock()
    _frame16: Optional[np.ndarray] = None

    def update_frame(self, frame16: np.ndarray, resolution: Tuple[int, int]) -> None:
        if frame16 is None:
            return
        with self._frame_lock:
            self._frame16 = frame16.copy()
            self.resolution = (int(resolution[0]), int(resolution[1]))

    def get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._frame16 is None:
                return None
            return self._frame16.copy()

