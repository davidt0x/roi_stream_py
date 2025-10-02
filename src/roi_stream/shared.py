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

    def last_time(self) -> float:
        with self._lock:
            if not self._t:
                return 0.0
            return float(self._t[-1])

    def snapshot_window(self, start_time: float, max_points: int | None = None) -> tuple[list[float], list[list[float]]]:
        """Return only the samples with t >= start_time, capped at max_points from the end.

        This avoids copying the entire ring when only a sliding window is needed.
        """
        with self._lock:
            if not self._t:
                return [], [[] for _ in range(self._k)]
            # Iterate from the end to collect recent points until hitting start_time or cap
            it_t = reversed(self._t)
            it_ys = [reversed(dq) for dq in self._y]
            t_rev: list[float] = []
            y_rev: list[list[float]] = [[] for _ in range(self._k)]
            cap = max_points if (max_points is not None and max_points > 0) else None
            count = 0
            for vals in zip(it_t, *it_ys):
                tval = float(vals[0])
                if cap is not None and count >= cap:
                    break
                if tval < start_time:
                    # We've reached outside the window; stop
                    break
                t_rev.append(tval)
                for i in range(self._k):
                    y_rev[i].append(float(vals[i + 1]))
                count += 1
            # Reverse to ascending time order
            t_rev.reverse()
            for i in range(self._k):
                y_rev[i].reverse()
            return t_rev, y_rev


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
