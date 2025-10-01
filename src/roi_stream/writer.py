from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import datetime as _dt

import h5py
import numpy as np


@dataclass
class H5TracesWriter:
    path: Path
    K: int
    chunk_frames: int = 240

    def __init__(self, path: str | Path, circles: np.ndarray, meta: Dict[str, Any], chunk_frames: int = 240):
        if circles is None or circles.ndim != 2 or circles.shape[1] != 3:
            raise ValueError("circles must be Nx3 array")
        self.path = Path(path)
        self.K = int(circles.shape[0])
        self.chunk_frames = max(1, int(chunk_frames))
        if self.path.exists():
            self.path.unlink()

        self._f = h5py.File(self.path, 'w')
        # Extendible datasets
        self._ds_time = self._f.create_dataset(
            '/time', shape=(0, 1), maxshape=(None, 1), dtype='f8', chunks=(self.chunk_frames, 1)
        )
        self._ds_means = self._f.create_dataset(
            '/roi/means', shape=(0, self.K), maxshape=(None, self.K), dtype='f4', chunks=(self.chunk_frames, self.K)
        )
        # Static datasets
        self._f.create_dataset('/roi/circles', data=circles.astype(np.float64, copy=False), dtype='f8')

        # Root attributes
        meta = dict(meta or {})
        meta.setdefault('created_with', 'roi_stream python')
        meta.setdefault('start_iso8601', _now_iso8601())
        for k, v in meta.items():
            self._f['/'].attrs[str(k)] = v

        self.rows = 0
        self._has_dff = False
        self._ds_dff = None

    def append(self, tvec: np.ndarray, means: np.ndarray, dff: Optional[np.ndarray] = None) -> None:
        if tvec is None or len(tvec) == 0:
            return
        n = int(len(tvec))
        if means is None or means.shape[0] != n:
            raise ValueError("means must have n rows to match tvec")

        start = self.rows
        new_rows = start + n
        # Resize and write
        self._ds_time.resize((new_rows, 1))
        # Preserve 2D shape on assignment
        self._ds_time[start:new_rows, :] = np.asarray(tvec, dtype=np.float64).reshape(n, 1)

        self._ds_means.resize((new_rows, self.K))
        self._ds_means[start:new_rows, :] = np.asarray(means, dtype=np.float32)

        if dff is not None:
            if self._ds_dff is None:
                self._ds_dff = self._f.create_dataset(
                    '/roi/dff', shape=(0, self.K), maxshape=(None, self.K), dtype='f4', chunks=(self.chunk_frames, self.K)
                )
            self._ds_dff.resize((new_rows, self.K))
            self._ds_dff[start:new_rows, :] = np.asarray(dff, dtype=np.float32)

        self.rows = new_rows

    def finalize(self, summary: Dict[str, Any]) -> None:
        # Closing attributes (e.g., end time and stats)
        summary = dict(summary or {})
        summary.setdefault('end_iso8601', _now_iso8601())
        summary.setdefault('rows', int(self.rows))
        for k, v in summary.items():
            self._f['/'].attrs[str(k)] = v
        self._f.flush()
        self._f.close()


def _now_iso8601() -> str:
    return _dt.datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S.%f%z')
