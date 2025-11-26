from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import datetime as _dt

import h5py
import numpy as np

from .roi import normalize_roi_table


@dataclass
class H5TracesWriter:
    path: Path
    K: int
    chunk_frames: int = 240

    def __init__(self, path: str | Path, rois: np.ndarray, meta: Dict[str, Any], chunk_frames: int = 240):
        if rois is None:
            raise ValueError("rois must not be None")
        roi_table = normalize_roi_table(rois)
        self.path = Path(path)
        self.K = int(roi_table.shape[0])
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
        self._f.create_dataset('/roi/shapes', data=roi_table.astype(np.float64, copy=False), dtype='f8')
        if self.K == 0:
            circles_copy = np.empty((0, 3), dtype=np.float64)
        else:
            circles_copy = roi_table[:, :3]
        is_circular = bool(
            roi_table.size == 0
            or (np.allclose(roi_table[:, 2], roi_table[:, 3]) and np.allclose(roi_table[:, 4], 0.0))
        )
        if is_circular:
            self._f.create_dataset('/roi/circles', data=circles_copy.astype(np.float64, copy=False), dtype='f8')
        else:
            self._f['/roi'].attrs['circles_geometry'] = 'ellipse'

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
