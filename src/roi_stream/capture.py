from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Iterable, List
import sys
import platform
import cv2


@dataclass
class FrameSource:
    source: Union[int, str]
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    backend: Optional[str] = None  # 'any'|'v4l2'|'msmf'|'dshow'|'gstreamer'|'ffmpeg'

    def __post_init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        # Determine backend candidates
        backends: List[Optional[int]] = []
        b = (self.backend or 'any').lower()
        def to_cv_backend(name: str) -> Optional[int]:
            return {
                'any': None,
                'v4l2': getattr(cv2, 'CAP_V4L2', None),
                'msmf': getattr(cv2, 'CAP_MSMF', None),
                'dshow': getattr(cv2, 'CAP_DSHOW', None),
                'gstreamer': getattr(cv2, 'CAP_GSTREAMER', None),
                'ffmpeg': getattr(cv2, 'CAP_FFMPEG', None),
            }.get(name, None)

        if isinstance(self.source, int):
            # Device index: try user-specified backend first, then OS-specific fallbacks
            cand = [to_cv_backend(b)] if b != 'any' else []
            if sys.platform == 'win32':
                cand += [getattr(cv2, 'CAP_MSMF', None), getattr(cv2, 'CAP_DSHOW', None), None]
            elif sys.platform.startswith('linux'):
                cand += [getattr(cv2, 'CAP_V4L2', None), None]
            else:
                cand += [None]
            # remove duplicates while preserving order
            seen = set()
            backends = []
            for x in cand:
                if x not in seen:
                    backends.append(x); seen.add(x)
        else:
            # File path/URL or named device (e.g., "video=OBS Virtual Camera")
            cand = [to_cv_backend(b)] if b != 'any' else []
            cand += [None]
            # remove duplicates
            seen = set(); backends = []
            for x in cand:
                if x not in seen:
                    backends.append(x); seen.add(x)

        # Try opening with candidates
        cap = None
        for bk in backends:
            try:
                if isinstance(self.source, int):
                    cap = cv2.VideoCapture(self.source if bk is None else int(self.source), bk) if bk is not None else cv2.VideoCapture(self.source)
                else:
                    # When using DirectShow, video device names must be provided as "video=Device Name"
                    cap = cv2.VideoCapture(self.source, bk) if bk is not None else cv2.VideoCapture(self.source)
            except Exception:
                cap = cv2.VideoCapture(self.source)
            if cap is not None and cap.isOpened():
                break
            if cap is not None:
                cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            self._cap = None
            if _is_wsl() and isinstance(self.source, int):
                print("[roi_stream] Detected WSL. Access to host cameras via /dev/video* is typically unavailable. "
                      "Run on Windows Python or use a file/RTSP source.")
            return False

        # Apply requested properties if provided
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps is not None:
            try:
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            except Exception:
                pass

        self._cap = cap
        return True

    def read(self):
        if self._cap is None:
            raise RuntimeError("FrameSource not opened")
        return self._cap.read()

    def get_resolution(self):
        if self._cap is None:
            return None, None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def get_fps(self) -> float:
        if self._cap is None:
            return 0.0
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 0.0

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None


def _is_wsl() -> bool:
    try:
        return sys.platform.startswith('linux') and 'microsoft' in platform.uname().release.lower()
    except Exception:
        return False
