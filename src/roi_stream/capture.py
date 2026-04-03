from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Iterable, List
import sys
import platform
import cv2

from .hamamatsu_sdk import (
    apply_hamamatsu_settings,
    discover_sdk_python_dir,
    get_missing_sdk_message,
    import_hamamatsu_modules,
)


@dataclass
class FrameSource:
    source: Union[int, str]
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    backend: Optional[str] = None  # 'any'|'v4l2'|'msmf'|'dshow'|'gstreamer'|'ffmpeg'|'hamamatsu_sdk'
    hamamatsu_settings: Optional[dict] = None

    def __post_init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._hamamatsu_cam = None
        self._hamamatsu_api = None
        self._last_frame_shape: Optional[tuple[int, int]] = None
        self._last_error: str = ""

    def open(self) -> bool:
        self._last_error = ""
        if (self.backend or "any").lower() == "hamamatsu_sdk":
            return self._open_hamamatsu_sdk()

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
                self._last_error = ("Detected WSL. Access to host cameras via /dev/video* is typically unavailable. "
                                    "Run on Windows Python or use a file/RTSP source.")
                print(f"[roi_stream] {self._last_error}")
            else:
                self._last_error = "OpenCV could not open the requested source."
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
        if self._hamamatsu_cam is not None:
            try:
                if not self._hamamatsu_cam.wait_capevent_frameready(1000):
                    self._last_error = f"Hamamatsu frame wait failed: {self._hamamatsu_cam.lasterr()}"
                    return False, None
                frame = self._hamamatsu_cam.buf_getlastframedata()
                if frame is None:
                    self._last_error = "Hamamatsu SDK returned no frame data."
                    return False, None
                try:
                    self._last_frame_shape = tuple(int(x) for x in frame.shape[:2])
                except Exception:
                    self._last_frame_shape = None
                return True, frame
            except Exception as exc:
                self._last_error = f"Hamamatsu frame acquisition failed: {exc}"
                return False, None
        if self._cap is None:
            raise RuntimeError("FrameSource not opened")
        return self._cap.read()

    def get_resolution(self):
        if self._last_frame_shape is not None:
            return int(self._last_frame_shape[1]), int(self._last_frame_shape[0])
        if self._cap is None:
            return None, None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def get_fps(self) -> float:
        if self._hamamatsu_cam is not None:
            return float(self.fps) if self.fps is not None else 0.0
        if self._cap is None:
            return 0.0
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 0.0

    def release(self) -> None:
        if self._hamamatsu_cam is not None:
            try:
                try:
                    self._hamamatsu_cam.cap_stop()
                except Exception:
                    pass
                try:
                    self._hamamatsu_cam.buf_release()
                except Exception:
                    pass
                try:
                    self._hamamatsu_cam.dev_close()
                except Exception:
                    pass
            finally:
                self._hamamatsu_cam = None
                if self._hamamatsu_api is not None:
                    try:
                        self._hamamatsu_api.Dcamapi.uninit()
                    finally:
                        self._hamamatsu_api = None
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def get_last_error(self) -> str:
        return self._last_error

    def _open_hamamatsu_sdk(self) -> bool:
        if not isinstance(self.source, int):
            self._last_error = "Hamamatsu SDK backend requires an integer camera index."
            return False

        python_dir = discover_sdk_python_dir()
        if python_dir is None:
            self._last_error = get_missing_sdk_message()
            return False

        try:
            dcamapi4, dcam = import_hamamatsu_modules(python_dir)
        except Exception as exc:
            self._last_error = f"Failed to import Hamamatsu SDK from {python_dir}: {exc}"
            return False

        try:
            if not dcam.Dcamapi.init():
                self._last_error = f"DCAM initialization failed: {dcam.Dcamapi.lasterr()}"
                return False

            cam = dcam.Dcam(int(self.source))
            if not cam.dev_open():
                self._last_error = f"DCAM camera open failed: {cam.lasterr()}"
                dcam.Dcamapi.uninit()
                return False

            if not cam.buf_alloc(3):
                self._last_error = f"DCAM buffer allocation failed: {cam.lasterr()}"
                cam.dev_close()
                dcam.Dcamapi.uninit()
                return False

            try:
                apply_hamamatsu_settings(cam, dcamapi4, self.hamamatsu_settings, fps_hint=self.fps)
            except Exception as exc:
                self._last_error = f"DCAM camera setup failed: {exc}"
                cam.buf_release()
                cam.dev_close()
                dcam.Dcamapi.uninit()
                return False

            if not cam.cap_start():
                self._last_error = f"DCAM capture start failed: {cam.lasterr()}"
                cam.buf_release()
                cam.dev_close()
                dcam.Dcamapi.uninit()
                return False

            self._hamamatsu_cam = cam
            self._hamamatsu_api = dcam
            return True
        except Exception as exc:
            try:
                dcam.Dcamapi.uninit()
            except Exception:
                pass
            self._last_error = f"Hamamatsu SDK backend setup failed: {exc}"
            return False


def _is_wsl() -> bool:
    try:
        return sys.platform.startswith('linux') and 'microsoft' in platform.uname().release.lower()
    except Exception:
        return False
