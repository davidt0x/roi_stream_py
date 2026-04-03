from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os
import re

import numpy as np
import yaml

from .roi import ROIShape, normalize_roi_table


PathLike = Union[str, os.PathLike]


@dataclass
class StreamOptions:
    frames_per_chunk: int = 240
    print_fps_period: float = 1.0
    trace_buffer_sec: float = 600.0
    max_frames: int = 0  # 0 = unlimited


@dataclass
class ROISettings:
    table: np.ndarray
    resolution: Optional[Tuple[int, int]] = None
    labels: List[str] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    source: Optional[Union[int, str]] = None
    backend: Optional[str] = None
    hamamatsu_sdk_python_dir: Optional[str] = None
    hamamatsu: Dict[str, Any] = field(default_factory=dict)
    format: Optional[str] = None
    output: Optional[str] = None
    stream: StreamOptions = field(default_factory=StreamOptions)
    rois: Optional[ROISettings] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _parse_resolution_hint(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            w = int(value[0])
            h = int(value[1])
        except (TypeError, ValueError):
            raise ValueError("resolution entries must be integers") from None
        if w > 0 and h > 0:
            return (w, h)
        raise ValueError("resolution values must be positive")
    raise ValueError("resolution must be a sequence of two numbers [width, height]")


def _parse_roi_shapes_section(section: Any, base_dir: Path) -> ROISettings:
    if not isinstance(section, dict):
        raise ValueError("'rois' section must be a mapping")

    labels: list[str] = []
    resolution_hint = None
    table: Optional[np.ndarray] = None

    if "file" in section:
        roi_path = Path(section["file"])
        if not roi_path.is_absolute():
            roi_path = (base_dir / roi_path).resolve()
        if roi_path.suffix.lower() in {".yaml", ".yml"}:
            raise ValueError("rois.file cannot reference another YAML config; include shapes directly or use --config")
        table, file_res = load_rois_with_meta(roi_path)
        resolution_hint = file_res
    elif "shapes" in section:
        shapes_raw = section["shapes"]
        if not isinstance(shapes_raw, Iterable):
            raise ValueError("'rois.shapes' must be an iterable of ROI definitions")
        shapes: list[ROIShape] = []
        for idx, item in enumerate(shapes_raw):
            if not isinstance(item, dict):
                raise ValueError("Each ROI entry must be a mapping")
            center = item.get("center") or item.get("xy") or item.get("origin")
            if center is None:
                raise ValueError("ROI entry missing 'center'")
            if isinstance(center, dict):
                try:
                    cx = float(center["x"])
                    cy = float(center["y"])
                except KeyError as e:
                    raise ValueError("ROI center dict must have 'x' and 'y'") from e
            elif isinstance(center, (list, tuple)) and len(center) == 2:
                cx = float(center[0])
                cy = float(center[1])
            else:
                raise ValueError("ROI center must be [x, y] or {x:, y:}")

            kind = str(item.get("type", "")).strip().lower()
            radius = item.get("radius", item.get("r"))
            radii = item.get("radii")
            rx = item.get("rx")
            ry = item.get("ry")

            if kind == "circle":
                if radius is None:
                    raise ValueError("Circle ROI requires 'radius'")
                rx_val = ry_val = float(radius)
            else:
                if radii is not None:
                    if isinstance(radii, (list, tuple)) and len(radii) == 2:
                        rx_val = float(radii[0])
                        ry_val = float(radii[1])
                    else:
                        raise ValueError("'radii' must be [rx, ry]")
                elif rx is not None and ry is not None:
                    rx_val = float(rx)
                    ry_val = float(ry)
                elif radius is not None and kind in {"", "ellipse", "elliptical"}:
                    rx_val = ry_val = float(radius)
                else:
                    raise ValueError("Ellipse ROI requires 'radii' or both 'rx' and 'ry'")
            angle = item.get("angle_deg", item.get("angle", 0.0))
            shape = ROIShape(cx=cx, cy=cy, rx=rx_val, ry=ry_val, angle_deg=float(angle))
            shapes.append(shape)
            label = item.get("name")
            if label is None:
                label = item.get("label")
            if label is None:
                label = idx
            labels.append(str(label))
        table = normalize_roi_table(shapes)
    else:
        raise ValueError("ROI config must provide either 'file' or 'shapes'")

    if "resolution" in section:
        resolution_hint = _parse_resolution_hint(section.get("resolution"))

    if table is None:
        raise ValueError("Failed to build ROI table from configuration")

    return ROISettings(table=table, resolution=resolution_hint, labels=labels)


def load_rois_with_meta(path: PathLike) -> tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Load ROIs as an array of shape (K,5) [cx, cy, rx, ry, angle_deg]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ROI file not found: {p}")

    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("ROI files must be YAML with either 'shapes' or 'rois' sections")

    data = yaml.safe_load(p.read_text()) or {}

    def _build(section: Dict[str, Any]) -> ROISettings:
        return _parse_roi_shapes_section(section, p.parent)

    if isinstance(data, dict):
        if "rois" in data:
            settings = _build(data["rois"])
            return settings.table, settings.resolution
        if "shapes" in data:
            settings = _build(data)
            return settings.table, settings.resolution
    elif isinstance(data, list):
        settings = _build({"shapes": data})
        return settings.table, settings.resolution

    raise ValueError("ROI YAML must contain either a 'rois' mapping or a 'shapes' list")


def load_runtime_config(path: PathLike) -> RuntimeConfig:
    """Load a YAML configuration describing capture/stream/ROI settings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level")

    extras: Dict[str, Any] = {}
    known_keys = {"source", "backend", "capture", "format", "output", "stream", "rois"}
    for k, v in data.items():
        if k not in known_keys:
            extras[k] = v

    stream_section = data.get("stream")
    stream_opts = StreamOptions()
    if isinstance(stream_section, dict):
        if "frames_per_chunk" in stream_section:
            stream_opts.frames_per_chunk = int(stream_section["frames_per_chunk"])
        if "print_fps_period" in stream_section:
            stream_opts.print_fps_period = float(stream_section["print_fps_period"])
        if "trace_buffer_sec" in stream_section:
            stream_opts.trace_buffer_sec = float(stream_section["trace_buffer_sec"])
        if "max_frames" in stream_section:
            stream_opts.max_frames = int(stream_section["max_frames"])

    rois_section = data.get("rois")
    roi_settings = _parse_roi_shapes_section(rois_section, p.parent) if rois_section is not None else None

    source = data.get("source")
    backend = data.get("backend")
    hamamatsu_sdk_python_dir = None
    hamamatsu: Dict[str, Any] = {}
    capture_section = data.get("capture")
    if isinstance(capture_section, dict):
        driver = capture_section.get("driver")
        if backend is None and driver is not None:
            backend = str(driver)
        sdk_python_dir = capture_section.get("sdk_python_dir")
        if sdk_python_dir is not None:
            hamamatsu_sdk_python_dir = str(sdk_python_dir)
        if "hamamatsu" in capture_section and isinstance(capture_section.get("hamamatsu"), dict):
            hamamatsu = dict(capture_section["hamamatsu"])
        else:
            for key in ("profile", "frame_rate", "exposure_sec", "binning", "roi", "pixel_type", "readout_speed", "output_triggers", "properties"):
                if key in capture_section:
                    hamamatsu[key] = capture_section[key]
    fmt = data.get("format")
    out = data.get("output")

    return RuntimeConfig(
        source=source,
        backend=backend,
        hamamatsu_sdk_python_dir=hamamatsu_sdk_python_dir,
        hamamatsu=hamamatsu,
        format=fmt,
        output=out,
        stream=stream_opts,
        rois=roi_settings,
        extras=extras,
    )


def parse_source(src: str) -> Union[int, str]:
    """Parse source: integer device index or file path string."""
    s = src.strip()
    # Integer device index (supports leading +/-, but typically non-negative)
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            pass
    return s


def parse_format(fmt: Optional[str]) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Parse a format string like '1280x720@60' → (W, H, FPS).

    Returns (width, height, fps) with None for unspecified parts.
    Accepts partials like '1280x720' or '@60'.
    """
    if not fmt:
        return None, None, None
    fmt = fmt.strip()
    width = height = None
    fps: Optional[float] = None

    # Split on '@' for FPS
    if "@" in fmt:
        dims, fps_str = fmt.split("@", 1)
        try:
            fps = float(fps_str)
        except ValueError:
            fps = None
    else:
        dims = fmt

    dims = dims.strip()
    if "x" in dims:
        wh = dims.split("x")
        if len(wh) == 2:
            try:
                width = int(wh[0]) if wh[0] else None
                height = int(wh[1]) if wh[1] else None
            except ValueError:
                width = height = None

    return width, height, fps
