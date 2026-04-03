from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union
import time

import numpy as np

from .capture import FrameSource
from .config import StreamOptions
from .roi import ROISet, normalize_roi_table, to_uint16_gray
from .writer import H5TracesWriter
from .shared import SharedState
import threading


def _prepare_output_path(base: Path) -> Path:
    """Return a writable path, appending a timestamp if the target exists."""
    base = Path(base)
    if not base.exists():
        return base

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = base.stem
    suffix = base.suffix
    candidate = base.with_name(f"{stem}_{ts}{suffix}")
    if not candidate.exists():
        return candidate

    for idx in range(1, 1000):
        candidate = base.with_name(f"{stem}_{ts}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to determine unique output path for {base}")


def run_stream(
    source: Union[int, str],
    rois: np.ndarray,
    out_path: Optional[Union[str, Path]] = None,
    opts: Optional[StreamOptions] = None,
    format_tuple: Optional[Tuple[Optional[int], Optional[int], Optional[float]]] = None,
    backend: Optional[str] = None,
    shared: Optional[SharedState] = None,
    stop_event: Optional[threading.Event] = None,
    roi_src_resolution: Optional[Tuple[int, int]] = None,
    hamamatsu_settings: Optional[dict] = None,
) -> Path:
    """Run the streaming loop headlessly and write HDF5.

    Returns the HDF5 path.
    """
    opts = opts or StreamOptions()

    width = height = fps_req = None
    if format_tuple is not None:
        width, height, fps_req = format_tuple

    src = FrameSource(source, width=width, height=height, fps=fps_req, backend=backend, hamamatsu_settings=hamamatsu_settings)
    if not src.open():
        detail = src.get_last_error().strip()
        if detail:
            raise RuntimeError(f"Failed to open source: {source}. {detail}")
        raise RuntimeError(f"Failed to open source: {source}")

    # Read one frame to determine resolution (and to warm up the capture)
    ok, frame0 = src.read()
    if not ok or frame0 is None:
        src.release()
        raise RuntimeError("No frames received from source")

    f16_0 = to_uint16_gray(frame0)
    preview8_0 = np.right_shift(f16_0, 8).astype(np.uint8, copy=False)
    H, W = f16_0.shape

    # If ROI file included a source resolution and it differs, scale coordinates
    roi_table = normalize_roi_table(rois)
    if roi_src_resolution is not None:
        roiW, roiH = int(roi_src_resolution[0]), int(roi_src_resolution[1])
        if roiW > 0 and roiH > 0 and (roiW != W or roiH != H):
            sx = W / float(roiW)
            sy = H / float(roiH)
            rois_scaled = roi_table.copy()
            rois_scaled[:, 0] *= sx
            rois_scaled[:, 1] *= sy
            rois_scaled[:, 2] *= abs(sx)
            rois_scaled[:, 3] *= abs(sy)
            rois_scaled[:, 0] = np.clip(rois_scaled[:, 0], 0.0, W - 1.0)
            rois_scaled[:, 1] = np.clip(rois_scaled[:, 1], 0.0, H - 1.0)
            rois_scaled[:, 2] = np.maximum(0.0, rois_scaled[:, 2])
            rois_scaled[:, 3] = np.maximum(0.0, rois_scaled[:, 3])
            roi_table = rois_scaled
            if (abs(sx - sy) > 1e-6):
                print(f"[roi_stream] ROI file resolution {roiW}x{roiH} scaled to {W}x{H} (non-uniform: sx={sx:.3f}, sy={sy:.3f}).")
        
    # Build ROI masks for this resolution
    roi = ROISet(height=H, width=W, table=roi_table)
    if shared is not None:
        shared.rois = roi.table
        shared.resolution = (W, H)

    # HDF5 writer setup
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path.cwd() / f"traces_{ts}.h5"
    desired_out = Path(out_path)
    resolved_out = _prepare_output_path(desired_out)
    if resolved_out != desired_out:
        print(f"[roi_stream] Output exists; writing to {resolved_out}")
    out_path = resolved_out

    meta = {
        'resolution': np.array([W, H], dtype=np.int32),
        'source': str(source),
    }
    writer = H5TracesWriter(str(out_path), roi.table, meta, chunk_frames=int(opts.frames_per_chunk))

    # Stats and buffers
    tic0 = time.perf_counter()
    last_print = tic0
    frames_seen = 0
    frametimes: list[float] = []
    max_ft = max(2 * opts.frames_per_chunk, 300)

    pending_t = []  # list of float
    pending_means = []  # list of np.ndarray rows

    # Process first frame (already decoded)
    t = time.perf_counter() - tic0
    means0 = roi.compute_means(f16_0)
    frames_seen += 1
    frametimes.append(t)
    pending_t.append(t)
    pending_means.append(means0)
    if shared is not None:
        shared.traces.append(t, means0)
        shared.update_frame(f16_0, (W, H), preview8=preview8_0)

    # Main loop
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = src.read()
            if not ok or frame is None:
                break
            f16 = to_uint16_gray(frame)
            preview8 = np.right_shift(f16, 8).astype(np.uint8, copy=False)
            means = roi.compute_means(f16)

            t = time.perf_counter() - tic0
            frames_seen += 1
            frametimes.append(t)
            if len(frametimes) > max_ft:
                frametimes = frametimes[-max_ft:]

            pending_t.append(t)
            pending_means.append(means)
            if shared is not None:
                shared.traces.append(t, means)
                shared.update_frame(f16, (W, H), preview8=preview8)

            # Flush chunk
            if len(pending_t) >= opts.frames_per_chunk:
                writer.append(np.asarray(pending_t, dtype=np.float64), np.vstack(pending_means))
                pending_t.clear()
                pending_means.clear()

            # Print FPS periodically
            now = time.perf_counter()
            if (now - last_print) >= opts.print_fps_period:
                fps = float('nan')
                if len(frametimes) >= 2:
                    fps = (len(frametimes) - 1) / max(frametimes[-1] - frametimes[0], 1e-9)
                print(f"[{frametimes[-1]:7.3f}s] FPS: {fps:5.1f}   frames={frames_seen}")
                last_print = now

            if opts.max_frames and frames_seen >= opts.max_frames:
                break
    except KeyboardInterrupt:
        print("[roi_stream] Interrupted by user; finalizing…")
    finally:
        # Flush remainder
        if pending_t:
            writer.append(np.asarray(pending_t, dtype=np.float64), np.vstack(pending_means))
        elapsed = time.perf_counter() - tic0
        avg_fps = frames_seen / max(elapsed, 1e-9)
        summary = {
            'frames_seen': int(frames_seen),
            'frames_dropped': int(0),
            'elapsed_sec': float(elapsed),
            'avg_fps': float(avg_fps),
        }
        writer.finalize(summary)
        src.release()

    print(f"[roi_stream] HDF5 saved: {out_path}")
    return out_path
