from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys
import numpy as np
import threading

from .config import StreamOptions, load_circles, parse_format, parse_source
from .stream import run_stream
from .shared import SharedState, TraceRing


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roi_stream", description="ROI streaming: capture, compute circular ROI means, write HDF5")
    p.add_argument("--source", required=True, help="Device index (e.g., 2) or video file path")
    p.add_argument("--rois", required=True, help="Path to ROI circles CSV/JSON (rows: xc,yc,r)")
    p.add_argument("--format", default=None, help="Format string like '1280x720@60' (best effort)")
    p.add_argument("--backend", default="any", choices=["any","v4l2","msmf","dshow","gstreamer","ffmpeg"], help="Capture backend hint for device indexes")
    p.add_argument("--out", default=None, help="Output HDF5 path (default: traces_YYYYMMDD_HHMMSS.h5)")
    p.add_argument("--frames-per-chunk", type=int, default=240, help="Rows per HDF5 chunk append")
    p.add_argument("--trace-buffer-sec", type=float, default=600.0, help="GUI ring buffer seconds (reserved)")
    p.add_argument("--print-fps-period", type=float, default=1.0, help="Seconds between FPS logs")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    p.add_argument("--gui", action="store_true", help="Launch GUI viewer (not yet wired to stream)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    src = parse_source(args.source)
    circles = load_circles(args.rois)

    opts = StreamOptions(
        frames_per_chunk=args.frames_per_chunk,
        print_fps_period=args.print_fps_period,
        trace_buffer_sec=args.trace_buffer_sec,
        max_frames=args.max_frames,
    )
    fmt = parse_format(args.format)

    if args.gui:
        # GUI mode: run stream in a background thread and launch the viewer
        try:
            from .gui_app import run_gui
        except Exception as e:
            print("[roi_stream] GUI requested but Dear PyGui not available. Install extras: pip install '.[gui]'")
            print(f"[roi_stream] Import error: {e}")
            return 2

        # Determine ring buffer length
        # Prefer requested FPS from format, else fall back to 60
        fps_hint = fmt[2] if fmt and fmt[2] else 60.0
        ring_len = int(max(300, opts.trace_buffer_sec * max(1.0, float(fps_hint))))
        shared = SharedState(traces=TraceRing(k=int(circles.shape[0]), maxlen=ring_len))
        stop_event = threading.Event()
        result: dict[str, Optional[Path]] = {"out": None}

        def _worker():
            try:
                out = run_stream(src, circles, args.out, opts, fmt, backend=args.backend, shared=shared, stop_event=stop_event)
                result["out"] = out
            except Exception as e:
                print(f"[roi_stream] worker error: {e}")
            finally:
                stop_event.set()

        th = threading.Thread(target=_worker, name="roi_stream_worker", daemon=True)
        th.start()
        # Run GUI loop (blocking)
        run_gui(shared, stop_event)
        # Ensure worker stops and finalize
        stop_event.set()
        th.join(timeout=5.0)
        if result["out"] is not None:
            print(str(result["out"]))
        return 0

    # Headless mode
    out = run_stream(src, circles, args.out, opts, fmt, backend=args.backend)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
