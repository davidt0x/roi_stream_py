from __future__ import annotations

import argparse
from typing import List
from .capture import FrameSource


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="roi_stream_devices", description="Probe camera device indexes for the selected backend")
    p.add_argument("--backend", default="any", choices=["any","v4l2","msmf","dshow","gstreamer","ffmpeg"], help="Capture backend hint")
    p.add_argument("--max-index", type=int, default=10, help="Probe indexes in [0..N]")
    p.add_argument("--format", default=None, help="Optional format 'WxH@FPS' to request when opening")
    args = p.parse_args(argv)

    # Lazy import to reuse parser in config if needed later
    from .config import parse_format
    w, h, fps = parse_format(args.format)

    print(f"Probing backend={args.backend} indexes 0..{args.max_index}")
    found = 0
    for idx in range(0, max(0, args.max_index) + 1):
        fs = FrameSource(idx, width=w, height=h, fps=fps, backend=args.backend)
        ok = fs.open()
        if ok:
            W, H = fs.get_resolution()
            fps_r = fs.get_fps()
            print(f"  index {idx}: OPENED  {W}x{H} @ {fps_r:.1f} fps")
            found += 1
            fs.release()
        else:
            print(f"  index {idx}: not available")
    if found == 0:
        print("No devices opened. If on Windows with DirectShow, try providing the device by name: --source 'video=OBS Virtual Camera' --backend dshow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

