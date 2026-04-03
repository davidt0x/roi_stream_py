from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import threading

from .config import RuntimeConfig, StreamOptions, load_runtime_config, parse_format, parse_source
from .hamamatsu_sdk import (
    ENV_SDK_PYTHON_DIR,
    HAMAMATSU_SDK_URL,
    install_hamamatsu_sdk,
    probe_hamamatsu_status,
)
from .stream import run_stream
from .shared import SharedState, TraceRing


def _configure_stream_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", required=True, help="Path to YAML config with capture/ROI definitions")
    p.add_argument("--source", help="Device index (e.g., 2) or video file path")
    p.add_argument("--format", default=None, help="Format string like '1280x720@60' (best effort)")
    p.add_argument("--backend", default="any", choices=["any","v4l2","msmf","dshow","gstreamer","ffmpeg","hamamatsu_sdk"], help="Capture backend hint for device indexes")
    p.add_argument("--out", default=None, help="Output HDF5 path (default: traces_YYYYMMDD_HHMMSS.h5)")
    p.add_argument("--frames-per-chunk", type=int, default=240, help="Rows per HDF5 chunk append")
    p.add_argument("--trace-buffer-sec", type=float, default=600.0, help="GUI ring buffer seconds (reserved)")
    p.add_argument("--print-fps-period", type=float, default=1.0, help="Seconds between FPS logs")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    p.add_argument("--gui", action="store_true", help="Launch GUI viewer (not yet wired to stream)")
    return p


def build_stream_parser() -> argparse.ArgumentParser:
    return _configure_stream_parser(
        argparse.ArgumentParser(
            prog="roi_stream stream",
            description="ROI streaming: capture frames, compute ROI means (circles or ellipses), write HDF5",
        )
    )


def _configure_install_hamamatsu_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--yes", action="store_true", help="Accept the vendor download prompt non-interactively")
    p.add_argument("--url", default=HAMAMATSU_SDK_URL, help="Override the Hamamatsu SDK archive URL")
    p.add_argument("--install-dir", default=None, help="Override the per-user Hamamatsu SDK install root")
    return p


def build_install_hamamatsu_parser() -> argparse.ArgumentParser:
    return _configure_install_hamamatsu_parser(
        argparse.ArgumentParser(
            prog="roi_stream install-hamamatsu-sdk",
            description="Download and install Hamamatsu DCAM SDK Python samples into a per-user directory",
        )
    )


def _configure_doctor_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("target", choices=["hamamatsu"], help="Diagnostic target")
    p.add_argument("--sdk-python-dir", default=None, help="Explicit Hamamatsu SDK Python samples directory to probe")
    p.add_argument("--install-dir", default=None, help="Override the per-user Hamamatsu SDK install root")
    return p


def build_doctor_parser() -> argparse.ArgumentParser:
    return _configure_doctor_parser(
        argparse.ArgumentParser(prog="roi_stream doctor", description="Inspect optional backend setup")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roi_stream",
        description="ROI streaming and optional Hamamatsu SDK management",
    )
    subparsers = parser.add_subparsers(dest="command")

    stream_parser = _configure_stream_parser(
        subparsers.add_parser(
            "stream",
            help="Capture frames, compute ROI means, and write HDF5",
            description="ROI streaming: capture frames, compute ROI means (circles or ellipses), write HDF5",
        )
    )
    stream_parser.set_defaults(command="stream")

    install_parser = _configure_install_hamamatsu_parser(
        subparsers.add_parser(
            "install-hamamatsu-sdk",
            help="Download the Hamamatsu SDK Python samples into a per-user directory",
            description="Download and install Hamamatsu DCAM SDK Python samples into a per-user directory",
        )
    )
    install_parser.set_defaults(command="install-hamamatsu-sdk")

    doctor_parser = _configure_doctor_parser(
        subparsers.add_parser(
            "doctor",
            help="Inspect optional backend setup",
            description="Inspect optional backend setup",
        )
    )
    doctor_parser.set_defaults(command="doctor")

    return parser


def _run_install_hamamatsu_args(args: argparse.Namespace) -> int:
    install_dir = Path(args.install_dir).expanduser() if args.install_dir else None
    status = install_hamamatsu_sdk(url=str(args.url), base_dir=install_dir, accept_license=bool(args.yes))
    print(f"[roi_stream] Hamamatsu SDK files installed under: {status.install_root}")
    if status.python_dir is not None:
        print(f"[roi_stream] SDK Python directory: {status.python_dir}")
        print(f"[roi_stream] Set {ENV_SDK_PYTHON_DIR} to override this path if needed.")
    print(f"[roi_stream] Runtime status: {status.runtime_message}")
    if status.camera_count is not None:
        print(f"[roi_stream] Detected Hamamatsu cameras: {status.camera_count}")
    return 0 if status.installed else 2


def _run_doctor_args(args: argparse.Namespace) -> int:
    install_dir = Path(args.install_dir).expanduser() if args.install_dir else None
    status = probe_hamamatsu_status(explicit=args.sdk_python_dir, base_dir=install_dir)
    print(f"[roi_stream] Hamamatsu install root: {status.install_root}")
    print(f"[roi_stream] SDK installed: {'yes' if status.installed else 'no'}")
    if status.metadata is not None:
        print(f"[roi_stream] Recorded SDK version: {status.metadata.version}")
        print(f"[roi_stream] Recorded SDK URL: {status.metadata.url}")
    if status.python_dir is not None:
        print(f"[roi_stream] SDK Python directory: {status.python_dir}")
    print(f"[roi_stream] Runtime status: {status.runtime_message}")
    if status.camera_count is not None:
        print(f"[roi_stream] Detected Hamamatsu cameras: {status.camera_count}")
    return 0 if status.installed else 2


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(argv) if argv is not None else list(os.sys.argv[1:])
    parser = build_parser()
    known_commands = {"stream", "install-hamamatsu-sdk", "doctor"}
    if argv and argv[0] not in known_commands and argv[0] not in {"-h", "--help"}:
        argv = ["stream", *argv]
    args = parser.parse_args(argv)
    if args.command == "install-hamamatsu-sdk":
        return _run_install_hamamatsu_args(args)
    if args.command == "doctor":
        return _run_doctor_args(args)
    if args.command not in {None, "stream"}:
        parser.error(f"Unsupported command: {args.command}")
    stream_defaults = build_stream_parser()

    cfg: RuntimeConfig = load_runtime_config(args.config)
    if cfg.rois is None:
        parser.error("Config must define at least one ROI shape")

    roi_table = cfg.rois.table
    roi_src_res = cfg.rois.resolution

    # Resolve source preference: CLI overrides config
    source_value: Optional[Union[int, str]] = args.source
    if source_value is None and cfg.source is not None:
        source_value = cfg.source
    if source_value is None:
        parser.error("Source must be provided via --source or --config")
    if isinstance(source_value, int):
        src = source_value
    else:
        src = parse_source(str(source_value))

    # Resolve format preference (string)
    fmt_str = args.format
    if fmt_str is None and cfg.format:
        fmt_str = str(cfg.format)
    fmt = parse_format(fmt_str)

    # Backend resolution: CLI overrides config unless left as default
    backend_default = stream_defaults.get_default("backend")
    backend = args.backend
    if cfg.backend and backend == backend_default:
        backend = str(cfg.backend)
    if cfg.hamamatsu_sdk_python_dir and not os.environ.get(ENV_SDK_PYTHON_DIR):
        os.environ[ENV_SDK_PYTHON_DIR] = cfg.hamamatsu_sdk_python_dir

    # Output path
    out_path = args.out
    if out_path is None and cfg.output:
        out_path = str(cfg.output)

    # Stream options merging (config base, CLI override when not default)
    defaults = {
        "frames_per_chunk": stream_defaults.get_default("frames_per_chunk"),
        "print_fps_period": stream_defaults.get_default("print_fps_period"),
        "trace_buffer_sec": stream_defaults.get_default("trace_buffer_sec"),
        "max_frames": stream_defaults.get_default("max_frames"),
    }
    stream_base = cfg.stream

    def _resolve(name: str) -> Union[int, float]:
        cli_val = getattr(args, name)
        default_val = defaults[name]
        base_val = getattr(stream_base, name)
        if cli_val is not None and cli_val != default_val:
            return cli_val
        return base_val

    opts = StreamOptions(
        frames_per_chunk=int(_resolve("frames_per_chunk")),
        print_fps_period=float(_resolve("print_fps_period")),
        trace_buffer_sec=float(_resolve("trace_buffer_sec")),
        max_frames=int(_resolve("max_frames")),
    )

    if args.gui:
        try:
            from .gui_app import run_gui
        except Exception as e:
            print("[roi_stream] GUI requested but Dear PyGui not available. Install extras: pip install '.[gui]'")
            print(f"[roi_stream] Import error: {e}")
            return 2

        fps_hint = fmt[2] if fmt and fmt[2] else 60.0
        ring_len = int(max(300, opts.trace_buffer_sec * max(1.0, float(fps_hint))))
        shared = SharedState(traces=TraceRing(k=int(roi_table.shape[0]), maxlen=ring_len))
        stop_event = threading.Event()
        result: dict[str, Optional[Path]] = {"out": None}

        def _worker():
            try:
                out = run_stream(
                    src,
                    roi_table,
                    out_path,
                    opts,
                    fmt,
                    backend=backend,
                    shared=shared,
                    stop_event=stop_event,
                    roi_src_resolution=roi_src_res,
                    hamamatsu_settings=cfg.hamamatsu,
                )
                result["out"] = out
            except Exception as e:
                print(f"[roi_stream] worker error: {e}")
            finally:
                stop_event.set()

        th = threading.Thread(target=_worker, name="roi_stream_worker", daemon=True)
        th.start()
        run_gui(shared, stop_event)
        stop_event.set()
        th.join(timeout=5.0)
        if result["out"] is not None:
            print(str(result["out"]))
        return 0

    out = run_stream(
        src,
        roi_table,
        out_path,
        opts,
        fmt,
        backend=backend,
        roi_src_resolution=roi_src_res,
        hamamatsu_settings=cfg.hamamatsu,
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
