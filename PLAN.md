# Python Port Plan — roi_stream

This plan outlines the Python port of the MATLAB ROI streaming toolkit under `matlab/`, targeting a lightweight, cross‑platform CLI (`roi_stream`) and an optional GUI (`roi_stream_gui`) built with Dear PyGui.

Goals

- Recreate core MATLAB features: live capture, circular ROI means, real‑time plotting, and HDF5 logging.
- Provide a simple CLI first; GUI is optional and decoupled from acquisition.
- Keep the code modular, typed, and testable on prerecorded video.
- Ship small, clear defaults; avoid hardware‑specific dependencies.

Scope (initial)

- Input: webcam or video file via OpenCV (`cv2.VideoCapture`).
- Processing: uint16 grayscale conversion, circular ROI mean intensities.
- Output: HDF5 with extendible datasets and metadata compatible with the MATLAB layout.
- Optional: live preview + scrolling trace plot in Dear PyGui.
- Out of scope (initial): vendor SDK integrations (e.g., Hamamatsu/DCAM), dF/F, offline viewer UI. These can be added later.

Architecture

- `src/roi_stream/cli.py` — CLI entry point (`roi_stream`).
- `src/roi_stream/stream.py` — orchestrates capture loop, ROI compute, batching, stats.
- `src/roi_stream/capture.py` — OpenCV capture backend (webcam index or file path).
- `src/roi_stream/roi.py` — ROI data structures, mask precompute, mean calculation.
- `src/roi_stream/writer.py` — HDF5 writer (extendible datasets, attributes).
- `src/roi_stream/config.py` — dataclasses for options and ROI definitions, simple file loaders (CSV/JSON).
- `src/roi_stream/gui_app.py` — Dear PyGui application (preview + traces), adapted from `roi_stream_gui.py` stub.

Data model and file format

- ROI circles: `N×3` float64 `[xc yc r]` in pixels, origin at top‑left.
- Frame: grayscale `uint16` after conversion.
- HDF5 datasets (extendible):
  - `/time` — `(rows, 1)` double (seconds since start)
  - `/roi/means` — `(rows, K)` float32
  - `/roi/dff` — optional `(rows, K)` float32 (not written initially)
- Static datasets:
  - `/roi/circles` — `(K, 3)` float64
- Root attributes (strings/ints): `created_with`, `source`, `resolution [W H]`, `start_iso8601`, `end_iso8601`, `frames_seen`, `frames_dropped`, `elapsed_sec`, `avg_fps`.

CLI (proposed)

- Command: `roi_stream` (headless by default, `--gui` to launch GUI)
- Options (subset):
  - `--source` (int webcam index or path to video file)
  - `--format` (e.g., `1280x720@60`, best‑effort; maps to OpenCV props)
  - `--rois` (path to CSV/JSON with `[xc,yc,r]` rows)
  - `--out` (HDF5 path; default `traces_YYYYMMDD_HHMMSS.h5`)
  - `--frames-per-chunk` (default 240)
  - `--trace-buffer-sec` (GUI ring buffer length; default 600)
  - `--print-fps-period` (seconds; default 1.0)
  - `--gui` (launch Dear PyGui viewer)
  - `--max-frames` (optional cap for offline tests)
- Behavior:
  - Ctrl+C stops gracefully, flushes pending rows, finalizes attributes.
  - For webcams, sets best‑effort FPS and exposure via OpenCV properties if supported.
  - Local testing note: device index can vary by system. Use `roi_stream_devices` to probe indexes and backends; on Windows, DirectShow (`--backend dshow`) or MSMF (`--backend msmf`) may work. Device names like `"video=OBS Virtual Camera"` can be opened with `--backend dshow`.
  - Under WSL, camera devices are generally unavailable; prefer file sources for tests.

- Utilities:
  - `roi_stream_devices` — probes device indexes for the selected backend and prints resolution/FPS.

GUI (Dear PyGui)

- Preview: latest grayscale frame with ROI overlays.
- Traces: scrolling window of ROI means (fixed count of points, auto‑fit Y).
- Controls: start/stop, window size, ROI selection, optional autoscale toggle.
- Architecture: capture/compute on a worker thread; GUI polls a lock‑free ring buffer for smooth rendering (no heavy work in the GUI thread).
- Starting point: adapt existing `src/roi_stream/roi_stream_gui.py` random‑walk demo to consume shared buffers.

Core algorithms

- Grayscale to `uint16` conversion mirrors MATLAB `to_uint16_gray` logic for predictable ranges (0..65535) from common input types.
- ROI mask precompute: for each circle, build a boolean mask or flat indices once per resolution; cache `npix` counts.
- Per‑frame means: `means[k] = frame16[mask_k].mean(dtype=np.float64)` → cast to `np.float32`.
- Batching: accumulate `frames_per_chunk` rows before writing; allow partial flush on stop.
- Stats: track `frames_seen`, `frametimes` ring, instantaneous FPS over recent window, dropped frames (best‑effort).

Threading and performance

- 1 capture thread (pulls frames, converts to `uint16`, computes ROI means, appends to batch + ring buffers).
- GUI thread only reads ring buffers and updates plots; image/plot updates throttled (e.g., 4–10 Hz).
- Use `cv2.CAP_PROP_*` for FPS/exposure when available; ignore failures gracefully.
- Avoid per‑frame allocations in the hot path; reuse buffers where possible.

Testing strategy (pytest)

- Framework and layout:
  - Use `pytest` with tests placed under `tests/`.
  - Naming: files `tests/test_*.py`, functions `test_*`.
  - Prefer fixtures over ad‑hoc setup; use `tmp_path` for temporary HDF5 outputs.
- Unit tests:
  - `tests/test_roi.py`: mask generation on tiny images; edge cases (border‑touching circles, radius 0/1, out‑of‑bounds clipping).
  - `tests/test_writer.py`: append/finalize; verify dataset shapes, dtypes, extendible growth, and root attributes using `h5py`.
  - `tests/test_convert.py`: `to_uint16_gray` parity on representative inputs (uint8/uint16/float in gray/RGB); range and monotonicity checks.
- Integration tests (offline):
  - `tests/test_integration_offline.py`: run the stream on `matlab/test_circle_1280x720_60fps.mp4` with two ROIs for ~200 frames (`--max-frames`), assert row count, strictly increasing timestamps, and `(rows, K)` shapes.
  - Mark with `@pytest.mark.offline` so these can be selected in CI: `pytest -m offline -q`.
- Optional virtual camera test:
  - `tests/test_integration_virtualcam.py`: attempt a short capture from device index `2` (virtual camera) with `--max-frames` small (e.g., 120). Skip if `cv2.VideoCapture(2).isOpened()` is false.
  - Mark with `@pytest.mark.virtualcam`; run locally via `pytest -m virtualcam -q`.
- Pytest configuration:
  - Add to `pyproject.toml` under `[tool.pytest.ini_options]`:
    - `testpaths = ["tests"]`
    - `markers = ["offline: tests using prerecorded video", "virtualcam: tests using local virtual camera at index 2", "gui: interactive viewer tests (skipped in CI)"]`
  - Default run: `pytest -q` (no hardware required). Skip `gui` by default.
  - WSL guidance: in WSL, run only offline tests by default; skip `virtualcam`.
- CI guidance:
  - Run unit tests and offline integration on a worker without camera access.
  - Cache test artifacts only in the temp directory; do not write outside `tmp_path`.

Packaging and dependencies

- `pyproject.toml`:
  - Runtime deps: `numpy`, `opencv-python`, `h5py`.
  - Optional GUI: move `imgui-bundle` to an extra, e.g., `[project.optional-dependencies].gui`.
  - Dev/test extra: `[project.optional-dependencies].dev = ["pytest", "pytest-cov"]`.
  - Pytest config: add `[tool.pytest.ini_options]` with `testpaths` and `markers` including `offline`, `virtualcam`, and `gui`.
  - Entry points:
    - `[project.scripts]`
      - `roi_stream = "roi_stream.cli:main"`
      - `roi_stream_gui = "roi_stream.gui_app:main"`
      - `roi_stream_devices = "roi_stream.devices:main"`
- Keep `py.typed` for typing consumers.

Milestones and deliverables

1) Skeleton + CLI stub (completed)
   - Modules scaffolded: `capture`, `roi`, `writer`, `stream`, `cli`.
   - Config parsing, ROI loader, HDF5 writer implemented; file streaming validated.
   - Windows device capture validated (index varies; `--backend dshow/msmf`).
   - Utility `roi_stream_devices` added for probing indexes.
   - Fix applied: HDF5 `/time` assignment keeps 2D shape to avoid broadcasting errors.

2) Live preview + traces (next)
   - Shared ring buffer for timeseries and latest frame.
   - imgui_bundle app adapted from stub; plot K series and image overlay.
   - `--gui` flag to launch viewer alongside streaming.
   - Deliverable: smooth scrolling plot and periodic image refresh; CPU usage reasonable.

3) Camera device support (ongoing)
   - Backend hints (`--backend dshow/msmf/v4l2`) and device name support (`"video=..."`).
   - Resolution negotiation from `--format` string (`WxH@FPS`) or probing.
   - Deliverable: end‑to‑end on a typical USB camera at 30–60 FPS across Windows and Linux (non‑WSL).

4) Robustness + metadata parity (day 4–5)
   - Root attributes parity with MATLAB writer; consistent timestamping.
   - Error handling, logging, graceful shutdown; finalize HDF5 summary.
   - Deliverable: HDF5s readable by existing MATLAB `h5_traces_viewer.m` for `/time`, `/roi/means`, `/roi/circles`.

5) Tests + docs
   - Add unit + offline integration tests (default in CI/WSL).
   - Optional `virtualcam` tests for local Windows.
   - Update `README.md` with Windows backend tips, WSL guidance, device probing usage, quickstart, and troubleshooting.

Mapping from MATLAB

- `roi_stream.m` → `stream.py` (loop) + `capture.py` (frames) + `roi.py` (masks/means) + `writer.py` (HDF5).
- `stop_roi_stream.m` → handled by signal/exception cleanup in `stream.py` and finalized attributes in `writer.py`.
- `H5TracesWriter.m` → `writer.py` with the same dataset names and attributes.
- `roi_stream_gui.m` → `gui_app.py` with Dear PyGui; polling timer replaces MATLAB timer.
- `to_uint16_gray.m` → `roi.py` or `stream.py` helper function with equivalent semantics.

Assumptions and risks

- OpenCV properties for FPS/exposure are device/driver dependent; implement best‑effort with clear warnings.
- Circle masks assume ROIs in‑bounds; plan to clip masks to the frame rectangle.
- Performance with many ROIs may require optimization (e.g., vectorized mask application or integral images). Start simple, profile, and iterate if needed.

Immediate next actions

- Wire GUI to streaming ring buffers; add ROI overlays and series plotting.
- Add pytest tests under `tests/` (roi masks, writer append/finalize, conversion parity, offline integration).
- Add README sections: device probing (`roi_stream_devices`), Windows backends and device names, WSL file‑based testing.
- Verify local environment captures from Windows device index (or name) and document with examples.
