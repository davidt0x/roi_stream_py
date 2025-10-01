ROI Stream (Python)

Quickstart

- Install (with GUI extras optional):
  - `pip install -e .`
  - GUI: `pip install -e '.[gui]'`
- Run headless on a video file:
  - `roi_stream --source path/to/video.mp4 --rois examples/rois.csv --format 1280x720@60 --max-frames 300`
- Launch with GUI (live traces):
  - `roi_stream --source path/to/video.mp4 --rois examples/rois.csv --gui`

Device probing

- Probe indexes and report resolution/FPS:
  - `roi_stream_devices --backend any --max-index 5`
- Windows backend/device name tips:
  - Use `--backend dshow` or `--backend msmf`.
  - DirectShow device names can be opened with `--source "video=OBS Virtual Camera" --backend dshow`.

WSL notes

- Access to host webcams from Linux/WSL is generally unavailable.
- Prefer file sources or run on native Windows Python when using webcams.

ROI circles format

- CSV or JSON with rows `[xc, yc, r]` in pixels; origin at top-left.
- Example: `examples/rois.csv`.
