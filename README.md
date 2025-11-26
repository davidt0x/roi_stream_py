ROI Stream (Python)

Quickstart

- Install (with GUI extras optional):
  - `pip install -e .`
  - GUI: `pip install -e '.[gui]'`
- Run headless on a video file with a dedicated config:
  - `roi_stream --config examples/sample_config.yaml --max-frames 300`
- Launch with GUI (live traces):
  - `roi_stream --config examples/sample_config.yaml --gui`

Device probing

- Probe indexes and report resolution/FPS:
  - `roi_stream_devices --backend any --max-index 5`
- Windows backend/device name tips:
  - Use `--backend dshow` or `--backend msmf`.
  - DirectShow device names can be opened with `--source "video=OBS Virtual Camera" --backend dshow`.

WSL notes

- Access to host webcams from Linux/WSL is generally unavailable.
- Prefer file sources or run on native Windows Python when using webcams.

ROI shapes format

- ROIs can be circles or ellipses. Internally each ROI is `[xc, yc, rx, ry, angle_deg]`.
- Provide shapes via YAML. Names are optional; numbered indices are used when omitted.
- Example YAML config with mixed shapes: `examples/sample_config.yaml`.

YAML configuration

- Define capture settings, stream options, and ROIs together.
- Fields set via CLI flags override matching config values when provided explicitly.
- Minimal structure:
  ```yaml
  source: "path/to/video.mp4"
  format: 1280x720@60
  stream:
    frames_per_chunk: 240
  rois:
    shapes:
      - type: circle
        center: [640, 360]
        radius: 120
      - type: ellipse
        center: [320, 220]
        radii: [140, 60]
        angle_deg: -20
  ```

Generate random ROIs

- Create a YAML file with 35 random circles for a given resolution (default 1280x720):
  - `roi_stream_make_random_rois --width 1280 --height 720 --count 35 --out examples/rois_random.yaml`
- A sample file is included: `examples/rois_random.yaml`.
