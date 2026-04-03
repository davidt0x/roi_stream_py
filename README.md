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
- Probe Hamamatsu cameras after SDK setup:
  - `roi_stream doctor hamamatsu`
- Windows backend/device name tips:
  - Use `--backend dshow` or `--backend msmf`.
  - DirectShow device names can be opened with `--source "video=OBS Virtual Camera" --backend dshow`.

Hamamatsu SDK setup

- Install the vendor SDK samples into a per-user directory:
  - `roi_stream install-hamamatsu-sdk`
- This downloads the SDK archive from Hamamatsu after an explicit confirmation prompt.
- The installer records the resolved SDK path under the user data directory and does not write into `site-packages`.
- The archive currently contains the vendor Python sample files, but not necessarily the DCAM runtime library itself. Use:
  - `roi_stream doctor hamamatsu`
  to confirm whether both the SDK files and the runtime are available on the machine.
- Hamamatsu defaults are intentionally generic. Rig-specific camera settings should be expressed explicitly under `capture:`. Example:
  ```yaml
  capture:
    driver: hamamatsu_sdk
    pixel_type: MONO16
    readout_speed: FASTEST
    frame_rate: 80
    exposure_sec: 0.0125
    binning: 2
    roi: [0, 320, 1152, 476]
    output_triggers:
      - line: 2
        source: EXPOSURE
        polarity: POSITIVE
      - line: 3
        source: EXPOSURE
        polarity: POSITIVE
  ```
- A concrete rig-specific example is included at `examples/polina_rig_config.yaml`.

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
