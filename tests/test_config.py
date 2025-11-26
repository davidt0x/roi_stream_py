from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roi_stream.config import (
    ROISettings,
    RuntimeConfig,
    load_rois_with_meta,
    load_runtime_config,
    parse_source,
)


def test_load_runtime_config_with_shapes(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
source: 0
backend: msmf
format: 1920x1080@30
stream:
  frames_per_chunk: 120
  print_fps_period: 0.5
  trace_buffer_sec: 120.0
  max_frames: 500
rois:
  resolution: [1920, 1080]
  shapes:
    - name: center
      type: circle
      center: [960.0, 540.0]
      radius: 80.0
    - name: ellipse_a
      type: ellipse
      center: [320.0, 240.0]
      radii: [100.0, 60.0]
      angle_deg: -30.0
        """,
        encoding="utf-8",
    )

    cfg = load_runtime_config(cfg_path)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.backend == "msmf"
    assert cfg.source == 0
    assert cfg.format == "1920x1080@30"
    assert cfg.stream.frames_per_chunk == 120
    assert cfg.stream.max_frames == 500
    assert cfg.rois is not None
    assert isinstance(cfg.rois, ROISettings)
    assert cfg.rois.table.shape == (2, 5)
    assert np.isclose(cfg.rois.table[0, 2], cfg.rois.table[0, 3])
    assert cfg.rois.labels == ["center", "ellipse_a"]


def test_load_rois_with_meta_yaml(tmp_path: Path):
    yaml_path = tmp_path / "rois.yaml"
    yaml_path.write_text(
        """
source: "video.mp4"
rois:
  shapes:
    - center: [100.0, 120.0]
      radius: 25.0
    - center: {x: 200.0, y: 220.0}
      rx: 30.0
      ry: 15.0
      angle: 45.0
        """,
        encoding="utf-8",
    )
    table, res = load_rois_with_meta(yaml_path)
    assert res is None
    assert table.shape == (2, 5)
    assert np.allclose(table[0, 2], table[0, 3])
    assert not np.allclose(table[1, 2], table[1, 3])


def test_roi_labels_default_numeric(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
source: "clip.mp4"
rois:
  shapes:
    - center: [10, 20]
      radius: 5
    - type: ellipse
      center: [30, 40]
      radii: [8, 3]
        """,
        encoding="utf-8",
    )

    cfg = load_runtime_config(cfg_path)
    assert cfg.rois is not None
    assert cfg.rois.labels == ["0", "1"]


def test_parse_source_numeric_and_path(tmp_path: Path):
    assert parse_source("5") == 5
    assert parse_source("+3") == 3
    assert parse_source("-1") == -1
    path = "video/example.mp4"
    assert parse_source(path) == path
