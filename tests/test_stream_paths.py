from __future__ import annotations

from pathlib import Path

import roi_stream.stream as stream_module
from roi_stream.stream import _prepare_output_path


def test_prepare_output_path_no_conflict(tmp_path):
    base = tmp_path / "result.h5"
    resolved = _prepare_output_path(base)
    assert resolved == base


def test_prepare_output_path_conflict(tmp_path, monkeypatch):
    base = tmp_path / "result.h5"
    base.write_text("existing")

    monkeypatch.setattr(stream_module.time, "strftime", lambda fmt: "20240101_120000")
    resolved = _prepare_output_path(base)

    assert resolved != base
    assert resolved.name == "result_20240101_120000.h5"
    assert not resolved.exists()


def test_prepare_output_path_conflict_multiple(tmp_path, monkeypatch):
    base = tmp_path / "result"
    base.write_text("existing")
    (tmp_path / "result_20240101_120000").write_text("other")

    monkeypatch.setattr(stream_module.time, "strftime", lambda fmt: "20240101_120000")
    resolved = _prepare_output_path(base)

    assert resolved.name == "result_20240101_120000_1"
    assert not resolved.exists()
