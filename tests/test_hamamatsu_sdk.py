from __future__ import annotations

from pathlib import Path
import sys
import zipfile

import pytest

from roi_stream import cli
from roi_stream.config import load_runtime_config
from roi_stream.hamamatsu_sdk import (
    HAMAMATSU_SDK_ARCHIVE_ROOT,
    HAMAMATSU_SDK_VERSION,
    ENV_SDK_PYTHON_DIR,
    apply_hamamatsu_settings,
    build_hamamatsu_settings,
    discover_sdk_python_dir,
    get_sdk_metadata_path,
    install_hamamatsu_sdk,
    probe_hamamatsu_status,
)


def _make_fake_sdk_archive(tmp_path: Path) -> Path:
    archive = tmp_path / "hamamatsu_sdk.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        root = f"{HAMAMATSU_SDK_ARCHIVE_ROOT}/dcamsdk4/samples/python"
        zf.writestr(
            f"{root}/dcamapi4.py",
            "VALUE = 1\n",
        )
        zf.writestr(
            f"{root}/dcam.py",
            "\n".join(
                [
                    "class _Err:",
                    "    def __str__(self):",
                    "        return 'OK'",
                    "",
                    "class Dcamapi:",
                    "    _count = 2",
                    "    @classmethod",
                    "    def init(cls):",
                    "        return True",
                    "    @classmethod",
                    "    def get_devicecount(cls):",
                    "        return cls._count",
                    "    @classmethod",
                    "    def lasterr(cls):",
                    "        return _Err()",
                    "    @classmethod",
                    "    def uninit(cls):",
                    "        return True",
                    "",
                    "class Dcam:",
                    "    def __init__(self, index=0):",
                    "        self.index = index",
                    "    def lasterr(self):",
                    "        return _Err()",
                    "    def dev_open(self):",
                    "        return True",
                    "    def buf_alloc(self, n):",
                    "        return True",
                    "    def cap_start(self):",
                    "        return True",
                    "    def cap_stop(self):",
                    "        return True",
                    "    def buf_release(self):",
                    "        return True",
                    "    def dev_close(self):",
                    "        return True",
                    "    def wait_capevent_frameready(self, timeout):",
                    "        return False",
                ]
            ) + "\n",
        )
    return archive


def test_load_runtime_config_accepts_capture_driver(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
source: 0
capture:
  driver: hamamatsu_sdk
  sdk_python_dir: /tmp/fake_sdk
rois:
  shapes:
    - center: [10, 20]
      radius: 5
        """,
        encoding="utf-8",
    )

    cfg = load_runtime_config(cfg_path)
    assert cfg.backend == "hamamatsu_sdk"
    assert cfg.hamamatsu_sdk_python_dir == "/tmp/fake_sdk"


def test_install_hamamatsu_sdk_download_extract_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = _make_fake_sdk_archive(tmp_path)
    install_root = tmp_path / "install_root"

    def _fake_urlopen(url: str):
        assert url == "https://example.test/sdk.zip"
        return archive.open("rb")

    monkeypatch.setattr("roi_stream.hamamatsu_sdk.urllib.request.urlopen", _fake_urlopen)
    monkeypatch.delenv(ENV_SDK_PYTHON_DIR, raising=False)
    sys.modules.pop("dcam", None)
    sys.modules.pop("dcamapi4", None)

    status = install_hamamatsu_sdk(
        url="https://example.test/sdk.zip",
        base_dir=install_root,
        accept_license=True,
    )

    assert status.installed
    assert status.python_dir is not None
    assert status.python_dir.name == "python"
    assert status.camera_count == 2

    metadata_path = get_sdk_metadata_path(install_root)
    assert metadata_path.exists()
    metadata_text = metadata_path.read_text(encoding="utf-8")
    assert HAMAMATSU_SDK_VERSION in metadata_text
    assert "https://example.test/sdk.zip" in metadata_text

    resolved = discover_sdk_python_dir(base_dir=install_root)
    assert resolved == status.python_dir


def test_probe_hamamatsu_status_reports_missing_sdk(tmp_path: Path):
    status = probe_hamamatsu_status(base_dir=tmp_path / "no_sdk_here")
    assert not status.installed
    assert status.python_dir is None
    assert "install-hamamatsu-sdk" in status.runtime_message


def test_stream_cli_reports_missing_hamamatsu_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
source: 0
backend: hamamatsu_sdk
stream:
  max_frames: 1
rois:
  shapes:
    - center: [10, 20]
      radius: 5
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr("roi_stream.capture.discover_sdk_python_dir", lambda: None)
    monkeypatch.delenv(ENV_SDK_PYTHON_DIR, raising=False)

    with pytest.raises(RuntimeError, match="install-hamamatsu-sdk"):
        cli.main(["--config", str(cfg_path)])


def test_doctor_command_uses_probe_status(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    archive = _make_fake_sdk_archive(tmp_path)
    install_root = tmp_path / "doctor_root"

    def _fake_urlopen(url: str):
        return archive.open("rb")

    monkeypatch.setattr("roi_stream.hamamatsu_sdk.urllib.request.urlopen", _fake_urlopen)
    sys.modules.pop("dcam", None)
    sys.modules.pop("dcamapi4", None)
    install_hamamatsu_sdk(url="https://example.test/sdk.zip", base_dir=install_root, accept_license=True)

    rc = cli.main(["doctor", "hamamatsu", "--install-dir", str(install_root)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SDK installed: yes" in out
    assert "Detected Hamamatsu cameras: 2" in out


class _FakeCam:
    def __init__(self):
        self.calls: list[tuple[int, object]] = []

    def prop_setvalue(self, prop_id, value):
        self.calls.append((int(prop_id), value))
        return True

    def lasterr(self):
        return "OK"


class _FakeDCAM:
    class DCAM_IDPROP:
        IMAGE_PIXELTYPE = 10
        READOUTSPEED = 20
        BINNING = 30
        SUBARRAYHPOS = 40
        SUBARRAYHSIZE = 41
        SUBARRAYVPOS = 42
        SUBARRAYVSIZE = 43
        SUBARRAYMODE = 44
        EXPOSURETIME = 50
        OUTPUTTRIGGER_SOURCE = 60
        OUTPUTTRIGGER_POLARITY = 61
        OUTPUTTRIGGER_ACTIVE = 62
        OUTPUTTRIGGER_KIND = 63
        _OUTPUTTRIGGER = 100

    class DCAM_PIXELTYPE:
        MONO16 = 16

    class DCAMPROP:
        class MODE:
            OFF = 0
            ON = 1

        class READOUTSPEED:
            FASTEST = 99

        class BINNING:
            _2 = 2

        class OUTPUTTRIGGER_SOURCE:
            EXPOSURE = 101

        class OUTPUTTRIGGER_POLARITY:
            POSITIVE = 202


def test_build_hamamatsu_settings_derives_exposure_from_frame_rate():
    settings = build_hamamatsu_settings({"frame_rate": 80.0})
    assert settings["pixel_type"] == "MONO16"
    assert settings["exposure_sec"] == 0.0125
    assert "binning" not in settings
    assert "roi" not in settings
    assert "output_triggers" not in settings


def test_apply_hamamatsu_settings_generic_defaults_are_minimal():
    cam = _FakeCam()
    resolved = apply_hamamatsu_settings(cam, _FakeDCAM, None, fps_hint=80.0)

    assert resolved["exposure_sec"] == 0.0125
    assert (10, 16) in cam.calls
    assert (50, 0.0125) in cam.calls
    assert len(cam.calls) == 2


def test_apply_hamamatsu_settings_honors_explicit_rig_config():
    cam = _FakeCam()
    resolved = apply_hamamatsu_settings(
        cam,
        _FakeDCAM,
        {
            "frame_rate": 80.0,
            "readout_speed": "FASTEST",
            "binning": 2,
            "roi": [0, 320, 1152, 476],
            "output_triggers": [
                {"line": 2, "source": "EXPOSURE", "polarity": "POSITIVE"},
                {"line": 3, "source": "EXPOSURE", "polarity": "POSITIVE"},
            ],
        },
        fps_hint=None,
    )

    assert resolved["exposure_sec"] == 0.0125
    assert (10, 16) in cam.calls
    assert (20, 99) in cam.calls
    assert (30, 2) in cam.calls
    assert (44, 0) in cam.calls
    assert (40, 0) in cam.calls
    assert (42, 320) in cam.calls
    assert (41, 1152) in cam.calls
    assert (43, 476) in cam.calls
    assert (44, 1) in cam.calls
    assert (50, 0.0125) in cam.calls
    assert (160, 101) in cam.calls
    assert (161, 202) in cam.calls
    assert (260, 101) in cam.calls
    assert (261, 202) in cam.calls
