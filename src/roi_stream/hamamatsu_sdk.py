from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional
import importlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone


HAMAMATSU_SDK_URL = "https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/static/sys/dcam-sdk/zip/Hamamatsu_DCAMSDK4_v25056964.zip"
HAMAMATSU_SDK_VERSION = "v25056964"
HAMAMATSU_SDK_ARCHIVE_ROOT = "Hamamatsu_DCAMSDK4_v25056964"
ENV_SDK_PYTHON_DIR = "ROI_STREAM_HAMAMATSU_SDK_PYTHON_DIR"
ENV_SDK_DIR = "ROI_STREAM_HAMAMATSU_SDK_DIR"
ENV_LEGACY_SDK_DIR = "HAMAMATSU_DCAM_SDK_DIR"


@dataclass
class HamamatsuInstallMetadata:
    version: str
    url: str
    installed_at: str
    install_dir: str
    sdk_root: str
    python_dir: str


@dataclass
class HamamatsuStatus:
    install_root: Path
    installed: bool
    python_dir: Optional[Path]
    metadata: Optional[HamamatsuInstallMetadata]
    runtime_available: bool
    runtime_message: str
    camera_count: Optional[int]

DEFAULT_HAMAMATSU_SETTINGS: dict[str, Any] = {
    "pixel_type": "MONO16",
}


def get_sdk_install_root(base_dir: Optional[Path] = None) -> Path:
    if base_dir is not None:
        return Path(base_dir)

    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(root) / "roi_stream" / "hamamatsu_sdk"

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "roi_stream" / "hamamatsu_sdk"
    return Path.home() / ".local" / "share" / "roi_stream" / "hamamatsu_sdk"


def get_sdk_version_dir(base_dir: Optional[Path] = None) -> Path:
    return get_sdk_install_root(base_dir) / HAMAMATSU_SDK_VERSION


def get_sdk_metadata_path(base_dir: Optional[Path] = None) -> Path:
    return get_sdk_install_root(base_dir) / "install.json"


def _candidate_python_dirs(root: Path) -> list[Path]:
    return [
        root,
        root / "dcamsdk4" / "samples" / "python",
        root / HAMAMATSU_SDK_ARCHIVE_ROOT / "dcamsdk4" / "samples" / "python",
    ]


def is_valid_sdk_python_dir(path: Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "dcam.py").exists() and (p / "dcamapi4.py").exists()


def resolve_sdk_python_dir(candidate: Path) -> Optional[Path]:
    for path in _candidate_python_dirs(Path(candidate)):
        if is_valid_sdk_python_dir(path):
            return path.resolve()
    return None


def _load_metadata(base_dir: Optional[Path] = None) -> Optional[HamamatsuInstallMetadata]:
    path = get_sdk_metadata_path(base_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return HamamatsuInstallMetadata(**data)
    except TypeError:
        return None


def _write_metadata(metadata: HamamatsuInstallMetadata, base_dir: Optional[Path] = None) -> None:
    path = get_sdk_metadata_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def discover_sdk_python_dir(explicit: Optional[Path | str] = None, base_dir: Optional[Path] = None) -> Optional[Path]:
    if explicit:
        resolved = resolve_sdk_python_dir(Path(explicit))
        if resolved is not None:
            return resolved

    env_python_dir = os.environ.get(ENV_SDK_PYTHON_DIR)
    if env_python_dir:
        resolved = resolve_sdk_python_dir(Path(env_python_dir))
        if resolved is not None:
            return resolved

    for env_name in (ENV_SDK_DIR, ENV_LEGACY_SDK_DIR):
        env_root = os.environ.get(env_name)
        if env_root:
            resolved = resolve_sdk_python_dir(Path(env_root))
            if resolved is not None:
                return resolved

    metadata = _load_metadata(base_dir)
    if metadata is not None:
        resolved = resolve_sdk_python_dir(Path(metadata.python_dir))
        if resolved is not None:
            return resolved
        resolved = resolve_sdk_python_dir(Path(metadata.sdk_root))
        if resolved is not None:
            return resolved
        resolved = resolve_sdk_python_dir(Path(metadata.install_dir))
        if resolved is not None:
            return resolved

    return resolve_sdk_python_dir(get_sdk_version_dir(base_dir))


def import_hamamatsu_modules(python_dir: Path) -> tuple[Any, Any]:
    python_dir = Path(python_dir).resolve()
    if not is_valid_sdk_python_dir(python_dir):
        raise FileNotFoundError(f"Hamamatsu SDK Python directory is invalid: {python_dir}")

    path_str = str(python_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    importlib.invalidate_caches()

    dcamapi4 = importlib.import_module("dcamapi4")
    dcam = importlib.import_module("dcam")
    return dcamapi4, dcam


def get_missing_sdk_message() -> str:
    return (
        "Hamamatsu SDK support was requested, but the vendor Python SDK files were not found. "
        "Run 'roi_stream install-hamamatsu-sdk' to download the SDK to your user data directory, "
        f"or set {ENV_SDK_PYTHON_DIR} / {ENV_SDK_DIR} to an existing SDK install."
    )


def probe_hamamatsu_status(explicit: Optional[Path | str] = None, base_dir: Optional[Path] = None) -> HamamatsuStatus:
    install_root = get_sdk_install_root(base_dir)
    python_dir = discover_sdk_python_dir(explicit=explicit, base_dir=base_dir)
    metadata = _load_metadata(base_dir)
    if python_dir is None:
        return HamamatsuStatus(
            install_root=install_root,
            installed=False,
            python_dir=None,
            metadata=metadata,
            runtime_available=False,
            runtime_message=get_missing_sdk_message(),
            camera_count=None,
        )

    try:
        _, dcam = import_hamamatsu_modules(python_dir)
    except Exception as exc:
        return HamamatsuStatus(
            install_root=install_root,
            installed=True,
            python_dir=python_dir,
            metadata=metadata,
            runtime_available=False,
            runtime_message=f"SDK Python files found, but runtime import failed: {exc}",
            camera_count=None,
        )

    camera_count: Optional[int] = None
    runtime_message = "Runtime import succeeded."
    runtime_available = True
    try:
        if dcam.Dcamapi.init():
            count = dcam.Dcamapi.get_devicecount()
            camera_count = int(count) if count is not False else None
        else:
            runtime_available = False
            runtime_message = f"Runtime import succeeded, but DCAM initialization failed: {dcam.Dcamapi.lasterr()}"
    except Exception as exc:
        runtime_available = False
        runtime_message = f"Runtime import succeeded, but camera probe failed: {exc}"
    finally:
        try:
            dcam.Dcamapi.uninit()
        except Exception:
            pass

    return HamamatsuStatus(
        install_root=install_root,
        installed=True,
        python_dir=python_dir,
        metadata=metadata,
        runtime_available=runtime_available,
        runtime_message=runtime_message,
        camera_count=camera_count,
    )


def install_hamamatsu_sdk(
    *,
    url: str = HAMAMATSU_SDK_URL,
    base_dir: Optional[Path] = None,
    accept_license: bool = False,
    input_fn: Callable[[str], str] = input,
) -> HamamatsuStatus:
    install_root = get_sdk_install_root(base_dir)
    version_dir = get_sdk_version_dir(base_dir)

    if not accept_license:
        prompt = (
            "This command downloads third-party software from Hamamatsu.\n"
            f"URL: {url}\n"
            "You are responsible for reviewing and accepting Hamamatsu's license terms.\n"
            "Type 'yes' to continue: "
        )
        answer = input_fn(prompt).strip().lower()
        if answer != "yes":
            raise RuntimeError("Hamamatsu SDK installation cancelled by user.")

    install_root.mkdir(parents=True, exist_ok=True)
    if version_dir.exists():
        shutil.rmtree(version_dir)

    with tempfile.TemporaryDirectory(prefix="roi_stream_hamamatsu_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        archive_path = tmpdir / "hamamatsu_sdk.zip"

        with urllib.request.urlopen(url) as response, archive_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)

        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(version_dir)

    python_dir = discover_sdk_python_dir(explicit=version_dir, base_dir=base_dir)
    if python_dir is None:
        raise RuntimeError("Downloaded Hamamatsu SDK archive did not contain the expected Python sample files.")

    metadata = HamamatsuInstallMetadata(
        version=HAMAMATSU_SDK_VERSION,
        url=url,
        installed_at=datetime.now(timezone.utc).isoformat(),
        install_dir=str(version_dir.resolve()),
        sdk_root=str((version_dir / HAMAMATSU_SDK_ARCHIVE_ROOT).resolve()),
        python_dir=str(python_dir.resolve()),
    )
    _write_metadata(metadata, base_dir)
    return probe_hamamatsu_status(explicit=python_dir, base_dir=base_dir)


def build_hamamatsu_settings(user_settings: Optional[dict[str, Any]], fps_hint: Optional[float] = None) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_HAMAMATSU_SETTINGS))
    if user_settings:
        for key, value in user_settings.items():
            merged[key] = value

    if merged.get("frame_rate") is None and fps_hint is not None:
        merged["frame_rate"] = float(fps_hint)
    if merged.get("exposure_sec") is None and merged.get("frame_rate"):
        merged["exposure_sec"] = round(1.0 / float(merged["frame_rate"]), 4)
    return merged


def _enum_value(namespace: Any, enum_name: str, value_name: str) -> Any:
    enum_cls = getattr(namespace.DCAMPROP, enum_name, None)
    if enum_cls is None:
        raise AttributeError(f"DCAM enum {enum_name} is unavailable")
    if not hasattr(enum_cls, value_name):
        raise AttributeError(f"DCAM enum {enum_name} has no value {value_name}")
    return getattr(enum_cls, value_name)


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def _set_prop(cam: Any, prop_id: Any, value: Any) -> None:
    actual = _coerce_scalar(value)
    ok = cam.prop_setvalue(prop_id, actual)
    if ok is False:
        raise RuntimeError(f"Failed to set DCAM property {prop_id}: {cam.lasterr()}")


def apply_hamamatsu_settings(cam: Any, dcamapi4: Any, settings: Optional[dict[str, Any]], fps_hint: Optional[float] = None) -> dict[str, Any]:
    resolved = build_hamamatsu_settings(settings, fps_hint=fps_hint)
    prop_ids = dcamapi4.DCAM_IDPROP

    pixel_type = resolved.get("pixel_type")
    if pixel_type:
        _set_prop(cam, prop_ids.IMAGE_PIXELTYPE, getattr(dcamapi4.DCAM_PIXELTYPE, str(pixel_type).upper()))

    readout_speed = resolved.get("readout_speed")
    if readout_speed:
        _set_prop(cam, prop_ids.READOUTSPEED, _enum_value(dcamapi4, "READOUTSPEED", str(readout_speed).upper()))

    binning = resolved.get("binning")
    if binning is not None:
        _set_prop(cam, prop_ids.BINNING, getattr(dcamapi4.DCAMPROP.BINNING, f"_{int(binning)}"))

    roi = resolved.get("roi")
    if roi is not None:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("Hamamatsu ROI must be [x, y, width, height]")
        x, y, w, h = [int(v) for v in roi]
        _set_prop(cam, prop_ids.SUBARRAYMODE, getattr(dcamapi4.DCAMPROP.MODE, "OFF", 1))
        _set_prop(cam, prop_ids.SUBARRAYHPOS, x)
        _set_prop(cam, prop_ids.SUBARRAYVPOS, y)
        _set_prop(cam, prop_ids.SUBARRAYHSIZE, w)
        _set_prop(cam, prop_ids.SUBARRAYVSIZE, h)
        _set_prop(cam, prop_ids.SUBARRAYMODE, getattr(dcamapi4.DCAMPROP.MODE, "ON", 2))

    exposure_sec = resolved.get("exposure_sec")
    if exposure_sec is not None:
        _set_prop(cam, prop_ids.EXPOSURETIME, float(exposure_sec))

    output_triggers = resolved.get("output_triggers") or []
    trigger_offset = int(getattr(prop_ids, "_OUTPUTTRIGGER", 0))
    for trig in output_triggers:
        if not isinstance(trig, dict):
            continue
        line = int(trig.get("line", 1))
        if line < 1:
            raise ValueError("Hamamatsu output trigger line numbers are 1-based")
        base = trigger_offset * (line - 1)
        source = trig.get("source")
        polarity = trig.get("polarity")
        active = trig.get("active")
        kind = trig.get("kind")
        if source:
            _set_prop(
                cam,
                prop_ids.OUTPUTTRIGGER_SOURCE + base,
                _enum_value(dcamapi4, "OUTPUTTRIGGER_SOURCE", str(source).upper()),
            )
        if polarity:
            _set_prop(
                cam,
                prop_ids.OUTPUTTRIGGER_POLARITY + base,
                _enum_value(dcamapi4, "OUTPUTTRIGGER_POLARITY", str(polarity).upper()),
            )
        if active:
            _set_prop(
                cam,
                prop_ids.OUTPUTTRIGGER_ACTIVE + base,
                _enum_value(dcamapi4, "OUTPUTTRIGGER_ACTIVE", str(active).upper()),
            )
        if kind:
            _set_prop(
                cam,
                prop_ids.OUTPUTTRIGGER_KIND + base,
                _enum_value(dcamapi4, "OUTPUTTRIGGER_KIND", str(kind).upper()),
            )

    properties = resolved.get("properties") or {}
    if isinstance(properties, dict):
        for prop_name, prop_value in properties.items():
            prop_id = getattr(prop_ids, str(prop_name).upper(), None)
            if prop_id is None:
                raise AttributeError(f"Unknown DCAM property id: {prop_name}")
            if isinstance(prop_value, str):
                enum_candidates = [
                    "MODE",
                    "TRIGGERSOURCE",
                    "TRIGGERACTIVE",
                    "TRIGGER_MODE",
                    "TRIGGERPOLARITY",
                    "READOUTSPEED",
                    "BINNING",
                    "OUTPUTTRIGGER_SOURCE",
                    "OUTPUTTRIGGER_POLARITY",
                    "OUTPUTTRIGGER_ACTIVE",
                    "OUTPUTTRIGGER_KIND",
                ]
                mapped = None
                for enum_name in enum_candidates:
                    enum_cls = getattr(dcamapi4.DCAMPROP, enum_name, None)
                    if enum_cls is not None and hasattr(enum_cls, prop_value.upper()):
                        mapped = getattr(enum_cls, prop_value.upper())
                        break
                prop_value = mapped if mapped is not None else prop_value
            _set_prop(cam, prop_id, prop_value)

    return resolved
