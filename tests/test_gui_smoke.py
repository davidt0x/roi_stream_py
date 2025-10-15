import os
import sys
import threading
import time

import numpy as np
import pytest

try:
    from imgui_bundle import imgui  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("imgui_bundle not available", allow_module_level=True)

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("No DISPLAY available for imgui_bundle", allow_module_level=True)

from roi_stream.gui_app import ViewerApp
from roi_stream.shared import SharedState, TraceRing


def _make_shared_state(k: int = 3, n_samples: int = 200) -> SharedState:
    ring = TraceRing(k=k, maxlen=max(256, n_samples))
    for i in range(n_samples):
        t = float(i) * 0.05
        vals = np.linspace(0.0, 1.0, k) + 0.1 * np.sin(0.2 * i)
        ring.append(t, vals)
    shared = SharedState(traces=ring)
    w, h = 320, 240
    shared.circles = np.stack(
        [
            [w * 0.25, h * 0.3, min(w, h) * 0.08],
            [w * 0.50, h * 0.5, min(w, h) * 0.10],
            [w * 0.75, h * 0.7, min(w, h) * 0.06],
        ],
        axis=0,
    )
    shared.resolution = (w, h)
    frame = (np.linspace(0, 65535, w, dtype=np.uint16)[None, :]).repeat(h, axis=0)
    shared.update_frame(frame, (w, h))
    return shared


@pytest.mark.gui
def test_gui_smoke_state_snapshot():
    shared = _make_shared_state(k=3)
    stop_event = threading.Event()

    app = ViewerApp(shared, stop_event)
    app.plot_update_hz = 1000.0
    app._preview_hz = 1000.0

    app._update_preview_image(force=True)
    assert app._preview_image is not None

    app._last_plot_update = 0.0
    app.update_state()
    assert len(app._series_data) == shared.traces.k
    assert app._plot_x_limits is not None
    assert app._overlay_limits is not None
    assert len(app._colors_rgba) >= shared.traces.k

    app.lock_global_y = True
    app._locked_y_range = None
    app._last_plot_update = 0.0
    time.sleep(0.001)
    app.update_state()
    assert app._locked_y_range is not None
    assert app._overlay_limits == app._locked_y_range

    app.display_mode = "Stacked"
    assert len(app._stacked_limits) == shared.traces.k
