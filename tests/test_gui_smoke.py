import os
import sys
import threading
import numpy as np
import pytest


# Skip module if Dear PyGui isn't available or no display on Linux
try:
    import dearpygui.dearpygui as dpg  # type: ignore
except Exception:  # pragma: no cover
    pytest.skip("DearPyGui not available", allow_module_level=True)

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("No DISPLAY available for DearPyGui", allow_module_level=True)


from roi_stream.shared import SharedState, TraceRing


def _make_shared_state(k: int = 3, n_samples: int = 200) -> SharedState:
    ring = TraceRing(k=k, maxlen=max(256, n_samples))
    for i in range(n_samples):
        t = float(i) * 0.05
        vals = np.linspace(0.0, 1.0, k) + 0.1 * np.sin(0.2 * i)
        ring.append(t, vals)
    shared = SharedState(traces=ring)
    W, H = 320, 240
    shared.circles = np.stack([
        [W * 0.25, H * 0.3, min(W, H) * 0.08],
        [W * 0.50, H * 0.5, min(W, H) * 0.10],
        [W * 0.75, H * 0.7, min(W, H) * 0.06],
    ], axis=0)
    shared.resolution = (W, H)
    frame = (np.linspace(0, 65535, W, dtype=np.uint16)[None, :]).repeat(H, axis=0)
    shared.update_frame(frame, (W, H))
    return shared


@pytest.mark.gui
def test_gui_smoke_stacked_and_overlay(gui_test_context):
    shared = _make_shared_state(k=3)
    stop_event = threading.Event()

    with gui_test_context(shared, stop_event, create_viewport=False, show_viewport=False, disable_preview=True) as app:
        # After build_ui, stacked plots are not yet built; ensure them explicitly
        app._ensure_plots(shared.traces.k)

        assert dpg.does_item_exist("roi_view_window")
        assert dpg.does_item_exist("roi_preview_child")
        assert dpg.does_item_exist("roi_preview_drawlist")

        # Stacked mode series should exist
        assert dpg.does_item_exist("roi_series_stacked_0")

        # Switch to overlay and ensure plots
        app.display_mode = "Overlay"
        app._plots_dirty = True
        app._ensure_plots(shared.traces.k)

        assert dpg.does_item_exist("roi_plot")
        assert dpg.does_item_exist("roi_series_0")
