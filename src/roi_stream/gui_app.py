from __future__ import annotations

from typing import List, Optional, Tuple
import time


def _lazy_import_dpg():
    import dearpygui.dearpygui as dpg  # type: ignore
    return dpg


class ViewerApp:
    """Dear PyGui viewer for ROI means traces.

    Displays a scrolling plot of K ROI series using shared ring buffers.
    """

    def __init__(self, shared_state, stop_event, window_title: str = "ROI Stream Viewer") -> None:
        self.shared = shared_state
        self.stop_event = stop_event
        self.window_title = window_title

        self.window_tag = "roi_view_window"
        self.plot_tag = "roi_plot"
        self.x_axis = None
        self.y_axis = None
        self.series_tags: List[str] = []
        self.stats_tag = None

        # X-axis window management (seconds)
        self.window_sec = 60.0
        self._x_limits_set = False

        # cache to avoid recreating series unnecessarily
        self._last_k = None

    def build_ui(self):
        dpg = _lazy_import_dpg()
        with dpg.window(tag=self.window_tag, label=self.window_title, width=1000, height=650):
            with dpg.group(horizontal=True):
                dpg.add_input_float(label="X window (s)", default_value=self.window_sec, min_value=5.0,
                                    min_clamped=True, step=5.0, width=140, callback=self._on_window_change)
                dpg.add_button(label="Quit", callback=lambda *_: dpg.stop_dearpygui())

            with dpg.plot(label="ROI Means", width=-1, height=-1, tag=self.plot_tag):
                self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
                self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Mean")
                # Create placeholder series; real ones added in refresh
            self.stats_tag = dpg.add_text("K=0  points=0")

    def _on_window_change(self, sender, app_data, *_):
        try:
            val = float(app_data)
        except Exception:
            return
        self.window_sec = max(5.0, val)
        self._x_limits_set = False

    def _ensure_series(self, k: int):
        dpg = _lazy_import_dpg()
        if self._last_k == k and len(self.series_tags) == k:
            return
        # Recreate series
        if self.series_tags:
            for tag in self.series_tags:
                try:
                    dpg.delete_item(tag)
                except Exception:
                    pass
        self.series_tags = []
        for i in range(k):
            tag = f"roi_series_{i}"
            self.series_tags.append(tag)
            dpg.add_line_series([], [], tag=tag, parent=self.y_axis)
        self._last_k = k
        self._x_limits_set = False

    def on_frame(self):
        dpg = _lazy_import_dpg()
        t, ys = self.shared.traces.snapshot()
        k = len(ys)
        if k == 0:
            return
        self._ensure_series(k)

        # Determine visible window in X
        if not t:
            return
        tmax = float(t[-1])
        x0 = max(0.0, tmax - self.window_sec)
        x1 = max(tmax, x0 + 1e-3)

        # Update series
        # Each series uses the same x vector (t)
        # To reduce copies, reuse t list; dpg takes Python lists
        y_min = None
        y_max = None
        for i, tag in enumerate(self.series_tags):
            dpg.set_value(tag, [t, ys[i]])
            if ys[i]:
                ymin = min(ys[i][-min(len(ys[i]), len(t)):])
                ymax = max(ys[i][-min(len(ys[i]), len(t)):])
                y_min = ymin if y_min is None else min(y_min, ymin)
                y_max = ymax if y_max is None else max(y_max, ymax)

        if not self._x_limits_set:
            dpg.set_axis_limits(self.x_axis, x0, x1)
            self._x_limits_set = True
        else:
            dpg.set_axis_limits(self.x_axis, x0, x1)

        if y_min is not None and y_max is not None:
            if y_min == y_max:
                pad = 1.0 if y_min == 0 else abs(y_min) * 0.1 + 1.0
                y0 = y_min - pad
                y1 = y_max + pad
            else:
                pad = (y_max - y_min) * 0.10
                y0 = y_min - pad
                y1 = y_max + pad
            dpg.set_axis_limits(self.y_axis, float(y0), float(y1))

        dpg.set_value(self.stats_tag, f"K={k}  points={len(t)}  t={tmax:0.2f}s")


def run_gui(shared_state, stop_event) -> None:
    """Run the Dear PyGui viewer event loop until closed.

    This function blocks until the user closes the window.
    """
    dpg = _lazy_import_dpg()
    dpg.create_context()
    dpg.create_viewport(title="ROI Stream Viewer", width=1000, height=650)

    app = ViewerApp(shared_state, stop_event)
    app.build_ui()

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if stop_event.is_set():
            # Worker requested stop; close GUI too
            dpg.stop_dearpygui()
            break
        app.on_frame()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

