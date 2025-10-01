from __future__ import annotations

from typing import List, Optional, Tuple
import time
import numpy as np


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

        # Preview components
        self.show_preview = True
        self._tex_registry_tag = "roi_tex_registry"
        self._tex_tag = None
        self._preview_child_tag = "roi_preview_child"
        self._drawlist_tag = "roi_preview_drawlist"
        self._last_tex_shape: Optional[Tuple[int, int]] = None  # (W, H)
        self._last_tex_update = 0.0
        self._preview_hz = 30.0  # limit texture updates
        self._raw_rgba: Optional[np.ndarray] = None  # float32 (H, W, 4)

    def build_ui(self):
        dpg = _lazy_import_dpg()
        with dpg.window(tag=self.window_tag, label=self.window_title, width=1000, height=650):
            with dpg.collapsing_header(label="Preview", default_open=True):
                dpg.add_checkbox(label="Show Preview", default_value=self.show_preview,
                                 callback=self._on_toggle_preview)
                # Child window provides a sized area; drawlist fills it
                with dpg.child_window(tag=self._preview_child_tag, width=-1, height=400, border=True):
                    dpg.add_drawlist(tag=self._drawlist_tag, width=-1, height=-1)

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

    def _on_toggle_preview(self, sender, app_data, *_):
        self.show_preview = bool(app_data)

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

    def _ensure_texture(self, W: int, H: int):
        dpg = _lazy_import_dpg()
        if self._tex_tag is not None and self._last_tex_shape == (W, H) and self._raw_rgba is not None:
            return
        # Create or recreate RAW texture backed by a persistent numpy array
        # Ensure registry exists
        if not dpg.does_item_exist(self._tex_registry_tag):
            with dpg.texture_registry(tag=self._tex_registry_tag):
                pass
        # Delete old texture if present
        if self._tex_tag is not None and dpg.does_item_exist(self._tex_tag):
            try:
                dpg.delete_item(self._tex_tag)
            except Exception:
                pass
        # Allocate persistent buffer (H, W, 4) float32 in [0,1]
        self._raw_rgba = np.zeros((H, W, 4), dtype=np.float32)
        self._raw_rgba[..., 3] = 1.0  # opaque
        # add_raw_texture expects format + width/height + buffer
        self._tex_tag = dpg.add_raw_texture(format=dpg.mvFormat_Float_rgba, width=W, height=H,
                                            default_value=self._raw_rgba, parent=self._tex_registry_tag)
        self._last_tex_shape = (W, H)

    def _update_preview(self):
        if not self.show_preview:
            return
        frame = self.shared.get_frame()
        circles = self.shared.circles
        if frame is None or circles is None:
            return
        H, W = frame.shape
        self._ensure_texture(W, H)

        now = time.time()
        if (now - self._last_tex_update) >= (1.0 / self._preview_hz):
            # Normalize uint16 to float32 in-place into the raw buffer
            # Maintain persistent memory; Dear PyGui reads from this buffer each frame
            if self._raw_rgba is not None:
                # Ensure shape matches
                if self._raw_rgba.shape[0] != H or self._raw_rgba.shape[1] != W:
                    # Recreate texture with new size
                    self._ensure_texture(W, H)
                # Write luminance to RGB channels
                self._raw_rgba[..., 0] = frame.astype(np.float32) / 65535.0
                self._raw_rgba[..., 1] = self._raw_rgba[..., 0]
                self._raw_rgba[..., 2] = self._raw_rgba[..., 0]
                # Alpha remains 1.0
            self._last_tex_update = now

        # Clear drawlist and draw image + circles
        dpg = _lazy_import_dpg()
        try:
            dpg.delete_item(self._drawlist_tag, children_only=True)
        except Exception:
            pass

        # Draw image to fit drawlist width while preserving aspect ratio
        # Compute display size
        # Use child window size as available drawing area
        dl_w, dl_h = dpg.get_item_rect_size(self._preview_child_tag)
        if dl_w <= 0:
            dl_w = W
        if dl_h <= 0:
            dl_h = H
        # Keep drawlist sized to the child content region
        try:
            dpg.configure_item(self._drawlist_tag, width=dl_w, height=dl_h)
        except Exception:
            pass
        sx = dl_w / float(W)
        sy = dl_h / float(H)
        s = min(sx, sy)
        disp_w = int(W * s)
        disp_h = int(H * s)
        offset_x = 0
        offset_y = 0
        # Center within drawlist if there is remaining space
        if dl_w > disp_w:
            offset_x = (dl_w - disp_w) // 2
        if dl_h > disp_h:
            offset_y = (dl_h - disp_h) // 2

        pmin = (offset_x, offset_y)
        pmax = (offset_x + disp_w, offset_y + disp_h)
        dpg.draw_image(self._tex_tag, pmin, pmax, parent=self._drawlist_tag)

        # Draw ROI circles (scaled)
        color = (255, 0, 0, 255)
        thickness = 2
        for row in circles:
            xc, yc, r = float(row[0]), float(row[1]), float(row[2])
            cx = offset_x + int(xc * s)
            cy = offset_y + int(yc * s)
            rr = max(1, int(r * s))
            dpg.draw_circle((cx, cy), rr, color=color, thickness=thickness, parent=self._drawlist_tag)

    def on_frame(self):
        dpg = _lazy_import_dpg()
        # Update preview image/overlays
        self._update_preview()

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
