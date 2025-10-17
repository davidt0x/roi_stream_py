from __future__ import annotations

import colorsys
import math
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from importlib import import_module
from imgui_bundle import ImVec2, ImVec4, hello_imgui, immvision
from imgui_bundle import imgui as _imgui_ns, implot as _implot_ns

imgui = _imgui_ns
if not hasattr(imgui, "WindowFlags_"):
    imgui = import_module("imgui_bundle._imgui_bundle.imgui")

implot = _implot_ns
if not hasattr(implot, "begin_plot"):
    implot = import_module("imgui_bundle._imgui_bundle.implot")

from .perf import PerfTracker


class ViewerApp:
    """imgui-bundle viewer for ROI mean traces and preview frames."""

    def __init__(self, shared_state, stop_event, window_title: str = "ROI Stream Viewer") -> None:
        self.shared = shared_state
        self.stop_event = stop_event
        self.window_title = window_title

        self.window_sec = 30.0
        self.preview_frac = 0.33
        self.display_mode = "Stacked"
        self.lock_global_y = False
        self.plot_update_hz = 60.0
        self._preview_hz = 20.0
        self._stacked_plot_height = 140

        self._stats_text = "ROIs=0  samples=0"
        self._vp_w: Optional[float] = None
        self._vp_h: Optional[float] = None

        self._series_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self._plot_x_limits: Optional[Tuple[float, float]] = None
        self._overlay_limits: Optional[Tuple[float, float]] = None
        self._stacked_limits: List[Optional[Tuple[float, float]]] = []
        self._locked_y_range: Optional[Tuple[float, float]] = None

        self._last_plot_update = 0.0
        self._last_preview_update = 0.0
        self._preview_image: Optional[np.ndarray] = None
        self._preview_dirty = False

        self._colors_rgba: List[Tuple[int, int, int, int]] = []
        self._colors_bgr: List[Tuple[int, int, int]] = []
        self._colors_implot: List[ImVec4] = []
        self._palette_size = 0

        self._runner_params: Optional[hello_imgui.RunnerParams] = None
        self._owns_implot_context = False
        self._setup_done = False

        self._perf = PerfTracker("ROI_STREAM_GUI_PROFILE")

        target_fps_env = os.environ.get("ROI_STREAM_GUI_TARGET_FPS")
        default_target_fps = 60.0
        self._frame_target_fps = default_target_fps
        if target_fps_env:
            try:
                parsed = float(target_fps_env)
                if parsed > 0.0:
                    self._frame_target_fps = parsed
            except ValueError:
                pass
        self._frame_target_interval = 1.0 / self._frame_target_fps

        idle_env = os.environ.get("ROI_STREAM_GUI_IDLE", "")
        if idle_env == "":
            self._idle_enabled = False
        else:
            self._idle_enabled = idle_env.lower() not in ("0", "false", "no")
        idle_fps_env = os.environ.get("ROI_STREAM_GUI_IDLE_FPS")
        self._idle_fps: Optional[float] = None
        if idle_fps_env:
            try:
                parsed_idle = float(idle_fps_env)
                if parsed_idle > 0.0:
                    self._idle_fps = parsed_idle
            except ValueError:
                pass
        self._last_frame_present = 0.0
        self.fullscreen_window = True

    def _estimate_window_point_cap(self, total_samples: int, tlast: float) -> int:
        """Estimate how many samples to request per window to bound copy cost."""
        traces = self.shared.traces
        maxlen = getattr(traces, "maxlen", 0)
        maxlen = int(maxlen) if maxlen else 0
        base_rate = self._frame_target_fps if self._frame_target_fps > 0 else 60.0
        if tlast <= 0.0 or total_samples <= 0:
            est_rate = base_rate
        else:
            est_rate = float(total_samples) / max(tlast, 1e-6)
            if not math.isfinite(est_rate) or est_rate <= 0.0:
                est_rate = base_rate
        # Provide headroom (1.5x) but clamp to 600 FPS equivalent
        est_rate = min(max(est_rate * 1.5, 30.0), 600.0)
        cap = int(max(120, round(self.window_sec * est_rate)))
        if maxlen > 0:
            cap = min(cap, maxlen)
        return cap

    def _build_palette(self, count: int) -> List[Tuple[int, int, int, int]]:
        if count <= 0:
            return []
        colors: List[Tuple[int, int, int, int]] = []
        for i in range(count):
            h = (i / max(1, count)) % 1.0
            s = 0.85
            v = 0.95
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append((int(r * 255), int(g * 255), int(b * 255), 255))
        return colors

    def _ensure_palette(self, count: int) -> None:
        if count <= 0:
            self._colors_rgba = []
            self._colors_bgr = []
            self._colors_implot = []
            self._palette_size = 0
            return
        if count <= self._palette_size:
            return
        colors = self._build_palette(count)
        self._colors_rgba = colors
        self._colors_bgr = [(c[2], c[1], c[0]) for c in colors]
        self._colors_implot = [ImVec4(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, c[3] / 255.0) for c in colors]
        self._palette_size = count

    @staticmethod
    def _pad_range(ymin: float, ymax: float) -> Tuple[float, float]:
        if ymin == ymax:
            pad = 1.0 if ymin == 0 else abs(ymin) * 0.1 + 1.0
            return float(ymin - pad), float(ymax + pad)
        span = ymax - ymin
        pad = span * 0.10
        return float(ymin - pad), float(ymax + pad)

    def _update_preview_image(self, *, force: bool = False) -> None:
        interval = 1.0 / max(1e-6, self._preview_hz)
        now_wall = time.time()
        if not force and (now_wall - self._last_preview_update) < interval:
            return
        with self._perf.measure("preview_total"):
            self._perf.start("preview_fetch")
            frame_preview = self.shared.copy_preview_frame(self._preview_image)
            frame16 = None
            if frame_preview is None:
                frame16 = self.shared.get_frame()
            self._perf.stop("preview_fetch")

            image = frame_preview
            if image is None:
                if frame16 is None or frame16.ndim != 2:
                    return
                with self._perf.measure("preview_convert"):
                    gray16 = np.asarray(frame16, dtype=np.uint16)
                    image = np.right_shift(gray16, 8).astype(np.uint8)

            circles = self.shared.circles
            if image is not None and circles is not None and len(circles) > 0:
                with self._perf.measure("preview_overlay"):
                    if image.ndim == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    palette = self._colors_bgr or [(0, 0, 255)]
                    for idx, circle in enumerate(circles):
                        cx = int(round(float(circle[0])))
                        cy = int(round(float(circle[1])))
                        radius = int(max(1.0, float(circle[2])))
                        color = palette[idx % len(palette)]
                        cv2.circle(image, (cx, cy), radius, color, thickness=1)

            if self._preview_image is not image:
                self._preview_image = image
            self._last_preview_update = now_wall
            self._preview_dirty = True

    def update_state(self) -> None:
        traces = self.shared.traces
        k = int(traces.k)
        total = 0
        try:
            total = int(traces.total_count())
        except Exception:
            total = 0
        tlast = traces.last_time()
        self._stats_text = f"ROIs={k}  samples={total}  t={tlast:0.2f}s"

        circle_count = int(len(self.shared.circles)) if self.shared.circles is not None else 0
        self._ensure_palette(max(k, circle_count))
        self._update_preview_image()

        with self._perf.measure("update_state_total"):
            interval = 1.0 / max(1e-6, self.plot_update_hz)
            now = time.time()
            if (now - self._last_plot_update) < interval:
                return

            x0 = max(0.0, tlast - self.window_sec)
            x1 = x0 + max(self.window_sec, 1e-3)
            with self._perf.measure("update_snapshot_window"):
                point_cap = self._estimate_window_point_cap(total, tlast)
                t_values, series_values = traces.snapshot_window(start_time=x0, max_points=point_cap)

            if t_values.size == 0 or series_values.shape[1] == 0:
                self._series_data = []
                self._plot_x_limits = None
                self._overlay_limits = None
                self._stacked_limits = []
                self._last_plot_update = now
                return

            with self._perf.measure("update_series_processing"):
                t_array = t_values
                y_matrix = series_values
                new_series: List[Tuple[np.ndarray, np.ndarray]] = []
                stacked_ranges: List[Optional[Tuple[float, float]]] = []
                global_min: Optional[float] = None
                global_max: Optional[float] = None
                sample_count = int(t_array.size)
                for idx in range(y_matrix.shape[0]):
                    ys = y_matrix[idx]
                    if ys.size != sample_count:
                        if sample_count == 0:
                            xs = np.empty(0, dtype=np.float64)
                            ys = np.empty(0, dtype=np.float64)
                        else:
                            xs = t_array
                            ys = ys[-sample_count:]
                    else:
                        xs = t_array
                    new_series.append((xs, ys))
                    if ys.size:
                        ymin = float(np.min(ys))
                        ymax = float(np.max(ys))
                        y0, y1 = self._pad_range(ymin, ymax)
                        stacked_ranges.append((y0, y1))
                        global_min = ymin if global_min is None else min(global_min, ymin)
                        global_max = ymax if global_max is None else max(global_max, ymax)
                    else:
                        stacked_ranges.append(None)

            self._series_data = new_series
            self._plot_x_limits = (float(x0), float(x1))
            if not self.lock_global_y:
                self._locked_y_range = None
            if global_min is not None and global_max is not None:
                desired = self._pad_range(global_min, global_max)
                if self.lock_global_y:
                    if self._locked_y_range is None:
                        self._locked_y_range = desired
                    self._overlay_limits = self._locked_y_range
                else:
                    self._overlay_limits = desired
            else:
                self._overlay_limits = None

            if self.lock_global_y and self._locked_y_range is not None:
                lock_range = self._locked_y_range
                self._stacked_limits = [lock_range if ys.size else None for _, ys in new_series]
            else:
                self._stacked_limits = stacked_ranges

            self._last_plot_update = now

    def _compute_preview_display_size(self, avail_w: float, avail_h: float) -> Tuple[float, float]:
        if self._preview_image is None:
            return avail_w, avail_h
        h, w = self._preview_image.shape[:2]
        if w <= 0 or h <= 0:
            return avail_w, avail_h
        if avail_w <= 0.0 or avail_h <= 0.0:
            return float(w), float(h)
        scale = min(avail_w / float(w), avail_h / float(h))
        return float(w) * scale, float(h) * scale

    def _render_preview(self, preview_height: float) -> None:
        with self._perf.measure("render_preview"):
            imgui.begin_child(
                "roi_preview_child",
                size=ImVec2(0.0, preview_height),
                child_flags=imgui.ChildFlags_.borders,
            )
            if self._preview_image is None:
                imgui.text_disabled("Preview unavailable")
            else:
                avail = imgui.get_content_region_avail()
                disp_w, disp_h = self._compute_preview_display_size(avail.x, avail.y)
                cursor = imgui.get_cursor_pos()
                offset_x = max(0.0, (avail.x - disp_w) * 0.5)
                offset_y = max(0.0, (avail.y - disp_h) * 0.5)
                imgui.set_cursor_pos(ImVec2(cursor.x + offset_x, cursor.y + offset_y))
                immvision.push_color_order_bgr()
                with self._perf.measure("render_preview_image"):
                    immvision.image_display_resizable(
                        "roi_preview_image",
                        self._preview_image,
                        size=ImVec2(disp_w, disp_h),
                        refresh_image=self._preview_dirty,
                        resizable=False,
                        show_options_button=False,
                    )
                immvision.pop_color_order()
                imgui.set_cursor_pos(cursor)
            imgui.end_child()
            self._preview_dirty = False

    def _render_control_row(self) -> None:
        imgui.push_item_width(160.0)
        changed, value = imgui.input_float("X window (s)", self.window_sec, step=5.0, format="%.1f")
        if changed:
            self.window_sec = max(5.0, float(value))
        imgui.pop_item_width()
        imgui.same_line()
        single_plot = self.display_mode == "Overlay"
        changed, single_plot = imgui.checkbox("Single Plot", single_plot)
        if changed:
            self.display_mode = "Overlay" if single_plot else "Stacked"
        imgui.same_line()
        changed, lock = imgui.checkbox("Lock global Y", self.lock_global_y)
        if changed:
            self.lock_global_y = bool(lock)
            self._locked_y_range = None

        imgui.new_line()

    def _render_overlay_plot(self) -> None:
        with self._perf.measure("render_overlay_plot"):
            if not self._series_data or self._plot_x_limits is None:
                imgui.text_disabled("No traces available")
                return
            x0, x1 = self._plot_x_limits
            implot.set_next_axis_limits(implot.ImAxis_.x1, x0, x1, cond=implot.Cond_.always)
            if self._overlay_limits is not None:
                y0, y1 = self._overlay_limits
                implot.set_next_axis_limits(implot.ImAxis_.y1, y0, y1, cond=implot.Cond_.always)
            if implot.begin_plot("ROI Means", size=ImVec2(-1.0, -1.0)):
                implot.setup_axes("Time (s)", "Mean")
                for idx, (xs, ys) in enumerate(self._series_data):
                    if xs.size == 0 or ys.size == 0:
                        continue
                    if idx < len(self._colors_implot):
                        implot.set_next_line_style(self._colors_implot[idx])
                    implot.plot_line(f"ROI {idx}", xs, ys)
                implot.end_plot()

    def _render_stacked_plots(self) -> None:
        with self._perf.measure("render_stacked_plots"):
            if not self._series_data or self._plot_x_limits is None:
                imgui.text_disabled("No traces available")
                return
            x0, x1 = self._plot_x_limits
            for idx, (xs, ys) in enumerate(self._series_data):
                imgui.push_id(idx)
                color = self._colors_implot[idx % len(self._colors_implot)] if self._colors_implot else ImVec4(1.0, 0.0, 0.0, 1.0)
                imgui.align_text_to_frame_padding()
                imgui.text_colored(color, f"ROI {idx}")
                imgui.same_line()
                implot.set_next_axis_limits(implot.ImAxis_.x1, x0, x1, cond=implot.Cond_.always)
                if self.lock_global_y and self._locked_y_range is not None:
                    y_limits = self._locked_y_range
                else:
                    y_limits = self._stacked_limits[idx] if idx < len(self._stacked_limits) else None
                if y_limits is not None:
                    implot.set_next_axis_limits(implot.ImAxis_.y1, y_limits[0], y_limits[1], cond=implot.Cond_.always)
                flags = implot.Flags_.no_menus | implot.Flags_.no_box_select
                if implot.begin_plot("##roi_plot", size=ImVec2(-1.0, float(self._stacked_plot_height)), flags=flags):
                    implot.setup_axes("Time (s)", "Mean")
                    if xs.size and ys.size:
                        if idx < len(self._colors_implot):
                            implot.set_next_line_style(self._colors_implot[idx])
                        implot.plot_line("##series", xs, ys)
                    implot.end_plot()
                imgui.pop_id()
                if idx < len(self._series_data) - 1:
                    imgui.spacing()

    def render_gui(self) -> None:
        with self._perf.measure("render_gui"):
            viewport = None
            get_main_viewport = getattr(imgui, "get_main_viewport", None)
            if callable(get_main_viewport):
                try:
                    viewport = get_main_viewport()
                except Exception:
                    viewport = None
            if viewport is not None:
                pos = viewport.pos
                size = viewport.size
                self._vp_w = size.x
                self._vp_h = size.y
                if self.fullscreen_window:
                    imgui.set_next_window_pos(pos, cond=imgui.Cond_.always)
                    imgui.set_next_window_size(size, cond=imgui.Cond_.always)

            flags = imgui.WindowFlags_.no_collapse
            if self.fullscreen_window:
                flags |= imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
            if not imgui.begin(self.window_title, flags=flags):
                imgui.end()
                return

            preview_open = imgui.collapsing_header("Preview", flags=imgui.TreeNodeFlags_.default_open)
            if preview_open:
                changed, value = imgui.slider_float("Preview height (fraction)", self.preview_frac, 0.10, 0.60, "%.2f")
                if changed:
                    self.preview_frac = float(min(0.9, max(0.05, value)))
                vp_h = self._vp_h if self._vp_h is not None else 600.0
                preview_height = max(120.0, vp_h * self.preview_frac)
                self._render_preview(preview_height)
            else:
                preview_height = 0.0

            if preview_open and preview_height > 0:
                imgui.spacing()

            self._render_control_row()
            imgui.text_unformatted(self._stats_text)
            imgui.spacing()

            avail = imgui.get_content_region_avail()
            imgui.begin_child("roi_plots", size=ImVec2(0.0, avail.y))
            if self.display_mode == "Overlay":
                self._render_overlay_plot()
            else:
                self._render_stacked_plots()
            imgui.end_child()
            imgui.end()

    def _on_pre_frame(self) -> None:
        with self._perf.measure("on_pre_frame"):
            if self.stop_event.is_set() and self._runner_params is not None:
                self._runner_params.app_shall_exit = True
            self.update_state()
        self._perf.maybe_report()

    def _on_after_swap(self) -> None:
        if not self._perf.enabled:
            return
        now = time.perf_counter()
        if self._last_frame_present != 0.0:
            interval = now - self._last_frame_present
            self._perf.record_seconds("frame_present_interval", interval)
            slip = max(0.0, interval - self._frame_target_interval)
            if slip > 0.0:
                self._perf.record_seconds("frame_present_slip", slip)
        self._last_frame_present = now

    def _on_post_init(self) -> None:
        if not self._owns_implot_context:
            implot.create_context()
            try:
                implot.set_imgui_context(imgui.get_current_context())
            except Exception:
                pass
            self._owns_implot_context = True

    def _on_before_exit(self) -> None:
        try:
            self.stop_event.set()
        except Exception:
            pass

    def setup(self, *, width: int = 1000, height: int = 650) -> None:
        if self._setup_done:
            return
        callbacks = hello_imgui.RunnerCallbacks(
            show_gui=self.render_gui,
            pre_new_frame=self._on_pre_frame,
            after_swap=self._on_after_swap,
            post_init=self._on_post_init,
            before_exit=self._on_before_exit,
        )
        geometry = hello_imgui.WindowGeometry(size=(int(width), int(height)))
        app_params = hello_imgui.AppWindowParams(window_title=self.window_title, window_geometry=geometry)
        imgui_params = hello_imgui.ImGuiWindowParams(
            default_imgui_window_type=hello_imgui.DefaultImGuiWindowType.no_default_window,
            show_menu_bar=False,
        )
        self._runner_params = hello_imgui.RunnerParams(
            callbacks=callbacks,
            app_window_params=app_params,
            imgui_window_params=imgui_params,
        )
        try:
            fps_idling = self._runner_params.fps_idling
            if fps_idling is not None:
                fps_idling.enable_idling = bool(self._idle_enabled)
                if self._idle_enabled and self._idle_fps is not None:
                    fps_idling.fps_idle = float(self._idle_fps)
        except Exception:
            pass
        self._setup_done = True

    def start(self) -> None:
        if not self._setup_done or self._runner_params is None:
            raise RuntimeError("ViewerApp.setup() must be called before start().")
        hello_imgui.run(self._runner_params)

    def teardown(self) -> None:
        if self._owns_implot_context:
            try:
                implot.destroy_context()
            except Exception:
                pass
            self._owns_implot_context = False
        self._runner_params = None
        self._setup_done = False


def run_gui(shared_state, stop_event) -> None:
    """Convenience wrapper that sets up, starts, and tears down the GUI."""

    # import cProfile
    # import pstats
    # import io

    # profiler = cProfile.Profile()
    # profiler.enable()

    app = ViewerApp(shared_state, stop_event)
    app.setup()
    try:
        app.start()
    finally:
        app.teardown()

    # profiler.disable()

    # s = io.StringIO()
    # sortby = pstats.SortKey.CUMULATIVE
    # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
