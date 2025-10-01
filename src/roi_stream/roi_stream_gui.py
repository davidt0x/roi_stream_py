"""
Dear PyGui Random Walk — viewport-filling plot with scrolling X window

- Plot fills the viewport (window size tracks viewport).
- One new random-walk point per rendered frame.
- X axis shows a fixed number of points (window_size, default 500) and scrolls.
- Y axis auto-fits to the visible window (toggleable).
"""

import random
import dearpygui.dearpygui as dpg


class RandomWalkApp:
    def __init__(self):
        # Simulation state
        self.running = False
        self.step_sigma = 1.0

        # Data buffers
        self.x = [0.0]
        self.y = [0.0]
        self.max_points = 20_000  # overall cap for responsiveness

        # View/window behavior
        self.window_tag = "primary_window"
        self.series_tag = None
        self.x_axis = None
        self.y_axis = None
        self.stats_points = None

        # Viewport sync
        self._vp_w = None
        self._vp_h = None

        # Scrolling X window settings
        self.window_size = 500  # number of points visible on X
        self.autoscale_y = True
        self._init_x_limits_set = False

    def build_ui(self):
        with dpg.window(
            tag=self.window_tag, pos=(0, 0),
            no_title_bar=True, no_resize=True, no_move=True
        ):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", callback=self._on_start)
                dpg.add_button(label="Stop", callback=self._on_stop)
                dpg.add_button(label="Reset", callback=self._on_reset)
                dpg.add_input_float(
                    label="Step σ", default_value=self.step_sigma,
                    min_value=0.0, min_clamped=True, step=0.1, width=110,
                    callback=self._on_step_change
                )
                dpg.add_input_int(
                    label="X window", default_value=self.window_size, min_value=50,
                    min_clamped=True, step=50, width=110, callback=self._on_window_change
                )
                dpg.add_checkbox(
                    label="Auto-fit Y", default_value=self.autoscale_y,
                    callback=self._on_autoscale_y_toggle
                )

            # Plot fills the window
            with dpg.plot(label="Random Walk", width=-1, height=-1):
                self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Value")
                self.series_tag = dpg.add_line_series(self.x, self.y, parent=self.y_axis)

            self.stats_points = dpg.add_text("points: 1")

        dpg.set_primary_window(self.window_tag, True)

    # ----- Controls -----
    def _on_start(self, *_):
        self.running = True

    def _on_stop(self, *_):
        self.running = False

    def _on_reset(self, *_):
        self.running = False
        self.x = [0.0]
        self.y = [0.0]
        self._init_x_limits_set = False
        dpg.set_value(self.series_tag, [self.x, self.y])
        self._apply_x_limits()
        self._apply_y_limits()
        dpg.set_value(self.stats_points, f"points: {len(self.x)}")

    def _on_step_change(self, sender, app_data, *_):
        self.step_sigma = max(0.0, float(app_data))

    def _on_window_change(self, sender, app_data, *_):
        self.window_size = max(50, int(app_data))
        # Re-apply limits immediately
        self._apply_x_limits()
        self._apply_y_limits()

    def _on_autoscale_y_toggle(self, sender, app_data, *_):
        self.autoscale_y = bool(app_data)
        self._apply_y_limits()

    # ----- Per-frame work -----
    def on_frame(self):
        # Keep window same size as viewport
        vp_w = dpg.get_viewport_client_width()
        vp_h = dpg.get_viewport_client_height()
        if (vp_w, vp_h) != (self._vp_w, self._vp_h):
            self._vp_w, self._vp_h = vp_w, vp_h
            dpg.set_item_pos(self.window_tag, (0, 0))
            dpg.set_item_width(self.window_tag, max(1, vp_w))
            dpg.set_item_height(self.window_tag, max(1, vp_h))

        # Update data
        if self.running:
            t = self.x[-1] + 1.0
            y = self.y[-1] + random.gauss(0.0, self.step_sigma)
            self.x.append(t)
            self.y.append(y)

            if len(self.x) > self.max_points:
                self.x = self.x[-self.max_points:]
                self.y = self.y[-self.max_points:]

            dpg.set_value(self.series_tag, [self.x, self.y])
            dpg.set_value(self.stats_points, f"points: {len(self.x)}")

            # Scroll X and update Y every point
            self._apply_x_limits()
            self._apply_y_limits()

    # ----- Axis management -----
    def _apply_x_limits(self):
        """Keep a fixed-width X window that scrolls with the latest point."""
        if not self.x:
            return

        # On first few frames, set initial window 0..window_size (or up to last t)
        t_max = self.x[-1]
        if not self._init_x_limits_set:
            x0 = 0.0
            x1 = max(float(self.window_size), t_max)
            dpg.set_axis_limits(self.x_axis, x0, x1)
            self._init_x_limits_set = True
            return

        # Slide window to keep last 'window_size' points in view
        # Points are spaced by 1.0 in this implementation, so we can use indices or t values.
        if len(self.x) <= self.window_size:
            # Still within the initial window; expand x_max as needed.
            x0, x1 = 0.0, max(float(self.window_size), t_max)
        else:
            x0 = t_max - (self.window_size - 1)
            x1 = t_max
        dpg.set_axis_limits(self.x_axis, float(x0), float(x1))

    def _apply_y_limits(self):
        """Either auto-fit Y to the visible X window or leave as-is."""
        if not self.autoscale_y or not self.x:
            return

        # Determine indices of the visible X window
        n = len(self.x)
        if n <= self.window_size:
            y_window = self.y
        else:
            y_window = self.y[-self.window_size:]

        y_min = min(y_window)
        y_max = max(y_window)
        if y_min == y_max:
            # Avoid zero-height range
            pad = 1.0 if y_min == 0 else abs(y_min) * 0.1 + 1.0
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.10  # 10% padding
            y_min -= pad
            y_max += pad

        dpg.set_axis_limits(self.y_axis, float(y_min), float(y_max))


def main():
    dpg.create_context()
    dpg.create_viewport(title="Random Walk — Dear PyGui", width=1000, height=650)

    app = RandomWalkApp()
    app.build_ui()

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Custom render loop
    while dpg.is_dearpygui_running():
        app.on_frame()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
