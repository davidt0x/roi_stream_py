"""
ImGui Test Engine harness utilities for the ROI Stream viewer.

The helpers defined here are used both by the pytest suite and by the
interactive runner in :mod:`tests.run_imgui_test_engine`.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import traceback
from importlib import import_module
from imgui_bundle import ImVec2, hello_imgui, imgui as _imgui_ns
from imgui_bundle.imgui.test_engine_checks import CHECK

from roi_stream.gui_app import ViewerApp
from roi_stream.shared import SharedState, TraceRing

imgui = _imgui_ns
if not hasattr(imgui, "WindowFlags_"):
    imgui = import_module("imgui_bundle._imgui_bundle.imgui")


STATUS_READY = "ready"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

@dataclass
class ScriptedTestEntry:
    test: Any
    label: str
    status: str = STATUS_READY
    error: str = ""


def _make_label(category: str, name: str) -> str:
    return f"{category} / {name}" if category else name


def _wrap_test(entry: ScriptedTestEntry, func):
    def wrapped(ctx: imgui.test_engine.TestContext) -> None:
        entry.status = STATUS_RUNNING
        entry.error = ""
        try:
            func(ctx)
        except Exception:
            entry.status = STATUS_FAILED
            entry.error = traceback.format_exc(limit=6)
            raise
        else:
            entry.status = STATUS_PASSED

    return wrapped


def _queue_tests(entries: List[ScriptedTestEntry]) -> None:
    engine = hello_imgui.get_imgui_test_engine()
    if engine is None:
        return
    engine_io = imgui.test_engine.get_io(engine)
    engine_io.config_run_speed = imgui.test_engine.TestRunSpeed.fast
    for entry in entries:
        entry.status = STATUS_QUEUED
        entry.error = ""
        imgui.test_engine.queue_test(engine, entry.test)


def build_sample_shared_state() -> Tuple[SharedState, threading.Event]:
    """Populate a SharedState with deterministic trace data and preview imagery."""
    traces = TraceRing(k=3, maxlen=1024)
    shared = SharedState(traces=traces)

    t_samples = np.linspace(0.0, 30.0, num=256, dtype=np.float64)
    for t in t_samples:
        trace_vector = np.array(
            [
                math.sin(t),
                math.cos(t * 0.5),
                0.5 * math.sin(t * 0.25) + 0.1,
            ],
            dtype=np.float64,
        )
        traces.append(float(t), trace_vector)

    shared.circles = np.array(
        [
            (80.0, 70.0, 22.0),
            (140.0, 110.0, 18.0),
            (210.0, 65.0, 25.0),
        ],
        dtype=np.float32,
    )

    width, height = 320, 240
    grad8 = np.tile(np.linspace(0, 255, num=width, dtype=np.uint8), (height, 1))
    grad16 = np.tile(np.linspace(0, 65535, num=width, dtype=np.uint16), (height, 1))
    shared.update_frame(grad16, (width, height), preview8=grad8)

    return shared, threading.Event()


def register_viewer_tests(
    app: ViewerApp,
    *,
    auto_queue: bool = True,
    request_exit: bool = True,
) -> List[ScriptedTestEntry]:
    """Register scripted interactions against the viewer UI."""

    engine = hello_imgui.get_imgui_test_engine()
    if engine is None:
        raise RuntimeError("ImGui test engine is not available")

    entries: List[ScriptedTestEntry] = []

    category = "ViewerApp"

    test_controls = imgui.test_engine.register_test(engine, category, "Control Row Toggles")

    def test_controls_func(ctx: imgui.test_engine.TestContext) -> None:
        ctx.set_ref(app.window_title)
        ctx.item_input_value("X window (s)", 20.0)
        CHECK(math.isclose(app.window_sec, 20.0, rel_tol=1e-3, abs_tol=1e-3))

        ctx.item_click("Single Plot")
        CHECK(app.display_mode == "Overlay")
        ctx.item_click("Single Plot")
        CHECK(app.display_mode == "Stacked")

        ctx.item_click("Lock global Y")
        CHECK(app.lock_global_y is True)
        ctx.item_click("Lock global Y")
        CHECK(app.lock_global_y is False)

    entry_controls = ScriptedTestEntry(test_controls, _make_label(category, "Control Row Toggles"))
    test_controls.test_func = _wrap_test(entry_controls, test_controls_func)
    entries.append(entry_controls)

    test_preview = imgui.test_engine.register_test(engine, category, "Preview Controls")

    def test_preview_func(ctx: imgui.test_engine.TestContext) -> None:
        ctx.set_ref(app.window_title)
        ctx.item_open("Preview")
        ctx.item_input_value("Preview height (fraction)", 0.55)
        CHECK(math.isclose(app.preview_frac, 0.55, rel_tol=0.0, abs_tol=1e-2))

    entry_preview = ScriptedTestEntry(test_preview, _make_label(category, "Preview Controls"))
    test_preview.test_func = _wrap_test(entry_preview, test_preview_func)
    entries.append(entry_preview)

    test_shutdown = imgui.test_engine.register_test(engine, category, "Shutdown")

    def test_shutdown_func(ctx: imgui.test_engine.TestContext) -> None:
        if request_exit:
            app.stop_event.set()

    entry_shutdown = ScriptedTestEntry(test_shutdown, _make_label(category, "Shutdown"))
    test_shutdown.test_func = _wrap_test(entry_shutdown, test_shutdown_func)
    entries.append(entry_shutdown)

    if auto_queue:
        _queue_tests(entries)

    return entries


def run_imgui_test_engine(
    window_title: str = "ROI Stream Viewer (ImGui Tests)",
    *,
    auto_queue: bool = True,
    request_exit: bool = True,
) -> bool:
    """Boot the viewer with scripted tests registered and optionally queued."""

    shared, stop_event = build_sample_shared_state()
    app = ViewerApp(shared, stop_event, window_title=window_title)
    app.setup(width=960, height=620)
    app.fullscreen_window = False

    runner_params = app._runner_params
    if runner_params is None:
        raise RuntimeError("ViewerApp.setup() did not initialize runner parameters.")

    runner_params.use_imgui_test_engine = True

    def register_tests() -> None:
        register_viewer_tests(app, auto_queue=auto_queue, request_exit=request_exit)

    runner_params.callbacks.register_tests = register_tests

    callbacks = runner_params.callbacks
    original_show_gui = callbacks.show_gui

    def show_gui_with_test_engine() -> None:
        engine = hello_imgui.get_imgui_test_engine()
        viewport = imgui.get_main_viewport()

        left_width = 0.0
        if viewport is not None:
            pos = viewport.pos
            size = viewport.size
            if engine is not None:
                left_width = min(max(320.0, size.x * 0.3), max(380.0, size.x * 0.45))
                left_width = min(left_width, max(0.0, size.x - 320.0))
            else:
                left_width = 0.0
            if engine is not None and left_width > 0.0:
                imgui.set_next_window_pos(ImVec2(pos.x, pos.y), cond=imgui.Cond_.always)
                imgui.set_next_window_size(ImVec2(left_width, size.y), cond=imgui.Cond_.always)
                flags = (
                    imgui.WindowFlags_.no_collapse
                    | imgui.WindowFlags_.no_move
                    | imgui.WindowFlags_.no_resize
                )
                if imgui.begin("Dear ImGui Test Engine", flags=flags):
                    imgui.test_engine.show_test_engine_windows(engine, True)
                imgui.end()
        elif engine is not None:
            imgui.test_engine.show_test_engine_windows(engine, True)

        if original_show_gui is not None:
            if viewport is not None:
                pos = viewport.pos
                size = viewport.size
                viewer_width = max(320.0, size.x - left_width)
                viewer_pos_x = pos.x + left_width
                imgui.set_next_window_pos(ImVec2(viewer_pos_x, pos.y), cond=imgui.Cond_.always)
                imgui.set_next_window_size(ImVec2(viewer_width, size.y), cond=imgui.Cond_.always)
            original_show_gui()

    callbacks.show_gui = show_gui_with_test_engine

    try:
        hello_imgui.run(runner_params)
    finally:
        app.teardown()

    return stop_event.is_set()


def main() -> None:
    """Launch the viewer with tests registered but not auto-queued."""
    run_imgui_test_engine(auto_queue=False, request_exit=False)


if __name__ == "__main__":
    main()
