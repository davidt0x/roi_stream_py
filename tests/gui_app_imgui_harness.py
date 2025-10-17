"""
ImGui Test Engine harness utilities for the ROI Stream viewer.

The helpers defined here are used both by the pytest suite and by the
interactive runner in :mod:`tests.run_imgui_test_engine`.
"""

from __future__ import annotations

import math
import threading
import inspect
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import traceback
from imgui_bundle import ImVec2, hello_imgui, imgui as _imgui_ns
from imgui_bundle.imgui.test_engine_checks import CHECK

from roi_stream.gui_app import ViewerApp
from roi_stream.shared import SharedState, TraceRing

imgui = _imgui_ns
if not hasattr(imgui, "WindowFlags_"):
    imgui = import_module("imgui_bundle._imgui_bundle.imgui")


__all__ = [
    "STATUS_FAILED",
    "STATUS_PASSED",
    "ViewerTestSuite",
    "create_viewer_test_suite",
    "imgui_case",
    "register_imgui_case_module",
    "list_scripted_test_labels",
    "registered_cases",
    "run_imgui_test_engine",
]


STATUS_READY = "ready"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

_SCRIPTED_CATEGORY = "ViewerApp"
_DEFAULT_CASE_MODULES = ["tests.gui_app_imgui_cases"]
_TEST_REGISTRY: Dict[str, "RegisteredCase"] = {}
_CASES_LOADED = False


@dataclass
class ScriptedTestEntry:
    test: Any
    category: str
    name: str
    label: str
    status: str = STATUS_READY
    error: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    requires_exit: bool = False
    case: Optional["RegisteredCase"] = None


@dataclass
class RegisteredCase:
    key: str
    name: str
    func: Callable[[imgui.test_engine.TestContext, ViewerApp], None]
    requires_exit: bool = False
    auto_set_ref: bool = True

    @property
    def label(self) -> str:
        return _make_label(_SCRIPTED_CATEGORY, self.name)


def imgui_case(
    *,
    name: Optional[str] = None,
    requires_exit: bool = False,
    auto_set_ref: bool = True,
) -> Callable[[Callable[[imgui.test_engine.TestContext, ViewerApp], None]], Callable[[imgui.test_engine.TestContext, ViewerApp], None]]:
    def decorator(func: Callable[[imgui.test_engine.TestContext, ViewerApp], None]) -> Callable[[imgui.test_engine.TestContext, ViewerApp], None]:
        if not func.__name__.startswith("test_"):
            raise ValueError("ImGui case functions must start with 'test_' to integrate with pytest")
        params = inspect.signature(func).parameters
        if len(params) != 2:
            raise ValueError("ImGui case functions must accept exactly two parameters: (ctx, app)")

        def wrapped(ctx: imgui.test_engine.TestContext, app: ViewerApp) -> None:
            if auto_set_ref:
                ctx.set_ref(app.window_title)
            func(ctx, app)

        display_name = name or func.__name__[len("test_") :].replace("_", " ").title()
        key = func.__name__
        if key in _TEST_REGISTRY:
            raise ValueError(f"Duplicate ImGui test case registered: {key}")
        _TEST_REGISTRY[key] = RegisteredCase(
            key=key,
            name=display_name,
            func=wrapped if auto_set_ref else func,
            requires_exit=requires_exit,
            auto_set_ref=auto_set_ref,
        )
        return wrapped if auto_set_ref else func

    return decorator


def _ensure_cases_loaded() -> None:
    global _CASES_LOADED
    if _CASES_LOADED:
        return
    for module_name in _DEFAULT_CASE_MODULES:
        try:
            import_module(module_name)
        except Exception:
            continue
    _CASES_LOADED = True


def registered_cases(*, include_requires_exit: bool = True) -> Iterable[RegisteredCase]:
    _ensure_cases_loaded()
    cases = sorted(_TEST_REGISTRY.values(), key=lambda c: c.label)
    for case in cases:
        if not include_requires_exit and case.requires_exit:
            continue
        yield case


def register_imgui_case_module(module_name: str) -> None:
    if module_name in _DEFAULT_CASE_MODULES:
        return
    _DEFAULT_CASE_MODULES.append(module_name)
    if _CASES_LOADED:
        import_module(module_name)


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
        else:
            entry.status = STATUS_PASSED
        finally:
            entry.done.set()

    return wrapped


def _queue_tests(entries: List[ScriptedTestEntry]) -> None:
    engine = hello_imgui.get_imgui_test_engine()
    if engine is None:
        return
    engine_io = imgui.test_engine.get_io(engine)
    engine_io.config_run_speed = imgui.test_engine.TestRunSpeed.fast
    for entry in entries:
        entry.error = ""
        entry.done.clear()
        entry.status = STATUS_QUEUED
        entry.error = ""
        imgui.test_engine.queue_test(engine, entry.test)


def check_ui(condition: bool, message: str) -> None:
    CHECK(condition)
    if not condition:
        raise AssertionError(message)


def list_scripted_test_labels(*, include_requires_exit: bool = True) -> List[str]:
    return [case.label for case in registered_cases(include_requires_exit=include_requires_exit)]


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

    _ensure_cases_loaded()
    if not _TEST_REGISTRY:
        raise RuntimeError("No ImGui test cases registered")

    for case in registered_cases(include_requires_exit=request_exit):
        test = imgui.test_engine.register_test(engine, _SCRIPTED_CATEGORY, case.name)
        entry = ScriptedTestEntry(
            test=test,
            category=_SCRIPTED_CATEGORY,
            name=case.name,
            label=case.label,
            requires_exit=case.requires_exit,
            case=case,
        )

        def make_func(case: RegisteredCase) -> Callable[[imgui.test_engine.TestContext], None]:
            def _run(ctx: imgui.test_engine.TestContext) -> None:
                case.func(ctx, app)

            return _run

        test.test_func = _wrap_test(entry, make_func(case))
        entry.done.set()
        entries.append(entry)

    if auto_queue:
        _queue_tests(entries)

    return entries


@dataclass
class ViewerTestSuite:
    app: ViewerApp
    stop_event: threading.Event
    runner_params: hello_imgui.RunnerParams
    entries: List[ScriptedTestEntry]
    show_test_engine: bool
    registered: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def start(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            return

        def _run_app() -> None:
            try:
                hello_imgui.run(self.runner_params)
            finally:
                self.app.teardown()

        self._thread = threading.Thread(target=_run_app, name="imgui-test-engine", daemon=True)
        self._thread.start()
        if not self.registered.wait(timeout):
            raise RuntimeError("ImGui test registration timed out")

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        params = self.runner_params
        if params is not None:
            params.app_shall_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def find_entry(self, label: str) -> ScriptedTestEntry:
        for entry in self.entries:
            if entry.label == label:
                return entry
        raise KeyError(label)

    def run_entry(self, entry: ScriptedTestEntry, timeout: float = 10.0) -> None:
        engine = hello_imgui.get_imgui_test_engine()
        if engine is None:
            raise RuntimeError("ImGui test engine is not available")
        entry.error = ""
        entry.status = STATUS_READY
        entry.done.clear()
        entry.status = STATUS_QUEUED
        imgui.test_engine.queue_test(engine, entry.test)
        if not entry.done.wait(timeout):
            raise TimeoutError(f"ImGui test '{entry.label}' timed out")
        if entry.status != STATUS_PASSED:
            raise AssertionError(entry.error or f"ImGui test '{entry.label}' failed")


def create_viewer_test_suite(
    *,
    window_title: str = "ROI Stream Viewer (ImGui Tests)",
    auto_queue: bool = False,
    request_exit: bool = True,
    show_test_engine: bool = False,
) -> ViewerTestSuite:
    shared, stop_event = build_sample_shared_state()
    app = ViewerApp(shared, stop_event, window_title=window_title)
    app.setup(width=960, height=620)
    app.fullscreen_window = False

    runner_params = app._runner_params
    if runner_params is None:
        raise RuntimeError("ViewerApp.setup() did not initialize runner parameters.")

    runner_params.use_imgui_test_engine = True
    runner_params.app_window_params.hidden = not show_test_engine

    suite = ViewerTestSuite(
        app=app,
        stop_event=stop_event,
        runner_params=runner_params,
        entries=[],
        show_test_engine=show_test_engine,
    )

    def register_tests() -> None:
        suite.entries = register_viewer_tests(app, auto_queue=auto_queue, request_exit=request_exit)
        suite.registered.set()

    runner_params.callbacks.register_tests = register_tests
    _install_show_gui_callback(suite)
    return suite


def _install_show_gui_callback(suite: ViewerTestSuite) -> None:
    callbacks = suite.runner_params.callbacks
    original_show_gui = callbacks.show_gui

    if not suite.show_test_engine:
        return

    def show_gui_with_test_engine() -> None:
        engine = hello_imgui.get_imgui_test_engine()
        viewport = imgui.get_main_viewport()

        left_width = 0.0
        if viewport is not None and engine is not None:
            pos = viewport.pos
            size = viewport.size
            left_width = min(max(320.0, size.x * 0.3), max(380.0, size.x * 0.45))
            left_width = min(left_width, max(0.0, size.x - 320.0))
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
                viewer_width = size.x - left_width if engine is not None else size.x
                viewer_width = max(320.0, viewer_width)
                viewer_pos_x = pos.x + left_width
                imgui.set_next_window_pos(ImVec2(viewer_pos_x, pos.y), cond=imgui.Cond_.always)
                imgui.set_next_window_size(ImVec2(viewer_width, size.y), cond=imgui.Cond_.always)
            original_show_gui()

    callbacks.show_gui = show_gui_with_test_engine


def run_imgui_test_engine(
    window_title: str = "ROI Stream Viewer (ImGui Tests)",
    *,
    auto_queue: bool = True,
    request_exit: bool = True,
    show_test_engine: bool = True,
) -> bool:
    """Boot the viewer with scripted tests registered and optionally queued."""
    suite = create_viewer_test_suite(
        window_title=window_title,
        auto_queue=auto_queue,
        request_exit=request_exit,
        show_test_engine=show_test_engine,
    )
    try:
        hello_imgui.run(suite.runner_params)
    finally:
        suite.app.teardown()

    return suite.stop_event.is_set()


def main() -> None:
    """Launch the viewer with tests registered but not auto-queued."""
    _ensure_cases_loaded()
    run_imgui_test_engine(auto_queue=False, request_exit=False, show_test_engine=True)


if __name__ == "__main__":
    import sys
    from importlib import import_module
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import_module("tests.gui_app_imgui_harness").main()
