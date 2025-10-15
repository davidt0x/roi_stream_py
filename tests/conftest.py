from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, ContextManager, Iterator

import pytest

from roi_stream.gui_app import ViewerApp, _lazy_import_dpg


@pytest.fixture
def gui_test_context() -> Callable[..., ContextManager[ViewerApp]]:
    """Expose the gui_test_context helper as a pytest fixture."""
    @contextmanager
    def _context(
        shared_state,
        stop_event,
        *,
        width: int = 800,
        height: int = 600,
        show_viewport: bool = True,
        create_viewport: bool = True,
        disable_preview: bool = True,
    ) -> Iterator[ViewerApp]:
        """Context manager that creates a Dear PyGui session for tests."""
        dpg = _lazy_import_dpg()
        dpg.create_context()
        if create_viewport:
            dpg.create_viewport(title="ROI Stream Viewer", width=int(width), height=int(height))
        app = ViewerApp(shared_state, stop_event)
        if disable_preview:
            # Reduce preview refresh rate for tests to limit texture updates.
            try:
                app._preview_hz = 0.001
            except Exception:
                pass
        app.build_ui()
        dpg.setup_dearpygui()
        if create_viewport and show_viewport:
            dpg.show_viewport()
        try:
            yield app
        finally:
            try:
                dpg.destroy_context()
            except Exception:
                pass

    return _context
