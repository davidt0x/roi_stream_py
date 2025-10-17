import os
import sys

import pytest

try:
    from imgui_bundle import imgui  # noqa: F401
    from imgui_bundle import hello_imgui  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("imgui_bundle not available", allow_module_level=True)

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("No DISPLAY available for imgui_bundle", allow_module_level=True)

from tests.gui_app_imgui_harness import run_imgui_test_engine


@pytest.mark.gui
def test_imgui_test_engine_suite_runs():
    assert run_imgui_test_engine()
