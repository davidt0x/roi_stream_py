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

from tests.gui_app_imgui_harness import registered_cases


SCRIPTED_CASES = list(registered_cases(include_requires_exit=False))


@pytest.mark.gui
@pytest.mark.parametrize("case", SCRIPTED_CASES, ids=lambda case: case.label)
def test_imgui_scripted_entry(imgui_test_suite, case):
    entry = imgui_test_suite.find_entry(case.label)
    imgui_test_suite.run_entry(entry)
