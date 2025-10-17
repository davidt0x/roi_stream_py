import pytest

from tests.gui_app_imgui_harness import ViewerTestSuite, create_viewer_test_suite


@pytest.fixture(scope="session")
def imgui_test_suite() -> ViewerTestSuite:
    suite = create_viewer_test_suite(auto_queue=False, request_exit=False, show_test_engine=False)
    suite.start()
    yield suite
    suite.stop()
