from __future__ import annotations

from tests.gui_app_imgui_harness import imgui_case, check_ui


@imgui_case()
def test_control_row_toggles(ctx, app):
    ctx.item_input_value("X window (s)", 20.0)
    check_ui(abs(app.window_sec - 20.0) < 1e-3, "Window range did not update")
    ctx.item_click("Single Plot")
    check_ui(app.display_mode == "Overlay", "Single Plot toggle failed")
    ctx.item_click("Single Plot")
    check_ui(app.display_mode == "Stacked", "Single Plot toggle did not revert")

    ctx.item_click("Lock global Y")
    check_ui(app.lock_global_y is True, "Lock global Y did not enable")
    ctx.item_click("Lock global Y")
    check_ui(app.lock_global_y is False, "Lock global Y did not disable")


@imgui_case()
def test_preview_controls(ctx, app):
    ctx.item_open("Preview")
    ctx.item_input_value("Preview height (fraction)", 0.55)
    check_ui(abs(app.preview_frac - 0.55) < 1e-2, "Preview slider did not update")
