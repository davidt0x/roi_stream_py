import numpy as np

from roi_stream.roi import ROISet, CirclesROI, ROIShape


def test_circles_roi_means_simple():
    # 10x10 frame with known pattern: value = x + 10*y
    H, W = 10, 10
    yy, xx = np.mgrid[0:H, 0:W]
    frame16 = (xx + 10 * yy).astype(np.uint16)

    # Circle centered at (5,5) radius 3
    circles = np.array([[5.0, 5.0, 3.0]], dtype=float)
    roi = CirclesROI(height=H, width=W, circles=circles)
    means = roi.compute_means(frame16)

    # Compute expected using mask
    d = (xx - 5.0) ** 2 + (yy - 5.0) ** 2 <= 3.0 ** 2
    expected = frame16[d].mean(dtype=np.float64)

    assert means.shape == (1,)
    assert np.isfinite(means[0])
    assert abs(float(means[0]) - float(expected)) < 1e-6


def test_ellipse_roi_means_with_rotation():
    H, W = 20, 20
    yy, xx = np.mgrid[0:H, 0:W]
    frame16 = ((xx * 2) + (yy * 3)).astype(np.uint16)

    shape = ROIShape(cx=10.0, cy=10.0, rx=5.0, ry=3.0, angle_deg=30.0)
    roi = ROISet(height=H, width=W, table=[shape])
    means = roi.compute_means(frame16)

    angle_rad = np.deg2rad(shape.angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    dx = xx - shape.cx
    dy = yy - shape.cy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    mask = (xr / shape.rx) ** 2 + (yr / shape.ry) ** 2 <= 1.0
    expected = frame16[mask].mean(dtype=np.float64)

    assert means.shape == (1,)
    assert np.isfinite(means[0])
    assert abs(float(means[0]) - float(expected)) < 1e-6
