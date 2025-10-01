import numpy as np

from roi_stream.roi import CirclesROI


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

