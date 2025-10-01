import numpy as np

from roi_stream.roi import to_uint16_gray


def test_uint8_to_uint16_gray():
    img = np.array([[0, 1, 255]], dtype=np.uint8)
    out = to_uint16_gray(img)
    assert out.dtype == np.uint16
    assert out.shape == img.shape
    # 0 -> 0, 1 -> 257, 255 -> 65535
    assert int(out[0, 0]) == 0
    assert int(out[0, 1]) == 257
    assert int(out[0, 2]) == 65535


def test_float01_to_uint16_gray():
    img = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = to_uint16_gray(img)
    assert int(out[0, 0]) == 0
    assert int(out[0, 1]) in (32767, 32768)  # rounding
    assert int(out[0, 2]) == 65535


def test_bgr_uint8_to_uint16_gray():
    # BGR pixel: blue=0, green=255, red=0 -> grayscale luma ~ 150
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0, 1] = 255  # G
    out = to_uint16_gray(bgr)
    assert out.ndim == 2 and out.shape == (1, 1)
    v = int(out[0, 0])
    # Roughly 0.587*255 * 257
    expected = int(round(0.5870430745 * 255.0 * 257.0))
    assert abs(v - expected) <= 1

