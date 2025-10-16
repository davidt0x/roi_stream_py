import numpy as np

from roi_stream.shared import TraceRing


def test_trace_ring_snapshot_window_limits():
    ring = TraceRing(k=2, maxlen=5)
    times = []
    for i in range(7):
        t = i * 0.5
        ring.append(t, np.array([i, i + 10], dtype=np.float64))
        times.append(t)

    assert ring.total_count() == 7

    all_t, all_y = ring.snapshot()
    assert isinstance(all_t, np.ndarray)
    assert isinstance(all_y, np.ndarray)
    assert all_t.shape == (5,)
    assert all_y.shape == (2, 5)
    # Newest timestamps retained
    assert np.allclose(all_t, np.array(times[-5:]))
    assert np.allclose(all_y[0], np.array(range(2, 7)))

    # Windowed snapshot
    window_t, window_y = ring.snapshot_window(start_time=2.0)
    assert window_t.ndim == 1
    assert window_y.shape[0] == 2
    assert np.all(window_t >= 2.0)

    # Max points respected
    capped_t, capped_y = ring.snapshot_window(start_time=0.0, max_points=3)
    assert capped_t.shape == (3,)
    assert capped_y.shape == (2, 3)
    assert np.allclose(capped_t, np.array(times[-3:]))
