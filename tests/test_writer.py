from pathlib import Path
import numpy as np
import h5py

from roi_stream.writer import H5TracesWriter


def test_writer_append_and_finalize(tmp_path: Path):
    circles = np.array([[10.0, 10.0, 5.0], [20.0, 15.0, 7.0]], dtype=float)
    meta = {"source": "unittest", "resolution": np.array([64, 48], dtype=np.int32)}

    out = tmp_path / "traces_test.h5"
    w = H5TracesWriter(out, circles, meta, chunk_frames=4)

    t1 = np.linspace(0.0, 0.3, 4, dtype=np.float64)
    m1 = np.ones((4, 2), dtype=np.float32)
    w.append(t1, m1)

    t2 = np.linspace(0.4, 0.7, 4, dtype=np.float64)
    m2 = 2.0 * np.ones((4, 2), dtype=np.float32)
    w.append(t2, m2)

    w.finalize({"frames_seen": 8})

    assert out.exists()
    with h5py.File(out, "r") as f:
        time_ds = f["/time"]
        means_ds = f["/roi/means"]
        shapes_ds = f["/roi/shapes"]
        circles_ds = f["/roi/circles"]

        assert time_ds.shape == (8, 1)  # keep 2D shape
        assert means_ds.shape == (8, 2)
        assert shapes_ds.shape == (2, 5)
        assert circles_ds.shape == (2, 3)

        # Check values
        times = time_ds[:].reshape(-1)
        means = means_ds[:]
        assert np.allclose(times[:4], t1)
        assert np.allclose(times[4:], t2)
        assert np.allclose(means[:4], m1)
        assert np.allclose(means[4:], m2)
        assert np.allclose(shapes_ds[:, 2], shapes_ds[:, 3])
        assert np.allclose(shapes_ds[:, 4], 0.0)


def test_writer_elliptical_rois(tmp_path: Path):
    rois = np.array(
        [
            [12.0, 18.0, 5.0, 3.0, 15.0],
            [40.0, 22.0, 4.5, 7.5, -10.0],
        ],
        dtype=float,
    )
    meta = {"source": "unittest", "resolution": np.array([80, 60], dtype=np.int32)}

    out = tmp_path / "traces_ellipses.h5"
    writer = H5TracesWriter(out, rois, meta, chunk_frames=2)
    t = np.linspace(0.0, 0.1, 3, dtype=np.float64)
    m = np.ones((3, 2), dtype=np.float32)
    writer.append(t, m)
    writer.finalize({"frames_seen": 3})

    with h5py.File(out, "r") as f:
        shapes = f["/roi/shapes"][:]
        assert shapes.shape == (2, 5)
        assert np.allclose(shapes[:, 2:], rois[:, 2:])
        assert "/roi/circles" not in f
        assert f["/roi"].attrs.get("circles_geometry") == "ellipse"
