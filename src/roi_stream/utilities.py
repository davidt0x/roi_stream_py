from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import yaml


def generate_random_circles(width: int, height: int, count: int = 35,
                            r_min_frac: float = 0.02, r_max_frac: float = 0.06,
                            seed: int = 12345) -> np.ndarray:
    """Generate random non-overflowing circular ROIs within the frame.

    - Centers are uniform within [r, W-1-r] x [r, H-1-r]
    - Radii are uniform in [r_min_frac*min(H,W), r_max_frac*min(H,W)]
    """
    W = int(width)
    H = int(height)
    K = max(0, int(count))
    rng = np.random.default_rng(int(seed))
    min_dim = float(min(H, W))
    rmin = max(4.0, float(r_min_frac) * min_dim)
    rmax = max(rmin + 1.0, float(r_max_frac) * min_dim)
    rs = rng.uniform(rmin, rmax, size=K)
    xs = rng.uniform(rs, W - 1 - rs)
    ys = rng.uniform(rs, H - 1 - rs)
    circles = np.stack([xs, ys, rs], axis=1)
    return circles.astype(float, copy=False)


def save_rois_yaml(path: Path | str, circles: np.ndarray, resolution: Tuple[int, int] | None = None) -> None:
    """Write a YAML ROI file with circle entries (ellipse-compatible schema)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    shapes = []
    for idx, row in enumerate(circles):
        shapes.append(
            {
                "name": idx,
                "center": [float(row[0]), float(row[1])],
                "radius": float(row[2]),
            }
        )
    data: dict = {"shapes": shapes}
    if resolution is not None:
        data["resolution"] = [int(resolution[0]), int(resolution[1])]
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main_random_rois(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roi_stream_make_random_rois",
                                 description="Generate a YAML file of random circular ROI definitions")
    ap.add_argument("--width", type=int, default=1280, help="Frame width in pixels")
    ap.add_argument("--height", type=int, default=720, help="Frame height in pixels")
    ap.add_argument("--count", type=int, default=35, help="Number of ROIs to generate")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")
    ap.add_argument("--min-frac", type=float, default=0.02, help="Min radius as fraction of min(height,width)")
    ap.add_argument("--max-frac", type=float, default=0.06, help="Max radius as fraction of min(height,width)")
    ap.add_argument("--out", type=str, default=str(Path("examples/rois_random.yaml")), help="Output YAML path")
    args = ap.parse_args(argv)

    circles = generate_random_circles(args.width, args.height, args.count, args.min_frac, args.max_frac, args.seed)
    save_rois_yaml(args.out, circles, (args.width, args.height))
    print(f"Wrote {len(circles)} ROIs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_random_rois())
