from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_draw_dataset.drawing import sample_sketch


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    if np.linalg.norm(a - c) < 1e-4 or np.linalg.norm(a - d) < 1e-4:
        return False
    if np.linalg.norm(b - c) < 1e-4 or np.linalg.norm(b - d) < 1e-4:
        return False
    return _ccw(a, c, d) != _ccw(b, c, d) and _ccw(a, b, c) != _ccw(a, b, d)


def count_self_crossings(points: np.ndarray, stride: int = 1) -> int:
    pts = points[::stride]
    total = 0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for j in range(i + 2, len(pts) - 1):
            if j == i + 1:
                continue
            c, d = pts[j], pts[j + 1]
            if _segments_cross(a, b, c, d):
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/one_stroke_interwoven.yaml")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(args.seed)
    crossings = []
    lengths = []
    for _ in range(args.samples):
        strokes, _ = sample_sketch(cfg["sketch"], rng)
        pts = np.asarray(strokes[0], dtype=np.float32)
        crossings.append(count_self_crossings(pts))
        lengths.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))

    arr = np.asarray(crossings, dtype=np.float32)
    out = {
        "samples": args.samples,
        "crossings_min": int(arr.min()),
        "crossings_p25": float(np.percentile(arr, 25)),
        "crossings_mean": float(arr.mean()),
        "crossings_median": float(np.median(arr)),
        "crossings_p75": float(np.percentile(arr, 75)),
        "crossings_max": int(arr.max()),
        "path_length_mean": float(np.mean(lengths)),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
