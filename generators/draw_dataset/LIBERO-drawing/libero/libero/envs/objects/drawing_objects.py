import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import register_object


ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "custom_drawing"
DRAW_DATASET_ROOT = Path(__file__).resolve().parents[5]
if str(DRAW_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(DRAW_DATASET_ROOT))
PAPER_RGB = (245, 245, 240)
FRAME_RGB = (214, 204, 184)
LINE_RGB = (14, 14, 14)

try:
    from sim_draw_dataset.drawing import rasterize_strokes, sample_sketch
except Exception:
    rasterize_strokes = None
    sample_sketch = None


def ensure_drawing_textures() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["drawing_board_reference.png", "drawing_board_blank.png"]:
        path = ASSET_DIR / name
        if path.exists():
            continue
        img = Image.new("RGB", (256, 256), PAPER_RGB)
        draw = ImageDraw.Draw(img)
        draw.rectangle([2, 2, 253, 253], outline=FRAME_RGB, width=4)
        img.save(path)


class DrawingXMLObject(MujocoXMLObject):
    def __init__(self, name, xml_name, joints=None):
        ensure_drawing_textures()
        super().__init__(
            os.path.join(str(ASSET_DIR), f"{xml_name}.xml"),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class DrawingBoardReference(DrawingXMLObject):
    def __init__(self, name="drawing_board_reference", joints=None):
        super().__init__(name, "drawing_board_reference", joints=joints)


@register_object
class DrawingBoardBlank(DrawingXMLObject):
    def __init__(self, name="drawing_board_blank", joints=None):
        super().__init__(name, "drawing_board_blank", joints=joints)


@register_object
class BrushRack(DrawingXMLObject):
    def __init__(self, name="brush_rack", joints=None):
        super().__init__(name, "brush_rack", joints=joints)


@register_object
class DrawingWaterBucket(DrawingXMLObject):
    def __init__(self, name="drawing_water_bucket", joints=None):
        super().__init__(name, "drawing_water_bucket", joints=joints)


@register_object
class SoftBrush(DrawingXMLObject):
    def __init__(self, name="soft_brush", joints=[dict(type="free", damping="0.02")]):
        super().__init__(name, "soft_brush", joints=joints)
        self.rotation = (0, 0)
        self.rotation_axis = "z"


def sample_oneline_pixels(seed: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    image, strokes, _ = sample_drawing_pixels(seed, size=size)
    curve = flatten_strokes(strokes)
    return image, curve


def sample_drawing_pixels(seed: int, size: int = 256) -> tuple[np.ndarray, list[np.ndarray], str]:
    if sample_sketch is not None and rasterize_strokes is not None:
        rng = np.random.default_rng(seed)
        cfg = _drawing_generator_config(seed)
        strokes_raw, sketch_type = sample_sketch(cfg, rng)
        strokes = [
            np.asarray(stroke, dtype=np.float32)
            for stroke in strokes_raw
            if len(stroke) > 0
        ]
        if strokes:
            image = rasterize_strokes(
                [[(float(x), float(y)) for x, y in stroke] for stroke in strokes],
                size=(size, size),
                stroke_width=4,
                line_rgb=LINE_RGB,
                board_rgb=PAPER_RGB,
            )
            image = _add_board_frame(image)
            return image, strokes, sketch_type
    return _legacy_oneline_pixels(seed, size)


def flatten_strokes(strokes: list[np.ndarray]) -> np.ndarray:
    if not strokes:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate([np.asarray(stroke, dtype=np.float32) for stroke in strokes if len(stroke)], axis=0)


def _drawing_generator_config(seed: int) -> dict[str, Any]:
    if seed % 5 == 0:
        return {
            "min_strokes": 2,
            "max_strokes": 5,
            "min_points_per_stroke": 3,
            "max_points_per_stroke": 8,
            "shape_mix": {
                "polyline": 0.14,
                "rectangle": 0.05,
                "circle": 0.20,
                "house": 0.16,
                "star": 0.16,
                "face": 0.29,
            },
        }
    if seed % 5 == 1:
        return {
            "generator": "ornamental_oneline",
            "min_points": 100,
            "max_points": 190,
            "margin": 0.08,
            "jitter": 0.002,
            "families": {
                "shape_chain": 0.04,
                "nested_shapes": 0.04,
                "icon_outline": 0.14,
                "butterfly": 0.14,
                "flower": 0.13,
                "music_note": 0.12,
                "key": 0.10,
                "planet": 0.10,
                "tree": 0.08,
                "bridge": 0.07,
                "smooth_walk": 0.04,
            },
        }
    if seed % 5 == 2:
        return {
            "generator": "ornamental_oneline",
            "min_points": 110,
            "max_points": 210,
            "margin": 0.08,
            "jitter": 0.002,
            "families": {
                "braid": 0.10,
                "knot": 0.10,
                "weave_grid": 0.08,
                "center_pass": 0.12,
                "rosette_cross": 0.10,
                "ribbon": 0.08,
                "spiral_maze": 0.18,
                "planet": 0.10,
                "flower": 0.08,
                "butterfly": 0.06,
            },
        }
    if seed % 5 == 3:
        return {
            "generator": "ornamental_oneline",
            "min_points": 100,
            "max_points": 190,
            "margin": 0.09,
            "jitter": 0.002,
            "families": {
                "hanzi_font": 0.50,
                "hanzi_simple": 0.18,
                "hanzi_box": 0.17,
                "hanzi_cursive": 0.15,
            },
            "hanzi_font_path": "assets/fonts/NotoSansCJKsc-Regular.otf",
        }
    return {
        "min_strokes": 3,
        "max_strokes": 5,
        "min_points_per_stroke": 2,
        "max_points_per_stroke": 6,
        "shape_mix": {
            "polyline": 0.16,
            "rectangle": 0.04,
            "circle": 0.22,
            "house": 0.16,
            "star": 0.18,
            "face": 0.24,
        },
    }


def _add_board_frame(image: np.ndarray) -> np.ndarray:
    out = np.asarray(image, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (2, 2), (w - 3, h - 3), FRAME_RGB, 4, lineType=cv2.LINE_AA)
    return out


def _legacy_oneline_pixels(seed: int, size: int = 256) -> tuple[np.ndarray, list[np.ndarray], str]:
    rng = np.random.default_rng(seed)
    family = int(rng.integers(0, 8))
    n = int(rng.integers(110, 170))
    if family == 0:
        turns = int(rng.integers(5, 9))
        t = np.linspace(0, 1, n)
        pts = np.stack([t * 2 - 1, 0.55 * np.sin(2 * np.pi * turns * t + rng.uniform(-0.5, 0.5))], axis=1)
    elif family == 1:
        t = np.linspace(0.2, 1.0, n)
        theta = np.linspace(float(rng.uniform(0, np.pi)), float(rng.uniform(3.8 * np.pi, 5.6 * np.pi)), n)
        r = t
        pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    elif family == 2:
        t = np.linspace(-1, 1, n)
        pts = np.stack([t, 0.42 * np.sin(float(rng.uniform(2.0, 4.0)) * np.pi * t)], axis=1)
    elif family == 3:
        rows = int(rng.integers(3, 6))
        cols = int(rng.integers(4, 7))
        raw = []
        for r in range(rows):
            xs = np.linspace(-1.0, 1.0, cols)
            if r % 2:
                xs = xs[::-1]
            y = -1.0 + 2.0 * r / max(1, rows - 1)
            raw.extend([[float(x), float(y)] for x in xs])
        pts = _resample_polyline(np.asarray(raw, dtype=np.float32), n)
    elif family == 4:
        t = np.linspace(0, 1, n)
        x = 2 * t - 1
        y = np.interp(t, np.linspace(0, 1, 8), rng.uniform(-0.8, 0.8, size=8))
        pts = np.stack([x, y], axis=1)
    elif family == 5:
        angles = np.linspace(float(rng.uniform(-0.6, 0.6)), float(rng.uniform(1.2 * np.pi, 1.8 * np.pi)), n)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    elif family == 6:
        raw = [[-1.0, -0.8], [-0.35, 0.8], [0.15, -0.55], [0.55, 0.65], [1.0, -0.25]]
        pts = _resample_polyline(np.asarray(raw, dtype=np.float32), n)
    else:
        anchors = rng.uniform(-0.85, 0.85, size=(int(rng.integers(5, 9)), 2)).astype(np.float32)
        anchors[:, 0] = np.linspace(-1.0, 1.0, len(anchors))
        pts = _resample_polyline(anchors, n)

    pts = _normalize_points(_resample_polyline(pts, n), margin=0.18)
    px = np.round(pts * (size - 1)).astype(np.int32)
    img = Image.new("RGB", (size, size), PAPER_RGB)
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, size - 3, size - 3], outline=FRAME_RGB, width=4)
    draw.line([tuple(p) for p in px], fill=LINE_RGB, width=4, joint="curve")
    return np.asarray(img, dtype=np.uint8), [pts.astype(np.float32)], "legacy_oneline"


def blank_board_pixels(size: int = 256) -> np.ndarray:
    img = Image.new("RGB", (size, size), PAPER_RGB)
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, size - 3, size - 3], outline=FRAME_RGB, width=4)
    return np.asarray(img, dtype=np.uint8)


def _normalize_points(points: np.ndarray, margin: float) -> np.ndarray:
    pts = points.astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = float(np.max(np.ptp(pts, axis=0)))
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    pts = pts * (1.0 - 2.0 * margin) + 0.5
    return np.clip(pts, margin, 1.0 - margin)


def _resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    if dist[-1] <= 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0, dist[-1], n)
    x = np.interp(target, dist, pts[:, 0])
    y = np.interp(target, dist, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)
