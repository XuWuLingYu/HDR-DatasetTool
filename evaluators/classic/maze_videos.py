#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def load_struct(path: Path) -> dict[str, object]:
    return np.load(path, allow_pickle=True).item()


def build_edge_set(edges: np.ndarray) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edge_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r1, c1, r2, c2 in edges.tolist():
        a = (int(r1), int(c1))
        b = (int(r2), int(c2))
        edge_set.add((a, b))
        edge_set.add((b, a))
    return edge_set


def scaled_open_mask(wall_grid: np.ndarray, frame_size: int) -> np.ndarray:
    base = np.where(wall_grid == 1, 0, 255).astype(np.uint8)
    scaled = np.asarray(Image.fromarray(base, mode="L").resize((frame_size, frame_size), resample=Image.Resampling.NEAREST))
    return scaled > 0


def align_prediction_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.shape[0] == frame_size and frame.shape[1] == frame_size:
        return frame
    return np.asarray(
        Image.fromarray(frame, mode="RGB").resize((frame_size, frame_size), resample=Image.Resampling.BILINEAR)
    )


def red_mask(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    return (r >= 115) & (g <= 150) & (b <= 150) & ((r - g) >= 18) & ((r - b) >= 18)


def connected_components(mask: np.ndarray, min_area: int = 4) -> list[dict[str, object]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, object]] = []
    rows, cols = np.nonzero(mask)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if visited[row, col]:
            continue
        queue: deque[tuple[int, int]] = deque([(row, col)])
        visited[row, col] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            cur_row, cur_col = queue.popleft()
            pixels.append((cur_row, cur_col))
            for nxt_row, nxt_col in (
                (cur_row - 1, cur_col),
                (cur_row + 1, cur_col),
                (cur_row, cur_col - 1),
                (cur_row, cur_col + 1),
            ):
                if (
                    0 <= nxt_row < mask.shape[0]
                    and 0 <= nxt_col < mask.shape[1]
                    and mask[nxt_row, nxt_col]
                    and not visited[nxt_row, nxt_col]
                ):
                    visited[nxt_row, nxt_col] = True
                    queue.append((nxt_row, nxt_col))
        if len(pixels) >= min_area:
            pts = np.asarray(pixels, dtype=np.float32)
            components.append(
                {
                    "area": len(pixels),
                    "centroid": pts.mean(axis=0),
                }
            )
    components.sort(key=lambda item: int(item["area"]), reverse=True)
    return components


def formalize_route(centroids: list[np.ndarray], cell_centers: np.ndarray) -> list[tuple[int, int]]:
    flattened = cell_centers.reshape(-1, 2)
    maze_size = cell_centers.shape[0]
    route: list[tuple[int, int]] = []
    for centroid in centroids:
        distances = np.sum((flattened - centroid[None, :]) ** 2, axis=1)
        cell_index = int(np.argmin(distances))
        route.append((cell_index // maze_size, cell_index % maze_size))
    compressed: list[tuple[int, int]] = []
    for cell in route:
        if not compressed or compressed[-1] != cell:
            compressed.append(cell)
    return compressed


def compress_route(route: list[tuple[int, int]]) -> list[tuple[int, int]]:
    compressed: list[tuple[int, int]] = []
    for cell in route:
        if not compressed or compressed[-1] != cell:
            compressed.append(cell)
    return compressed


def nearest_cell(centroid: np.ndarray, cell_centers: np.ndarray) -> tuple[int, int]:
    flattened = cell_centers.reshape(-1, 2)
    maze_size = cell_centers.shape[0]
    distances = np.sum((flattened - centroid[None, :]) ** 2, axis=1)
    cell_index = int(np.argmin(distances))
    return (cell_index // maze_size, cell_index % maze_size)


def nominal_step_pixels(cell_centers: np.ndarray) -> float:
    maze_size = cell_centers.shape[0]
    if maze_size <= 1:
        return 8.0
    samples: list[float] = []
    for row in range(maze_size):
        for col in range(maze_size):
            if row + 1 < maze_size:
                samples.append(float(np.linalg.norm(cell_centers[row + 1, col] - cell_centers[row, col])))
            if col + 1 < maze_size:
                samples.append(float(np.linalg.norm(cell_centers[row, col + 1] - cell_centers[row, col])))
    return max(8.0, float(np.median(samples))) if samples else 8.0


def choose_tracked_component(
    components: list[dict[str, object]],
    target_centroid: np.ndarray,
    step_pixels: float,
) -> tuple[dict[str, object] | None, str | None, list[np.ndarray]]:
    if not components:
        return None, "disappear", []

    max_area = max(int(component["area"]) for component in components)
    min_tracking_area = max(4, int(round(max_area * 0.2)))
    tracking_indices = [idx for idx, component in enumerate(components) if int(component["area"]) >= min_tracking_area]
    if not tracking_indices:
        tracking_indices = list(range(len(components)))

    distances = [
        float(np.linalg.norm(np.asarray(component["centroid"], dtype=np.float32) - target_centroid))
        for component in components
    ]
    order = sorted(tracking_indices, key=lambda idx: (distances[idx], -int(components[idx]["area"])))
    best_index = order[0]
    best_component = components[best_index]
    best_distance = distances[best_index]
    max_jump = max(10.0, step_pixels * 1.35)

    if best_distance > max_jump:
        return None, "disappear", []

    plausible_indices = [
        idx
        for idx in order
        if distances[idx] <= max_jump and int(components[idx]["area"]) >= max(4, int(best_component["area"]) // 3)
    ]
    if len(plausible_indices) >= 2:
        second_index = plausible_indices[1]
        second_distance = distances[second_index]
        best_area = int(best_component["area"])
        second_area = int(components[second_index]["area"])
        similar_distance = (second_distance - best_distance) <= max(3.0, step_pixels * 0.18)
        similar_area = second_area >= max(4, int(round(best_area * 0.7)))
        inter_blob_distance = float(
            np.linalg.norm(
                np.asarray(components[second_index]["centroid"], dtype=np.float32)
                - np.asarray(best_component["centroid"], dtype=np.float32)
            )
        )
        far_apart_branch = inter_blob_distance > (3.0 * step_pixels)
        if similar_distance and similar_area and far_apart_branch:
            branch_centroids = [np.asarray(components[idx]["centroid"], dtype=np.float32) for idx in plausible_indices[:3]]
            return None, "branch", branch_centroids

    return best_component, None, []


def choose_relaxed_component(
    components: list[dict[str, object]],
    target_centroid: np.ndarray,
    step_pixels: float,
) -> dict[str, object] | None:
    if not components:
        return None

    max_area = max(int(component["area"]) for component in components)
    min_tracking_area = max(4, int(round(max_area * 0.2)))
    candidates = [component for component in components if int(component["area"]) >= min_tracking_area]
    if not candidates:
        candidates = components

    max_jump = max(10.0, step_pixels * 1.35)
    distances = [
        float(np.linalg.norm(np.asarray(component["centroid"], dtype=np.float32) - target_centroid))
        for component in candidates
    ]
    close_indices = [idx for idx, distance in enumerate(distances) if distance <= max_jump]
    search_space = [candidates[idx] for idx in close_indices] if close_indices else candidates
    search_distances = [distances[idx] for idx in close_indices] if close_indices else [
        float(np.linalg.norm(np.asarray(component["centroid"], dtype=np.float32) - target_centroid))
        for component in candidates
    ]
    best_index = min(
        range(len(search_space)),
        key=lambda idx: (search_distances[idx], -int(search_space[idx]["area"])),
    )
    return search_space[best_index]


def validate_route(
    route: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    edge_set: set[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[bool, str]:
    if not route:
        return False, "empty_route"
    if route[0] != start:
        return False, "bad_start"
    if route[-1] != goal:
        return False, "bad_goal"
    for current, nxt in zip(route, route[1:]):
        if (current, nxt) not in edge_set:
            return False, "wall_cross"
    return True, "ok"


def lcs_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for item_a in a:
        curr = [0]
        for idx_b, item_b in enumerate(b, start=1):
            if item_a == item_b:
                curr.append(prev[idx_b - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[idx_b]))
        prev = curr
    return prev[-1]


def solution_path(struct: dict[str, object]) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.asarray(struct["solution_path"]).tolist()]


def score_loose_route(pred_route: list[tuple[int, int]], gt_route: list[tuple[int, int]]) -> tuple[float, int]:
    overlap = lcs_length(pred_route, gt_route)
    denominator = max(1, len(gt_route))
    return overlap / denominator, overlap


def evaluate_video(pred_video: Path, struct_path: Path) -> dict[str, object]:
    struct = load_struct(struct_path)
    edge_set = build_edge_set(struct["edges"])
    open_mask = scaled_open_mask(struct["wall_grid"], int(struct["frame_size"]))
    cell_centers = struct["cell_centers"]
    start = tuple(int(x) for x in struct["start"].tolist())
    goal = tuple(int(x) for x in struct["goal"].tolist())
    gt_route = solution_path(struct)
    step_pixels = nominal_step_pixels(cell_centers)
    start_centroid = np.asarray(cell_centers[start[0], start[1]], dtype=np.float32)
    strict_prev_centroid = start_centroid.copy()
    loose_prev_centroid = start_centroid.copy()

    reader = imageio.get_reader(pred_video)
    strict_centroids: list[np.ndarray] = []
    strict_route: list[tuple[int, int]] = []
    loose_centroids: list[np.ndarray] = []
    loose_route_raw: list[tuple[int, int]] = []
    failure_reason = ""
    goal_reached_frame: int | None = None
    frame_count = 0
    try:
        for frame in reader:
            frame = align_prediction_frame(frame, int(struct["frame_size"]))
            frame_count += 1
            mask = red_mask(frame)
            components = connected_components(mask)
            relaxed_component = choose_relaxed_component(
                components,
                loose_prev_centroid if loose_centroids else start_centroid,
                step_pixels,
            )
            if relaxed_component is not None:
                loose_centroid = np.asarray(relaxed_component["centroid"], dtype=np.float32)
                loose_centroids.append(loose_centroid)
                loose_prev_centroid = loose_centroid
                loose_route_raw.append(nearest_cell(loose_centroid, cell_centers))

            if goal_reached_frame is not None or failure_reason:
                continue

            chosen_component, detect_failure, _branch_centroids = choose_tracked_component(
                components,
                strict_prev_centroid if strict_centroids else start_centroid,
                step_pixels,
            )
            if detect_failure is not None:
                failure_reason = detect_failure
                continue
            assert chosen_component is not None
            centroid = chosen_component["centroid"]
            row = int(round(float(centroid[0])))
            col = int(round(float(centroid[1])))
            row = max(0, min(open_mask.shape[0] - 1, row))
            col = max(0, min(open_mask.shape[1] - 1, col))
            if not bool(open_mask[row, col]):
                failure_reason = "wall_cross"
                continue
            centroid_arr = np.asarray([float(centroid[0]), float(centroid[1])], dtype=np.float32)
            strict_centroids.append(centroid_arr)
            strict_prev_centroid = centroid_arr
            cell = nearest_cell(centroid_arr, cell_centers)
            if not strict_route:
                strict_route.append(cell)
                if strict_route[0] != start:
                    failure_reason = "bad_start"
                    continue
            elif strict_route[-1] != cell:
                if (strict_route[-1], cell) not in edge_set:
                    failure_reason = "wall_cross"
                    continue
                strict_route.append(cell)
            if strict_route[-1] == goal:
                goal_reached_frame = frame_count - 1
    finally:
        reader.close()

    loose_route = compress_route(loose_route_raw)
    loose_score, loose_overlap_count = score_loose_route(loose_route, gt_route)

    if goal_reached_frame is not None:
        success = 1
        failure_reason = "ok"
    elif not failure_reason:
        strict_route = formalize_route(strict_centroids, cell_centers) if strict_centroids else []
        ok, failure_reason = validate_route(strict_route, start, goal, edge_set)
        success = int(ok)
    else:
        success = 0

    return {
        "file": pred_video.name,
        "maze_size": int(struct["maze_size"]),
        "num_frames": frame_count,
        "strict_success": success,
        "strict_failure_reason": failure_reason,
        "success": success,
        "failure_reason": failure_reason,
        "formalized_route": [list(cell) for cell in strict_route],
        "loose_score": loose_score,
        "loose_overlap_count": loose_overlap_count,
        "loose_gt_length": len(gt_route),
        "loose_formalized_route": [list(cell) for cell in loose_route],
        "goal_reached_frame": goal_reached_frame,
    }


def find_prediction_video(pred_dir: Path, index: int) -> Path | None:
    canonical = pred_dir / f"maze_{index:04d}.mp4"
    if canonical.exists():
        return canonical
    prefixed = sorted(pred_dir.glob(f"{index:05d}_*.mp4"))
    if prefixed:
        return prefixed[0]
    bare = pred_dir / f"{index:05d}.mp4"
    if bare.exists():
        return bare
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate maze videos by tracking the red ball.")
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory containing 50 predicted videos.")
    parser.add_argument("--struct-dir", type=str, default="maze_50_test_npy", help="Directory with ground-truth npy files.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="maze_eval_summary.json",
        help="Summary JSON path.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        default="maze_eval_formalized.npy",
        help="Per-sample formalization output path.",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    struct_dir = Path(args.struct_dir)

    struct_paths = sorted(path for path in struct_dir.glob("maze_*.npy"))
    if len(struct_paths) != 50:
        raise RuntimeError(f"Expected 50 struct npy files in {struct_dir}, found {len(struct_paths)}")

    results: list[dict[str, object]] = []
    for struct_path in struct_paths:
        index = int(struct_path.stem.split("_")[-1])
        pred_video = find_prediction_video(pred_dir, index)
        if pred_video is None:
            struct = load_struct(struct_path)
            results.append(
                {
                    "file": f"maze_{index:04d}.mp4",
                    "maze_size": int(struct["maze_size"]),
                    "num_frames": 0,
                    "strict_success": 0,
                    "strict_failure_reason": "missing_video",
                    "success": 0,
                    "failure_reason": "missing_video",
                    "formalized_route": [],
                    "loose_score": 0.0,
                    "loose_overlap_count": 0,
                    "loose_gt_length": len(solution_path(struct)),
                    "loose_formalized_route": [],
                }
            )
            continue
        results.append(evaluate_video(pred_video, struct_path))

    per_size: dict[int, dict[str, object]] = {}
    for maze_size in sorted({int(result["maze_size"]) for result in results}):
        size_results = [result for result in results if int(result["maze_size"]) == maze_size]
        strict_success_count = sum(int(result["strict_success"]) for result in size_results)
        loose_average_score = sum(float(result["loose_score"]) for result in size_results) / len(size_results) if size_results else 0.0
        per_size[maze_size] = {
            "count": len(size_results),
            "strict_success_count": strict_success_count,
            "strict_success_rate": strict_success_count / len(size_results) if size_results else 0.0,
            "strict_failure_breakdown": dict(
                Counter(result["strict_failure_reason"] for result in size_results if not int(result["strict_success"]))
            ),
            "loose_average_score": loose_average_score,
        }

    summary = {
        "pred_dir": str(pred_dir),
        "struct_dir": str(struct_dir),
        "strict": {
            "overall_success_rate": sum(int(result["strict_success"]) for result in results) / len(results),
            "per_size": per_size,
        },
        "loose": {
            "overall_average_score": sum(float(result["loose_score"]) for result in results) / len(results),
            "per_size": {maze_size: {"count": values["count"], "average_score": values["loose_average_score"]} for maze_size, values in per_size.items()},
        },
        "overall_success_rate": sum(int(result["strict_success"]) for result in results) / len(results),
        "per_size": per_size,
        "results": results,
    }

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(args.output_npy, {"results": results, "per_size": per_size, "strict": summary["strict"], "loose": summary["loose"]}, allow_pickle=True)

    print(
        json.dumps(
            {
                "strict": {
                    "overall_success_rate": summary["strict"]["overall_success_rate"],
                    "per_size": {maze_size: {"success_rate": values["strict_success_rate"]} for maze_size, values in per_size.items()},
                },
                "loose": {
                    "overall_average_score": summary["loose"]["overall_average_score"],
                    "per_size": {maze_size: {"average_score": values["loose_average_score"]} for maze_size, values in per_size.items()},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
