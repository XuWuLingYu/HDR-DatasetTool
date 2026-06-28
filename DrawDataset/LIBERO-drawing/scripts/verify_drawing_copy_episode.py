#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a collected drawing-copy episode directory.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--min-visual-diff", type=float, default=1000.0)
    parser.add_argument("--max-ee-step", type=float, default=0.30)
    parser.add_argument("--max-draw-ee-z-range", type=float, default=0.01)
    parser.add_argument("--max-brush-tilt-deg", type=float, default=35.0)
    parser.add_argument("--max-pre-attach-brush-move", type=float, default=0.012)
    parser.add_argument("--min-post-attach-brush-move", type=float, default=0.01)
    parser.add_argument("--max-attach-site-distance", type=float, default=0.05)
    parser.add_argument("--min-grasp-contact-frames", type=int, default=1)
    parser.add_argument("--max-draw-tip-z-mean", type=float, default=0.002)
    parser.add_argument("--min-draw-tip-z-mean", type=float, default=-0.016)
    parser.add_argument("--max-draw-tip-z-rel", type=float, default=0.015)
    parser.add_argument("--min-draw-tip-contact-fraction", type=float, default=0.95)
    parser.add_argument("--max-draw-tip-xy-error", type=float, default=0.06)
    parser.add_argument("--max-copy-uv-mean", type=float, default=0.03)
    parser.add_argument("--max-copy-uv-p95", type=float, default=0.08)
    parser.add_argument("--min-board-ink-pixels", type=int, default=20)
    parser.add_argument("--min-target-dark-pixels", type=int, default=80)
    parser.add_argument("--min-target-reference-dark-ratio", type=float, default=0.0)
    parser.add_argument("--min-bucket-blue-pixels", type=int, default=40)
    parser.add_argument("--min-bucket-visible-fraction", type=float, default=0.95)
    parser.add_argument("--max-frames", type=int, default=0, help="Reject episodes longer than this many frames when >0.")
    parser.add_argument("--max-draw-frames", type=int, default=0, help="Reject episodes with more draw frames than this when >0.")
    parser.add_argument("--min-draw-frames", type=int, default=8)
    parser.add_argument("--allow-hdf5-only", action="store_true", help="Verify directly from HDF5 frames when debug MP4/JPGs were skipped.")
    args = parser.parse_args()

    root = args.episode_dir
    mp4 = root / "episode.mp4"
    hdf5 = root / "episode.hdf5"
    preview = root / "preview.jpg"
    full_sheet = root / "contact_sheet_full.jpg"
    frames_dir = root / "frames"
    summary = root / "summary.json"

    require(hdf5.exists(), f"missing {hdf5}")
    if not args.allow_hdf5_only:
        require(mp4.exists(), f"missing {mp4}")
        require(preview.exists(), f"missing {preview}")
        require(full_sheet.exists(), f"missing {full_sheet}")
        require(frames_dir.is_dir(), f"missing {frames_dir}")
    require(summary.exists(), f"missing {summary}")
    summary_data = json.loads(summary.read_text(encoding="utf-8"))

    frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.is_dir() else []
    mp4_frames = imageio.get_reader(mp4).count_frames() if mp4.exists() else None
    with h5py.File(hdf5, "r") as f:
        h5_frames = int(f["frames"].shape[0])
        h5_first = f["frames"][0].astype(np.uint8)
        h5_last = f["frames"][h5_frames - 1].astype(np.uint8)
        h5_sample_indices = np.linspace(0, h5_frames - 1, min(32, h5_frames)).round().astype(int)
        h5_sample_frames = f["frames"][h5_sample_indices].astype(np.uint8)
        actions = f["actions"][()]
        phases = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in f["phases"][()]]
        require(len(f["frames"].shape) == 4 and f["frames"].shape[-1] == 3, "frames dataset must be [T,H,W,3]")
        require(actions.shape == (h5_frames, 7), f"actions shape mismatch: {actions.shape}")
        require(f["ee_targets"].shape == (h5_frames, 3), f"ee_targets shape mismatch: {f['ee_targets'].shape}")
        ee_targets = f["ee_targets"][()]
        tip_targets = f["brush_tip_targets"][()] if "brush_tip_targets" in f else None
        brush_tilt = f["brush_tilt_degrees"][()] if "brush_tilt_degrees" in f else None
        sim_states = f["sim_states"][()] if "sim_states" in f else None
        weld_active = f["weld_active"][()] if "weld_active" in f else None
        require("reference_curve" in f and f["reference_curve"].shape[1] == 2, "missing reference_curve [N,2]")
        reference_curve = f["reference_curve"][()]
        reference_stroke_lengths = f["reference_stroke_lengths"][()] if "reference_stroke_lengths" in f else None
        require("drawing_waypoints" in f and f["drawing_waypoints"].shape[1] == 3, "missing drawing_waypoints [M,3]")
        bddl = Path(f.attrs["bddl"]) if "bddl" in f.attrs else None
        use_weld = bool(f.attrs.get("use_weld", summary_data.get("use_weld", False)))
        scripted_ink_mode = bool(f.attrs.get("scripted_ink_mode", summary_data.get("scripted_ink_mode", False)))
        seed = int(f.attrs.get("seed", summary_data.get("seed", 0)))
        draw_segment_count = int(f.attrs.get("draw_segment_count", summary_data.get("draw_segment_count", 0)))
        draw_canvas_dark_pixels = int(f.attrs.get("draw_canvas_dark_pixels", summary_data.get("draw_canvas_dark_pixels", 0)))

    require(h5_frames >= args.min_frames, f"too few frames: {h5_frames}")
    if args.max_frames > 0:
        require(h5_frames <= args.max_frames, f"too many frames: {h5_frames}")
    if mp4_frames is not None:
        require(mp4_frames == h5_frames, f"mp4 frame count {mp4_frames} != hdf5 frames {h5_frames}")
    if not args.allow_hdf5_only:
        require(len(frame_files) == h5_frames, f"frame jpg count {len(frame_files)} != hdf5 frames {h5_frames}")
    require("draw" in phases, "no draw phase recorded")
    require("touch_down" in phases and "lift_after_draw" in phases, "missing touch/lift phases")
    require(phases.count("draw") >= args.min_draw_frames, f"too few draw frames: {phases.count('draw')}")
    if args.max_draw_frames > 0:
        require(phases.count("draw") <= args.max_draw_frames, f"too many draw frames: {phases.count('draw')}")

    finite_ee = ee_targets[np.isfinite(ee_targets).all(axis=1)]
    if len(finite_ee) > 1:
        max_step = float(np.linalg.norm(np.diff(finite_ee, axis=0), axis=1).max())
        require(max_step <= args.max_ee_step, f"ee target jump too large: {max_step}")
    else:
        max_step = 0.0
    if tip_targets is not None:
        finite_tip = tip_targets[np.isfinite(tip_targets).all(axis=1)]
        require(len(finite_tip) > 2, "too few finite brush tip targets")
    draw_z_range = None
    draw_ee = ee_targets[np.asarray(phases) == "draw"]
    draw_ee = draw_ee[np.isfinite(draw_ee).all(axis=1)]
    if len(draw_ee) > 1:
        draw_z_range = float(np.max(draw_ee[:, 2]) - np.min(draw_ee[:, 2]))
        require(draw_z_range <= args.max_draw_ee_z_range, f"draw ee z range too large: {draw_z_range}")
    max_brush_tilt = None
    if brush_tilt is not None:
        finite_tilt = brush_tilt[np.isfinite(brush_tilt)]
        require(len(finite_tilt) > 0, "no finite brush tilt measurements")
        max_brush_tilt = float(np.max(finite_tilt))
        require(max_brush_tilt <= args.max_brush_tilt_deg, f"brush tilt too large: {max_brush_tilt}")
    layout = verify_bddl_layout(bddl) if bddl is not None and bddl.exists() else {}
    physics_metrics = None
    if use_weld:
        require(not scripted_ink_mode, "use_weld episode must not use scripted ink mode")
        if sim_states is not None and bddl is not None and bddl.exists():
            physics_metrics = verify_physical_brush_motion(
                bddl,
                sim_states,
                phases,
                tip_targets,
                weld_active,
                max_pre_attach_move=args.max_pre_attach_brush_move,
                min_post_attach_move=args.min_post_attach_brush_move,
                max_attach_site_distance=args.max_attach_site_distance,
                min_grasp_contact_frames=args.min_grasp_contact_frames,
                min_draw_tip_z_mean=args.min_draw_tip_z_mean,
                max_draw_tip_z_mean=args.max_draw_tip_z_mean,
                max_draw_tip_z_rel=args.max_draw_tip_z_rel,
                min_draw_tip_contact_fraction=args.min_draw_tip_contact_fraction,
                max_draw_tip_xy_error=args.max_draw_tip_xy_error,
                reference_curve=reference_curve,
                reference_stroke_lengths=reference_stroke_lengths,
                max_copy_uv_mean=args.max_copy_uv_mean,
                max_copy_uv_p95=args.max_copy_uv_p95,
                seed=seed,
            )

    if frame_files:
        first = cv2.imread(str(frame_files[0]))
        last = cv2.imread(str(frame_files[-1]))
        require(first is not None and last is not None, "could not read extracted frames")
    else:
        first = cv2.cvtColor(h5_first, cv2.COLOR_RGB2BGR)
        last = cv2.cvtColor(h5_last, cv2.COLOR_RGB2BGR)
    visual_diff = float(np.abs(last.astype(np.int16) - first.astype(np.int16)).sum())
    require(visual_diff >= args.min_visual_diff, f"visual diff too small: {visual_diff}")
    ink_pixels = estimate_added_ink_pixels(first, last)
    target_dark_pixels = estimate_target_board_dark_pixels(last)
    reference_dark_pixels = estimate_reference_board_dark_pixels(last)
    target_reference_dark_ratio = float(target_dark_pixels / max(1, reference_dark_pixels))
    bucket_visibility = (
        estimate_bucket_visibility(frame_files, args.min_bucket_blue_pixels)
        if frame_files
        else estimate_bucket_visibility_from_frames(h5_sample_frames, args.min_bucket_blue_pixels)
    )
    require(
        bucket_visibility["visible_fraction"] >= args.min_bucket_visible_fraction,
        f"bucket disappeared in video: {bucket_visibility}",
    )
    if use_weld:
        require(draw_segment_count >= 20, f"too few internal draw ink segments: {draw_segment_count}")
        require(draw_canvas_dark_pixels >= 500, f"too few internal draw canvas dark pixels: {draw_canvas_dark_pixels}")
        require(ink_pixels >= args.min_board_ink_pixels, f"too little physical ink on target board: {ink_pixels}")
        require(
            target_dark_pixels >= args.min_target_dark_pixels,
            f"too little visible ink on target board: {target_dark_pixels}",
        )
        require(
            target_reference_dark_ratio >= args.min_target_reference_dark_ratio,
            f"target/reference visible ink ratio too low: {target_reference_dark_ratio}",
        )

    result = {
        "ok": True,
        "episode_dir": str(root.resolve()),
        "mp4_frames": mp4_frames if mp4_frames is not None else h5_frames,
        "hdf5_frames": h5_frames,
        "frame_jpgs": len(frame_files),
        "actions_shape": list(actions.shape),
        "visual_diff_first_last": visual_diff,
        "estimated_added_ink_pixels": ink_pixels,
        "target_board_dark_pixels": target_dark_pixels,
        "reference_board_dark_pixels": reference_dark_pixels,
        "target_reference_dark_ratio": target_reference_dark_ratio,
        "bucket_visibility": bucket_visibility,
        "draw_segment_count": draw_segment_count,
        "draw_canvas_dark_pixels": draw_canvas_dark_pixels,
        "max_ee_target_step": max_step,
        "draw_ee_z_range": draw_z_range,
        "max_brush_tilt_degrees": max_brush_tilt,
        "use_weld": use_weld,
        "scripted_ink_mode": scripted_ink_mode,
        "physics_metrics": physics_metrics,
        "layout": layout,
        "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VERIFY_FAILED: {message}")


def verify_bddl_layout(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    regions = {}
    for name in ["reference_board_region", "blank_board_region", "brush_region", "bucket_region"]:
        m = re.search(rf"\({name}\s+.*?\(:ranges\s+\(\s+\(([-0-9. ]+)\)", text, re.S)
        require(m is not None, f"missing BDDL region {name}")
        vals = [float(x) for x in m.group(1).split()]
        require(len(vals) == 4, f"invalid range for {name}")
        regions[name] = vals
    ref = regions["reference_board_region"]
    blank = regions["blank_board_region"]
    brush = regions["brush_region"]
    bucket = regions["bucket_region"]
    ref_y = 0.5 * (ref[1] + ref[3])
    blank_y = 0.5 * (blank[1] + blank[3])
    brush_y = 0.5 * (brush[1] + brush[3])
    brush_x = 0.5 * (brush[0] + brush[2])
    bucket_x = 0.5 * (bucket[0] + bucket[2])
    bucket_y = 0.5 * (bucket[1] + bucket[3])
    require(abs(ref_y - blank_y) > 0.20, "boards are not separated along left-right y axis")
    require(ref_y < blank_y, "reference board must be on the left and blank/copy board must be on the right in agentview")
    require(blank[0] - 0.13 <= brush_x <= blank[2] - 0.05, "brush x region should be shifted toward the inner side of agentview")
    require(brush_y > blank_y, "brush region should be on the right side of the blank board")
    require(bucket_y > blank_y, "bucket should be on the right side of the blank board")
    require(bucket_y > blank_y + 0.20, "bucket should be visibly to the right of the blank board")
    return {
        "reference_center_y": ref_y,
        "blank_center_y": blank_y,
        "brush_center_x": brush_x,
        "brush_center_y": brush_y,
        "bucket_center_x": bucket_x,
        "bucket_center_y": bucket_y,
    }


def verify_physical_brush_motion(
    bddl: Path,
    sim_states: np.ndarray,
    phases: list[str],
    tip_targets: np.ndarray | None,
    weld_active: np.ndarray | None,
    max_pre_attach_move: float,
    min_post_attach_move: float,
    max_attach_site_distance: float,
    min_grasp_contact_frames: int,
    min_draw_tip_z_mean: float,
    max_draw_tip_z_mean: float,
    max_draw_tip_z_rel: float,
    min_draw_tip_contact_fraction: float,
    max_draw_tip_xy_error: float,
    reference_curve: np.ndarray | None,
    reference_stroke_lengths: np.ndarray | None,
    max_copy_uv_mean: float,
    max_copy_uv_p95: float,
    seed: int,
) -> dict:
    prepare_local_libero_config(ROOT)
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=32,
        camera_widths=32,
        camera_names=["agentview"],
        control_freq=20,
        horizon=5000,
    )
    env.seed(seed)
    env.reset()
    brush_positions = []
    brush_tips = []
    board_positions = []
    grasp_site_distances = []
    gripper_brush_contacts = []
    for state in sim_states:
        env.set_state(state)
        env.env.sim.forward()
        brush_positions.append(env.env.sim.data.body_xpos[env.env.obj_body_id["soft_brush_1"]].copy())
        brush_tips.append(env.env.brush_tip_world_pos().copy())
        board_positions.append(env.env.sim.data.body_xpos[env.env.obj_body_id["drawing_board_blank_1"]].copy())
        grasp_site_distances.append(grasp_site_distance(env))
        gripper_brush_contacts.append(has_gripper_brush_contact(env))
    env.close()

    brush_positions = np.asarray(brush_positions, dtype=np.float64)
    brush_tips = np.asarray(brush_tips, dtype=np.float64)
    board_positions = np.asarray(board_positions, dtype=np.float64)
    grasp_site_distances = np.asarray(grasp_site_distances, dtype=np.float64)
    gripper_brush_contacts = np.asarray(gripper_brush_contacts, dtype=bool)
    phases_arr = np.asarray(phases)
    settle_indices = np.where(phases_arr == "settle")[0]
    start_idx = int(settle_indices[-1]) if len(settle_indices) else 0
    start = brush_positions[start_idx]
    move = np.linalg.norm(brush_positions - start[None, :], axis=1)
    if weld_active is not None:
        attach_candidates = np.where(np.asarray(weld_active, dtype=bool))[0]
    else:
        attach_candidates = np.where(np.isin(phases_arr, ["hold_closed", "lift_brush", "approach_draw_board", "touch_down", "draw", "lift_after_draw"]))[0]
    require(len(attach_candidates) > 0, "no post-attach phases found for physical motion check")
    attach_idx = int(attach_candidates[0])
    pre_slice = move[start_idx:attach_idx]
    pre_attach_max = float(np.max(pre_slice)) if len(pre_slice) else 0.0
    post_attach_max = float(np.max(move[attach_idx:]))
    contact_window_start = max(start_idx, attach_idx - 3)
    contact_window_stop = min(len(gripper_brush_contacts), attach_idx + 2)
    contact_count = int(np.count_nonzero(gripper_brush_contacts[contact_window_start:contact_window_stop]))
    attach_site_distance = float(grasp_site_distances[attach_idx])
    require(pre_attach_max <= max_pre_attach_move, f"brush moved before physical attach: {pre_attach_max}")
    require(post_attach_max >= min_post_attach_move, f"brush did not move after physical attach: {post_attach_max}")
    require(attach_site_distance <= max_attach_site_distance, f"grasp sites too far at attach: {attach_site_distance}")
    metrics = {
        "attach_index_estimate": attach_idx,
        "attach_phase_estimate": phases[attach_idx],
        "settled_start_index": start_idx,
        "pre_attach_max_brush_move_m": pre_attach_max,
        "post_attach_max_brush_move_m": post_attach_max,
        "attach_site_distance_m": attach_site_distance,
        "gripper_brush_contact_frames_near_attach": contact_count,
        "gripper_brush_contact_any": bool(np.any(gripper_brush_contacts)),
    }
    if tip_targets is not None:
        draw_mask = (phases_arr == "draw") & np.isfinite(tip_targets).all(axis=1)
        if np.any(draw_mask):
            xy_err = brush_tips[draw_mask, :2] - tip_targets[draw_mask, :2]
            xy_max = float(np.linalg.norm(xy_err, axis=1).max())
            z_rel = brush_tips[draw_mask, 2] - (board_positions[draw_mask, 2] + 0.018)
            z_mean = float(z_rel.mean())
            contact_fraction = float(np.count_nonzero((z_rel >= min_draw_tip_z_mean) & (z_rel <= max_draw_tip_z_rel)) / len(z_rel))
            require(
                min_draw_tip_z_mean <= z_mean <= max_draw_tip_z_mean,
                f"draw tip z mean out of contact range: {z_mean}",
            )
            z_rel_p99 = float(np.percentile(z_rel, 99))
            require(z_rel_p99 <= max_draw_tip_z_rel, f"draw tip p99 rises too far above board: {z_rel_p99}")
            require(
                contact_fraction >= min_draw_tip_contact_fraction,
                f"too few draw frames keep tip on board: {contact_fraction}",
            )
            require(xy_max <= max_draw_tip_xy_error, f"draw tip xy error too large: {xy_max}")
            metrics.update(
                {
                    "draw_tip_xy_error_mean_m": [float(x) for x in xy_err.mean(axis=0)],
                    "draw_tip_xy_error_rms_m": [float(x) for x in np.sqrt((xy_err**2).mean(axis=0))],
                    "draw_tip_xy_error_max_norm_m": xy_max,
                    "draw_tip_z_rel_min_m": float(z_rel.min()),
                    "draw_tip_z_rel_max_m": float(z_rel.max()),
                    "draw_tip_z_rel_p99_m": z_rel_p99,
                    "draw_tip_z_rel_mean_m": z_mean,
                    "draw_tip_contact_fraction": contact_fraction,
                }
            )
            if reference_curve is not None and len(reference_curve) > 1:
                actual_uv = np.column_stack(
                    (
                        (brush_tips[draw_mask, 0] - board_positions[draw_mask, 0]) / 0.38 + 0.5,
                        (brush_tips[draw_mask, 1] - board_positions[draw_mask, 1]) / 0.28 + 0.5,
                    )
                )
                reference_strokes = split_reference_strokes(reference_curve, reference_stroke_lengths)
                copy_metrics = curve_copy_uv_metrics(reference_strokes, actual_uv)
                require(
                    copy_metrics["actual_to_reference_mean"] <= max_copy_uv_mean,
                    f"copied curve too far from reference mean: {copy_metrics['actual_to_reference_mean']}",
                )
                require(
                    copy_metrics["reference_to_actual_mean"] <= max_copy_uv_mean,
                    f"reference curve not covered by copy mean: {copy_metrics['reference_to_actual_mean']}",
                )
                require(
                    copy_metrics["actual_to_reference_p95"] <= max_copy_uv_p95,
                    f"copied curve too far from reference p95: {copy_metrics['actual_to_reference_p95']}",
                )
                require(
                    copy_metrics["reference_to_actual_p95"] <= max_copy_uv_p95,
                    f"reference curve not covered by copy p95: {copy_metrics['reference_to_actual_p95']}",
                )
                metrics["copy_uv_metrics"] = copy_metrics
    return metrics


def split_reference_strokes(reference_curve: np.ndarray, lengths: np.ndarray | None) -> list[np.ndarray]:
    reference_curve = np.asarray(reference_curve, dtype=np.float64)
    if lengths is None:
        return [reference_curve]
    out = []
    start = 0
    for length in np.asarray(lengths, dtype=np.int64):
        stop = start + int(length)
        if stop > start:
            out.append(reference_curve[start:stop])
        start = stop
    return out or [reference_curve]


def curve_copy_uv_metrics(reference_strokes: list[np.ndarray], actual_uv: np.ndarray) -> dict:
    def nearest_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return distances.min(axis=1)

    dense_strokes = [densify_uv_stroke(stroke, max_step=0.008) for stroke in reference_strokes if len(stroke)]
    reference_uv = np.concatenate(dense_strokes, axis=0)
    actual_to_reference = nearest_distance(actual_uv, reference_uv)
    per_stroke_reference_to_actual = [
        nearest_distance(np.asarray(stroke, dtype=np.float64), actual_uv)
        for stroke in dense_strokes
        if len(stroke)
    ]
    reference_to_actual = np.concatenate(per_stroke_reference_to_actual, axis=0)
    return {
        "actual_to_reference_mean": float(actual_to_reference.mean()),
        "actual_to_reference_p95": float(np.percentile(actual_to_reference, 95)),
        "reference_to_actual_mean": float(reference_to_actual.mean()),
        "reference_to_actual_p95": float(np.percentile(reference_to_actual, 95)),
        "actual_uv_min": [float(x) for x in actual_uv.min(axis=0)],
        "actual_uv_max": [float(x) for x in actual_uv.max(axis=0)],
        "reference_uv_min": [float(x) for x in reference_uv.min(axis=0)],
        "reference_uv_max": [float(x) for x in reference_uv.max(axis=0)],
    }


def densify_uv_stroke(stroke: np.ndarray, max_step: float = 0.008) -> np.ndarray:
    pts = np.asarray(stroke, dtype=np.float64)
    if len(pts) <= 1:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        dist = float(np.linalg.norm(b - a))
        steps = max(1, int(np.ceil(dist / max_step)))
        for i in range(1, steps + 1):
            alpha = i / steps
            out.append((1.0 - alpha) * a + alpha * b)
    return np.asarray(out, dtype=np.float64)


def grasp_site_distance(env) -> float:
    gripper_site = get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
    brush_site = get_site_xpos(env, "soft_brush_1_grasp_site")
    if gripper_site is None or brush_site is None:
        return float("inf")
    return float(np.linalg.norm(gripper_site - brush_site))


def get_site_xpos(env, site_name: str) -> np.ndarray | None:
    try:
        site_id = env.sim.model.site_name2id(site_name)
    except Exception:
        return None
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def has_gripper_brush_contact(env) -> bool:
    sim = env.sim
    model = sim.model
    data = sim.data
    for idx in range(int(data.ncon)):
        contact = data.contact[idx]
        name1 = model.geom_id2name(contact.geom1) or ""
        name2 = model.geom_id2name(contact.geom2) or ""
        pair = name1 + " " + name2
        has_gripper = "gripper0_finger" in pair
        has_brush = "soft_brush_1_handle" in pair or "soft_brush_1_grasp_sleeve" in pair
        if has_gripper and has_brush and float(contact.dist) <= 0.002:
            return True
    return False


def estimate_added_ink_pixels(first_bgr: np.ndarray, last_bgr: np.ndarray) -> int:
    first = first_bgr.astype(np.int16)
    last = last_bgr.astype(np.int16)
    h, w = first.shape[:2]
    gray_first = first.mean(axis=2)
    gray_last = last.mean(axis=2)
    darkened = gray_first - gray_last
    margin_y = max(1, int(0.04 * h))
    margin_x = max(1, int(0.04 * w))
    roi_darkened = darkened[margin_y : h - margin_y, margin_x : w - margin_x]
    roi_last_gray = gray_last[margin_y : h - margin_y, margin_x : w - margin_x]
    mask = (roi_darkened > 18) & (roi_last_gray < 190)
    return int(np.count_nonzero(mask))


def estimate_target_board_dark_pixels(last_bgr: np.ndarray) -> int:
    h, w = last_bgr.shape[:2]
    gray = last_bgr.mean(axis=2)
    # Agentview is fixed: the copy board occupies the central-right table region.
    x0, x1 = int(0.45 * w), int(0.88 * w)
    y0, y1 = int(0.42 * h), int(0.88 * h)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0
    mask = roi < 95
    return int(np.count_nonzero(mask))


def estimate_reference_board_dark_pixels(last_bgr: np.ndarray) -> int:
    h, w = last_bgr.shape[:2]
    gray = last_bgr.mean(axis=2)
    x0, x1 = int(0.02 * w), int(0.45 * w)
    y0, y1 = int(0.42 * h), int(0.88 * h)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0
    return int(np.count_nonzero(roi < 95))


def estimate_bucket_visibility(frame_files: list[Path], min_blue_pixels: int) -> dict:
    if not frame_files:
        return {"visible_fraction": 0.0, "min_blue_pixels": 0, "sampled_frames": 0}
    sample_count = min(32, len(frame_files))
    indices = np.linspace(0, len(frame_files) - 1, sample_count).round().astype(int)
    counts = []
    for idx in indices:
        frame = cv2.imread(str(frame_files[int(idx)]))
        if frame is None:
            counts.append(0)
            continue
        h, w = frame.shape[:2]
        roi = frame[int(0.50 * h) : int(0.95 * h), int(0.58 * w) : int(0.99 * w)]
        if roi.size == 0:
            counts.append(0)
            continue
        b = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        r = roi[:, :, 2].astype(np.int16)
        blue = (b > 90) & (b > g + 25) & (b > r + 35)
        counts.append(int(np.count_nonzero(blue)))
    visible = [count >= min_blue_pixels for count in counts]
    return {
        "visible_fraction": float(np.count_nonzero(visible) / max(1, len(visible))),
        "min_blue_pixels": int(min(counts) if counts else 0),
        "max_blue_pixels": int(max(counts) if counts else 0),
        "sampled_frames": int(len(counts)),
    }


def estimate_bucket_visibility_from_frames(frames_rgb: np.ndarray, min_blue_pixels: int) -> dict:
    counts = []
    for frame_rgb in np.asarray(frames_rgb):
        frame = cv2.cvtColor(frame_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]
        roi = frame[int(0.50 * h) : int(0.95 * h), int(0.58 * w) : int(0.99 * w)]
        if roi.size == 0:
            counts.append(0)
            continue
        b = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        r = roi[:, :, 2].astype(np.int16)
        blue = (b > 90) & (b > g + 25) & (b > r + 35)
        counts.append(int(np.count_nonzero(blue)))
    visible = [count >= min_blue_pixels for count in counts]
    return {
        "visible_fraction": float(np.count_nonzero(visible) / max(1, len(visible))),
        "min_blue_pixels": int(min(counts) if counts else 0),
        "max_blue_pixels": int(max(counts) if counts else 0),
        "sampled_frames": int(len(counts)),
    }


def prepare_local_libero_config(libero_root: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    config_dir = libero_root / ".libero_local"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = libero_root / "empty_datasets"
    datasets_dir.mkdir(exist_ok=True)
    import yaml

    config = {
        "benchmark_root": str(libero_root / "libero" / "libero"),
        "bddl_files": str(libero_root / "libero" / "libero" / "bddl_files"),
        "init_states": str(libero_root / "libero" / "libero" / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(libero_root / "libero" / "libero" / "assets"),
    }
    with (config_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


if __name__ == "__main__":
    main()
