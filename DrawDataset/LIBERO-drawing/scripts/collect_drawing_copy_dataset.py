#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
import av
import robosuite.utils.transform_utils as T
from robosuite.utils.control_utils import orientation_error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BRUSH_ROOT_TO_TIP = np.array([0.0, 0.0, -0.018], dtype=np.float64)
BRUSH_ROOT_TO_GRASP = np.array([0.0, 0.0, 0.085], dtype=np.float64)
BRUSH_ROOT_TO_VISIBLE_BOTTOM = np.array([0.0, 0.0, -0.030], dtype=np.float64)
BRUSH_EEF_TO_TIP_Z = float(BRUSH_ROOT_TO_GRASP[2] - BRUSH_ROOT_TO_TIP[2])
PHYSICAL_DRAW_XY_OFFSET = np.array([0.0, 0.0], dtype=np.float64)
PHYSICAL_DRAW_TARGET_X_GAIN = 1.027
PHYSICAL_DRAW_TARGET_Y_GAIN = 1.00
PHYSICAL_DRAW_TARGET_X_SHIFT = 0.0
PHYSICAL_DRAW_TARGET_Y_SHIFT = 0.0
PHYSICAL_DRAW_Z_LIFT = -0.025
PHYSICAL_FIXED_DRAW_EEF_Z_BIAS = -0.006
STROKE_CONTINUATION_DISTANCE_M = 0.03
PHYSICAL_GRASP_EE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PHYSICAL_GRASP_SITE_Z_BIAS = 0.018
BUCKET_TIP_CENTERING_BIAS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
BOARD_SURFACE_Z_FROM_BODY = 0.018
BOARD_HALF_Y_WITH_FRAME = 0.156
BRUSH_INITIAL_RIGHT_X_OFFSET = -0.095
BRUSH_INITIAL_RIGHT_Y_MARGIN = 0.055
_BRUSH_FOLLOW_POS = None
_BRUSH_WELD_ACTIVE = False
_VERTICAL_GRASP_TARGET_ORI = None
_EEF_TARGET_ORI = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect scripted LIBERO drawing-copy data.")
    parser.add_argument("--out", type=Path, default=Path("outputs/drawing_copy_episode"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--control-fps", type=int, default=16)
    parser.add_argument("--stride", type=int, default=2, help="Subsample drawing curve waypoints.")
    parser.add_argument("--record-every", type=int, default=1, help="Record one video frame every N low-level controller steps. Keep this at 1 for action-aligned VLA data.")
    parser.add_argument("--speed-scale", type=float, default=0.08, help="Multiply scripted low-level steps for smoother, slower motion.")
    parser.add_argument("--horizon", type=int, default=5000, help="LIBERO episode horizon for long smooth scripted rollouts.")
    parser.add_argument("--max-recorded-frames", type=int, default=0, help="Abort early when recorded frames exceed this count. 0 disables.")
    parser.add_argument("--use-weld", action="store_true", help="Experimental: activate the precompiled MuJoCo gripper-brush weld.")
    parser.add_argument("--grasp-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Override physical grasp ee offset for debugging.")
    parser.add_argument("--draw-z-lift", type=float, default=None, help="Override draw-phase z lift for physical contact tuning.")
    parser.add_argument("--skip-debug-media", action="store_true", help="Only save HDF5 and summary; used by large LeRobot builds.")
    parser.add_argument("--teleport", action="store_true", help="Scripted grasp helper keeps the vertical brush under the actual gripper pose.")
    parser.add_argument("--no-teleport", dest="teleport", action="store_false")
    parser.set_defaults(teleport=True)
    return parser.parse_args()


def main() -> None:
    global _BRUSH_FOLLOW_POS, _BRUSH_WELD_ACTIVE, _VERTICAL_GRASP_TARGET_ORI, _EEF_TARGET_ORI
    _BRUSH_FOLLOW_POS = None
    _BRUSH_WELD_ACTIVE = False
    _VERTICAL_GRASP_TARGET_ORI = None
    _EEF_TARGET_ORI = None
    args = parse_args()
    grasp_ee_offset = np.asarray(args.grasp_offset, dtype=np.float64) if args.grasp_offset is not None else PHYSICAL_GRASP_EE_OFFSET
    draw_z_lift = 0.0 if args.draw_z_lift is None else float(args.draw_z_lift)
    prepare_local_libero_config(ROOT)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    bddl = (
        Path(get_libero_path("bddl_files"))
        / "libero_drawing"
        / "copy_the_left_board_pattern_to_the_right_board_with_the_soft_brush.bddl"
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview", "frontview", "robot0_eye_in_hand"],
        control_freq=args.control_fps,
        horizon=args.horizon,
    )
    env.seed(args.seed)
    obs = env.reset()
    if hasattr(env.env, "set_drawing_seed"):
        env.env.set_drawing_seed(args.seed)
        env._update_observables(force=True)
        obs = env.env._get_observations()
    initial_brush_pose = place_initial_brush_on_blank_board(env)
    for _ in range(8):
        zero_brush_velocity(env)
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    _VERTICAL_GRASP_TARGET_ORI = get_site_xmat(env, "gripper0_vertical_brush_grasp_site")
    _EEF_TARGET_ORI = get_eef_xmat(env)
    env.env.scripted_ink_mode = args.teleport and not args.use_weld
    frames = []
    wrist_frames = []
    actions = []
    ee_targets = []
    brush_tip_targets = []
    brush_tilt_degrees = []
    grippers = []
    phases = []
    weld_active = []
    sim_states = []

    def record(action, ee, tip, gripper, phase):
        if args.max_recorded_frames > 0 and len(frames) >= args.max_recorded_frames:
            raise RuntimeError(f"too many recorded frames: >{args.max_recorded_frames}")
        frame = obs["agentview_image"][::-1].copy()
        frames.append(frame)
        wrist_frame = obs["robot0_eye_in_hand_image"][::-1].copy()
        wrist_frames.append(wrist_frame)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size == 4:
            action_arr = np.asarray([action_arr[0], action_arr[1], action_arr[2], 0.0, 0.0, 0.0, action_arr[3]], dtype=np.float32)
        actions.append(action_arr)
        ee_targets.append(np.asarray(ee, dtype=np.float32))
        brush_tip_targets.append(np.asarray(tip, dtype=np.float32))
        brush_tilt_degrees.append(float(measure_brush_tilt_degrees(env)))
        grippers.append(float(gripper))
        phases.append(phase)
        weld_active.append(bool(_BRUSH_WELD_ACTIVE))
        sim_states.append(env.get_sim_state().copy())

    zero = np.zeros(7, dtype=np.float32)
    for _ in range(20):
        obs, _, _, _ = env.step(zero)
    initial_sim_state = env.get_sim_state().copy()

    brush_pos = env.env.sim.data.body_xpos[env.env.obj_body_id["soft_brush_1"]].copy()
    brush_grasp_site = get_site_xpos(env, "soft_brush_1_grasp_site")
    stroke_waypoints = [stroke[:: max(1, args.stride)] for stroke in env.env.get_drawing_stroke_waypoints()]
    stroke_waypoints = [stroke for stroke in stroke_waypoints if len(stroke) > 0]
    stroke_waypoints = [densify_world_stroke(stroke, max_step=0.012) for stroke in stroke_waypoints]
    stroke_waypoints = merge_nearby_strokes(stroke_waypoints, max_gap=STROKE_CONTINUATION_DISTANCE_M)
    curve_waypoints = np.concatenate(stroke_waypoints, axis=0) if stroke_waypoints else np.empty((0, 3), dtype=np.float32)
    if len(curve_waypoints) == 0:
        raise RuntimeError("No drawing waypoints generated")

    grasp_offset = current_grasp_site_offset(env)
    bucket_drop_site = get_site_xpos(env, "drawing_water_bucket_1_drop_site")
    if bucket_drop_site is None and "drawing_water_bucket_1" in env.env.obj_body_id:
        bucket_pos = env.env.sim.data.body_xpos[env.env.obj_body_id["drawing_water_bucket_1"]].copy()
        bucket_drop_site = bucket_pos + np.array([0.0, 0.0, 0.085], dtype=np.float64)
    plan = build_waypoint_plan(
        brush_pos,
        brush_grasp_site,
        bucket_drop_site,
        stroke_waypoints,
        grasp_offset,
        speed_scale=args.speed_scale,
        grasp_ee_offset=grasp_ee_offset,
        draw_z_lift=draw_z_lift,
    )
    holding_brush = False
    last_ee = None
    last_tip = None
    frame_clock = 0
    for item in adaptive_plan_items(env, plan, use_weld=args.use_weld, draw_z_lift=draw_z_lift, speed_scale=args.speed_scale):
        if args.teleport:
            start_ee = current_eef_pos(env)
            if start_ee is not None and np.all(np.isfinite(start_ee)):
                last_ee = start_ee
            if args.use_weld and _BRUSH_WELD_ACTIVE:
                current_tip = env.env.brush_tip_world_pos()
                if current_tip is not None and np.all(np.isfinite(current_tip)):
                    last_tip = np.asarray(current_tip, dtype=np.float64)
            for sub_item in interpolate_item(last_ee, last_tip, item):
                for step_item in expand_low_level_steps(sub_item):
                    obs, holding_brush, action = execute_scripted_step(env, step_item, holding_brush, use_weld=args.use_weld)
                    if (
                        not args.use_weld
                        and step_item["phase"] in {"touch_down", "draw"}
                        and np.all(np.isfinite(step_item["tip"]))
                    ):
                        env.env.force_draw_tip(np.asarray(step_item["tip"], dtype=np.float64))
                        env._update_observables(force=True)
                        obs = env.env._get_observations()
                    frame_clock += 1
                    if frame_clock % max(1, args.record_every) == 0:
                        record(action, step_item["ee"], step_item["tip"], step_item["gripper"], step_item["phase"])
            actual_ee = current_eef_pos(env)
            last_ee = np.asarray(actual_ee if actual_ee is not None and np.all(np.isfinite(actual_ee)) else item["ee"], dtype=np.float64)
            current_tip = env.env.brush_tip_world_pos()
            if current_tip is not None and np.all(np.isfinite(current_tip)):
                last_tip = np.asarray(current_tip, dtype=np.float64)
            elif np.all(np.isfinite(item["tip"])):
                last_tip = np.asarray(item["tip"], dtype=np.float64)
            continue
        else:
            action = np.array([0, 0, 0, 0, 0, 0, item["gripper"]], dtype=np.float32)
            obs, _, _, _ = env.step(action)
        record(action, item["ee"], item["tip"], item["gripper"], item["phase"])

    mp4_path = out_dir / "episode.mp4"
    wrist_mp4_path = out_dir / "episode_wrist.mp4"
    hdf5_path = out_dir / "episode.hdf5"
    preview_path = out_dir / "preview.jpg"
    full_sheet_path = out_dir / "contact_sheet_full.jpg"
    frames_dir = out_dir / "frames"
    summary_path = out_dir / "summary.json"

    write_hdf5(
        hdf5_path,
        frames=np.asarray(frames, dtype=np.uint8),
        wrist_frames=np.asarray(wrist_frames, dtype=np.uint8),
        actions=np.asarray(actions, dtype=np.float32),
        ee_targets=np.asarray(ee_targets, dtype=np.float32),
        brush_tip_targets=np.asarray(brush_tip_targets, dtype=np.float32),
        brush_tilt_degrees=np.asarray(brush_tilt_degrees, dtype=np.float32),
        grippers=np.asarray(grippers, dtype=np.float32),
        phases=phases,
        weld_active=np.asarray(weld_active, dtype=np.bool_),
        sim_states=np.asarray(sim_states, dtype=np.float64),
        initial_sim_state=np.asarray(initial_sim_state, dtype=np.float64),
        reference_curve=env.env.reference_curve,
        reference_stroke_lengths=np.asarray([len(stroke) for stroke in env.env.reference_strokes], dtype=np.int32),
        drawing_waypoints=curve_waypoints,
        drawing_stroke_lengths=np.asarray([len(stroke) for stroke in stroke_waypoints], dtype=np.int32),
        bddl=str(bddl),
        seed=args.seed,
        camera_size=args.camera_size,
        fps=args.fps,
        control_fps=args.control_fps,
        teleport=bool(args.teleport and not args.use_weld),
        scripted_controller=bool(args.teleport),
        use_weld=args.use_weld,
        scripted_ink_mode=bool(env.env.scripted_ink_mode),
        speed_scale=args.speed_scale,
        grasp_ee_offset=grasp_ee_offset,
        draw_z_lift=draw_z_lift,
        initial_brush_pose=json.dumps(initial_brush_pose, ensure_ascii=False),
        reference_sketch_type=str(getattr(env.env, "reference_sketch_type", "unknown")),
        num_reference_strokes=int(len(getattr(env.env, "reference_strokes", []))),
        reference_points=int(len(getattr(env.env, "reference_curve", []))),
        drawing_points=int(len(curve_waypoints)),
        horizon=args.horizon,
        draw_segment_count=int(getattr(env.env, "_draw_segment_count", 0)),
        draw_canvas_dark_pixels=count_draw_canvas_dark_pixels(env.env),
        action_frame_alignment="post_step_frame_matches_action_t_result",
        low_level_step_recording=bool(args.record_every == 1),
    )
    if not args.skip_debug_media:
        write_mp4_pyav(mp4_path, frames, fps=args.fps)
        write_mp4_pyav(wrist_mp4_path, wrist_frames, fps=args.fps)
        write_preview(preview_path, frames)
        write_frame_images(frames_dir, frames)
        write_contact_sheet(full_sheet_path, frames)
    summary = {
        "mp4": str(mp4_path.resolve()),
        "wrist_mp4": str(wrist_mp4_path.resolve()),
        "hdf5": str(hdf5_path.resolve()),
        "preview": str(preview_path.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "contact_sheet_full": str(full_sheet_path.resolve()),
        "frames": len(frames),
        "actions": len(actions),
        "max_brush_tilt_degrees": float(np.nanmax(brush_tilt_degrees)) if brush_tilt_degrees else None,
        "camera_size": args.camera_size,
        "control_fps": args.control_fps,
        "seed": args.seed,
        "teleport": bool(args.teleport and not args.use_weld),
        "scripted_controller": bool(args.teleport),
        "use_weld": args.use_weld,
        "scripted_ink_mode": bool(env.env.scripted_ink_mode),
        "speed_scale": args.speed_scale,
        "grasp_ee_offset": [float(x) for x in grasp_ee_offset],
        "initial_brush_pose": initial_brush_pose,
        "reference_sketch_type": str(getattr(env.env, "reference_sketch_type", "unknown")),
        "num_reference_strokes": int(len(getattr(env.env, "reference_strokes", []))),
        "reference_points": int(len(getattr(env.env, "reference_curve", []))),
        "drawing_points": int(len(curve_waypoints)),
        "draw_z_lift": draw_z_lift,
        "horizon": args.horizon,
        "bddl": str(bddl.resolve()),
        "draw_segment_count": int(getattr(env.env, "_draw_segment_count", 0)),
        "draw_canvas_dark_pixels": count_draw_canvas_dark_pixels(env.env),
        "action_frame_alignment": "post_step_frame_matches_action_t_result",
        "low_level_step_recording": bool(args.record_every == 1),
        "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    env.close()


def count_draw_canvas_dark_pixels(task_env) -> int:
    canvas = getattr(task_env, "draw_canvas", None)
    if canvas is None:
        return 0
    gray = np.asarray(canvas, dtype=np.uint8).mean(axis=2)
    return int(np.count_nonzero(gray < 80))


def prepare_local_libero_config(libero_root: Path) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    config_dir = libero_root / ".libero_local"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = libero_root / "empty_datasets"
    datasets_dir.mkdir(exist_ok=True)
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


def current_grasp_site_offset(env) -> np.ndarray:
    env._update_observables(force=True)
    obs = env.env._get_observations()
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    site = get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
    if site is None:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(site, dtype=np.float64) - eef


def place_initial_brush_on_blank_board(env) -> dict:
    ref_pos = env.env.sim.data.body_xpos[env.env.obj_body_id["drawing_board_reference_1"]].copy()
    blank_pos = env.env.sim.data.body_xpos[env.env.obj_body_id["drawing_board_blank_1"]].copy()
    surface_z = float(min(ref_pos[2], blank_pos[2]) + BOARD_SURFACE_Z_FROM_BODY)
    center_x = 0.5 * (ref_pos[0] + blank_pos[0])
    right_outer_y = float(blank_pos[1] + BOARD_HALF_Y_WITH_FRAME + BRUSH_INITIAL_RIGHT_Y_MARGIN)
    root_pos = np.array(
        [
            float(center_x + BRUSH_INITIAL_RIGHT_X_OFFSET),
            float(right_outer_y),
            float(surface_z - BRUSH_ROOT_TO_VISIBLE_BOTTOM[2] + 0.001),
        ],
        dtype=np.float64,
    )
    joint_name = find_brush_joint_name(env)
    if joint_name is None:
        raise RuntimeError("Could not find soft brush free joint for precise initial placement")
    addr = env.sim.model.get_joint_qpos_addr(joint_name)
    start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
    qpos = env.sim.data.qpos[start:end].copy()
    qpos[:3] = root_pos
    if qpos.size >= 7:
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    env.sim.data.qpos[start:end] = qpos
    zero_brush_velocity(env)
    env.sim.forward()
    env._update_observables(force=True)
    return {
        "reference_board_body_pos": [float(x) for x in ref_pos],
        "blank_board_body_pos": [float(x) for x in blank_pos],
        "board_surface_z": surface_z,
        "right_outer_y": right_outer_y,
        "root_pos": [float(x) for x in root_pos],
        "tip_pos": [float(x) for x in (root_pos + BRUSH_ROOT_TO_TIP)],
        "visible_bottom_pos": [float(x) for x in (root_pos + BRUSH_ROOT_TO_VISIBLE_BOTTOM)],
        "grasp_site_pos": [float(x) for x in (root_pos + BRUSH_ROOT_TO_GRASP)],
        "agentview_inner_x_offset": float(BRUSH_INITIAL_RIGHT_X_OFFSET),
        "right_y_margin": float(BRUSH_INITIAL_RIGHT_Y_MARGIN),
    }


def build_waypoint_plan(
    brush_pos: np.ndarray,
    brush_grasp_site: np.ndarray | None,
    bucket_drop_site: np.ndarray | None,
    draw_strokes: list[np.ndarray],
    grasp_site_offset_from_eef: np.ndarray,
    speed_scale: float = 1.0,
    grasp_ee_offset: np.ndarray | None = None,
    draw_z_lift: float | None = None,
) -> list[dict]:
    scale = max(0.05, float(speed_scale))
    if grasp_ee_offset is None:
        grasp_ee_offset = PHYSICAL_GRASP_EE_OFFSET
    grasp_ee_offset = np.asarray(grasp_ee_offset, dtype=np.float64)
    if draw_z_lift is None:
        draw_z_lift = PHYSICAL_DRAW_Z_LIFT

    def item(
        phase: str,
        ee,
        gripper: float,
        tip=None,
        steps: int = 8,
        grasp_site_target=None,
        servo_steps_per_record: int = 1,
        max_step_m: float | None = None,
    ) -> dict:
        out = {
            "phase": phase,
            "ee": [float(x) for x in ee],
            "tip": [float(x) for x in (tip if tip is not None else [np.nan, np.nan, np.nan])],
            "gripper": float(gripper),
            "steps": max(1, int(round(steps * scale))),
            "servo_steps_per_record": max(1, int(servo_steps_per_record)),
        }
        if grasp_site_target is not None:
            out["grasp_site_target"] = [float(x) for x in grasp_site_target]
        if max_step_m is not None:
            out["max_step_m"] = float(max_step_m)
        return out

    safe_z = 1.18
    pickup_site = np.asarray(brush_grasp_site, dtype=np.float64) if brush_grasp_site is not None else brush_pos + BRUSH_ROOT_TO_GRASP
    pickup_control_site = pickup_site + np.array([0.0, 0.0, PHYSICAL_GRASP_SITE_Z_BIAS], dtype=np.float64)
    pickup_ee = pickup_control_site - grasp_site_offset_from_eef + grasp_ee_offset
    hover_site = np.array([pickup_site[0], pickup_site[1], safe_z + grasp_site_offset_from_eef[2]], dtype=np.float64)
    approach_ee = hover_site - grasp_site_offset_from_eef + grasp_ee_offset
    plan = [
        item("approach_brush", approach_ee, -1.0, steps=80, grasp_site_target=hover_site, servo_steps_per_record=1),
        item("hover_over_brush", approach_ee, -1.0, steps=24, grasp_site_target=hover_site, servo_steps_per_record=2),
        item("descend_to_grasp", pickup_ee, -1.0, steps=120, grasp_site_target=pickup_control_site, servo_steps_per_record=1),
        item("align_to_grasp", pickup_ee, -1.0, tip=brush_pos + BRUSH_ROOT_TO_TIP, steps=40, servo_steps_per_record=1),
        item("settle_on_grasp", pickup_ee, -1.0, tip=brush_pos + BRUSH_ROOT_TO_TIP, steps=8),
        item("close_gripper", pickup_ee, 1.0, tip=brush_pos + BRUSH_ROOT_TO_TIP, steps=12),
        item("hold_closed", pickup_ee, 1.0, tip=brush_pos + BRUSH_ROOT_TO_TIP, steps=8),
        item("lift_brush", approach_ee, 1.0, tip=approach_ee - BRUSH_ROOT_TO_GRASP + BRUSH_ROOT_TO_TIP, steps=24),
    ]
    draw_offset = np.array([PHYSICAL_DRAW_XY_OFFSET[0], PHYSICAL_DRAW_XY_OFFSET[1], 0.0], dtype=np.float64)
    draw_strokes = [
        compensate_physical_draw_targets(np.asarray(stroke, dtype=np.float64) + draw_offset)
        for stroke in draw_strokes
        if len(stroke) > 0
    ]
    if not draw_strokes:
        raise ValueError("build_waypoint_plan requires at least one drawing stroke")
    first = draw_strokes[0][0]
    first_hover_tip = np.asarray(first, dtype=np.float64) + np.array([0.0, 0.0, 0.08])
    draw_lift = float(draw_z_lift)
    first_draw_ee = np.asarray(first, dtype=np.float64) + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift])
    plan.extend(
        [
            item(
                "approach_draw_board",
                first_hover_tip + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift]),
                1.0,
                tip=first_hover_tip,
                steps=24,
            ),
            item("touch_down", first_draw_ee, 1.0, tip=first, steps=50, servo_steps_per_record=3),
        ]
    )
    last = first
    last_hover_tip = first_hover_tip
    for stroke_idx, draw_pts in enumerate(draw_strokes):
        if stroke_idx > 0:
            first = draw_pts[0]
            first_hover_tip = np.asarray(first, dtype=np.float64) + np.array([0.0, 0.0, 0.09])
            first_draw_ee = np.asarray(first, dtype=np.float64) + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift])
            plan.extend(
                [
                    item(
                        "move_between_strokes",
                        first_hover_tip + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift]),
                        1.0,
                        tip=first_hover_tip,
                        steps=24,
                        servo_steps_per_record=1,
                    ),
                    item("touch_down", first_draw_ee, 1.0, tip=first, steps=44, servo_steps_per_record=3),
                ]
            )
        for point_idx, p in enumerate(draw_pts):
            p = np.asarray(p, dtype=np.float64)
            plan.append(
                item(
                    "draw",
                    p + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift]),
                    1.0,
                    tip=p,
                    steps=18,
                    servo_steps_per_record=3,
                )
            )
            if point_idx == 0:
                plan.append(
                    item(
                        "draw",
                        p + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift]),
                        1.0,
                        tip=p,
                        steps=8,
                        servo_steps_per_record=3,
                    )
                )
        last = np.asarray(draw_pts[-1], dtype=np.float64)
        lift_height = 0.09 if stroke_idx < len(draw_strokes) - 1 else 0.18
        last_hover_tip = last + np.array([0.0, 0.0, lift_height], dtype=np.float64)
        plan.append(
            item(
                "lift_after_draw",
                last_hover_tip + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift]),
                1.0,
                tip=last_hover_tip,
                steps=8 if stroke_idx < len(draw_strokes) - 1 else 24,
                servo_steps_per_record=1,
                max_step_m=0.0065 if stroke_idx == len(draw_strokes) - 1 else None,
            )
        )
    if bucket_drop_site is not None:
        bucket_drop_site = np.asarray(bucket_drop_site, dtype=np.float64)
        bucket_center_tip = bucket_drop_site
        bucket_hover_tip = bucket_center_tip + np.array([0.0, 0.0, 0.14], dtype=np.float64)
        bucket_release_tip = bucket_center_tip
        bucket_hover = bucket_hover_tip + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift], dtype=np.float64)
        bucket_release = bucket_release_tip + np.array([0.0, 0.0, BRUSH_EEF_TO_TIP_Z + draw_lift], dtype=np.float64)
        plan.extend(
            [
                item("move_to_bucket", bucket_hover, 1.0, tip=bucket_hover_tip, steps=120, servo_steps_per_record=1),
                item("lower_into_bucket", bucket_release, 1.0, tip=bucket_release_tip, steps=80, servo_steps_per_record=1),
                item("hold_bucket_center", bucket_release, 1.0, tip=bucket_release_tip, steps=12, servo_steps_per_record=1),
                item("release_in_bucket", bucket_release, -1.0, tip=bucket_release_tip, steps=12, servo_steps_per_record=1),
                item("retreat_after_bucket", bucket_hover, -1.0, steps=16),
            ]
        )
    else:
        retreat_ee = np.array([last[0], last[1], safe_z + 0.22], dtype=np.float64)
        plan.append(item("retreat_after_draw", retreat_ee, 1.0, steps=60))
    return plan


def compensate_physical_draw_targets(draw_pts: np.ndarray) -> np.ndarray:
    """Counteract OSC lag so the physical brush tip lands on the intended canvas scale."""
    pts = np.asarray(draw_pts, dtype=np.float64).copy()
    if len(pts) == 0:
        return pts
    center = 0.5 * (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0))
    pts[:, 0] = center[0] + (pts[:, 0] - center[0]) * PHYSICAL_DRAW_TARGET_X_GAIN + PHYSICAL_DRAW_TARGET_X_SHIFT
    pts[:, 1] = center[1] + (pts[:, 1] - center[1]) * PHYSICAL_DRAW_TARGET_Y_GAIN + PHYSICAL_DRAW_TARGET_Y_SHIFT
    return pts


def densify_world_stroke(stroke: np.ndarray, max_step: float = 0.012) -> np.ndarray:
    pts = np.asarray(stroke, dtype=np.float64)
    if len(pts) <= 1:
        return pts.astype(np.float32)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        dist = float(np.linalg.norm(b[:2] - a[:2]))
        steps = max(1, int(np.ceil(dist / max_step)))
        for i in range(1, steps + 1):
            alpha = i / steps
            out.append((1.0 - alpha) * a + alpha * b)
    return np.asarray(out, dtype=np.float32)


def merge_nearby_strokes(strokes: list[np.ndarray], max_gap: float = STROKE_CONTINUATION_DISTANCE_M) -> list[np.ndarray]:
    merged: list[np.ndarray] = []
    for stroke in strokes:
        pts = np.asarray(stroke, dtype=np.float32)
        if len(pts) == 0:
            continue
        if not merged:
            merged.append(pts)
            continue
        gap = float(np.linalg.norm(merged[-1][-1, :2] - pts[0, :2]))
        if gap < max_gap:
            tail = pts[1:] if gap < 1e-6 and len(pts) > 1 else pts
            merged[-1] = np.concatenate([merged[-1], tail], axis=0)
        else:
            merged.append(pts)
    return merged


def adaptive_plan_items(env, plan: list[dict], use_weld: bool, draw_z_lift: float, speed_scale: float):
    if not use_weld:
        yield from plan
        return
    split_idx = None
    for idx, item in enumerate(plan):
        yield item
        if item["phase"] == "lift_brush":
            split_idx = idx + 1
            break
    if split_idx is None:
        return
    remaining = plan[split_idx:]
    eef_from_tip = None
    fixed_draw_eef_z = None
    z_bias = np.array([0.0, 0.0, float(draw_z_lift)], dtype=np.float64)
    for item in remaining:
        desired_tip = np.asarray(item["tip"], dtype=np.float64)
        if np.all(np.isfinite(desired_tip)):
            eef = current_eef_pos(env)
            tip = env.env.brush_tip_world_pos()
            if eef is not None and tip is not None and np.all(np.isfinite(eef)) and np.all(np.isfinite(tip)):
                eef_from_tip = np.asarray(eef, dtype=np.float64) - np.asarray(tip, dtype=np.float64)
            if eef_from_tip is None:
                yield item
                continue
            adapted = dict(item)
            if item["phase"] in {"touch_down", "draw"}:
                target_tip = desired_tip + z_bias
            else:
                target_tip = desired_tip
            adapted_ee = target_tip + eef_from_tip
            if item["phase"] in {"touch_down", "draw"}:
                if fixed_draw_eef_z is None:
                    fixed_draw_eef_z = float(adapted_ee[2] + PHYSICAL_FIXED_DRAW_EEF_Z_BIAS)
                adapted_ee[2] = fixed_draw_eef_z
            adapted["contact_tip"] = [float(x) for x in target_tip]
            adapted["ee"] = [float(x) for x in adapted_ee]
            yield adapted
        else:
            yield item


def current_eef_pos(env) -> np.ndarray | None:
    try:
        env._update_observables(force=True)
        obs = env.env._get_observations()
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
    except Exception:
        return None


def interpolate_item(last_ee: np.ndarray | None, last_tip: np.ndarray | None, item: dict) -> list[dict]:
    target_ee = np.asarray(item["ee"], dtype=np.float64)
    item_steps = max(1, int(item.get("steps", 1)))
    target_grasp_site = np.asarray(item.get("grasp_site_target", [np.nan, np.nan, np.nan]), dtype=np.float64)
    multi_servo_phases = {
        "approach_brush",
        "hover_over_brush",
        "descend_to_grasp",
        "align_to_grasp",
        "touch_down",
        "draw",
        "lift_after_draw",
        "move_between_strokes",
        "move_to_bucket",
        "lower_into_bucket",
        "hold_bucket_center",
        "release_in_bucket",
    }
    direct_tip_phases = {"hold_bucket_center", "release_in_bucket"}
    servo_steps_per_record = max(1, int(item.get("servo_steps_per_record", 1)))
    if last_ee is None or not np.all(np.isfinite(last_ee)):
        out = []
        for i in range(item_steps):
            sub = dict(item)
            sub["steps"] = servo_steps_per_record if item["phase"] in multi_servo_phases else 1
            sub["force_record"] = i == item_steps - 1
            out.append(sub)
        return out
    target_tip = np.asarray(item["tip"], dtype=np.float64)
    dist = float(np.linalg.norm(target_ee - last_ee))
    phase = item["phase"]
    if phase == "draw":
        max_step_m = 0.012
    elif phase in {"approach_brush", "hover_over_brush"}:
        max_step_m = 0.022
    elif phase in {"descend_to_grasp", "align_to_grasp", "settle_on_grasp", "close_gripper"}:
        max_step_m = 0.007
    elif phase == "move_between_strokes":
        max_step_m = 0.012
    elif phase == "move_to_bucket":
        max_step_m = 0.0045
    elif phase in {"lower_into_bucket", "retreat_after_bucket"}:
        max_step_m = 0.008
    else:
        max_step_m = 0.010
    max_step_m = float(item.get("max_step_m", max_step_m))
    steps = max(item_steps, int(np.ceil(dist / max_step_m)))
    if item["phase"] in {"draw", "touch_down"}:
        steps = max(steps, 1)
    if item["phase"] == "approach_brush" and np.all(np.isfinite(target_ee)):
        return interpolate_approach_brush_item(last_ee, target_ee, target_grasp_site, item, item_steps)
    if item["phase"] == "touch_down" and np.all(np.isfinite(target_ee)) and last_tip is not None and np.all(np.isfinite(last_tip)):
        return interpolate_touch_down_item(last_ee, last_tip, item, item_steps)
    if steps == 1:
        sub = dict(item)
        sub["steps"] = 1
        return [sub]
    out = []
    for i in range(1, steps + 1):
        alpha = i / steps
        ee = (1.0 - alpha) * last_ee + alpha * target_ee
        if item["phase"] == "draw" and np.all(np.isfinite(target_ee)):
            ee[2] = target_ee[2]
        sub = dict(item)
        sub["ee"] = [float(x) for x in ee]
        sub["steps"] = servo_steps_per_record if item["phase"] in multi_servo_phases else 1
        if np.all(np.isfinite(target_grasp_site)):
            offset = target_grasp_site - target_ee
            grasp_site = ee + offset
            sub["grasp_site_target"] = [float(x) for x in grasp_site]
        if np.all(np.isfinite(target_tip)):
            if item["phase"] in direct_tip_phases:
                tip = target_tip
            elif last_tip is not None and np.all(np.isfinite(last_tip)):
                tip = (1.0 - alpha) * last_tip + alpha * target_tip
            else:
                tip = target_tip
            sub["tip"] = [float(x) for x in tip]
            if "contact_tip" in sub:
                sub["contact_tip"] = [float(x) for x in tip]
        sub["force_record"] = i == steps
        out.append(sub)
    return out


def expand_low_level_steps(item: dict) -> list[dict]:
    steps = max(1, int(item.get("steps", 1)))
    if steps == 1:
        sub = dict(item)
        sub["steps"] = 1
        return [sub]
    out = []
    for idx in range(steps):
        sub = dict(item)
        sub["steps"] = 1
        sub["substep_index"] = idx
        sub["substep_count"] = steps
        out.append(sub)
    return out


def interpolate_approach_brush_item(
    last_ee: np.ndarray,
    target_ee: np.ndarray,
    target_grasp_site: np.ndarray,
    item: dict,
    item_steps: int,
) -> list[dict]:
    lift_ee = np.array([last_ee[0], last_ee[1], target_ee[2]], dtype=np.float64)
    offset = target_grasp_site - target_ee if np.all(np.isfinite(target_grasp_site)) else None
    lift_dist = abs(float(lift_ee[2] - last_ee[2]))
    horizontal_dist = float(np.linalg.norm(target_ee[:2] - lift_ee[:2]))
    lift_steps = max(4, int(np.ceil(lift_dist / 0.012)))
    horizontal_steps = max(item_steps, int(np.ceil(horizontal_dist / 0.018)))
    out = []

    for i in range(1, lift_steps + 1):
        alpha = i / lift_steps
        ee = (1.0 - alpha) * last_ee + alpha * lift_ee
        sub = dict(item)
        sub["ee"] = [float(x) for x in ee]
        sub["steps"] = max(1, int(item.get("servo_steps_per_record", 1)))
        if offset is not None:
            sub["grasp_site_target"] = [float(x) for x in (ee + offset)]
        sub["force_record"] = False
        out.append(sub)

    for i in range(1, horizontal_steps + 1):
        alpha = i / horizontal_steps
        ee = (1.0 - alpha) * lift_ee + alpha * target_ee
        ee[2] = target_ee[2]
        sub = dict(item)
        sub["ee"] = [float(x) for x in ee]
        sub["steps"] = max(1, int(item.get("servo_steps_per_record", 1)))
        if offset is not None:
            sub["grasp_site_target"] = [float(x) for x in (ee + offset)]
        sub["force_record"] = i == horizontal_steps
        out.append(sub)
    return out


def interpolate_touch_down_item(
    last_ee: np.ndarray,
    last_tip: np.ndarray,
    item: dict,
    item_steps: int,
) -> list[dict]:
    target_ee = np.asarray(item["ee"], dtype=np.float64)
    target_tip = np.asarray(item["tip"], dtype=np.float64)
    if float(np.linalg.norm(last_tip[:2] - target_tip[:2])) < 0.004:
        out = []
        steps = max(1, item_steps)
        for i in range(1, steps + 1):
            alpha = i / steps
            ee = (1.0 - alpha) * last_ee + alpha * target_ee
            tip = (1.0 - alpha) * last_tip + alpha * target_tip
            sub = dict(item)
            sub["ee"] = [float(x) for x in ee]
            sub["tip"] = [float(x) for x in tip]
            sub["contact_tip"] = [float(x) for x in tip]
            sub["steps"] = max(1, int(item.get("servo_steps_per_record", 1)))
            sub["force_record"] = i == steps
            out.append(sub)
        return out
    hover_tip = np.array([target_tip[0], target_tip[1], max(float(last_tip[2]), float(target_tip[2] + 0.075))], dtype=np.float64)
    eef_from_tip = last_ee - last_tip
    hover_ee = hover_tip + eef_from_tip
    horizontal_dist = float(np.linalg.norm(hover_ee[:2] - last_ee[:2]))
    vertical_dist = abs(float(target_ee[2] - hover_ee[2]))
    horizontal_steps = max(4, int(np.ceil(horizontal_dist / 0.010)))
    vertical_steps = max(item_steps, int(np.ceil(vertical_dist / 0.006)))
    out = []

    for i in range(1, horizontal_steps + 1):
        alpha = i / horizontal_steps
        ee = (1.0 - alpha) * last_ee + alpha * hover_ee
        tip = (1.0 - alpha) * last_tip + alpha * hover_tip
        sub = dict(item)
        sub["ee"] = [float(x) for x in ee]
        sub["tip"] = [float(x) for x in tip]
        sub["contact_tip"] = [float(x) for x in tip]
        sub["steps"] = max(1, int(item.get("servo_steps_per_record", 1)))
        sub["force_record"] = False
        out.append(sub)

    for i in range(1, vertical_steps + 1):
        alpha = i / vertical_steps
        ee = (1.0 - alpha) * hover_ee + alpha * target_ee
        tip = (1.0 - alpha) * hover_tip + alpha * target_tip
        sub = dict(item)
        sub["ee"] = [float(x) for x in ee]
        sub["tip"] = [float(x) for x in tip]
        sub["contact_tip"] = [float(x) for x in tip]
        sub["steps"] = max(1, int(item.get("servo_steps_per_record", 1)))
        sub["force_record"] = i == vertical_steps
        out.append(sub)
    return out


def execute_scripted_step(env, item: dict, holding_brush: bool, use_weld: bool = False):
    target_ee = np.asarray(item["ee"], dtype=np.float64)
    phase = item["phase"]
    gripper = float(item["gripper"])
    if use_weld:
        env.env.inking_enabled = phase == "draw"

    lock_orientation = use_weld and phase in {
        "approach_brush",
        "hover_over_brush",
        "descend_to_grasp",
        "align_to_grasp",
        "settle_on_grasp",
        "close_gripper",
        "hold_closed",
        "lift_brush",
        "approach_draw_board",
        "move_between_strokes",
        "touch_down",
        "draw",
        "lift_after_draw",
        "retreat_after_draw",
        "move_to_bucket",
        "lower_into_bucket",
        "hold_bucket_center",
        "release_in_bucket",
        "retreat_after_bucket",
    }
    contact_tip = np.asarray(item.get("contact_tip", item.get("tip", [np.nan, np.nan, np.nan])), dtype=np.float64)
    grasp_site_target = np.asarray(item.get("grasp_site_target", [np.nan, np.nan, np.nan]), dtype=np.float64)
    if use_weld and phase in {"approach_brush", "hover_over_brush", "descend_to_grasp"} and np.all(np.isfinite(grasp_site_target)):
        obs, last_action = drive_grasp_site_to_target(
            env,
            grasp_site_target,
            gripper,
            steps=1,
            lock_vertical_orientation=lock_orientation,
        )
    elif use_weld and phase == "align_to_grasp":
        obs, last_action = drive_grasp_site_to_brush_site(
            env,
            gripper,
            steps=1,
            lock_vertical_orientation=lock_orientation,
            max_delta=0.12,
        )
    elif use_weld and phase in {"touch_down", "draw"} and np.all(np.isfinite(contact_tip)):
        obs, last_action = drive_tip_xy_with_fixed_eef_z(
            env,
            contact_tip,
            target_eef_z=float(target_ee[2]),
            gripper=gripper,
            steps=1,
            lock_vertical_orientation=lock_orientation,
        )
    elif use_weld and phase in {"touch_down", "draw", "lift_after_draw", "move_between_strokes", "move_to_bucket", "lower_into_bucket", "hold_bucket_center", "release_in_bucket"} and np.all(np.isfinite(contact_tip)):
        bucket_servo = phase in {"hold_bucket_center", "release_in_bucket"}
        if bucket_servo:
            obs, last_action = drive_tip_to_bucket_center(
                env,
                gripper,
                steps=1,
                lock_vertical_orientation=lock_orientation,
                hover_z_offset=0.0,
            )
        else:
            obs, last_action = drive_tip_toward(
                env,
                contact_tip,
                gripper,
                steps=1,
                lock_vertical_orientation=lock_orientation,
            )
    else:
        obs, last_action = drive_eef_toward(
            env,
            target_ee,
            gripper,
            steps=1,
            lock_vertical_orientation=lock_orientation,
        )
    if phase in {
        "hold_closed",
        "lift_brush",
        "approach_draw_board",
        "move_between_strokes",
        "touch_down",
        "draw",
        "lift_after_draw",
        "retreat_after_draw",
        "move_to_bucket",
        "lower_into_bucket",
        "hold_bucket_center",
    }:
        holding_brush = True
    if holding_brush:
        if use_weld and phase == "release_in_bucket":
            release_brush_weld(env)
            holding_brush = False
        elif use_weld:
            holding_brush = attach_brush_weld(env, align_once=False)
        else:
            place_brush_at_gripper(env, smooth=True)
        env.env._post_process()
        env._update_observables(force=True)
        obs = env.env._get_observations()
    return obs, holding_brush, last_action


def drive_tip_toward(
    env,
    target_tip: np.ndarray,
    gripper: float,
    steps: int,
    lock_vertical_orientation: bool = False,
    max_delta: float = 0.35,
):
    obs = None
    action = np.zeros(7, dtype=np.float32)
    target_tip = np.asarray(target_tip, dtype=np.float64)
    for _ in range(max(1, int(steps))):
        env._update_observables(force=True)
        current_obs = env.env._get_observations()
        current_eef = np.asarray(current_obs.get("robot0_eef_pos", target_tip), dtype=np.float64)
        current_tip = np.asarray(env.env.brush_tip_world_pos(), dtype=np.float64)
        tip_error = target_tip - current_tip
        target_eef = current_eef + tip_error
        delta = np.clip((target_eef - current_eef) * 7.0, -max_delta, max_delta)
        ori_delta = vertical_orientation_action(env) if lock_vertical_orientation else np.zeros(3, dtype=np.float64)
        action = np.asarray(
            [delta[0], delta[1], delta[2], ori_delta[0], ori_delta[1], ori_delta[2], gripper],
            dtype=np.float32,
        )
        obs, _, _, _ = env.step(action)
    return obs, action


def current_bucket_center(env) -> np.ndarray | None:
    if "drawing_water_bucket_1" not in env.env.obj_body_id:
        return None
    bucket = env.env.sim.data.body_xpos[env.env.obj_body_id["drawing_water_bucket_1"]].copy()
    return bucket + np.array([0.0, 0.0, 0.085], dtype=np.float64)


def drive_tip_to_bucket_center(
    env,
    gripper: float,
    steps: int,
    lock_vertical_orientation: bool = False,
    hover_z_offset: float = 0.0,
    max_delta: float = 0.75,
):
    obs = None
    action = np.zeros(7, dtype=np.float32)
    for _ in range(max(1, int(steps))):
        bucket_center = current_bucket_center(env)
        if bucket_center is None:
            break
        target_tip = bucket_center + BUCKET_TIP_CENTERING_BIAS + np.array([0.0, 0.0, float(hover_z_offset)], dtype=np.float64)
        env._update_observables(force=True)
        current_obs = env.env._get_observations()
        current_tip = np.asarray(env.env.brush_tip_world_pos(), dtype=np.float64)
        tip_error = target_tip - current_tip
        delta = np.array([tip_error[0] * 12.0, tip_error[1] * 12.0, tip_error[2] * 9.0], dtype=np.float64)
        delta = np.clip(delta, -max_delta, max_delta)
        ori_delta = vertical_orientation_action(env) if lock_vertical_orientation else np.zeros(3, dtype=np.float64)
        action = np.asarray(
            [delta[0], delta[1], delta[2], ori_delta[0], ori_delta[1], ori_delta[2], gripper],
            dtype=np.float32,
        )
        obs, _, _, _ = env.step(action)
    return obs, action


def drive_tip_xy_with_fixed_eef_z(
    env,
    target_tip: np.ndarray,
    target_eef_z: float,
    gripper: float,
    steps: int,
    lock_vertical_orientation: bool = False,
    max_delta: float = 0.75,
):
    obs = None
    action = np.zeros(7, dtype=np.float32)
    target_tip = np.asarray(target_tip, dtype=np.float64)
    for _ in range(max(1, int(steps))):
        env._update_observables(force=True)
        current_obs = env.env._get_observations()
        current_eef = np.asarray(current_obs.get("robot0_eef_pos", target_tip), dtype=np.float64)
        current_tip = np.asarray(env.env.brush_tip_world_pos(), dtype=np.float64)
        xy_error = target_tip[:2] - current_tip[:2]
        z_error = float(target_eef_z) - float(current_eef[2])
        delta = np.array([xy_error[0] * 14.0, xy_error[1] * 18.0, z_error * 18.0], dtype=np.float64)
        delta = np.clip(delta, -max_delta, max_delta)
        ori_delta = vertical_orientation_action(env) if lock_vertical_orientation else np.zeros(3, dtype=np.float64)
        action = np.asarray(
            [delta[0], delta[1], delta[2], ori_delta[0], ori_delta[1], ori_delta[2], gripper],
            dtype=np.float32,
        )
        obs, _, _, _ = env.step(action)
    return obs, action


def drive_eef_toward(
    env,
    target_ee: np.ndarray,
    gripper: float,
    steps: int,
    lock_vertical_orientation: bool = False,
    max_delta: float = 0.38,
):
    obs = None
    action = np.zeros(7, dtype=np.float32)
    for _ in range(max(1, int(steps))):
        env._update_observables(force=True)
        current_obs = env.env._get_observations()
        current = np.asarray(current_obs.get("robot0_eef_pos", target_ee), dtype=np.float64)
        delta = np.clip((target_ee - current) * 7.0, -max_delta, max_delta)
        ori_delta = vertical_orientation_action(env) if lock_vertical_orientation else np.zeros(3, dtype=np.float64)
        action = np.asarray(
            [delta[0], delta[1], delta[2], ori_delta[0], ori_delta[1], ori_delta[2], gripper],
            dtype=np.float32,
        )
        obs, _, _, _ = env.step(action)
    return obs, action


def drive_grasp_site_to_brush_site(
    env,
    gripper: float,
    steps: int,
    lock_vertical_orientation: bool = False,
    max_delta: float = 0.22,
):
    brush_site = get_site_xpos(env, "soft_brush_1_grasp_site")
    if brush_site is None:
        env._update_observables(force=True)
        current_obs = env.env._get_observations()
        current_eef = np.asarray(current_obs.get("robot0_eef_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        return drive_eef_toward(env, current_eef, gripper, steps=1, lock_vertical_orientation=lock_vertical_orientation)
    return drive_grasp_site_to_target(
        env,
        np.asarray(brush_site, dtype=np.float64),
        gripper,
        steps=steps,
        lock_vertical_orientation=lock_vertical_orientation,
        max_delta=max_delta,
    )


def drive_grasp_site_to_target(
    env,
    target_grasp_site: np.ndarray,
    gripper: float,
    steps: int,
    lock_vertical_orientation: bool = False,
    max_delta: float = 0.45,
):
    obs = None
    action = np.zeros(7, dtype=np.float32)
    target_grasp_site = np.asarray(target_grasp_site, dtype=np.float64)
    for _ in range(max(1, int(steps))):
        env._update_observables(force=True)
        current_obs = env.env._get_observations()
        current_eef = np.asarray(current_obs.get("robot0_eef_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        gripper_site = get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
        if gripper_site is None:
            return drive_eef_toward(env, current_eef, gripper, steps=1, lock_vertical_orientation=lock_vertical_orientation)
        error = target_grasp_site - np.asarray(gripper_site, dtype=np.float64)
        target_eef = current_eef + error
        delta = np.clip((target_eef - current_eef) * 10.0, -max_delta, max_delta)
        ori_delta = vertical_orientation_action(env) if lock_vertical_orientation else np.zeros(3, dtype=np.float64)
        action = np.asarray(
            [delta[0], delta[1], delta[2], ori_delta[0], ori_delta[1], ori_delta[2], gripper],
            dtype=np.float32,
        )
        obs, _, _, _ = env.step(action)
    return obs, action


def vertical_orientation_action(env) -> np.ndarray:
    if _EEF_TARGET_ORI is None:
        return np.zeros(3, dtype=np.float64)
    current = get_eef_xmat(env)
    if current is None:
        return np.zeros(3, dtype=np.float64)
    err = orientation_error(_EEF_TARGET_ORI, current)
    return np.clip(err * 2.2, -0.75, 0.75)


def get_eef_xmat(env) -> np.ndarray | None:
    try:
        controller = env.env.robots[0].controller
        controller.update(force=True)
        return np.asarray(controller.ee_ori_mat, dtype=np.float64).copy()
    except Exception:
        return None


def get_site_xmat(env, site_name: str) -> np.ndarray | None:
    try:
        site_id = env.sim.model.site_name2id(site_name)
    except Exception:
        return None
    return env.sim.data.site_xmat[site_id].reshape(3, 3).copy()


def attach_brush_weld(env, align_once: bool = False) -> bool:
    global _BRUSH_WELD_ACTIVE
    if _BRUSH_WELD_ACTIVE:
        return True
    if align_once:
        align_grasp_sites_before_weld(env)
        hold_current_eef(env, steps=18, gripper=1.0, lock_vertical_orientation=True)
    if grasp_site_distance(env) > 0.030:
        return False
    if not has_gripper_brush_contact(env):
        return False
    eq_id = find_equality_id(env, "drawing_brush_gripper_weld")
    if eq_id is not None:
        update_body_weld_relpose(env, eq_id)
        env.sim.data.eq_active[eq_id] = 1
        _BRUSH_WELD_ACTIVE = True
        zero_brush_velocity(env)
        env.sim.forward()
        return True
    return False


def release_brush_weld(env):
    global _BRUSH_WELD_ACTIVE
    eq_id = find_equality_id(env, "drawing_brush_gripper_weld")
    if eq_id is not None:
        env.sim.data.eq_active[eq_id] = 0
    _BRUSH_WELD_ACTIVE = False
    zero_brush_velocity(env)
    env.sim.forward()


def grasp_site_distance(env) -> float:
    gripper_site = get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
    brush_site = get_site_xpos(env, "soft_brush_1_grasp_site")
    if gripper_site is None or brush_site is None:
        return float("inf")
    return float(np.linalg.norm(gripper_site - brush_site))


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


def align_grasp_sites_before_weld(env, tolerance: float = 0.0025, max_rounds: int = 12):
    env._update_observables(force=True)
    obs = env.env._get_observations()
    target = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    for _ in range(max(1, int(max_rounds))):
        gripper_site = get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
        brush_site = get_site_xpos(env, "soft_brush_1_grasp_site")
        if gripper_site is None or brush_site is None:
            break
        error = brush_site - gripper_site
        if float(np.linalg.norm(error)) <= tolerance:
            break
        target = target + error
        drive_eef_toward(
            env,
            target,
            gripper=1.0,
            steps=10,
            lock_vertical_orientation=True,
            max_delta=0.28,
        )


def update_body_weld_relpose(env, eq_id: int):
    model = env.sim.model
    data = env.sim.data
    body1 = int(model.eq_obj1id[eq_id])
    body2 = int(model.eq_obj2id[eq_id])
    pos1 = np.asarray(data.body_xpos[body1], dtype=np.float64)
    mat1 = np.asarray(data.body_xmat[body1], dtype=np.float64).reshape(3, 3)
    pos2 = np.asarray(data.body_xpos[body2], dtype=np.float64)
    mat2 = np.asarray(data.body_xmat[body2], dtype=np.float64).reshape(3, 3)
    rel_pos = mat1.T @ (pos2 - pos1)
    rel_mat = mat1.T @ mat2
    rel_quat_xyzw = np.asarray(T.mat2quat(rel_mat), dtype=np.float64)
    rel_quat_wxyz = rel_quat_xyzw[[3, 0, 1, 2]]
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat_wxyz


def hold_current_eef(env, steps: int, gripper: float, lock_vertical_orientation: bool):
    env._update_observables(force=True)
    obs = env.env._get_observations()
    target = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    drive_eef_toward(env, target, gripper, steps=steps, lock_vertical_orientation=lock_vertical_orientation)


def settle_brush(env, steps: int = 4):
    for _ in range(max(1, int(steps))):
        zero_brush_velocity(env)
        env.sim.forward()


def zero_brush_velocity(env):
    joint_name = find_brush_joint_name(env)
    if joint_name is None:
        return
    qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
    vstart, vend = qvel_addr if isinstance(qvel_addr, tuple) else (qvel_addr, qvel_addr + 1)
    env.sim.data.qvel[vstart:vend] = 0.0


def find_equality_id(env, name: str):
    for idx in range(env.sim.model.neq):
        adr = int(env.sim.model.name_eqadr[idx])
        end = env.sim.model.names.find(b"\x00", adr)
        if env.sim.model.names[adr:end].decode("utf-8") == name:
            return idx
    return None


def place_brush_at_gripper(env, smooth: bool = False):
    env._update_observables(force=True)
    body_id = env.env.obj_body_id["soft_brush_1"]
    joint_name = find_brush_joint_name(env)
    global _BRUSH_FOLLOW_POS
    grasp_pos = get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
    if grasp_pos is None:
        obs = env.env._get_observations()
        grasp_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    target_pos = grasp_pos - BRUSH_ROOT_TO_GRASP
    if smooth:
        if _BRUSH_FOLLOW_POS is None:
            _BRUSH_FOLLOW_POS = target_pos.copy()
        else:
            _BRUSH_FOLLOW_POS = 0.72 * _BRUSH_FOLLOW_POS + 0.28 * target_pos
        pos = _BRUSH_FOLLOW_POS
    else:
        pos = target_pos
    if joint_name is not None:
        addr = env.sim.model.get_joint_qpos_addr(joint_name)
        start, end = addr if isinstance(addr, tuple) else (addr, addr + 1)
        qpos = env.sim.data.qpos[start:end].copy()
        qpos[:3] = pos
        if qpos.size >= 7:
            qpos[3:7] = brush_grasp_quat(env)
        env.sim.data.qpos[start:end] = qpos
        qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
        vstart, vend = qvel_addr if isinstance(qvel_addr, tuple) else (qvel_addr, qvel_addr + 1)
        env.sim.data.qvel[vstart:vend] = 0.0
    else:
        env.sim.data.body_xpos[body_id] = pos
    env.sim.forward()


def find_brush_joint_name(env):
    body_id = env.env.obj_body_id["soft_brush_1"]
    body = env.sim.model.body_names[body_id]
    prefix = body.split("_main")[0] if "_main" in body else "soft_brush_1"
    for name in env.sim.model.joint_names:
        if prefix in name:
            return name
    return None


def get_site_xpos(env, site_name: str) -> np.ndarray | None:
    try:
        site_id = env.sim.model.site_name2id(site_name)
    except Exception:
        return None
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def brush_grasp_quat(env) -> np.ndarray:
    mat = get_site_xmat(env, "gripper0_vertical_brush_grasp_site")
    if mat is None:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat_xyzw = np.asarray(T.mat2quat(mat), dtype=np.float64)
    return quat_xyzw[[3, 0, 1, 2]]


def measure_brush_tilt_degrees(env) -> float:
    try:
        top = find_first_site_xpos(env, "soft_brush_1", "axis_top_site")
        bottom = find_first_site_xpos(env, "soft_brush_1", "axis_bottom_site")
        if top is not None and bottom is not None:
            axis = top - bottom
        else:
            body_id = env.env.obj_body_id["soft_brush_1"]
            mat = env.sim.data.body_xmat[body_id].reshape(3, 3)
            axis = mat[:, 2]
        vertical = abs(float(axis[2])) / max(1e-8, float(np.linalg.norm(axis)))
        return float(np.degrees(np.arccos(np.clip(vertical, -1.0, 1.0))))
    except Exception:
        return float("nan")


def find_first_site_xpos(env, object_prefix: str, site_suffix: str) -> np.ndarray | None:
    for name in env.sim.model.site_names:
        if object_prefix in name and name.endswith(site_suffix):
            return get_site_xpos(env, name)
    return None


def write_hdf5(path: Path, **arrays) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "libero_drawing_copy_v1"
        for key, value in arrays.items():
            if isinstance(value, str):
                f.attrs[key] = value
            elif isinstance(value, (int, float, bool)):
                f.attrs[key] = value
            elif key == "phases":
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(key, data=np.asarray(value, dtype=object), dtype=dt)
            elif value is not None:
                f.create_dataset(key, data=value)


def write_mp4_pyav(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("cannot write empty mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_preview(path: Path, frames: list[np.ndarray], count: int = 12) -> None:
    if not frames:
        return
    idxs = np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int)
    thumbs = []
    for idx in idxs:
        img = frames[idx]
        thumbs.append(cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA))
    rows = []
    for start in range(0, len(thumbs), 4):
        row = thumbs[start : start + 4]
        while len(row) < 4:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(cv2.hconcat([x[:, :, ::-1] for x in row]))
    cv2.imwrite(str(path), cv2.vconcat(rows))


def write_frame_images(path: Path, frames: list[np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("frame_*.jpg"):
        old.unlink()
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(path / f"frame_{idx:06d}.jpg"), frame[:, :, ::-1])


def write_contact_sheet(path: Path, frames: list[np.ndarray], thumb_size: int = 96) -> None:
    if not frames:
        return
    cols = int(np.ceil(np.sqrt(len(frames))))
    rows = int(np.ceil(len(frames) / cols))
    thumbs = [cv2.resize(f, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA) for f in frames]
    blank = np.zeros_like(thumbs[0])
    tiles = []
    for r in range(rows):
        row = thumbs[r * cols : (r + 1) * cols]
        while len(row) < cols:
            row.append(blank)
        tiles.append(cv2.hconcat([x[:, :, ::-1] for x in row]))
    cv2.imwrite(str(path), cv2.vconcat(tiles))


if __name__ == "__main__":
    main()
