#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.collect_drawing_copy_dataset as collect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast sweep physical brush grasp offsets.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--x", type=float, nargs="+", default=[-0.01, 0.0, 0.01, 0.02, 0.03, 0.04])
    parser.add_argument("--y", type=float, nargs="+", default=[0.0])
    parser.add_argument("--z", type=float, nargs="+", default=[0.0])
    parser.add_argument("--speed-scale", type=float, default=2.0)
    parser.add_argument("--camera-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for x in args.x:
        for y in args.y:
            for z in args.z:
                offset = np.array([x, y, z], dtype=np.float64)
                try:
                    result = run_offset(args, offset)
                except Exception as exc:
                    result = {
                        "ok": False,
                        "offset": [float(v) for v in offset],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    best = sorted(results, key=score_result, reverse=True)
    print(json.dumps({"ok": True, "best": best[:5], "count": len(results)}, ensure_ascii=False, indent=2), flush=True)


def run_offset(args: argparse.Namespace, offset: np.ndarray) -> dict:
    collect.prepare_local_libero_config(ROOT)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    collect._BRUSH_FOLLOW_POS = None
    collect._BRUSH_WELD_ACTIVE = False
    collect._VERTICAL_GRASP_TARGET_ORI = None
    collect._EEF_TARGET_ORI = None

    bddl = (
        Path(get_libero_path("bddl_files"))
        / "libero_drawing"
        / "copy_the_left_board_pattern_to_the_right_board_with_the_soft_brush.bddl"
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        control_freq=20,
        horizon=1500,
    )
    env.seed(args.seed)
    env.reset()
    collect.place_initial_brush_on_blank_board(env)
    collect._VERTICAL_GRASP_TARGET_ORI = collect.get_site_xmat(env, "gripper0_vertical_brush_grasp_site")
    collect._EEF_TARGET_ORI = collect.get_eef_xmat(env)
    env.env.scripted_ink_mode = False

    zero = np.zeros(7, dtype=np.float32)
    for _ in range(20):
        env.step(zero)

    brush_pos = env.env.sim.data.body_xpos[env.env.obj_body_id["soft_brush_1"]].copy()
    brush_grasp_site = collect.get_site_xpos(env, "soft_brush_1_grasp_site")
    bucket_drop_site = collect.get_site_xpos(env, "drawing_water_bucket_1_drop_site")
    dummy_draw = np.asarray(env.env.get_drawing_waypoints()[::8], dtype=np.float64)
    grasp_offset = collect.current_grasp_site_offset(env)
    plan = collect.build_waypoint_plan(
        brush_pos,
        brush_grasp_site,
        bucket_drop_site,
        dummy_draw,
        grasp_offset,
        speed_scale=args.speed_scale,
        grasp_ee_offset=offset,
    )
    plan = [item for item in plan if item["phase"] in {"approach_brush", "descend_to_grasp", "settle_on_grasp", "close_gripper", "hold_closed", "lift_brush"}]

    holding = False
    last_ee = None
    last_tip = None
    rows = []
    for item in plan:
        for sub_item in collect.interpolate_item(last_ee, last_tip, item):
            _, holding, _ = collect.execute_scripted_step(env, sub_item, holding, use_weld=True)
            if sub_item["phase"] in {"descend_to_grasp", "close_gripper", "hold_closed", "lift_brush"}:
                rows.append(frame_metrics(env, sub_item["phase"]))
            if collect._BRUSH_WELD_ACTIVE and sub_item["phase"] == "lift_brush":
                break
        last_ee = np.asarray(item["ee"], dtype=np.float64)
        if np.all(np.isfinite(item["tip"])):
            last_tip = np.asarray(item["tip"], dtype=np.float64)
        if collect._BRUSH_WELD_ACTIVE:
            break
    env.close()

    summary = summarize(rows)
    return {
        "ok": True,
        "offset": [float(v) for v in offset],
        "weld_active": bool(collect._BRUSH_WELD_ACTIVE),
        "summary": summary,
    }


def frame_metrics(env, phase: str) -> dict:
    sim = env.sim
    model = sim.model
    data = sim.data
    gripper_site = collect.get_site_xpos(env, "gripper0_vertical_brush_grasp_site")
    brush_site = collect.get_site_xpos(env, "soft_brush_1_grasp_site")
    finger = 0
    hand = 0
    for idx in range(int(data.ncon)):
        contact = data.contact[idx]
        name1 = model.geom_id2name(contact.geom1) or ""
        name2 = model.geom_id2name(contact.geom2) or ""
        pair = name1 + " " + name2
        if "gripper0_finger" in pair and "soft_brush_1" in pair:
            finger += 1
        if "gripper0_hand" in pair and "soft_brush_1" in pair:
            hand += 1
    return {
        "phase": phase,
        "site_distance_m": float(np.linalg.norm(gripper_site - brush_site)),
        "finger_contacts": finger,
        "hand_contacts": hand,
        "weld_active": bool(collect._BRUSH_WELD_ACTIVE),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {
        "frames": len(rows),
        "min_site_distance_m": float(min(row["site_distance_m"] for row in rows)),
        "finger_contact_frames": int(sum(row["finger_contacts"] > 0 for row in rows)),
        "hand_contact_frames": int(sum(row["hand_contacts"] > 0 for row in rows)),
        "weld_frames": int(sum(row["weld_active"] for row in rows)),
    }


def score_result(result: dict) -> tuple:
    summary = result.get("summary") or {}
    return (
        int(result.get("weld_active", False)),
        int(summary.get("finger_contact_frames", 0)),
        -int(summary.get("hand_contact_frames", 0)),
        -float(summary.get("min_site_distance_m", 1e9)),
    )


if __name__ == "__main__":
    main()
