#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose physical brush grasp contacts for a drawing-copy episode.")
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--camera-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_local_libero_config(ROOT)
    from libero.libero.envs import OffScreenRenderEnv

    hdf5 = args.episode_dir / "episode.hdf5"
    if not hdf5.exists():
        raise SystemExit(f"missing {hdf5}")
    with h5py.File(hdf5, "r") as f:
        states = f["sim_states"][()]
        phases = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["phases"][()]]
        weld_active = f["weld_active"][()] if "weld_active" in f else np.zeros(len(phases), dtype=bool)
        actions = f["actions"][()]
        bddl = f.attrs["bddl"]

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        camera_names=["agentview"],
        control_freq=20,
        horizon=5000,
    )
    env.reset()
    model = env.env.sim.model
    data = env.env.sim.data

    rows = []
    for idx, phase in enumerate(phases):
        if phase not in {"descend_to_grasp", "close_gripper", "hold_closed", "lift_brush"}:
            continue
        env.set_state(states[idx])
        env.env.sim.forward()
        env._update_observables(force=True)
        row = frame_metrics(model, data, idx, phase, bool(weld_active[idx]), float(actions[idx, -1]))
        rows.append(row)
        if len(rows) >= args.max_rows:
            break
    env.close()

    summary = summarize(rows)
    print(json.dumps({"ok": True, "episode_dir": str(args.episode_dir.resolve()), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2))


def frame_metrics(model, data, idx: int, phase: str, weld_active: bool, gripper_cmd: float) -> dict:
    gripper_site = site_xpos(model, data, "gripper0_vertical_brush_grasp_site")
    brush_site = site_xpos(model, data, "soft_brush_1_grasp_site")
    finger_contacts = []
    hand_contacts = []
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        name1 = model.geom_id2name(contact.geom1) or ""
        name2 = model.geom_id2name(contact.geom2) or ""
        pair = name1 + " " + name2
        item = {"geom1": name1, "geom2": name2, "dist": float(contact.dist)}
        if "gripper0_finger" in pair and "soft_brush_1" in pair:
            finger_contacts.append(item)
        if "gripper0_hand" in pair and "soft_brush_1" in pair:
            hand_contacts.append(item)
    geom_positions = {}
    for name in [
        "gripper0_finger1_pad_collision",
        "gripper0_finger2_pad_collision",
        "gripper0_hand_collision",
        "soft_brush_1_grasp_sleeve",
        "soft_brush_1_handle",
    ]:
        try:
            geom_id = model.geom_name2id(name)
        except Exception:
            continue
        geom_positions[name] = [float(x) for x in data.geom_xpos[geom_id]]
    return {
        "frame": idx,
        "phase": phase,
        "weld_active": weld_active,
        "gripper_cmd": gripper_cmd,
        "site_distance_m": float(np.linalg.norm(gripper_site - brush_site)),
        "gripper_site": [float(x) for x in gripper_site],
        "brush_site": [float(x) for x in brush_site],
        "finger_brush_contact_count": len(finger_contacts),
        "hand_brush_contact_count": len(hand_contacts),
        "finger_brush_contacts": finger_contacts,
        "hand_brush_contacts": hand_contacts,
        "geom_positions": geom_positions,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    site_distances = [row["site_distance_m"] for row in rows]
    finger_counts = [row["finger_brush_contact_count"] for row in rows]
    hand_counts = [row["hand_brush_contact_count"] for row in rows]
    return {
        "frames": len(rows),
        "min_site_distance_m": float(min(site_distances)),
        "finger_contact_frames": int(sum(count > 0 for count in finger_counts)),
        "hand_contact_frames": int(sum(count > 0 for count in hand_counts)),
        "weld_frames": int(sum(row["weld_active"] for row in rows)),
        "best_site_frame": rows[int(np.argmin(site_distances))],
    }


def site_xpos(model, data, name: str) -> np.ndarray:
    return np.asarray(data.site_xpos[model.site_name2id(name)], dtype=np.float64).copy()


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


if __name__ == "__main__":
    main()
