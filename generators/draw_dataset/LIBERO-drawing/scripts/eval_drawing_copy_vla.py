#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import collect_drawing_copy_dataset as collect

PROMPT = "Copy the pattern on the left board with the robot."


class ReplayPolicy:
    def __init__(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"replay actions must have shape [T, 7], got {actions.shape}")
        self.actions = actions

    def reset(self, _: dict[str, Any]) -> None:
        return None

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        step = int(obs["step"])
        if step >= len(self.actions):
            return self.actions[-1].copy()
        return self.actions[step].copy()


class ModulePolicy:
    def __init__(self, module_path: Path, kwargs: dict[str, Any]):
        spec = importlib.util.spec_from_file_location("drawing_copy_vla_policy", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not import policy module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "create_policy"):
            self.policy = module.create_policy(**kwargs)
        elif hasattr(module, "Policy"):
            self.policy = module.Policy(**kwargs)
        else:
            raise RuntimeError(f"{module_path} must define create_policy(**kwargs) or Policy(**kwargs)")

    def reset(self, context: dict[str, Any]) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset(context)

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        if hasattr(self.policy, "act"):
            action = self.policy.act(obs)
        elif callable(self.policy):
            action = self.policy(obs)
        else:
            raise RuntimeError("policy must be callable or expose act(obs)")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise RuntimeError(f"policy action must have shape [7], got {action.shape}")
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out a VLA policy, or replay LeRobot actions, in the drawing-copy LIBERO task."
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/drawing_copy_vla_replay_episode"))
    parser.add_argument("--seed", type=int, default=None, help="Drawing/environment seed. Defaults to manifest seed for replay.")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--control-fps", type=int, default=16)
    parser.add_argument(
        "--replay-fps",
        type=int,
        default=None,
        help="For replay, execute exactly one saved action per env step at this control/video FPS.",
    )
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means use replay length or horizon.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Repeat every policy / replay action for N env.step calls before recording one frame.",
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--policy-module", type=Path, default=None)
    parser.add_argument("--policy-kwargs", default="{}", help="JSON object passed to create_policy / Policy.")
    parser.add_argument("--replay-actions", type=Path, default=None, help=".npy, .npz, .hdf5, or parquet action file.")
    parser.add_argument("--replay-hdf5", type=Path, default=None, help="Exact replay from a collected drawing-copy episode.hdf5.")
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("outputs/drawing_copy_lerobot_5k_v2/lerobot"),
        help="LeRobot dataset root used by --episode-index replay.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/drawing_copy_lerobot_5k_v2/manifest.jsonl"),
        help="Manifest used to recover replay seed from episode_index.",
    )
    parser.add_argument("--no-auto-weld", dest="auto_weld", action="store_false")
    parser.set_defaults(auto_weld=True)
    parser.add_argument("--run-verify", action="store_true", help="Run the strict scripted verifier after rollout.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replay_fps is not None:
        args.fps = int(args.replay_fps)
        args.control_fps = int(args.replay_fps)
        args.action_repeat = 1
    out_dir = args.out.resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    replay_actions = None
    replay_source = None
    replay_hdf5 = None
    replay_phases = None
    replay_initial_state = None
    if args.policy_module is None:
        if args.replay_hdf5 is not None:
            replay_hdf5 = load_replay_hdf5(args.replay_hdf5)
            replay_actions = replay_hdf5["actions"]
            replay_source = str(args.replay_hdf5.resolve())
            replay_phases = replay_hdf5.get("phases")
            replay_initial_state = replay_hdf5.get("initial_sim_state")
            if args.seed is None and replay_hdf5.get("seed") is not None:
                args.seed = int(replay_hdf5["seed"])
        elif args.replay_actions is not None and args.replay_actions.suffix == ".npz":
            replay_hdf5 = load_replay_npz(args.replay_actions)
            replay_actions = replay_hdf5["actions"]
            replay_source = str(args.replay_actions.resolve())
            replay_phases = replay_hdf5.get("phases")
            replay_initial_state = replay_hdf5.get("initial_sim_state")
            if args.seed is None and replay_hdf5.get("seed") is not None:
                args.seed = int(replay_hdf5["seed"])
        else:
            action_path = args.replay_actions or lerobot_episode_parquet(args.lerobot_root, args.episode_index)
            replay_actions = load_actions(action_path)
            replay_source = str(action_path.resolve())
        policy = ReplayPolicy(replay_actions)
    else:
        kwargs = json.loads(args.policy_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--policy-kwargs must decode to a JSON object")
        policy = ModulePolicy(args.policy_module.resolve(), kwargs)

    seed = int(args.seed if args.seed is not None else seed_for_episode(args.manifest, args.episode_index))
    result = rollout(args, policy, seed, replay_actions, replay_source, out_dir, replay_phases, replay_initial_state)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def rollout(
    args: argparse.Namespace,
    policy: Any,
    seed: int,
    replay_actions: np.ndarray | None,
    replay_source: str | None,
    out_dir: Path,
    replay_phases: list[str] | None = None,
    replay_initial_state: np.ndarray | None = None,
) -> dict[str, Any]:
    collect._BRUSH_WELD_ACTIVE = False
    collect._BRUSH_FOLLOW_POS = None
    collect._VERTICAL_GRASP_TARGET_ORI = None
    collect._EEF_TARGET_ORI = None
    collect.prepare_local_libero_config(ROOT)

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

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
    try:
        env.seed(seed)
        obs = env.reset()
        if hasattr(env.env, "set_drawing_seed"):
            env.env.set_drawing_seed(seed)
            env._update_observables(force=True)
            obs = env.env._get_observations()
        initial_brush_pose = None
        if replay_initial_state is not None:
            env.set_state(np.asarray(replay_initial_state, dtype=np.float64))
            env._update_observables(force=True)
            obs = env.env._get_observations()
        else:
            initial_brush_pose = collect.place_initial_brush_on_blank_board(env)
            for _ in range(8):
                collect.zero_brush_velocity(env)
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            for _ in range(20):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        policy.reset(
            {
                "prompt": args.prompt,
                "seed": seed,
                "action_schema": "6D robosuite ee_delta controller action + 1D gripper absolute command",
                "camera_size": args.camera_size,
                "control_fps": args.control_fps,
            }
        )
        max_steps = len(replay_actions) if replay_actions is not None else args.horizon
        if args.max_steps > 0:
            max_steps = min(max_steps, args.max_steps)

        frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        ee_pos: list[np.ndarray] = []
        brush_tip: list[np.ndarray] = []
        brush_tilt_degrees: list[float] = []
        weld_active: list[bool] = []
        sim_states: list[np.ndarray] = []
        exact_scripted_replay = replay_phases is not None and replay_initial_state is not None
        holding_brush = False

        for step in range(max_steps):
            phase = replay_phases[step] if replay_phases is not None and step < len(replay_phases) else "vla_rollout"
            action = np.asarray(policy.act(make_policy_observation(obs, step, args.prompt)), dtype=np.float32).reshape(-1)
            if action.shape != (7,):
                raise RuntimeError(f"policy action must have shape [7], got {action.shape}")
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            action[6] = float(np.clip(action[6], -1.0, 1.0))
            done = False
            for _ in range(max(1, int(args.action_repeat))):
                if args.auto_weld and exact_scripted_replay:
                    prepare_scripted_replay_step(env, phase)
                elif args.auto_weld:
                    update_physical_helpers(env, action, phase)
                obs, _, done, _ = env.step(action)
                if args.auto_weld and exact_scripted_replay:
                    holding_brush = finish_scripted_replay_step(env, phase, holding_brush)
                    env._update_observables(force=True)
                    obs = env.env._get_observations()
                elif args.auto_weld:
                    update_physical_helpers(env, action, phase)
                    env.env._post_process()
                    env._update_observables(force=True)
                    obs = env.env._get_observations()
                if done:
                    break

            frames.append(obs["agentview_image"][::-1].copy())
            wrist_frames.append(obs["robot0_eye_in_hand_image"][::-1].copy())
            actions.append(action.copy())
            ee_pos.append(np.asarray(obs.get("robot0_eef_pos", [np.nan, np.nan, np.nan]), dtype=np.float32))
            tip = env.env.brush_tip_world_pos() if hasattr(env.env, "brush_tip_world_pos") else np.full(3, np.nan)
            brush_tip.append(np.asarray(tip, dtype=np.float32))
            brush_tilt_degrees.append(float(collect.measure_brush_tilt_degrees(env)))
            weld_active.append(bool(collect._BRUSH_WELD_ACTIVE))
            sim_states.append(env.get_sim_state().copy())
            if done:
                break

        mp4_path = out_dir / "episode.mp4"
        wrist_mp4_path = out_dir / "episode_wrist.mp4"
        hdf5_path = out_dir / "episode.hdf5"
        preview_path = out_dir / "preview.jpg"
        frames_dir = out_dir / "frames"
        full_sheet_path = out_dir / "contact_sheet_full.jpg"

        collect.write_mp4_pyav(mp4_path, frames, fps=args.fps)
        collect.write_mp4_pyav(wrist_mp4_path, wrist_frames, fps=args.fps)
        collect.write_hdf5(
            hdf5_path,
            frames=np.asarray(frames, dtype=np.uint8),
            wrist_frames=np.asarray(wrist_frames, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.float32),
            ee_targets=np.asarray(ee_pos, dtype=np.float32),
            brush_tip_targets=np.asarray(brush_tip, dtype=np.float32),
            brush_tilt_degrees=np.asarray(brush_tilt_degrees, dtype=np.float32),
            grippers=np.asarray([a[6] for a in actions], dtype=np.float32),
            phases=(replay_phases[: len(actions)] if replay_phases is not None else ["vla_rollout"] * len(actions)),
            weld_active=np.asarray(weld_active, dtype=np.bool_),
            sim_states=np.asarray(sim_states, dtype=np.float64),
            reference_curve=env.env.reference_curve,
            reference_stroke_lengths=np.asarray([len(stroke) for stroke in env.env.reference_strokes], dtype=np.int32),
            drawing_waypoints=np.empty((0, 3), dtype=np.float32),
            drawing_stroke_lengths=np.empty((0,), dtype=np.int32),
            bddl=str(bddl),
            seed=seed,
            camera_size=args.camera_size,
            fps=args.fps,
            control_fps=args.control_fps,
            teleport=False,
            scripted_controller=False,
            use_weld=bool(args.auto_weld),
            scripted_ink_mode=False,
            prompt=args.prompt,
            action_schema="6D robosuite ee_delta controller action + 1D gripper absolute command",
            replay_source=replay_source or "",
            initial_brush_pose=json.dumps(initial_brush_pose, ensure_ascii=False),
            reference_sketch_type=str(getattr(env.env, "reference_sketch_type", "unknown")),
            num_reference_strokes=int(len(getattr(env.env, "reference_strokes", []))),
            reference_points=int(len(getattr(env.env, "reference_curve", []))),
            horizon=args.horizon,
            draw_segment_count=int(getattr(env.env, "_draw_segment_count", 0)),
            draw_canvas_dark_pixels=collect.count_draw_canvas_dark_pixels(env.env),
        )
        collect.write_preview(preview_path, frames)
        collect.write_frame_images(frames_dir, frames)
        collect.write_contact_sheet(full_sheet_path, frames)

        summary = {
            "ok": True,
            "mode": "replay" if replay_actions is not None else "policy",
            "out": str(out_dir),
            "mp4": str(mp4_path),
            "wrist_mp4": str(wrist_mp4_path),
            "hdf5": str(hdf5_path),
            "preview": str(preview_path),
            "frames_dir": str(frames_dir),
            "frames": len(frames),
            "seed": seed,
            "prompt": args.prompt,
            "replay_source": replay_source,
            "auto_weld": bool(args.auto_weld),
            "action_repeat": int(args.action_repeat),
            "exact_replay_initial_state": bool(replay_initial_state is not None),
            "weld_active_frames": int(np.count_nonzero(weld_active)),
            "draw_segment_count": int(getattr(env.env, "_draw_segment_count", 0)),
            "draw_canvas_dark_pixels": collect.count_draw_canvas_dark_pixels(env.env),
            "max_brush_tilt_degrees": float(np.nanmax(brush_tilt_degrees)) if brush_tilt_degrees else None,
            "action_min": np.asarray(actions, dtype=np.float32).min(axis=0).tolist() if actions else [],
            "action_max": np.asarray(actions, dtype=np.float32).max(axis=0).tolist() if actions else [],
        }
        if args.run_verify:
            verify = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "verify_drawing_copy_episode.py"), str(out_dir)],
                cwd=str(ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            summary["verify_returncode"] = int(verify.returncode)
            summary["verify_output"] = verify.stdout[-4000:]
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        env.close()


def make_policy_observation(obs: dict[str, Any], step: int, prompt: str) -> dict[str, Any]:
    return {
        "step": int(step),
        "prompt": prompt,
        "images": {
            "agentview": obs["agentview_image"][::-1].copy(),
            "wrist": obs["robot0_eye_in_hand_image"][::-1].copy(),
        },
        "proprio": {
            "eef_pos": np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float32).copy(),
            "gripper_qpos": np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32).copy(),
            "joint_pos": np.asarray(obs.get("robot0_joint_pos", []), dtype=np.float32).copy(),
        },
        "raw_obs": obs,
    }


def update_physical_helpers(env: Any, action: np.ndarray, phase: str = "vla_rollout") -> None:
    gripper = float(action[6])
    if phase == "release_in_bucket" and collect._BRUSH_WELD_ACTIVE:
        collect.release_brush_weld(env)
    elif gripper < -0.2 and collect._BRUSH_WELD_ACTIVE and phase not in {"draw", "touch_down", "move_to_bucket", "lower_into_bucket", "hold_bucket_center"}:
        collect.release_brush_weld(env)
    elif gripper > 0.2 and phase in {
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
        collect.attach_brush_weld(env, align_once=False)
    env.env.inking_enabled = bool(collect._BRUSH_WELD_ACTIVE and phase == "draw" and brush_tip_on_blank_board(env))


SCRIPTED_HOLDING_PHASES = {
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
}


def prepare_scripted_replay_step(env: Any, phase: str) -> None:
    env.env.inking_enabled = phase == "draw"


def finish_scripted_replay_step(env: Any, phase: str, holding_brush: bool) -> bool:
    if phase in SCRIPTED_HOLDING_PHASES:
        holding_brush = True
    if holding_brush:
        if phase == "release_in_bucket":
            collect.release_brush_weld(env)
            holding_brush = False
        else:
            holding_brush = collect.attach_brush_weld(env, align_once=False)
    return holding_brush


def brush_tip_on_blank_board(env: Any) -> bool:
    if "drawing_board_blank_1" not in env.env.obj_body_id:
        return False
    tip = np.asarray(env.env.brush_tip_world_pos(), dtype=np.float64)
    board_pos = np.asarray(env.env.sim.data.body_xpos[env.env.obj_body_id["drawing_board_blank_1"]], dtype=np.float64)
    rel = tip - board_pos
    uv = np.array([rel[0] / 0.38 + 0.5, rel[1] / 0.28 + 0.5], dtype=np.float64)
    z_rel = float(rel[2] - collect.BOARD_SURFACE_Z_FROM_BODY)
    return bool(0.04 <= uv[0] <= 0.96 and 0.04 <= uv[1] <= 0.96 and -0.020 <= z_rel <= 0.030)


def lerobot_episode_parquet(root: Path, episode_index: int) -> Path:
    path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        matches = sorted(root.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
        if matches:
            return matches[0]
    return path


def seed_for_episode(manifest: Path, episode_index: int) -> int:
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("episode_index", -1)) == int(episode_index):
                return int(row.get("seed", row.get("id", episode_index)))
    return int(episode_index)


def load_actions(path: Path) -> np.ndarray:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".npz":
        data = np.load(path)
        key = "actions" if "actions" in data else sorted(data.files)[0]
        return np.asarray(data[key], dtype=np.float32)
    if path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            key = "actions" if "actions" in f else "action"
            return np.asarray(f[key][()], dtype=np.float32)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["action"])
        return np.asarray(table["action"].to_pylist(), dtype=np.float32)
    raise ValueError(f"unsupported action file type: {path}")


def load_replay_hdf5(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        phases = None
        if "phases" in f:
            phases = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["phases"][()]]
        return {
            "actions": np.asarray(f["actions"][()], dtype=np.float32),
            "phases": phases,
            "initial_sim_state": np.asarray(f["initial_sim_state"][()], dtype=np.float64) if "initial_sim_state" in f else None,
            "seed": int(f.attrs["seed"]) if "seed" in f.attrs else None,
        }


def load_replay_npz(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    phases = data["phases"].astype(str).tolist() if "phases" in data else None
    initial = np.asarray(data["initial_sim_state"], dtype=np.float64) if "initial_sim_state" in data else None
    if initial is not None and initial.size == 0:
        initial = None
    return {
        "actions": np.asarray(data["actions"], dtype=np.float32),
        "phases": phases,
        "initial_sim_state": initial,
        "seed": int(np.asarray(data["seed"]).item()) if "seed" in data else None,
    }


if __name__ == "__main__":
    main()
