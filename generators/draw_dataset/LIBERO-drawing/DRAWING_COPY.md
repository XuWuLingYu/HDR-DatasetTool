# LIBERO Drawing Copy Data Collection

This standalone LIBERO clone adds a `libero_drawing` benchmark with one task:

`copy_the_left_board_pattern_to_the_right_board_with_the_soft_brush`

Scene contents:

- Two flat tabletop drawing boards laid out left/right in the official LIBERO
  `agentview`. The left board contains a random one-stroke reference drawing.
- The right flat board receives visible MuJoCo ink geometry during drawing.
- Soft brush with a dark flexible-looking tip.
- Test-tube-rack-like brush rack holding the brush upright near the side of the
  center line between the two boards.
- Mounted Panda robot from LIBERO / robosuite.
- The physical weld path requires the robot to close around the brush before
  the brush can move. If the grasp misses, the episode fails verification
  instead of being corrected by script.

## One-Command Collection

Use a Python environment that has `robosuite`, `mujoco`, `h5py`, `imageio`, `opencv-python`, `Pillow`, and `PyYAML`.

Example using the local fastwam environment:

```bash
SEED=0 CAMERA_SIZE=256 STRIDE=3 FPS=20 RECORD_EVERY=3 \
  bash scripts/run_drawing_copy_collection.sh outputs/drawing_copy_episode
```

Strict physical mode:

```bash
USE_WELD=1 SEED=7 CAMERA_SIZE=128 STRIDE=8 FPS=20 RECORD_EVERY=4 \
  bash scripts/run_drawing_copy_collection.sh outputs/drawing_copy_physical_episode
```

The output directory contains:

- `episode.mp4`: full operation video.
- `episode.hdf5`: action trajectory and auxiliary arrays.
- `preview.jpg`: sampled frame grid for quick visual inspection.
- `frames/frame_*.jpg`: every frame extracted as an image.
- `contact_sheet_full.jpg`: contact sheet containing every frame.
- `summary.json`: paths and rollout metadata.

For VLA action learning, use the current action-aligned collector output. It
records one video frame for every real `env.step(action)` at `control_fps`.
The HDF5 invariant is:

```text
frames[t] = observation after applying actions[t]
```

The episode also stores `initial_sim_state`, `phases`, and `weld_active`, so an
episode can be replayed from the same simulator state. Older rows generated
before this alignment change saved only the final action of some internal servo
bursts and should not be used for strict action replay.

The wrapper runs `scripts/verify_drawing_copy_episode.py` after collection and
fails if the mp4, HDF5, extracted frames, smoothness, layout, or visual
progression are inconsistent. Videos are written with PyAV / H.264. The verifier
also checks `brush_tilt_degrees` so unstable grasp rollouts are rejected instead
of silently entering the dataset.

For `--use-weld` / `USE_WELD=1`, the verifier also checks that:

- `scripted_ink_mode` is false.
- The brush barely moves after settle and before weld activation.
- A gripper finger collision geom contacts the brush grasp sleeve/handle near
  weld activation; hand-only contact is not accepted as a grasp.
- The brush moves after weld activation.
- The right board receives real ink from the simulated brush tip.
- The draw-phase brush tip is in a physical contact range relative to the board.

## Batch Video Dataset

For training data, use the batch builder. It launches multiple independent
workers, runs the single-episode collector and verifier for every sample, copies
only the verified MP4s into `videos/`, and writes training metadata to
`metadata.jsonl` with only the fixed prompt and video path.

Small smoke run:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/build_drawing_copy_video_dataset.py \
  --out outputs/drawing_copy_batch_smoke \
  --num-samples 4 \
  --workers 2 \
  --camera-size 96 \
  --stride 10 \
  --record-every 5 \
  --speed-scale 1.6 \
  --overwrite
```

Action-aligned LeRobot smoke run:

```bash
/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/RICL/envs/ricl/bin/python \
  scripts/build_drawing_copy_lerobot_dataset.py \
  --out outputs/drawing_copy_lerobot_aligned_smoke \
  --repo-id local/drawing_copy_lerobot_aligned_smoke \
  --num-samples 1 \
  --workers 1 \
  --seed-start 0 \
  --camera-size 128 \
  --fps 16 \
  --control-fps 16 \
  --stride 1 \
  --speed-scale 0.08 \
  --horizon 9000 \
  --max-frames 2200 \
  --max-draw-frames 1200 \
  --keep-episodes \
  --overwrite
```

The LeRobot parquet contains the aligned `action` tensor. The builder also
writes lightweight exact-replay sidecars under `replay/episode_XXXXXX.npz`
containing `actions`, `phases`, `initial_sim_state`, and `seed`.

Exact replay:

```bash
/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/RICL/envs/ricl/bin/python \
  scripts/eval_drawing_copy_vla.py \
  --out outputs/replay_check_episode000000 \
  --replay-actions outputs/drawing_copy_lerobot_aligned_smoke/replay/episode_000000.npz \
  --replay-fps 16 \
  --camera-size 128 \
  --overwrite
```

Replay match verification, when the source HDF5 is kept:

```bash
/mnt/world_foundational_model/wfm_ckp-fileset/qianzezhong/RICL/envs/ricl/bin/python \
  scripts/verify_drawing_replay_match.py \
  outputs/drawing_copy_lerobot_aligned_smoke/source_episodes/episode_000000.hdf5 \
  outputs/replay_check_episode000000/episode.hdf5
```

Strict physical smoke run:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/build_drawing_copy_video_dataset.py \
  --out outputs/drawing_copy_physical_batch_smoke \
  --num-samples 1 \
  --workers 1 \
  --camera-size 96 \
  --stride 8 \
  --record-every 4 \
  --speed-scale 2.0 \
  --horizon 5000 \
  --use-weld \
  --keep-episodes \
  --overwrite \
  --max-retries 20
```

Current validated physical smoke:

```text
outputs/drawing_copy_physical_contact_audit_smoke
```

The sample passed final dataset validation with `--require-use-weld`. Its
manifest includes `estimated_added_ink_pixels=405`,
`gripper_brush_contact_frames_near_attach=1`,
`pre_attach_max_brush_move_m=0.0`, and
`draw_tip_z_rel_mean_m=-0.00136`.

Strict physical mode is intentionally slower than scripted mode. A candidate
episode is rejected if the gripper misses the brush, if the brush moves before
the weld, if the brush hovers above the board, or if the target board receives
too little physical ink. Use multiple workers or shards for larger runs.

For grasp tuning, use the fast pickup-only sweep instead of full rollouts:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/sweep_drawing_copy_grasp_offsets.py \
  --seed 11 \
  --x -0.025 -0.02 -0.015 -0.01 -0.005 0.0 0.005 0.01 0.015 0.02 \
  --z 0.0
```

For a failed episode, inspect the physical grasp with:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/diagnose_drawing_copy_grasp.py outputs/drawing_copy_failed_episode
```

Fast scripted debug run:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/build_drawing_copy_video_dataset.py \
  --out outputs/drawing_copy_video_dataset_100k \
  --num-samples 100000 \
  --workers 8 \
  --camera-size 256 \
  --stride 3 \
  --record-every 3 \
  --speed-scale 2.0
```

This mode is useful for checking the pattern generator and video/data plumbing,
but it is not the physically truthful dataset path: the brush is helped by
script. For the current physical dataset, use strict physical mode.

Recommended strict physical run:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/build_drawing_copy_video_dataset.py \
  --out outputs/drawing_copy_physical_dataset \
  --num-samples 100 \
  --workers 8 \
  --camera-size 96 \
  --stride 8 \
  --record-every 4 \
  --speed-scale 2.0 \
  --horizon 5000 \
  --use-weld \
  --max-retries 20
```

Multi-machine sharded run:

```bash
# Machine 0
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/build_drawing_copy_video_dataset.py \
  --out outputs/drawing_copy_video_dataset_100k_shard00 \
  --num-samples 100000 \
  --num-shards 8 \
  --shard-index 0 \
  --workers 8 \
  --camera-size 256 \
  --stride 3 \
  --record-every 3 \
  --speed-scale 2.0
```

Run the same command on machines 1-7 with matching `--shard-index`. Each shard
keeps global ids and video names, e.g. shard 0 writes ids `0, 8, 16, ...`.
Use separate output directories per shard to avoid concurrent JSONL writes.

Merge shards after all machines finish:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/merge_drawing_copy_video_shards.py \
  --out outputs/drawing_copy_video_dataset_100k \
  --shards \
    outputs/drawing_copy_video_dataset_100k_shard00 \
    outputs/drawing_copy_video_dataset_100k_shard01 \
    outputs/drawing_copy_video_dataset_100k_shard02 \
    outputs/drawing_copy_video_dataset_100k_shard03 \
    outputs/drawing_copy_video_dataset_100k_shard04 \
    outputs/drawing_copy_video_dataset_100k_shard05 \
    outputs/drawing_copy_video_dataset_100k_shard06 \
    outputs/drawing_copy_video_dataset_100k_shard07 \
  --expected-count 100000 \
  --workers 8
```

For strict physical shards, add `--require-use-weld` during merge so unaudited
or scripted rows are rejected:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/merge_drawing_copy_video_shards.py \
  --out outputs/drawing_copy_physical_dataset \
  --shards outputs/drawing_copy_physical_dataset_shard00 outputs/drawing_copy_physical_dataset_shard01 \
  --expected-count 100 \
  --workers 8 \
  --require-use-weld
```

Each metadata line has:

```json
{"prompt": "用机器人复制左边的图案", "video_path": "/abs/path/videos/drawing_copy_000000.mp4"}
```

`manifest.jsonl` stores debug and audit fields: `id`, `seed`, frame count, max
brush tilt, prompt, video path, and verifier-derived physical audit fields. In
strict physical mode, legacy rows without `estimated_added_ink_pixels` and
`physics_metrics` are treated as incomplete and regenerated on resume. Both
files are sorted by `id` after each successful sample. Failed samples are
written to `failures.jsonl`, and the script exits nonzero if any sample fails.
The batch builder runs a final dataset-level validator by default; pass
`--no-final-validate` only for debugging. Resume uses manifest plus the expected
video file for strict physical runs, not orphan MP4s alone, so a partially
copied video or legacy unaudited row from an interrupted run will be
regenerated. Video copies are written through temporary files and atomically
renamed into place.

Manual final validation:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/validate_drawing_copy_video_dataset.py \
  outputs/drawing_copy_video_dataset_100k \
  --expected-count 100000 \
  --workers 8
```

For strict physical datasets, require the manifest to prove weld mode:

```bash
/mnt/world_foundational_model/wfm_envs-fileset/qianzezhong/fastwam/bin/python \
  scripts/validate_drawing_copy_video_dataset.py \
  outputs/drawing_copy_physical_dataset \
  --expected-count 100 \
  --workers 8 \
  --require-use-weld
```

The validator checks that `metadata.jsonl` has exactly `prompt` and
`video_path`, every path is absolute and unique, every MP4 exists and decodes,
all videos are square, the expected count matches, and there are no extra MP4s
under `videos/` that are missing from metadata. It also checks manifest
consistency: ids are unique and sorted, video names match ids, manifest and
metadata rows point to the same videos, and final merged datasets use contiguous
ids when `--require-contiguous-ids` is set. Individual shards intentionally do
not require contiguous ids. Under `--require-use-weld`, it also requires
`use_weld=true`, `scripted_ink_mode=false`, enough estimated physical ink, and
physical replay metrics proving the brush barely moved before attach, moved
after attach, and drew with the tip in board-contact range. It also rejects
rollouts whose replayed brush-tip XY error exceeds 6 cm, because such videos no
longer faithfully copy the target curve.

## Current Stability Notes

The one-sample strict physical smoke is passing:

```text
outputs/drawing_copy_physical_contact_audit_smoke
```

A 3-sample strict physical stability run now passes:

```text
outputs/drawing_copy_physical_adaptive_tip_batch3
```

This uses adaptive tip control: after the brush is physically grasped and welded,
the collector measures the actual end-effector-to-brush-tip offset and uses that
measured offset to place the real simulated tip on the target curve. This avoids
depending on a single fixed draw height across different grasp outcomes.

Do not jump straight to a large strict physical job. The next practical gates
are a 10-sample run and then a 100-sample run, both with `--require-use-weld`
and the XY gate enabled.

## HDF5 Layout

Datasets:

- `frames`: `uint8`, shape `[T, H, W, 3]`.
- `actions`: `float32`, shape `[T, 7]`, schema `[target_x, target_y, target_z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]`.
- `ee_targets`: `float32`, shape `[T, 3]`.
- `brush_tip_targets`: `float32`, shape `[T, 3]`.
- `brush_tilt_degrees`: `float32`, shape `[T]`; angle between the brush axis and
  world vertical.
- `weld_active`: `bool`, shape `[T]`; whether the physical brush-gripper weld
  was active for the recorded frame.
- `grippers`: `float32`, shape `[T]`.
- `phases`: UTF-8 phase labels.
- `sim_states`: MuJoCo flattened state per frame.
- `reference_curve`: normalized left-board reference curve, shape `[N, 2]`.
- `drawing_waypoints`: world-space right-board drawing waypoints, shape `[M, 3]`.

There are two collection modes:

1. Scripted video mode, the default. The brush follows the achieved Panda
   end-effector pose and the planned tip path is used for ink. This is fast and
   useful for generating visual warmup videos.
2. Strict physical mode, enabled by `--use-weld` or `USE_WELD=1`. The brush is
   not teleported to the gripper, and ink is not forced from the planned path.
   The weld activates only after the gripper reaches the brush. If the grasp
   misses, the episode fails and the batch builder tries another seed.

In physical mode, `teleport` in `summary.json` only means the scripted robot
waypoints are interpolated; it does not mean the brush or ink is teleported.
`scripted_ink_mode` must be `false`.
