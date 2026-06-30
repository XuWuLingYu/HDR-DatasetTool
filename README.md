# HDR DatasetTool

Utilities for generating and evaluating video reasoning datasets.

## Layout

- `generators/`: dataset/video generation entry points.
- `evaluators/`: standalone and mixed-task evaluation entry points.
- `configs/`: reusable configuration files.
- `docs/`: task-specific evaluation notes.
- `outputs/`: recommended local output root. This directory is git-ignored.

## Classic Generators

Run commands from the repository root.

```bash
python generators/classic/maze_videos.py --num-videos 100 --output-dir outputs/maze/videos --overwrite
python generators/classic/maze_50_test.py --output-dir outputs/maze/test_videos --struct-dir outputs/maze/test_struct
python generators/classic/push_block_maze_videos.py --num-videos 100 --output-dir outputs/maze/push_blocks --overwrite

python generators/classic/bounce_ball/generate_bounce_ball_videos.py --num-videos 100 --output-dir outputs/bounce_ball/videos --struct-dir outputs/bounce_ball/struct --overwrite
python generators/classic/hanoi/generate_hanoi_videos.py --num-videos 100 --output-dir outputs/hanoi/hanoi_videos --struct-dir outputs/hanoi/hanoi_npy --overwrite
python generators/classic/shape_return/generate_shape_return_videos.py --num-videos 100 --output-dir outputs/shape_return/videos --struct-dir outputs/shape_return/struct --overwrite

python generators/classic/shell_game/generate_shell_game_videos.py --num-videos 100 --output-base-dir outputs/shell_game/train --overwrite
python generators/classic/shell_game_easy/generate_shell_game_easy_videos.py --num-videos 100 --output-base-dir outputs/shell_game_easy/train --overwrite
python generators/classic/shell_game_mini/generate_shell_game_mini_videos.py --num-videos 100 --output-base-dir outputs/shell_game_mini/train --overwrite
```

## Diffusion Reasoning Generators

The diffusion reasoning tasks imported from `diffusion-reasoning-dataset` are available in this repo:

- `one_line_drawing`
- `sliding_puzzle`
- `sokoban`
- `water_pour`

```bash
python generators/diffusion_reasoning/generate_dataset.py \
  --config configs/diffusion_reasoning/env_configs.yaml \
  --env one_line_drawing \
  --output-root outputs/diffusion_reasoning \
  --levels 2 \
  --count-override 2 \
  --workers 1
```

Remove `--count-override` for full generation.

## Draw Dataset

The original draw/sketch-copy tools live under `generators/draw_dataset/`.

```bash
python generators/draw_dataset/tools/generate_synthetic_oneline.py \
  --config generators/draw_dataset/configs/one_stroke_copy.yaml \
  --out outputs/draw_dataset/synthetic_oneline_copy \
  --total 10 \
  --overwrite
```

Some draw tools still depend on the upstream `sim_draw_dataset` package layout. The `diffusion_reasoning` `one_line_drawing` generator above is self-contained in this repository.

## Evaluators

Classic evaluators:

```bash
python evaluators/classic/maze_videos.py --pred-dir outputs/predictions/maze --struct-dir outputs/maze/test_struct
python evaluators/classic/hanoi.py --dataset-dir outputs/hanoi --split hanoi_test --pred-dir outputs/predictions/hanoi
python evaluators/classic/shell_game_easy.py --dataset-dir outputs/shell_game_easy --pred-dir outputs/predictions/shell_game_easy
python evaluators/classic/shell_game_mini.py --dataset-dir outputs/shell_game_mini --pred-dir outputs/predictions/shell_game_mini
```

Diffusion reasoning evaluator:

```bash
python evaluators/diffusion_reasoning/run_evaluation.py \
  --env one_line_drawing \
  --gt-root outputs/diffusion_reasoning \
  --pred-root outputs/predictions/one_line_drawing \
  --levels 2 \
  --output outputs/eval/one_line_drawing_level2.json
```

Mixed reasoning evaluator:

```bash
python evaluators/mixed_reasoning.py \
  --pred-dir outputs/predictions/mixed_reasoning \
  --test-jsonl outputs/mixed_reasoning/test.jsonl \
  --full-test-jsonl outputs/mixed_reasoning/test_full.jsonl \
  --manifest-jsonl outputs/mixed_reasoning/manifest.jsonl \
  --maze-struct-dir outputs/maze/test_struct \
  --hanoi-dataset-dir outputs/hanoi \
  --output-json outputs/mixed_reasoning/mixed_reasoning_eval.json
```

The mixed evaluator now imports `one_line_drawing`, `sliding_puzzle`, `sokoban`, and `water_pour` evaluators from `evaluators/diffusion_reasoning/`; it no longer depends on an external absolute-path checkout.

## Dependencies

Common dependencies:

```bash
python -m pip install numpy imageio imageio-ffmpeg pillow networkx pyyaml opencv-python tqdm av
```

Task-specific tools may need extra packages documented in their subdirectory README files.
