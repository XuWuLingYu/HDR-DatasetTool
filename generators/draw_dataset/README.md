# Sim Drawing Video Dataset

This repo generates robot-copy sketch videos for video warmup learning. The
default output is video-only:

```text
dataset/
  shard_0000/
    video_000000.mp4
    video_000001.mp4
  shard_0001/
    video_001000.mp4
```

No actions, states, previews, targets, or per-episode metadata are written by
default. Videos are written with PyAV.

## Procedural Videos

```bash
python generate_dataset.py --config configs/default.yaml --out dataset --episodes 1000 --seed 7 --overwrite
```

The procedural generator is useful for smoke tests and domain randomization, but
it is not the best source for 100k meaningful sketches.

## 100k Meaningful Sketches

Use Google QuickDraw as the main source. It provides human-drawn vector strokes
in NDJSON format, so it maps cleanly into this pipeline's stroke representation.
Download selected `*.ndjson` category files, then run:

```bash
python tools/generate_from_quickdraw.py \
  --config configs/default.yaml \
  --quickdraw-dir /path/to/quickdraw_ndjson \
  --categories cat dog bicycle car airplane house tree chair cup apple face clock flower fish bird \
  --out outputs/draw_dataset/quickdraw_videos \
  --total 100000 \
  --seed 7 \
  --overwrite
```

Recommended strategy:

- Pick 100-200 concrete object categories.
- Generate 500-1000 videos per category.
- Keep only recognized QuickDraw samples.
- Store videos in shards of 1000 files, as configured by `output.shard_size`.
- If you need traceability later, add `--write-manifest`; otherwise keep
  video-only output.

## Dataset Options

Best fit:

- Google QuickDraw: large-scale, meaningful categories, vector stroke order,
  easy conversion, enough data for 100k+ videos.

Possible but less convenient:

- TU Berlin Sketch Dataset: meaningful sketches, but typically static raster
  images rather than original time-ordered strokes.
- Sketchy Database: object sketches paired with photos, useful semantically,
  but not as direct for reproducing stroke-by-stroke drawing videos.

For this simulator, vector stroke order matters. QuickDraw is the practical
default because it directly gives sequences of pen strokes.

## Output Details

The rendered video shows:

- left: reference whiteboard with the full black sketch
- right: robot-side whiteboard being drawn over time
- robot visual: screen-normal end-effector approximated by a black-white
  concentric circle with a rectangular rod above it

Edit [configs/default.yaml](configs/default.yaml) to control resolution, FPS,
stroke speed, robot visual size, and sharding.
