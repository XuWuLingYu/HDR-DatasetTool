# Shell Game Mini Dataset

本目录用于生成 `shell_game_mini` 数据集。

## Setting

- 白色背景
- 三个槽位：左 / 中 / 右
- 初始时红球出现在左中右之一
- 接着三个黑色杯子从上方落下，盖住三个位置
- 杯子落下后不再交换位置
- `ref video` 继续保持遮挡状态，并额外等待 `1 / 2 / 3 / 4 / 5` 秒
- `target video` 从三个杯子已经盖住红球的状态开始，立即抬起正确杯子，露出红球

这个版本只测试视频模型在遮挡后的记忆能力，不测试轨迹跟踪。

## Train

默认训练集为 `3000` 条：

- `1s`: `600`
- `2s`: `600`
- `3s`: `600`
- `4s`: `600`
- `5s`: `600`

运行：

```bash
python shell_game_mini_dataset/generate_shell_game_mini_videos.py \
  --num-videos 3000 \
  --wait-values 1,2,3,4,5 \
  --wait-counts 600,600,600,600,600 \
  --output-base-dir outputs/shell_game_mini/train \
  --workers 8 \
  --overwrite
```

## Test

默认测试集为 `50` 条：

- `1s`: `10`
- `2s`: `10`
- `3s`: `10`
- `4s`: `10`
- `5s`: `10`

运行：

```bash
python shell_game_mini_dataset/generate_shell_game_mini_50_test.py \
  --output-base-dir outputs/shell_game_mini/test \
  --overwrite
```

## Outputs

每套数据会包含：

- `ref_videos/`
- `target_videos/`
- `npy/`
- `metadata.jsonl`

结构化数据里包含：

- 初始红球槽位
- 最终红球槽位
- 正确杯子 id
- 遮挡等待秒数
- 遮挡等待帧数
