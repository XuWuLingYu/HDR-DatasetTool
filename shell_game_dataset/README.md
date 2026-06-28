# Shell Game Dataset

本目录用于生成“三仙归洞”数据集。

## Setting

- 白色背景
- 三个槽位：左 / 中 / 右
- 初始时红球出现在左中右之一
- 接着三个黑色杯子从上方落下，盖住三个位置
- 然后执行 `N` 次随机交换，每次交换两个杯子
- 交换动画中一个杯子走上方弧线，另一个走下方弧线，避免重叠
- `condition video` 只展示到交换结束
- `target video` 从最终三个杯子的位置开始，最后把真正盖住红球的杯子抬起，露出红球

## Train

默认训练集为 `3000` 条：

- `N=1`: `600`
- `N=2`: `600`
- `N=3`: `600`
- `N=4`: `600`
- `N=5`: `600`

运行：

```bash
python shell_game_dataset/generate_shell_game_videos.py \
  --num-videos 3000 \
  --swap-values 1,2,3,4,5 \
  --swap-counts 600,600,600,600,600 \
  --output-base-dir shell_game_dataset/train \
  --workers 8 \
  --overwrite
```

## Test

默认测试集为 `50` 条：

- `N=1`: `10`
- `N=2`: `10`
- `N=3`: `10`
- `N=4`: `10`
- `N=5`: `10`

运行：

```bash
python shell_game_dataset/generate_shell_game_50_test.py \
  --output-base-dir shell_game_dataset/test \
  --overwrite
```

## Outputs

每套数据会包含：

- `condition_videos/`
- `target_videos/`
- `npy/`
- `metadata.jsonl`

结构化数据里包含：

- 初始红球槽位
- 交换序列
- 最终红球槽位
- 获胜杯 id
