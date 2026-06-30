# Hanoi Dataset

本目录用于生成三柱汉诺塔视频数据集。

## Setting

- 三个杆子，平面视图
- 目标状态：所有矩形盘子都在最右侧杆
- 训练集初始状态：每个盘子按从大到小的顺序，随机堆放到左杆或中杆
- 测试集包含两部分：`in_domain` 只从左杆/中杆随机堆放，`ood` 从左杆/中杆/右杆随机堆放
- 初始状态始终合法，不会出现大盘压在小盘上
- `ood` 子集默认保证初始状态里实际出现右杆，从而和 `in_domain` 明确区分
- 对任意初始状态，自动求解到右杆的最短合法解序列，再按解序列逐步渲染
- 每一步移动都走“上升 -> 水平 -> 下降”的折线轨迹，并按弧长匀速采样
- 盘子颜色固定为：小到大依次是绿、蓝、粉、棕、紫
- 默认 `fps=10`

## Train

默认训练集为 `3000` 条：

- `2` 个盘子：`750`
- `3` 个盘子：`750`
- `4` 个盘子：`750`
- `5` 个盘子：`750`

运行：

```bash
python generators/classic/hanoi/generate_hanoi_videos.py \
  --num-videos 3000 \
  --disk-values 2,3,4,5 \
  --disk-counts 750,750,750,750 \
  --initial-rods 0,1 \
  --output-dir outputs/hanoi/videos \
  --struct-dir outputs/hanoi/struct \
  --workers 8 \
  --overwrite
```

## Test

默认测试集为 `80` 条，分成两部分：

- `in_domain`：`40` 条，初始状态只会出现在左杆/中杆
- `ood`：`40` 条，初始状态允许出现在左杆/中杆/右杆，并默认保证右杆实际出现

每部分内部按组输出：

- `2` 个盘子：`10`
- `3` 个盘子：`10`
- `4` 个盘子：`10`
- `5` 个盘子：`10`

默认索引范围：

- `0-39`：`in_domain`
- `40-79`：`ood`

运行：

```bash
python generators/classic/hanoi/generate_hanoi_50_test.py \
  --in-domain-initial-rods 0,1 \
  --ood-initial-rods 0,1,2 \
  --output-dir outputs/hanoi/test \
  --struct-dir outputs/hanoi/test_npy \
  --overwrite
```

## Outputs

每条样本会保存：

- `mp4` 视频
- `metadata.jsonl`
- 对应 `npy` 结构化文件

`metadata.jsonl` 额外包含：

- `domain_split`：`in_domain` 或 `ood`
- `allowed_initial_rods`
- `required_initial_rods_present`

`npy` 中包含：

- 盘子数量
- 解序列 `move_sequence`
- 每步状态 `states`
- 每个盘子的颜色
- 杆位置与盘子宽高等几何信息
