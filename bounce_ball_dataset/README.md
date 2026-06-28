# Bounce Ball Dataset

本目录用于生成“小球弹弹乐”数据集。

## Setting

- 白色背景，浅灰色方形边框场地
- 小球大小与颜色固定，默认黑色小球
- 首帧带箭头，表示初始运动方向
- 小球在场地内按镜面反射规则运动
- 难度由反弹次数控制
- 每条样本带 `5` 个候选位置，其中只有一个是正确终点
- 会过滤掉会被中途轨迹碰到的错误候选位置

## Train

默认训练集为 `3000` 条：

- `bounce_count=1`: `1000`
- `bounce_count=2`: `1000`
- `bounce_count=3`: `1000`

运行：

```bash
python bounce_ball_dataset/generate_bounce_ball_videos.py \
  --num-videos 3000 \
  --bounce-values 1,2,3 \
  --bounce-counts 1000,1000,1000 \
  --output-dir bounce_ball_dataset/bounce_ball_videos \
  --struct-dir bounce_ball_dataset/bounce_ball_npy \
  --workers 8 \
  --overwrite
```

## Test

默认测试集为 `50` 条，按组输出：

- `bounce_count=1`: `10`
- `bounce_count=2`: `10`
- `bounce_count=3`: `10`
- `bounce_count=4`: `10`
- `bounce_count=5`: `10`

运行：

```bash
python bounce_ball_dataset/generate_bounce_ball_50_test.py \
  --output-dir bounce_ball_dataset/bounce_ball_test \
  --struct-dir bounce_ball_dataset/bounce_ball_test_npy \
  --overwrite
```

## Outputs

每条样本会保存：

- `mp4` 视频
- `metadata.jsonl`
- 对应 `npy` 结构化文件

`npy` 中包含：

- `bounce_count`
- `candidate_positions`
- `correct_candidate_index`
- `path_points`
- `positions`
- `initial_vector`
