# Shape Return Dataset

本目录用于生成“物体离开画面再返回”的白底视频数据集。

## Setting

- 背景纯白
- 物体初始位于画面中心
- 匀速向右移动到画外
- 在画外停留 `1` 秒
- 再从右侧匀速沿原路返回到中心
- 颜色为随机非白色、且不接近白色的颜色
- 默认 `fps=20`

## Train

默认训练集为 `3000` 条：

- `ball`: `1000`
- `square`: `1000`
- `triangle`: `1000`

运行：

```bash
python generators/classic/shape_return/generate_shape_return_videos.py \
  --num-videos 3000 \
  --shape-values ball,square,triangle \
  --shape-counts 1000,1000,1000 \
  --output-dir outputs/shape_return/videos \
  --struct-dir outputs/shape_return/struct \
  --workers 8 \
  --overwrite
```

## Test

默认测试集为 `50` 条，按组输出：

- `ball`: `10`
- `square`: `10`
- `triangle`: `10`
- `parallelogram`: `10`
- `star`: `10`

运行：

```bash
python generators/classic/shape_return/generate_shape_return_50_test.py \
  --output-dir outputs/shape_return/test_videos \
  --struct-dir outputs/shape_return/test_videos_npy \
  --overwrite
```

## Outputs

每条样本会保存：

- `mp4` 视频
- `metadata.jsonl`
- 对应 `npy` 结构化文件

`npy` 中包含：

- `shape_name`
- `positions`
- `visible`
- `color_rgb`
- `direction`
- `start_center`
- `offscreen_center`
