# Maze Video Dataset Generator

生成迷宫视频数据集：

- 主角是红色小球
- 迷宫墙壁是黑色格子
- 起点固定左上角 `(0, 0)`
- 终点固定右下角 `(N-1, N-1)`
- 小球沿一条从起点到终点的 BFS 最短路径移动，速度为 `5` 帧走 `1` 格
- 默认 `N=6`
- 默认 `fps=10`
- 默认输出 `1000` 个 `mp4`
- 支持混合多个 `N`，并固定输出分辨率不变
- 支持 `candidate_dfs` 和 `recursive_division` 两种迷宫生成算法

## 运行

```bash
python generate_maze_videos.py
```

默认输出目录是 `maze_videos/`，同时会生成一份 `metadata.jsonl`。
如果输出目录里已经有同名视频，脚本默认会跳过已有文件并重写 metadata；加上 `--overwrite` 才会重新编码。

## 常用参数

```bash
python generate_maze_videos.py \
  --n 6 \
  --num-videos 1000 \
  --algorithm candidate_dfs \
  --fps 10 \
  --frames-per-move 5 \
  --output-dir maze_videos \
  --candidate-mazes 8 \
  --top-candidates 3 \
  --branch-weight 0.35 \
  --loop-weight 0.25 \
  --workers 4 \
  --overwrite
```

混合 `N=6/7/8` 且固定 `H=W=208` 的例子：

```bash
python generate_maze_videos.py \
  --num-videos 3000 \
  --algorithm candidate_dfs \
  --n-values 6,7,8 \
  --n-counts 1000,1000,1000 \
  --frame-size 208 \
  --output-dir maze_videos \
  --workers 8 \
  --overwrite
```

使用递归分割算法并输出到另一目录的例子：

```bash
python generate_maze_videos.py \
  --num-videos 3000 \
  --algorithm recursive_division \
  --n-values 6,7,8 \
  --n-counts 1000,1000,1000 \
  --frame-size 208 \
  --output-dir maze_videos_recursive_division \
  --workers 8 \
  --overwrite
```

## 说明

- 现在允许多解。生成器会先随机采样若干迷宫候选，再轻度偏好“路径上有分叉”和“存在环路”的样本，因此整体会更随机，不会都长成同一种骨架。
- `candidate_dfs` 会先生成随机 DFS 迷宫，再随机加环并做轻度候选筛选。
- `recursive_division` 会先全连通，再递归建墙分割区域。
- `metadata.jsonl` 中记录的是视频实际走的那条路径，不代表它一定是唯一解。
- 当使用 `--frame-size` 时，不同 `N` 会先生成逻辑迷宫，再统一缩放到同一个输出大小。
- 渲染使用 `imageio` 输出 `mp4`，不依赖系统级 `ffmpeg` 命令。

## Maze 50 Test

生成带 GT 视频、结构化迷宫和官方路线的 50 条测试集：

```bash
python generate_maze_50_test.py
```

输出：

- `maze_50_test/`: 50 条 GT `mp4` 和 `metadata.jsonl`
- `maze_50_test_npy/`: 每条样本一个 `npy`，保存迷宫结构、墙体网格、边集和官方解题路线

默认是 5 组，每组 10 条，顺序为：

- `N=6`
- `N=7`
- `N=8`
- `N=9`
- `N=10`

## Eval

评测模型输出视频时，要求预测目录里有 50 条 `mp4`，文件名与 GT 一一对应，例如 `maze_0000.mp4` 到 `maze_0049.mp4`。

```bash
python eval_maze_videos.py \
  --pred-dir /mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_wan22_ti2v_5b_maze_recursive_division_st_end_plus_repframe_rope_budget_7_11_17_26_40_50_eval \
  --struct-dir maze_test_npy \
  --output-json maze_eval_summary.json \
  --output-npy maze_eval_formalized.npy
```

评测规则分两套：

- 严格版：
- 红球在到达终点前消失：`0`
- 红球在到达终点前分叉成多个显著红块，且相距大于 `3` 个格子步长：`0`
- 红球中心在到达终点前落到墙体区域，视为穿墙：`0`
- 形式化后的离散路线不从起点出发、不能沿合法边移动、或最终未到终点：`0`
- 只有形式化路线合法且曾到达终点：`1`
- 宽松版：
- 每帧只要检测到红球，就选择最可能的那个红块继续跟踪
- 收集整段视频的形式化轨迹，与 GT 路线计算最大有序交集
- 分数定义为 `LCS(pred_route, gt_route) / len(gt_route)`，例如 GT 路线长 `20`、重合 `10`，得分 `0.5`

脚本会输出：

- 严格版总成功率
- 严格版 `N=6/7/8/9/10` 五个 size 各自的成功率
- 宽松版总平均分
- 宽松版 `N=6/7/8/9/10` 五个 size 各自的平均分
- 每条样本的严格结果、宽松分数、形式化路线与失败原因


 python /mnt/zezhong/MazeDataset/hanoi_dataset/eval_hanoi.py \
    --dataset-dir /mnt/zezhong/MazeDataset/hanoi_dataset \
    --split hanoi_test \
    --pred-dir /mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_wan22_ti2v_5b_hanoi_125f_budget_5_8_13_20_32_50_eval

## Mixed Reasoning Eval

评测 mixed reasoning 输出目录：

```bash
python /mnt/zezhong/MazeDataset/eval_mixed_reasoning.py \
  --pred-dir /mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_tf_framewise_wan22_ti2v_5b_mixed_reasoning_ar32_split21_eval
```

默认使用：

- `test_jsonl`: `/mnt/zezhong/HiDiT/Causal-Forcing/dataset/mixed_reasoning_all_125f_224/vertical_125f/test.jsonl`
- `full_test_jsonl`: `/mnt/zezhong/HiDiT/Causal-Forcing/dataset/mixed_reasoning_all_125f_224/vertical_125f/test_full_backup_before_level20_20260418.jsonl`
- `manifest_jsonl`: `/mnt/zezhong/HiDiT/Causal-Forcing/dataset/mixed_reasoning_all_125f_224/vertical_125f/manifest.jsonl`

对齐规则：

- 如果预测视频数量等于 `full_test_jsonl` 的条数，会按 `video_path` 映射回 full test 原始 idx，再评测当前 `test.jsonl` 子集。
- 否则按预测目录内 `mp4` 字典序，从 `test.jsonl` 的 `--start-index` 开始一一对应截取。
- Maze 里的 `realbot` 样本会跳过，不参与评分。

常用可选参数：

```bash
python /mnt/zezhong/MazeDataset/eval_mixed_reasoning.py \
  --pred-dir /path/to/pred_dir \
  --start-index 0 \
  --limit 0 \
  --output-json /path/to/pred_dir/mixed_reasoning_eval.json
```

输出：

- `mixed_reasoning_eval.json`: 每条样本的详细评测结果与总表。
- `mixed_reasoning_eval.md`: Markdown 汇总表，包括 maze 按 `N`、hanoi 按 `InD/OOD`、其它任务按 level 的 strict/loose 分数。
