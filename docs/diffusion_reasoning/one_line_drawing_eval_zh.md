# One-Line Drawing 评测说明

本文档说明 `one_line_drawing` 任务的评测脚本如何使用、预测结果目录应该如何组织、输出内容是什么，以及评测可视化视频中会显示哪些信息。

## 1. 评测入口

当前评测入口脚本为：

```bash
python scripts/02_run_evaluation.py
```

目前该脚本只支持：

```text
--env one_line_drawing
```

## 2. 基本运行命令

评测 `level 2`，并将预测视频根目录设为 `data`：

```bash
python scripts/02_run_evaluation.py \
  --env one_line_drawing \
  --pred-root data \
  --gt-root data \
  --visualization-root eval \
  --levels 2 \
  --split eval \
  --output eval/one_line_drawing_level2_report.json
```

如果你要同时评测多个难度，可以写成：

```bash
python scripts/02_run_evaluation.py \
  --env one_line_drawing \
  --pred-root data \
  --gt-root data \
  --visualization-root eval \
  --levels 2 3 4 \
  --split eval \
  --output eval/one_line_drawing_report.json
```

## 3. 参数说明

- `--env`
  - 环境名称，目前固定为 `one_line_drawing`
- `--pred-root`
  - 预测视频根目录
- `--gt-root`
  - GT 数据根目录，默认是 `data`
  - 这里需要包含 manifest、metadata 和 GT 视频
- `--visualization-root`
  - 评测可视化视频输出根目录，默认是 `eval`
- `--levels`
  - 要评测的难度等级，例如 `2`、`3`、`4`
- `--split`
  - 数据划分，当前支持 `train` 或 `eval`
- `--output`
  - 汇总 JSON 输出路径

## 4. `pred-root` 目录格式

脚本会按以下顺序查找预测视频：

1. `PRED_ROOT/one_line_drawing_{level}level/{split}/{video_name}`
2. `PRED_ROOT/{split}/{video_name}`
3. `PRED_ROOT/{video_name}`

例如，`level 2` 的一个样本视频名可能是：

```text
one_line_drawing_l2_eval_0000.mp4
```

推荐目录结构如下：

```text
data/
  one_line_drawing_2level/
    eval/
      one_line_drawing_l2_eval_0000.mp4
      one_line_drawing_l2_eval_0001.mp4
      ...
```

这种结构和数据集原始 GT 目录一致，最不容易出错。

## 5. GT 数据目录格式

`--gt-root` 下需要有以下内容：

```text
data/
  one_line_drawing_2level/
    eval/
      one_line_drawing_l2_eval_0000.mp4
      one_line_drawing_l2_eval_0000.json
      ...
    eval_manifest.json
```

其中：

- `eval_manifest.json` 用来枚举当前 split 的样本
- 每个样本都有一个同名 `json` metadata
- 每个样本还有一个 GT 视频

## 6. 评测逻辑概述

评测器会读取预测视频和 metadata，然后做两类评测：

- `strict`
  - 严格二分类结果，输出 `passed: true/false`
- `relaxed`
  - 宽松连续分数，输出 `score`

当前不会计算单独的“总分（total score）”，汇总时分别统计：

- `strict_pass_rate`
- `relaxed_score_mean`
- `coverage_ratio_mean`

## 7. Strict 指标含义

`strict` 会检查以下条件是否同时满足：

- 起点是否正确
- 是否覆盖了所有要求经过的格子
- 是否存在重复访问格子
- 是否存在非法跳跃
- 是否存在缺失格子
- 是否有明显离开合法轨迹区域的帧

只有全部满足时，`strict.passed` 才为 `true`。

`strict` 结果中常见字段：

- `passed`
- `resolved_sequence`
  - 从视频中恢复出的离散格子访问序列
- `coverage_complete`
- `missing_cells`
- `repeated_cells`
- `illegal_transitions`
- `unresolved_frames`
- `off_shape_frames`
- `starts_correctly`

## 8. Relaxed 指标含义

`relaxed` 是一个 `[0, 1]` 范围内的分数，当前由三部分组成：

- `coverage_ratio`
  - 已访问目标格子比例
- `legal_transition_ratio`
  - 合法移动比例
- `on_shape_ratio`
  - 保持在合法形状区域内的帧比例

当前公式为：

```text
score = 0.5 * coverage_ratio
      + 0.3 * legal_transition_ratio
      + 0.2 * on_shape_ratio
```

`relaxed` 结果中常见字段：

- `score`
- `coverage_ratio`
- `legal_transition_ratio`
- `on_shape_ratio`
- `completed`
- `revisits_detected`
- `resolved_sequence`

## 9. 可视化评测视频

每个被评测的样本都会输出一个可视化视频，默认保存到：

```text
eval/one_line_drawing_{n}level/
```

例如：

```text
eval/one_line_drawing_2level/one_line_drawing_l2_eval_0000_eval.mp4
```

视频布局如下：

- 左侧：GT 视频
- 右侧：预测视频
- 顶部 banner：
  - `sample_id`
  - `level`
  - `family`
  - `Strict: PASS/FAIL`
  - `Relaxed: 分数`
  - 失败诊断计数：
    - `missing`
    - `illegal`
    - `repeat`
- 右侧预测区域内部还会显示：
  - 当前 `head_cell`
  - 已访问格子数 / 目标格子数
  - 当前帧号
  - `on_shape`
  - `trace_on_shape_ratio`

## 10. 汇总 JSON 输出

`--output` 指定的 JSON 中包含：

- `env`
- `split`
- `levels`
- `pred_root`
- `gt_root`
- `visualization_root`
- `summary`
- `missing_predictions`
- `results`

其中 `summary` 目前包含：

- `sample_count`
- `strict_pass_rate`
- `relaxed_score_mean`
- `coverage_ratio_mean`

### 10.1 汇总指标说明表

下面这张表对应 `report.json` 里的 `summary` 字段，也就是整套评测跑完以后看到的汇总值。

| 字段 | 含义 | GT2GT 理想解释 |
| --- | --- | --- |
| `sample_count` | 本次参与汇总的样本总数。 | 应等于当前 level / split 的样本数量。 |
| `strict_pass_rate` | 严格评测通过率。计算方式是：`strict.passed=true` 的样本数 / 总样本数。 | GT2GT 时最好是 `1.0`，表示所有 GT 视频都被判定为严格正确。 |
| `relaxed_score_mean` | 宽松分数均值。它综合了覆盖率、合法转移比例和是否保持在合法轨迹区域。 | GT2GT 时最好是 `1.0`，说明整体轨迹恢复几乎没有问题。 |
| `coverage_ratio_mean` | 覆盖率均值。表示目标格子被访问到的平均比例。 | GT2GT 时最好是 `1.0`，说明所有样本都完整覆盖了目标形状。 |

### 10.2 如何解读一组 GT2GT 结果

| 结果现象 | 解释 |
| --- | --- |
| `strict_pass_rate = 1.0` | 说明评测器认为所有 GT 视频都满足起点正确、轨迹合法、无重复访问、无缺失格子，并最终完整覆盖目标。 |
| `relaxed_score_mean = 1.0` | 说明从宽松指标看，整体轨迹质量也完全正确，没有明显的离轨、跳跃或覆盖缺失。 |
| `coverage_ratio_mean = 1.0` | 说明平均来看，每个样本都访问到了全部应覆盖的格子。 |
| `strict_pass_rate = 1.0` 但 `relaxed_score_mean < 1.0` | 说明任务结果本身是对的，但某些帧在视觉恢复时可能出现轻微偏差，例如局部离轨、少量帧不稳定。 |

`results` 中每个样本会包含：

- 样本基础信息
- `strict`
- `relaxed`
- `visualization_video`

`visualization_video` 中通常包含：

- `path`
- `rendered_frames`
- `gt_frame_count`
- `pred_frame_count`
- `frame_count_mismatch`

## 11. 常见问题

### 11.1 为什么找不到预测视频

通常是 `--pred-root` 目录结构不匹配。请优先检查预测视频是否满足：

```text
PRED_ROOT/one_line_drawing_{level}level/{split}/{video_name}
```

### 11.2 为什么 `strict` 失败但 `relaxed` 分数还不低

因为 `strict` 是硬条件，只要出现重复访问、非法跳跃、缺失格子或离开合法区域，就会失败。

而 `relaxed` 是连续分数，只要大部分轨迹仍然合理，分数仍可能较高。

### 11.3 为什么没有 total score

当前实现没有合并 `strict` 和 `relaxed` 成单一总分，而是分别报告：

- `strict_pass_rate`
- `relaxed_score_mean`

这样更利于区分“是否真正完成任务”和“过程是否基本合理”。

## 12. 推荐使用方式

如果你只是验证现有数据和评测链路是否正常，推荐先直接运行：

```bash
python scripts/02_run_evaluation.py \
  --env one_line_drawing \
  --pred-root data \
  --gt-root data \
  --visualization-root eval \
  --levels 2 \
  --split eval \
  --output eval/one_line_drawing_level2_report.json
```

这会：

- 使用 `data` 中现有视频作为预测输入
- 使用 `data` 中 metadata 和 GT 视频作为标注
- 在 `eval/one_line_drawing_2level/` 下生成逐样本评测视频
- 在 `eval/one_line_drawing_level2_report.json` 中生成汇总结果
