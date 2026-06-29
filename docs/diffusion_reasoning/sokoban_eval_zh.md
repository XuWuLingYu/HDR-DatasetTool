# Sokoban 评测说明

当前 `sokoban` 评测入口与其他任务一致：

```bash
python scripts/02_run_evaluation.py \
  --env sokoban \
  --pred-root data \
  --gt-root data \
  --visualization-root eval \
  --levels 2 \
  --split eval \
  --output eval/sokoban_level2_report.json
```

## 评测内容

- `strict`
  - 要求视频可恢复出合法 Sokoban 状态序列
  - 起始状态正确
  - 中间状态之间能由合法移动 / 推箱连接
  - 最终箱子到达目标格
- `relaxed`
  - 输出 `[0, 1]` 分数
  - 重点看箱子是否朝目标推进
  - 同时统计规则遵守、可观测性和最终状态质量
  - 对“只恢复出一两个合法动作”的低质量视频会强力压分

## 汇总指标说明表

| 字段 | 含义 | GT2GT 理想解释 |
| --- | --- | --- |
| `sample_count` | 本次参与汇总的样本总数。 | 应等于当前 level / split 的样本数量。 |
| `strict_pass_rate` | 严格评测通过率。 | GT2GT 时最好是 `1.0`。 |
| `relaxed_score_mean` | 宽松总分均值。 | GT2GT 时应非常接近 `1.0`。 |
| `progress_ratio_mean` | 箱子朝目标推进的平均比例。 | GT2GT 时通常接近 `1.0`。 |
| `final_state_quality_mean` | 最终恢复状态质量均值。 | GT2GT 时通常接近 `1.0`。 |
| `rule_adherence_ratio_mean` | 已恢复稳定帧中，实体位置和基本规则正确的比例。 | GT2GT 时应非常接近 `1.0`。 |
| `observability_ratio_mean` | 恢复出的稳定状态数相对期望状态数的比例。 | GT2GT 时应尽量接近 `1.0`。 |
| `illegal_transition_ratio_mean` | 相邻恢复状态能否由合法 Sokoban 动作连接的比例。 | GT2GT 时最好是 `1.0`。 |

## 如何解读一组 GT2GT 结果

| 结果现象 | 解释 |
| --- | --- |
| `strict_pass_rate = 1.0` | 说明评测器已经能把 GT 视频恢复成合法解题轨迹，并确认最终推到目标。 |
| `relaxed_score_mean` 接近 `1.0` 但不满分 | 通常说明任务逻辑没问题，但有少量中间帧没有被稳定恢复。 |
| `observability_ratio_mean` 偏低 | 说明运动过程中的某些关键稳定状态没有被清楚识别出来。 |
| `progress_ratio_mean` 偏低 | 说明评测器没有稳定看出箱子朝目标推进。 |
| `strict_pass_rate` 低 | 说明起始状态、终止状态或中间合法转移恢复存在明显问题。 |

## 可视化视频

每个样本都会输出 side-by-side 视频：

- 左侧：GT
- 右侧：预测
- 顶部 banner：
  - `sample_id`
  - `level`
  - `bucket`
  - `Strict`
  - `Relaxed`
  - 非法转移、无效状态、未解析帧计数
- 右侧 overlay：
  - 当前帧号
  - 是否解析成功
  - player / box 位置
  - 是否在 target
  - box 到 target 的距离
  - 整条轨迹的 observability / rule adherence

## 验证建议

当前建议只做 `gt2gt`：

- `strict_pass_rate` 应达到 `1.0`
- `relaxed_score_mean` 应接近 `1.0`
- 如果不是，就优先检查状态恢复和中间状态桥接逻辑
