# Sliding Puzzle 评测说明

当前 `sliding_puzzle` 评测入口与其他任务一致：

```bash
python scripts/02_run_evaluation.py \
  --env sliding_puzzle \
  --pred-root data \
  --gt-root data \
  --visualization-root eval \
  --levels 2 4 \
  --split eval \
  --output eval/sliding_puzzle_report.json
```

## 评测内容

- `strict`
  - 要求视频可恢复出合法滑块状态序列
  - 起始状态正确
  - 中间状态之间能由合法滑动连接
  - 最终达到目标有序棋盘
- `relaxed`
  - 输出 `[0, 1]` 分数
  - 优先看是否朝目标状态推进
  - 同时统计规则遵守、最终状态质量和状态可观测性

## 汇总指标说明表

| 字段 | 含义 | GT2GT 理想解释 |
| --- | --- | --- |
| `sample_count` | 本次参与汇总的样本总数。 | 应等于当前 level / split 的样本数量。 |
| `strict_pass_rate` | 严格评测通过率。 | GT2GT 时最好是 `1.0`。 |
| `relaxed_score_mean` | 宽松总分的均值。 | GT2GT 时应接近 `1.0`。 |
| `progress_ratio_mean` | 朝目标状态推进的平均比例。 | GT2GT 时通常接近 `1.0`。 |
| `final_state_quality_mean` | 最终恢复状态距离目标的质量分。 | GT2GT 时通常接近 `1.0`。 |
| `rule_adherence_ratio_mean` | 已解析帧中，tile 集合和 blank 规则正确的比例。 | GT2GT 时应非常接近 `1.0`。 |
| `observability_ratio_mean` | 恢复出的稳定状态数相对期望状态数的比例。 | GT2GT 时应尽量接近 `1.0`。 |
| `illegal_transition_ratio_mean` | 相邻恢复状态可由合法滑块规则连接的比例。 | GT2GT 时最好是 `1.0`。 |
| `correct_tile_ratio_mean` | 至少在某个稳定状态中，放在正确位置的 tile 比例均值。 | GT2GT 时通常接近 `1.0`。 |

## 如何解读一组 GT2GT 结果

| 结果现象 | 解释 |
| --- | --- |
| `strict_pass_rate = 1.0` | 说明评测器可以把 GT 视频恢复为合法解题序列，并确认最终解出。 |
| `relaxed_score_mean` 接近 `1.0` 但不是完全 `1.0` | 通常说明逻辑判断没问题，但少量中间帧没有被稳定恢复。 |
| `observability_ratio_mean` 偏低 | 说明运动过程里有些中间状态没有被稳定解码出来。 |
| `illegal_transition_ratio_mean < 1.0` | 说明恢复出的相邻状态之间存在无法用合法滑动解释的跳跃。 |
