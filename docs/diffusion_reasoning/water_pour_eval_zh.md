# Water Pour 评测说明

当前 `water_pour` 评测入口与 `one_line_drawing` 一致：

```bash
python scripts/02_run_evaluation.py \
  --env water_pour \
  --pred-root data \
  --gt-root data \
  --visualization-root eval \
  --levels 2 \
  --split eval \
  --output eval/water_pour_level2_report.json
```

## 评测内容

- `strict`
  - 要求视频可恢复出合法倒水状态序列
  - 起始状态正确
  - 中间状态转移合法
  - 不发生容量、颜色落点、守恒错误
  - 最终达到已完成分类的解
- `relaxed`
  - 输出 `[0, 1]` 分数
  - 重点统计规则遵守程度和解题进展
  - 包含：
    - `rule_adherence_ratio`
    - `progress_ratio`
    - `final_state_quality`
    - `observability_ratio`

## 汇总指标说明表

下面这张表对应 `report.json` 里的 `summary` 字段，也就是你运行整套评测后看到的这些 `*_mean` 数值。

| 字段 | 含义 | GT2GT 理想解释 |
| --- | --- | --- |
| `sample_count` | 本次参与汇总的样本总数。 | 应等于当前 level / split 的样本数量。 |
| `strict_pass_rate` | 严格评测通过率。计算方式是：通过 `strict.passed` 的样本数 / 总样本数。 | GT2GT 时最好是 `1.0`，表示所有 GT 视频都被判定为合法解。 |
| `relaxed_score_mean` | 宽松总分的均值。它综合了规则遵守、解题进展、最终状态质量和可观测性。 | GT2GT 时应接近 `1.0`。如果略低，通常是视觉解码阶段漏掉少量关键状态。 |
| `rule_adherence_ratio_mean` | 规则遵守比例均值。表示在静止/关键帧中，守恒、容量、颜色放置规则同时成立的平均比例。 | GT2GT 时应非常接近 `1.0`。低于 `1.0` 往往说明有少量帧被 CV 误判。 |
| `progress_ratio_mean` | 解题进展均值。当前实现里，如果最终成功恢复到完成状态，这项通常就是满分。 | GT2GT 时通常是 `1.0`。 |
| `final_state_quality_mean` | 最终状态质量均值。衡量最后恢复出的稳定状态是否已经完成分类。 | GT2GT 时通常是 `1.0`。 |
| `observability_ratio_mean` | 可观测性均值。表示恢复出的稳定关键状态数，相对于期望关键状态数的比例。 | GT2GT 时应尽量接近 `1.0`，但它最容易受两排 tube、阴影、运动过渡影响。 |
| `solved_tube_ratio_mean` | 已完成单色满管的 tube 比例均值。 | GT2GT 时通常是 `1.0`。 |
| `monochrome_tube_ratio_mean` | 单色 tube 比例均值。 | GT2GT 时通常是 `1.0`。 |
| `conservation_ok_ratio_mean` | 守恒正确比例均值。表示颜色块数量在关键帧上是否保持一致。 | GT2GT 时应非常接近 `1.0`。 |
| `illegal_transition_ratio_mean` | 合法状态转移比例均值。表示恢复出的关键状态之间，是否都可以由合法倒水规则连接。 | GT2GT 时最好是 `1.0`。 |

## 如何解读一组 GT2GT 结果

| 结果现象 | 解释 |
| --- | --- |
| `strict_pass_rate = 1.0` | 说明评测器已经能把 GT 视频判成“逻辑正确且最终解出”。这通常是最重要的正确性指标。 |
| `relaxed_score_mean` 很接近 `1.0`，但不是完全 `1.0` | 通常不是任务逻辑错误，而是视觉解码还漏掉了少量关键状态，所以 `observability_ratio_mean` 或 `rule_adherence_ratio_mean` 被轻微拉低。 |
| `progress_ratio_mean = 1.0`、`final_state_quality_mean = 1.0` | 说明从解题结果角度看，评测器已经认为 GT 视频完整解出了任务。 |
| `observability_ratio_mean` 明显低于 `1.0` | 说明视频中的某些稳定状态没有被稳定恢复出来。常见原因是阴影、遮挡、两排 tube 布局、块在运动结束前后的过渡帧。 |
| `illegal_transition_ratio_mean = 1.0` | 说明恢复出的关键状态之间都能被解释为合法倒水步骤，没有出现明显“穿模”或非法堆叠。 |

## 可视化视频

每个样本输出一个 side-by-side 视频：

- 左侧：GT
- 右侧：预测
- 顶部 banner：
  - `sample_id`
  - `level`
  - `bucket`
  - `Strict`
  - `Relaxed`
  - 非法转移、守恒错误、未解析帧计数
- 右侧 overlay：
  - 当前帧号
  - 是否成功解析
  - 守恒 / 放置 / 容量检查
  - 已完成 tube 数
  - 当前推断 move 或 airborne block

## 汇总报告

`--output` JSON 会包含：

- `strict_pass_rate`
- `relaxed_score_mean`
- `rule_adherence_ratio_mean`
- `progress_ratio_mean`
- `final_state_quality_mean`
- `observability_ratio_mean`
