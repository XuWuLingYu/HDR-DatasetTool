# Mixed Reasoning STD Summary

- rule: for each task-level with `N` samples, sample `N//2` without replacement, compute mean, repeat `10` times, then take the population std of those means
- seed: `20260505`

## Core

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDR full | 0.044966 | 0.052525 | 0.101225 | 0.079307 | 0.096667 | 0.047931 | 0.030432 | 0.010692 | 0.006121 | 0.014695 | 0.003074 | 0.028382 | 0.070437 | 0.015566 |
| Causal full | 0.067288 | 0.080804 | 0.076555 | 0.051633 | 0.106878 | 0.052487 | 0.027618 | 0.083550 | 0.012764 | 0.026214 | 0.057941 | 0.030258 | 0.072608 | 0.039724 |

Paths:

- `HDR full`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_wan22_ti2v_5b_mixed_reasoning_125f_budget_5_8_13_20_32_50_eval/mixed_reasoning_eval.json`
- `Causal full`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_tf_framewise_wan22_ti2v_5b_mixed_reasoning_ar32_split21_eval_10w/mixed_reasoning_eval.json`

## Noise Ablation

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDR fixed5 | 0.052500 | 0.069314 | 0.089575 | 0.082563 | 0.106207 | 0.035444 | 0.035604 | 0.011986 | 0.006142 | 0.015871 | 0.011918 | 0.024475 | 0.072600 | 0.017666 |
| HDR fixed1 | 0.060470 | 0.000000 | 0.120296 | 0.061468 | 0.072318 | 0.010000 | 0.027403 | 0.092119 | 0.011086 | 0.007883 | 0.001905 | 0.018923 | 0.054092 | 0.026553 |
| Causal fixed5 | 0.051363 | 0.113113 | 0.092613 | 0.060000 | 0.101858 | 0.068272 | 0.027762 | 0.091733 | 0.019951 | 0.027946 | 0.058522 | 0.032572 | 0.081203 | 0.043081 |
| Causal fixed1 | 0.043570 | 0.056257 | 0.000000 | 0.067538 | 0.089173 | 0.017951 | 0.045748 | 0.093080 | 0.010005 | 0.035598 | 0.057885 | 0.016985 | 0.045748 | 0.043217 |

Paths:

- `HDR fixed5`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/mixed_reasoning_denoise_ablation/vertical_mixed_fixed5/mixed_reasoning_eval.json`
- `HDR fixed1`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/mixed_reasoning_denoise_ablation/vertical_mixed_fixed1/mixed_reasoning_eval.json`
- `Causal fixed5`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/mixed_reasoning_denoise_ablation/causal_mixed_fixed5/mixed_reasoning_eval.json`
- `Causal fixed1`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/mixed_reasoning_denoise_ablation/causal_mixed_fixed1/mixed_reasoning_eval.json`

## Layer Ablation

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HDR full budget | 0.044966 | 0.052525 | 0.100773 | 0.079307 | 0.096667 | 0.047931 | 0.030432 | 0.010692 | 0.004559 | 0.014780 | 0.003074 | 0.029006 | 0.070361 | 0.015424 |
| Active L2+L6 | 0.041912 | 0.076863 | 0.118121 | 0.090341 | 0.061605 | 0.068300 | 0.028640 | 0.015663 | 0.005656 | 0.019474 | 0.007076 | 0.029120 | 0.076190 | 0.017605 |
| Active L3+L6 | 0.045811 | 0.080298 | 0.082447 | 0.097494 | 0.072426 | 0.036667 | 0.032683 | 0.018431 | 0.004625 | 0.023661 | 0.006638 | 0.027214 | 0.069190 | 0.018875 |
| Active L4+L6 | 0.076914 | 0.106435 | 0.116187 | 0.095706 | 0.107106 | 0.049830 | 0.035746 | 0.031551 | 0.006046 | 0.020932 | 0.031767 | 0.026537 | 0.092030 | 0.025430 |
| Active L5+L6 | 0.082165 | 0.093959 | 0.081022 | 0.065726 | 0.089059 | 0.052701 | 0.050951 | 0.059045 | 0.009319 | 0.019894 | 0.042326 | 0.037319 | 0.077439 | 0.036476 |

Paths:

- `HDR full budget`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/mixed_reasoning_denoise_ablation/vertical_mixed_full_budget/mixed_reasoning_eval.json`
- `Active L2+L6`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_active_l2_l6_eval_10k/mixed_reasoning_eval.json`
- `Active L3+L6`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_active_l3_l6_eval_10k/mixed_reasoning_eval.json`
- `Active L4+L6`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_active_l4_l6_eval_10k/mixed_reasoning_eval.json`
- `Active L5+L6`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_active_l5_l6_eval_10k/mixed_reasoning_eval.json`

## Scheduler

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All 50 | 0.078698 | 0.076000 | 0.099774 | 0.024944 | 0.102201 | 0.062474 | 0.036948 | 0.046511 | 0.004734 | 0.013282 | 0.022684 | 0.031366 | 0.074015 | 0.025921 |
| Budget 2_4_8_16_32_50 | 0.035296 | 0.092000 | 0.086667 | 0.059319 | 0.044019 | 0.052997 | 0.031186 | 0.012743 | 0.003940 | 0.012527 | 0.010113 | 0.023563 | 0.061716 | 0.015679 |
| Budget 5_5_5_5_5_50 | 0.038860 | 0.172554 | 0.103775 | 0.101678 | 0.089327 | 0.055203 | 0.028860 | 0.022866 | 0.004411 | 0.017521 | 0.006667 | 0.030695 | 0.093566 | 0.018503 |

Paths:

- `All 50`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_all50/mixed_reasoning_eval.json`
- `Budget 2_4_8_16_32_50`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_reasoning_budget_2_4_8_16_32_50_eval_100k/mixed_reasoning_eval.json`
- `Budget 5_5_5_5_5_50`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_reasoning_budget_5_5_5_5_5_50_eval_100k/mixed_reasoning_eval.json`

## Data Ablation

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vertical 1/10 data | 0.057848 | 0.099505 | 0.083391 | 0.058609 | 0.058262 | 0.055697 | 0.035544 | 0.020477 | 0.005520 | 0.008170 | 0.000816 | 0.021723 | 0.068885 | 0.015375 |
| Vertical 1/50 data | 0.055863 | 0.134334 | 0.079757 | 0.057071 | 0.071964 | 0.024944 | 0.037467 | 0.065387 | 0.009043 | 0.006809 | 0.004703 | 0.021755 | 0.070656 | 0.024194 |
| Bidirectional low10 | 0.053586 | 0.099414 | 0.091344 | 0.072393 | 0.066725 | 0.061545 | 0.029953 | 0.020300 | 0.005921 | 0.010848 | 0.008981 | 0.035508 | 0.074168 | 0.018585 |
| Bidirectional low50 | 0.067367 | 0.077571 | 0.092694 | 0.032660 | 0.074222 | 0.031447 | 0.025855 | 0.063456 | 0.006563 | 0.007150 | 0.033182 | 0.023913 | 0.062660 | 0.026687 |

Paths:

- `Vertical 1/10 data`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_reasoning_data_1over10_eval_100k/mixed_reasoning_eval.json`
- `Vertical 1/50 data`: `/mnt/zezhong/HiDiT/Causal-Forcing/output/ar_diffusion_vertical_mixed_reasoning_data_1over50_eval_100k/mixed_reasoning_eval.json`
- `Bidirectional low10`: `/mnt/zezhong/users/ruibin/checkpoints/eval/Wan2.2-TI2V-5B_224x224_81f_mixed_reasoning_low10_vertical125f_step100000_two_stage/mixed_reasoning_eval.json`
- `Bidirectional low50`: `/mnt/zezhong/users/ruibin/checkpoints/eval/Wan2.2-TI2V-5B_224x224_81f_mixed_reasoning_low50_vertical125f_step100000_two_stage/mixed_reasoning_eval.json`

## Bidirectional

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bidirectional 50 steps | 0.067823 | 0.031241 | 0.075966 | 0.069778 | 0.042150 | 0.068300 | 0.029534 | 0.001081 | 0.002733 | 0.015578 | 0.000000 | 0.036567 | 0.059210 | 0.014249 |
| Bidirectional 5 steps | 0.069051 | 0.025612 | 0.087906 | 0.057425 | 0.048113 | 0.058462 | 0.030385 | 0.001526 | 0.007177 | 0.016593 | 0.008889 | 0.031629 | 0.057762 | 0.016033 |
| Bidirectional 1 step | 0.027839 | 0.000000 | 0.090756 | 0.043541 | 0.091668 | 0.000000 | 0.031276 | 0.044273 | 0.014175 | 0.006197 | 0.056179 | 0.015312 | 0.042301 | 0.027902 |

Paths:

- `Bidirectional 50 steps`: `/mnt/zezhong/users/ruibin/checkpoints/eval/Wan2.2-TI2V-5B_224x224_81f_mixed_reasoning_vertical125f_step105000_two_stage_50steps/mixed_reasoning_eval.json`
- `Bidirectional 5 steps`: `/mnt/zezhong/users/ruibin/checkpoints/eval/Wan2.2-TI2V-5B_224x224_81f_mixed_reasoning_vertical125f_step105000_two_stage_5steps/mixed_reasoning_eval.json`
- `Bidirectional 1 step`: `/mnt/zezhong/users/ruibin/checkpoints/eval/Wan2.2-TI2V-5B_224x224_81f_mixed_reasoning_vertical125f_step100000_two_stage_1steps/mixed_reasoning_eval.json`

## VideoMAE

| Method | Hanoi S | Maze S | One-line S | Sliding S | Sokoban S | Water S | Hanoi L | Maze L | One-line L | Sliding L | Sokoban L | Water L | Overall S | Overall L |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VideoMAE | 0.061808 | 0.089612 | 0.096748 | 0.000000 | 0.094157 | 0.049432 | 0.038021 | 0.013183 | 0.017137 | 0.000000 | 0.000000 | 0.024777 | 0.065293 | 0.015520 |

Paths:

- `VideoMAE`: `/mnt/zezhong/users/hengzhe/VideoMAE_patch8/output/mixed_reasoning_ar32_split21_videomae_base_patch8_224x224_32f/mixed_reasoning_eval.json`
