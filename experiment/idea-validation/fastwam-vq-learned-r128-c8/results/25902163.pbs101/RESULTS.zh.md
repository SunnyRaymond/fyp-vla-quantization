# H+VQ8 与 R8+VQ8 固定输入配对结果

Protocol：`fastwam-vq-learned-r128-c8-v1`；每臂40 contexts × 2 seeds = 80 queries。
先对每个context的两seed取均值，再对每个suite的10个context取均值，最后四suite等权。
相对变化为 candidate / baseline - 1；负值表示误差下降。

## Overall

| Metric | H+VQ8 baseline | R8+VQ8 candidate | Candidate change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.044296774 | 0.046408990 | +4.768% |
| Future latent NMSE | 0.060877371 | 0.065187050 | +7.079% |
| Total mixed loss | 0.004598652 | 0.004980800 | +8.310% |
| Full32 motor RMSE | 0.047500980 | 0.050613965 | +6.554% |
| Late22 motor RMSE | 0.048528504 | 0.051931984 | +7.013% |
| First10 gripper RMSE | 0.057054031 | 0.060208080 | +5.528% |
| First10 motor MSE | 0.002014132 | 0.002213315 | +9.889% |

## By suite

| Suite | Arm | First10 motor RMSE | Future latent NMSE | Total mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| libero_spatial | H+VQ8 | 0.044794729 | 0.053923931 | 0.004318821 | 0.046173595 | 0.046622019 | 0.046753075 |
| libero_spatial | R8+VQ8 | 0.048821239 | 0.059891817 | 0.005007176 | 0.049376729 | 0.049427376 | 0.051024390 |
| libero_object | H+VQ8 | 0.041680142 | 0.065074514 | 0.004552778 | 0.045767309 | 0.047370207 | 0.017308175 |
| libero_object | R8+VQ8 | 0.043989306 | 0.072291899 | 0.005060678 | 0.051042690 | 0.053440361 | 0.020168558 |
| libero_goal | H+VQ8 | 0.045092065 | 0.068296552 | 0.004991575 | 0.052540081 | 0.054876660 | 0.114934304 |
| libero_goal | R8+VQ8 | 0.048103066 | 0.067375104 | 0.005204254 | 0.056432199 | 0.059112648 | 0.117244495 |
| libero_10 | H+VQ8 | 0.045620161 | 0.056214487 | 0.004531433 | 0.045522934 | 0.045245130 | 0.049220567 |
| libero_10 | R8+VQ8 | 0.044722350 | 0.061189379 | 0.004651092 | 0.045604242 | 0.045747552 | 0.052394879 |

## Paired case win/loss

每个context先对两个seed取均值；误差下降记为 candidate improved。

| Metric | Improved | Worsened | Tied | Cases |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 16 | 24 | 0 | 40 |
| Future latent NMSE | 19 | 21 | 0 | 40 |
| Total mixed loss | 16 | 24 | 0 | 40 |
| Full32 motor RMSE | 13 | 27 | 0 | 40 |
| Late22 motor RMSE | 14 | 26 | 0 | 40 |
| First10 gripper RMSE | 4 | 36 | 0 | 40 |
| First10 motor MSE | 16 | 24 | 0 | 40 |

该结果是固定输入、decoded-BF16/A4 的描述性配对比较。paired case counts 不是显著性检验；不构成 closed-loop 或 native-kernel 结论。
