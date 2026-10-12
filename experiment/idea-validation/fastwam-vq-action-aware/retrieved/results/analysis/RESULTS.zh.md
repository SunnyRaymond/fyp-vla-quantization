# Fast-WAM Action-Aware VQ 汇总

Full run 状态为 `complete`，作业 `25743693.pbs101`；protocol `fastwam-vq-action-aware-v1`。
本分析覆盖 614 个 target Linear，22 个 test cases、
44 个 case/seed contexts；未执行 episode 或预测动作。

## 选择冻结

Action lambda 只依据 4 个 selection cases 选择：每个 case 先平均两个 sampler seeds，再对 cases 等权平均。
冻结值为 `0.1`；测试记录均晚于 selection 记录，且使用该值。

| Lambda | Selection mean of four case means |
|---:|---:|
| 0.1 | 0.041168183 |
| 1.0 | 0.044446538 |

## Held-out fixed-input action RMSE

Primary 为相同 case、seed 下相对 BF16 reference 的前 10 步 × 6 motor 维 RMSE。先在每个 case 内平均两个 seeds，再对 case 等权平均。
Plus cases 与原任务可能共享底层任务，因此 22 个 case 按描述性观察汇总；不报告独立样本置信区间或 p 值。

| Arm | Test cases | Contexts | Primary motor RMSE | First10 gripper RMSE | 32-step motor RMSE |
|---|---:|---:|---:|---:|---:|
| bf16 | 22 | 44 | 0 | 0 | 0 |
| bf16w_a4 | 22 | 44 | 0.048559209 | 0.015464677 | 0.059608922 |
| scalar_w4a4 | 22 | 44 | 0.054994168 | 0.012510659 | 0.059340984 |
| original_vq_a4 | 22 | 44 | 0.071310809 | 0.040501796 | 0.074633729 |
| row_scale_vq_a4 | 22 | 44 | 0.057052058 | 0.031797958 | 0.059709202 |
| row_output_vq_a4 | 22 | 44 | 0.057255758 | 0.018110268 | 0.066828375 |
| row_action_vq_a4 | 22 | 44 | 0.061029794 | 0.021260778 | 0.066599787 |

## 按 domain 和 dimension

每格仍是 case 内 seed 平均后，对该组 cases 等权平均。

| Arm | Domain | Dimension | Cases | Contexts | Mean primary RMSE |
|---|---|---|---:|---:|---:|
| bf16 | original | unperturbed | 10 | 20 | 0 |
| bf16w_a4 | original | unperturbed | 10 | 20 | 0.04123327 |
| scalar_w4a4 | original | unperturbed | 10 | 20 | 0.045664474 |
| original_vq_a4 | original | unperturbed | 10 | 20 | 0.057369649 |
| row_scale_vq_a4 | original | unperturbed | 10 | 20 | 0.04770841 |
| row_output_vq_a4 | original | unperturbed | 10 | 20 | 0.04812203 |
| row_action_vq_a4 | original | unperturbed | 10 | 20 | 0.051663303 |
| bf16 | plus | all_dimensions | 12 | 24 | 0 |
| bf16w_a4 | plus | all_dimensions | 12 | 24 | 0.054664159 |
| scalar_w4a4 | plus | all_dimensions | 12 | 24 | 0.062768912 |
| original_vq_a4 | plus | all_dimensions | 12 | 24 | 0.082928443 |
| row_scale_vq_a4 | plus | all_dimensions | 12 | 24 | 0.064838431 |
| row_output_vq_a4 | plus | all_dimensions | 12 | 24 | 0.064867199 |
| row_action_vq_a4 | plus | all_dimensions | 12 | 24 | 0.068835203 |
| bf16 | plus | camera | 2 | 4 | 0 |
| bf16w_a4 | plus | camera | 2 | 4 | 0.055202037 |
| scalar_w4a4 | plus | camera | 2 | 4 | 0.072345718 |
| original_vq_a4 | plus | camera | 2 | 4 | 0.075540293 |
| row_scale_vq_a4 | plus | camera | 2 | 4 | 0.059497577 |
| row_output_vq_a4 | plus | camera | 2 | 4 | 0.053123468 |
| row_action_vq_a4 | plus | camera | 2 | 4 | 0.08061974 |
| bf16 | plus | robot | 2 | 4 | 0 |
| bf16w_a4 | plus | robot | 2 | 4 | 0.09988256 |
| scalar_w4a4 | plus | robot | 2 | 4 | 0.098344777 |
| original_vq_a4 | plus | robot | 2 | 4 | 0.14864798 |
| row_scale_vq_a4 | plus | robot | 2 | 4 | 0.11857694 |
| row_output_vq_a4 | plus | robot | 2 | 4 | 0.10709365 |
| row_action_vq_a4 | plus | robot | 2 | 4 | 0.11697148 |
| bf16 | plus | layout | 2 | 4 | 0 |
| bf16w_a4 | plus | layout | 2 | 4 | 0.039921666 |
| scalar_w4a4 | plus | layout | 2 | 4 | 0.041351436 |
| original_vq_a4 | plus | layout | 2 | 4 | 0.051857343 |
| row_scale_vq_a4 | plus | layout | 2 | 4 | 0.046088919 |
| row_output_vq_a4 | plus | layout | 2 | 4 | 0.044153356 |
| row_action_vq_a4 | plus | layout | 2 | 4 | 0.04930102 |
| bf16 | plus | language | 2 | 4 | 0 |
| bf16w_a4 | plus | language | 2 | 4 | 0.039177311 |
| scalar_w4a4 | plus | language | 2 | 4 | 0.048454347 |
| original_vq_a4 | plus | language | 2 | 4 | 0.066537021 |
| row_scale_vq_a4 | plus | language | 2 | 4 | 0.044205635 |
| row_output_vq_a4 | plus | language | 2 | 4 | 0.043772275 |
| row_action_vq_a4 | plus | language | 2 | 4 | 0.042250374 |
| bf16 | plus | light | 2 | 4 | 0 |
| bf16w_a4 | plus | light | 2 | 4 | 0.042572289 |
| scalar_w4a4 | plus | light | 2 | 4 | 0.044573724 |
| original_vq_a4 | plus | light | 2 | 4 | 0.078148292 |
| row_scale_vq_a4 | plus | light | 2 | 4 | 0.044851347 |
| row_output_vq_a4 | plus | light | 2 | 4 | 0.055689047 |
| row_action_vq_a4 | plus | light | 2 | 4 | 0.053245155 |
| bf16 | plus | background | 2 | 4 | 0 |
| bf16w_a4 | plus | background | 2 | 4 | 0.051229089 |
| scalar_w4a4 | plus | background | 2 | 4 | 0.07154347 |
| original_vq_a4 | plus | background | 2 | 4 | 0.076839727 |
| row_scale_vq_a4 | plus | background | 2 | 4 | 0.075810167 |
| row_output_vq_a4 | plus | background | 2 | 4 | 0.085371399 |
| row_action_vq_a4 | plus | background | 2 | 4 | 0.070623446 |

## 与 scalar W4A4 的配对 case 差

这是两种 arm 的 primary RMSE 之差，不是 raw action 向量差；每个 case 的两 seed 均先平均。负值表示该 arm 的 RMSE 较低。

| Arm | Mean paired difference | Lower-RMSE cases | Higher-RMSE cases |
|---|---:|---:|---:|
| bf16 | -0.054994168 | 22 | 0 |
| bf16w_a4 | -0.0064349585 | 17 | 5 |
| scalar_w4a4 | 0 | 0 | 0 |
| original_vq_a4 | 0.016316642 | 2 | 20 |
| row_scale_vq_a4 | 0.0020578902 | 10 | 12 |
| row_output_vq_a4 | 0.0022615907 | 8 | 14 |
| row_action_vq_a4 | 0.0060356263 | 7 | 15 |

## 编码存储

字节数和 bpw 来自 full-run 每 arm receipt；transform 字段单独列出，receipt 声明的 bytes accounting 原样保留。bf16 reference 的缺失 bpw 显示为 n/a。

| Arm | Encoded bytes | bpw | Serialized bytes | Transform |
|---|---:|---:|---:|---|
| bf16 | 12032634880 | 16 | n/a | `"identity"` |
| bf16w_a4 | 12041669195 | 16.012013 | n/a | `{"name":"smooth05_hadamard","alpha":0.5,"hadamard_block":128,"scale_bound":16,"source":"reuse exact saved transforms from original successful full banks; no refitting or alpha search"}` |
| scalar_w4a4 | 3111207611 | 4.1370259 | 3112661146 | `{"name":"smooth05_hadamard","alpha":0.5,"hadamard_block":128,"scale_bound":16,"source":"reuse exact saved transforms from original successful full banks; no refitting or alpha search"}` |
| original_vq_a4 | 3019708491 | 4.0153579 | 3021279898 | `{"name":"smooth05_hadamard","alpha":0.5,"hadamard_block":128,"scale_bound":16,"source":"reuse exact saved transforms from original successful full banks; no refitting or alpha search"}` |
| row_scale_vq_a4 | 3023856089 | 4.0208731 | 3025501840 | `{"name":"smooth05_hadamard","alpha":0.5,"hadamard_block":128,"scale_bound":16,"source":"reuse exact saved transforms from original successful full banks; no refitting or alpha search"}` |
| row_output_vq_a4 | 3023856089 | 4.0208731 | 3025501840 | `{"name":"smooth05_hadamard","alpha":0.5,"hadamard_block":128,"scale_bound":16,"source":"reuse exact saved transforms from original successful full banks; no refitting or alpha search"}` |
| row_action_vq_a4 | 3023856089 | 4.0208731 | 3025501840 | `{"name":"smooth05_hadamard","alpha":0.5,"hadamard_block":128,"scale_bound":16,"source":"reuse exact saved transforms from original successful full banks; no refitting or alpha search"}` |

## 范围与限制

Test 集包含 10 个熟悉的 original tasks 与 12 个 Plus input variants（6 dimensions，每维 2 个）；这不是 unseen-task generalization。
Action-aware arm 使用采样的一阶 action projection proxy；它不等于完整 final-action sensitivity，也不单独证明 action 改善机制。
本结果来自固定 observation 的数值查询，episodes=0；不提供任务成功率、native VQ kernel 延迟或部署速度结论。

Primary metric：`case mean over two seeds of first10 x 6 motor RMSE versus own same-input same-seed original BF16`。源数据：`/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009/results/full`。
