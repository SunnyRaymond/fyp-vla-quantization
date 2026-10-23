# C80 clip_only 与原 held-out 结果

最终 update80，weights 冻结；四 suites 等权。
Calibration 为80 contexts × 1 seed；原 held-out 为40 contexts × 2 seeds。复用原 teacher、seeds 和 baseline。
未使用新的 development set。

| Split / arm | First10 motor RMSE | Future latent NMSE | Full32 motor RMSE | Mixed loss |
|---|---:|---:|---:|---:|
| calibration / baseline | 0.044146961 | 0.054735372 | 0.046424454 | 0.004351853 |
| calibration / clip_only | 0.041858108 | 0.047223291 | 0.044334697 | 0.003796513 |
| heldout / baseline | 0.044296774 | 0.060877371 | 0.047500980 | 0.004598652 |
| heldout / clip_only | 0.043848437 | 0.054848124 | 0.050111666 | 0.004306253 |

相对 baseline 变化，负值为改善：

- calibration: action -5.185%，latent -13.724%，mixed loss -12.761%。
- heldout: action -1.012%，latent -9.904%，mixed loss -6.358%。

同一 C80 warm start 增加了数据与更新次数；不能单独归因于数据量。
原 held-out baseline 的附带 action MSE 由保存的逐查询 RMSE 平方还原，mixed loss 包含相应舍入误差；主指标直接复用，完全一致。
原 held-out 已参与多次诊断，本轮只能作为固定 cohort 的工程比较。
仅 fixed-input decoded-BF16 VQ + A4 证据，无 closed-loop 或 native INT4 性能结论。
