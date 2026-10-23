# R128 improvements: fixed-input results

所有误差越低越好。完整固定 heldout 为40 contexts ×2 seeds，先seed→context→suite，再四suite等权。
所有预先指定的arm均报告；没有按heldout挑选checkpoint或最佳配置。

## calibration

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| H_W16A4 | 0.044464021 | 0.039335942 | 0.003835097 | +0.000% |
| Rt025_W16A4 | 0.043467428 | 0.031826478 | 0.003336103 | -2.241% |
| Rt050_W16A4 | 0.044034446 | 0.070498761 | 0.005084047 | -0.966% |
| R8_W16A4 | 0.040792220 | 0.034579013 | 0.003145532 | -8.258% |

## heldout

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| H_W16A4 | 0.042604556 | 0.038380617 | 0.003474206 | +0.000% |
| Rt025_W16A4 | 0.043145218 | 0.041919314 | 0.003681327 | +1.269% |
| Rt050_W16A4 | 0.042680165 | 0.042637283 | 0.003673972 | +0.177% |
| R8_W16A4 | 0.044585121 | 0.040343879 | 0.003740738 | +4.649% |

## Comparisons

- Rt025_W16A4_vs_H_W16A4: motor +1.269%, latent +9.220%, mixed +5.962%; motor improved/worsened 21/19 of40 contexts.
- Rt050_W16A4_vs_H_W16A4: motor +0.177%, latent +11.091%, mixed +5.750%; motor improved/worsened 22/18 of40 contexts.
- R8_W16A4_vs_H_W16A4: motor +4.649%, latent +5.115%, mixed +7.672%; motor improved/worsened 12/28 of40 contexts.

## Evidence boundary

BF16原始weights、A4 activations、BF16 KV；固定Smooth/signs，D=I，clipping ratio=1，无VQ。
固定输入数值证据，不代表closed-loop、native kernel、延迟或显著性检验。
用户要求仅训练C80并复用原C8-R8与H对照。C80与原C8-R8的样本处理量、LR和梯度累积方式不同，是整体recipe对照，不能单独归因于calibration数量。
