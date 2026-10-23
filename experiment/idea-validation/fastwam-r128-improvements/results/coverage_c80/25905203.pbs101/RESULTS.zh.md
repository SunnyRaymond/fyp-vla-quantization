# R128 improvements: fixed-input results

所有误差越低越好。完整固定 heldout 为40 contexts ×2 seeds，先seed→context→suite，再四suite等权。
所有预先指定的arm均报告；没有按heldout挑选checkpoint或最佳配置。

## calibration

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| C80_once_lr1e4 | 0.045809319 | 0.037399097 | 0.003784558 | +3.026% |
| H_W16A4 | 0.044464021 | 0.039335942 | 0.003835097 | +0.000% |
| R8_W16A4 | 0.040792220 | 0.034579013 | 0.003145532 | -8.258% |

## heldout

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| C80_once_lr1e4 | 0.042443858 | 0.034349046 | 0.003288021 | -0.377% |
| H_W16A4 | 0.042604556 | 0.038380617 | 0.003474206 | +0.000% |
| R8_W16A4 | 0.044585121 | 0.040343879 | 0.003740738 | +4.649% |

## c80

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| C80_once_lr1e4 | 0.044635637 | 0.038370860 | 0.003666527 | n/a |

## Comparisons

- C80_once_lr1e4_vs_H_W16A4: motor -0.377%, latent -10.504%, mixed -5.359%; motor improved/worsened 18/22 of40 contexts.
- R8_W16A4_vs_H_W16A4: motor +4.649%, latent +5.115%, mixed +7.672%; motor improved/worsened 12/28 of40 contexts.
- C80_once_lr1e4_vs_R8_W16A4: motor -4.803%, latent -14.859%, mixed -12.102%; motor improved/worsened 28/12 of40 contexts.

## Evidence boundary

BF16原始weights、A4 activations、BF16 KV；固定Smooth/signs，D=I，clipping ratio=1，无VQ。
固定输入数值证据，不代表closed-loop、native kernel、延迟或显著性检验。
用户要求仅训练C80并复用原C8-R8与H对照。C80与原C8-R8的样本处理量、LR和梯度累积方式不同，是整体recipe对照，不能单独归因于calibration数量。
