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
| grouped_lr1e4 | 0.044283326 | 0.026850229 | 0.003136338 | -0.406% |
| interleaved_lr1e3 | 0.054192602 | 0.035927127 | 0.004617184 | +21.880% |
| interleaved_lr1e4 | 0.041998560 | 0.027309381 | 0.002936811 | -5.545% |
| C80_once_lr1e4 | 0.045809319 | 0.037399097 | 0.003784558 | +3.026% |

## heldout

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| H_W16A4 | 0.042604556 | 0.038380617 | 0.003474206 | +0.000% |
| Rt025_W16A4 | 0.043145218 | 0.041919314 | 0.003681327 | +1.269% |
| Rt050_W16A4 | 0.042680165 | 0.042637283 | 0.003673972 | +0.177% |
| R8_W16A4 | 0.044585121 | 0.040343879 | 0.003740738 | +4.649% |
| grouped_lr1e4 | 0.044338215 | 0.038351924 | 0.003626084 | +4.069% |
| interleaved_lr1e3 | 0.047950823 | 0.038014711 | 0.003936433 | +12.549% |
| interleaved_lr1e4 | 0.043917302 | 0.037503106 | 0.003551120 | +3.081% |
| C80_once_lr1e4 | 0.042443858 | 0.034349046 | 0.003288021 | -0.377% |

## c80

| Arm | Motor RMSE | Latent NMSE | Mixed loss | Motor vs H |
|---|---:|---:|---:|---:|
| C80_once_lr1e4 | 0.044635637 | 0.038370860 | 0.003666527 | n/a |

## Comparisons

- Rt025_W16A4_vs_H_W16A4: motor +1.269%, latent +9.220%, mixed +5.962%; motor improved/worsened 21/19 of40 contexts.
- Rt050_W16A4_vs_H_W16A4: motor +0.177%, latent +11.091%, mixed +5.750%; motor improved/worsened 22/18 of40 contexts.
- R8_W16A4_vs_H_W16A4: motor +4.649%, latent +5.115%, mixed +7.672%; motor improved/worsened 12/28 of40 contexts.
- grouped_lr1e4_vs_H_W16A4: motor +4.069%, latent -0.075%, mixed +4.372%; motor improved/worsened 10/30 of40 contexts.
- interleaved_lr1e3_vs_H_W16A4: motor +12.549%, latent -0.953%, mixed +13.305%; motor improved/worsened 3/37 of40 contexts.
- interleaved_lr1e4_vs_H_W16A4: motor +3.081%, latent -2.286%, mixed +2.214%; motor improved/worsened 17/23 of40 contexts.
- grouped_lr1e4_vs_R8_W16A4: motor -0.554%, latent -4.937%, mixed -3.065%; motor improved/worsened 20/20 of40 contexts.
- interleaved_lr1e3_vs_R8_W16A4: motor +7.549%, latent -5.773%, mixed +5.231%; motor improved/worsened 9/31 of40 contexts.
- interleaved_lr1e4_vs_grouped_lr1e4: motor -0.949%, latent -2.213%, mixed -2.067%; motor improved/worsened 21/19 of40 contexts.
- interleaved_lr1e4_vs_interleaved_lr1e3: motor -8.412%, latent -1.346%, mixed -9.788%; motor improved/worsened 32/8 of40 contexts.
- C80_once_lr1e4_vs_H_W16A4: motor -0.377%, latent -10.504%, mixed -5.359%; motor improved/worsened 18/22 of40 contexts.
- C80_once_lr1e4_vs_R8_W16A4: motor -4.803%, latent -14.859%, mixed -12.102%; motor improved/worsened 28/12 of40 contexts.

## Evidence boundary

BF16原始weights、A4 activations、BF16 KV；固定Smooth/signs，D=I，clipping ratio=1，无VQ。
固定输入数值证据，不代表closed-loop、native kernel、延迟或显著性检验。
用户要求仅训练C80并复用原C8-R8与H对照。C80与原C8-R8的样本处理量、LR和梯度累积方式不同，是整体recipe对照，不能单独归因于calibration数量。
