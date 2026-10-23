# Row-scale VQ 与 Microscaling VQ：LIBERO-Plus 280 episodes / arm

| 配置 | 成功 / episodes | Success rate |
|---|---:|---:|
| row_scale_vq_a4 | 202/280 | 72.14% |
| micro_scale_vq_a4 | 204/280 | 72.86% |
| bf16 | 216/280 | 77.14% |
| quarot_adapted_w4a4 | 195/280 | 69.64% |
| spinquant_adapted_w4a4 | 203/280 | 72.50% |
| smooth05_hadamard_scalar_w4a4 | 200/280 | 71.43% |

两组VQ均复用原614层bank及Smooth0.5/H128变换，decoded BF16 compute、动态A4、KV BF16；没有重新拟合。
共560个新增episodes，与已有四组的1120个episodes采用相同冻结variants、环境与sampler seeds、IDM/replan/控制步数配置。
已完成的旧实现episodes保留用于success rate；耗时按逐block FWHT与批量H128分组，并单独统计每次续跑的首query。
模型加载、显存和耗时反映当前decoded VQ实现，不作为native低位宽kernel或同输入速度结论。
