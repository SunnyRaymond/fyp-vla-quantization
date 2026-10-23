# B32 grouped scaling + clipping80

原baseline初始化，C80/原loss/seeds/teacher固定；没有新增teacher或development set。
每组原held-out为40 contexts ×2 seeds=80 queries，四suites等权；另完成每组C80的80-query端点评估。

| Held-out配置 | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| baseline | 0.044296774 | 0.060877371 |
| clip_only | 0.043848437 | 0.054848124 |
| b32_scale_clip | 0.043986743 | 0.054151555 |

新方法相对对照，负值为改善：

- vs baseline: action -0.700%，latent -11.048%。
- vs clip_only: action +0.315%，latent -1.270%。

相对两组对照两项均改善：False。
Late22/full32/gripper仅诊断，不进入训练loss或推进门槛。
评估使用实际折入B32 exponents的weight decode；codebooks/indices/row gains冻结；factorized/folded全query输出已验证一致。
仅fixed-input decoded-BF16 VQ+A4数值证据，无native kernel或closed-loop收益声明。
