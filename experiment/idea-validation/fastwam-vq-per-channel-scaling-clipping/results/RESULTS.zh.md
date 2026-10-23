# Per-channel scaling + clipping80

原baseline初始化，C80/原loss/seeds/teacher固定；没有新增teacher或development set。
每组原held-out为40 contexts ×2 seeds=80 queries，四suites等权；另完成每组C80的80-query端点评估。

| Held-out配置 | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| baseline | 0.044296774 | 0.060877371 |
| clip_only | 0.043848437 | 0.054848124 |
| b32_scale_clip | 0.043986743 | 0.054151555 |
| continuous_b32_clip | 0.044588993 | 0.060690613 |
| coupled_b32_clip | 0.043394818 | 0.054067248 |
| per_channel_clip | 0.031641466 | 0.055377875 |

新方法相对对照，负值为改善：

- vs baseline: action -28.569%，latent -9.034%。
- vs clip_only: action -27.839%，latent +0.966%。
- vs b32_scale_clip: action -28.066%，latent +2.265%。
- vs continuous_b32_clip: action -29.037%，latent -8.754%。
- vs coupled_b32_clip: action -27.085%，latent +2.424%。

相对 baseline/clip_only/coupled B32 两项均改善：False。
Late22/full32/gripper仅诊断，不进入训练loss或推进门槛。
评估使用实际continuous scale补偿后的BF16 weights；codebooks/indices/row gains/original exponents冻结；uncached/cached完整query输出一致；单独保存FP32 log2 scale metadata。
仅fixed-input decoded-BF16 VQ+A4数值证据，无native kernel或closed-loop收益声明。
