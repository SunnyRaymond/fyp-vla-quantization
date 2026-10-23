# Diffusion W4A4 Baselines：LIBERO-Plus 闭环评测

状态：complete；新增 episode：560/560；frozen references：2240/2240。

本轮比较 SVDQuant adapted W4A4 与 ViDiT-Q adapted W4A4。权重和激活以 decoded BF16 形式模拟，KV cache 为 BF16；SVDQuant 低秩 residual 按预算保留 BF16 factors。
报告中的 timing 按每个 arm 的 transform implementation 分开统计，包含完整 action query 和 GPU denoising interval；不代表 native low-bit kernel 的速度或显存。

| Arm | Successes / episodes | Success rate |
|---|---:|---:|
| svdquant_w4a4 | 108/280 | 38.57% |
| viditq_w4a4 | 203/280 | 72.50% |

与 8 个 frozen reference arms 及两个新 arm 之间的 paired wins/losses/ties，以及 suite、dimension、suite×dimension 明细见 summary.json。

## 本轮结论与匹配预算

| Arm | Successes /280 | Success rate | Target effective bpw |
|---|---:|---:|---:|
| bf16 | 216 | 77.14% | reference |
| spinquant_adapted_w4a4 | 203 | 72.50% | reference |
| micro_scale_vq_a4 | 204 | 72.86% | reference |
| rho1.5 | 206 | 73.57% | reference |
| svdquant_w4a4 | 108 | 38.57% | 4.4250 |
| viditq_w4a4 | 203 | 72.50% | 4.3250 |

ViDiT-Q adapted 与 SpinQuant 总成功数相同，配对为9胜/9负/262平；与micro-scale VQ为13胜/14负/253平，少1个成功。相对BF16为4胜/17负/259平，少13个成功。SVDQuant adapted为108/280；与ViDiT-Q相比0胜/95负/185平，它的成功案例在本次cohort中均包含于ViDiT-Q成功集合。

| Suite (each70) | SVDQuant adapted | ViDiT-Q adapted | BF16 | SpinQuant | Micro-scale VQ |
|---|---:|---:|---:|---:|---:|
| libero_10 | 25 | 52 | 54 | 49 | 51 |
| libero_goal | 10 | 48 | 51 | 48 | 46 |
| libero_object | 35 | 49 | 55 | 51 | 51 |
| libero_spatial | 38 | 54 | 56 | 55 | 56 |

SVDQuant适配版的最大suite落差出现在libero_goal（10/70，ViDiT-Q为48/70）。这只是本轮已冻结recipe的观察，未做参数归因。

| Perturbation (each40) | SVDQuant adapted | ViDiT-Q adapted | BF16 | SpinQuant | Micro-scale VQ |
|---|---:|---:|---:|---:|---:|
| background_textures | 7 | 26 | 26 | 23 | 24 |
| camera_viewpoints | 4 | 14 | 21 | 16 | 19 |
| language_instructions | 28 | 37 | 38 | 36 | 38 |
| light_conditions | 23 | 37 | 35 | 39 | 35 |
| objects_layout | 21 | 34 | 35 | 33 | 34 |
| robot_initial_states | 18 | 32 | 34 | 31 | 30 |
| sensor_noise | 7 | 23 | 27 | 25 | 24 |

ViDiT-Q相比SpinQuant：background +3、language +1、objects +1、robot +1；camera -2、light -2、sensor -2，净变化0。相比BF16，camera少7个成功，是主要净缺口。

## 运行开销与有效性

单一4GPU任务25878591.pbs101实际运行08:48:26，scheduler F/Exit_status=0、artifact exit0、4×140 worker rc0、560新episodes和2240references检查全部通过。四卡各加载一次模型。原C80校准共80次policy queries，另4次BF16 anchor与4次quantized cross-worker anchor；正式episode之前的这些queries不计入下表。每15秒的GPU util/显存记录保留在远端job.log中。

| Arm | Later formal queries | Full query mean / median (ms) | Peak allocated / reserved (decimal GB) |
|---|---:|---:|---:|
| svdquant_w4a4 | 10480 | 2304.04 / 2286.91 | 25.2378 / 25.5391 |
| viditq_w4a4 | 6990 | 5854.55 / 5848.81 | 25.1155 / 25.4175 |

两种配置的query数不同，episode也受控制动作和成功/失败轨迹影响；上述timing只描述各自decoded-BF16 reference实现。有效bpw反映编码存储，不等于runtime VRAM。

## 不能外推的范围

本轮是为全614层W4A4和相似bpw实现的Fast-WAM adaptations，不能作为完整官方recipe复现结论：

- SVDQuant：global rank16（611层rank16、2层rank7、1层rank8），固定alpha0.5，单次randomized truncated SVD；没有DeepCompressor smoothing grid search和迭代low-rank refinement。
- ViDiT-Q：alpha按author code convention0.5665，保留channel balancing、Hadamard和token-dynamic A4；未启用W8A8/FP保护层。249层full power2、334层full H12×power2、31层dyadic blocks；其中30个width14336使用H8192/H4096/H2048，另1个width7使用H4/H2/H1，未使用author utility中的full H28。
- 主路径weight group128、FP32 scales；SVD的BF16 factors也计入4.4250 bpw，ViDiT zero points/masks/signs计入4.3250 bpw，实际差2.314%。
- 已冻结的280例各执行一次；没有用Plus结果选alpha/rank/rotation，也没有增加rollout重复或显著性检验。

因此可以确认本轮ViDiT-Q适配版的结果接近既有SpinQuant/micro-scale VQ；SVDQuant适配版明显较低。尚未测量完整官方SVDQuant recipe在Fast-WAM上的表现，也没有从这次结果证明其一般性能。Primary sources与完整配置边界见README.zh.md。

完成证据：FINAL_COMPLETION_EVIDENCE.json；详细配对统计：summary.json；encoded banks及原始episode JSONL保留在远端。30min Luna监控已在终态成功后停止。
