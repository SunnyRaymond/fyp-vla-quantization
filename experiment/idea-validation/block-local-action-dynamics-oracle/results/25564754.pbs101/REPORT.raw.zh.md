# 已知正确坐标的 toy oracle 对照结果

本轮在已分配 PBS compute node 完成 27 个主 runs 与 3 个 lowrank global16 附加 runs。模型直接使用生成器正交矩阵 `M`，固定 `s=(z-mean_z)@M`，只训练 predictor/message。它回答：第一轮的误差有多少来自坐标学习，已知正确坐标后局部模型、有限通信和 dense predictor 的差异如何。Oracle 得到生成器坐标，属于诊断对照，不是可部署方案。

主实验的 train/dev/test 为 512/128/128 episodes，三个 seeds，各训练 1500 steps。误差使用 train-only delta energy 归一化；h10/h20 是 free-running 末步误差。action-response 是以真实 response energy 归一化的扰动响应差。延迟是在同一 allocation 重测 oracle 与原 dense checkpoint，计入初次坐标变换和最终重构。

## 主实验

| Condition | Arm | h10 | h20 | Action-response | Params | Dense quality seeds | B1 ms | B300 ms | Each-step B300 ms | Speed gate B1/B300 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | oracle_block | 0.07520 | 0.06218 | 0.0928 | 27200 | 3/3 | 2.664 | 2.937 | 3.339 | 0/3; 0/3 |
| independent | oracle_global4 | 0.07498 | 0.06263 | 0.0930 | 28480 | 3/3 | 3.045 | 3.337 | 3.676 | 0/3; 0/3 |
| independent | oracle_local4 | 0.07442 | 0.06256 | 0.0927 | 28480 | 3/3 | 3.088 | 3.414 | 3.783 | 0/3; 0/3 |
| lowrank_coupled | oracle_block | 0.14619 | 0.08783 | 0.0946 | 27200 | 1/3 | 2.682 | 2.969 | 3.355 | 0/3; 0/3 |
| lowrank_coupled | oracle_global4 | 0.07708 | 0.06378 | 0.0935 | 28480 | 3/3 | 3.056 | 3.344 | 3.690 | 0/3; 0/3 |
| lowrank_coupled | oracle_local4 | 0.14614 | 0.08837 | 0.0944 | 28480 | 1/3 | 3.097 | 3.399 | 3.800 | 0/3; 0/3 |
| dense_coupled | oracle_block | 0.17918 | 0.11071 | 0.0974 | 27200 | 0/3 | 2.684 | 2.966 | 3.348 | 0/3; 0/3 |
| dense_coupled | oracle_global4 | 0.14390 | 0.08688 | 0.0964 | 28480 | 3/3 | 3.052 | 3.329 | 3.667 | 0/3; 0/3 |
| dense_coupled | oracle_local4 | 0.17903 | 0.11040 | 0.0973 | 28480 | 0/3 | 3.088 | 3.406 | 3.782 | 0/3; 0/3 |

质量门槛沿用第一轮 `h10 <= dense + max(0.1*dense, 0.02)`；速度门槛沿用完整 rollout 至少快 20%，B1 和 B300 分开判断。表中参数数不计入固定的 64×64 坐标矩阵。

## 与第一轮 learned arm 的逐 seed 配对差分

差分为 oracle 减原 learned arm；负值表示 oracle 误差较低。区间是每个训练 seed 内按 episode bootstrap 得到的 95% CI；它不是三个训练 seed 的总体区间。

| Condition | Seed | Oracle arm - learned arm | Mean | Episode bootstrap 95% CI | Oracle - dense mean | Oracle - dense 95% CI |
|---|---:|---|---:|---|---:|---|
| lowrank_coupled | 1102 | oracle_local4 - learned_local4 | -0.137144 | [-0.150778, -0.123397] | 0.014656 | [0.006435, 0.023777] |
| independent | 1101 | oracle_block - learned_block | -0.178006 | [-0.187993, -0.168629] | -0.049553 | [-0.054715, -0.044777] |
| independent | 1102 | oracle_global4 - learned_global4 | -0.107444 | [-0.114061, -0.101056] | -0.052243 | [-0.056857, -0.047738] |
| lowrank_coupled | 1103 | oracle_global16 - learned_global16 | -0.025078 | [-0.027703, -0.022659] | -0.045852 | [-0.049886, -0.041901] |
| lowrank_coupled | 1103 | oracle_block - learned_block | -0.110648 | [-0.122831, -0.098101] | 0.024681 | [0.016366, 0.033428] |
| lowrank_coupled | 1102 | oracle_block - learned_block | -0.109351 | [-0.121050, -0.097559] | 0.014284 | [0.005240, 0.022644] |
| lowrank_coupled | 1101 | oracle_global16 - learned_global16 | -0.030762 | [-0.033663, -0.028006] | -0.049318 | [-0.054652, -0.044690] |
| dense_coupled | 1101 | oracle_block - learned_block | -0.112141 | [-0.124644, -0.101024] | 0.048894 | [0.042179, 0.055711] |
| dense_coupled | 1103 | oracle_block - learned_block | -0.104892 | [-0.118502, -0.092503] | 0.050938 | [0.044427, 0.057605] |
| lowrank_coupled | 1103 | oracle_local4 - learned_local4 | -0.128119 | [-0.141307, -0.115043] | 0.024911 | [0.015833, 0.035090] |
| independent | 1102 | oracle_block - learned_block | -0.171074 | [-0.179585, -0.161927] | -0.052310 | [-0.057051, -0.047456] |
| dense_coupled | 1103 | oracle_global4 - learned_global4 | -0.052863 | [-0.060373, -0.045077] | 0.017384 | [0.011677, 0.023059] |
| independent | 1101 | oracle_local4 - learned_local4 | -0.179077 | [-0.188581, -0.170025] | -0.051219 | [-0.056360, -0.046114] |
| independent | 1103 | oracle_block - learned_block | -0.168197 | [-0.177041, -0.158920] | -0.047407 | [-0.051628, -0.043335] |
| lowrank_coupled | 1102 | oracle_global4 - learned_global4 | -0.116562 | [-0.123770, -0.109405] | -0.051737 | [-0.056667, -0.047377] |
| lowrank_coupled | 1101 | oracle_local4 - learned_local4 | -0.133861 | [-0.146373, -0.120224] | 0.020024 | [0.012032, 0.028853] |
| dense_coupled | 1103 | oracle_local4 - learned_local4 | -0.112312 | [-0.125421, -0.099866] | 0.051368 | [0.044910, 0.057382] |
| dense_coupled | 1102 | oracle_block - learned_block | -0.113373 | [-0.125402, -0.102096] | 0.044096 | [0.037495, 0.050521] |
| lowrank_coupled | 1102 | oracle_global16 - learned_global16 | -0.029231 | [-0.032039, -0.026397] | -0.052551 | [-0.057366, -0.047772] |
| independent | 1103 | oracle_global4 - learned_global4 | -0.115895 | [-0.123160, -0.109001] | -0.046989 | [-0.051213, -0.042830] |
| lowrank_coupled | 1101 | oracle_block - learned_block | -0.137965 | [-0.152137, -0.123693] | 0.020799 | [0.012391, 0.029569] |
| independent | 1101 | oracle_global4 - learned_global4 | -0.114048 | [-0.120805, -0.106292] | -0.050674 | [-0.055874, -0.046019] |
| dense_coupled | 1102 | oracle_global4 - learned_global4 | -0.058437 | [-0.065383, -0.051557] | 0.009301 | [0.003197, 0.014991] |
| lowrank_coupled | 1101 | oracle_global4 - learned_global4 | -0.122873 | [-0.130890, -0.115446] | -0.049924 | [-0.055078, -0.045206] |
| dense_coupled | 1101 | oracle_global4 - learned_global4 | -0.056842 | [-0.064157, -0.048855] | 0.011389 | [0.005380, 0.016829] |
| independent | 1103 | oracle_local4 - learned_local4 | -0.187014 | [-0.195912, -0.177947] | -0.047782 | [-0.051550, -0.043358] |
| dense_coupled | 1101 | oracle_local4 - learned_local4 | -0.095709 | [-0.105949, -0.085378] | 0.047104 | [0.040716, 0.053447] |
| dense_coupled | 1102 | oracle_local4 - learned_local4 | -0.112593 | [-0.124521, -0.100747] | 0.044985 | [0.038436, 0.052135] |
| lowrank_coupled | 1103 | oracle_global4 - learned_global4 | -0.120164 | [-0.127721, -0.112634] | -0.045900 | [-0.049851, -0.042098] |
| independent | 1102 | oracle_local4 - learned_local4 | -0.182995 | [-0.193071, -0.173922] | -0.052602 | [-0.057439, -0.048182] |

## global16 附加对照

它只在 lowrank_coupled 做三个 seeds。message 从 4D 增至 16D，输入层也相应扩大，参数量随之上升；即使结果更好，也不能区分通信维数和容量的作用。

| Seed | h10 | Params | Quality gate | B1 reduction | B300 reduction |
|---:|---:|---:|---|---:|---:|
| 1103 | 0.07750 | 32320 | True | -108.9% | -117.1% |
| 1101 | 0.07715 | 32320 | True | -110.2% | -117.5% |
| 1102 | 0.07644 | 32320 | True | -109.8% | -117.0% |

## 机制与原结果复现

机制检查通过 4/4。复载并复算第一轮 learned/dense 与 global16 原 checkpoint 的全部 held-out horizon arrays 共 39 份；最大绝对差为 0。逐模型值见 `reference_reproduction.json`。

## 怎样解释这轮结果

- 如果 oracle_block 明显优于第一轮 learned_block，且误差接近 dense，第一轮的重要瓶颈是从混合 latent 学坐标；这是已知 M 条件下的诊断上限。
- 如果 oracle_block 的坐标已正确却仍落后 dense，局部 predictor 的函数形式、容量或当前优化预算仍不足。此结果不能简单归因于坐标学习。
- oracle_global4 与 oracle_local4 在正确坐标下的差异，检验全局摘要的信息通路是否直接帮助 dynamics；oracle_global4 与 oracle_block 的差异显示通信摘要的净收益。仍需结合各 condition 的差异看，而不是把某个 seed 的优势泛化。
- `dense_coupled` 是更广泛的交互压力测试，不证明真实视觉 dynamics 一定 dense 或不可分。
- global16 是参数与通信同时增加的附加探索。全部结果均不支持视觉、C-SWM/LeWM、CEM、部署加速或闭环结论；是否进入视觉 pilot 需由并行视觉实验单独回答。

全部逐 episode arrays、checkpoint、training.tsv 和 helper/source snapshot 保存在本 PBS run 目录。机制文件、JSON 摘要与本报告由 job 输出。GPU 利用率和显存按 30 秒采样写入 job.log。

实验设计来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065

PBS job `25564754.pbs101`，compute host `x1000c0s1b0n0`。
