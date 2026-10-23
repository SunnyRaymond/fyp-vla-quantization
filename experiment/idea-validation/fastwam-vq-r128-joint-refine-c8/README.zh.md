# C8 calibration 条件触发的 R 步长改进与实现对照

本轮已完成。保留原 **H+VQ8** incumbent；目前没有证据支持用这次训练得到的 R128 替换 H128。完整机器聚合结果见 [RESULTS.zh.md](results/25904715.pbs101/RESULTS.zh.md) 和 [comparison.json](results/25904715.pbs101/comparison.json)。

## 去掉 VQ 后，R128 是否优于 H128？

这里去掉的是 weight VQ，activation 仍为 dynamic per-token A4；W16 指变换后的 BF16 权重。两臂共享冻结的 Smooth0.5、signs、D=I、r=1、BF16 KV、输入和 teacher。R8 是原先在 W16A4 下用 C8、Adam LR=1e-3 学习 8 次的 R，不是从 VQ 权重反推得到的 R。

| 数据与指标（越低越好） | H128 W16A4 | R8 W16A4 | Heldout 相对变化 |
|---|---:|---:|---:|
| C8 first10 motor RMSE | 0.044464021 | 0.040792220 | — |
| C8 future latent NMSE | 0.039335942 | 0.034579013 | — |
| C8 mixed loss | 0.003835097 | 0.003145532 | — |
| Heldout first10 motor RMSE | 0.042604556 | 0.044585121 | +4.649% |
| Heldout future latent NMSE | 0.038380617 | 0.040343879 | +5.115% |
| Heldout mixed loss | 0.003474206 | 0.003740738 | +7.672% |
| Heldout full32 motor RMSE | 0.045009860 | 0.046935371 | +4.278% |

Heldout 每臂完整 40 contexts × 2 seeds，先平均 context 内 seeds，再平均 suite 内 10 contexts，最后四 suite 等权。Motor RMSE 有 12/40 contexts 改善、28/40 退化，四个 suite 的 motor RMSE 和 mixed loss 均升高。Latent NMSE 有 23/40 contexts 改善，不能把总体均值上升解释成每个 context 都变差；较大退化可以抵消更多的小幅改善。

直接结论是：**当前 R8 改善 calibration 拟合，但没有泛化到 heldout；进入 VQ 之前已经发生退化。** 因而不能把原 R8+VQ8 的全部问题归因于 VQ refitting 或 R/VQ 分开训练。

如果完全去掉 activation 和 weight quantization，在理想实数运算、权重与输入配对变换下，`(X S^-1 R) (W S R)^T = X W^T`。因此学习正交 R 的价值在于改变量化误差；R 本身没有保证更优输出的性质。本实验仍有 A4 以及 BF16 rounding，不是全精度等价性比较。

更有依据的解释是 calibration 泛化不足，并伴随需要控制的优化问题：训练只覆盖 8 contexts、每个一次更新；两个 stream-shared 128×128 Cayley residual 从 H 初始化后共同影响许多层；A4 backward 使用 identity STE，不对实际 rounding/absmax 阈值变化给出精确梯度。训练集 loss 降低与 heldout 改善之间没有保证。目前未隔离 calibration、R LR=1e-3、STE 和共享粒度各自的贡献，不能认定某一个是唯一根因。已有数值等价性、梯度连接和正交检查没有发现 R 未接入或旋转失效。

## Joint8 与降低 R LR 的最终判断

以下均为完整 80-query heldout；VQ LR=5e-4、8 次更新，其余固定协议相同。`H_currentpath` 使用当前 JointBanks 路径，R=I 全程冻结。

| Arm | First10 motor RMSE | Future latent NMSE | Mixed loss |
|---|---:|---:|---:|
| 原 H+VQ8 incumbent | 0.044296774 | 0.060877371 | 0.004598652 |
| 原 R8→refit→VQ8 | 0.046408990 | 0.065187050 | 0.004980800 |
| Joint8，R LR=1e-4 | 0.045524040 | 0.060773535 | 0.004718190 |
| H_currentpath，R 冻结 | 0.045305496 | 0.065714797 | 0.004894043 |
| Joint8，R LR=1e-5 | 0.045226322 | 0.065914399 | 0.004893894 |

- Joint8 LR=1e-4 比原 sequential recipe 的 motor RMSE/mixed loss 低 1.907%/5.272%，但比原 H+VQ8 高 2.771%/2.599%。联合训练有局部改善，没有超过 incumbent。
- 降至 R LR=1e-5 后，同一训练 context 更新后 mixed loss 下降的次数由 3/8 变为 7/8，优化更稳定。但相对 H_currentpath 的 heldout mixed loss 仅低 0.003%，motor RMSE 低 0.175%，latent NMSE 高 0.304%；full32 motor RMSE 还高 7.033%。不能据此宣称 learned R 获得可靠收益。
- H_currentpath 相对旧 H+VQ8 的 heldout mixed loss 高 6.423%，提示训练实现与数值重现也需要关注。连同一路径、U=I 起点的首步 VQ gradient norm 也存在小幅差异，因此不能将该差距全部归因于新增路径的确定性偏差。现有 VQ backward 使用 CUDA `index_add_`；[PyTorch 文档](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_add_.html)说明该操作可能非确定性。这只是候选来源，尚未实验证实为本轮差异的原因，更不能用于解释全部 W16A4 退化。

下一步建议先做同起点梯度/单步 replay 与 numerical reproducibility 检查，再设计独立 calibration-validation 和适度的 R 正则/步长控制。不要先靠增加 updates 或扩大 joint sweep 寻找赢家。本轮计划已结束，没有启动这些后续实验，也没有改变 C8、loss、D 或 incumbent。

## 完成状态与证据边界

两次新增 GPU 作业 `25903221.pbs101` 与 `25904715.pbs101` 均在 g1 完成，PBS/产物 Exit0、PHASE_COMPLETE，walltime 分别 01:17:10、00:41:48；两者均申请 4h，完成后立即退出。最终 inference parity 为 0，参数全部冻结；完整计数和 GPU numerical selfcheck 已通过。20 分钟 subagent 监控均已停止。没有 H+VQ16、没有按 heldout 挑 checkpoint、没有新 teacher queries。

证据仅限固定输入、BF16 transformed/decoded weights 与 A4 的数值 fidelity；没有环境 episodes、closed-loop success 或 native low-bit kernel latency，也没有额外重复训练来估计 run-to-run 方差或统计显著性。

## 原定执行方案

来源实验 `25903221.pbs101` 已完成：原 R8 在 W16A4 heldout motor RMSE/latent NMSE 上分别较 H128 高 4.65%/5.12%，而 C8 指标改善。第一轮 Joint8（R LR=1e-4）比原 H+VQ8 的 heldout motor RMSE 高 2.77%；其 C8 mixed loss 为 0.0045157388667576015，H+VQ8 为 0.004196602109004743。

根据首轮运行期间、查看 Joint 结果前（13:10:49 UTC）固定的 C8-only 条件，执行一档 R LR=1e-5 的 Joint8。保持原 H+VQ8 incumbent、C8、ρ=1.5、loss、Smooth、D=I、r=1、VQ LR=5e-4、8 updates、初始 H+VQ0 banks 和 U=I。固定使用第 8 次更新；不按 heldout 挑选 checkpoint。

额外运行 `H_currentpath`：同一 JointBanks forward/backward 实现，U=I 全程冻结，仅进行 8 次 VQ updates。首轮 Joint0 的 VQ gradient norm 是 0.02929229848，旧 H+VQ0 首步为 0.02933440171；该对照量化实现数值差异，不能仅凭 norm 差异认定根因。

实际新增训练只有 H_currentpath 和 Joint8 两臂，每臂完整 C8 evaluation 8 条和 heldout 40 contexts × 2 seeds。第一轮所有诊断和原 Joint8_R1e4 完整复用；最终总表包含 7 个 heldout arms 与 9 个 calibration arms。原 H+VQ8 保留，另提供与当前实现对照的比较。

训练仍使用实际 BF16 decoded VQ + dynamic per-token A4；仅 codebooks/row gains 与候选 R 可训练，indices/B32 exponents、Smooth、原模型、D、clipping 冻结。Joint8 是 compressed student 的共同适配，不要求每次更新保持原 FP 函数等价；最终无 refit。

PBS 请求 normal / 1 GPU / 16 CPU / 110 GB / 4h，串行运行，实际队列以 qstat 为准。GPU utilization/VRAM 每 15 秒记录；subagent 每 20 分钟监控。Login node 仅做轻量控制；数值工作和重 I/O 全部在真实 PBS allocation 内。

入口为 control.py upload/submit/status/verify/pull/resume。Resume 保存两臂各自 masters、R、optimizer、RNG 与 update cursor；保留已完成 query。证据仅为固定输入数值比较，不包含 closed-loop 或 native-kernel 性能。
