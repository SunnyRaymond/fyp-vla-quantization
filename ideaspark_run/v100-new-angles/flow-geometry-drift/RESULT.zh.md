# Flow Geometry under PTQ：保留的最小结果

2026-09-13。**数值 screen：scope_limited_preliminary_go；工程/复现状态：inconclusive_provenance（报告细节丢失）。停止本切面扩验。**

12 个 LIBERO episodes、每个两个固定 noise seeds、FP32 / backbone_W4 / expert_W4 三臂的 raw outputs 已由 CCDS GPU64763 算完。该 job 在最终 JSON 写入时超过 64KiB 报错，因此 scheduler 仍是 FAILED，不能改写成成功作业。CPU64764 仅复算已保存的完整数组，COMPLETED；没有再次运行模型。

| Quantized locus | Q geometry vs drift 的 Spearman | FP geometry baseline | Q action-norm baseline | 冻结数值 gate |
|---|---:|---:|---:|---|
| language backbone W4 | 0.629371 | 0.349650 | 0.146853 | preliminary_go |
| action expert W4 | 0.181818 | 0.244755 | 0.223776 | mechanism_no_go |

门槛为 Q geometry 的相关至少 0.5，且较两项 baseline 分别至少高 0.1。先在每个 episode 内平均两个 seeds，再对 12 episodes 做 average-tie-rank Spearman；不能将 24 次 draws 当作独立 episodes。没有调节样本、bit、threshold 或 noise。

本结果仅支持：在这批输入和此 RTN recipe 下，现成 flow geometry 信号对于 backbone W4 的 normalized action drift 排序有初步信息，未在 expert W4 下复现。两种 locus 的参数规模和误差预算不同，不能据此推断 intrinsic sensitivity，更不能声称机器人 failure detection、可靠性保证、task success 改善或 native INT4 speedup。

## 实验与证据

- checkpoint：`lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`；dataset：`lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。使用 LeRobot 0.4.4 official policy 和保存的 preprocessor。存储权重转 FP32 运行；不能声称恢复了训练前完整 FP32 数值。
- 固定前 4 个 task、每 task 前 3 个 episode、frame=floor(length/4)，两个真实 cameras，按 checkpoint rename map 处理，保留与 saved normalizer 一致的 8 维 state。10 denoising steps、50-step chunk；指标只取预定前 8 denoising steps、前 8 chunk positions、7 个 physical action coordinates。
- W4 仅量化预定 locus 内 Linear weights，signed [-7,7] per-output RTN 后 dequant FP32，其余保持 FP32。未做训练或 calibration search。
- GPU64763 为 TC1N03 的 Tesla V100-PCIE-32GB，scheduler 1m29s；CPU64764 用 1s 完成 raw replay。早期工程失败64760/64761/64762和CPU准备失败均保留；没有将它们计为科学 no-go。
- 原始数组 SHA256：`eb33c5c6324be68025a7a947bfe645d243bad930d0540823d56588d66b87eece`，保留于 compute storage 的 `artifacts/64763/raw_flow.npz`。小产物见 [CPU verification](artifacts/64764/verification.json) 和 [原始失败记录](artifacts/64763/summary.json)。

## 报告恢复边界

64763 专属 allocation、GPU identity、asset identity brief 仍在。CPU 复核原始 schema、完整性、finite values、manifest episode binding 和 sample hashes。原 GPU 保存的 full asset identity SHA 与当前 identity 相同；CPU 对作业快照计算的 runner SHA 与本地审查过的控制源码相同：`ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9`。

异常发生在所有输出、weight restore、bypass-state checks、no-op check 和指标计算之后的最终写入；据 source+trace 只能将此前 gates 记为 **control_flow_gate_pass_inferred**。完整 runtime summary、逐层 quantizer 数值、bypass digest、当次实际 import 的 package source hash 值及原 GPU metrics 没有成功落盘，不能冒称已恢复，也没有验证 GPU/CPU metrics agreement。这里的数值判定来自 CPU 对 raw arrays 的独立计算；复现证据不足的部分随结果保留，不以另一个作业的记录代替。

完整工程判定应同时阅读 [恢复审查](RECOVERY_AUDIT.zh.md)。不因报告缺口扩大到任务 rollout，也不为重新生成日志重复本轮 inference。

## Prior boundary 与终止

Geometry 信号本身已有 [The Geometry of Flow Matching Uncertainty](https://arxiv.org/abs/2607.27933)；curvature 与 PTQ calibration 也已有 [Zero-Shot Quantization via Trajectory Curvature and Attention Guidance](https://openreview.net/pdf/dcfe09ea1e77595f612807a5ebc8a92b39c6456b.pdf)。本轮不提出新 curvature metric 或量化算法，只问已量化 policy 的该信号能否排序 FP/Q drift。

expert no-go 没有表明某个局部实现错误：所有 arms 使用同一 recorder 与预处理，报告序列化错误发生在数值计算之后，无法解释两种 locus 的差别。故不为 expert 重选指标或重新调参；backbone 的正向信号也只保留，不深入验证。
