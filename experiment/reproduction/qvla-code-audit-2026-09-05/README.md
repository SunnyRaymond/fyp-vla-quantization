# QVLA 论文与公开代码一致性核查

核查日期：2026-09-05。对象为 **QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization**，不是 QuantVLA。

**结论：关于“公开代码与论文创新点不一致”的质疑有充分依据。最严重的是 action-space sensitivity 被替换为局部 input-covariance proxy，且公开路径仅实现 weight-only fake quantization。另有 allocator 公式差异和可复现的接口错误。证据不足以认定实验数据造假或作者主观欺骗。**

## 1. 版本、范围与证据等级

- 论文：[arXiv:2602.03782v1](https://arxiv.org/html/2602.03782v1)，2026-02-03；核查时 arXiv 仅列 v1。已有 [本地 PDF](../../../papers/vla/vla-quantization-literature-review-alternative/papers/07-qvla/paper-arxiv-v1.pdf)，本次公式定位使用在线 HTML 的 section / equation，不以 PDF 页码作为依据。
- 仓库：[AutoLab-SAI-SJTU/QVLA](https://github.com/AutoLab-SAI-SJTU/QVLA)，锁定 commit `26cc4821a3be4c003d09d3c7997b38db2a347982`；其 commit 日期为 2026-02-04。本次完整 clone 得到 main，无其他 remote branch 或 tag。
- 代码：完整阅读 `openvla/qvla/` 四个 Python 文件，并确认 `openvla-oft/qvla/`、`UniVLA/qvla/` 四文件与之逐字节相同；检查调用链与全仓库相关实现线索。
- 验证：[check_claims.py](check_claims.py) 直接调用未经修改的公开函数，PyTorch 2.10.0+cpu；输出见 [check-results.json](check-results.json)。没有运行 OpenVLA checkpoint、LIBERO rollout、RTX 4090 latency 或 VRAM benchmark，没有使用 NSCC GPU。
- OpenReview 两个 PDF 地址均返回 browser verification，故没有把其版本等同于 arXiv v1，也未对其评审记录作结论。
- 证据等级：A＝公开源代码及本地可复现行为；B＝论文主张与源代码的对应关系；C＝第三方 issue，仅证明存在公开质疑，不证明其指控真实。

## 2. 核心创新是否真正进入实现？

### F1：action-space sensitivity 的关键依赖缺失【A+B，严重】

论文 §3.2 Eq. (4)、§3.3.1、Appendix G 把 channel 的量化扰动与最终 action 联系起来，并提出 proxy screening 后的精确 forward 校准。[论文方法](https://arxiv.org/html/2602.03782v1#S3.SS3.SSS1)

实际执行链：

`image/text → 普通 forward → layer input hook → H → inverse Cholesky diagonal → row weight error → proxy.pt → greedy allocator`

具体代码：

- [sensitivity_hessian_proxy.py L83–116](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/sensitivity_hessian_proxy.py#L83-L116)：H 由该层输入外积累计，并加入 damping；没有 action Jacobian。
- [L131–148](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/sensitivity_hessian_proxy.py#L131-L148)：最终分数是按 inverse-Cholesky diagonal 加权的 row weight quantization error。
- [L253–281](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/sensitivity_hessian_proxy.py#L253-L281)：hook 只读取输入；forward 返回值未用于评分；没有逐 channel 修改后比较 action 的精确校准阶段。

**关键区别不是 Jacobian 与 Hessian 两个名称。** Appendix G 的二次型包含 action mapping 的 `J^T J`；代码的 H 来自本层输入，两者依赖的对象不同。`no_grad` 本身也不能证明错误，有限差分可以在 no_grad 下进行；这里的判断来自实际数据依赖和计算路径。

最小反例：保持被量化的 layer、weight、input 不变，只将下游 action mapping 的增益从 1 改为 10。真实 Action-MSE 从 0.072075 变为 7.207500，约增大 100 倍；公开 proxy 两次均为 0.582366。这个例子证明该 proxy 无法感知这种下游 action sensitivity 变化，**不代表测得了真实 OpenVLA 的失败率**。

这仍然是一种 data-dependent/channel-wise quantization heuristic；不能据此把整个实现称为空壳。但目前代码不支持论文核心的 action-centric 机制。

### F2：greedy allocation 没有计算论文的 marginal cost【A+B，明确差异】

论文 Eq. (8) 的分子为 demotion 前后的 sensitivity 差值。实现 [assign_gates_from_sensitivity.py L72–84](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/assign_gates_from_sensitivity.py#L72-L84) 直接使用较低 bit 的绝对 proxy，未减去当前 bit 的 proxy。

初始 16→8 时若 16-bit error 为零，两者可一致；后续 demotion 一般不等价。前一阶段输出的 proxy 是相对原始 weight 的误差，不能解释为预先算好的增量。

最小反例（两个 channel，目标平均 6-bit，16-bit error 均为零）：

| Channel | 8-bit error | 4-bit error | 8→4 增量 error |
|---|---:|---:|---:|
| A | 100 | 102 | 2 |
| B | 10 | 60 | 50 |

两者先到达 `[8,8]`。依据 Eq. (8) 应优先降低 A，得到 `[4,8]`；实际公开函数得到 `[8,4]`。本地测试已确认。

**反证检查：** min-heap 和跨阶段 enqueue 并非独立错误，Appendix B Algorithm 1 本身采用这种安排。正文的分阶段叙述和 Appendix 有表述差异，本报告不把 heap 当作缺陷；确定的差异是分子缺少减法。

## 3. 公开路径能否支持 W4A4、压缩与加速？

### F3：weight-only fake quantization，缺少相应 deployment 实现【A+B，严重复现缺口】

论文 Table 1 包含 W4A4/W8A8，并报告 OpenVLA-OFT 的 4.5 GB 与 1.49× speedup。[论文 Table 1](https://arxiv.org/html/2602.03782v1#S4.T1)

公开代码的行为：

- [inject_fake_w.py L53–99](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/inject_fake_w.py#L53-L99)：round/clamp 后立即乘回 scale，结果写入原来的 floating-point weight。
- [run_eval.py L47–69](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/run_eval.py#L47-L69)：GPU 路径加载 BF16 模型，再注入上述 fake quant，没有把 module 替换为 packed low-bit operator。
- 未在 QVLA 调用路径找到 activation quantizer、相应 calibration、mixed-bit packing 或 low-bit execution backend。
- 基础 OpenVLA 工具确有 `load_in_4bit/load_in_8bit` 的 bitsandbytes 选项；这些通用选项没有接入上述 QVLA 自行加载并注入的路径，不能用于解释其 per-channel `{0,2,4,8,16}` 实现。

最小 BF16 验证：给一个 3×4 Linear weight 分配 `[0,2,4]` 后，shape 仍为 `[3,4]`，dtype 仍为 BF16，weight tensor storage 仍为 24 bytes。它只改变数值。

**判断边界：** fake quantization 是正常的量化误差模拟技术，“fake” 不等于欺诈；但该实现不能仅靠这些操作产生宣称的 weight storage 压缩，也没有公开支撑上述真实 W4A4/W8A8 与 speedup 的执行路径。可能存在未发布的 backend，但本次没有证据确认其存在或结果。没有测量整模型 VRAM 或 latency，不能把这个 CPU 例子冒充硬件复现。

### F4：0-bit 实现为 weight 清零，未 structural prune【A，明确】

[inject_fake_w.py L82–99](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/inject_fake_w.py#L82-L99) 没有删除 channel、缩小矩阵或建立 sparse execution；bias 也未清零。测试中 0-bit row 的 bias=1，输出仍为 1。

这可作为权重扰动的模拟，不能据此声称已实现节省计算/存储的 structural pruning。对于无 bias 的 layer，bias 反例不适用，但 shape/storage 不变的问题仍成立。

## 4. README 流程本身存在可复现断点

### F5：allocator 输出不能直接作为 injector 输入【A，已运行验证】

- [allocator L136–144](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/assign_gates_from_sensitivity.py#L136-L144) 写出含 `proxy_pt`、`bits`、`assign`、`stats` 的外层对象。
- [loader L30–50](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/inject_fake_w.py#L30-L50) 期待直接 `{layer_name: bits}`，未提取 `assign`。
- 使用 allocator 相同格式的文件，实际报错 `new(): invalid data type 'str'`；把文件内容改为仅 `assign` 后可读取。

这证明文档步骤 2→3 不能原样衔接。它是工程错误；单独并不足以说明论文方法无效。

### F6：README 的 evaluation filename 不存在【A，已核实】

[README](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/README.md) 使用 `openvla/qvla/run_eval_with_qvla_fakew.py`；仓库实际文件为 `run_eval.py`。

## 5. 不应夸大的疑点

- **projector/action head 保持 BF16：** 与论文 §4.1 一致，不是遗漏创新的证据。action-aware 指评分依据，不能与“必须量化 action head”混同。
- **symmetric quantization：** 代码确实使用 symmetric per-row scale；但本次读取的 arXiv v1 没有明确的 asymmetric 要求。每 row 有 zero-point 不排除其固定为零。Issue #5 的这部分指控，本报告不作为已证实矛盾。代码实际上也是 per-row，并非 per-tensor。
- **calibration 没有 action label：** 单凭这一点不能判错，FP teacher 可以在线生成 action。更强的证据是实现根本不比较 action；缺少 labels 不是必要判断条件。
- **默认 32 个 image/text samples：** 与论文 Appendix F 的 trajectory calibration 描述存在复现材料缺口，但 CLI 参数可改变。仅比较默认样本数不足以证明用了不同实验数据。
- **全局最优：** greedy 方法不自动提供全局最优保证；但本次不把论文术语争议混入已实证的 implementation mismatch。
- **没有作者回复：** 不是实验造假的证据。

## 6. 已有公开讨论

核查时以下 issues 均处于 open。已保存 [issues.json](issues.json) 和 [comments.json](comments.json) 的 GitHub API 快照。#2/#5/#6 的评论中未见作者对这些技术问题的实质澄清；#4/#7/#8 评论数为 0。这里描述所检查的讨论，不推断私人沟通情况。

| Issue | 主题 | 对本报告的作用 |
|---|---|---|
| [#2](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/2) | Sensitivity Hessian | 公开质疑线索；具体矩阵依赖由代码独立确认 |
| [#5](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/5) | Wrong Implementation | 部分质疑获支持，asymmetric 部分没有照单全收 |
| [#4](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/4) | activation quantization | 与公开实现缺口一致 |
| [#6](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/6) | Speedup and memory | 要求解释 fake quant 与真实 deployment 的关系 |
| [#7](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/7) | mixed-precision kernel / BF16 baseline | 询问 backend、packing、structural pruning 与 benchmark protocol |
| [#8](https://github.com/AutoLab-SAI-SJTU/QVLA/issues/8) | AWQ / SmoothQuant baseline | baseline 配置与 backend 的复现疑问 |

#1 中有人提供自称从实验室获得的 calibration 脚本及个人复现体验。这些是第三方转述，未升级为作者官方发布或可控的独立 replication。

## 7. 对 FYP 使用的具体影响

可以把 QVLA 作为 **action-aware/channel-wise bit allocation 的论文思路** 研究；若引用实验表现，应标注为作者报告结果。当前 release 应描述为 **local-covariance proxy + greedy allocation + weight-only fake quantization**，不要直接标成论文算法的 faithful reproduction。

不宜直接用此 release 的低成功率反驳论文算法，也不宜用它的 fake-quantized checkpoint 验证论文 deployment gain。修好 JSON/file-name 两处错误只解决入口问题，不能补齐 action sensitivity、activation quantization 或 runtime。

向作者澄清时，最有价值的问题是：

1. 与 Eq. (4)/(6) 对应的 action Jacobian、teacher-action comparison、精确 sensitivity refinement 在哪个 commit/file？
2. Eq. (8) 的 sensitivity difference 为什么在当前 allocator 中被替换为 absolute proxy？
3. Table 1 使用何种 activation quantizer、packed mixed-bit backend、0-bit handling，以及何种 timing/VRAM 测量定义？
4. 能否提供生成表格的 gates/checkpoint、calibration split、baseline configurations 和可直接执行的完整命令？

本报告未联系作者、未发布 issue，也未修改上游代码。由 AI 辅助完成代码审查与 CPU 反例验证，未进行完整 benchmark replication。
