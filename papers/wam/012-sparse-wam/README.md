# 24. Sparse-WAM：动作引导的稀疏想象

**Sparse-WAM: Accelerating World Action Models via Action-Guided Sparse Imagination**  
本地版本：**arXiv:2609.38984v1** · 2026-09-30 提交 · 16 页 · 阅读状态：`unread`  
分类：WAM inference efficiency / online token sparsification

[本地固定版 PDF](paper-arxiv-v1.pdf) · [arXiv v1 页面](https://arxiv.org/abs/2609.38984v1) · [官方 HTML 全文](https://arxiv.org/html/2609.38984v1) · [arXiv PDF](https://arxiv.org/pdf/2609.38984v1)

## 一句话读懂

Sparse-WAM 在联合生成视频未来与动作的 WAM 中，观察“动作 token 会关注哪些未来图像 token”，每个 action chunk 先做一次完整计算并据此选择未来视觉 token，之后 denoising steps 只重新计算保留的视觉 token；省略位置仍用最近一次完整步骤缓存的视觉预测参与 sampler 更新，动作 token 每一步照常计算。稀疏化本身无需额外训练，优化目标是减少每个 chunk 的推理计算。

## 背景与直觉

不少 diffusion WAM 在一个 action chunk 的多步 denoising 中，反复更新未来图像 latent 和动作。高分辨率未来图像会产生大量视觉 token，而控制动作只依赖其中一部分区域。Sparse-WAM 的出发点是：从 action query 到 future-frame key 的注意力分布里，能否找到与动作相关的区域，并在相邻 denoising steps 之间复用这份选择？

作者报告连续 denoising steps 的 action-to-future 注意力图有较高空间重合，但未来表示仍持续变化。跨步重合支持复用选择，不能解读为未来表示不再变化，也不能单独证明注意力是因果重要性。方法同时保留两类位置：每帧单独变化的 action-relevant **core tokens**，以及所有未来帧共享空间位置的 **anchor tokens**，用来保留较稳定的上下文。

## 方法：从注意力分数到稀疏执行

令未来 latent 有 \(F\) 帧，每帧 \(N_s\) 个空间 token；action chunk 有 \(H\) 个 action tokens。第 \(\ell\) 层、第 \(\tau\) 个 denoising step 中，\(A_{\ell,h}^{(\tau)}(i,f,j)\) 是 action query \(i\) 对未来帧 \(f\) 的位置 \(j\) 的 attention。作者先对 attention heads 求平均：

\[
U_{\ell}^{(\tau)}(i,f,j)=\frac{1}{N_h}\sum_{h=1}^{N_h}A_{\ell,h}^{(\tau)}(i,f,j).
\]

假设 \(H\) 可被 \(F\) 整除，方法把 action queries 分成 \(F\) 组，每组对应一个 future frame。每帧的空间分数由对应 queries 加权求和得到：

\[
S_\ell(f,j)=\sum_{i\in\mathcal I_f}\alpha_{f,i}U_\ell(i,f,j),\qquad \sum_{i\in\mathcal I_f}\alpha_{f,i}=1.
\]

query 靠近分组中心时权重更大，因为组边界附近的 action query 也会关注相邻未来帧。之后用 \(Q_\ell=R_\ell(1-E_\ell)\) 排序层：\(R_\ell\) 是对齐到对应 future frame 的注意力质量，\(E_\ell\) 是归一化空间熵。分数较高意味着注意力质量充足且较集中。选出前 \(K_{layer}\) 层并按 \(Q_\ell\) 加权，得到每帧位置分数 \(V(f,j)\)。

选择分两步：

1. **Core：** 先以 \(\max_f V(f,j)\) 选出大小为 \(N_s-K_s\) 的候选位置池，再对每个 future frame 独立取前 \(K_c\) 个位置。这让不同帧追随不同的 attention hotspots。
2. **Anchors：** 从任何一帧都未被选为 core 的位置中，用全部层按 \(Q_\ell\) 加权汇总的空间分数计算跨帧均值 \(\mu_j\) 与变异系数 \(CV_j\)，按 \(\mu_j/(1+CV_j)\) 取前 \(K_s\) 个。这里使用全部层，而 core 只使用选出的高分层。每一帧保留相同的 anchor 空间位置，但每帧对应的 latent 表示仍分别更新。

因此每帧固定保留 \(K_c+K_s\) 个未来视觉 token，有利于紧凑打包执行。观测 token 与动作 token 始终全部保留。

### Pilot 执行引擎

朴素剪枝会先完整计算 attention map，再打包 token，额外开销可能抵消省下来的计算。Pilot 利用 dense conditional forward 已有的 query-key logits 和 softmax log-sum-exp normalizer，直接计算所需的 action-to-future 分数；它不额外跑一遍网络，也不构造完整 attention matrix。Pilot 缓存被选 token 的原始位置和打包元数据，后续 denoising steps 复用。

对被省略的未来视觉位置，Pilot 不会从 sampler 中删掉这些 latent。令 \(M_v\) 为保留位置 mask，则稀疏步骤用当前保留位置预测和上一次完整步骤的缓存预测拼成完整速度场：

\[
\widetilde v_v^{(\tau)}=M_v\odot v_v^{(\tau)}+(1-M_v)\odot v_v^{(\tau_d)},\qquad \tau>\tau_d.
\]

原 sampler 仍更新整张未来 latent；动作预测每个 denoising step 都重新算。默认每个 action chunk 的第一步 \((\tau_d=0)\) 是完整计算，之后复用选择和视觉预测缓存；下一个 chunk 重新选择。简言之，它是“稀疏 Transformer 计算 + 对省略位置复用旧视觉预测”，不是完整地跳过未来 latent 的生成。

## 报告的配置与结果

所有 policy inference 在 NVIDIA RTX 4090 上运行。主表的 speedup 是相对 **dense eager inference**；论文说明该比较包含 acceleration method 与 execution optimizations 的共同收益。不能把 headline speedup 全部归因于 token selection。

| Benchmark / backbone | Dense success | Sparse-WAM success | Sparse speedup | Sparse FLOPs | 读数 |
|---|---:|---:|---:|---:|---|
| LIBERO / FastWAM-Joint | 98.75% | 98.45% | 1.98× | dense 的 49.65% | success 低 0.30 个百分点 |
| RoboLab-120 / Cosmos 3 Edge | 22.90% | 23.00% | 1.85× | dense 的 60.73% | 平均 success 高 0.10 个百分点 |
| RoboLab-120 / Cosmos 3 Nano Policy | 36.75% | 35.50% | 1.81× | dense 的 59.78% | 平均 success 低 1.25 个百分点 |
| 真实机械臂 / FastWAM-Joint | 77.78% | 75.00% | 2.08× | 未报告 | 平均 latency 501 → 242 ms |

RoboLab-120 的 Edge/Nano 指标覆盖不同难度任务，整体 success 较低；不要只看相对 speedup 而忽略绝对成功率。真实机器人结果来自 AgileX Cobot Magic 上的 object packing、cup stacking、battery insertion 三项任务。论文报告三项 success 从 75.00/75.00/83.33% 变为 83.33/75.00/66.67%，平均从 77.78% 变为 75.00%。

### 区分算法收益与执行后端

Cosmos 3 Edge 的 Appendix B.4 把速度来源拆开：dense eager latency 为 859.31 ms；Sparse-WAM eager 为 555.55 ms（1.55×）；加入 CUDA Graph 与 `torch.compile` 后为 464.95 ms（相对 dense eager 1.85×）。在两边都启用执行优化的 matched-backend 比较中，dense 为 727.44 ms、Sparse-WAM 为 464.95 ms，即 **1.56×**。这组数更适合估计 Sparse-WAM 在该 backend 上相对优化后 dense 的增益。

计时采用 batch size 1、BF16、4 denoising steps；5 次 warm-up 后用 CUDA events 测 30 次并同步，报告 median。计时包含 observation encoding、profiling、在线评分与选择、packing/restoration、cache 操作和 sampler 更新；不包含模型加载、compile warm-up、CPU 输出传输、视频解码、RPC 与仿真。它是模型推理延迟证据，不是端到端闭环控制频率或完整系统延迟。

### 稀疏程度与鲁棒性代价

- **LIBERO-Plus：** FastWAM-Joint 平均 success 70.89%、431.7 ms；Sparse-WAM 为 63.04%、207.2 ms（2.08×），下降 7.85 个百分点。action-only Fast-WAM 为 51.96%、90.1 ms。Sparse-WAM 保留 future imagination 的同时，比该 action-only 结果更稳，但未达到 dense joint baseline。
- **Pruning ratio：** 在 Cosmos 3 Nano Policy 的消融中，剪枝率从 28.89% 增至 68.89%，speedup 从 1.45× 增至 2.36×，success 从 37.5% 降至 28.3%。加速与任务成功之间存在明显取舍。
- **跨步注意力图重合：** Appendix C.3 用归一化分布的 overlap \(\sum_i\min(P_i,Q_i)\) 衡量。连续步骤平均 overlap 在 Cosmos 3 Edge 为 81.11%、FastWAM-Joint 为 97.97%；跨帧 overlap 更低，分别为 68.89% 和 81.85%。这是注意力分布重合，不等价于保留 token 集合完全一致或 action prediction 不受影响。

## 应如何理解这些 claims

- **Training-free** 指稀疏化方法不需额外训练；不代表基础 WAM 没有训练。真实机器人部分的 FastWAM-Joint 使用作者收集的数据做了 fine-tune。
- **Action-guided** 在这里是用 action query 对 future token 的注意力当作 relevance proxy。注意力高低不是因果作用或闭环收益的证明。
- **Speedup** 是所述 GPU、精度、backend 与计时范围内每个 action chunk 的模型推理延迟比值；不同实现、设备、编译后端或控制周期不能直接沿用该数字。
- 论文评估三种 WAM、RTX 4090，以及有限的模拟和真实机器人任务。作者也将其他 GPU、更多架构和扰动下的性能列为后续工作。
- 论文 HTML 与 arXiv 元数据没有列出代码仓库或项目页 URL；本次没有核实到作者发布的官方代码，因此无法独立检查实现、checkpoint、数据、命令或复现实验。这里的结果均为论文报告，未由本阅读包重跑。

## 与 Fast-WAM 和本 FYP 的关系

这篇属于 WAM **推理侧视觉 token 稀疏化**：它保持 action tokens 与 observation tokens 密集计算，只减少部分 future visual tokens 的 Transformer 计算，并在省略位置复用旧的 visual velocity。它不是量化、权重压缩、蒸馏，也不是训练一个更快的 world-model predictor。

Sparse-WAM 的主要 Fast-WAM 实验使用 **FastWAM-Joint**，每个 denoising step 继续联合更新 future 与 actions。论文还在 LIBERO-Plus 中列出 action-only Fast-WAM 对照。不要把这两种模式混为一谈，也不要用 joint 模式的结果替代 Fast-WAM 在 test time 省略 future imagination 的结论。可对照本地 [Fast-WAM 阅读包](../004-fast-wam/README.md)。

对 FYP 来说，最相关的是它把“action 对视觉未来哪些位置有用”变成了一个可测的筛选机制，也给出了一种系统开销如何抵消理论 FLOPs 降低的例子。但 LeWM/PushT 的 action-conditioned latent predictor 与这里联合 denoise 视频 token、action token 的结构不同；本论文结果不能直接迁移为 LeWM 的 predictor acceleration 或 planner/CEM 加速证据。注意力选择、latent predictor 误差、planner 排名/首动作保真度和 closed-loop 成功率仍需分开讨论。

## 分段阅读路线

- **20 分钟定位：** Abstract、Fig. 1–3、Sec. 3.2。先写出稀疏对象是什么，哪些 token 始终保留，attention map 在哪一步取得。
- **75 分钟理解：** Sec. 4.1–4.2 与 Eqs. (1)–(7)，再读 LIBERO 与 RoboLab-120 主结果。画出“dense 选点 → core/anchor → sparse forward → 用缓存补齐 sampler”的数据流。
- **2 小时核查证据：** Sec. 5、Appendix B.2/B.4、C.1–C.3。并排记录 dense eager、优化后 dense、Sparse-WAM 的 reference；检查 LIBERO-Plus 退化与 pruning-ratio 消融；区分 attention overlap、FLOPs、latency、success。
- **联系本地基线：** 阅读 [Fast-WAM 包](../004-fast-wam/README.md) 中不同 inference mode 的定义，再确认论文里的 FastWAM-Joint 是否与想比较的 checkpoint、观测输入、action chunk 和 denoising schedule 对齐。

## Reading Questions（留给自己作答）

1. \(Q_\ell=R_\ell(1-E_\ell)\) 中的高 attention mass 与低 entropy 各自解决什么问题？是否有可能选中很集中的、但对动作无用的区域？
2. \(H\) 必须能被 \(F\) 整除。遇到 action horizon 与 future frame 数不整除时，query-frame 对齐应如何定义？
3. attention 是 relevance proxy。怎样区分“注意力相关”与“对动作有因果影响”？
4. 一整个 action chunk 只在第一 denoising step 选点；什么变化会让这个 selection 很快过期？C.2 的刷新频率实验支持什么结论？
5. 被省略 token 用缓存的 visual velocity 更新。误差如何随 denoising steps 积累？哪些情形可能要求重新做 dense refresh？
6. LIBERO headline 1.98× 与 Cosmos Edge matched-backend 1.56× 分别包含哪些执行后端收益？哪一个更适合比较算法贡献？
7. LIBERO-Plus 平均 success 下降 7.85 个百分点。这个代价对目标应用可接受吗？应优先关注哪些 perturbation 类别？
8. FastWAM-Joint 与 action-only Fast-WAM 在 architecture、checkpoint、future tokens 和 action generation 上具体有哪些差异？本地包和原论文能确认到哪些？
9. FLOPs 降低、单 chunk latency、control frequency、closed-loop success 是四种不同证据。本文分别实际测了哪些？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的输入、输出与推理循环：
- 稀疏规则与核心公式：
- selection 的刷新频率与省略位置如何更新：
- 最关键结果及其 hardware/backend/reference：
- 一项重要性能代价或未验证点：
- 与我的 FYP 机制假设的关系：
- 我需要导师确认的问题：

## 来源与身份

- 论文、版本、作者、提交日期、页数与 license：以 [arXiv:2609.38984v1](https://arxiv.org/abs/2609.38984v1) 为准；HTML 标注 CC BY 4.0。
- PDF：从 [arXiv v1 PDF](https://arxiv.org/pdf/2609.38984v1) 固定下载到本目录；文件名带 `v1`，不以更新版本覆盖。
- 代码：arXiv metadata 与论文全文没有 code/project URL；未核实到作者官方 repository。
- 本地 PDF 页数与大小：16 页，5,957,499 bytes。未做模型实验、benchmark 或复现。
