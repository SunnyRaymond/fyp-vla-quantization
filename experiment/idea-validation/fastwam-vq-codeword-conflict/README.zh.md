# Codeword 内的 latent 梯度冲突诊断

**已完成（2026-10-17）**：完整 capture/exact/360 条有限干预及最终 CPU 分析均完成。小步 untied 有正面信号，较大步长失败；未通过预定双步长推进条件，暂不直接实现新的 assignment。完整结论、限制及下一步见 [FINAL_REPORT.zh.md](FINAL_REPORT.zh.md)。下方运行记录保留为历史过程。

问题：固定 indices 时，同一个 codeword 的成员是否对 latent protection 提出相互冲突的修正需求？这种约束是否会影响有限扰动后的 future 和 action？先诊断，不实现新 assignment 算法。

沿用 Parent、action-only epoch2、action+latent epoch2 及相同的4D/两本256-entry VQ、row scales、Smooth/H128/A4。梯度分析只用原8个calibration contexts；有限扰动评估沿用完整60contexts（test22cases×2seeds）。新扰动/分组的选择不读取selection或test。

第一阶段覆盖614个目标Linear和全部已占用codewords；每个book的每个codeword均匀无放回抽取最多32个成员，固定样本跨checkpoint与context复用。每个weight vector先累加整个sampler的重复调用梯度，再计算组内抵消。Codebook的总梯度精确计算，组内成员指标是明确标记的抽样指标。分别报告latent组内冲突、action/latent目标冲突和固定occupancy的随机分组对照；自然抵消本身不是“坏assignment”的证明。

第二阶段只根据calibration结果选少数位置，重新捕获完整成员梯度。保持scales及其余参数固定，比较真实共享codeword更新与同support、同实际decoded-BF16更新范数的untied梯度更新和随机方向；步长比例预先固定0.005/0.015。Untied仅为解除容量约束的诊断，增加了自由度，不是相同bits的可部署方法。有限扰动的forward证据与STE梯度证据分开。

该有限干预只更新选中 book0 codewords，book1 固定；它检查这一受控共享子空间的限制，不能证明允许两本 codebook 与所有 indices 联合变化后仍无法恢复。Native actions 会同时受 future 生成和 reader/prefill 参数变化影响；只有 BF16 teacher-reader 的 future 替换分支固定了完整下游模块。

所有GPU作业单卡/最多2h，CPU阶段20min；逐context保存、只有成功RESUME_READY终态才能续交。每15s将GPU利用率和显存写入job log。所有模型、tensor、数值计算与重IO都在PBS compute allocation内；login仅SSH/PBS/有界小控制与日志。保留host-key verification，不输出credentials，不使用banked reset。监控使用独立subagent，约10min间隔，Q/R不变时静默。

本轮不提供闭环成功率、native kernel速度、独立泛化或历史训练因果归因。最终是否值得修改assignment，要结合梯度模式、随机分组对照、实际有限扰动和跨reader结果共同判断。

实验设计流程参考 Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065, consulted v2. https://doi.org/10.48550/arXiv.2609.00065

## 文献对照与候选方法（2026-10-16）

本节是 primary-source 方法阅读后的研究判断，下面的改进均尚未实现或验证。检索涵盖 VQ/assignment/gradient conflict、Hessian/Fisher、sensitivity/bit allocation 和 WAM video–action protection；并非完整 novelty proof。

| 已有工作与位置 | 超出最终输出 loss 调参的操作 | 对本项目的约束 |
|---|---|---|
| [VQ4DiT v1 §4.1–4.2、Fig.3](https://arxiv.org/html/2408.17131v1) | Euclidean top-n 候选的 soft assignments 与 codebook 联合优化，最后 harden；报告同 codeword 成员梯度一致性 | “共享 codeword 有梯度冲突”与“优化 indices”已有直接先例 |
| [GPTVQ v2 Codebook Initialization、Algorithm1](https://arxiv.org/html/2402.15319v2) | Hessian-aware clustering/assignment；量化后补偿其余权重 | 不应把 data-aware distance 或 error compensation 单独称为新方法 |
| [AQLM v3 §3.2–3.4](https://arxiv.org/html/2401.06118v3) | 搜索 additive codes、交替更新 codebook/scales，跨 Linear 做 block joint tuning | 当前固定 indices 只是受限优化配置，不代表 VQ 的最佳上限 |
| [PV-Tuning v2 §3.2–3.4](https://arxiv.org/html/2405.14852v2) | 对 discrete partition 和 continuous values 交替优化，限制离散更新子空间 | 普通 discrete/continuous alternating optimization 已有先例 |
| [VPTQ v2 §3.2.2–3.2.3](https://arxiv.org/html/2409.17066v2) | residual codebooks 与独立 outlier codebook | 容量重分配/残差保护有先例，须比较实际总 bytes |
| [RSAVQ v1 §4.1–4.2](https://arxiv.org/html/2510.01240v1)、[MGVQ v1 §4.1–4.2](https://arxiv.org/html/2605.24019v1) | Fisher 方向约束、sensitivity-guided 资源分配、gradient-aware compensation | 泛化的“沿低敏感方向放误差”和 mixed precision 不能直接作为独占新意 |
| [QuantWAMs v1 §3.2 Eq.10–17](https://arxiv.org/html/2607.28405v1) | joint video–action gradient 的 empirical-Fisher saliency，分配 layer precision | “其他 WAM quantization 都 action-only”不成立；其 co-training saliency 与 rollout future distillation 仍是不同操作 |
| [LatentQuant v1 §4–5，2026-10-02](https://arxiv.org/html/2610.03959v1) | VAE encoder alignment 后冻结 encoder，再适配 decoder，阻断重建梯度改写 policy-facing latent | “量化需要保留 policy-facing latent”已有直接先例；它不是权重 VQ，且保护 VAE encoder latent，本项目保护 world DiT 生成的 future，所查共享约束机制不同 |

优先候选一：用 WAM future 的功能兼容性约束 codeword 共享。保持 M=2、K=256、d=4、scales/transform 和推理表示不变，在 reconstruction 候选范围内，利用 calibration 上稳定的 latent 修正方向与 action/latent 影响搜索 indices。可争取的区别须落在具体的兼容性准则及等预算结果；不能仅将 VQ4DiT 的 block MSE 换为 latent MSE。当前诊断只决定是否值得继续这条路线。

一个比平均 cosine 更直接的候选准则，是保留 calibration 修正方向在共享参数可实现子空间中的投影能量。单 book、scales 固定时，centroid c 的 decoded-space 最佳修正由 Σᵢ sᵢgᵢ / Σᵢsᵢ² 决定，retained energy 为 Σc ‖Σᵢsᵢgᵢ‖²/(Σᵢsᵢ²)，再除以 untied gradient energy。Padding coordinates 不计入分母。可将 assignment 搜索写成：在原 reconstruction/存储约束内，减少 latent 与 action 两种修正方向各自被共享子空间丢弃的能量。两本 additive books 都可动时应使用完整的 additive tangent subspace，不能把当前 book0-only 诊断直接当成完整表示上限。该准则仍是待验证的方法候选。

候选二：在初始化/assignment 时使用 future Jacobian 诱导的各向异性距离。局部近似为 Δwᵀ(JzᵀJz+αJaᵀJa)Δw，可用随机输出 probes 或分解近似；teacher 自身的 reconstruction-loss gradient 为零，不能据此构造该 metric。须与 activation-Hessian、action-only metric 和普通 action+latent fitting 比较。它与 Fisher-aware VQ 有显著重叠，WAM 的被保护对象及有效误差方向需要额外机制证据。

更具体的 WAM 假设是：某些方向满足 vᵀHzv 大而 vᵀHav 小，当前 action 目标看不见它们，却损伤 future 的可读性。可先用 normalized generalized Rayleigh ratio vᵀHzv/(vᵀHav+ε‖v‖²) 寻找候选，再用固定 BF16/学生 reader 和受控输入变化判定是否有害，之后才把它们用于 assignment 或预算分配。高 ratio 本身不证明对控制重要，可能只对应无关视觉细节；该负结果也必须保留。不要直接将 joint video–action Fisher 与“action 盲区中的有害 latent drift”混为同一个发现。

候选三：在总 bytes 不增加的条件下，把 codebook/residual 容量从低风险位置移给高 latent 损伤位置。以单位额外 byte 的实际 future/固定 reader 改善量分配资源，并包含 indices、codebooks、scales、mask/metadata 的开销；对照 random allocation 和 action-only allocation。单纯增加 residual book 或保留高精度不构成充分新意。

优化侧低成本对照：若冲突主要来自 action 与 latent 两个目标，可限制更新的 latent directional derivative，做小步有限 BF16 验证。若冲突发生在同一 latent 目标内部的多个成员，聚合梯度本身已抵消，目标间 gradient projection 无法解除共享约束，需另查 assignment/容量。两者必须分开诊断。

另一个必要的低成本对照来自 LatentQuant 的优化边界思路：先拟合 world/future 相关 VQ 参数，随后冻结全部会影响 future 的 books/scales，只恢复已确认不影响 future 的 action expert 参数，并与原 joint fitting 在相同 calibration 和预算下比较。这不是可直接独占的新方法。本地 FastWAMIDM.infer_action 在 video denoising 前就把 proprio 追加到 context，因此不能把 proprio encoder 当作纯 action 参数。Video expert 参数还会在 reader 的 prepare/prefill 中复用，单纯 detach future tensor 不保证这些共享参数不会沿 action 分支被改写；须按真实参数依赖冻结，不能只按 tensor 名称切断一条图。

目前最有希望的研究问题是：action fidelity 在 calibration 上恢复后，哪些 future drift 仍能损伤固定下游 reader；这些 drift 是否由可识别、可修正的 VQ sharing 约束造成。固定 reader 的既有结果支持继续诊断，但尚不证明历史训练因果、全局 novelty、真实 dynamics 恢复或 closed-loop 收益。

后续方法比较至少保留四个配置：当前 fixed-index action+latent、相同 loss 的普通 assignment 优化、功能兼容性 assignment，以及先 world 后 action 的冻结对照。前两者是必要基线，不能只与 action-only 比较。固定 calibration、候选数、训练预算、codebook/index/scale 表示；若新增资源分配，另匹配总 bytes。继续报告 future NMSE、固定 reader first10/full32/late22/gripper、原 reader action 和逐 case 配对差异，内部筛选后才做独立 closed-loop 测试。

若要解释 VQ 相对 scalar 的作用，scalar baseline 也须使用同一个 latent-aware 目标/敏感度信息，并比较包含 codebook/indices/scales/metadata 的真实预算。否则只能说明 latent-aware fitting 更好，不能将收益归因于 VQ 表示。当前 action-only 拟合本身不保证 held-out action 已恢复；研究叙述应检验这种情况，不能把它当作既定前提。

## 当前运行记录

- prepare `25843869.pbs101` 在非 4 的倍数的 Linear input width 上退出（F/Exit1）；未做新 model forward。
- 修复为与原 decoder 一致的逐 row 4D padding；梯度、成员 row 归属和有限 BF16 扰动范数均排除虚构的 padding coordinates，保留全部 614 modules。
- repair prepare `25843978.pbs101` 已提交，8 CPU / 32 GB / 20 min；新数学 self-check 包含 7-column padding 与实际 shared/untied BF16 扰动。最终以 compute allocation 的运行输出为准。
- 该作业通过 `CONFLICT_MATH_TESTS_PASSED`，随后在原 20 min allocation 内完成（F/Exit0、matching summary、PIPELINE_COMPLETE）。尝试延长 walltime 被 PBS 拒绝，未改变 allocation；因此没有使用另行准备的续跑逻辑。
- 完整 plan：614 Linears、314,368 occupied codewords、10,032,184 sampled book memberships、3,008,159,232 population memberships，model_forward_count=0。Sampling scope 与完整 codeword-gradient scope 分开报告。
- capture shard0 `25844248.pbs101` 已提交（1 GPU / 110 GB / 2h）；22:48 SGT 仍 Q，PBS 提示 queue 的 ngpus overall limit。尚无梯度实验结果；其余三个分片等待首个有效 context 后提交，避免放大首次运行错误。独立 subagent 按约 10min 间隔只读监控，unchanged Q/R 静默。
- 该 capture 于 23:21 左右 F/Exit1，首个 backward 失败，完整 contexts=0。原因是 observer 在原 backward 读取 saved tensors 后再次读取，违反 non-reentrant checkpoint 的单次解包要求；修复为在 forward 缓存模块名。原 decoder 的计算和梯度公式未改。
- 为避免再次加载整模型后才发现 checkpoint 交互问题，新增 CPU-only selfcheck `25844593.pbs101`（8 CPU / 32 GB / 20min）：直接测试 production observer 和原 DecodedWeight，在 padded 7-column、重复两次调用、同图 action/latent 两次 backward 下检查成员梯度与 codeword 梯度累加；sampled/exact 两条路径都覆盖。通过后重交首个 GPU 分片。
- selfcheck 已 F/Exit0、matching summary、PIPELINE_COMPLETE，6:37 walltime，`CHECKPOINT_OBSERVER_TESTS_PASSED`。Capture shard0 repair `25844701.pbs101` 已提交，仍为 1 GPU / 110 GB / 2h。
- 后续 GPU 提交以已通过的 CPU selfcheck 为 prerequisite，不再为每个 GPU allocation 启动独立 Python 重跑相同数学测试；实际 context 的 forward parity 与 adjoint 累加检查仍保留。首个 repair 已提交的脚本不改动。
- Exact 阶段会单独报告选定 32 个 book0 codewords 在全部 8 个 calibration contexts 上的完整成员 cancellation、pair cosine、共享修正的 opposed-member fraction 和 retained descent fraction，并将完整成员加和与实际 autograd codeword gradient 对照。它们与第一阶段抽样指标、跨 contexts 的抵消分别呈现；不增加模型 queries。
- 复用 C8 evaluation 中已有的 `batched_h128_v2`（batched_transform.py / transform_checks.py），不更改 H128 basis、scales、A4 grid 或权重表示。新增 CPU selfcheck `25844790.pbs101` 已 F/Exit0、PIPELINE_COMPLETE，6:17 walltime，`BATCHED_A4_STE_TESTS_PASSED`：旧/新 A4 输出及 STE gradient bitwise 一致（7/129/1536/5120/13824 widths，FP32/BF16，非连续输入）。后续 GPU 在模型加载前检查218个 forward/inverse/A4 fixtures；原 capture_0 repair 保留原 backend。实现差异单独记录，不作 native kernel 或 end-to-end latency 证据。

## 2026-10-17 first valid capture and remaining shards

Capture0 repair `25844701.pbs101` completed Parent cases0 and10 (seed0), 662.03s / 543.90s. Both action and future have cached-forward max_abs=0 and RMSE=0; 614 action-connected /307 latent-connected modules; repeated-call adjoint identity checks passed before records were saved. This is capture validation, not a gradient-conflict finding.

After this gate, capture1/2/3 were submitted as `25845191.pbs101`, `25845192.pbs101`, `25845193.pbs101`, each single GPU and walltime02:00:00, using the checked batched backend. Current capture0 retains its original backend snapshot. All24 frozen checkpoint-context records remain required.

The exact CPU report additionally records zero-scale support membership and its fraction of untied gradient energy. This is a descriptive confound check only: no new forward, selection, direction, step norm, or progression gate changes. Any gain from reviving fixed zero-scale rows must be distinguished from removing codeword ties.
