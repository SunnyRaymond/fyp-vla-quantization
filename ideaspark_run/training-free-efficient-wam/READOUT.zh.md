# Training-free Efficient WAM：本轮探索结论

资料截止：2026-10-02。ResearchStudio-Idea（idea depth）已到达原 navigator 的 **DONE**：完成候选生成、coherence、collision、critique、一次限定 revision、独立 post-revision review、technical expansion、implementability/cost、plain derivation 和 publication。没有运行 WAM、GPU 或机器人；这是研究方案，不是实验验证。

方案卡：[中文版](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/phase4/idea.std.zh.md>) · [英文版](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/phase4/idea.std.en.md>) · [英文详情及实现缺口](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/phase4/idea.detail.en.md>)。

## 结论与建议

本轮比较了五条路线，保留一个**有条件的研究假设**，其余四条在 prior 初筛中淘汰。现在还没有确认一个同时满足“有足够新意、推广到两个 baseline、凭现有公开 artifacts 真机复现”的完整方案。建议先解决 matching checkpoint 与计算余量，再决定是否投入实验。

保留项是 **Action-Context Source Masking for Frozen Joint WAMs**：在已削减 future-video 计算的 joint WAM 中，检验 action queries 是否仍需要读取全部 observation source groups。每个 action chunk 的首个 native solver evaluation 保持 dense，按各层 source group 对 action residual 的投影贡献排序；后续 evaluations 只保留选中的 action→observation attention edges，使用实际跳过 QK/AV 的 sparse/variable-length kernel。

它不更新权重。video queries、QKV、FFN、action→future/action interactions 和 solver schedule 都保持原路径。因此，大模型参数多不保证这项干预有大的加速空间。

## 五条路线比较

| 路线 | 本轮判断 | 主要理由 | 检查深度 |
|---|---|---|---|
| Action-context source masking | 条件保留 | 需要先证明计算余量和 action/task 效果；通用贡献评分已有强 prior | 完整 canonical gauntlet，未做模型实验 |
| Exact observation K/V reuse | 当前淘汰 | 固定输入 latent 不证明 hidden/KV 不变；与既有 video KV reuse 邻近 | prior 初筛 |
| Within-chunk action residual reuse | 当前淘汰 | 与 WAMachine、C3ache 和本地 CREC proposal 邻近 | prior 初筛 |
| Low-rank action-context operator | 当前淘汰 | 缺少区别于通用 attention approximation 的 WAM 结构依据 | prior 初筛 |
| Async lookahead/action separation | 当前淘汰 | 邻近 GlanceWAM、WAMachine、FBFM；co-training 版本不满足冻结条件 | prior 初筛 |

完整条目见 [candidate_shortlist.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/candidate_shortlist.zh.md>)；近邻边界见 [recent_prior_boundaries.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/recent_prior_boundaries.zh.md>)。四项淘汰是本次方向选择，不是所有具体实现的普遍 NO-GO，也没有声称它们都跑过完整 gauntlet。

## 保留项的两个决定性问题

### 1. 可删除路径是否足够大

某层某次 evaluation 中，action-query→observation 的 QK/AV 算子量约为

`F_edge = 4 N_a N_o d_att`，其中一次乘加按两 FLOPs 计。

这不删除大量 video-query/FFN/projection 计算。令 `phi_obs` 是完整原推理中该路径的实测耗时份额；其余路径不变、额外开销为零时，完全删除它的理想加速上限才是 `1/(1-phi_obs)`。首步 dense、贡献评分、packing 和 kernel launch 还会降低净收益。FLOP 占比不能替代时延占比；fused kernel 内的子路径归因也须说明方法。

**停止条件：** matching 路径没有真实省算 kernel，或理想余量已不足以支付开销/达到既有 latency gate，就结束此候选。详细推导见 [HEADROOM_NOTE.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/HEADROOM_NOTE.zh.md>)。

### 2. 贡献评分不等于控制保真

对固定 query 的单个 attention head，若删除部分 keys 的原 attention mass 为 `p`（`0≤p<1`），保留和删除部分的归一化 value 均值分别为 `y_K`、`y_D`，则原输出为 `(1-p)y_K+p y_D`，删边并重新 softmax 后输出为 `y_K`，误差为 `p(y_K-y_D)`。所以原删除部分的贡献小，不自动保证删边后的误差小；这也不能直接保证 action chunk 或闭环成功率。

[ToPi](https://arxiv.org/html/2602.01609v1) 已有 attention×value-norm、累计贡献选集及跨步复用；[CAPA](https://arxiv.org/html/2602.00247v1) 已有 output-projected contribution。因此“使用 value / W_O”本身不能作新意。候选还需在 WAM 中证明特定 source-group 干预的用途。

原 falsifier 保留 dense、贡献排序和相同保留边成本的 source-group permutation，并增加 attention mass、ToPi/VATP value-norm 与 CAPA projected-score 对照。各稀疏评分臂同 checkpoint、同 solver、同 compact kernel、同保留边数；每臂都计入自身评分与 packing 开销。epsilon 的选择集与最终 paired evaluation 集须分开。等成本 group permutation 需要存在合法可置换分组；不满足时不能冒称该负控已执行。

**停止条件：** 原 task-success gate 不通过，排序没有优于上述对照，或完整推理没有净加速，就不支持本机制主张。合成算术 trace 仅检查规则可计算，不提供任务或速度证据。

## 两种 baseline 与真机路径

| 对象 | 已确认的公开材料 | 当前仍缺什么 |
|---|---|---|
| FastWAM-Joint / LIBERO | Joint 源码、配置和评测路径 | [官方权重页](https://huggingface.co/yuanty/fastwam) 未确认 matching Joint checkpoint；action-only/Optional IDM 不能冒名替代 |
| Cosmos3-Edge-Policy-DROID / RoboLab | [公开 4B checkpoint](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)、Framework policy server、RoboLab client | 仍需钉定与 Sparse-WAM 相符的 joint video/action entry、sampler/config/backend；未实际运行 |
| Motus / RoboTwin2 fallback | 官方 Stage-3 joint policy 和仿真入口 | 可作另一 backbone 的适配对象；不能称为 FastWAM-Joint 同模型复现；真机 matching 部署未确认 |
| DreamZero alternative | 官方 joint DROID checkpoint，论文真机证据 | matching robot client/controller、calibration/config、硬件；DROID 权重许可为 CC-BY-NC-4.0 |
| LingBot-VA alternative | 官方 RoboTwin/LIBERO policy 与仿真入口，论文真机证据 | matching 真机任务 policy 和控制/标定入口；视频先生成、动作随后生成，不能视作与同步 joint solver 完全等价 |

**重要区别：** [Sparse-WAM](https://arxiv.org/html/2609.38984) 的 Cosmos3 Edge 行是 **RoboLab 仿真**；真机行是 **FastWAM-Joint + AgileX Cobot Magic**。作者先对其任务做 baseline post-training；matching 真机权重、自采数据及方法源码未确认公开。这些前置成本与 training-free acceleration 方法分开核算。

因此，这轮提供的是跨两种模型的**条件性验证计划**，没有宣称已经迁移或真机复现。详细入口和来源见 [baseline_feasibility.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/baseline_feasibility.zh.md>)、[fallback_baseline_feasibility.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/fallback_baseline_feasibility.zh.md>)、[public_robot_alternative_baselines.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/public_robot_alternative_baselines.zh.md>)。

## 执行边界与预算

- 本轮没有 GPU 实验、模型/数据下载或机器人运行；没有额外付费 LLM/API 调用。proposal 中的 GPU-days=0 仅指本轮，不是未来验证成本。
- 后续预算未知：先有 matching checkpoint、kernel 插入点与 profile，再确定 GPU、episodes 和计算预算。本轮不编造 GPU-days、样本量或成功率阈值。
- 原论文的硬件参照：[Sparse-WAM](https://arxiv.org/html/2609.38984) 的 policy inference 使用 RTX 4090；其 AgileX 任务的 baseline post-training 使用 8×H800，三个任务分别训练 15k/20k/15k steps。这些是已报告设置，不能直接当成本候选的 GPU-days 或必需配置。
- 最近实际主配额检查：used 65%，remaining 35%；没有检测到恢复 100% 的 reset，未动用 banked reset。后续若实际主 bucket 突然恢复 100%，立即停止。
- 检索来自 arXiv/OpenAlex，Semantic Scholar 限流、OpenReview challenge 使覆盖降级。原 collision pool 为 226 条；ToPi/CAPA 是后续 primary 检查发现并经 resolver 核验的补充，不能伪称原检索已命中，也不能以未命中证明新颖。

检索限制见 [phase0/RETRIEVAL_LIMITATIONS.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/phase0/RETRIEVAL_LIMITATIONS.zh.md>)。

## 输出状态

原 navigator 的终态证据保存在 [final_navigator.txt](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/context/final_navigator.txt>)。本轮采用 idea depth，保留实现缺口，不执行 experiment-depth 的补全/修复流程；默认未单独启用 plain-vs-technical fidelity review。独立 post-revision review 和原 publication validators 已执行，后者没有 fail，但保留未测 profile、kernel、paired evaluation 和真机等 warnings。

已保存三个 Markdown cards 和两份 TeX source。渲染环境没有可用 TeX compiler，因此 **PDF 未生成**；未安装 compiler。实际状态见 [render_status.json](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/phase4/work/render_status.json>)。

Publication 遇到 installed skill 的函数签名不一致，使用 run-local compatibility wrapper 调用原 renderer；成功 publication 后，仅归档已解决的 execution-contract 错误，再由原 navigator 确认 DONE。没有改 scientific inputs 或降低 gates。记录见 [RUNTIME_COMPAT.zh.md](<D:/Downloads/Final Year Project/ideaspark_run/training-free-efficient-wam/context/RUNTIME_COMPAT.zh.md>)。
