# WM / WAM quantization：主代理独立 prior-art 核对

核对日期：2026-09-08。此文件是两条 IdeaSpark 运行之外的来源核对，不替代 pipeline 的结构化检索、collision gate 或独立审稿。

## 已核实的直接前作

1. **QuantWM — An Empirical Study of World Model Quantization**, arXiv:2602.02110v1。官方全文已经讨论 encoder/predictor 非对称敏感性，以及低比特下 planning objective 与真实成功的失配。因此，“量化误差随 rollout 累积”“需要保护敏感模块”“更多 planning 不一定补救”都不能作为新贡献本身。
   来源：https://arxiv.org/html/2602.02110v1 ，§3–4，尤其 Insight V。

2. **Where Bits Matter in World-Model Planning**, arXiv:2602.11882v1。已经做 paired mixed-bit / planner-budget 对照，并明确提出直接针对 planning success 的 allocation 和 geometry-aware calibration 作为后续问题。一个新 idea 需要具体可证伪机制，不能只重复这条研究建议。论文明确承认小样本、单环境和实际大小未完全匹配的限制。
   来源：https://arxiv.org/html/2602.11882v1 ，§4–6。

3. **QuantWAMs**, arXiv:2607.28405v1。已经包含 shared-basis calibration、联合 video/action gradient 的 empirical-Fisher cross term，以及固定 protection 数量的 closed-loop schedule auditing。“考虑 video-action 交互”“用闭环状态校准”“按 denoising step 分配 precision”均已有直接先验。需逐公式比较新的被估计量、干预对象和部署动作。原文 layer upgrade 用 Linear 数量预算；应另核对真实 bytes 与运行代价。
   来源：https://arxiv.org/html/2607.28405v1 ，§3.2 Eq.12–16；§3.3 Eq.22。

## 最终整合时必须检查

- 两条 idea 的核心对象确实不同：latent world-model planning 与 joint future/action policy。
- 明确区分数值 quantization 和 latent/action tokenization。
- QuantWM / QuantWAMs 的存在不自动否定新机制；但换术语不能算差异。
- CPU toy trace 只能说明程序/论证是否自洽，不能证明机器人 success、GPU memory 或 speedup。
- fake quant、floating-point dequantized GEMM 和真实低比特 kernel 必须分开报告。
- 所有收益均为待验证假设；不把 pipeline 的 advance/DONE 当作实验成功或新颖性认证。

## 官方实现入口的额外核对

- FastWAM 当前官方 README 提供 Optional IDM checkpoint 的 `idm` / `first_frame` 两种推理模式。涉及 future conditioning 的 idea 必须指明执行模式；原论文、旧 checkpoint 与新 Optional IDM 不可混用。旧 checkpoint 的 action scheduler shift 为 5.0，新 Optional IDM 为 1.0。来源：https://github.com/yuantianyuan01/FastWAM ，2026-09-08 读取。
- LingBot-VA 官方 README 报告：RoboTwin 单 GPU evaluation 开启 VAE/text encoder CPU offload 时约需 24GB VRAM；i2av inference 相同 offload 条件约 18GB。这只是作者报告的 inference 配置，不能据此证明带 backward 的 calibration 在 A100 40GB 一定能放下。来源：https://github.com/Robbyant/lingbot-va ，2026-09-08 读取。

## 超出默认 48 个月 collision 窗口的机制谱系

这部分由主代理补查，不能被解释为已经证明候选被包含。它约束的是宽泛 novelty 口号，最终仍须逐机制比较。

- **Value-Aware Loss Function for Model-based Reinforcement Learning**（AISTATS 2017）：明确提出 transition-model loss 应考虑下游 value/decision 结构，而非仅概率建模误差。因此，“让模型误差度量对决策敏感”本身已有长期先验。来源：https://proceedings.mlr.press/v54/farahmand17a.html 。
- **Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice**（2021，arXiv:2106.14080）：研究 model-performance difference、value-aware objective，并讨论 stale value estimates。来源：https://arxiv.org/abs/2106.14080 。
- **Value Gradient weighted Model-Based Reinforcement Learning**（2022，arXiv:2204.01464）：也是需要比较的 value-aware/gradient-weighted model-learning 家族，发布时间早于本次默认 48 个月窗口。来源：https://arxiv.org/abs/2204.01464 。
- **Deep Task-Based Quantization**（2019，arXiv:1908.06845）：以 ADC/信号任务为对象，不是 WAM weight PTQ；但足以说明“让量化服务于下游任务”不是新的总体原则。不能据此直接否定具体 WAM 数值机制。来源：https://arxiv.org/abs/1908.06845 。

## 相邻 robotic policy quantization

- **SQIL — Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control**（ICCV 2025，arXiv:2505.15304）：以 saliency/state-importance score 识别 mission-critical states，并通过 QAT 与选择性 action distillation 加权保护决策。它不是 joint WAM 的 paired simulator site intervention，但“关键接触/任务状态更受量化影响，应特别保护”已有直接相邻先验。WAM 候选需比较具体分配对象、是否更新权重、如何识别 state importance 与干预后果，不能只比较应用标签。
  来源：https://arxiv.org/abs/2505.15304 ；https://aiha-lab.github.io/sqil/ 。
