# 03. QuantWM：An Empirical Study of World Model Quantization

**An Empirical Study of World Model Quantization**  
本地版本：**arXiv:2602.02110v1** · 17 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：优先核心 / 直接 PTQ prior art

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2602.02110v1) · [代码或官方项目入口](https://github.com/huawei-noah/noah-research/tree/master/QuantWM) · [返回总指南](../../README.md)

## 背景与要解决的问题

这篇直接否定“WM 还没有量化研究”的宽泛说法。它以 DINO-WM 为对象，系统比较 weight-only 和 weight-activation PTQ，并观察 planning success。

## 方法与核心机制

比较 RTN、OMSE、AWQ、SmoothQuant、OmniQuant；改变 weight/activation bits、weight group size、activation granularity，并分别量化 encoder 与 predictor。重点是低精度如何影响 latent rollout 及优化目标与实际成功的关系，而非单纯分类准确率。

## 结果：带着条件读证据

Tables 1–2 / PDF pp.5、7：Wall/PushT weight-only；Tables 3–6 / pp.9–11：weight-activation；Tables 7–8 / p.12：encoder/predictor sensitivity；Figure 3 / p.13：planning loss。官方目录已公开 readme.md、plan_act.py、plan_quant_omse_rtn.py、plan_quant_smooth.py、plan_quant_omniquant.py、plan_quant_awq.py 和 quant_utils。摘要仍写 code will be available，但当前不能再据此说未开源。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

两种环境和一个模型家族不能代表所有 WM/WAM。本文观察到 encoder 更敏感、group-wise 在低位宽有帮助、增加 optimization 不一定救回失败；这些不是对所有模型的定理。原文对 horizon/iterations 的表述需对照 config 拆开。仓库存在脚本不等于已验证 packed low-bit kernels 或真实端到端 speedup；本次只做资料与入口核验。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 每张表 baseline precision 是 FP32 还是 FP16？
2. 哪些提升来自更细 granularity，哪些来自校准算法？
3. Tables 7–8 是否支持“一律保留 encoder 高精度”的普遍结论？
4. Planning loss 降低却失败，是否由 candidate ranking inversion 或目标表示失真解释？
5. Calibration 使用真实 observation 还是 predicted latents？是否覆盖 planner 搜索分布？
6. Latency/显存是否实测，测的是哪个模块和哪种 kernel？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
