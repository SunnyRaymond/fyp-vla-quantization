# Temporal rounding persistence：方法主张 no-go，诊断结果保留

审查日期2026-09-13；使用scoop-check，原GPU/CPU不新增验证。

**generic_method_novelty_no_go；matched diagnostic/application delta未确证。** GPU64825/CPU64831的6/6 narrow preliminary signal仍成立；不能把数值positive自动升级成新方法。

## 最接近的primary prior

*De-biasing Diffusion: Data-Free FP8 Quantization of Text-to-Image Models with Billions of Parameters*，OpenReview ICLR2025 submission，§4.2–4.3、Eq.(8)、Fig.3/8，已讨论跨denoising-step误差相关性，并用带额外mantissa bits的stochastic weights逐forward生成低精度权重，改善FP图像保真。它还讨论额外存储的代价。[primary PDF](https://openreview.net/pdf?id=nExUJBF5tR)

这直接覆盖“减弱weight rounding的时间持久性可缓解diffusion累积误差”的宽机制及随机权重实现方向。换成SmolVLA、INT4、三套snapshot或fixed cycle不足以单独证明新quantizer；没有比较上述prior的完整结果，也不能声称方案更有效或更省资源。

[AccuQuant §3.2–3.3](https://arxiv.org/html/2510.20348v1) 已将单步与累积误差分开，并通过多步输出对齐进行calibration；[Low-Bitwidth Floating Point Quantization §V-B](https://arxiv.org/html/2408.06995v1) 学习FP4 weight rounding。二者不是本案的同一fixed-marginal temporal assignment，但进一步限制“首次考虑多步误差/rounding”的主张。

## 尚可保留的精确区别

本案是一个受控诊断：在每个step保持同样三draw multiset，比较Frozen与Cyclic，保存共同FP-path forcing与自由积分endpoint。它把时间joint assignment与单步边际误差大小分开；已有引用不足以证明这个精确实验设计已被完全覆盖，也不足以证明其全球新颖性。唯一已完成的证据是固定SmolVLA checkpoint六state的数值screen；无task success、native性能或可部署单snapshot新方法。

## 搜索与覆盖限制

查询：diffusion quantization stochastic rounding weights each denoising step correlated rounding；diffusion stochastic rounding temporal；quantization temporal correlation diffusion rounding；nExUJBF5tR；De-biasing Diffusion stochastic weights/arxiv/authors；stochastic weights diffusion quantization（后者限arxiv/OpenReview）。当前primary HTML读到AccuQuant方法与误差分解、FP4 paper §V-B。

OpenReview直接打开遇到browser verification；检索系统返回该primary PDF的标题、abstract及§4.2–4.4完整索引段落，足以识别最接近的方法重叠。未绕过验证，未获得完整PDF/代码，未核实最终作者/接受状态及appendix proof。故只以“ICLR2025 submission”标识，不虚称accepted。全文附件或实现差异仍未核验；这些限制不应被写成没有prior。

**STOP**：不为改进novelty而重跑或扩展原positive screen；保留原结果与本次较严格的贡献边界。
