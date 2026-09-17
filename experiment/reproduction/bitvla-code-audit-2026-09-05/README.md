# BitVLA 公开代码一致性核查

核查日期：2026-09-05。

**结论：BitVLA 的 ternary-weight / INT8-activation quantization 和 STE 确实实现并接入模型；不能按 QVLA 的 action-sensitivity 替换问题类推它。主要复现缺口是 Quantize-then-Distill 完整训练未找到、robotics pre-training code 尚未发布，以及论文 BitBLAS deployment 与当前默认 floating-point evaluation 路径之间缺少公开桥接。没有证据认定实验造假。**

## 版本与核查范围

- 官方 GitHub：[ustcwhy/BitVLA](https://github.com/ustcwhy/BitVLA)，本次 shallow clone 的 HEAD：`8afac0260b3748b14657a69ec58e3d9f0d6da3a7`，commit 日期 2026-03-02。未审计完整 Git 历史。
- 主核查论文：[arXiv 2506.07530v2](https://arxiv.org/html/2506.07530v2)，2026-03-01；核对了 [v1](https://arxiv.org/html/2506.07530v1) 的方法/摘要，以免混用版本。
- v1 没有大规模 robotics pre-training；v2 增加这一阶段及新的 efficiency 结果。README 将旧版 94.8 和新版 96.0 的 LIBERO average 分开列示；两者不能当作同一 checkpoint。
- 已有 [本地论文 PDF](../../../papers/vla/vla-quantization-literature-review-alternative/papers/03-bitvla/paper-arxiv-v2.pdf)。本次 section/equation 定位来自在线 HTML，不使用未经本轮 PDF 校验的页码。
- 检查了 BitNet/SigLIP quantizer、模型装配、BitVLA action wrapper、OFT fine-tuning loss/optimizer、evaluation loader 和 README；在相关模型及训练目录搜索 distillation，在全仓库搜索 BitBLAS / packing 调用。
- [check_claims.py](check_claims.py) 使用 AST 抽取并执行原始 quantizer/class 定义，没有改写函数逻辑，也没有导入完整 Transformers fork。结果：[check-results.json](check-results.json)。这是 CPU synthetic check，不能代替整模型加载或 LIBERO evaluation。
- 没有下载 6 GB checkpoint、安装训练环境、使用 NSCC GPU、运行训练/机器人或测量硬件 latency/VRAM。上游 checkout 未修改。

## 1. 实际实现了什么？

### 1.1 ternary weights：已实现并验证

Vision 的 [WeightQuant L57–71](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/transformers/src/transformers/models/siglip/modeling_siglip.py#L57-L71) 和 Language 的 [WeightQuant L84–98](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/transformers/src/transformers/models/llava/modeling_bitnet.py#L84-L98) 都按 absmean scale，round/clamp 到三值网格，再反量化到原 dtype。

CPU 验证了 scaled weight 对应的码值为 `{-1,0,1}`，且与 absmean reference 数值一致。保存 BF16 master weights 不代表 forward 没有量化；这是 QAT 常规表示。

### 1.2 activation quantization 与 STE：已实现并验证

Vision [ActQuant L73–87](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/transformers/src/transformers/models/siglip/modeling_siglip.py#L73-L87) 和 Language [L100–114](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/transformers/src/transformers/models/llava/modeling_bitnet.py#L100-L114) 按最后一个维度的 absmax 做 INT8 网格模拟；backward 直接传递梯度。

测试验证 INT8 数值网格、per-token scaling 和两类 quantizer 的 identity STE gradient。注意 activation 返回值仍为 floating dtype；这验证的是量化数值，不是 integer kernel。

### 1.3 quantizer 不是闲置文件：已接入模型

- [modeling_llava.py L251–256](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/transformers/src/transformers/models/llava/modeling_llava.py#L251-L256) 装配修改后的 SigLIP 和 BitNet。
- BitNet MLP 的 gate/up/down projections 及 attention q/k/v/o 使用 BitLinear（L295–297、L349–352）。
- SigLIP attention（L505–514）和 MLP（L738–744）从 vision config 读取 bit 设置并构造 BitLinear。
- 当前官方推荐 checkpoint 的 [config](https://huggingface.co/lxsy/bitvla-bf16/blob/931b01f263401be987fe76c384d5d9ce9d5a9b52/config.json) 明确为 `vit_weight_bits=1`、`vit_act_bits=8`；不是只在类中定义后默认关闭。
- OFT [finetune_bitnet.py L337–339](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/openvla-oft/vla-scripts/finetune_bitnet.py#L337-L339) 使用连续 action head 的 L1 loss，L773/L803 有 backward/optimizer step。这是实际下游训练实现，但本次未整链执行。

## 2. Quantize-then-Distill：最重要的训练复现缺口

v2 §III-B 以固定 FP vision teacher 指导低比特 student，结合 LM 与 intermediate representation alignment loss，只更新 student vision encoder。

在当前 release 的 BitVLA/OFT 训练代码与改造后的 LLaVA/SigLIP 模型中，未找到相应 teacher 建立、teacher/student hidden-state 对齐 loss、该阶段 freeze 配置和可执行训练 recipe。公开 OFT 的 action L1 loss 属于另一阶段，不能视作 distillation 的替代实现。

**结论：这项创新目前无法仅靠该 release 从头复现；不能据此证明作者未进行 distillation。** 已发布 quantized-student checkpoint 只能使评估成为可能，不能独立证明其训练来源。

Robot pre-training 也未完整公开：README Open Source Plan 将其 code 列为待发布；[Issue #3](https://github.com/ustcwhy/BitVLA/issues/3) 中 OWNER 说明已发布 Open-X 预训练模型，代码计划以后发布。仓库保留通用 OpenVLA `train.py`，但不能因此称为已发布 BitVLA 所报告的整套 pre-training 实现。

## 3. 内存与加速：不能把当前默认路径当作论文 deployment

### 3.1 默认 evaluation 使用 BF16 master + online fake quant

[bitnet_utils.py L68–80](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/openvla-oft/experiments/robot/bitnet_utils.py#L68-L80) 加载 BF16。默认 `load_in_4bit`/`load_in_8bit` 均为 False；它们是额外的通用加载选项，不是 BitVLA 原生 ternary execution。

BitLinear 默认 `enable_qlora=False`，每次 forward 对 weight/input 量化再反量化，最后运行普通 `F.linear`。因此保留 BF16 master weight，不能期待照 README 跑就得到表格中的 1.4 GB。

官方 Hugging Face metadata（revision `931b01f263401be987fe76c384d5d9ce9d5a9b52`）列出 3,147,925,664 个 BF16 参数；model.safetensors 文件为 6,295,953,280 bytes，约 6.30 GB。见 [metadata](hf-metadata.json)、[tree](hf-tree.json)。这是文件/参数表示的证据，**不是实测运行 VRAM**。

README L61/L94 明确披露 master weights 和 offline conversion 的要求，并称专用 inference framework/model 待发布。这一点应给予公平评价，不能把已披露的 BF16 checkpoint 本身当作欺诈。

### 3.2 不能说“完全没有 packing”：存在可用 helper

SigLIP [L98–123](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/transformers/src/transformers/models/siglip/modeling_siglip.py#L98-L123) 和 BitNet 对应定义实现每 byte 存四个 ternary code 的 INT2 packing/unpacking。`quantize_weights()` 可移除 master weight 并保留 packed buffer。

最小测试中，8 个 BF16 weight 的 16 bytes 变为 2 bytes packed codes + 4 bytes scale；补齐非 4 倍数的 round-trip 也通过。这里实现的是每 weight **2-bit storage**；ternary 的 `log2(3)≈1.58` 命名不是声称 helper 采用每 weight 1.58-bit 的实际编码。

但是全仓库搜索只找到两个 `quantize_weights()` 定义，未找到默认流程调用。即使手动启用，forward 仍先恢复 floating weight，再执行 `F.linear`，不是 fused ternary×INT8 matrix multiplication。可减少持久 weight buffer，不等于已经实现论文的 latency gain；运行时还有解包临时内存。

### 3.3 v2 所述 BitBLAS backend 未在公开 BitVLA 路径中找到

论文 §III-A 指定 BitBLAS custom kernel；§IV 报告 A100 上的 latency/throughput，采用 K=25 的 action chunks，baseline 数值取自 OpenVLA-OFT。

当前 repo 搜索 BitBLAS 只发现 Transformers 通用 quantization_config 的提及，未发现 BitVLA 的 import/operator、转换脚本或该 benchmark 的完整执行路径。模型实际 forward 为上面的 floating `F.linear`。

因此 **1.4 GB、73 ms、4.4× 仍属于作者报告结果，本次未独立复现**。不能断言“加速不可能”，也不能把默认 fake-quant evaluation 的速度作为论文 BitBLAS 结果的复现。baseline 引用既有数值也意味着并非本仓库统一环境重测全部方法。

## 4. 不能误解成造假的地方

- “1-bit”在这里是 ternary/1.58-bit 命名，不是只能有两个数值。
- 摘要“every parameter is ternary”字面上过强：论文正文已明确保留 connector/action head 和 input/output embedding；代码也存在 floating-point normalization 和这些模块。准确表述应是主要 backbone linear weights 采用 ternary。
- BF16 master checkpoint 与 QAT 不冲突；真正要查的是 forward 是否使用 quantizer，部署是否用 packed/kernel。本次前者通过，后者有缺口。
- 模型 evaluation 不需要 distillation teacher；teacher 没出现在 inference 里不是错误。缺失的是生成 student 的训练流程。
- v1 与 v2 结果变化不能自动当成造假；它们对应不同 robotics pre-training 状态。
- 通用 Transformers fork 内的其他 distillation/quantization 支持不能自动算作 BitVLA 的方法实现。

## 5. 两处明确工程/文档问题

1. **OFT 启动脚本的 cwd 不匹配。** README 指示在 `openvla-oft/` 执行 `sh ft_script/ft_bitvla_libero_spatial.sh`，但 script 第 1 行指向 `../vla-scripts/finetune_bitnet.py`。shell 不会自动切换至 script 所在目录；按 README cwd 解析目标不存在。测试验证在 `ft_script/` cwd 才可解析该路径。这是启动路径错误，不是算法失败。
2. **LIBERO-Goal checkpoint 链接指向 Long。** README L57 的 Goal 行与 Long 行均链接到 `...libero_long-bf16`。不能直接用该链接进行 Goal 成功率复现。

另有一项静态风险：新 checkpoint config 的 auto_map 指向 `configuration_bitvla.py` / `bitvla.py`，本轮 Hugging Face tree 未列这些文件。GitHub 本地注册代码可能提供替代加载路径；由于未实际运行整模型 loader，不把它升级为所有加载方式必然失败的结论。

## 6. 与此前 QVLA 核查的区别

| 项目 | QVLA 本轮此前发现 | BitVLA 本次发现 |
|---|---|---|
| 核心数值机制 | action sensitivity 被 local covariance proxy 替代 | ternary weight、INT8 activation、STE 均有实现并通过小测试 |
| 训练创新 | action-space precise refinement 未接入 | Quantize-then-Distill 整阶段代码未找到，无法从头验证 |
| 默认运行 | weight-only fake quant | weight + activation fake quant，有 optional packing helper |
| 硬件结果复现 | 缺少所需 runtime | 缺少论文 BitBLAS 接入与测量路径 |
| 开源边界披露 | README 标出 fake quant，但论文结果桥接不足 | 明确 master weights、offline conversion、pre-training code 待发布 |

## 对 FYP 的使用建议

BitVLA 比较适合研究已发布 low-bit backbone 的 forward/OFT 行为；不能视为已经开放了完整训练和 deployment pipeline。若作为 baseline，分别记录：checkpoint revision、是否 OXE pre-trained、vision bit config、fake-quant 或 packed backend、action chunk 和输入 image 数。

向作者请求的优先材料是 Quantize-then-Distill recipe、BitBLAS conversion/inference/benchmark script，以及新版每个 LIBERO suite 的对应 checkpoint。对于 FYP 报告，性能表写“paper-reported”，你实际跑到的 fake-quant 结果另外标注。

本报告由 AI 辅助完成，证据包括论文、固定 commit 源代码、官方 metadata 和 CPU 单元级验证。未联系作者、未发 issue、未运行闭环 benchmark。公开讨论快照见 [issues.json](issues.json)、[comments.json](comments.json)；OWNER 对预训练发布范围有回复，对数据也提供了链接，不能套用 QVLA 的“所检查问题未获澄清”描述。
