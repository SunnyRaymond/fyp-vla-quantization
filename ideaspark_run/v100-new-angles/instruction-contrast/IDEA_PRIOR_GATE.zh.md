# Instruction-contrast / numerical PTQ：问题与 prior gate

审查日期：2026-09-13  
范围：只审查可检验问题、primary prior 与最小 V100 screen；本文件没有运行模型、下载资产或提交作业。

## 精确研究假设

对每个固定样本 \(x_i=(V_i,s_i,\epsilon_i)\)，保持 image \(V_i\)、state \(s_i\)、flow 初始 noise \(\epsilon_i\)、processor、时间步和随机状态完全相同，只把 instruction 从 \(I_i^+\) 改为配对的 \(I_i^-\)。两条 instruction 使用同一模板和一个预注册的语义槽位替换（例如 target 或 spatial relation）；'I_i^-' 是否对当前画面构成 contradictory/OOD 必须逐样本记录，不能事后解释为 grounding 证据。

令 \(a_{m,i}^{+/-}\) 是 arm \(m\) 的完整 raw action output（按实际 checkpoint shape 展平，不预设 chunk 长度），其中 \(m\in\{FP32,W4\text{-language},W4\text{-expert}\}\)，且 forward 仍为 FP32 fake-quant。定义

\[
C_{m,i}=a_{m,i}^{+}-a_{m,i}^{-},\qquad
D_{m,i}=\frac{\lVert C_{m,i}-C_{FP32,i}\rVert_2}
{\lVert C_{FP32,i}\rVert_2+\epsilon}.
\]

**H1：在相同 \(x_i\) 与 noise 下，language-side W4 对 instruction-contrast 的失真 \(D\) 大于 action-expert/head-only W4；这一差异在同时报告原 instruction 的 action MSE 后仍存在，因此不能由单一原 prompt 的 output drift 解释。** 这是关于数值 PTQ 改变输入 instruction response 的可证伪假设，不是 task-success 或 grounding-success 假设。

若 \(\lVert C_{FP32,i}\rVert_2\) 接近零，该样本的 \(D_{m,i}\) 不可识别，应标为 'degenerate_fp_contrast' 并按预注册规则 fail-closed；不能用任意小常数制造强结论。另报 contrast-direction projection
\(\langle C_{m,i},C_{FP32,i}\rangle/(\lVert C_{FP32,i}\rVert_2^2+\epsilon)\)，用于区分 attenuation、sign reversal 与单纯幅度漂移。

## 两条 closest prior 及重叠边界

1. **Mix-QVLA（[arXiv:2606.19565v1](https://arxiv.org/abs/2606.19565)，[HTML](https://arxiv.org/html/2606.19565v1)）**。它把 quantized policy 与固定的 FP action-token reference 对齐，并在 vision output、projector、language-policy、action-head 等 boundary 上计算 gradient-weighted task-evidence map、evidence mass、attribution JSD 和 temporal sensitivity；它明确指出只看 final action deviation 会漏掉内部 evidence 变化。重叠在“同一 VLA 的数值量化与 FP 参照、按模块定位变化”。关键差异是 Mix-QVLA 保持 instruction/input 不变，以 teacher-forced FP action-token reference 做 attribution/diagnostic；它没有执行 \(do(I=I^-)\) 的 paired input intervention，也没有测 \(C_m=a_m(I^+)-a_m(I^-)\)。其 evidence map 被作者定位为 diagnostic signal，不能直接当作 causal instruction-grounding 证据。

2. **IGAR / ICBench（[项目页](https://ray-nh.github.io/igar/)，[arXiv:2603.06001](https://arxiv.org/abs/2603.06001)）**。它在视觉环境保持不变时，使用受控的 contradictory/OOD instruction 检查 policy 是否仍跟随 visual prior，再用 forward-pass attention recalibration 使 attention 更多指向 instruction token。重叠在“固定画面、改变语言、显式测语言敏感性，并把正常行为与语言诊断分开”。差异是 IGAR 干预 hidden attention/forward computation，而不是 PTQ weights；它的 benchmark 与 intervention 不能证明 FP32 与 W4 的 paired numerical contrast 差异。当前只确认项目页和论文；未把猜测的 GitHub 地址当作 official code 证据（该地址检查为不可用）。

**Adjacent quantization comparator（不计入上述两条 closest prior）：QVLA（[arXiv:2602.03782v1](https://arxiv.org/abs/2602.03782)，[HTML](https://arxiv.org/html/2602.03782v1)）**按 action-space sensitivity 做 channel-wise bit allocation，并优化最终 action distribution/KL 或 action drift。它适合提供 action-level PTQ baseline，却没有同 image/state/noise 下的 instruction pair，因此不能替代 instruction-contrast 测试。

## 不可忽略的 baseline 与混淆

- **FP32 也可能没有 grounding。** \(C_{FP32}\) 小、方向不稳定或对多数样本不响应时，只能结论为“该 screen 的 FP contrast 不可识别/未显示语言响应”；不能把 W4 与 FP 的差异包装成 PTQ 造成的 grounding loss。
- **Contradictory prompt 的 OOD confound。** 'I^-' 可能改变的不只是语义，还包括 token length、padding、截断、词形频率或 processor 分支。冻结 tokenizer/processor，保持模板和 token budget，记录 token IDs、长度、是否 truncation，以及该替换是否为数据分布外；一个两 instruction screen 不能消除 OOD 因果混淆。
- **原 prompt action MSE 不等于 instruction causality。** 只测 \(\lVert a_m^+-a_{FP32}^+\rVert\) 会把整体 action drift 与对 instruction 改动的响应混在一起；必须保存两条 prompt 的 raw outputs，并同时报告 action MSE 与 \(D\)。
- 所有 arms 使用同一 inference mode、normalizer、时间步、初始 noise 与 model snapshot；禁止对某个 arm 另抽 noise 或另跑 closed-loop trajectory。若 SmolVLA 实际没有可独立量化的 language-side 或 expert-side module，记录 structural no-go，不用近似模块冒充。

## 最小 V100 screen gate（尚未执行）

资源上限为 **12 个固定 image/state samples × 2 instructions × 3 arms**，每个 sample 使用一个预注册且在两 instruction、三 arms 间复用的 flow noise；只做一次离线 action forward，禁止完整 MPC 或 task-success benchmark。三 arms 为 FP32、language-side W4 fake-quant、action-expert/head-only W4 fake-quant；其余权重、processor 和 postprocessor 固定。完整 output 的实际 shape、chunk/action horizon、gripper 表示和 normalizer identity 从 checkpoint/runtime evidence 读取并写入结果，不能凭常见 SmolVLA shape 猜测。

进入 gate 前必须满足：

1. 真实 GPU hostname、allocation owner、partition/job evidence 确认为 V100；V100 路径只依赖已验证的 FP16 支持，不假定 BF16。FP32 reference 与 fake-quant 算术保持明确记录。
2. 12 个 pair 的 image/state、两条 instruction 的 tokenization、noise seed、时间步和 snapshot identity 均可审计；输出数量为 \(12\times2\times3\)，全部 finite，且无 processor truncation 或 silent fallback。
3. 至少 8/12 个样本满足 \(\lVert C_{FP32}\rVert\) 高于预注册识别阈值；否则 gate 为 'inconclusive_fp_not_grounded'，不比较“谁更 grounded”。
4. 机制候选方向预注册为：'mean(D_W4-language) > mean(D_W4-expert)'，并且逐样本方向至少在 8/12 个 pair 同向；同时保留每个 sample 的 raw outputs、\(D\)、direction projection、两种原 prompt action MSE 和 OOD/token metadata。该方向性条件只是 screen gate，样本量不足以认证统计显著性或新颖性。

最小结果只允许三种状态：'directional_signal'、'null_or_opposite'、'inconclusive_fp_not_grounded'。无论哪一种，都不报告 task success、真实 robot performance、训练收益或“novelty certified”。若 'directional_signal'，下一步仍需单独设计更大样本、有效 instruction pair 与真实 closed-loop validation；若 'null_or_opposite'，该切面停止扩展。

## 结论边界

这个切面值得作为一个窄的、可证伪的 numerical-PTQ diagnostic：它询问“量化后模型对同一视觉/状态输入的 instruction intervention response 是否改变”，而不是重复 action MSE 或 attribution map。现有 prior 已覆盖 task-evidence attribution、action-space sensitivity 与 forward attention intervention 的相邻部分；截至本审查，不能据此声称该问题已获新颖性认证。当前文档也没有确认 SmolVLA 资产、module split 或任何实验结果。

