# Instruction-contrast：bounded falsification

审查日期：2026-09-13  
范围：对 IDEA_PRIOR_GATE.zh.md 的 H1 做反证审查。没有读取任何 SmolVLA 输出，没有运行数值实验、下载资产或提交作业。

## 先否定原 H1 的两个隐含前提

### 1. W4-language > W4-expert 可能只是扰动预算不同

如果 language-side 包含的参数远多于 action expert/head，或者两侧的 relative quantization error 不同，\(D_{W4\text{-language}}>D_{W4\text{-expert}}\) 只能说明这两个具体 arm 的总体扰动不同。它不能说明 language location 对 instruction response 更敏感。即使两 arm 都叫 W4，量化参数数量、层数、每层 scale 分布、饱和比例和累计误差都可能不相同。

因此原 H1 的模块机制解释必须降级。预先记录每个 arm 的：

\[
N_m=\text{quantized parameter count},\qquad
\delta_m=
\frac{\sum_{\ell\in m}\lVert W_{q,\ell}-W_{FP,\ell}\rVert_F^2}
{\sum_{\ell\in m}\lVert W_{FP,\ell}\rVert_F^2+\epsilon},
\]

以及每层的 relative error、clipping/saturation fraction 和 quantizer integer/grid audit。\(D_m\) 必须与 \(\delta_m\) 并列看；可以另外报告 \(D_m/(\delta_m+\epsilon)\)，但该比值只是描述性 normalization，不会自动消除非线性传播或 layer-count confound。

只有在 preflight 证明量化模块身份、\(N_m\)、\(\delta_m\) 和 fake-quant recipe 可审计时，才允许比较这两个 arm。若相对扰动预算无法匹配，结果最多是“该 frozen arm configuration 的 contrast drift 不同”；不允许写成 language-side causal sensitivity。当前没有 verified SmolVLA module split 或预算证据，因此原 H1 的强机制表述是 **conceptual_no_go**，直到 preflight 通过。

### 2. 原 prompt MSE 不足以排除 common-mode output drift

对两条 instruction 分别定义

\[
e_{m,i}^{+}=a_{m,i}^{+}-a_{FP32,i}^{+},\qquad
e_{m,i}^{-}=a_{m,i}^{-}-a_{FP32,i}^{-}.
\]

则

\[
C_{m,i}-C_{FP32,i}=e_{m,i}^{+}-e_{m,i}^{-}.
\]

所以只报告 \(M_{m,i}^{+}=\lVert e_{m,i}^{+}\rVert^2\)（甚至同时报告 \(M^-\) 的两个均值）仍不能说明量化误差是否随 instruction 改变。若 \(e^+\approx e^-\)，模型可能有明显 action drift，但 instruction contrast 几乎保持；反之，较小的总 drift 也可能主要是 prompt-specific。screen 至少必须保存两条 prompt 的 raw outputs，并计算：

\[
O_{m,i}=\frac{\lVert e_{m,i}^{+}\rVert_2^2+\lVert e_{m,i}^{-}\rVert_2^2}{2},
\qquad
Q_{m,i}=
\frac{\lVert e_{m,i}^{+}-e_{m,i}^{-}\rVert_2^2}
{\lVert e_{m,i}^{+}\rVert_2^2+\lVert e_{m,i}^{-}\rVert_2^2+\epsilon}.
\]

其中 \(O\) 是两 prompt 的总体 output drift，\(Q\) 是其中具有 instruction-specific 的部分。原 brief 的 \(D\) 仍作为 contrast deviation，但不能用单 prompt action MSE 替代 \(Q\)。若只得到一条 prompt、无法复算 \(e^+-e^-\)，直接判为 contrast_unidentifiable。

## 更窄且可反证的最小 claim

在固定的 12 个 image/state/noise pair、固定 processor 和固定 snapshot 上，**条件性 claim** 是：

> 在已公开报告 \(N_m\)、\(\delta_m\)、\(O_m\) 与 \(Q_m\) 的前提下，指定的 W4-language arm 是否比指定的 W4-expert arm 产生更大的同输入 paired instruction-contrast deviation \(D\)，并且这种差异是否伴随更高的 instruction-specific drift \(Q\)。

这个 claim 只描述 frozen numerical configurations 的 paired input response。它不等价于 language mechanism，不等价于 grounding，也不声称两 arm 具有相同计算量、同等部署成本或相同 task competence。若 \(I^-\) 不是当前 scene 中有外部证据支持的合法 alternative，只能将结果命名为 input_response_under_controlled_or_ood_intervention；不得称为 grounding degradation。

## 不超过 12 samples 的反证 gate

保持原资源边界：12 个固定 image/state samples、两条 instruction、三 arms（FP32、language-side W4、action-expert/head-only W4），每个 pair 复用同一 flow noise；不训练、不跑 closed-loop、不看结果后改指标。

1. **结构与预算 preflight。** 在任何 forward 前冻结实际 module names、\(N_m\)、\(W_{FP}\rightarrow W_q\) recipe、\(\delta_m\)、每层 saturation 与 processor/tokenization identity。缺少其中任一项，状态为 conceptual_no_go。若 \(N_m\) 或 \(\delta_m\) 明显不匹配，禁止 module-causal wording；只保留 conditional arm comparison。
2. **输入干预 preflight。** 两条 instruction 必须使用预注册模板和单一语义槽位替换；记录 token IDs、长度、padding/truncation、processor branch 与 scene-valid/OOD 标记。若不能证明 12 个 pair 的 scene-valid alternative，grounding 解释永久关闭，但仍可做 input-response-only diagnostic。
3. **完整 paired outputs。** 三 arms 均须输出同一实际 action tensor shape 的 \(a^+\) 和 \(a^-\)，共 \(12\times2\times3\) 个 finite outputs；否则为 contrast_unidentifiable。不得按 SmolVLA 常见 shape 猜 chunk、gripper 或 horizon。
4. **FP 可识别性。** 至少 8/12 个 pair 的 \(\lVert C_{FP32}\rVert_2\) 超过预注册阈值。否则为 inconclusive_fp_not_grounded，因为 FP reference 自身没有可识别的 instruction contrast。
5. **描述性反证规则。** 只有在至少 8/12 个有效 pair 同时满足 \(D_{W4\text{-language}}>D_{W4\text{-expert}}\) 与 \(Q_{W4\text{-language}}>Q_{W4\text{-expert}}\)，且 aggregate means 保持同方向时，才记录 conditional_contrast_signal。任一条件失败，记录 null_or_opposite；若仅 \(D\) 变大而 \(Q\) 不变或变小，优先解释为 common-mode/output drift，否定“instruction-specific PTQ effect”。所有结果必须附 \(N,\delta,O,Q,D\) 原始表，不能只给一个平均 action MSE。

## 最终解释边界

当前可准备资产没有提供足够证据证明 SmolVLA 的 language-side/expert-side 可按上述方式隔离，也没有预先确认 12 个 instruction pairs 均为 scene-legal alternatives。因此当前阶段对“PTQ 改变 grounding 的 language-side 机制”作 **conceptual_no_go**；唯一可预冻的是上述 input-response-only conditional screen。

即使 screen 得到 conditional_contrast_signal，也只能说明在这组 frozen arms、pair 和 noise 下存在 paired numerical contrast 差异。它不能证明 task success、真实 robot grounding、因果机制、统计显著性或 research novelty。不能复用后续 flow 输出挑选 instruction pair、阈值或指标。

