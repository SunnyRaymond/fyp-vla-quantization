# FP32 no-op arithmetic repair gate

## 结论

64807 GPU raw 与 64809 CPU verification 的 source、checkpoint、quantizer、snapshot/readback 和 independent decoder checks 已通过。冻结的 FP-centered no-op 仍失败：probability 最大绝对差为 `1.0669e-5`，decoded Q 最大绝对差为 `3.967e-4`，分别高于 `1e-7/1e-6` 与 `1e-5/1e-6` 的冻结门槛。`A_FPcenter` 约为 `6.299e-9`，而 W4-RTN 的 `A_original → A_centered` 为 `159.419 → 59.469`、8/8 state 改善；这些 positive 数字不能绕过 no-op gate。

现有证据与 FP32 roundoff **相容**，但不能把失败重新分类为已经证实的 arithmetic floor。中心化在 exact arithmetic 中只给 101 个 logits 加同一个 `h` 相关常数；在实际 FP32 路径中，均值、centered weight 的 matrix multiply、softmax 和 `symexp` 都会有不同的舍入。另一方面，当前的 source/quantizer/readback 通过只能排除若干实现错误，不能单凭 `A_FPcenter` 很小证明残差必然来自 FP32 arithmetic。冻结 no-op 是目标运行语义的 implementation gate，不能因 W4 effect 很大而放宽。

因此本次应保持：

```text
decision = implementation_inconclusive
```

并停止该候选。不能称为 `method_no_go`，因为 no-op 失败阻止了科学比较；也不能称为 `preliminary_go`。不建议为挽救结果再跑 GPU repair、增加 seeds 或改变 tolerance。

## 为什么 FP64 不是同一 pipeline 的 repair

把同一 FP32 checkpoint 数值 cast 到 FP64，并在 FP64 中重新 center、matmul、softmax 和 decoder，会改变实验的 arithmetic contract。若 scale、integer code 或 dequant 也在 FP64 重新计算，还会改变 W4 quantizer 的网格，成为另一个 quantizer；其 W4 MSE 不能与 64807 的 W4 arm 比较。即使只把 W4 dequant 后的值提升到 FP64，得到的也只能是 arithmetic attribution，不是冻结的 FP32 fake-quantization 结果。

此外，现有 raw 的 `terminal_latents` 是 Q 调用的上游输入，未必是最后 `Linear(101,512)` 的实际 hidden input。要归因到最后 head，必须在同一固定 z/action cache 上一次性捕获最终 head 输入 `h`，之后四臂共享 bit-identical `h`；只对 terminal latent 做 FP64 计算不能保证隔离 upstream MLP 的差异。

## 若 root 仍需要一次 explanatory diagnostic

这只能作为独立的 postmortem artifact，不能改写 64809，也不能产生 preliminary-go。运行前必须固定以下边界：

1. 使用同一 8-state、64-candidate、H3 cache 和同一 pinned FP32 checkpoint；不增加 state、seed、task 或 rollout。一次 FP32 upstream forward 保存最终 Q-head 输入 `h`，所有 arms 使用同一份 `h`。
2. 将原始 `W,b` 的 FP32 数值精确 cast 到 FP64，再在 FP64 中构造 `Wc,b_c`、final-head matmul、101-bin softmax 与 official `two_hot_inv`。四臂的 arithmetic dtype 和调用顺序必须相同；禁止 AMP、TF32、混合 dtype 或按结果选择实现。
3. 若要检查 W4 的 arithmetic attribution，必须复用 64807 已保存的 FP32 `scale/code/dequant`，仅将这些已冻结 grid 值提升到 FP64。任何 FP64 重新求 scale/code 的结果单独标成 oracle，不能进入原 W4 gate、A 或 gain。
4. 在运行前固定 FP64 diagnostic 的数值判据，并同时报告 FP32 与 FP64 的 probability/Q residual、head-input hash、quantizer hash 和每臂 dtype。不得用 64809 的最大差反推新的 tolerance，也不得把“FP64 通过”当作原 FP32 no-op 通过。

只有在共享 `h`、quantizer grid 不变且预注册的 FP64 no-op 明显回到 machine-precision 范围时，才可写成“FP32 arithmetic residual 的解释性证据”；否则写 `unexplained_inconclusive`。两种结果都不解除原冻结 gate，且都不支持 full validation、task success、native low-bit deployment 或 novelty claim。

## 最终审查判定

当前没有足够证据把 64809 的失败归类为可修复的 coding bug；更没有理由通过改变 arithmetic contract 来救援 positive W4 数字。FP64 方案在概念上可作为一次窄的 attribution diagnostic，但它是 scope change，不是现有 pipeline 的 repair。本审查建议保持 STOP，并保留 64807/64809 的原始 `implementation_inconclusive` 记录。
