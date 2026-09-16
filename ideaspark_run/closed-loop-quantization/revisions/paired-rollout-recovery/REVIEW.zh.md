# PRR bounded review

**Verdict：revise / conditional adaptation。** 方案可以作为受控的 WM quantization adaptation screen；不能作为已成立的新 PTQ principle。IDEA 与 EXPERIMENT_PLAN 已正确保留旧 FRT no-go，并把 target（local vs paired-clean）与 parameter freedom（quantizer-only vs LoRA-r4）分开；四格即使 B 失败也完成，避免选择性验证 capacity branch。固定 Q0/旧 clean donor、同 action prefix、H=2、无 online refresh，以及最终 hard W4 merge/requant、LoRA STE 梯度检查，足以修复旧 fresh-bank 的 context/direction 混淆。

唯一需要在执行前锁死的解释条件是：R1 的主结论只能是 **target × parameterization 的配对差异**。`paired-clean` 与 `local` 的 teacher target 范数可能不同，故必须报告每格 target norm、target-loss/clean-loss 与实际梯度，而不能把 B−A 或 D−C 直接写成“恢复机制证明”。若 R2 的 random corruption、direct unroll 或 action-response 对照未完成，只能称 conditional adaptation；R3 的 free rollout/环境 endpoint 也必须保持独立预算。除此之外不建议继续添加方法或扩大 horizon。

Root 回应：已将 target norm、loss 比例和参数梯度日志写入实验草案。阶段位置澄清：R1 已包含 H=2 的自身自由 rollout；R3 是独立 TEST 的 CEM/环境闭环，预算另列。新版的受控设计不修复或撤销旧实验数据，只避免在新 target 对照中再次混入不同 context。此处是编辑回应，未声称完成实验验证。
