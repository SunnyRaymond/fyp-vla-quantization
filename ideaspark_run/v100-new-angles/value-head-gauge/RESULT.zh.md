# Value-head gauge：implementation_inconclusive

GPU **64807**（TC1N07，V100）23s 完成四臂，CPU **64809** 4s 独立复算全部raw、来源及量化规则。正式结论 **implementation_inconclusive**：FP original/centered 未通过冻结的 no-op 容差，不能判 preliminary_go。保留全部结果，原协议不扩验、不改容差。

| arm | 8-state mean two-Q pair-average error A |
|---|---:|
| FP-original | 0 |
| FP-centered | 6.29923e-9 |
| W4 RTN original | 159.418782 |
| W4 RTN centered | 59.468802 |

W4 描述性 median gain **62.38%**，8/8 state改善（60.13%–64.92%）。Pristine weight的pooled common-mode energy ratio为 **0.77525**，global binding成立。但 FP probability 最大变化 **1.0669e-5**，decoded Q 最大变化 **3.9673e-4**，均未通过预设 elementwise no-op gate；小的平均FP误差不能事后替代该门槛。该结果提示有数值效应，尚不构成受控机制通过，也不认证新的centering方法。

CPU64808初次重算的RTN grid exact检查失败，是NumPy直接division与PyTorch CUDA scalar reciprocal-multiplication的实现差异。固定原GPU recipe后，CPU64809两臂integer/scale/dequant exact复算及所有来源检查通过；没有重跑模型，没有改变scientific gate。详情见 [root处理](ROOT_VERIFIER_RESOLUTION.zh.md)。FP no-op失败是独立问题，未被这个修复消除。

输入是独立reset seeds5209..5216，64个FP-imagined H3 candidates；只量化五个Q final Linear，全部10种pair纳入，未执行environment step。Source `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`；seed3 checkpoint SHA256 `0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2`。Raw SHA256 `fe44cf47974730201087adf01723f913a4647c81639b5a5c5d7934e37d70ff13`，远端 `artifacts/64807/raw_gauge.npz`，本地 [最终verification](artifacts/64809/verification.json)。原 [64808](artifacts/64808/verification.json) 同样保留。

独立 [arithmetic repair gate](ARITHMETIC_REPAIR_GATE.zh.md) 已完成，root接受 **STOP**：现象与FP32 roundoff相容，但未证明具体coding defect；改用FP64会改变算术问题，不能解除原gate。不给漂亮的W4数值安排额外算术实验，不放宽容差，也不扩成完整验证。
