# TD-MPC2 existing-Q stratified rounding：inconclusive_binding，停止

GPU **64799**（TC1N06，V100-PCIE-32GB）18s 完成，实际 screen 13.05s；CPU **64801** 4s 独立复算全部 raw，engineering/identity/restore/API/quantizer readback 检查通过。冻结判定是 **inconclusive_binding**；没有足够可比的 member-level error，不能把它写成已证实或已推翻 joint-rounding mechanism。没有可确认的剩余实现缺陷，不扩 seed、任务或完整闭环。

8 个 fresh reset state、每个64个 FP-imagined H3 candidate、3个 rounding seed；主量为全部10种 two-Q pair average 的 squared error，先在 state 内平均。

| arm | 全8-state mean pair-average error A |
|---|---:|
| FP32 | 0 |
| W4 RTN | 151.640494 |
| W4 independent SR | 71.497266 |
| W4 stratified SR | 92.417425 |

Stratified 相对 independent 的每state member-MSE ratio 为 **1.379–1.457**，全部超出冻结的 [0.9,1.1]，binding **0/8**。全8-state median gain **−29.02%**，0/8改善。cross-term 虽下降，但 member error 增幅超过其收益；这里的有限 seed 结果不能证实“保持 individual error 而改善 coupling”的假设。Stratified 8/8优于 RTN 是描述性结果；协议要求先通过 mechanism，不能用这个比较绕过 binding。

最终使用 source `e9f59321933cbc8e11a002b842adc7d4ffae8ff1` 与官方 seed3 checkpoint，SHA256 `0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2`。所有 source/runtime/manifest、精度事务、FP cache、官方 Q(avg) 对照已随 job 保存。Raw SHA256 `3cb7c5d528df468c16880e8538a60f50ddac407dbce5cb07f94a09a592dee099`，远端 `artifacts/64799/raw_tdq.npz`；本地独立 [verification](artifacts/64801/verification.json) 与 [summary](artifacts/64799/summary.json)。仅 fake quantization，不声明 native 低bit加速、环境 return 或作者 policy reproduction。

工程历史全部保留：CPU准备64780–64790的依赖/路径修复见 prep 文档；GPU64793 seed1 source/checkpoint mismatch；CPU64794 schema检查；CPU64795 固定2→3只按strict-load选择（seed2不兼容、seed3通过，无inference）；GPU64797 参数绑定检查失败；CPU64798证实TensorDict state_dict是序列化clone，实际live/detach共享storage；修复实际参数读写并加入fresh readback后，64799完整执行。前述失败均未产生本候选Q结果，没有基于结果修改gate。
