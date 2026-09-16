# Frozen execution amendment

本文件与 PROTOCOL_DRAFT.zh.md 共同构成冻结协议；本文件优先。冻结发生在该候选任何模型输出之前。checkpoint 仅按 strict compatibility 选择，CPU 64795 seed2 失败、seed3 通过，无 inference。

Source `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`；checkpoint `nicklashansen/tdmpc2@73a50e2719ed8258c72c7d1fefd23b781d66e35e/dmcontrol/cartpole-balance-3.pt`，SHA256 `0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2`，31,344,610 bytes。Parent manifest SHA256 `9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369`；child CPU prep 仅生成原定 fresh resets，不选择样本。

根据独立审查固定以下执行语义：

- 所有 arms 都在开始及结束恢复同一 pristine FP snapshot。中心化从原始 W,b 计算，不能中心化已经量化的权重。模型 FP32/eval/no-grad，全部 arms 共用 bit-identical FP z/action cache。
- 量化 zero row 用 scale=1、code=0、dequant=0。其他 row 为 absmax/7，torch.round，clamp[-7,7]，FP32 dequant。
- Global common-mode ratio 使用 **pooled** 五个 member：`101*sum_{member,input}(mean_bin W)^2 / sum_{member,bin,input} W^2`，从 pristine weight 计算。denominator=0 则 inconclusive_no_gauge；非finite raw 为 engineering failure。ratio>1e-8，并在全部五个 member 的任一 row 上 original/centered scale 的 `abs(sc-so)/abs(so)>1e-6` 或 integer code 有变化，才算 global binding。so 对 zero row 已定义为1。没有选择最佳 member。
- `A_original` 明确为 **W4-RTN-original** 相对 FP-original 的全部10 pair-avg squared error；`A_centered` 是 **W4-RTN-centered** 同样误差。FP-centered 只用于 no-op。8-state、64-candidate、10-pair 全部纳入，无删除。

其他 seeds、no-op、全8 state A_original>1e-8、median gain≥25%、≥6/8严格改善且全集 mean下降、walltime 与不扩张验证约束保持草案。结论无论正负均停止该候选。
