# Gripper decision-margin PTQ prior gate

**机制。** OpenVLA-OFT 官方 robot_utils adapter 是 continuous L1 action head，非 token quantization。LIBERO 将 g∈[0,1]→[-1,1]，sign 后反转：g<.5 close、g>.5 open、g=.5 为 0；无 hysteresis。LeRobot VLA-JEPA 则先 normalized .5 snap，再按 unnormalized physical threshold 输出 ±1，单位错会恒定。LIBERO action 为 7D [-1,1]，dummy gripper=-1。mapping、polarity、tie、hysteresis 必须从实际 adapter 确认。

**风险与 prior。** PTQ 小 g 漂移跨 threshold 会翻转 binary command，六个连续位的总 MSE 可能掩盖它；无闭环证据时只能称 decision flip。QVLA (arXiv:2602.03782) 与 ActQuant (arXiv:2605.24011) 用 action-sensitive PTQ；QVLA 定性提到 gripper overshoot/release，但未独立量化 margin、flip-rate、tie 或 hysteresis。Mix-QVLA (arXiv:2606.19565) 测固定 FP reference 的 task-evidence，不测 continuous-to-binary boundary。因此最多是 execution-interface diagnostic，不是新 quantizer。

**严格 claim / gate。** mapping、单位、tie、hysteresis 未冻结可审计即 structural_no_go；否则只主张 PTQ 是否改变同一输入 action 的 binary command。最小 screen：≤12 samples、FP32+两个预注册 W4 arms、同 image/state/noise、一次离线输出；保存 raw g_FP/g_q、threshold、polarity/tie/hysteresis、margin、binary、flip。至少 2/12 flips，每个满足 |g_q−g_FP|≥|g_FP−threshold|（tie 加容差），且远 margin 无 flip，才记 conditional_margin_signal，否则 null_or_inconclusive。不写 grounding、task success、部署收益。
