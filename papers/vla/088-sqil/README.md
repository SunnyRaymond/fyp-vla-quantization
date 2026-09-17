# 02 — SQIL: Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control

- 定位：以 mission-critical states 為中心的 VLA-aware Quantization-Aware Training (QAT)。
- Source：[ICCV 2025 final](https://openaccess.thecvf.com/content/ICCV2025/html/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.html)
- Local PDF：[paper-iccv-final.pdf](paper-iccv-final.pdf)

## 為何納入

SQIL 明確以 OpenVLA 與 Pi-0 等 robotic policies 為對象，研究 low-bit quantization 如何在 closed-loop control 中保留關鍵 decision states。

## Background / Problem

一般 QAT 對所有 demonstrations 近似等權，但 robot trajectory 中真正決定成敗的 grasp、contact、alignment states 只佔少數。若量化後在這些 states 偏離，即使平均 imitation loss 很低也可能 task failure。

## Method 與 Key Innovation

- `State-Importance Score (SIS)`：以 visual perturbation 前後的 action discrepancy 找出 mission-critical states。
- `Quantization-Robust Action Distillation (QRD)`：讓 quantized student 更重視這些 states，對齊 full-precision teacher 的 action。
- 可搭配 AWQ 或 QuaRot 的 4-bit weights；OpenVLA 設定以 QLoRA 更新約 110M parameters。

## Main Results

- OpenVLA LIBERO average：full precision 約 73.8%；SQIL-AWQ W4 約 73.2%，SQIL-QuaRot W4 約 73.3%。
- Pi-0：full precision 93.8%，PTQ 92.0%，一般 QAT 92.6%，SQIL 93.3%；model size 3.3 GB→1.1 GB。
- Real UR5，3 tasks、每 task 30 trials：79% full precision、67% PTQ、71% QAT、77% SQIL。
- Jetson AGX Orin：OpenVLA BF16 15.2 GB / 955.2 ms；INT4 4.0 GB / 374.7 ms，約 2.5× speedup；paper 亦報告約 2.5× energy saving。

## Limitations / Evidence Boundary

SQIL 需要 training 與 task demonstrations，不能視為 drop-in PTQ。SIS 依賴 perturbation design；若 calibration/training trajectory 沒覆蓋真正的 rare failure states，saliency map 仍可能錯。

## Why It Matters

它把 VLA quantization 的目標從 average representation fidelity 改成 state-weighted control fidelity，而且同時提供 simulation、real robot、Jetson latency、memory 與 energy evidence，是 corpus 中 deployment measurement 較完整的 paper。

## Reading Route

- 20 minutes：SIS、QRD、Table 1–3。
- 60 minutes：追蹤 perturbation 如何產生、state weights 如何進 loss，再核對 Jetson measurement boundary。

## Reading Questions（不附答案）

1. SIS 衡量的是 task causality 還是 perturbation sensitivity？
2. 為何 critical-state weighting 比一般 QAT 更適合 long-horizon control？
3. Real-robot 30 trials/task 的 uncertainty 有多大？
4. AWQ 與 QuaRot 在這裡只處理 weights，還是也改 activation path？
5. SIS 能否用 contact、force 或 kinematics 做更便宜的 proxy？

## Meeting Card

SQIL 的核心不是新的 4-bit format，而是用 action saliency 告訴 QAT「哪些 trajectory states 絕對不能量壞」。

