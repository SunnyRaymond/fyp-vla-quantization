# 12 — DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models

- 定位：以 long-horizon kinematic drift 指導 CogACT mixed-precision PTQ。
- Source：[arXiv 2604.11572](https://arxiv.org/abs/2604.11572)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

DA-PTQ 直接研究 VLA W4A8 quantization，將 quantization error 從單步 reconstruction 問題改寫成 sequential trajectory drift 問題。

## Background / Problem

在 VLA closed loop 中，vision-language conditioning 的小 perturbation 會進入 DiT action head，再經 robot dynamics 累積成 kinematic drift。Single-step action deviation 無法完全捕捉 joint geometry 與 long horizon amplification。

## Method 與 Key Innovation

- `Cross-Space Representation Compensation (CSRC)`：以 affine / low-rank transforms 對齊 full-precision 與 quantized interface statistics，之後 fold 入 weights。
- `Drift-Aware Mixed-Precision Allocation (DA-MPA)`：利用 structural Jacobian 與 trajectory-level motion error，將 drift-sensitive layers 保留 BF16，其餘 W4。
- 以 512 條 BridgeData V2 trajectories calibration；不 fine-tune model，補償在 inference 前完成。

## Main Results

- SIMPLER WidowX Visual Matching：CogACT FP 51.3%，DA-PTQ 48.9%，高於文中 QuantVLA 43.5%。
- Google Robot cross-domain：DA-PTQ 約 68.5% Visual Matching、51.7% Variant Aggregation；FP 分別 74.8%、61.3%。
- Paper 報告 42.5% memory reduction 與 54.8% inference speedup；ablation 顯示完整方法 48.9%，CSRC-only 43.8%，DA-MPA-only 39.6%。

## Limitations / Evidence Boundary

只有 CogACT / SIMPLER simulation；沒有 physical robot。Structural Jacobian 與 error propagation 是 approximation，對 highly nonlinear/contact-rich dynamics 未必可靠；固定 calibration distribution 也可能無法處理大 domain shift。PDF 仍保留 `Conference'17`、placeholder DOI/ISBN，因此只能標作 arXiv preprint，不能當成已接受 ACM final。

## Why It Matters

它把 action-aware PTQ 再推到 trajectory-level geometry，適合與 DyQ-VLA 比較：前者在 calibration 時選 static mixed precision，後者在 runtime 依 kinematics 動態切換。

## Reading Route

- 20 minutes：Figure 1、Algorithm 1、Tables 1–4。
- 60 minutes：推導 drift score、檢查 compensation folding、calibration set 與 cross-domain evaluation。

## Reading Questions（不附答案）

1. Structural Jacobian 是否來自 robot model、data，還是 policy gradient？
2. 512 trajectories 的 calibration cost 是否仍可稱 lightweight？
3. CSRC 的 gain 是否主要來自 distribution alignment 而非 drift modeling？
4. Reported speedup 的 denominator 與 hardware measurement 是什麼？
5. Fixed bit map 如何處理 deployment 中新的 task phases？

## Meeting Card

DA-PTQ 將 VLA PTQ 目標改成「少造成 trajectory drift」，但目前 evidence 仍限於一個 VLA family 的 simulation preprint。
