# 07 — QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization

- 定位：第一個以 action-space sensitivity 做 channel-wise bit allocation 的 VLA-specific PTQ。
- Canonical status：ICLR 2026 conference paper。
- Source：[arXiv identity](https://arxiv.org/abs/2602.03782) · [official code](https://github.com/AutoLab-SAI-SJTU/QVLA)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)（official final PDF 在下載時拒絕 automated access，因此 local copy 明確保留為 arXiv v1 fallback。）

## 為何納入

QVLA 的 research question 就是 VLA quantization；它把 mixed precision 的 optimization target 從 reconstruction error 改成 action deviation，並提供 simulation、memory、latency 與小規模 real-robot evidence。

## Background / Problem

Uniform-bit LLM quantization 假設 channels 的重要性近似一致，但 VLA 的少數 channels 可能直接支配 action coordinates。微小單步誤差又會透過 closed loop 累積，因此 passive feature fidelity 並非合適目標。

## Method 與 Key Innovation

- 對每個 output channel 候選 bit-width `{0,2,4,8,16}` 估計 action-space sensitivity。
- 以 first-order Taylor proxy 近似 channel quantization 對 action loss 的影響，再以 greedy demotion 滿足 bit budget。
- Weights 可 channel-wise mixed precision；activations 維持同一 bit-width 以配合 hardware。
- Projector 與 action head 在最終 system 中保留 BF16。

## Main Results

- OpenVLA W4A4：保留 99.3% baseline performance，memory 約為 28.2%，speedup 約 1.47×。
- Weight-only W4A16：OpenVLA 76.5%，與 FP 相同；4.3 GB vs 15.2 GB。OpenVLA-OFT 約 96.7% vs 97.1%，4.5 GB vs 15.4 GB。
- Real-world Pi-0 experiment：3 tasks、每 task 10 trials；W8A16 與 full precision 都報告 63.3%，speedup 約 1.28×。

## Limitations / Evidence Boundary

任意 per-channel precision 對 dense commodity kernels 未必友善；paper 以 uniform activation precision 減輕問題，但 weight layout 仍需 runtime support。Real-world 每 task 10 trials 太少，不足以證明完全等價。Local PDF 不是 canonical final，引用精確 wording 時應查 ICLR final。

## Why It Matters

QVLA 奠定 `action-centric quantization` 主線：重要性必須由 executable action 定義，而不是只看 hidden-state reconstruction。後續 DyQ-VLA、ActQuant、Mix-QVLA 都可視為不同 granularity / temporal extension。

## Reading Route

- 20 minutes：motivation、sensitivity proxy、main tables。
- 60 minutes：推導 Taylor score、greedy allocation、hardware mapping，並比較 W4A16 與 W4A4。

## Reading Questions（不附答案）

1. First-order proxy 在 long-horizon closed loop 下何時會失效？
2. Channel-wise mixed precision 的 metadata 與 kernel cost 是否計入？
3. 為何 projector/action head 保留 BF16？
4. W4A4 的「average 4-bit」與 uniform 4-bit 有何不同？
5. 如何用相同 calibration data 公平比較 QVLA 與 ActQuant？

## Meeting Card

QVLA 的核心貢獻是把 mixed-precision allocation 的 currency 從 reconstruction error 改成 action error。

