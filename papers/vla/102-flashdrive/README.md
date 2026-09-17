# 18 — FlashDrive: Flash Vision-Language-Action Inference for Autonomous Driving

- 定位：以 W4A8 作為其中一環的 full-pipeline VLA acceleration；屬 autonomous-driving system companion。
- Source：[arXiv 2608.12932](https://arxiv.org/abs/2608.12932) · [official repository](https://github.com/z-lab/flashdrive)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

FlashDrive 直接部署 Alpamayo 1.5-10B VLA 並以 W4A8 quantization 減少 memory / latency，也提供 open-loop driving metrics、AlpaSim closed-loop evaluation 與多 GPU measurement。

## Background / Problem

Driving VLA 的 latency 不是單一 transformer bottleneck，而是 encode、prefill、autoregressive reasoning decode 與 flow-matching action 四段不同 redundancy。只做 quantization 不足以達到 real-time replanning。

## Method 與 Key Innovation

- Streaming inference：重用 overlap video frames 與 KV-cache；只更新新 frame。
- Speculative reasoning：用 diffusion drafter 平行提出 reasoning token blocks。
- Adaptive-step flow matching：重用 denoising middle steps 的 velocity。
- CUDA Graph + kernel fusion。
- 以 ParoQuant / Marlin 將 VLM backbone 做 W4A8；continuous action expert 保留 BF16。

## Main Results

- RTX PRO 6000：total latency 716.9→151.4 ms，約 4.7×；control rate 1.4→6.6 Hz。
- 必須拆解：所有非量化 optimizations 已把 latency 降到 176.0 ms；W4A8 再降到 151.4 ms，quantization 的 isolated incremental saving 是 24.6 ms，而不是完整 4.7×。
- Memory 約 31.6→18.3 GB（6 trajectory samples context）。
- Open-loop：minADE6@6.4s 0.767→0.844 m，minADE1 1.705→1.573 m。
- AlpaSim 100 clips：collision 0.19→0.15、off-road 0.41→0.32，但 wrong-lane 0.45→0.51；per-step rollout 1150→463 ms。

## Limitations / Evidence Boundary

只在 Alpamayo family；streaming fine-tuning、speculative decoding、flow-step caching 與 quantization 一起改動，closed-loop gain 不能歸因於 W4A8。6.6 Hz 對 autonomous driving 是否足夠是 system requirement 問題，不能由單一 paper 自行定義。Action expert 未量化。

## Why It Matters

它是最好的 attribution lesson：end-to-end speedup 必須按 stage 與 component 拆開。對 FYP，這比只報一個 headline 4.7× 更有價值。

## Reading Route

- 20 minutes：Figure 1、Table 1–3、quantization section。
- 60 minutes：逐段核對 encode/prefill/decode/action ablation、memory context、open/closed-loop metrics。

## Reading Questions（不附答案）

1. 為何 W4A8 比 W4A16 更適合 vision-heavy prefill？
2. Action expert 保留 BF16 佔多少 memory / latency？
3. Quantization 對 minADE 的 isolated effect 是否有 ablation？
4. Closed-loop safety improvement 是否超過 100 clips 的 uncertainty？
5. Streaming KV-cache shift 如何與 quantization error interaction？

## Meeting Card

FlashDrive 的 4.7× 是 full-stack co-design，不是 quantization-only；W4A8 的 isolated step 是 176.0→151.4 ms。

