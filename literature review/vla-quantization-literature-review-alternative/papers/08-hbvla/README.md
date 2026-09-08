# 08 — HB-VLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models

- 定位：VLA-tailored 1-bit weight-only PTQ。
- Source：[arXiv 2602.13710](https://arxiv.org/abs/2602.13710)
- Local PDF：[paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 為何納入

HB-VLA 直接研究多種 VLA 在 1-bit weight quantization 下的 action preservation，涵蓋 LIBERO、SIMPLER 與 Mobile ALOHA real robot。

## Background / Problem

Binary PTQ 的 local reconstruction error 很大；對 VLA，error 可經 action head 與 robot kinematics 放大。Generic binary LLM methods 無法辨識真正影響 trajectory 的 weights。

## Method 與 Key Innovation

- `Action-Aware Rectified Hessian`：由 drift-weighted action loss 和 kinematic Jacobian 建立 action-critical weight importance。
- 將 weights 分成 salient / non-salient groups。
- 使用 sparse orthogonal permutation、Haar transform 與 group-wise binary quantization，降低 extreme outliers。
- Activations 保留 BF16；方法本質是 1-bit weight-only PTQ。

## Main Results

- Pi-0.5 LIBERO：FP 97.1%；只量化 LM + ViT 約 92.7%，全模型約 87.9%。
- OpenVLA-OFT：LM + ViT 約 90.3%，全模型約 83.5%。CogACT SIMPLER Visual Matching：FP 74.8%，全模型約 67.2%。
- 含 metadata 的 footprint：Pi-0.5 約 4.6→0.83 GB；OpenVLA-OFT 15.2→2.74 GB；CogACT 30.5→5.50 GB。Paper 報告最高約 2.93× speedup。
- Mobile ALOHA real tasks 仍出現約 12.5–23.4 percentage-point drops，但優於其他 binary PTQ baselines。

## Limitations / Evidence Boundary

「1-bit 可部署」不能解讀為 near-lossless：whole-model binary setting 在多個 closed-loop tasks 有明顯 absolute degradation，action head 尤其敏感。Activations 未低 bit，compute 與 activation memory saving 有上限。

## Why It Matters

它測試 PTQ 的極限，並把 kinematic amplification 納入 Hessian importance；同時用 real-robot drop 誠實顯示 binary weight compression 的 ceiling。

## Reading Route

- 20 minutes：action-aware Hessian、footprint table、real-robot table。
- 60 minutes：細讀 salient split、transform pipeline、metadata accounting 與 whole-model / partial-model差異。

## Reading Questions（不附答案）

1. Kinematic Jacobian 對不同 robot embodiment 是否可轉移？
2. 1-bit weights 的 scales 與 indices 佔多少 footprint？
3. 為何 action head binary quantization 造成較大 drop？
4. BF16 activations 如何限制真正的 edge speedup？
5. Real-robot absolute drop 是否值得 memory gain？

## Meeting Card

HB-VLA 把 VLA weight PTQ 推到 binary，但結果也清楚顯示 action head 全量化仍不是 near-lossless。
