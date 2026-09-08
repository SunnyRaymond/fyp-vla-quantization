# 03 — BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation

- 定位：從 training stage 原生設計的 1.58-bit VLA，不是 post-hoc PTQ。
- Source：[arXiv 2506.07530](https://arxiv.org/abs/2506.07530)
- Local PDF：[paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 為何納入

BitVLA 同時重構 VLA language backbone 與 vision encoder 的 low-bit training，並在 LIBERO、real robot、memory 與 latency 上直接評估。

## Background / Problem

把 pretrained OpenVLA 直接壓到極低 bit-width 容易崩潰。BitVLA 改問：若 VLA 從 multimodal training 開始就以 ternary weights 適應，能否取得遠高於 PTQ 的 compression ceiling？

## Method 與 Key Innovation

- LLM backbone 使用 ternary 1.58-bit weights 與 INT8 activations。
- Vision encoder 使用 `Quantize-then-Distill`，同樣形成 W1.58A8 path。
- Connector 與 action head 保留 floating point。
- 三階段流程：multimodal training、vision quantization/distillation、約 1M Open X-Embodiment samples 的 robotics pretraining。

## Main Results

- LIBERO：BitVLA 約 1.4 GB、96.0% average；OpenVLA-OFT full precision 約 15.4 GB、97.1%。
- OpenVLA-OFT INT4 PTQ 在同表為約 4.7 GB、96.9%，說明 BitVLA 的主要優勢是 footprint，而非明顯更高 success。
- A100、100 queries、action chunk 25：BitVLA 73 ms / 341.1 Hz；OpenVLA-OFT+ 321 ms / 77.9 Hz，約 4.4× latency gain。
- Vision encoder memory 約 0.8→0.1 GB；VQA average 約 53.0→51.5。

## Limitations / Evidence Boundary

不是現有 checkpoint 的 drop-in conversion。Paper 報告 VLM training 約 8×H800、7 days，robotics pretraining 約 16×H800、14 days；因此它把 inference cost 轉移成相當大的 training cost。部分 real-world 結果的 task/rollout 規模仍有限。

## Why It Matters

BitVLA 定義了另一條研究線：不是在 PTQ 中尋找更好的 scales，而是重新共同訓練 low-bit vision、language 與 action interface。比較 PTQ 時必須把 training budget 算進去。

## Reading Route

- 20 minutes：Figure 1、three-stage training、LIBERO table。
- 60 minutes：細讀 Quantize-then-Distill、data schedule 與 latency definition，區分 model size、VRAM 與 control Hz。

## Reading Questions（不附答案）

1. 為何 connector 與 action head 必須保留 floating point？
2. 1.58-bit 的 compression ratio 是否包含 scales、packing 與 runtime workspace？
3. Native low-bit training 與 4-bit PTQ 的公平比較應固定哪些 costs？
4. Vision encoder VQA drop 是否能預測 robot success drop？
5. 73 ms 是完整 observation-to-action latency 還是 action-chunk generation latency？

## Meeting Card

BitVLA 展示 native ternary VLA 的高 compression ceiling，但代價是重新 training，而不是便宜的 PTQ。
