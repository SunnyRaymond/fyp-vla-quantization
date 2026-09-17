# 10 — LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics

- 定位：Q4_K_M GGUF + llama.cpp + ROS 2 的 edge deployment feasibility study。
- Source：[arXiv 2603.03380](https://arxiv.org/abs/2603.03380)
- Local PDF：[paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)

## 為何納入

Paper 直接 fine-tune compact VLA 做 image-to-action generation，再把 model 做 4-bit GGUF PTQ，於 Jetson Orin-class hardware 量測完整 image-to-action latency。

## Background / Problem

Compact VLM 能在 edge device 執行，但 VLM 不等同能產生 deterministic motor commands 的 VLA。作者把 256M SmolVLM backbone fine-tune 成 structured velocity-command policy，並接入 ROS 2。

## Method 與 Key Innovation

- FP32 LoRA rank 8 fine-tuning，之後轉成 GGUF `Q4_K_M`。
- llama.cpp CUDA backend，42 transformer layers 全部 GPU offload。
- Context window 512、最多 12 output tokens；ROS 2 node 發布 `geometry_msgs/Twist`。
- Low-level controller 維持 100 Hz heartbeat，VLA reasoning loop 約 6.6 Hz。

## Main Results

- 300 timing runs：150.5 ms mean latency、約 6.64 Hz、standard deviation 約 0.13 ms。
- Simulated closed-loop section 使用 30 runs，測量從 image ingestion 到 action output，而非單一 forward pass。

## Limitations / Evidence Boundary

沒有 standard robot benchmark 的 task success，也沒有 matched FP32 / FP16 / alternative quantizer accuracy comparison，因此只能證明 timing feasibility。Paper 在 Jetson AGX Orin 64 GB 與 Jetson Orin NX 的命名上不一致；6.6 Hz 也不等於已證明 real-world closed-loop stability。文中約 220% improvement 屬跨 paper comparison，hardware/protocol 不匹配。

## Why It Matters

它提供一條很容易理解的 practical stack：compact backbone → LoRA → GGUF → llama.cpp → ROS 2。但它也示範為何 deployment paper 必須把 latency evidence 與 task-performance evidence 分開。

## Reading Route

- 20 minutes：architecture、implementation、latency table、threats to validity。
- 60 minutes：核對 action parser、安全 override、warm-up、hardware power mode 與 benchmark 缺口。

## Reading Questions（不附答案）

1. Q4_K_M 量化哪些 layers，vision path 是否同樣量化？
2. 150.5 ms 是否包含 camera capture 與 ROS transport？
3. 0.13 ms jitter 是否在真實 concurrent workload 下仍成立？
4. VLA 6.6 Hz 與 controller 100 Hz 如何交換 action？
5. 沒有 task success 時，什麼程度可稱 closed-loop control？

## Meeting Card

LiteVLA-Edge 是 edge timing baseline，不是 accuracy-preserving quantization method；最重要的缺口是沒有 matched task success。

