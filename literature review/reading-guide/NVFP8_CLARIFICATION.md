# NVFP8 Clarification: Do Not Invent a Paper or Datatype

> **Status**: `UNRESOLVED LABEL`  
> **Snapshot**: 2026-08-22  
> **Core-count rule**: 这不是一篇已核验 paper，也没有找到独立 bit-level specification；**不计入 core paper count**。

## Bottom line

在已核验的 official NVIDIA materials 中，`NVFP8` 被用于 ComfyUI/video-generation optimization 或 quantized checkpoint 的 product label；没有 source 完整定义独立的 E/M layout、bias、block size、scale hierarchy、special values 与 arithmetic rules。因此不能把它与 OFP8、MXFP8、NVFP4 并列成一个已定义 datatype，也不能创建虚构的 “NVFP8 paper README”。

## What is verified

| Evidence | What it supports | What it does **not** support |
|---|---|---|
| [NVIDIA GeForce CES 2026 article](https://www.nvidia.com/en-us/geforce/news/gfecnt/20261/rtx-ai-garage-ces-2026-open-models-video-generation/) | official pages use “NVFP8” for named ComfyUI/checkpoint workflows；reports **2× performance** and **40% VRAM reduction** for that workflow | does not define a distinct numeric encoding or prove a universal 2×/40% property |
| [NVIDIA Technical Blog: Open Source AI Tool Upgrades](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs) | describes NVIDIA-optimized FP8 fused quantization/dequantization and uses NVFP8 wording in workflow claims | does not provide bit layout, scale granularity or special-value table for “NVFP8” |
| [NVIDIA Llama-3.3-70B-Instruct-FP8 model card](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8) | this **specific checkpoint** uses E4M3 PTQ for linear weights/activations；reports ~**50%** disk/GPU-memory reduction vs 16-bit and listed task scores | does not define every item marketed as NVFP8；checkpoint name itself is FP8 |

Checkpoint-specific numbers from the model card:

| Metric | BF16 | FP8 |
|---|---:|---:|
| MMLU | 83.3 | 83.2 |
| GSM8K-CoT | 95.3 | 94.3 |
| ARC-Challenge | 93.7 | 93.2 |
| IFEval | 92.1 | 92.2 |

These rows only support that checkpoint and evaluation protocol.

## What remains unknown

| Field | Verified answer for generic “NVFP8” |
|---|---|
| E/M bits | not specified; do not generalize E4M3 from one checkpoint |
| Exponent bias / finite range | not specified under this label |
| Scale granularity | not specified |
| Block size | not specified |
| Scale encoding | not specified |
| NaN / Inf behavior | not specified |
| Distinct CUDA / Transformer Engine datatype | not found in the verified packet |
| Training recipe | not found; observed official use is inference/checkpoint optimization |

## Safe terminology

- Safe: “NVIDIA-marketed NVFP8 checkpoint/optimization label.”
- Safe when a model card says so: “This checkpoint uses FP8 E4M3 PTQ.”
- Unsafe: “NVFP8 is an E4M3 format with one FP32 scale per tensor.”
- Unsafe: “NVFP8 is a new NVIDIA 8-bit datatype standardized alongside NVFP4.”
- Unsafe: “The NVFP8 paper shows …”

## Clarification question for Professor Li

> “NVFP8” 是否指 NVIDIA-optimized standard FP8/E4M3 checkpoints，还是老师另有一份尚未附上的 NVIDIA internal/product format document？

## Action after clarification

1. 如果答案是 standard FP8/E4M3：把该项合并到 [FP8 Formats for Deep Learning](papers/07-fp8-formats/README.md)，并在具体 checkpoint 层记录 scaling recipe。
2. 如果答案是某个 product checkpoint：只按该 model card 建 deployment note，不把 checkpoint behavior 泛化成 datatype。
3. 如果老师提供新的 specification/technical report：重新核验 version/date、bit layout、scaling 与 hardware scope，再决定是否建独立 note。

## Primary links

- [NVIDIA GeForce CES 2026 article](https://www.nvidia.com/en-us/geforce/news/gfecnt/20261/rtx-ai-garage-ces-2026-open-models-video-generation/)
- [NVIDIA Technical Blog: Open Source AI Tool Upgrades](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs)
- [NVIDIA Llama-3.3-70B-Instruct-FP8 model card](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8)

