# 16 — Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots

- 定位：同時支援 quantized VLA / WAM 的 portable C++ runtime；屬 system companion，不是新 quantizer。
- Source：[arXiv 2607.02501](https://arxiv.org/abs/2607.02501) · [official repository](https://github.com/SEU-PAISys/Embodied.cpp)
- Local PDF：[paper-arxiv-v3.pdf](paper-arxiv-v3.pdf)

## 為何納入

Paper 不只是泛談 runtime，而是直接用 Pi-0.5、GR00T N1.7、HY-VLA 的 8/6/4-bit C++ configurations 比較 latency、VRAM 與 closed-loop success，因此符合 VLA + quantization 的 supporting evidence。

## Background / Problem

LLM runtimes 預設 request-response、token I/O 與 throughput optimization；robot deployment 則需要 batch-1 latency、low jitter、multi-rate modules、stateful closed loop、sensor/action adapters 與 heterogeneous hardware。

## Method 與 Key Innovation

- 五層 runtime：input adapters、sequence builders、backbone execution、head plugins、deployment adapters。
- 將 multi-rate execution、buffer reuse、operator fusion、backend dispatch 與 robot/simulator interface 視為同一 deployment contract。
- 對多個 VLA 提供 C++ full-precision 及 8/6/4-bit execution paths。

## Main Results

- Pi-0.5：normalized 4-bit latency 0.88、success 0.70、VRAM 0.30，相對各自 Python baseline 1.00。
- GR00T N1.7：4-bit latency 0.65、success 0.96、VRAM 0.38。
- HY-VLA：4-bit latency 0.37、success 0.99、VRAM 0.23。
- Paper 對整體 configurations 概括為約 1.05×–2.70× speedup 與 7%–77% lower VRAM。

## Limitations / Evidence Boundary

主要 VLA table 是 normalized values，缺少足夠 absolute latency、task protocol、quantizer/format 與 model-specific kernel details，不適合拿來與其他 papers 直接排名。Pi-0.5 4-bit success retention 只有 0.70，也顯示 portability 不代表所有 architecture 都能 aggressive quantization。

## Why It Matters

它提醒 VLA quantization 的最後一公里是 runtime contract：相同 weights 如果沒有 action-head plugin、multi-rate scheduling 與 batch-1 kernels，不能自動形成 robot-side deployment。

## Reading Route

- 20 minutes：architecture taxonomy、runtime layers、Table 3。
- 60 minutes：對照 repository，查清各 model 的 quantization format、absolute hardware results 與 benchmark scripts。

## Reading Questions（不附答案）

1. 8/6/4-bit 分別量化哪些 modules 與 tensors？
2. Normalized success 的 absolute numerator/denominator 是什麼？
3. Pi-0.5 為何在 4-bit 下只有 70% retention？
4. Multi-rate scheduling 如何處理 stale representations？
5. Runtime portability 與 peak performance 的 trade-off 是什麼？

## Meeting Card

Embodied.cpp 是「quantized model 如何進入 heterogeneous closed-loop runtime」的 system paper；quantization algorithm evidence 本身較薄。

