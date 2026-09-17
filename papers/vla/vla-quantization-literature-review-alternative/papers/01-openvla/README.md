# 01 — OpenVLA: An Open-Source Vision-Language-Action Model

- 定位：VLA foundation paper；本 corpus 用它建立 quantization baseline，而不是把它當作專門 quantizer。
- Source：[CoRL final / PMLR](https://proceedings.mlr.press/v270/kim25c.html)
- Local PDF：[paper-corl-final.pdf](paper-corl-final.pdf)

## 為何納入

Paper 本身不只介紹 7B OpenVLA，也直接比較 BF16、INT8 與 INT4 deployment，並以 closed-loop robot rollouts 檢驗 quantized policy，因此同時滿足 VLA 與 model quantization。

## Background / Problem

OpenVLA 把 pretrained VLM fine-tune 成 action-token policy。Deployment 的直接障礙是約 7B parameters 的 memory 與 control frequency；但量化不能只看 language perplexity，因為小 action error 會改變下一步 observation。

## Method 與 Key Innovation

Quantization 不是主方法。作者對 finetuned OpenVLA 做 weight-only low-bit conversion，並在 8 個 Bridge V2 tasks、共 80 rollouts 上比較 task success，同時量測 GPU memory 與 inference throughput。這個 ablation 的價值是提供後續 VLA-aware quantization papers 反覆採用的 failure case。

## Main Results

- BF16：71.3 ± 4.8% success、16.8 GB。
- INT8：58.1 ± 5.1%、10.2 GB；在多數 GPU 上還可能因 quantization overhead 而更慢。
- INT4：71.9 ± 4.7%、7.0 GB；在該設定中恢復 BF16-level success，並因較低 memory traffic 提升 throughput。

## Limitations / Evidence Boundary

這只是一個小規模 deployment ablation，不是新的 PTQ algorithm；8 tasks 與 80 rollouts 不足以證明 INT4 普遍優於 INT8。OpenVLA 本身仍受 autoregressive action decoding、low control frequency 與 single-image input 限制。

## Why It Matters

它先證明兩件事：bit-width 與 closed-loop quality 不是單調關係；memory saving 也不自動等於 latency saving。後續 paper 幾乎都在回答「如何讓這個 INT4 success 不靠偶然，並跨 architecture 成立」。

## Reading Route

- 20 minutes：Architecture、quantization ablation、limitations。
- 60 minutes：再讀 training data mixture、action tokenization、Bridge V2 protocol，記下 latency 是 per-token、per-action 還是 end-to-end。

## Reading Questions（不附答案）

1. 為何 INT8 可能同時比 BF16 與 INT4 差？
2. 80 rollouts 能否分辨 71.3% 與 71.9%？
3. Weight-only quantization 對 activation memory 與 KV-cache 有何影響？
4. Action token error 如何透過 closed loop 累積？
5. 若改成 action chunking，quantization failure mode 會如何改變？

## Meeting Card

OpenVLA 是「generic quantization 在 VLA 上會出現 non-monotonic accuracy/runtime」的 baseline evidence，不是 VLA-specific quantizer。
