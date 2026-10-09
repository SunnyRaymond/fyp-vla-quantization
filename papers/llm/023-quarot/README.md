# QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

> **Reading-list role**: Companion - `SpinQuant` 的直接 predecessor；fixed randomized Hadamard rotations for end-to-end `W4A4KV4` inference  
> **Verification**: `verified-full-text` - NeurIPS 2024 official proceedings final  
> **Recommended effort**: 先完成 20-minute bridge route，再回到 `SpinQuant`

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman |
| Year / version | 2024; arXiv:2404.00456 v2, 2024-10-29 |
| Venue / status | NeurIPS 2024 Main Conference Track; peer-reviewed venue final |
| Primary source | [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html) · [official PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2404.00456) · DOI `10.52202/079017-3180` |
| Code | [official repository](https://github.com/spcl/QuaRot) |

**Local-version boundary.** `paper.pdf` 是 NeurIPS official proceedings final，共 28 PDF pages，包含 appendix 与 NeurIPS checklist。arXiv 当前 metadata 是 v2；本地文件按 venue final 标记，不按 arXiv artifact 标记。

## 2. One-sentence takeaway

QuaRot 用 function-preserving randomized Hadamard rotations 把 hidden-state、FFN、attention 与 KV-cache outliers 分散到更多 dimensions，使 frozen LLaMA-2 能采用 `W4A4KV4`，且不需要为 outlier channels 保留 higher precision；但 rotation 本身是 fixed/random，未按 quantized task loss 学习，这正是 `SpinQuant` 后续要解决的缺口。

## 3. Why SpinQuant cites QuaRot

| Axis | QuaRot | SpinQuant |
|---|---|---|
| Rotation choice | fixed randomized Hadamard matrices | 用 quantized loss 学习 mergeable `R1/R2` |
| Full-precision property | paired/fused orthogonal transforms 保持 model function | 同样利用 rotational invariance |
| Online path | internal FFN/attention/KV paths仍需 fast Hadamard kernels | `R3/R4` 保留为 online Hadamard；`R1/R2` 可 fuse |
| Calibration boundary | rotation construction本身不 training；默认 4-bit weight path仍使用 GPTQ calibration | 另用 800 WikiText-2 samples、backprop 与 Cayley SGD 学 rotation，随后可配 GPTQ |
| Main question | random Hadamard 能否消除 outliers？ | 哪个 orthogonal orientation 对 quantized loss 最好？ |

不要把两篇理解成“一个有 rotation，一个没有”。正确 lineage 是：

`QuaRot: function-preserving fixed/random Hadamard rotation`  
`→ SpinQuant: keep the placement idea, learn mergeable rotations under quantized loss`

### Direct comparison reported by SpinQuant

以下数字来自 `SpinQuant` Table 5 的 matched rows，不是把两篇各自不同实验拼在一起：

| Model / setting | QuaRot + GPTQ | SpinQuant + GPTQ | Locator |
|---|---:|---:|---|
| LLaMA-3 8B `W4A4KV4` | zero-shot average **63.3**, WikiText-2 PPL **8.0** | **65.5**, **7.3** | SpinQuant Table 5, PDF p. 9 |
| LLaMA-3 70B `W4A4KV4` | **65.1**, **20.2** | **69.3**, **5.5** | SpinQuant Table 5, PDF p. 9 |

这组 evidence 支持“learned rotation 在该 protocol 下优于 fixed/random QuaRot rotation”；它不证明 learned rotation 对 every model、seed、kernel 或 VLA task 都更好。

## 4. Background and prerequisites

- **Activation outliers and dynamic range**：per-token INT4 quantizer 的 scale 被少数 extreme coordinates 拉大时，大多数普通值会挤在少量 bins 中。
- **Orthogonal matrix**：$Q^\top Q=I$，保持 $L_2$ norm 与 inner product；paired $Q,Q^\top$ 能保持 full-precision computation。
- **Randomized Hadamard transform**：$\widetilde H=H\operatorname{diag}(s)$，其中 $s_i\in\{-1,+1\}$；structured transform 可用 fast Hadamard kernel 执行。
- **Incoherence processing**：通过 rotation 降低 matrix/activation 的 maximum coordinate 相对整体 norm 的突出程度，让 uniform quantization 更容易。
- **RTN vs GPTQ**：RTN 不需要 calibration；QuaRot 的默认 strong 4-bit result 使用 GPTQ，因此不能把所有 4-bit result 都称为 calibration-free。
- **Transformer paths**：RMSNorm、residual stream、FFN down-projection、attention value/output pair、RoPE、KV cache。
- **Systems terms**：prefill 通常更 compute-bound；decoding 的 KV-cache access 通常更 memory-bound。

## 5. Problem

- **Target setting**：weights、activations 与 KV cache 全部降到 4 bit，同时让 main matrix multiplications 使用 INT4。
- **Outlier bottleneck**：activation/KV outliers 使 4-bit uniform quantization 产生很大 error；previous methods 常需要 mixed precision、outlier channel isolation 或 calibration-derived scaling。
- **Systems bottleneck**：nominal 4-bit storage 若没有 matching INT4 GEMM、KV-cache path 和 fast transform kernels，不会自动转化为 latency/memory gain。
- **Design goal**：不用 retraining 改变 pretrained model function，先把 representation geometry 改得更容易 quantize，再接现有 weight quantizer。

## 6. Method

### 6.1 System view

`fuse RMSNorm scales → rotate residual stream with randomized Hadamard Q → absorb inverse/paired transforms into weights → add online Hadamard transforms inside FFN and attention → GPTQ or RTN weights → per-token activation quantization + group-wise KV quantization → INT4 kernels`

QuaRot 分两大 stages：Stage 1 在 full precision 中做 function-preserving weight/graph modifications；Stage 2 才执行 weight、activation 与 KV-cache quantization。

### 6.2 Global residual rotation

RMSNorm 的 learned scale 先吸收到相邻 weights。QuaRot 再把 residual-stream activation 写成 $XQ$，并在 block input/output weights 中配对吸收 $Q^\top$ 与 $Q$。关键 commutation relation 是：

$$
\operatorname{RMSNorm}(X)=\operatorname{RMSNorm}(XQ^\top)Q.
$$

因为 orthogonal transform 保持 norm，full-precision output 不变；但 element-wise distribution 已改变，因此 quantization error 可以改变。

### 6.3 Internal FFN and attention rotations

- **FFN**：在 down-projection 前在线执行 Hadamard transform，并把 inverse effect fuse 到 modified $W_{down}$。
- **Value/output pair**：利用每个 attention head 中 $W_v$ 与 $W_{out}$ 的 paired structure，把 head-wise transforms 部分 fuse 到 weights。
- **Key/query path**：RoPE 阻止简单地把全部 rotation 吸收到 $W_q/W_k$；QuaRot 在 Post-RoPE query/key 上执行 matching head-wise Hadamard transforms，使 attention scores 保持不变，并让 cached keys 更易 quantize。
- **KV cache**：rotated keys/values 用 low-bit format 保存；读取后 dequantize，再以 FP16 进入 attention computation。

因此，“end-to-end 4-bit”指 weights/linear inputs/KV cache 与 main linear matrix multiplications 的 4-bit path，不表示 RMSNorm、nonlinearity、softmax、scaling 和所有 intermediate arithmetic 都是 INT4。

### 6.4 Quantization recipe

- Activations：symmetric per-token quantization，constant clipping ratio **0.9**。
- KV cache：asymmetric group-wise quantization，group size **128**，clipping ratio **0.95**。
- Weights：默认 GPTQ，per-column symmetric quantization；weight clipping ratio 通过 squared-error linear search 选择。
- GPTQ calibration：**128 WikiText-2 samples × sequence length 2048**。
- LLaMA-2 70B preparation：单张 NVIDIA A100 上 QuaRot model modification 约 **5 min**，GPTQ 另约 **2 h**。

### 6.5 What is actually new

核心贡献不是发明 Hadamard transform 或 GPTQ，而是把 computational invariance 系统性扩展到 residual、FFN、attention 与 KV-cache paths，使 weights、activations 和 cache 可以共同进入 4-bit deployment path，而不单独保留 higher-precision outlier channels。

## 7. Experiments and evidence

### Authors' claims

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| LLaMA-2 70B `A4W4KV4` quality | WikiText-2 PPL **3.32 → 3.79**；zero-shot average **77.07 → 75.98** | Tables 1–2, PDF p. 8 | “retains 99%”来自 relative score retention；不是每项 task 都只降 1%。 |
| LLaMA-2 7B `A4W4KV4` quality | PPL **5.47 → 6.10**；zero-shot average **69.82 → 65.64** | Tables 1–2, PDF p. 8 | Smaller model 的 drop 更明显，不能只引用 70B headline。 |
| System benefit | 70B prefill up to **3.33×**；decoding peak-memory saving **3.89×** | Figure 4, PDF p. 8 | RTX 3090、single Transformer block；prefill length 2048，memory test batch 16 / 50 decoded tokens。不是 full-model end-to-end benchmark。 |
| Calibration-free high precision | 70B RTN INT8 PPL **3.33** vs FP16 **3.32**；zero-shot **77.17** vs **77.07** | Table 3, PDF p. 9 | “without calibration”适用于 RTN 6/8-bit path；default strong 4-bit GPTQ path用了 calibration。 |
| Hadamard is better than generic random orthogonal | 7B PPL **7.45 → 6.10**；70B **4.07 → 3.79** | Table 8, PDF p. 15 | 只表明 tested Hadamard construction 更好；不证明 rotation seed 不重要。 |

### Synthesis

QuaRot 的 strongest lesson 是：**full-precision equivalence does not imply quantized equivalence**。Table 8 已显示 orthogonal choice 会改变 PPL；`SpinQuant` 随后把这个 observation 提升成 optimization problem，并进一步展示 random Hadamard seeds 的 downstream variance。

## 8. Limitations

### Authors' stated limitations

- Paper 没有独立 `Limitations` section；NeurIPS checklist 把 Conclusion 中的 future work 当作 limitations discussion。
- Conclusion 提出 quantizing residuals 与 extending to Mixture-of-Experts architectures 仍是 future work。

### My critique

- **Evidence scope**：main evidence 以 LLaMA-2、WikiText-2 和 six zero-shot language tasks 为主；没有 VLM/VLA、multimodal tokens、continuous actions 或 closed-loop success。
- **Calibration wording**：rotation不需要 learned calibration，不等于整个 default 4-bit pipeline calibration-free；GPTQ 明确使用 128 WikiText-2 samples。
- **Systems validity**：headline speedup/memory 来自 RTX 3090 上的 single Transformer block 和 custom CUTLASS/FlashInfer path；full-model serving stack、batching、launch overhead 和 data movement 可能改变结果。
- **Precision wording**：queries、RMSNorm、softmax、dequantization 和部分 transforms 仍用 FP16/FP32；“all matrix multiplications in 4 bits”不应改写为“entire model runs only in INT4”。
- **Number inconsistency**：Abstract、Figure 4 与正文给出 70B peak-memory saving **3.89×**，Conclusion 写 **3.39×**。阅读/汇报时优先引用 Figure 4 的 plotted value 并标记 Conclusion typo。
- **Rotation robustness**：fixed/random rotation 没有直接优化 task-aware quantization error；这会留下 model/seed/precision sensitivity，也是 `SpinQuant` 的直接 motivation。

## 9. Why it matters for this project

- **VLA backbone**：同样的 representation transform 可考虑用于 language/VLM backbone，但 vision tokens、proprioception 与 action expert 的 outlier geometry 未必服从 text-only LLaMA observations。
- **Action calibration**：若接 GPTQ 或 learned rotations，calibration 应包含 robot trajectories、rare actions 和 failure recovery，而不是只用 WikiText-2。
- **Closed-loop evaluation**：需要同时报告 task success、tail action error、control frequency、P99 latency、peak memory 与 power；PPL/zero-shot accuracy 不足以确认 robot deployment quality。
- **Online transform cost**：FFN/KV online Hadamard path 会占用 latency budget；在 high-frequency control 中必须测实际 backend，而不是用 nominal bit-width 推断。

## 10. How to read it

### 20-minute bridge route for your current SpinQuant reading

1. **0–4 min**：看 Abstract + Figure 1，回答 outlier 对 INT4 scale 有什么影响。
2. **4–8 min**：读 Section 3.4 与 Eq. (3)，说明 rotation 为什么保持 full-precision function。
3. **8–13 min**：看 Figure 3（PDF p. 5），沿 signal path 标出 `FP16 → INT4 → FP16`，不要把 whole graph 误读为 pure INT4。
4. **13–17 min**：核对 Tables 1–2 + Figure 4（PDF p. 8），分开 quality、prefill speed 与 decoding memory。
5. **17–20 min**：看 Table 8（PDF p. 15），然后回到 `SpinQuant` Figure 4 / Tables 2 and 5，写出 fixed/random → learned 的 research gap。

### 60-minute deep route

1. **0–10 min**：复习 orthogonal matrix、Hadamard transform、incoherence 与 per-token quantization。
2. **10–25 min**：逐步读 Section 4 Stage 1a–1d；画 residual、FFN、value/output、query/key 四条 rotation paths。
3. **25–35 min**：读 Stage 2a–2c；区分 rotation、GPTQ、online quantization 与 attention dequantization。
4. **35–45 min**：核对 setup、Tables 1–4 与 Figure 4；记录 model、precision、hardware、batch/sequence conditions。
5. **45–52 min**：读 Appendix Tables 6–8；关注 key/value sensitivity 与 Hadamard-vs-random ablation。
6. **52–60 min**：对照 `SpinQuant` Section 3 与 Table 5，完成下面的问题，但先不看任何现成答案。

## 11. Reading questions - answer them yourself

1. 为什么 $Q,Q^\top$ 保持 full-precision output，却不能保证 quantized output 相同？
2. QuaRot 中哪些 transforms 能 fuse 到 weights，哪些必须 online 执行？阻止完整 fuse 的 operator 是什么？
3. “QuaRot does not need calibration”在哪个 precision/quantizer setting 下成立？默认 4-bit result 又用了什么 calibration？
4. Table 2 的 “retains 99%” 应按 absolute percentage points 还是 relative retention 解读？请用 70B average 自己计算。
5. Table 8 说明 Hadamard 比 random orthogonal matrix 好；它没有证明什么？
6. Figure 4 的 single-block speedup 为什么不能直接当作 full-model end-to-end latency？
7. `SpinQuant` 具体学习哪些 rotations，哪些仍固定为 Hadamard？这个选择怎样改变 inference overhead？
8. 如果把 QuaRot/SpinQuant 用到 VLA，你会用什么 calibration data、loss 与 closed-loop metrics？

建议把回答直接追加在每题下面，并各写一个 `Evidence: Section/Figure/Table` locator。

## 12. Weekly meeting card

- **Problem**：activation/KV outliers 阻碍 end-to-end 4-bit LLM inference。
- **Key idea**：用 function-preserving randomized Hadamard rotations 分散 outliers，再量化 weights、activations 和 KV cache。
- **Best evidence**：LLaMA-2 70B PPL 3.32→3.79、zero-shot 77.07→75.98；single-block prefill up to 3.33×，peak-memory saving 3.89×。
- **Biggest limitation**：fixed/random rotations不优化 quantized task loss；system result 是 paper-specific single-block kernel benchmark。
- **Question for the group**：对 VLA，应该学习 language-aware、action-aware，还是 risk-sensitive multi-objective rotations？

## 13. Status & evidence boundary

- **Status**：NeurIPS 2024 peer-reviewed venue final；arXiv v2、NeurIPS proceedings 与 official code 属于同一 work cluster。
- **Source claim**：Sections 3–5、Figures 1/3/4、Tables 1–4、Appendix Tables 6–8，以及 paper 中的 setup/runtime numbers。
- **Direct comparison**：QuaRot vs SpinQuant 的 LLaMA-3 rows来自 `SpinQuant` Table 5，已单独标出 source boundary。
- **My synthesis**：QuaRot → SpinQuant lineage、VLA calibration proposal、closed-loop/system metrics 与 wording caveats。
- **Open question**：multimodal/action distributions、modern kernels、full-model serving 与 robot control loops 下能否保留同样 trade-off。

