# MEM: Multi-Scale Embodied Memory for Vision Language Action Models

> **Reading-list role**: PI-series companion — long/short-term memory for `π₀.₆`  
> **Verification**: `verified-full-text` — Physical Intelligence official project PDF + arXiv metadata  
> **Recommended effort**: Deep read after `π₀.₅` and `π*₀.₆`

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Marcel Torne, Karl Pertsch, Homer Walke, Kyle Vedder, Suraj Nair, Brian Ichter, Allen Z. Ren, Haohuan Wang, Jiaming Tang, Kyle Stachowicz, Karan Dhabalia, Michael Equi, Quan Vuong, Jost Tobias Springenberg, Sergey Levine, Chelsea Finn, Danny Driess |
| Year / version | 2026; arXiv:2603.03596 v1 on 2026-03-04, current v2 on 2026-03-08 |
| Venue / status | Physical Intelligence technical report / arXiv preprint; no peer-reviewed venue verified as of 2026-08-25 |
| Primary source | [Physical Intelligence project page](https://www.pi.website/research/memory) · [official PDF](https://www.pi.website/download/Mem.pdf) · [arXiv](https://arxiv.org/abs/2603.03596) |
| Code / weights | No official code, checkpoint, or dataset release link was verified on the project/arXiv pages at this snapshot |

**Local-version boundary.** `paper.pdf` 是 2026-08-25 获取的 Physical Intelligence official project PDF，共 15 pages。arXiv 当前文件是 v2；两份 PDF 内容属于同一 work cluster，但 binary file 不相同，因此本地文件不能标成 “arXiv artifact”。

**Series identity.** `π₀.₆-MEM` 不是 `π*₀.₆` 的别名。两者都从 base `π₀.₆` 出发，但改变的是不同 axis：`π*₀.₆` 用 RECAP / advantage-conditioned RL post-training 从 experience 改进 policy；MEM 添加 multi-scale context architecture，让 policy 能记住过去。MEM 不应与 `π*₀.₆` 合并计数。

## 2. One-sentence takeaway

MEM 将 dense short-term video/proprioceptive history 与 compressed long-term text summary 分工，再通过 high-level memory/subtask policy 和 low-level action policy 接入 `π₀.₆`，使 VLA 能处理最多约 15 分钟的 real-world tasks、partial observability 和 failure-conditioned strategy adaptation，同时保持 real-time inference path。

## 3. Background and prerequisites

- 先区分 `Markov policy` 与 `partially observable control`：当前 frame 不能总是决定正确 action。
- 复习 `π₀.₅` 的 hierarchical inference：low-frequency high-level subtask + high-frequency action generation。
- 复习 ViT spatial attention、causal attention、action chunking、FAST tokenization 与 flow-matching action expert。
- 区分三种 memory：raw observation history、proprioceptive history、semantic/text summary。
- 最重要的 terminology boundary：`in-context adaptation` 是根据 context 改变行为，不是 online gradient update；`long-term` 在本文主要表示 current episode 内的 minutes，不是跨 episode 的 persistent memory。

## 4. Problem

- **Target setting**：long-horizon real-world manipulation，其中 robot 必须记住已完成的 steps、隐藏 object 的位置、计数/计时信息，以及刚刚失败的 manipulation strategy。
- **Single-frame bottleneck**：self-occlusion、partial observability 和 repeated failures 都无法由当前 observation 单独消除。
- **Dense-history bottleneck**：把十几分钟的 multi-camera images 全部送进 Transformer，会让 token count、attention cost 和 inference latency 失控。
- **One-modality bottleneck**：proprioception 不能完整表示 environment state；text 丢失 fine spatial/dynamic detail；aggressive visual pooling 又会丢掉 precise event history。

## 5. Method

### 5.1 System view

`goal g + current observation + previous text memory m_t`  
`→ high-level policy → next subtask l_{t+1} + compressed memory update m_{t+1}`  
`→ short recent video/proprio history + l_{t+1}`  
`→ video encoder + low-level VLA/action expert → continuous action chunk`  
`→ environment → repeat`

核心不是一个无限增长的 memory bank，而是让不同 timescale 使用不同 representation：recent detail 用 video，distant semantic state 用 text。

### 5.2 Hierarchical factorization

Section III-A 将 joint prediction 近似分解为：

$$
\pi(a_{t:t+H},l_{t+1},m_{t+1}\mid o_{t-T:t},m_t,g)
\approx
\pi^{LL}(a_{t:t+H}\mid o_{t-K:t},l_{t+1},g)
\pi^{HL}(l_{t+1},m_{t+1}\mid o_t,m_t,g),
$$

其中 $K\ll T$。`πᴸᴸ` 只看较短的 dense history 并产生 action chunk；`πᴴᴸ` 通过 text memory $m_t$ 保留更长时间的 semantic events，同时输出 next subtask。

这个 factorization 隐含一个 strong assumption：远期历史对 low-level motor action 的影响，大部分可以通过 `updated memory + subtask` 这个 interface 传递。若 text summary 丢掉后续需要的 fine detail，low-level policy 无法恢复它。

### 5.3 Long-term text memory

- $m_t$ 是 previous semantic events 的 compressed natural-language summary，而不是所有历史 subtasks 的原样拼接。
- 训练 label 由 off-the-shelf LLM 根据 episode 的 subtask annotations 与 success/failure indicators 生成；LLM 被要求只保留 future execution 仍需要的信息，并合并/删除冗余内容。
- inference 时不是外部 LLM 在写 memory；trained high-level policy 自己预测 $m_{t+1}$。
- Context compression 还缓解 train-inference shift：human demonstrations 通常一次成功，而 deployed policy 可能多次重复失败；naive subtask concatenation 会累积大量重复指令，MEM 可以在任务真正完成前不更新 summary。

### 5.4 Short-term video and proprioceptive memory

- 每个 frame 先独立 patchify；standard spatial attention 保留。
- 每 4th ViT layer 加入 causal temporal attention，只在同一 spatial patch position 跨 time 连接。
- Space-time attention 从 naive $O(n^2K^2)$ factorize 为：

$$
O(Kn^2+nK^2),
$$

其中 $n$ 是 spatial patches，$K$ 是 frames。
- Upper ViT layers 丢弃 past-frame tokens，只把 current-timestep representation 送入 VLA backbone。因此送给 backbone 的 visual token count 与 single-frame path 接近。
- Video capability 主要通过改变 attention pattern 和 fixed sinusoidal temporal position encoding 获得；paper 强调不比原 single-image ViT 增加 learnable parameters，并令 $K=1$ 时 initialization 与原 VLM 一致。
- Past proprioceptive states 不再逐项转成 text，而是各自经 linear projection 形成 $K$ 个 continuous state tokens。

### 5.5 `π₀.₆-MEM` instantiation and training

- Backbone 初始化自 pretrained Gemma 3 4B VLM；Figure 2 将其展开为 SigLIP vision encoder + Gemma language model。
- 同时训练 discrete FAST action-token prediction 与 860M flow-matching action expert；action expert 的 gradient 不流回 VLM backbone。
- Input resolution 为每 camera `448×448`，最多 four camera streams。
- Pretraining mixture 包含 teleoperated demonstrations、policy rollouts、human corrections、vision-language tasks 和 video-language tasks。
- Pretraining 使用 six observations（five past + current），间隔 one second；post-training 可扩展到 18 frames / 54 seconds observation memory。
- On-robot experiments 使用 inference-time RTC 或 training-time RTC 做 asynchronous real-time execution。

### 5.6 What is actually new

真正的 contribution 是 `timescale-to-modality matching`：

1. 用 compact video encoder 保存 seconds-level fine detail；
2. 用 learned text summary 保存 minutes-level semantic state；
3. 让 high-level policy 同时决定“下一步做什么”和“现在应该记住什么”；
4. 在 diverse pretraining 中学习使用 memory，而不是只在 downstream post-training 临时接一个 history module。

Hierarchical subtask control、ViT、FAST、flow matching、LLM-generated labels 和 RTC 都是 inherited/supporting components；不能把它们全部算成 MEM 的新贡献。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Video history 保持 real-time path | Four-camera `π₀.₆` on one H100：随着 frames 增加，naive multi-frame encoding 很快进入 seconds；MEM video encoder 在 tested range 内保持在 300 ms real-time barrier 以下。 | Figure 3, PDF p. 4 | Hardware- and implementation-specific；不是 edge-device latency。 |
| Two-scale memory 对 long tasks 必要 | Recipe Setup 中 MEM task progress 约 63%，Clean Kitchen 约 90%，average 约 70%；no-memory average 约 35%。Only-video、only-text 与 naive text+video 都明显较差。 | Figure 6, PDF p. 6 | Bars 需从 graph 读取；metric 是 custom task progress，不是 binary success。每 policy/task 或 recipe 10 rollouts，报告 mean ± SE。 |
| Short-term memory 支持 strategy adaptation | Pick Up Chopstick 与 Open Fridge 的 memory-vs-no-memory 差异在 Figure 中分别标为 **+11%**、**+62%** success rate。 | Figure 7, PDF p. 7 | 该能力来自 targeted correction/exploration data + memory；不是 zero-shot online learning。 |
| MEM 跨 memory capability 更一致 | 在 Swap 3 Mugs、Find Object、Unpack Groceries、Scoop Coffee、Grilled Cheese、Window Cleaning 上，MEM average progress 约 73%，高于 no-memory、Pool Memory、Proprio Memory。 | Figure 8, PDF p. 8 | Authors re-implemented baselines on `π₀.₆`，但 tasks/evaluation 都是 internal。 |
| Memory pretraining 很重要 | `Ours` average 约 73%，`Ours (post-train only)` 约 50%；只在 downstream 加 video memory 明显较差。 | Figure 9, PDF p. 8 | 同时反映 data mixture、training scale 和 architecture exposure，不能只归因于 one factor。 |
| 添加 memory 未明显破坏 dexterity | 在 Table Bussing、Shirt Folding、Make Bed、Batch Folding、Box Building 等 non-memory tasks，MEM 与 no-memory `π₀.₆` 整体相当。 | Figure 10, PDF p. 9 | Graph 混合 out-of-the-box 与 finetuned settings；不是 universal no-regression guarantee。 |

## 7. MEM 与 `π₀.₅` / `π*₀.₆` 的关系

| Work | Primary question | Main mechanism | Runtime state | Training signal |
|---|---|---|---|---|
| `π₀.₅` | 怎样从 heterogeneous data 获得 open-world generalization？ | co-training、FAST pretraining、flow post-training、high/low hierarchy | current observation + subtask | demonstrations + multimodal co-training |
| `π*₀.₆` | 怎样从 deployment experience 改进 policy？ | RECAP、value/advantage conditioning、offline RL iterations | positive-advantage-conditioned policy | rollouts、returns、corrections、value estimates |
| MEM / `π₀.₆-MEM` | 怎样让 policy 记住 seconds-to-minutes 的 history？ | short video memory + long text memory + learned memory update | current observation + recent history + episode summary | robot/video/VL pretraining + LLM-generated memory labels + corrections |

最短记忆法：`π₀.₅ = generalize`，`π*₀.₆ = improve from experience`，`MEM = remember and adapt from context`。三者会共享 architecture/data lineage，但不回答同一个 research question。

## 8. Limitations

### Authors' stated/future boundary

- Conclusion 将跨 episode、跨 weeks/months/years 的 persistent memory 留给 future work；本文 memory 主要限于 current episode。
- Paper 没有独立 `Limitations` section。

### My critique

- **Publication/reproducibility**：目前是 PI-authored preprint；未核验到 official code、weights、training data 或 evaluation suite release。
- **Evaluation independence**：model、data、tasks、robots、progress rubric 与 evaluator 都由 authors 控制；多数 graph 每 policy/task 只有 10 rollouts。
- **Memory correctness**：text summary 是 lossy bottleneck。Paper 展示 compression 的好处，但没有系统测量 hallucinated memory、incorrect deletion 或 stale memory 的 failure rate。
- **Adaptation boundary**：Figure 7 的 adaptation 依赖 targeted interventions/exploration data；不能表述成未训练过 strategy 的 spontaneous test-time learning。
- **Timescale boundary**：15-minute claim 依靠 text memory；dense observation memory 实验上最多 54 seconds。不要说“模型直接观看了 15 分钟 video context”。
- **Systems boundary**：Figure 3 使用 one H100 和 asynchronous RTC；mobile/edge hardware、energy、memory footprint 与 P99 control latency 未报告。
- **Comparison boundary**：Pool Memory 与 Proprio Memory 是 authors 在同一 backbone 上的 representative re-implementations，但未覆盖 recurrent state、retrieval memory、external world-state map 等所有 alternatives。

## 9. Why it matters for this project

- 它把 VLA deployment bottleneck 从单步 inference 扩展到 `state retention under bounded latency`：更长 history 会增加 visual compute、activation memory 和 context management cost。
- 对 quantization，memory path 提供新的 component sensitivity：video encoder temporal layers、text-memory tokens、proprio projection 与 action expert 未必适合相同 bit-width。
- 对 FYP，一个可执行问题是：在固定 control-latency budget 下，应该优先增加 visual-memory frames，还是保留更高 precision 的 memory encoder？
- 另一个关键 evaluation 是 `memory fidelity × closed-loop success × latency`，而不只是 single-frame action error。

## 10. How to read it

### 20-minute route

1. 读 Figure 1 + Abstract（PDF pp. 1–2）：先记住 `short video / long text` 分工。
2. 读 factorization 与 Figure 2（pp. 3–4）：画 high-level memory/subtask 与 low-level action interface。
3. 看 Figure 3–4（pp. 4–5）：解释 factorized attention、drop past tokens 和 real-time barrier。
4. 核对 Figures 6–7（pp. 6–7）：区分 long-horizon progress 与 in-context adaptation。
5. 核对 Figures 8–9（p. 8）：回答为什么需要 multimodal memory 和 memory-aware pretraining。

### 90-minute route

1. **0–15 min**：复习 partial observability、hierarchical VLA 与 action chunks。
2. **15–32 min**：推导 Section III-A factorization；标出被 summary interface 丢掉的信息。
3. **32–47 min**：读 language-memory label pipeline；区分 external LLM labeling 与 inference-time memory update。
4. **47–62 min**：读 video encoder + Appendix C，解释 causal temporal attention 和 complexity。
5. **62–72 min**：核对 `π₀.₆-MEM` pretraining/post-training horizons、FAST/flow/RTC path。
6. **72–84 min**：逐个看 Figures 6–10，记录 metric、rollout count、baseline 和 caveat。
7. **84–90 min**：画一张 `π₀.₅ / π*₀.₆ / MEM` comparison，并提出一个 memory-quantization experiment。

## 11. Reading questions

1. 为什么 long-term context 适合 text compression，而 short-term correction 必须保留 dense visual detail？
2. Hierarchical factorization 在什么情况下会因为 text-summary bottleneck 而失败？
3. Inference-time memory update 为什么不是“调用 LLM 写日志”，也不是 online learning？
4. Every-4th-layer causal temporal attention + dropping past tokens 如何同时保留 motion information 与固定 backbone token count？
5. 为什么只在 post-training 加 memory 明显不如 memory-aware pretraining？这里有哪些 confounds？
6. Figure 7 如何证明 context-conditioned strategy switching，又为什么不能证明 arbitrary failure 的 spontaneous recovery？
7. 若要 quantize `π₀.₆-MEM`，vision history length、video-encoder precision、text-memory fidelity 与 control latency 应怎样做 matched ablation？

## 12. Weekly meeting card

- **Problem**：single-frame VLA 无法处理 partial observability 和 long-horizon task state；dense raw history 又不满足 latency constraint。
- **Key idea**：seconds-level detail 用 compressed video memory，minutes-level semantic state 用 learned text summary。
- **Best evidence**：Figure 6 long-task average progress 约 35%→70%；Figure 7 fridge-opening adaptation 标注 +62%；Figure 3 multi-frame path 保持在 300 ms barrier 下。
- **Biggest limitation**：closed PI pretraining/evaluation stack、10-rollout graphs、no code/weights；15-minute memory 主要是 lossy text summary。
- **Question for the group**：在 edge VLA 上，memory horizon 与 memory-encoder precision 哪一个更值得优先保留？

## 13. Status & evidence boundary

- **Status**：arXiv:2603.03596 v2 / Physical Intelligence technical report；不是 peer-reviewed venue paper。
- **Source claims**：Section III factorization、LLM-generated memory labels、video encoder complexity、six-observation pretraining、18-frame/54-second post-training、10-rollout evaluation、Figures 3/6/7/8/9/10。
- **My interpretation**：与 `π₀.₅`/`π*₀.₆` 的 design-space mapping、reproducibility critique，以及 memory quantization proposal。
- **Open question**：independent benchmarks、released implementation、persistent cross-episode memory、memory hallucination rate 与 edge-device latency/energy。
