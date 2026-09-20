# 023. Fast LeWorldModel (Fast-LeWM)

> **完整标题：** *Fast LeWorldModel*
>
> **本地论文：** [paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)
>
> **Official resources：** [arXiv:2606.26217v1](https://arxiv.org/abs/2606.26217v1) · [HTML](https://arxiv.org/html/2606.26217v1) · [project page](https://fast-lewm.github.io/) · [official code](https://github.com/Yuntian-Gao/Fast-LeWorldModel) · [pretrained checkpoints](https://huggingface.co/naiverer/fast-leworldmodel/tree/main)
>
> **建议先修：** [#001 LeWorldModel](../001-leworldmodel/README.md)、`JEPA`、`CEM / MPC`、causal Transformer、`SIGReg`。
>
> **阅读状态：** `verified-full-text + static-code-audit`；本地 PDF 与阅读入口已核对，代码只做静态检查与 Python syntax compile，没有安装依赖、下载 checkpoint/data、执行 training、GPU inference 或 benchmark。

## 1. Paper identity

| Field | Record |
|---|---|
| Authors | Yuntian Gao, Xiangyu Xu |
| Affiliation | Xi'an Jiaotong University |
| arXiv | `2606.26217v1`，submitted 2026-06-24 |
| Local artifact | arXiv v1，11 pages，2,155,469 bytes，unencrypted |
| Venue / status | arXiv preprint；本次未找到 peer-reviewed venue 声明 |
| Code snapshot checked | official repository `main` HEAD `492752d96b2a11ec802c55322c46bdc87885da09`，commit time 2026-08-06；2026-09-19 轻量静态核对 |

不要把它与 generic “fast world model” 工作混在一起。这里的 `Fast` 特指：把 LeWM 的 repeated one-step latent rollout 改成 **action-prefix prediction**，在一次 causal prefix encoding 和一次 parallel prediction 中得到多个 horizon 的 latent。

## 2. 一句话抓手

LeWM 对每个 CEM candidate 逐步执行 `z_t -> z_{t+1} -> ... -> z_{t+H}`；Fast-LeWM 则把 `(a_t)`、`(a_t,a_{t+1})`、...、`(a_t,...,a_{t+H-1})` 编成不同 prefix tokens，并让每个 token 从同一个 observed anchor latent `z_t` 直接预测对应 future latent，因此避免在模型内部递归喂回 predicted latent，同时把 `H` 次 dynamics calls 合并为一次 batched call。

## 3. Problem：LeWM 的 local transition interface 为什么慢

LeWM 学的是 one-step transition：

$$
\hat z_{t+1}=F_\phi(z_t,a_t).
$$

要评估一个长度为 `H` 的 candidate action sequence，CEM 必须 autoregressively 调用它 `H` 次：

$$
\hat z_{t+k}=F_\phi(\hat z_{t+k-1},a_{t+k-1}).
$$

这带来两个相互独立的问题：

- **sequential compute：** 后一步依赖前一步，难以把 horizon 维度完全并行；
- **recursive error propagation：** 早期 predicted latent 会成为后续输入，误差可能沿 rollout 累积。

Fast-LeWM 解决的是 dynamics-query interface，不是给原 LeWM 加 inference cache，也不是量化或 kernel optimization。

## 4. Method

### 4.1 Action-Prefix Encoder

对一个候选序列 `a_t:t+H-1`，先把 current latent `z_t` 经 two-layer MLP 变成 state token，prepend 到 action tokens 前。causal Transformer 保证第 `k` 个 action-position output 只能看到 state token 与前 `k` 个 actions：

$$
p_{t,k}=E_\psi^{(k)}(a_t,\ldots,a_{t+k-1}\mid z_t).
$$

这一步的重要约束是 **no future-action leakage**。如果 causal mask 或 token indexing 错了，短 prefix 会偷看到后续 actions，dense supervision 的含义就失效。

### 4.2 Parallel Latent Predictor

每个 prefix token 与同一个 anchor latent 配对：

$$
\hat z_{t+k}=G_\phi(z_t,p_{t,k}),\quad k=1,\ldots,H.
$$

论文实现使用 6-layer action-modulated residual MLP、AdaLN-zero、latent dim 192、MLP width 2048、fusion width 768、dropout 0.1。不同 horizon 在 batch/token 维并行处理，但它们共享 parameters；“parallel”不是训练 `H` 个独立 predictors。

### 4.3 Dense Prefix Supervision

每个 prefix 都对应 ground-truth future observation 的 encoded latent：

$$
\mathcal{L}_{prefix}=\frac{1}{H}\sum_{k=1}^{H}\|\hat z_{t+k}-z_{t+k}\|_2^2.
$$

再加与 LeWM 同类的 `SIGReg` anti-collapse regularizer：

$$
\mathcal{L}_{AP}=\mathcal{L}_{prefix}+\lambda\,SIGReg(Z).
$$

因此它不是简单把 25 primitive actions concatenate 后只监督 terminal state。Table 4 的 `Terminal-only Fast-LeWM` 与 `Long-Action LeWM` 正是对应的 negative controls。

### 4.4 Planning and Optional Self-Consistency

base Fast-LeWM 与 LeWM 一样按 terminal latent-to-goal distance 给 CEM 排序：

$$
C_{goal}^{(m)}=\|\hat z_{t+H}^{(m)}-z_g\|_2^2.
$$

optional self-consistency 会比较：

1. 从 `z_t` 用完整 prefix 直接预测 25-step terminal latent；
2. 先预测 10-step intermediate latent，再从那里预测剩余 15 steps。

二者 discrepancy 以 `beta` 加入 CEM cost。它不是训练 loss，也不是 uncertainty calibration；它只在 candidate scoring 时改变 ranking。

## 5. Main Results

### 5.1 Closed-loop planning success

| Method | Two-Room | Reacher | PushT | OGBench-Cube | Avg. |
|---|---:|---:|---:|---:|---:|
| LeWM | 87 | 86 | 96 | 74 | 85.8 |
| Fast-LeWM | 98 | 88 | 96 | 80 | 90.5 |
| Fast-LeWM + Self-Consistency | 98 | 90 | 98 | 82 | 92.0 |

这些是作者在同一组四个 goal-conditioned tasks 下报告的 success percentages。论文没有给每个 cell 的 confidence interval、seed dispersion 或 raw episode outcomes，因此不能据此断言小幅差异具备统计显著性。

### 5.2 Planning time

在 single NVIDIA 4090、Two-Room、相同 CEM budget 下：

| Method | Dynamics calls | Dynamics time | Full CEM time |
|---|---:|---:|---:|
| LeWM | 5 | 31.4 s | 54.4 s |
| Fast-LeWM | 1 | 8.0 s | 28.3 s |

`3.9x` 指 action encoding + latent prediction 这一 dynamics module；end-to-end CEM solve time 是 `54.4 -> 28.3 s`，约降低 48%，不是 3.9x。论文没有报告训练 time、peak memory、batch-size scaling、不同 horizon scaling 或 closed-loop control frequency。

### 5.3 Open-loop and representation evidence

- Figure 3 显示四个 tasks 上 Fast-LeWM 的 latent MSE 起点与随 raw steps 增长的 slope 都较低；这支持“recursive rollout error 较少”，但 latent MSE 仍不等价于 planner ranking 或 task success。
- PushT probing 中，MLP probe 对 agent/block state 的 MSE 更低；linear probe 的 block angle 反而比 LeWM 差 (`0.314` vs `0.187`)。因此“latent 更物理”应限定为所报告 probes，而不是全方位更 linearly disentangled。
- 论文 horizon 固定为 `H=5` prefix blocks，每个 block 含 5 primitive actions，总覆盖 25 environment steps。它没有证明 variable or much-longer horizons 仍保持同样 speed/accuracy relationship。

## 6. Official Code：当前是否可用

### 6.1 结论

**核心 implementation 已发布，具备研究性使用的基本材料，但还不能称为 turn-key reproduction package。**

支持“可用”的证据：

- repository 包含 `train.py`、`eval.py`、`jepa.py`、`module.py`、四个 environment 的 train/eval configs 与 pinned `requirements.txt`，不是只有 README；
- Hugging Face 已提供 four `*_object.ckpt` files，总计约 288 MB；
- 本次 Python syntax compile 通过；
- 关键 call path 与论文一致：`ActionPrefixEmbedder` prepend state token、causal attention、drop state output、保留 action-position prefix tokens；training 对各 prefix target 做 MSE；planning 默认只使用 terminal prediction，optional consistency 另走 `[2,3]` prefix decomposition；
- 主要 architecture values 与论文一致：`3 layers / 6 heads / head dim 32 / token dim 192`，predictor `6 layers / MLP 2048 / fusion 768 / dropout 0.1`，training horizon `1..5`、action skip `5`、history size `1`。

限制“可用”程度的证据：

- 没有 tests、CI、container 或最小 smoke-test script；本次未证明依赖能在干净环境一次解析成功；
- datasets 仍依赖 LeWM / `stable-worldmodel` 外部数据布局；repository 本身不带 data downloader；
- pretrained files 是 serialized Python object checkpoints，Hugging Face model card 几乎为空，没有逐 checkpoint config、metrics、provenance 或 weights-only alternative；这会增加 version-coupling 与安全加载负担；
- repository 没有公开对应 Figure 3 open-loop curves、Table 2 latency instrumentation、Table 3 probes 或 Table 4 ablations 的独立 scripts/configs，因此“能训练/评测主模型”不等于“论文所有表图一键可复现”。

### 6.2 一眼可见的 paper/code/documentation gaps

| Finding | Evidence | Assessment |
|---|---|---|
| **Cube batch size mismatch** | Paper says Cube uses batch size `32`; released `Fast-lewm.yaml` globally sets `loader.batch_size: 128`, while `data=ogb` does not override it. | **明确 paper/config 差异。** 若按 README 的 generic training command，只换 `data=ogb` 不会自动得到 paper setting；需显式 `loader.batch_size=32`。 |
| **Self-consistency disabled by default** | Paper's `+Self-Consistency` row uses `beta=1`; all four eval YAMLs set `consistency_loss_weight: 0.0`. | **不是 base Fast-LeWM 的错误，但 README 没给 variant command。** 要得到该 row 必须显式 override 为 `1.0`；`[2,3]` 对应先 10-step、再 15-step。 |
| **Default checkpoint reference looks invalid** | Eval YAMLs set `policy: Fast-lewm_weights.ckpt`; code passes `policy` directly to `AutoCostModel`, whose LeWM convention expects a run name without `_object.ckpt`. The provided HF artifacts are `Fast-lewm_*_object.ckpt`. | **默认 config 很可能不能直接加载 released checkpoint。** README 的 explicit `eval.ckpt_path=..._object.ckpt` path 会被 parser 正确去掉 suffix，是较可信入口。 |
| **README data filenames do not match configs** | README example lists `tworoom_expert_train.h5`, `reacher_expert_train.h5`, `cube_expert_train.h5`; train configs request `tworoom`, `reacher`, `cube_single_expert`. | **明显 documentation/config mismatch。** 因 LeWM loader 通常按 dataset name 直接映射 `<name>.h5`，应先以实际下载文件名为准，不能只照 README 示例。 |
| **Paper artifacts not all exposed** | Main train/eval route exists, but no dedicated public entrypoints for latency, open-loop plot, probing, or ablation tables. | **coverage gap，不是 method implementation mismatch。** 现有代码不足以直接审计全部 headline evidence。 |

### 6.3 看起来不同、但未必是 mismatch

- Paper 写 planning horizon `H=5`；eval YAML 写 `horizon: 1`、`action_num_blocks: 5`、`action_block_size: 5`。在这份实现里，一个 outer rollout step 内含 5 个 prefix blocks，每个 block 有 5 primitive actions，因此仍对应 25-step terminal prediction。不要仅凭 `horizon: 1` 判定它把论文 horizon 改成了 1。
- `jepa.py::rollout` 仍保留 `rollout_steps > 1` 的 autoregressive outer loop。这是为了支持超出单个 5-prefix window 的 repeated planning/diagnostic rollout；论文的主 CEM config 走 `rollout_steps == 1`，内部 5 个 prefixes 并行处理。
- `eval.py` 增加 buffered-action fast path，在 MPC buffer 非空时跳过重复 image preprocessing。这是 engineering optimization，不是 action-prefix method 本身；若比较 end-to-end controller timing，应单独标明它是否启用。

## 7. Evidence Boundary / Limitations

- 论文只覆盖 four LeWM goal-conditioned tasks；没有 stochastic dynamics、partial observability、real robot、contact-rich long-horizon recovery 或 safety constraints。
- 固定 `H=5` 与 action skip `5`，无法回答 horizon 更长时 prefix Transformer 的 compute/memory、distribution shift 与 direct-prediction bias 如何变化。
- 预测并非数学 exact：Fast-LeWM 换了 model architecture 并重新训练。它与 exact cache/reuse 不同，速度提升伴随 learned approximation 与 changed representation。
- 17.9M vs 18.0M parameters 只控制 model size 量级，不自动保证同等 FLOPs、memory traffic、training compute 或 optimization difficulty。
- Table 2 只给 single 4090 的 aggregate seconds；没有 warmup、CUDA synchronization、number of CEM iterations/candidates 的完整 timing protocol，也没有 variance。
- `Fast-LeWM + Self-Consistency` 多做一条 decomposed prediction path。论文报告 success，但没有单列它的额外 latency；不能把 base Fast-LeWM 的 28.3 s 直接赋给该 variant。
- 论文说 all other settings follow LeWM，但 current official LeWM repository 的 config/history可能继续变化。公平复现应固定两边的 exact commit、dataset snapshot、checkpoint 与 dependency versions。

## 8. Why It Matters for the FYP

Fast-LeWM 是当前 DINO-WM / LeWM acceleration 方向中一个很直接的 **model-redesign baseline**：它不缓存 shared computation，而是重定义 dynamics query，让 planner 用 multi-horizon prefix tokens 一次得到 terminal prediction。

因此对本项目的作用是形成清晰的比较边界：

- **Fast-LeWM：** retrain required；改变 predictor interface；approximate learned direct prediction；headline gain 是 5 calls -> 1 call。
- **RankSafe-EPC / exact shared-prefix reuse：** 理想目标是不改 checkpoint 与 planner objective，复用 action-independent exact prefix；正确性看 candidate score/ranking/top-k/first action 与 closed-loop behavior。
- **QuantWM / low-bit deployment：** 改 arithmetic/representation precision；必须区分 fake quantization、native low-bit kernel 与 real latency/memory gain。

如果后续要做公平对比，最重要的不是跨论文比较 `3.9x`，而是同一个 LeWM backend、同一 CEM budget/hardware/timing boundary 下放置：original autoregressive LeWM、Fast-LeWM retrained architecture、exact reuse（若成立）与 quantized execution。

## 9. Reading Route

### 20 minutes - 抓住机制与 claim boundary

1. Abstract + Figure 1：区分 `3.9x dynamics` 与 `48% full CEM reduction`。
2. Figure 2 + Sections 3.3-3.6：画出 `state token + causal action prefixes -> parallel latents -> dense targets`。
3. Section 3.7：区分 base goal cost 与 optional self-consistency cost。
4. Table 1 + Table 2：分别记录 task metric、timing boundary、hardware 与 variant。

### 90 minutes - 读到可以解释与质疑

1. Sections 3.2-3.5：对比 LeWM recursive input dependency 与 Fast-LeWM anchor dependency。
2. Section 4.1：核对 `H=5`、action skip `5`、state token、3-layer Transformer、6-layer predictor 与 10 epochs。
3. Figures 3-4：区分 latent loss、decoded visualization 与 closed-loop success。
4. Tables 3-4：找出 block-angle linear probe 反例，以及 Long-Action / terminal-only / no-state-token controls。
5. 回到本 README 的 code gap table：判断哪些影响“能跑”、哪些影响“能复现 paper row”。

### 3 hours - 形成 FYP prior-art card

1. 从 official code 追踪 `train.py::lejepa_forward -> JEPA.encode/predict` 与 `eval.py -> JEPA.rollout/get_cost`。
2. 画一张 execution graph，标出 candidate dimension、prefix dimension、outer rollout step 与 CEM iteration。
3. 为 LeWM / Fast-LeWM / exact-cache proposal 建表：`retrain? / exact? / calls / saved compute / extra compute / planner objective / correctness metric`。
4. 设计一个不运行大规模复现也能先做的 static gate：确认 exact commits、dataset filenames、Cube batch override、checkpoint loading contract 与 self-consistency override。

## 10. Reading Questions（留给你回答）

1. 为什么 action prefix 比把 25 primitive actions concatenate 成一个 vector 更能保存 order 与 partial outcomes？Table 4 排除了哪些 alternative explanations？
2. causal mask 如何保证 `p_t,k` 不看到 future actions；state token 在 position 0 时 indexing 是否完全对齐？
3. 每个 prefix horizon 独立从 `z_t` 预测，会减少 recursive error，但会不会牺牲 cross-horizon consistency？
4. dense prefix MSE 对每个 horizon 等权是否合理；更长 horizon 是否应更高/更低权重？
5. SIGReg 是作用于所有 time latents 还是合并后的 distribution；code 的 `merge_time_into_batch=False` 对 objective 有什么含义？
6. 论文的 “error accumulation” 改善有多少来自 architecture，多少来自多目标 dense supervision？Terminal-only ablation 能否完全拆开二者？
7. self-consistency penalty 衡量 epistemic uncertainty、model bias，还是仅仅 decomposition disagreement？
8. `[2,3]` decomposition 只测试一种 10/15 split；换成 `[1,4]`、`[3,2]` 或多分解 ensemble 会怎样？
9. 论文没有报告 self-consistency 的 latency，它是否会部分抵消 base Fast-LeWM 的 single-call advantage？
10. Table 2 的 31.4/8.0 s 包含 CUDA synchronization、warmup、CEM candidates 与 iterations 的哪些部分？
11. full CEM 还有约 20 s 未被 dynamics acceleration 消除；goal/image encoding、score/data operations 中谁是下一个 bottleneck？
12. 17.9M vs 18.0M parameter matching 是否足以支持公平 speed comparison，还是应再报告 FLOPs、activation memory 与 kernel utilization？
13. open-loop latent MSE 更低是否必然保持 CEM candidate ranking；应该补什么 rank/top-k/first-action metrics？
14. current code 的 `horizon:1 + action_num_blocks:5` 如何映射到 paper 的 `H=5`，outer rollout 超过一步时又会发生什么？
15. Cube batch size released config 与 paper 不一致，会不会影响 checkpoint provenance 或只影响从头训练复现？
16. HF object checkpoints 如何安全、稳定地加载；能否从 state dict + explicit config 重建以减少 version coupling？
17. README dataset names 与 config names 哪一套对应实际 LeWM archive；是否需要 per-task manifest？
18. 若把 Fast-LeWM 当作 RankSafe-EPC 的 comparison baseline，什么是公平的 same-backend protocol，什么比较会混入 retraining advantage？

## 11. Meeting Card

- **Paper：** Gao & Xu, *Fast LeWorldModel*, arXiv:2606.26217v1, 2026。
- **Core idea：** 用 state-conditioned causal action-prefix tokens，从同一 observed latent 并行预测多个 future horizons，替代 LeWM 的 repeated one-step autoregressive rollout。
- **Strongest evidence：** four LeWM tasks average success `85.8% -> 90.5%`；single 4090 上 dynamics `31.4 -> 8.0 s`，full CEM `54.4 -> 28.3 s`。
- **Critical caveat：** 只有 fixed `H=5`、single-GPU aggregate timing；self-consistency latency 未单列；当前 release 的 Cube batch size、default checkpoint reference 与 dataset naming 有明显 config/documentation gaps。
- **Code verdict：** core method 与 four checkpoints 已发布，足以做 code reading 和后续 gated smoke test；还不是论文所有 headline tables/figures 的 turn-key reproduction package。
- **FYP connection：** 它是 retrained model-redesign acceleration baseline，不是 exact cache 或 quantization baseline；公平对比要固定 LeWM backend、CEM budget、hardware 与 timing boundary。
- **Question to bring to meeting：** 我们要比较的是“减少 dynamics calls 的 architecture redesign”，还是“保持 checkpoint/planner semantics 的 exact reuse”；两者的 correctness contract 与研究价值如何分开？

## 12. Citation

```bibtex
@misc{gao2026fastleworldmodel,
  title         = {Fast LeWorldModel},
  author        = {Yuntian Gao and Xiangyu Xu},
  year          = {2026},
  eprint        = {2606.26217},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  version       = {v1},
  url           = {https://arxiv.org/abs/2606.26217}
}
```

## 13. Verification note

本地 PDF 已核对 title metadata、authors、11-page count、unencrypted status，并逐页渲染检查；未见 clipping、blank page 或 unreadable figure/table。静态 code audit 使用 official `main` snapshot，syntax compile 通过；由于没有安装 pinned ML stack、datasets 与 checkpoints，本条目不声称 runtime、numerical results 或 clean-environment dependency resolution 已验证。
