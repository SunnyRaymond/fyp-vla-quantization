# 105 — A Survey on Efficient Vision-Language-Action Models

- 作者：Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan Zhang, Ji Zhang, Zheng Wang, Lianli Gao, Jingkuan Song, Nicu Sebe, Heng Tao Shen
- 版本：arXiv:2510.24795v2，2026-02-02，28 pages
- 定位：从 `model–training–data` 全链路整理 Efficient VLA 的综述；适合作为文献地图，不是统一 benchmark。
- Source：[arXiv abstract](https://arxiv.org/abs/2510.24795v2) · [HTML](https://arxiv.org/html/2510.24795v2) · [project page](https://evla-survey.github.io/)
- Local PDF：[paper-arxiv-v2.pdf](paper-arxiv-v2.pdf)

## 一句话抓手

作者把 Efficient VLA 分成三个互相依赖的支柱：`Efficient Model Design`、`Efficient Training` 与 `Efficient Data Collection`；论文真正有用之处是建立检索地图，而不是证明某一种方法在共同设置下最好。

## Background

Foundational VLA 通常由 vision encoder、VLM/LLM backbone 与 action decoder 组成。它们的扩展路线带来三类成本：推理 latency / control frequency 不匹配，pre-training 与 adaptation compute 昂贵，以及 robot trajectory 的采集成本高。论文因此把“efficiency”从单纯的模型压缩扩展到完整生命周期。

## Problem

已有工作分别讨论 architecture、compression、training 或 data collection，但使用的效率目标、hardware、task 与 evaluation protocol 不统一。作者试图用同一 taxonomy 把这些工作串起来，并给出 applications、limitations 与 future directions。

## Method / Taxonomy

### 1. Efficient Model Design

- `Efficient Architectures`：efficient attention、Transformer alternatives、efficient action decoding、lightweight components、Mixture-of-Experts、hierarchical systems。
- `Model Compression`：layer pruning、model quantization、token optimization / caching。

### 2. Efficient Training

- `Efficient Pre-training`：data-efficient pre-training、compact action representations、multi-stage or modular strategies。
- `Efficient Post-training`：parameter-efficient supervised fine-tuning 与 RL-based adaptation。

### 3. Efficient Data Collection

- Human-in-the-loop collection、simulation、internet-scale / cross-domain reuse、self-exploration 与 data augmentation。

## Key Innovation

这篇综述的贡献不是新算法，而是把部署效率、训练效率和数据效率放进同一张 `model–training–data` 地图。它也反复强调三组 trade-off：compactness vs. expressivity、scalability vs. stability、data scale vs. physical fidelity。

## Main Takeaways

- 结构优化正在从静态轻量化转向 task-、state- 和 hardware-aware 的动态计算路径。
- Compression 不只发生在 weights：layer、vision/action tokens、KV/intermediate features 与 temporal reuse 都可能成为优化对象。
- Training efficiency 涉及迁移 VLM priors、压缩 action representation、PEFT 与 RL refinement；它不等于 inference efficiency。
- Data efficiency 的趋势是用 simulation、human video、world models 与 generative augmentation 替代部分人工 teleoperation，但 sim-to-real、embodiment mismatch 和 label fidelity 仍是硬约束。
- 作者给出的未来方向包括 dynamic token routing、hardware-software co-design、physics-informed objectives、continual/federated learning 与 self-sustaining generative data ecosystems。

## 与本项目最相关的阅读边界

### “Quantization”至少要拆成两类

1. `Model quantization`：降低 weights / activations 的数值精度，目标通常是 memory、throughput、latency 或 energy。
2. `Action quantization / tokenization`：把连续 action 映射成 discrete codes 或较短序列，主要改变 action representation 与 decoding/training cost。

论文在 Table III 与正文中同时讨论 OpenVLA、QAIL、SQIL、BitVLA、RLRC、FAST 和 SQAP-VLA，但这些工作的量化对象、训练方式和部署 claim 并不相同。尤其不能由 action token compression 推出 low-bit model deployment，也不能把 QAT / native low-bit training 的结果当成 PTQ 证据。

### 结果不可横向当 leaderboard

Table I–VI 汇总了不同 paper 的参数量、latency、frequency 或方法特点，但 hardware、batch size、observation setup、action horizon、robot/task 和 success metric 并未统一。表格适合发现候选文献，不足以支持严格的速度或 Pareto 排名。

## Limitations / Evidence Boundary

- 这是 narrative survey，不是带有明确检索式、纳排标准和质量评估的 systematic review。
- 多数数字是对原论文的二次汇总；本笔记未把它们视为独立复现结果。
- Taxonomy 覆盖很广，导致部分概念边界偏松：architecture、compression、representation 与 caching 的收益来源可能互相重叠。
- “first comprehensive survey”是作者自述的 novelty claim；不应当作独立核验后的结论。
- Future Works 多为方向性判断，而非已经验证的 research gap。
- v2 截止 2026-02-02；快速发展的 2026 work 需要另行更新，project page 也不能替代固定版本的论文证据。

## Why It Matters for the FYP

它可以作为 VLA efficiency 文献的入口索引，并帮助把现有课题放进更大的设计空间：`what is compressed`、`when efficiency is gained`、`where the cost moves`、`which metric is actually measured`。对 PTQ / world-action-model 研究，最值得利用的不是大而全的 paper list，而是检查 survey 中不同路径是否真的在共同 backbone、共同 task 与共同 hardware 下比较。

## Reading Route

### 20 minutes — 建立地图

1. Abstract + Section I：确认作者如何定义 Efficient VLA。
2. Figure 2：记住 `model–training–data` 三支柱。
3. Table III：只标记 model quantization、action tokenization、token pruning/caching 的边界。
4. Section VII：圈出作者认为仍未解决的 trade-offs。

### 60 minutes — 聚焦 FYP

1. Section II-A/B：记录 VLA pipeline 与效率指标的测量对象。
2. Section III-A：区分 architecture-level acceleration 与 action-decoding acceleration。
3. Section III-B：逐项判断 compression object、precision regime、training requirement 与 hardware dependency。
4. Section III-C：核对作者自己承认的 semantic drift、asynchrony 与 static importance limitations。
5. Section VII：把 future directions 分成已有证据与 proposal。

### 2 hours — 做成可用文献地图

1. 浏览 Tables II–VI，每类各挑 2–3 篇与你的课题最接近的原论文。
2. 为每篇建立四列：`compressed object`、`method regime`、`evaluation setting`、`deployment evidence`。
3. 回到对应原论文核验 headline numbers，不直接引用 survey 的二手数字。
4. 标记 survey 遗漏或混合的类别，并判断是否形成真正的 research gap。

## Reading Questions（留给你回答）

1. 作者对 Efficient VLA 的定义是按 resource objective、method family，还是 deployment outcome？这个定义是否可证伪？
2. 三支柱之间有哪些成本转移，例如更少 robot data 是否换来更多 simulation compute？
3. Table I 的 latency / frequency 是否足以跨 model 比较？至少还缺哪些 control-loop 与 hardware 条件？
4. Table III 中哪些工作真正量化 model weights / activations，哪些只是量化或压缩 action representation？
5. OpenVLA、QAIL、SQIL、BitVLA、RLRC 与 SQAP-VLA 分别属于 PTQ、QAT、native low-bit training 还是联合优化？
6. 哪些 headline gains 是 algorithm-only，哪些依赖 custom kernel、specific accelerator 或 asynchronous execution？
7. Token pruning、caching 与 action reuse 会如何改变 closed-loop feedback frequency 和 failure recovery？
8. 作者提出的 compactness–expressivity trade-off 有哪些直接实验支持，哪些只是综合判断？
9. Efficient Training 和 Efficient Data Collection 的“效率”是否用了可比较的成本单位？
10. Human-video / simulation / world-model data 的 embodiment mismatch 在哪些任务阶段最严重？
11. 这篇综述遗漏了哪些与你的 world model / world-action model quantization 课题直接相关的工作？
12. 如果建立严格 benchmark，最小的共同报告项应包括哪些 accuracy、latency、memory、energy 与 control metrics？
13. 哪一个 future direction 已有足够 prior work，不应再当成未经探索的 novelty gap？
14. 这篇综述最适合作为哪类 claim 的 source，哪些 claim 必须回到 primary paper？

## Meeting Card

- **Paper**：Yu et al., *A Survey on Efficient Vision-Language-Action Models*, arXiv:2510.24795v2。
- **Core idea**：用 `Efficient Model Design / Training / Data Collection` 统一整理 Efficient VLA。
- **Most useful artifact**：Tables II–VI 可作为文献入口，Section VII 可作为 hypothesis source。
- **Critical caveat**：不是 common-protocol benchmark；model quantization 与 action quantization/tokenization 必须拆开。
- **FYP connection**：可用来定位 PTQ 在全链路 efficiency 中的位置，并寻找 equal-cost、same-backbone、same-task 的证据缺口。
- **Question to bring to meeting**：我们要优化的是 model memory、kernel latency、closed-loop control quality，还是 total data/compute cost；现有实验能否区分这些目标？

## Citation

```bibtex
@article{yu2026surveyefficientvla,
  title   = {A Survey on Efficient Vision-Language-Action Models},
  author  = {Yu, Zhaoshu and Wang, Bo and Zeng, Pengpeng and Zhang, Haonan and Zhang, Ji and Wang, Zheng and Gao, Lianli and Song, Jingkuan and Sebe, Nicu and Shen, Heng Tao},
  journal = {arXiv preprint arXiv:2510.24795},
  year    = {2026},
  version = {v2}
}
```
