# 031. Product Quantization for Nearest Neighbor Search

> **本地论文：** [paper-original-mirror.pdf](paper-original-mirror.pdf)（公开 mirror 的原文副本，12 页）
>
> **正式记录与实现参考：** [DOI: 10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57) · [HAL record](https://inria.hal.science/inria-00514462v2) · [Faiss indexes：PQ / IVF-PQ](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
>
> **来源说明：** HAL 作者 postprint PDF 当前返回 browser-challenge HTML；本包改用指定的公开 mirror。该文件首面 title/authors 与本文相符，正文带 TPAMI 卷期、页码和 DOI；但 mirror 不是官方 host，故本地 PDF **不标为 official final**。
>
> **阅读状态：** `unread`。已核对本地 PDF 首面身份、页数，并阅读方法和实验章节定位；未运行 Faiss 或复现搜索实验。

## 1. Paper identity

| Field | Record |
|---|---|
| Title | *Product Quantization for Nearest Neighbor Search* |
| Authors | Hervé Jégou, Matthijs Douze, Cordelia Schmid |
| Venue / year | IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1), 117–128, 2011 |
| DOI / HAL | `10.1109/TPAMI.2010.57` · `inria-00514462v2` |
| Local artifact | 12-page paper copy from public third-party mirror; PDF identifies the TPAMI article, but the mirror file is not represented as an official final PDF |
| Author PDF status | Specified HAL PDF URL returned HTML browser challenge; no authentication or challenge bypass attempted |

## 2. 一句话抓手

Product quantization（PQ）把一个固定的 `D`-dimensional vector 切成 `m` 个 sub-vectors，各自用自己的 codebook 量化，再把 `m` 个 code indices 拼成紧凑 code；查询时用查表近似距离，服务于近似最近邻（ANN）检索。它压缩并检索向量，**没有时间轴、state transition 或 temporal update，因此本身不是 world model，也不保证 latent blocks 在统计意义上独立。**

## 3. 背景：一个大 codebook 难以训练和存储

普通 vector quantization 用一个包含 `k` 个 centroid 的 codebook 表示整个向量。若要极高精度，codebook 会变得庞大，codebook 学习和分配也昂贵。PQ 改用 Cartesian product codebook：将 `x∈Rᴰ` 按坐标分成 `m` 个子向量 `x₁,…,xₘ`，在每个子空间单独学 `qⱼ`，最终 code 是 `(q₁(x₁),…,qₘ(xₘ))`。若每个子空间 `k` 个 centroid，组合空间可表达 `kᵐ` 种 code，而只需分别存储各子空间 codebook。

这是一种向量编码/检索分解。各子空间的量化器分别训练，不等于原向量各 block 的数据统计相互独立；分组方式与数据结构会影响近似误差。

## 4. Method：编码、查距离与缩小搜索集

1. **Subspace quantization。** 维度切分为 `m` 段；每段由独立的 k-means codebook 映射到最近 centroid。向量以各段 centroid 的 indices 表示，码长由每段 index 位数累加。
2. **近似距离。** 在 asymmetric distance computation（ADC）中，query 保持原向量，只把库向量 PQ 编码；预先计算 query 各段到对应 codebook centroids 的距离表，再逐段查表并求和。Symmetric distance computation（SDC）则两边都量化。距离是近似值，nearest-neighbor rank 可能变化。
3. **IVFADC。** 为避免扫描所有 codes，先用 coarse quantizer 将库向量放入 inverted lists；对 residual 再作 PQ。查询只探测部分 lists，再用 ADC 排序候选。速度/召回取决于 coarse partition、探测列表数、PQ 参数与数据分布。

## 5. 贡献、实验与限制

贡献是把 high-dimensional 向量表示为可查表的短 code，并展示 PQ 距离估计与 inverted-file search 的组合。论文在 SIFT、GIST 等图像描述子上考察 code length、量化误差、recall/search accuracy、内存和查询时间，也比较了其他 ANN 路线并报告大规模向量检索场景。结果支持的是这些 descriptor 检索设置中的压缩与 ANN trade-off。

证据边界：量化距离不会自动保留 exact nearest-neighbor ordering；结果受 `m`、每段 codebook 大小、coarse index、subspace grouping 和数据结构影响。论文还专门检查了 component grouping，提醒按坐标顺序切块可能拆散有结构的 descriptor 分量。它没有预测未来 latent、规划动作、评估 PushT 或给出 neural inference 的通用加速保证。

## 6. 与 LeWM + PushT 的关系

可借鉴的只有 **partition + per-block codebooks + compact codes** 这一设计思路，例如讨论如何压缩固定 latent 向量或检索向量库。若把它放进 planner score path，还要检查近似距离的候选排序、elite set 与首动作是否漂移，以及实际 latency/memory；PQ 论文的 ANN recall 不替代这些证据。LeWM 的 world model 需要 action-conditioned temporal predictor，PQ 没有 transition function、history update 或闭环机制，所以不应把 PQ 直接称作 LeWM/world-model 方法。

## 7. Reading route

### 20 分钟：抓住“乘积 codebook”

1. Abstract、Introduction（PDF p.1）：目标是大规模 ANN，不是 dynamics prediction。
2. Section 2.2 与 Figure 1 / Table 1（PDF pp.2–3）：画出 `D → m` sub-vectors → `m` code indices，留意存储与量化误差的 trade-off。
3. Section 3.1、Figure 2（PDF p.4）：看 ADC 与 SDC 怎样使用距离表。

### 90 分钟：理解距离近似与 IVFADC

1. Sections 2.1–2.2（PDF pp.2–3）：对比全空间 quantizer 与 product quantizer。
2. Sections 3.1–3.3、Figures 2–4（PDF pp.4–6）：逐步检查 ADC/SDC、distance error 与 estimator。
3. Sections 4.1–4.3、Figure 5（PDF pp.6–7）：理解 coarse quantizer、residual PQ、inverted lists 和查询过程。
4. Sections 5.2–5.5、Figures 6–10（PDF pp.8–10）：对照码长、recall、grouping、速度和比较方法。

### 180 分钟：判断能借哪一层

1. 完读 Sections 5.6–5.7 与 Figures 11–12（PDF p.11），区分大库检索和图像搜索的实验证据。
2. 用 [Faiss 官方 indexes 文档](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) 对照 IndexPQ / IndexIVFPQ 的工程接口；只读文档，不运行 benchmark。
3. 写出若用于 LeWM latent 的目标：压缩存储、nearest-neighbor lookup，还是近似 planner score？每种目标分别列出 exact-error/ranking 与 native memory/latency 的验证要求。

## 8. Reading Questions（留待阅读后回答）

1. 固定 code length 时，增加 subquantizers `m` 与增加每段 centroid 数的 trade-off 是什么？
2. 为什么 PQ 可隐式表达很大的 product codebook，却不需要显式存储全部组合 centroid？
3. ADC 与 SDC 分别近似哪一侧，查表成本和误差来源有什么差别？
4. 量化误差小是否足以保证最近邻排序不变？还缺少什么 margin 条件？
5. 各段独立训练的 codebook 如何利用、又会遗漏跨段相关性？
6. 对 SIFT/GIST 这类结构化 descriptor，component grouping 怎样改变 retrieval quality？
7. IVFADC 的 coarse residual 编码与直接对原向量 PQ 相比，解决了哪部分搜索瓶颈？
8. 探测更多 inverted lists 时，recall、查询时间和候选数如何权衡？
9. 论文使用的 descriptor/query distribution 对其他 modality 的量化结论有什么限制？
10. 压缩存储、降低距离计算量与端到端 wall-clock speedup 是怎样的不同 claim？
11. 对 LeWM 的 latent-goal score，PQ 距离误差会怎样改变 CEM candidates 的排序？
12. 仅固定候选集上的 MSE，为什么不足以证明 PQ 适合 planner？
13. 若 latent blocks 有明显相关结构，是否应先旋转/重排维度，如何用数据验证？
14. 对 PushT，PQ 最合理的角色是向量库压缩、缓存 lookup，还是 score approximation？各自需要什么独立验收指标？

## 9. Meeting Card（空白）

- **Paper / version：**
- **Core idea：**
- **最强实验结论：**
- **适用范围与限制：**
- **与 LeWM + PushT 的关系：**
- **待讨论问题：**

## 10. Sources and implementation

- [DOI / publisher record](https://doi.org/10.1109/TPAMI.2010.57)
- [HAL author record, version v2](https://inria.hal.science/inria-00514462v2) · [specified postprint PDF](https://inria.hal.science/inria-00514462v2/file/jegou_pq_postprint.pdf)（当前返回 browser challenge）
- [Public mirror copy used locally](https://jatin7gupta.github.io/Product-Quantization/PQ%20NN.pdf) · 本地固定副本：[paper-original-mirror.pdf](paper-original-mirror.pdf)
- [Faiss official index documentation: PQ and IVF-PQ](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
