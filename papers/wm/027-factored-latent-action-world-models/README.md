# 027. Factored Latent Action World Models（FLAM）

> **本地论文：** [paper-arxiv-v1.pdf](paper-arxiv-v1.pdf)（23 页）
>
> **Official resources：** [arXiv v1](https://arxiv.org/abs/2602.16229v1) · [v1 HTML](https://arxiv.org/html/2602.16229v1) · [project page](https://sites.google.com/view/factored-lam)

> **代码状态更新（2026-09-26）：** 已确认作者公开仓库 [wangzizhao/flam](https://github.com/wangzizhao/flam)，README 将其标为官方实现。仓库含模型源码、tokenizer / latent action model 训练入口和评估入口；本地论文仍固定为 v1，不把当前仓库自动当作 v1 的逐项复现。尚未运行代码。

> **核对代码版本：** `1c707becc33e0d45411a5cb6137cd984efa7eb8b`。主流程在 `models/lam/lam_factored.py`，temporal factorizer 在 `models/modules/factorizer/temporal_slot_attn.py`。`train_lam.sh` 当前配置 action bottleneck 为 `vae_vq`；其实现是 Gaussian mean/log-variance、训练时 sampling、KL to unit Gaussian，并非 FSQ 离散 action codebook。GitHub API 未识别许可证，根目录未见 LICENSE；已确认公开源码，未确认开放源码许可。
>
> **版本提醒：** 当前 arXiv 记录已有 v2（2026-05-25）。本包按任务指定固定为 v1（2026-02-18），只用 v1 PDF/HTML；没有用 v2 内容替换或补写。arXiv 当前记录未列期刊/会议引用，本包把该固定版本记作 arXiv preprint。
>
> **包核验：** 已核对 v1 正文、Eq. (5)、实验与结论；本地 PDF 的 `%PDF-` 文件头、首页标题/版本、页数已核对并目视检查。未运行或审计代码与实验。**用户阅读状态：** `unread`。

## 1. Paper identity

| Field | Record |
|---|---|
| Title | *Factored Latent Action World Models* |
| Authors | Zizhao Wang, Chang Shi, Jiaheng Hu, Kevin Rohling, Roberto Martín-Martín, Amy Zhang, Peter Stone |
| Year / pinned version | 2026 / arXiv `2602.16229v1`, submitted 2026-02-18 |
| Version status | arXiv preprint；v1 PDF 首页印有 “Preprint. February 19, 2026.”；arXiv 后续已有 v2，但本包固定 v1 |
| Local artifact | `paper-arxiv-v1.pdf`, 23 pages |
| Official project | [`factored-lam`](https://sites.google.com/view/factored-lam) |

## 2. 一句话抓手

FLAM 面向 action-free video 中多个实体同时运动的场景：先把当前视觉 feature 分解成 slots，再给每个 slot 推断自己的 latent action，并预测其下一时刻 slot；最后把所有预测 slots 聚合回 feature space。核心假设是不同实体的动作大体可分，而一整个场景的联合动作难以压进单一 latent action。

需要马上校准标题里的 “independent”：FLAM 对每个 slot 输出单独的 action 和 next-slot prediction，但 IDM/FDM 在 slots 之间共享参数；Eq. (5) 的 FDM 还读取**全部当前 slots**，aggregator 也会合并全部预测 slots 与当前 feature。因此它保留 interaction path，并不是完全独立、互不通信的 K 个小模型。

## 3. Problem：单一 scene-level latent action 的组合难题

经典 latent action model 从相邻视频帧推断一个 `a_t`，再让 forward dynamics 用 `(o_t, a_t)` 预测 `o_{t+1}`。在含多车、多智能体、多个 moving objects 的场景里，单个动作编码必须表示多个实体可能同时采取的组合；随着实体数增加，联合动作组合数可能快速增长。

FLAM 的策略是对 state 和 latent action 一起 factorize。若场景有 `K` 个 slots，每个 slot 产生一个维度约为 `d/K` 的 action；所有 factor 的 actions 使用同一 prior/code space。这样模型学习每个 factor 的动作模式，并通过当前全局 slots 保留相互影响。监督仍来自下一帧预测，不需要视频中的真实 action labels 来训练 latent action model。

## 4. Method：slots → per-slot actions → shared FDM → aggregate

### 4.1 视觉 tokenizer（§4.1，Fig. 3a）

第一阶段训练 dataset-specific VQ-VAE：CNN 把图像编码成 patch features，再用 FSQ 量化，decoder 重建图像。第二阶段训练 FLAM 时 encoder 冻结；作者为每个 dataset 单独训练 encoder，目标是比较 dynamics factorization，而不是做通用 tokenizer。

### 4.2 Factorizer 与 inverse dynamics（§4.2，Fig. 3b，Eq. 2–4）

1. `Slot Attention` 从 feature map 竞争提取 `K` 个 slot。
2. 每个 slot 用 causal temporal attention 查看同一 slot 过去的状态，提升跨帧绑定一致性。
3. 对第 `i` 个 slot，IDM 由所有当前 slots 和它自己的下一帧值推断 action：

   $$a_t^i=\mathrm{IDM}(s_t^{1:K},s_{t+1}^i).$$

   这一步让 action 能考虑交互，例如人的位移来自所乘车辆，而不只是人自身运动。IDM 在所有 slots 间共享参数；latent action 用变分 posterior 并以 KL 项约束，避免直接复制下一状态。

### 4.3 Forward dynamics 与 aggregator（§4.2，Eq. 5–6）

每个 slot 用它自己的 `a_t^i` 预测自己的下一状态，但 FDM 同时读取**所有当前 slots**：

$$\hat{s}_{t+1}^i=\mathrm{FDM}(s_t^{1:K},a_t^i).$$

同一个 FDM 在各 slots 间共享参数。所有 `\hat{s}_{t+1}^{1:K}` 再经 aggregator 回到 feature map；aggregator 以当前 feature `z_t` 为 query、预测 slots 为 key，让新 slots 主要表达帧间变化，而非重复编码静态外观。Factorizer、IDM、FDM、Aggregator 在冻结 tokenizer 上 joint-train，目标是 feature prediction error 加 latent-action KL penalty。

因此更准确的机制描述是：**每个 slot 有局部 action/prediction 通道，模块参数共享，当前全局状态与输出聚合保留 interaction。** 不能从“factored”直接推出完全解耦或运行更快。

## 5. Main findings 与限制

作者在四个模拟数据集（MultiGrid 与 Procgen 的 Bigfish、Leaper、Starpilot）和 nuPlan 真实驾驶视频上评估 world-model prediction；报告 FLAM 在这些多实体设置中整体优于所比较的单一 latent-action 或 object-centric baselines。分析还显示，slot grouping 会跟着 action correlation 变化：动作相关的实体可被分到同一 factor，独立行动的实体更可能分开；factor 数少于独立实体数时 prediction 会变差，超过后表现趋稳。Table 2 的 controlled MultiGrid DCI 结果支持 factor–agent correspondence，但 probe/metric 仍不是因果识别证明。

latent actions 也被用于可控视频生成与下游 behavior cloning。policy experiment 先用 1k 或 10k 有 action label 的样本训练 action decoder，再对约 1M 帧示范生成 pseudo-label；因此这是少量标注下的利用方式，不是零标签 policy learning。

阅读时保留这些边界：

- **版本边界：** v1 是 2026-02-18 的预印本快照；当前 arXiv record 已有 v2。若之后读 v2，须把改动单独对比，不能把新结果归给本地 v1。
- **representation dependence：** encoder 是每个数据集单独预训练并冻结的 tokenizer；论文没有证明同一 factorization 可直接迁移到 LeWM 的 latent 或不同 domain。
- **interaction remains:** Eq. (4) 和 Eq. (5) 都能读取所有当前 slots；aggregator 仍需跨 slots 汇总。结构是 factor-wise conditioning + shared modules，不是数学上互相独立的动力系统。
- **control evidence:** latent action 的用途包括 video generation 与 policy learning；这不等于在 LeWM + PushT、CEM/MPC 或真实机器人上验证规划成功。
- **no systems claim:** 作者用 transformer attention 与 feature aggregation。文中没有证明 native inference latency、峰值显存、吞吐或能耗降低；factorization 和 parameter sharing 不自动带来硬件加速。
- **author-stated limitation（§6）：** 各 dataset 的 VQ-VAE 从头训练；可探索共享/微调预训练 tokenizer。latent rollout 使用 transformer aggregator，并依赖预训练 decoder 生成可视化，作者认为更强 decoder 可能改善视觉质量。

## 6. 与 LeWM + PushT 的相关性和 retraining boundary

相关之处在于：FLAM 把“状态如何分块”与“动作条件下如何预测”一起设计，说明 predictor complexity 可能取决于 state/action representation 的 factorization。对多实体任务，这是直接的 world-model prior art。

但它不是已有 LeWM checkpoint 上的后处理：论文训练 dataset-specific visual tokenizer，再在其 feature 上学习 temporal slots、latent actions、FDM 和 aggregator。将此迁移到 LeWM + PushT 至少需要重新定义 latent slot、action conditioning、prediction target 与 interaction/aggregation，并训练新的 dynamics stack；若要沿用冻结 LeWM encoder，需重新验证 slots 是否能稳定对应 PushT 的状态因素。

也不要把论文视频预测误写成 planner 证据。FYP 若做最小研究，先在固定观察和固定 action/candidate bank 上测 predictor 质量，再检查候选排序、elite set、first action 与 CEM trace，之后才讨论 native timing/memory 和闭环 success。每一层结果回答的问题不同；这篇 paper 本身不提供这些 PushT gates 的结论。

## 7. Reading route

### 20 分钟：抓住多实体瓶颈与模型图

1. Abstract、§1、Fig. 1：解释单一 latent action 的组合问题和 per-factor action 的设想。
2. §3、Fig. 2：复述普通 LAM 如何由 inverse model 产生 action、由 forward model预测下一帧。
3. §4.2、Fig. 3：沿着 slots → IDM → FDM → aggregator 走一遍，特别圈出 Eq. (5) 的全部当前 slots 输入。
4. §5.1：看实验测的是 autoregressive video prediction/control quality，不要先映射成 PushT planning。

### 90 分钟：检查共享、交互与实验设计

1. 精读 §4.1–4.3 与 Eq. (1)–(7)，记下 tokenizer 冻结边界和 KL regularization 的作用。
2. 阅读 §5.1–5.4：prediction accuracy、state representation、policy learning、ablation 各自回答什么问题。
3. 对照 Table 2/3 与 Fig. 7/8：factor correspondence 如何随 action correlation 和 `K` 改变？
4. 读 §6 limitation；写下至少两项从论文结果无法推出的系统或规划结论。

### 180 分钟：形成可审查的 FYP prior-art card

1. 阅读 Appendix A–C：核对 MultiGrid/Procgen/nuPlan 数据、实现与 evaluation metric。
2. 沿正文算法核对 tokenizer、factorizer、IDM/FDM、aggregator 的共享参数与张量输入，再与 [公开作者仓库](https://github.com/wangzizhao/flam) 对照。仓库当前版本与本地 v1 须分别记录；代码可访问不等于已复现论文结果。
3. 用一张图区分“feature extractor frozen”与“latent action model jointly trained”两阶段。
4. 将 FLAM 与 LeWM+PushT 对照：哪些是 slot-representation 假设，哪些是 predictor 结构，哪些必须经过 CEM/planning gate 才能转成 FYP claim。

## 8. Reading Questions（先留空，读完再回答）

1. 单一 latent action 的容量为什么会随多个独立实体的 joint action combinations 变得难以学习？
2. FLAM 所说的 shared action space/prior 在 factorization 中具体限制了什么？
3. 为什么 factorizer 需要 causal temporal attention？slot 的视觉绑定不一致会怎样污染 inverse dynamics？
4. Eq. (4) 为何让 `IDM` 看所有当前 slots，却只看第 `i` 个下一时刻 slot？
5. Eq. (5) 的全部当前 slots 输入解决了什么 interaction 问题？它又如何限制“独立动力学”的解释？
6. IDM 与 FDM 的参数是 per-slot 独立还是在 slots 间共享？“各自 action”与“共享网络”如何同时成立？
7. Aggregator 为什么还需要当前 feature `z_t`，而不是只把预测 slots 解码回帧？
8. KL regularization 约束 latent action 的什么信息通道？beta 改变时预测质量和 action capacity 有什么权衡？
9. tokenizer frozen 对 baseline comparison 有什么好处？它对 universal world-model claim 有什么限制？
10. Table 2 的 DCI 与 factor–agent correspondence 能支持到哪一级结论？哪些仍只是 probe/metric evidence？
11. 当两个实体总是同步行动时，FLAM 为什么可能把它们合到一个 factor？这是否意味着学到了对象边界？
12. `K` 小于实体数和大于实体数时，prediction/slot assignment 分别出现什么现象？
13. policy experiment 中 action decoder 和 1M-frame pseudo-label 流程用了多少/哪类 action supervision？
14. 预测时 latent actions 是从未来 ground-truth frames 推断的评估设置，会怎样影响 controllability 或 rollout claim 的解释？
15. FLAM 的 shared attention/FDM/aggregator 与“完全独立小模型”在参数效率和运行成本上可能分别有什么差异？
16. 要把这个思路接到 LeWM + PushT，最小需要重训哪些组件，哪些 predictor/planner/runtime gate 必须单独通过？

## 9. Meeting Card（留空）

- **我现在的 FLAM 数据流图：**
- **Eq. (5) 的输入/输出与共享关系：**
- **我认为最强的一条实验证据：**
- **一个尚未解决的限制或替代解释：**
- **它与 LeWM + PushT 的具体关系：**
- **v1 与之后版本需要核对的差异：**
- **下次讨论时我想问：**
