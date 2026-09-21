# LeWM PushT latent-delta：Step 1 / Step 2 冻结协议

状态：`frozen`  
Baseline：official LeWM PushT checkpoint  
主冻结文件：[LEWM_DELTA_ORACLE_FREEZE.json](LEWM_DELTA_ORACLE_FREEZE.json)

## 研究问题

LeWM 的 planner-facing state 是 `192-D CLS vector`，不是 `N×d` token map。因此，对单个
delta matrix 做 low-rank SVD 会退化成 rank-1，不能检验原 idea。本实验把可证伪假设改写为：

> 从 calibration episodes 学到的同一个低维 subspace，能否在 episode-disjoint held-out
> transitions 和 action candidates 上重建 latent delta，并保留 LeWM 的 candidate ranking
> 与 fixed-observation CEM action？

## Step 1：delta 是否存在可复用低维结构

同一 episode split 上收集两种 delta：

1. `observed delta`：把 frame offsets `0/5/10/15/20/25` 分别通过 official encoder，计算相邻
   world-model steps 的 `E(o_{t+1})-E(o_t)`；
2. `model delta`：对 full LeWM 的五步 rollout，计算 `ẑ_{h+1}-ẑ_h`，其中第一步相对当前
   encoded latent。

### 1A. Per-trajectory oracle SVD lower-bound

对每条 observed 或 model-rollout trajectory，把五个连续 delta 堆成一个
`5×192` matrix：

```text
D_i = [delta_1; delta_2; delta_3; delta_4; delta_5]
```

对每条 trajectory 单独计算 rank `1/2/3/4/5` SVD。这是该 trajectory 自己的
hindsight 最优 basis，只用于回答“如果知道这条 trajectory 的最佳低秩方向，信息
损失有多小”。它不是 reusable global basis，不能用于运行时，也不能被写成 predictor
latency 或 acceleration evidence。

### 1B. Reusable global basis

只在 256 个 calibration episodes 上拟合 centered PCA basis；8 个 held-out episodes
不参与拟合。primary 是 model-delta calibration PCA；observed-delta PCA 只作为
cross-basis control。报告 singular spectrum、cumulative centered variance、held-out
relative MSE/cosine、按 horizon 分解，以及 observed-basis/model-basis 的双向 cross
reconstruction。Step 1 的 structure gate 只适用于 reusable model basis，不适用于
per-trajectory oracle lower-bound。

结构 gate：存在 `rank ≤ 64`，同时满足 held-out centered variance retained `≥ .95`、relative
MSE `≤ .05`。无论 gate 是否通过都继续执行 Step 2 的候选诊断，以避免仅凭 global MSE 作结论。

## Step 2：oracle / reusable projection 会不会破坏 planner decision

本步骤保留用户原始要求的 official fixed-observation CEM，但它不是 closed-loop
实验，也不是 acceleration benchmark。所有近似路径都先完整运行 LeWM predictor；
因此不会省掉 predictor 计算。

Per-trajectory oracle 路径对每个 candidate 的 `5×192` full-model delta 单独做
rank-`1..5` SVD；reusable path 则使用 calibration 阶段冻结的 global basis。两者
必须分开报告：

```text
full LeWM rollout -> full delta stack
                  ├── per-trajectory rank-1..5 oracle SVD -> cumulative reconstruction
                  └── frozen reusable model-PCA basis      -> cumulative reconstruction
                  -> unchanged official criterion / fixed-observation CEM
```

per-trajectory branch 只检验信息下界；model-PCA branch 是 primary reusable basis。
observed-PCA 是 cross-basis control，deterministic random orthonormal basis 是 negative
control。任何 branch 都不构成 acceleration result。

### 2A. held-out fixed candidate banks

沿用 8 个 held-out contexts × 2 seeds × 300 candidates。每个 rank 报告：

- official terminal-objective Spearman；
- top-30 overlap；
- argmin agreement with the full teacher；
- selected candidate 的 first-action normalized L2；
- 用 full teacher objective 评价 approximate-selected candidate 的 teacher-objective regret；
- objective error；
- rollout latent relative MSE/cosine；
- model-PCA primary、observed-PCA cross-basis；
- deterministic random rank-64 basis negative control。

Candidate gate：median/minimum Spearman `≥ .99/.95`，median/minimum top-30 overlap
`≥ .90/.75`；argmin agreement 的 median `≥ .75`，selected first-action normalized L2
的 median `≤ .15`。teacher-objective regret 必须逐 block 报告，但不另设阈值，保留其
official objective native units。

### 2B. official fixed-observation CEM（不是 closed-loop，也不是加速结果）

保持 official solver：`M=300`、`30 iterations`、`topk=30`、`horizon=5`、`action_block=5`。
运行 baseline、rank-192 exact-control，以及最多两个最小 reusable model-basis
candidate-gate passing ranks；per-trajectory oracle SVD 不得作为 CEM replacement arm。
若没有 reusable approximate rank 通过候选 gate，不运行 approximate-rank CEM。比较相同
observation、seed 下的 first/final action；CEM 不包含 environment interaction，也不
采集或解释 latency reduction。

Planner gate：first-action normalized L2 `≤ .15` 且 maximum absolute difference `≤ .25`。

## 解释边界

- PASS 只说明 frozen LeWM/PushT protocol 下存在 oracle information bottleneck 候选；
- FAIL 意味着不应直接用该 rank 的 fixed linear delta subspace；
- 本实验不训练 cheap path，不测真实 speedup，不运行 closed-loop，不声称 end-to-end success；
- full-rank control 用于发现 projection/reconstruction 接线错误；random-basis 用于排除“任意相同维数
  都一样”的解释。

## 执行约束

所有 dataset I/O、encoder、checkpoint load、SVD、predictor 和 CEM 只在 PBS GPU compute node
运行。Login node 只上传小型控制文件、提交和查询。GPU 作业每 30 秒把 utilization/VRAM 写入
job log。不下载、不安装、不解压、不做额外 hash。
