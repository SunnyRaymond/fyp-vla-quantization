# LeWM CEM candidate-distribution score distillation（draft）

本文件是给 parent review 的冻结草案，尚未提交 PBS。实验只在 fixed observation 上比较
candidate ranking 与 teacher regret，不执行 environment、official planner deployment 或
closed-loop PushT。

## 采样语义

本轮以 pinned stable-worldmodel LeWM solver 为语义来源：
`meeting-ready/01-iteration-cache-cross-model-planners/cases/lewm-pusht-cem/LEWM_PUSHT_ITERATION_CACHE_FREEZE.json:35-80`。
保持 `M=300`、`top30`、`30` iterations、`horizon=5`、packed action dim `10`、
`var_scale=1.0`；DINO-WM CEM helper 只作为结构参考，不把它的排序调用方式冒充
LeWM solver：

1. 每个 context 用固定 CPU `torch.Generator` 预先生成共同 standard-normal
   innovations。
2. 每轮用当前 pre-update `mu/sigma` 形成 `actions = mu + sigma * epsilon`，再将
   candidate 0 精确替换为 pre-update `mu`。
3. 用 `torch.topk(cost, k=30, largest=False)` 选 elites；`mu_next=elite.mean(dim=0)`，
   `sigma_next=elite.std(dim=0)`，保留默认 `unbiased=True`。
4. CEM loop 不使用 temporal training 的 `official_action_low/high` 与 Gaussian
   `_slate_actions` clipping；两者是不同的 action-generation contract。

student-only driver 从已保存的 anchor-aligned step3000 checkpoint 开始。每个 train
context 跑完整 30 轮，固定保存 round 10/20/30 的 300-candidate actions。每个保存轮按
预先固定的 candidate index 取 11/11/10 条，合计 32 条；teacher 只为这 32 条生成 targets
与 official cost，不参与该 driver 的 elite 选择或 `mu/sigma` 更新。报告中使用
“student-driven CEM trajectory bank with teacher shadow labels”，不写成
“teacher-driven”或“student on-policy”。

## 两臂和训练

control 与 treatment 都从同一个
`anchor_aligned_bank_step3000.pt` state dict 开始，各自重置 AdamW 并运行 1000 updates。
control 继续原 64-candidate anchor-aligned bank；treatment 的 64 候选为原 bank 前 32
条加上 round 10/20/30 固定抽取的 32 条 CEM tail actions。CEM tail 的 teacher targets
只在 compute node 生成并保存；context、anchor、rank permutation、loss、optimizer、
batch size、schedule 和额外 update budget 都保持一致。

已有 `anchor-aligned-bank/run_anchor_aligned_bank.py:89-338` 可复用 freeze/interface/
fresh-row/checkpoint provenance 与 bank shape 检查；`teacher-screening/run_teacher_screening.py:346-368`
可复用 student/teacher cost wrapper；`run_lewm_recurrent_student.py:375,757-798` 和
`temporal-balanced-train/run_lewm_temporal_balanced_train.py:180-213` 可复用 teacher
target、recurrent loss、score-distill loss 和 AdamW 细节。现有 `train_arm` 默认从传入
的 initial state 训练，因此新 runner 只需增加一个接收已加载 step3000 state、运行
1000 steps 的本地 continuation loop，不改旧文件。

## Fresh evaluation

fresh episode 固定为沿用原 shuffle 的 `valid[560:568]`（selection seed `20300903`），
排除 `valid[:560]`，8 episodes × early/middle/late × 2 action-prefix seeds = 48 blocks；
episode 是 replicate，candidate 和 CEM round 是 nested measurements。两臂共享同一份
frozen-start student-driven CEM trajectory bank，
每个 bank 在 round 10/20/30 由 teacher shadow 评分后固定。每轮报告 Spearman、top30、
relative latent MSE、recall@60/120、full containment、raw regret 和按 block teacher
cost std（floor `1e-6`）标准化的 elite mean-cost regret，并按 round、stratum、episode、
seed block 与 worst cases 展开。

另用同一 fresh contexts 的 standard current-anchor 300-candidate bank 报告最终模型的
forgetting guard。它不参与 CEM tail 选择，也不用于挑 episode、round 或 checkpoint。

## Gate 与解释

每个 block 的 primary standardized regret 固定为
`(teacher_mean(student_top30) - teacher_mean(teacher_top30)) /
max(population_std(teacher_cost_300), 1e-6)`。primary endpoint 合并 round 10/20/30 的
18 个 episode 内 blocks（3 rounds × 3 anchors × 2 seeds）后再作 episode median；每个
block 的 student top30 都直接以 teacher cost 计算 regret，不能把 K=60/120 hybrid
regret 代入。treatment-control 的 8-episode median delta 必须 `<=−0.05`，至少 5/8
episodes 的 delta 严格小于 0；并且分别在 round 10/20/30 计算的 episode-median delta
都必须 `<=0`。absolute recall 只作描述，不替代 mechanism gate。
standard current-anchor 300-candidate bank 是独立 forgetting guard：episode-median
standardized-regret delta 必须 `<=+0.05`，且没有 episode delta `>+0.20`；否则标记
forgetting，不能称 recipe GO。

以上 gate 是 predictor/mechanism gate，不是 predictor replacement 或 planner gate。即使
通过也不进入 closed-loop；若失败，保留 CEM-distribution recipe 的 NO-GO/partial evidence。

## 执行边界

runner 必须在开始模型、HDF5、teacher target、CEM、训练或 benchmark 前检查非空
`PBS_JOBID`、非 login/head/submit hostname 和 allocation node。PBS job 每 5 秒向
`job.log` 写 GPU utilization/VRAM，保存实际 `nvidia-smi` PID 并由 trap 清理。login node
只用于轻量控制、提交和状态查询。推荐单个 `1 GPU/16 CPU/110 GB/30 min` bounded job；
large rows、models、checkpoints 留在 cluster，只回传小 summary、status、job log。

官方接口边界参考
`meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/PROTOCOL_LEWM_RECURRENT_STUDENT.zh.md:243-286`；该文件的 Stage B 是 protocol reference，
不是可直接调用的 Stage B runner。
