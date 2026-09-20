# DINO-WM PushT：Action-Query Slate + Planner-Aware Rank Distillation 冻结协议

状态：`frozen / protocol only / 未提交作业`

对应冻结文件：`QUERY_SLATE_RANK_FREEZE.json`

## 1. 这项实验要回答什么

前几轮实验已经把几个简单解释压低了优先级：

- 把训练延长到 1500 updates 没有显著改善 candidate ranking；
- 把 student hidden width 从 128 提到 256 没有改善 ranking；
- 在同一批 32 个 train episodes 中增加 context support，明显降低了 latent MSE，但 ranking effect 没有通过 gate；
- 因此，当前更像是 **action-conditioned predictor 没有学到同一 context 内的 action 相对顺序**，而不仅是平均 latent 回归不够精确。

本 cell 检验一个最小机制：每次让一个 context 同时出现多个 action queries，给 planner-aware loss 一个可比较的 action slate。control 和 treatment 看到完全相同的 slate；唯一改变是 treatment 多一个固定的 listwise KL。

这仍然是 predictor-level 实验，不涉及 `encode_obs`、CEM 执行或闭环控制。

## 2. 两个 paired arms

| arm | 输入与训练数据 | loss |
|---|---|---|
| `slate_mse_control` | 256 个 Dhigh train contexts；每个 context 的四个 query 都保留 | 32 个 query rows 的 dense native-latent MSE |
| `slate_rank_treatment` | 与 control 完全相同 | 同一 MSE + `0.1 × listwise_KL` |

两 arm 使用同一个新建的 h256 初始 state dictionary，并各自创建独立 AdamW optimizer。每个 update 使用同一个 context schedule、同一个 action-slate bank、同一个 detached teacher target。按自然数编号，偶数 update 先更新 control，奇数 update 先更新 treatment，用来抵消同一 GPU 上的固定执行顺序影响。

这不是把已有 Dhigh checkpoint 继续训练，也不是重跑旧的 500-step rank-distill。

## 3. Action slate 如何构造

每个训练 context 固定四个 action queries，顺序为：

1. `primary_logged`：保留该 context 在 Dhigh manifest 中的真实五步 action prefix；
2. `gaussian_planner_init_a`：独立 `Normal(0, 1)` planner-init query；
3. `gaussian_planner_init_b`：另一个独立的 `Normal(0, 1)` planner-init query；
4. `one_step_cem_resample`：使用 frozen teacher 做一次 `M=64, K=8` 的 elite-resample query。

原始 primitive action 的形状是 `[5,2]`；经过 official DINO-WM action packing 后，student/teacher 实际接收的 token 形状是 `[5,10]`。两者不可混写。

后三者是 planner-like counterfactual queries。CEM query 的候选由 CPU generator 产生，用同一 context 的 frozen teacher terminal objective 选出最低成本的 8 个候选，再按 per-coordinate variance floor `0.05` 重采样一次。student 的输出不会参与 CEM query 生成。

完整 slate bank 在第一个 update 前生成并冻结，规模是 `256 contexts × 4 queries`。每个 context 内四个 prefix 必须逐一 pairwise distinct；如果固定 seed 造成 exact collision，整个 cell 视为 invalid，不允许看到结果后换 seed。

四个 query 都通过相同的 frozen autoregressive teacher rollout 生成 dense future-latent targets。goal 只用于 CEM 选样和 planner objective，不送入 student。

## 4. Effective batch 与梯度语义

每个 update 选择 8 个不同 context，因此：

```text
8 context slates × 4 queries/slate = 32 query rows
```

冻结为：

- effective batch：32 query rows；
- microbatch：完整的 8 个 slates、32 rows；
- gradient accumulation：1；
- 每个 arm 每个 update：恰好一次 backward 和一次 optimizer step；
- 不允许把一个 slate 拆到不同 microbatch，也不允许跨 update 拼接 rank group。

Control 的 MSE 是对 32 rows、5 个 horizon 和 native latent dimensions 的总体 mean。Treatment 的 rank KL 是对 8 个完整 slate 的 group mean，然后：

\[
L_{control}=L_{latent},
\qquad
L_{treatment}=L_{latent}+0.1L_{rank}.
\]

这样增加的是监督形式，而不是 effective batch、每步 optimizer 更新次数或 student architecture。

## 5. Context schedule 与 seeds

训练固定为 h256、256 个 Dhigh contexts、1500 updates、AdamW、learning rate `3e-4`，snapshot 为 500/1000/1500。

context schedule 使用 CPU generator，seed 为 `20260931`：每 32 个 updates 对 `0..255` 做一次 permutation，每次取连续 8 个 ordinals。整个 1500-step schedule 在训练前生成，因此每个 update 内没有 context 重复，全部 12000 个 context slots 中每个 context 暴露 46 或 47 次。

固定 seeds：

| 用途 | seed |
|---|---:|
| student initialization | `99` |
| training schedule | `20260930` |
| context schedule | `20260931` |
| Gaussian/CEM slate bank | `20260932` / `20260933` |
| held-out action prefixes | `20264925`, `20264926` |
| timing action prefixes | 继承 Dhigh predictor-level timing contract |

所有 action proposals、context schedule、held-out candidates 和 timing candidates 都先用显式 CPU generator 产生，再传到 GPU。不得按结果替换 seed、slate composition、lambda、temperature、steps 或 gate。

## 6. Planner-aware listwise KL

一个 rank group 就是同一 context 在同一个 update 的四个 queries；不跨 context 排名。

对每个 group：

1. 用 frozen teacher 和 student 的 horizon-5 terminal latent 计算 official latent-to-goal objective cost；
2. 在四个 cost 内做 z-score，std floor 为 `1e-6`；
3. lower cost 视为更好，形成

   \[
   p_T=softmax(-\tilde c_T/\tau),
   \quad
   p_S=softmax(-\tilde c_S/\tau);
   \]

4. 计算 `KL(p_T || p_S)`，temperature `τ=1.0`；
5. teacher cost detach，student cost 保留梯度。

student 的 forward 输入始终只有 cached native observation latent 和 action prefix。它不读取 goal，也不调用 `encode_obs`。

## 7. Held-out 评估

两 arm 和 frozen teacher 使用完全相同的 held-out candidate bank：

- 8 个 held-out contexts；
- 2 个 fresh action-prefix seeds；
- 每个 block 300 candidates；
- 共 16 个 paired blocks；
- top-k=`30`；
- block 是分析单位，300 candidates 不是 300 个独立样本。

主比较是：

```text
slate_rank treatment − slate_mse control
```

每个 block 共享相同的 context、candidate actions、goal 和 teacher costs。报告 objective Spearman、top-30 overlap、teacher-relative latent MSE 和 per-horizon cosine。

已经完成的 h256-Dhigh 结果（job `24477396.pbs101`）只作为外部 descriptive diagnostic：可以并列报告绝对 ranking、logged-action MSE、causality 和 predictor-only latency，但不能把它作为本实验 control，也不能用 `new − Dhigh` 定义 paired effect。原因是 Dhigh 使用了不同的 initialization/context schedule/action schedule。

## 8. 冻结 gates

先判断 paired integrity；若 integrity 失败，则不解释任何 ranking delta。

| gate | 冻结要求 |
|---|---|
| paired integrity | 两 arm finite；相同初始 state、context schedule、slate bank、held-out bank；所有 slate 完整且四个 query pairwise distinct |
| training convergence | 两 arm 的 latent-MSE `last10 / first <= 0.8` |
| causality | 两 arm 的 future-action leakage `<= 1e-6` |
| planner-ranking effect | median ΔSpearman `>= +0.05`，median Δtop-30 `>= +0.10`，两项各至少 `12/16` blocks 为正 |
| absolute fidelity | treatment median Spearman `>= .99`、minimum `>= .95`；median top-30 `>= .95`、minimum `>= .80` |
| latent non-inferiority | held-out logged-action teacher-relative MSE 的 treatment/control ratio `<= 1.25` |
| predictor latency | treatment 相对 frozen teacher 的 predictor-only reduction `>= 20%` |

独立报告三个决策：

1. `paired_effect`：完整 integrity 且 ranking-effect 四项都 PASS；
2. `full_replacement`：paired integrity、paired effect、absolute fidelity、MSE non-inferiority 和 latency 全 PASS；
3. `slate_supported_replacement`：前两者都 PASS。

任一失败都保留为 NO-GO，不进入 closed-loop，不调阈值，也不增加一轮相同 cell 来包装结果。

## 9. 可以声称什么

若 `paired_effect=PASS`，最多声称：在 frozen DINO-WM PushT predictor-level contract 下，让同一 context 同时提供多个 action queries，并加入固定 planner-aware listwise KL，改善了相对于相同 slate-trained latent-MSE control 的 held-out candidate ranking。

若 `full_replacement=GO`，才可以进一步声称 treatment 满足本协议定义的 predictor-level replacement gates。

本协议不能支持 observation encoder 加速、full CEM、closed-loop task success、端到端控制频率、LeWM/Fast-LeWM transfer、all-JEPA universality、population-level inference 或 native low-bit deployment claim。

本次只创建冻结文件和阅读协议，**没有提交 PBS 作业**。
